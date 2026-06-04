import argparse
import json
import os
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm


# dataloader cho cross-encoder, đọc từ json hoặc jsonl
class CrossEncoderDataset(Dataset):
    def __init__(self, path: str):
        # path: json or jsonl với mỗi row có question, passage_text, label
        self.rows = []
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self.rows.append(json.loads(line))
        else:
            with open(path, "r", encoding="utf-8") as f:
                self.rows = json.load(f)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return r["question"], r["passage_text"], int(r.get("label", 0))


# model cross-encoder, trả về logit cho cặp question-passage
class CrossEncoderModel(nn.Module):
    def __init__(self, backbone_name: str, hidden_dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        x = self.dropout(pooled)
        logit = self.classifier(x).squeeze(-1)
        return logit


def collate_fn(batch, tokenizer, max_len):
    qs, ps, labels = zip(*batch)
    toks = tokenizer(list(qs), list(ps), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return toks, torch.tensor(labels, dtype=torch.float)


def evaluate(model, dl, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    loss_f = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for toks, labels in dl:
            toks = {k: v.to(device) for k, v in toks.items()}
            labels = labels.to(device)
            logits = model(toks["input_ids"], toks["attention_mask"])
            loss = loss_f(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def train(cfg):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    train_path = os.path.join(cfg.data_dir, f"train.{cfg.input_ext}")
    val_path = os.path.join(cfg.data_dir, f"val.{cfg.input_ext}")

    train_ds = CrossEncoderDataset(train_path)
    val_ds = CrossEncoderDataset(val_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone, use_fast=False)
    model = CrossEncoderModel(cfg.backbone).to(DEVICE)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          collate_fn=lambda b: collate_fn(b, tokenizer, cfg.max_len))
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, tokenizer, cfg.max_len))

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_dl) * cfg.epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    loss_f = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    os.makedirs(cfg.out_dir, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(train_dl, start=1), total=len(train_dl), desc=f"Epoch {epoch}")
        for step, (toks, labels) in pbar:
            toks = {k: v.to(DEVICE) for k, v in toks.items()}
            labels = labels.to(DEVICE)
            logits = model(toks["input_ids"], toks["attention_mask"])
            loss = loss_f(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{running_loss/step:.4f}")

        val_loss, val_acc = evaluate(model, val_dl, DEVICE)
        print(f"[Epoch {epoch}] val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir = os.path.join(cfg.out_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            model_path = os.path.join(best_dir, "model.pt")
            torch.save(model.state_dict(), model_path)
            tokenizer.save_pretrained(best_dir)
            with open(os.path.join(best_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2)
            print(f"Saved best model to {model_path}")

        # save last
        last_dir = os.path.join(cfg.out_dir, "last")
        os.makedirs(last_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(last_dir, "model.pt"))
        tokenizer.save_pretrained(last_dir)
        with open(os.path.join(last_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2)

    print("Training finished.")


from dataclasses import dataclass

@dataclass
class TrainConfig:
    data_dir: str = "/content/data/cross_encoder"

    backbone: str = "vinai/phobert-base"

    out_dir: str = "/content/drive/MyDrive/DPR_DATA/cross_encoder_ckpt"

    epochs: int = 20
    batch_size: int = 64

    lr: float = 3e-5
    weight_decay: float = 0.01

    max_len: int = 512
    input_ext = "jsonl"

    warmup_ratio: float = 0.05

    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0


if __name__ == "__main__":
    cfg = TrainConfig()
    train(cfg)
