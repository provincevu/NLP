import argparse
import json
import os
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm


class CrossEncoderDataset(Dataset):
    def __init__(self, path: str):
        # path: json or jsonl where each row has question, passage_text, label
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


class CrossEncoderModel(nn.Module):
    def __init__(self, backbone_name: str, hidden_dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # use CLS token representation
        cls = out.last_hidden_state[:, 0, :]
        x = self.dropout(cls)
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


def train(args):
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    train_path = os.path.join(args.data_dir, f"train.{args.input_ext}")
    val_path = os.path.join(args.data_dir, f"val.{args.input_ext}")

    train_ds = CrossEncoderDataset(train_path)
    val_ds = CrossEncoderDataset(val_path)

    tokenizer = AutoTokenizer.from_pretrained(args.backbone, use_fast=False)
    model = CrossEncoderModel(args.backbone).to(device)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=lambda b: collate_fn(b, tokenizer, args.max_len))
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_len))

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps)
    loss_f = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(train_dl, start=1), total=len(train_dl), desc=f"Epoch {epoch}")
        for step, (toks, labels) in pbar:
            toks = {k: v.to(device) for k, v in toks.items()}
            labels = labels.to(device)
            logits = model(toks["input_ids"], toks["attention_mask"])
            loss = loss_f(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{running_loss/step:.4f}")

        val_loss, val_acc = evaluate(model, val_dl, device)
        print(f"[Epoch {epoch}] val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.out_dir, "best_model.pt")
            torch.save(model.state_dict(), model_path)
            tokenizer.save_pretrained(args.out_dir)
            print(f"Saved best model to {model_path}")

        # save last
        torch.save(model.state_dict(), os.path.join(args.out_dir, "last_model.pt"))

    print("Training finished.")


def parse_args():
    p = argparse.ArgumentParser(description="Train cross-encoder reranker")
    p.add_argument("--data_dir", default=os.path.join(os.path.dirname(__file__), "data"))
    p.add_argument("--backbone", default="vinai/phobert-base")
    p.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "ckpt"))
    p.add_argument("--input_ext", default="jsonl", choices=["jsonl", "json"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
