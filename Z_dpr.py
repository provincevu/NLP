import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

# ====== CONFIG ======
MODEL_NAME = "vinai/phobert-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class TrainConfig:
    passages_path: str = "/content/content/data/passages.json"
    train_path: str = "/content/content/data/splits/train.json"
    val_path: str = "/content/content/data/splits/val.json"
    out_dir: str = "/content/drive/MyDrive/DPR_DATA/dpr_ckpt_3"

    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 0.02
    warmup_ratio: float = 0.2

    max_q_len: int = 64
    max_p_len: int = 256

    # gradient accumulation nếu GPU yếu / muốn batch hiệu dụng lớn hơn
    grad_accum_steps: int = 4

    # (tuỳ chọn) clip grad để ổn định
    max_grad_norm: float = 1.0

def read_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        return data["data"]
    if isinstance(data, list):
        return data

    raise ValueError(f"Unsupported JSON structure in {path}")

def load_passages_map(passages_path: str) -> Dict[str, dict]:
    mp = {}
    passages = read_json(passages_path)
    for p in passages:
        if "passage_id" not in p or "text" not in p:
            continue
        mp[p["passage_id"]] = p
    return mp

def mean_pool(last_hidden, attn_mask):
    mask = attn_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom

def max_pool(last_hidden, attn_mask):
    mask = attn_mask.unsqueeze(-1).float()
    masked_hidden = last_hidden.masked_fill(mask == 0, float("-inf"))
    maxed = masked_hidden.max(dim=1).values
    return maxed

class DPRInBatchDataset(Dataset):
    def __init__(self, qa_path: str, passages_map: Dict[str, dict]):
        self.items = read_json(qa_path)
        self.passages_map = passages_map

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx) -> Tuple[str, str]:
        ex = self.items[idx]
        q = ex["question"]
        pos_id = ex["positive_passage_id"]
        if pos_id not in self.passages_map:
            raise ValueError(f"passage không tìm thấy id: {pos_id}")
        p_text = self.passages_map[pos_id]["text"]
        return q, p_text

class DualEncoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        vec = mean_pool(out.last_hidden_state, attention_mask)
        vec = F.normalize(vec, p=2, dim=1)
        return vec

    def forward(self, q_batch, p_batch):
        q_vec = self.encode(q_batch["input_ids"], q_batch["attention_mask"])
        p_vec = self.encode(p_batch["input_ids"], p_batch["attention_mask"])
        return q_vec, p_vec

def make_collate(tokenizer, max_q_len, max_p_len):
    def collate(batch: List[Tuple[str, str]]):
        qs, ps = zip(*batch)
        q_tok = tokenizer(list(qs), padding=True, truncation=True, max_length=max_q_len, return_tensors="pt")
        p_tok = tokenizer(list(ps), padding=True, truncation=True, max_length=max_p_len, return_tensors="pt")
        return q_tok, p_tok
    return collate

@torch.no_grad()
def eval_retrieval(model, dl) -> float:
    model.eval()
    total_loss, n_batches = 0.0, 0

    for q_tok, p_tok in dl:
        q_tok = {k: v.to(DEVICE) for k, v in q_tok.items()}
        p_tok = {k: v.to(DEVICE) for k, v in p_tok.items()}

        q_vec, p_vec = model(q_tok, p_tok)
        scores = q_vec @ p_vec.T
        labels = torch.arange(scores.size(0), device=DEVICE)
        loss = F.cross_entropy(scores, labels)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)

def train(cfg: TrainConfig):
    passages_map = load_passages_map(cfg.passages_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = DualEncoder(MODEL_NAME).to(DEVICE)

    train_ds = DPRInBatchDataset(cfg.train_path, passages_map)
    val_ds = DPRInBatchDataset(cfg.val_path, passages_map)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=make_collate(tokenizer, cfg.max_q_len, cfg.max_p_len)
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=make_collate(tokenizer, cfg.max_q_len, cfg.max_p_len)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # số step update thực sự = ceil(num_batches / grad_accum_steps) * epochs
    steps_per_epoch = math.ceil(len(train_dl) / cfg.grad_accum_steps)
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running = 0.0
        pbar = tqdm(enumerate(train_dl, start=1), total=len(train_dl), desc=f"Epoch {epoch}")

        for step, (q_tok, p_tok) in pbar:
            q_tok = {k: v.to(DEVICE) for k, v in q_tok.items()}
            p_tok = {k: v.to(DEVICE) for k, v in p_tok.items()}

            q_vec, p_vec = model(q_tok, p_tok)
            scores = q_vec @ p_vec.T
            labels = torch.arange(scores.size(0), device=DEVICE)

            # CHIA loss cho grad_accum_steps để tổng gradient tương đương
            loss = F.cross_entropy(scores, labels) / cfg.grad_accum_steps

            loss.backward()
            running += loss.item()

            do_step = (step % cfg.grad_accum_steps == 0) or (step == len(train_dl))
            if do_step:
                if cfg.max_grad_norm is not None and cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            pbar.set_postfix(loss=f"{(running/step):.4f}", upd_step=("Y" if do_step else "N"))

        val_loss = eval_retrieval(model, val_dl)
        print(f"[Epoch {epoch}] val_loss = {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            import os
            os.makedirs(cfg.out_dir, exist_ok=True)
            model.encoder.save_pretrained(cfg.out_dir)
            tokenizer.save_pretrained(cfg.out_dir)
            print(f"Saved best checkpoint to: {cfg.out_dir}")

    print("Done.")

if __name__ == "__main__":
    cfg = TrainConfig()
    train(cfg)