import argparse
import json
import os
from typing import List, Tuple
from train_cross_encoder4colab import CrossEncoderModel

import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

# đọc dữ liệu từ json hoặc jsonl
def read_json_list(path: str) -> List[dict]:
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return data


# ghi dữ liệu ra json hoặc jsonl
def write_json_list(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


# đọc config từ ckpt_dir/config.json nếu có
def load_config(ckpt_dir: str) -> dict:
    cfg_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def encode_batch(rows: List[dict], tokenizer, model, device, max_len: int) -> Tuple[List[float], List[float]]:
    qs = [r["question"] for r in rows]
    ps = [r["passage_text"] for r in rows]
    toks = tokenizer(qs, ps, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    toks = {k: v.to(device) for k, v in toks.items()}
    with torch.no_grad():
        logits = model(toks["input_ids"], toks["attention_mask"])
        probs = torch.sigmoid(logits)
    return logits.cpu().tolist(), probs.cpu().tolist()


def infer(args):
    rows = read_json_list(args.input_path)
    cfg = load_config(args.ckpt_dir)

    backbone = args.backbone or cfg.get("backbone")

    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir, use_fast=False)
    model = CrossEncoderModel(backbone).to(device)
    state = torch.load(os.path.join(args.ckpt_dir, "model.pt"), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    output_rows = []
    batch_size = args.batch_size
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        logits, probs = encode_batch(batch, tokenizer, model, device, args.max_len)
        for r, logit, prob in zip(batch, logits, probs):
            out = dict(r)
            if args.save_logit:
                out[args.logit_key] = float(logit)
            if args.save_prob:
                out[args.prob_key] = float(prob)
            output_rows.append(out)

    write_json_list(args.output_path, output_rows)
    print(f"Saved {len(output_rows)} rows to {args.output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Inference for cross-encoder reranker")
    p.add_argument("--input_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--backbone", default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save_logit", action="store_true")
    p.add_argument("--save_prob", action="store_true")
    p.add_argument("--logit_key", default="ce_logit")
    p.add_argument("--prob_key", default="ce_prob")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.save_logit and not args.save_prob:
        args.save_logit = True
        args.save_prob = True
    infer(args)
