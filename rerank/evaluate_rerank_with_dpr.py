import argparse
import json
import math
import os
from collections import defaultdict
from typing import Dict, List

import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

BASE_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
BUILD_ROOT = os.path.join(PROJECT_DIR, "build")

DEFAULT_SPLIT = os.path.join(PROJECT_DIR, "data", "splits", "val.json")
DEFAULT_PASSAGES = os.path.join(PROJECT_DIR, "data", "passages.json")
DEFAULT_DPR_CKPT = os.path.join(PROJECT_DIR, "inferences", "dpr_ckpt_1")
DEFAULT_DPR_BASE = "vinai/phobert-base"
DEFAULT_DPR_WEIGHTS = "model_last.pt"
DEFAULT_CE_CKPT = os.path.join(PROJECT_DIR, "rerank", "ckpt", "best")

FAISS_NAME = "faiss_index.ivf"
IDMAP_NAME = "passage_id_map.json"


def read_json_list(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return data


def load_passage_map(path: str) -> Dict[str, dict]:
    passages = read_json_list(path)
    return {p["passage_id"]: p for p in passages if "passage_id" in p}


def get_latest_build_dir(build_root: str) -> str:
    if not os.path.isdir(build_root):
        raise FileNotFoundError(f"Build folder not found: {build_root}")
    candidates = [
        os.path.join(build_root, d)
        for d in os.listdir(build_root)
        if os.path.isdir(os.path.join(build_root, d))
    ]
    if not candidates:
        raise FileNotFoundError("No build subfolders found. Run encode_passages.py first.")
    return sorted(candidates)[-1]


def resolve_passage_id(id_map, idx: int) -> str:
    if isinstance(id_map, list):
        return id_map[idx]
    if isinstance(id_map, dict):
        if idx in id_map:
            return id_map[idx]
        return id_map.get(str(idx))
    raise TypeError("Unsupported id_map type")


class DualEncoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.encoder.config.hidden_size, 768)

    def encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        vec = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        vec = self.proj(vec)
        vec = F.normalize(vec, p=2, dim=1)
        return vec


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


def encode_queries(queries, tokenizer, encoder, device, max_len: int):
    toks = tokenizer(queries, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    toks = {k: v.to(device) for k, v in toks.items()}
    with torch.no_grad():
        vec = encoder.encode(toks["input_ids"], toks["attention_mask"])
    return vec.cpu().numpy()


def score_pairs(questions, passages, tokenizer, model, device, max_len: int):
    toks = tokenizer(questions, passages, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    toks = {k: v.to(device) for k, v in toks.items()}
    with torch.no_grad():
        logits = model(toks["input_ids"], toks["attention_mask"])
    return logits.cpu().tolist()


def dcg(labels: List[int]) -> float:
    score = 0.0
    for i, rel in enumerate(labels, start=1):
        score += rel / math.log2(i + 1)
    return score


def ranking_metrics(grouped: Dict[str, List[dict]], ks: List[int]) -> Dict[str, float]:
    mrr_sum = 0.0
    map_sum = {k: 0.0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}
    recall_sum = {k: 0.0 for k in ks}
    precision_sum = {k: 0.0 for k in ks}
    total = 0

    for _, items in grouped.items():
        labels = [int(r.get("label", 0)) for r in items]
        if sum(labels) == 0:
            continue

        total += 1
        ranked = sorted(items, key=lambda r: r.get("ce_logit", 0.0), reverse=True)
        ranked_labels = [int(r.get("label", 0)) for r in ranked]

        rr = 0.0
        for i, rel in enumerate(ranked_labels, start=1):
            if rel > 0:
                rr = 1.0 / i
                break
        mrr_sum += rr

        total_relevant = sum(ranked_labels)
        for k in ks:
            topk = ranked_labels[:k]
            hits = sum(topk)
            recall_sum[k] += hits / total_relevant
            precision_sum[k] += hits / k

            ap = 0.0
            hit_count = 0
            for i, rel in enumerate(topk, start=1):
                if rel > 0:
                    hit_count += 1
                    ap += hit_count / i
            ap = ap / min(total_relevant, k)
            map_sum[k] += ap

            dcg_k = dcg(topk)
            ideal = sorted(ranked_labels, reverse=True)[:k]
            idcg_k = dcg(ideal)
            ndcg_sum[k] += (dcg_k / idcg_k) if idcg_k > 0 else 0.0

    if total == 0:
        return {}

    out = {"mrr": mrr_sum / total}
    for k in ks:
        out[f"map@{k}"] = map_sum[k] / total
        out[f"ndcg@{k}"] = ndcg_sum[k] / total
        out[f"recall@{k}"] = recall_sum[k] / total
        out[f"precision@{k}"] = precision_sum[k] / total
    return out


def main():
    p = argparse.ArgumentParser(description="Evaluate cross-encoder with DPR candidates")
    p.add_argument("--split_path", default=DEFAULT_SPLIT)
    p.add_argument("--passages", default=DEFAULT_PASSAGES)
    p.add_argument("--build_root", default=BUILD_ROOT)
    p.add_argument("--dpr_ckpt", default=DEFAULT_DPR_CKPT)
    p.add_argument("--dpr_base", default=DEFAULT_DPR_BASE)
    p.add_argument("--dpr_weights", default=DEFAULT_DPR_WEIGHTS)
    p.add_argument("--ce_ckpt", default=DEFAULT_CE_CKPT)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_q_len", type=int, default=128)
    p.add_argument("--max_ce_len", type=int, default=512)
    p.add_argument("--ks", default="1,2,3,4,5,6,7,8,9,10")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    build_dir = get_latest_build_dir(args.build_root)
    faiss_idx = os.path.join(build_dir, FAISS_NAME)
    idmap_path = os.path.join(build_dir, IDMAP_NAME)

    index = faiss.read_index(faiss_idx)
    with open(idmap_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)

    passages_map = load_passage_map(args.passages)
    split = read_json_list(args.split_path)

    dpr_tokenizer = AutoTokenizer.from_pretrained(args.dpr_ckpt, use_fast=False)
    dpr_encoder = DualEncoder(args.dpr_base).to(device)
    state = torch.load(os.path.join(args.dpr_ckpt, args.dpr_weights), map_location=device)
    dpr_encoder.load_state_dict(state, strict=True)
    dpr_encoder.eval()

    ce_tokenizer = AutoTokenizer.from_pretrained(args.ce_ckpt, use_fast=False)
    ce_cfg_path = os.path.join(args.ce_ckpt, "config.json")
    ce_backbone = args.dpr_base
    if os.path.exists(ce_cfg_path):
        with open(ce_cfg_path, "r", encoding="utf-8") as f:
            ce_cfg = json.load(f)
            ce_backbone = ce_cfg.get("backbone", ce_backbone)
    ce_model = CrossEncoderModel(ce_backbone).to(device)
    ce_state = torch.load(os.path.join(args.ce_ckpt, "model.pt"), map_location=device)
    ce_model.load_state_dict(ce_state, strict=True)
    ce_model.eval()

    grouped = defaultdict(list)
    for i in range(0, len(split), args.batch_size):
        batch = split[i : i + args.batch_size]
        questions = [ex["question"] for ex in batch]
        q_embs = encode_queries(questions, dpr_tokenizer, dpr_encoder, device, args.max_q_len)
        scores, indices = index.search(q_embs.astype("float32"), args.topk)

        for bi, ex in enumerate(batch):
            gold = ex["positive_passage_id"]
            qid = ex.get("qid", str(i + bi))
            for rank, idx in enumerate(indices[bi], start=1):
                pid = resolve_passage_id(id_map, int(idx))
                passage = passages_map.get(pid, {})
                grouped[qid].append(
                    {
                        "qid": qid,
                        "question": ex["question"],
                        "passage_id": pid,
                        "passage_text": passage.get("text", ""),
                        "label": 1 if pid == gold else 0,
                        "rank": rank,
                    }
                )

    # score with cross-encoder
    for qid, items in grouped.items():
        qs = [r["question"] for r in items]
        ps = [r["passage_text"] for r in items]
        logits = score_pairs(qs, ps, ce_tokenizer, ce_model, device, args.max_ce_len)
        for r, logit in zip(items, logits):
            r["ce_logit"] = float(logit)

    ks = [int(k.strip()) for k in args.ks.split(",") if k.strip()]
    scores = ranking_metrics(grouped, ks)

    print("Đánh giá reranker trên candidates DPR:")
    print(f"Đã đọc {len(split)} samples từ file {args.split_path}")
    if scores:
        print(f"  mrr={scores['mrr']:.4f}")
        for k in ks:
            # print(f"  map@{k}={scores[f'map@{k}']:.4f}")
            # print(f"  ndcg@{k}={scores[f'ndcg@{k}']:.4f}")
            print(f"  recall@{k}={scores[f'recall@{k}']:.4f}")
            # print(f"  precision@{k}={scores[f'precision@{k}']:.4f}")


if __name__ == "__main__":
    main()