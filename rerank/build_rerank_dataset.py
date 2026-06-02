import argparse
import json
import os
from typing import Dict, List

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

BASE_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

DEFAULT_SPLITS_DIR = os.path.join(PROJECT_DIR, "data", "splits")
DEFAULT_PASSAGES = os.path.join(PROJECT_DIR, "data", "passages.json")
DEFAULT_BUILD_ROOT = os.path.join(PROJECT_DIR, "build")
DEFAULT_MODEL_DIR = os.path.join(PROJECT_DIR, "inferences", "dpr_ckpt_1")
DEFAULT_BASE_MODEL = "vinai/phobert-base"
DEFAULT_MODEL_WEIGHTS = "model_last.pt"
DEFAULT_OUT_DIR = os.path.join(PROJECT_DIR, "rerank", "data")
DEFAULT_FAISS_NAME = "faiss_index.ivf"
DEFAULT_IDMAP_NAME = "passage_id_map.json"


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


class DualEncoder(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.proj = torch.nn.Linear(self.encoder.config.hidden_size, 768)

    def encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        vec = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        vec = self.proj(vec)
        vec = F.normalize(vec, p=2, dim=1)
        return vec


def encode_batch(questions, tokenizer, encoder, device, max_q_len: int):
    toks = tokenizer(
        questions,
        padding=True,
        truncation=True,
        max_length=max_q_len,
        return_tensors="pt",
    )
    toks = {k: v.to(device) for k, v in toks.items()}
    with torch.no_grad():
        vec = encoder.encode(toks["input_ids"], toks["attention_mask"])
    return vec.cpu().numpy()


def build_split(
    split_path: str,
    out_path: str,
    index,
    id_map,
    passages_map,
    tokenizer,
    encoder,
    device,
    topk: int,
    batch_size: int,
    max_q_len: int,
):
    data = read_json_list(split_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    total_questions = len(data)
    kept_questions = 0
    skipped_questions = 0
    all_rows = []
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        questions = [ex["question"] for ex in batch]
        q_embs = encode_batch(questions, tokenizer, encoder, device, max_q_len)
        scores, indices = index.search(q_embs.astype("float32"), topk)

        for bi, ex in enumerate(batch):
            gold = ex["positive_passage_id"]
            retrieved_ids = [resolve_passage_id(id_map, int(idx)) for idx in indices[bi]]
            if gold not in retrieved_ids:
                skipped_questions += 1
                continue
            kept_questions += 1
            for rank, (idx, score) in enumerate(zip(indices[bi], scores[bi]), start=1):
                passage_id = resolve_passage_id(id_map, int(idx))
                passage = passages_map.get(passage_id, {})
                label = 1 if passage_id == gold else 0
                all_rows.append(
                    {
                        "qid": ex.get("qid"),
                        "question": ex["question"],
                        "passage_id": passage_id,
                        "passage_text": passage.get("text", ""),
                        "label": label,
                        "rank": rank,
                        "score": float(score),
                        "positive_passage_id": gold,
                    }
                )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    return {
        "rows": len(all_rows),
        "total_questions": total_questions,
        "kept_questions": kept_questions,
        "skipped_questions": skipped_questions,
    }


def main():
    parser = argparse.ArgumentParser(description="Build cross-encoder rerank dataset from DPR candidates")
    parser.add_argument("--splits_dir", default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--passages", default=DEFAULT_PASSAGES)
    parser.add_argument("--build_root", default=DEFAULT_BUILD_ROOT)
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--model_weights", default=DEFAULT_MODEL_WEIGHTS)
    parser.add_argument("--faiss_name", default=DEFAULT_FAISS_NAME)
    parser.add_argument("--idmap_name", default=DEFAULT_IDMAP_NAME)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_q_len", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    build_dir = get_latest_build_dir(args.build_root)
    faiss_idx = os.path.join(build_dir, args.faiss_name)
    idmap_path = os.path.join(build_dir, args.idmap_name)

    index = faiss.read_index(faiss_idx)
    with open(idmap_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)

    passages_map = load_passage_map(args.passages)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    encoder = DualEncoder(args.base_model).to(args.device)
    state = torch.load(os.path.join(args.model_dir, args.model_weights), map_location=args.device)
    encoder.load_state_dict(state, strict=True)
    encoder.eval()

    splits = ["train", "val", "test"]
    for split in splits:
        split_path = os.path.join(args.splits_dir, f"{split}.json")
        out_path = os.path.join(args.out_dir, f"{split}.json")
        if not os.path.exists(split_path):
            continue
        stats = build_split(
            split_path,
            out_path,
            index,
            id_map,
            passages_map,
            tokenizer,
            encoder,
            args.device,
            args.topk,
            args.batch_size,
            args.max_q_len,
        )
        total_questions = stats["total_questions"]
        kept_questions = stats["kept_questions"]
        skipped_questions = stats["skipped_questions"]
        rows = stats["rows"]
        keep_ratio = (kept_questions / total_questions) if total_questions else 0.0
        skip_ratio = (skipped_questions / total_questions) if total_questions else 0.0

        print(f"Đã lưu {rows} dòng vào: {out_path}")
        print(
            "Thống kê split {split}: Tổng câu hỏi={total}, Giữ lại={kept}, "
            "Bị loại={skipped}, Tỉ lệ giữ={keep:.2%}, Tỉ lệ loại={skip:.2%}".format(
                split=split,
                total=total_questions,
                kept=kept_questions,
                skipped=skipped_questions,
                keep=keep_ratio,
                skip=skip_ratio,
            )
        )


if __name__ == "__main__":
    main()
