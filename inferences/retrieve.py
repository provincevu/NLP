# retrieve.py
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

BASE_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
BUILD_ROOT = os.path.join(PROJECT_DIR, "build")

FAISS_NAME = "faiss_index.ivf"
IDMAP_NAME = "passage_id_map.json"
PASSAGES_PATH = os.path.join(PROJECT_DIR, "data", "passages.json")
MODEL_NAME_OR_PATH = os.path.join(BASE_DIR, "dpr_ckpt_1")
BASE_MODEL_NAME = "vinai/phobert-base"
MODEL_WEIGHTS = "model_last.pt"
TOPK = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_Q_LEN = 64

import faiss

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

def load_passages(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        return {p["passage_id"]: p for p in data["data"]}
    return {p["passage_id"]: p for p in data}

def encode_query(text, tokenizer, encoder):
    tok = tokenizer([text], padding=True, truncation=True, max_length=MAX_Q_LEN, return_tensors="pt")
    tok = {k: v.to(DEVICE) for k, v in tok.items()}
    with torch.no_grad():
        vec = encoder.encode(tok["input_ids"], tok["attention_mask"])
    return vec.cpu().numpy()

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

def main():
    build_dir = get_latest_build_dir(BUILD_ROOT)
    faiss_idx = os.path.join(build_dir, FAISS_NAME)
    idmap_path = os.path.join(build_dir, IDMAP_NAME)

    # load index + id_map + passages
    index = faiss.read_index(faiss_idx)
    id_map = json.load(open(idmap_path, "r", encoding="utf-8"))
    passages = load_passages(PASSAGES_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=False)
    encoder = DualEncoder(BASE_MODEL_NAME).to(DEVICE)
    state_path = os.path.join(MODEL_NAME_OR_PATH, MODEL_WEIGHTS)
    state = torch.load(state_path, map_location=DEVICE)
    encoder.load_state_dict(state, strict=True)
    encoder.eval()

    while True:
        q = input("\nQuery (enter to exit): ").strip()
        if not q:
            break
        q_vec = encode_query(q, tokenizer, encoder)  # shape (1,d)
        D, I = index.search(q_vec.astype("float32"), TOPK)  # I: indices
        for rank, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
            pid = id_map[idx]
            p = passages.get(pid, {"text": ""})
            print(f"\nRank {rank} - score={score:.4f} - passage_id={pid}\n{p.get('clause_ref','')}\n{p.get('text','')[:500]}")

if __name__ == "__main__":
    main()