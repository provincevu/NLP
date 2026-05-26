# retrieve.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

FAISS_INDEX = "../data/faiss_index_last.ivf"
EMB_PATH = "../data/passage_embeddings.npy"
IDMAP_PATH = "../data/passage_id_map.json"
PASSAGES_PATH = "../data/passages.json"
MODEL_NAME_OR_PATH = "dpr_ckpt_1"
TOPK = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_Q_LEN = 64

import faiss

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
        out = encoder(**tok)
        last = out.last_hidden_state
        mask = tok["attention_mask"].unsqueeze(-1).float()
        vec = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        vec = F.normalize(vec, p=2, dim=1)
    return vec.cpu().numpy()

def main():
    # load index + id_map + passages
    index = faiss.read_index(FAISS_INDEX)
    id_map = json.load(open(IDMAP_PATH, "r", encoding="utf-8"))
    passages = load_passages(PASSAGES_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=False)
    encoder = AutoModel.from_pretrained(MODEL_NAME_OR_PATH).to(DEVICE)
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