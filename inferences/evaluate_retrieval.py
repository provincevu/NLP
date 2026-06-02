# evaluate_retrieval.py
import json, numpy as np
import os
from transformers import AutoTokenizer, AutoModel
import torch, torch.nn.functional as F
import faiss

BASE_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
BUILD_ROOT = os.path.join(PROJECT_DIR, "build")

MODEL = os.path.join(BASE_DIR, "dpr_ckpt_1")
BASE_MODEL_NAME = "vinai/phobert-base"
MODEL_WEIGHTS = "model_last.pt"
PASSAGES = os.path.join(PROJECT_DIR, "data", "passages.json")
IDMAP_NAME = "passage_id_map.json"
FAISS_NAME = "faiss_index.ivf"
TEST_Q = os.path.join(PROJECT_DIR, "data", "splits", "train.json")
BATCH_Q = 64
TOPK = list(range(1, 21))

def read_json_list(path):
    with open(path,"r",encoding="utf-8") as f:
        j = json.load(f)
    if isinstance(j, dict) and "data" in j:
        return j["data"]
    return j

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

def load_passage_map(path):
    ps = read_json_list(path)
    return {p["passage_id"]: p for p in ps}

def encode_batch(queries, tokenizer, encoder, device):
    toks = tokenizer(queries, padding=True, truncation=True, max_length=64, return_tensors="pt")
    toks = {k: v.to(device) for k,v in toks.items()}
    with torch.no_grad():
        vec = encoder.encode(toks["input_ids"], toks["attention_mask"])
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

    index = faiss.read_index(faiss_idx)
    id_map = json.load(open(idmap_path,"r",encoding="utf-8"))
    test = read_json_list(TEST_Q)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = DualEncoder(BASE_MODEL_NAME).to(device)
    state_path = f"{MODEL}/{MODEL_WEIGHTS}"
    state = torch.load(state_path, map_location=device)
    encoder.load_state_dict(state, strict=True)
    encoder.eval()

    total = len(test)
    hits = {k:0 for k in TOPK}
    mrr_sum = 0.0

    for i in range(0, total, BATCH_Q):
        batch = test[i:i+BATCH_Q]
        queries = [ex["question"] for ex in batch]
        q_embs = encode_batch(queries, tokenizer, encoder, device)
        D, I = index.search(q_embs.astype("float32"), max(TOPK))
        for qi, ex in enumerate(batch):
            gold = ex["positive_passage_id"]
            ranks = I[qi].tolist()
            # check hits
            found_rank = None
            for r, idx in enumerate(ranks, start=1):
                if id_map[idx] == gold:
                    found_rank = r
                    break
            if found_rank is not None:
                for k in TOPK:
                    if found_rank <= k:
                        hits[k] += 1
                mrr_sum += 1.0 / found_rank
    # print metrics
    print("N:", total)
    for k in TOPK:
        print(f"Recall@{k}: {hits[k]/total:.4f}")
    print("MRR:", mrr_sum/total)

if __name__ == "__main__":
    main()