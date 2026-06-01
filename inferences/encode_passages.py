# encode_passages.py
import json
import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

torch.set_num_threads(4)

BASE_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODEL_NAME_OR_PATH = os.path.join(BASE_DIR, "dpr_ckpt_1")
BASE_MODEL_NAME = "vinai/phobert-base"
MODEL_WEIGHTS = "model_last.pt"
PASSAGES_PATH = os.path.join(PROJECT_DIR, "data", "passages.json")
BUILD_ROOT = os.path.join(PROJECT_DIR, "build")
OUT_EMB_NAME = "passage_embeddings.npy"
OUT_IDMAP_NAME = "passage_id_map.json"
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_P_LEN = 512

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # hỗ trợ list hoặc {"data": [...]}
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return data

def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom

def make_build_dir(build_root: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(build_root, ts)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

class DualEncoder(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.proj = torch.nn.Linear(self.encoder.config.hidden_size, 768)

    def encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        vec = mean_pool(out.last_hidden_state, attention_mask)
        vec = self.proj(vec)
        vec = F.normalize(vec, p=2, dim=1)
        return vec

def main():
    passages = read_json(PASSAGES_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=False)
    encoder = DualEncoder(BASE_MODEL_NAME).to(DEVICE)
    state_path = os.path.join(MODEL_NAME_OR_PATH, MODEL_WEIGHTS)
    state = torch.load(state_path, map_location=DEVICE)
    encoder.load_state_dict(state, strict=True)
    encoder.eval()

    id_map = []
    embs = []

    with torch.no_grad():
        for i in range(0, len(passages), BATCH_SIZE):
            batch = passages[i:i+BATCH_SIZE]
            texts = [p["text"] for p in batch]
            toks = tokenizer(texts, padding=True, truncation=True, max_length=MAX_P_LEN, return_tensors="pt")
            toks = {k: v.to(DEVICE) for k, v in toks.items()}
            vec = encoder.encode(toks["input_ids"], toks["attention_mask"])
            embs.append(vec.cpu().numpy())
            id_map.extend([p["passage_id"] for p in batch])

    embs = np.vstack(embs)
    out_dir = make_build_dir(BUILD_ROOT)
    out_emb = os.path.join(out_dir, OUT_EMB_NAME)
    out_idmap = os.path.join(out_dir, OUT_IDMAP_NAME)
    np.save(out_emb, embs)
    with open(out_idmap, "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False)
    print("Saved embeddings:", out_emb, "shape:", embs.shape)
    print("Saved id_map:", out_idmap)

if __name__ == "__main__":
    main()