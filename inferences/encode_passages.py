# encode_passages.py
import json
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

torch.set_num_threads(4)

MODEL_NAME_OR_PATH = "dpr_ckpt_1"  # thư mục checkpoint bạn đã lưu
PASSAGES_PATH = "../data/passages.json"  # hoặc /content/data/passages.json
OUT_EMB = "../data/passage_embeddings.npy"
OUT_IDMAP = "../data/passage_id_map.json"
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_P_LEN = 256

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

def main():
    passages = read_json(PASSAGES_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=False)
    encoder = AutoModel.from_pretrained(MODEL_NAME_OR_PATH).to(DEVICE)
    encoder.eval()

    id_map = []
    embs = []

    with torch.no_grad():
        for i in range(0, len(passages), BATCH_SIZE):
            batch = passages[i:i+BATCH_SIZE]
            texts = [p["text"] for p in batch]
            toks = tokenizer(texts, padding=True, truncation=True, max_length=MAX_P_LEN, return_tensors="pt")
            toks = {k: v.to(DEVICE) for k, v in toks.items()}
            out = encoder(**toks)
            vec = mean_pool(out.last_hidden_state, toks["attention_mask"])
            vec = F.normalize(vec, p=2, dim=1)  # L2 normalize
            embs.append(vec.cpu().numpy())
            id_map.extend([p["passage_id"] for p in batch])

    embs = np.vstack(embs)
    os.makedirs(os.path.dirname(OUT_EMB), exist_ok=True)
    np.save(OUT_EMB, embs)
    with open(OUT_IDMAP, "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False)
    print("Saved embeddings:", OUT_EMB, "shape:", embs.shape)
    print("Saved id_map:", OUT_IDMAP)

if __name__ == "__main__":
    main()