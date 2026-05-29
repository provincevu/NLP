# evaluate_retrieval.py
import json, numpy as np
from transformers import AutoTokenizer, AutoModel
import torch, torch.nn.functional as F
import faiss

MODEL = "inferences/dpr_ckpt_1"
PASSAGES = "data/passages.json"
IDMAP = "data/passage_id_map.json"
FAISS_IDX = "data/faiss_index_last.ivf"
TEST_Q = "data/splits/test.json"
BATCH_Q = 32
TOPK = [1,5,10,20]

def read_json_list(path):
    with open(path,"r",encoding="utf-8") as f:
        j = json.load(f)
    if isinstance(j, dict) and "data" in j:
        return j["data"]
    return j

def load_passage_map(path):
    ps = read_json_list(path)
    return {p["passage_id"]: p for p in ps}

def encode_batch(queries, tokenizer, encoder, device):
    toks = tokenizer(queries, padding=True, truncation=True, max_length=64, return_tensors="pt")
    toks = {k: v.to(device) for k,v in toks.items()}
    with torch.no_grad():
        out = encoder(**toks)
        mask = toks["attention_mask"].unsqueeze(-1).float()
        vec = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        vec = F.normalize(vec, p=2, dim=1)
    return vec.cpu().numpy()

def main():
    index = faiss.read_index(FAISS_IDX)
    id_map = json.load(open(IDMAP,"r",encoding="utf-8"))
    test = read_json_list(TEST_Q)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    encoder = AutoModel.from_pretrained(MODEL).to("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"

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