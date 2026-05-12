# build_faiss.py
import faiss
import numpy as np
import json
import os

EMB_PATH = "../data/passage_embeddings.npy"
IDMAP_PATH = "../data/passage_id_map.json"
FAISS_OUT = "../data/faiss_index_last.ivf"  # hoặc .index

def main():
    embs = np.load(EMB_PATH)
    with open(IDMAP_PATH, "r", encoding="utf-8") as f:
        id_map = json.load(f)

    # embs assumed normalized (L2). If not, normalize here:
    # norms = np.linalg.norm(embs, axis=1, keepdims=True)
    # embs = embs / np.maximum(norms, 1e-12)

    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)  # exact inner product
    index.add(embs)               # add all vectors
    print("Added", index.ntotal, "vectors to FAISS index")

    # save index
    os.makedirs(os.path.dirname(FAISS_OUT), exist_ok=True)
    faiss.write_index(index, FAISS_OUT)
    # also save id_map alongside - but we already have it
    print("Saved index to", FAISS_OUT)

if __name__ == "__main__":
    main()