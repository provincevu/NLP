# build_faiss.py
import faiss
import numpy as np
import json
import os

BASE_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
BUILD_ROOT = os.path.join(PROJECT_DIR, "build")

EMB_NAME = "passage_embeddings.npy"
IDMAP_NAME = "passage_id_map.json"
FAISS_NAME = "faiss_index.ivf"  # hoặc .index

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

def main():
    build_dir = get_latest_build_dir(BUILD_ROOT)
    emb_path = os.path.join(build_dir, EMB_NAME)
    idmap_path = os.path.join(build_dir, IDMAP_NAME)
    faiss_out = os.path.join(build_dir, FAISS_NAME)

    embs = np.load(emb_path)
    with open(idmap_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)

    # embs assumed normalized (L2). If not, normalize here:
    # norms = np.linalg.norm(embs, axis=1, keepdims=True)
    # embs = embs / np.maximum(norms, 1e-12)

    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)  # exact inner product
    index.add(embs)               # add all vectors
    print("Added", index.ntotal, "vectors to FAISS index")

    # save index
    faiss.write_index(index, faiss_out)
    # also save id_map alongside - but we already have it
    print("Saved index to", faiss_out)

if __name__ == "__main__":
    main()