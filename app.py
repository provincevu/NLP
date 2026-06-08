import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix OpenMP conflict between PyTorch and FAISS on macOS

import json
import sys
from pathlib import Path
from typing import List

import faiss
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request
from transformers import AutoModel, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
RERANK_DIR = BASE_DIR / "rerank"
if str(RERANK_DIR) not in sys.path:
    sys.path.insert(0, str(RERANK_DIR))

from train_cross_encoder4colab import CrossEncoderModel  # noqa: E402


CKPT_DIR = BASE_DIR / "rerank" / "ckpt" / "cross_encoder_ckpt" / "best"
PASSAGES_PATH = BASE_DIR / "data" / "passages.json"
BUILD_ROOT = BASE_DIR / "build"
DPR_CKPT_DIR = BASE_DIR / "inferences" / "dpr_ckpt_1"
DPR_BASE_MODEL = "vinai/phobert-base"
DPR_WEIGHTS = "model_last.pt"
FAISS_NAME = "faiss_index.ivf"
IDMAP_NAME = "passage_id_map.json"
DEFAULT_TOP_K = 10
DEFAULT_DPR_TOP_K = 10
DEFAULT_BATCH_SIZE = 8
MAX_Q_LEN = 128
MAX_CE_LEN = 512


app = Flask(__name__)


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
        return F.normalize(vec, p=2, dim=1)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_passages(path: Path) -> List[dict]:
    data = load_json(path)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]

    passages = []
    for idx, item in enumerate(data):
        text = item.get("text") or item.get("passage_text") or ""
        if not text:
            continue
        passages.append(
            {
                "passage_id": item.get("passage_id", f"passage_{idx + 1}"),
                "clause_ref": item.get("clause_ref", ""),
                "text": text,
            }
        )
    return passages


def get_latest_build_dir(build_root: Path) -> Path:
    if not build_root.is_dir():
        raise FileNotFoundError(f"Build folder not found: {build_root}")
    candidates = [p for p in build_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError("No build subfolders found. Run inferences/encode_passages.py first.")
    return sorted(candidates, key=lambda p: p.name)[-1]


def load_backbone(ckpt_dir: Path) -> str:
    cfg_path = ckpt_dir / "config.json"
    if not cfg_path.exists():
        return "vinai/phobert-base"
    cfg = load_json(cfg_path)
    return cfg.get("backbone") or "vinai/phobert-base"


def load_cross_encoder():
    if not CKPT_DIR.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {CKPT_DIR}")
    model_path = CKPT_DIR / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = load_backbone(CKPT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(str(CKPT_DIR), use_fast=False)
    model = CrossEncoderModel(backbone).to(device)
    state = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return tokenizer, model, device


def load_dpr():
    build_dir = get_latest_build_dir(BUILD_ROOT)
    faiss_path = build_dir / FAISS_NAME
    idmap_path = build_dir / IDMAP_NAME
    weights_path = DPR_CKPT_DIR / DPR_WEIGHTS

    if not faiss_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
    if not idmap_path.exists():
        raise FileNotFoundError(f"Passage id map not found: {idmap_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"DPR weights not found: {weights_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = faiss.read_index(str(faiss_path))
    id_map = load_json(idmap_path)
    tokenizer = AutoTokenizer.from_pretrained(str(DPR_CKPT_DIR), use_fast=False)
    encoder = DualEncoder(DPR_BASE_MODEL).to(device)
    state = torch.load(str(weights_path), map_location=device)
    encoder.load_state_dict(state, strict=True)
    encoder.eval()
    return tokenizer, encoder, index, id_map, build_dir, device


print("Loading passages...")
PASSAGES = load_passages(PASSAGES_PATH)
print("Passages loaded.")
PASSAGE_BY_ID = {p["passage_id"]: p for p in PASSAGES}
print("Loading cross encoder...")
TOKENIZER, MODEL, DEVICE = load_cross_encoder()
print("Cross encoder loaded.")
print("Loading dpr...")
DPR_TOKENIZER, DPR_ENCODER, FAISS_INDEX, DPR_ID_MAP, DPR_BUILD_DIR, DPR_DEVICE = load_dpr()
print("DPR loaded.")


def retrieve_candidates(question: str, dpr_top_k: int = DEFAULT_DPR_TOP_K) -> List[dict]:
    search_k = min(max(dpr_top_k, DEFAULT_TOP_K), FAISS_INDEX.ntotal)
    toks = DPR_TOKENIZER(
        [question],
        padding=True,
        truncation=True,
        max_length=MAX_Q_LEN,
        return_tensors="pt",
    )
    toks = {k: v.to(DPR_DEVICE) for k, v in toks.items()}
    with torch.no_grad():
        q_vec = DPR_ENCODER.encode(toks["input_ids"], toks["attention_mask"])
    scores, indices = FAISS_INDEX.search(q_vec.cpu().numpy().astype("float32"), search_k)

    candidates = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        if idx < 0:
            continue
        passage_id = DPR_ID_MAP[idx]
        passage = PASSAGE_BY_ID.get(passage_id)
        if not passage:
            continue
        candidates.append(
            {
                **passage,
                "dpr_rank": rank,
                "dpr_score": float(score),
            }
        )
    return candidates


def score_passages(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    dpr_top_k: int = DEFAULT_DPR_TOP_K,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[dict]:
    candidates = retrieve_candidates(question, dpr_top_k=dpr_top_k)
    scored = []

    with torch.no_grad():
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i : i + batch_size]
            questions = [question] * len(batch)
            texts = [p["text"] for p in batch]
            toks = TOKENIZER(
                questions,
                texts,
                padding=True,
                truncation=True,
                max_length=MAX_CE_LEN,
                return_tensors="pt",
            )
            toks = {k: v.to(DEVICE) for k, v in toks.items()}
            logits = MODEL(toks["input_ids"], toks["attention_mask"])
            probs = torch.sigmoid(logits).detach().cpu().tolist()
            raw_logits = logits.detach().cpu().tolist()

            for passage, logit, prob in zip(batch, raw_logits, probs):
                scored.append(
                    {
                        **passage,
                        "logit": float(logit),
                        "prob": float(prob),
                    }
                )

    scored.sort(key=lambda row: row["logit"], reverse=True)
    return scored[:top_k]


def parse_positive_int(value, default: int, upper_bound: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    parsed = max(1, parsed)
    if upper_bound is not None:
        parsed = min(parsed, upper_bound)
    return parsed


@app.route("/", methods=["GET", "POST"])
def index():
    question = ""
    results = []
    error = ""

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if not question:
            error = "Vui long nhap cau hoi."
        else:
            results = score_passages(question)

    return render_template(
        "index.html",
        question=question,
        results=results,
        error=error,
        passage_count=len(PASSAGES),
        device=DEVICE,
        dpr_top_k=DEFAULT_DPR_TOP_K,
        dpr_build_dir=DPR_BUILD_DIR.name,
    )


@app.route("/api/search", methods=["GET", "POST"])
def api_search():
    payload = request.get_json(silent=True) or {}
    question = (
        payload.get("question")
        or request.form.get("question")
        or request.args.get("question")
        or ""
    ).strip()

    if not question:
        return jsonify({"error": "Missing required field: question"}), 400

    top_k = parse_positive_int(
        payload.get("top_k") or request.form.get("top_k") or request.args.get("top_k"),
        DEFAULT_TOP_K,
        upper_bound=20,
    )
    dpr_top_k = parse_positive_int(
        payload.get("dpr_top_k") or request.form.get("dpr_top_k") or request.args.get("dpr_top_k"),
        DEFAULT_DPR_TOP_K,
        upper_bound=len(PASSAGES),
    )
    dpr_top_k = max(dpr_top_k, top_k)

    results = score_passages(question, top_k=top_k, dpr_top_k=dpr_top_k)
    return jsonify(
        {
            "question": question,
            "top_k": top_k,
            "dpr_top_k": dpr_top_k,
            "dpr_build_dir": DPR_BUILD_DIR.name,
            "results": results,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=False)
