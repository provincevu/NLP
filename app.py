import json
import os
import sys
from pathlib import Path
from typing import List

import torch
from flask import Flask, render_template, request
from transformers import AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
RERANK_DIR = BASE_DIR / "rerank"
if str(RERANK_DIR) not in sys.path:
    sys.path.insert(0, str(RERANK_DIR))

from train_cross_encoder4colab import CrossEncoderModel  # noqa: E402


CKPT_DIR = BASE_DIR / "rerank" / "ckpt" / "best"
PASSAGES_PATH = BASE_DIR / "data" / "passages.json"
DEFAULT_TOP_K = 3
DEFAULT_BATCH_SIZE = 8


app = Flask(__name__)


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


PASSAGES = load_passages(PASSAGES_PATH)
TOKENIZER, MODEL, DEVICE = load_cross_encoder()


def score_passages(question: str, top_k: int = DEFAULT_TOP_K, batch_size: int = DEFAULT_BATCH_SIZE) -> List[dict]:
    scored = []
    max_len = 512

    with torch.no_grad():
        for i in range(0, len(PASSAGES), batch_size):
            batch = PASSAGES[i : i + batch_size]
            questions = [question] * len(batch)
            texts = [p["text"] for p in batch]
            toks = TOKENIZER(
                questions,
                texts,
                padding=True,
                truncation=True,
                max_length=max_len,
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
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=False)
