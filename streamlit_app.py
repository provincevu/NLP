"""
Streamlit Search Portal – DPR + Cross-Encoder Retrieval
========================================================
Giao diện Search Portal chuyên nghiệp – Premium Dark UI.

Chạy:
    OMP_NUM_THREADS=1 streamlit run streamlit_app.py --server.port 8501
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import sys
import time
from pathlib import Path
from typing import List

import faiss
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
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
DEFAULT_TOP_K = 5
DEFAULT_DPR_TOP_K = 10
MAX_Q_LEN = 128
MAX_CE_LEN = 512

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HUMG – Hỏi-đáp Quy chế Đào tạo",
    page_icon="🎓",
    layout="centered",
)

# ---------------------------------------------------------------------------
# CSS Theme – Premium Dark UI with Glassmorphism
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── CSS Variables ─────────────────────────────────────── */
:root {
    --bg-primary:    #060b18;
    --bg-secondary:  #0d1526;
    --bg-card:       rgba(13, 21, 38, 0.85);
    --accent-1:      #4f8ef7;
    --accent-2:      #7c3aed;
    --accent-3:      #06b6d4;
    --accent-gold:   #f59e0b;
    --text-primary:  #f1f5f9;
    --text-secondary:#94a3b8;
    --text-muted:    #475569;
    --border:        rgba(79, 142, 247, 0.15);
    --border-hover:  rgba(79, 142, 247, 0.4);
    --glow:          rgba(79, 142, 247, 0.12);
    --success:       #10b981;
    --warning:       #f59e0b;
    --danger:        #ef4444;
}

/* ── Global Reset ──────────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif !important;
    min-height: 100vh;
}

/* Animated gradient background mesh */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(79,142,247,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 80% at 80% 90%, rgba(124,58,237,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 50% 50% at 50% 50%, rgba(6,182,212,0.04) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* ── Hide Streamlit chrome ─────────────────────────────── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stSidebar"],
[data-testid="stDecoration"] {
    display: none !important;
}

/* ── Main block wrapper ────────────────────────────────── */
[data-testid="stMainBlockContainer"],
.main .block-container {
    max-width: 860px !important;
    padding: 2rem 1.5rem 4rem !important;
    position: relative;
    z-index: 1;
}

/* ── Typography ────────────────────────────────────────── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    color: var(--text-secondary) !important;
    line-height: 1.75 !important;
}
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="stMarkdownContainer"] strong {
    color: #93c5fd !important;
    font-weight: 600 !important;
}
[data-testid="stMarkdownContainer"] code {
    background: rgba(79,142,247,0.12) !important;
    color: #93c5fd !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
    font-size: 0.85em !important;
}

/* ── Search Form ───────────────────────────────────────── */
[data-testid="stForm"] {
    background: rgba(13, 21, 38, 0.9) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 20px !important;
    padding: 10px 10px 10px 22px !important;
    box-shadow:
        0 8px 32px rgba(0,0,0,0.4),
        0 0 0 1px rgba(255,255,255,0.03),
        inset 0 1px 0 rgba(255,255,255,0.05) !important;
    backdrop-filter: blur(20px) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
[data-testid="stForm"]:focus-within {
    border-color: var(--border-hover) !important;
    box-shadow:
        0 8px 32px rgba(0,0,0,0.4),
        0 0 0 1px rgba(255,255,255,0.05),
        0 0 40px rgba(79,142,247,0.12),
        inset 0 1px 0 rgba(255,255,255,0.05) !important;
}

/* ── Search Input ──────────────────────────────────────── */
[data-testid="stForm"] [data-testid="stTextInput"] > div > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stForm"] [data-testid="stTextInput"] input {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15.5px !important;
    font-weight: 400 !important;
    padding: 12px 4px !important;
    height: 50px !important;
    box-shadow: none !important;
    letter-spacing: 0.01em !important;
}
[data-testid="stForm"] [data-testid="stTextInput"] input:focus {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}
[data-testid="stForm"] [data-testid="stTextInput"] input::placeholder {
    color: var(--text-muted) !important;
    font-weight: 400 !important;
    font-style: italic !important;
}
[data-testid="stTextInput"] label,
[data-testid="stForm"] [data-testid="stTextInput"] label {
    display: none !important;
}

/* ── Submit Button ─────────────────────────────────────── */
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #4f8ef7 0%, #7c3aed 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    height: 50px !important;
    width: 100% !important;
    min-width: unset !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.4px !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow:
        0 4px 15px rgba(79,142,247,0.35),
        inset 0 1px 0 rgba(255,255,255,0.15) !important;
    cursor: pointer !important;
    position: relative !important;
    overflow: hidden !important;
}
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button:hover {
    background: linear-gradient(135deg, #6ba5ff 0%, #9333ea 100%) !important;
    box-shadow:
        0 8px 30px rgba(79,142,247,0.5),
        inset 0 1px 0 rgba(255,255,255,0.2) !important;
    transform: translateY(-2px) !important;
}
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button:active {
    transform: translateY(0) !important;
}

/* ── Example chip buttons ──────────────────────────────── */
[data-testid="stButton"] > button {
    background: rgba(79, 142, 247, 0.07) !important;
    color: #93c5fd !important;
    border: 1px solid rgba(79, 142, 247, 0.2) !important;
    border-radius: 100px !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 6px 16px !important;
    transition: all 0.2s ease !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
[data-testid="stButton"] > button:hover {
    background: rgba(79, 142, 247, 0.18) !important;
    border-color: rgba(79, 142, 247, 0.45) !important;
    color: #bfdbfe !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(79,142,247,0.2) !important;
}

/* ── New-search button override ────────────────────────── */
.new-search-btn [data-testid="stButton"] > button {
    background: rgba(16, 185, 129, 0.08) !important;
    color: #6ee7b7 !important;
    border: 1px solid rgba(16, 185, 129, 0.2) !important;
    border-radius: 12px !important;
    font-size: 14px !important;
    padding: 10px 20px !important;
}
.new-search-btn [data-testid="stButton"] > button:hover {
    background: rgba(16, 185, 129, 0.18) !important;
    border-color: rgba(16, 185, 129, 0.4) !important;
    color: #a7f3d0 !important;
    box-shadow: 0 4px 20px rgba(16,185,129,0.2) !important;
}

/* ── Expander (Result cards) ───────────────────────────── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    margin-bottom: 14px !important;
    overflow: hidden !important;
    backdrop-filter: blur(12px) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow:
        0 4px 20px rgba(0,0,0,0.3),
        inset 0 1px 0 rgba(255,255,255,0.04) !important;
}
[data-testid="stExpander"]:hover {
    border-color: rgba(79, 142, 247, 0.28) !important;
    box-shadow:
        0 8px 32px rgba(0,0,0,0.4),
        0 0 20px rgba(79,142,247,0.08),
        inset 0 1px 0 rgba(255,255,255,0.06) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    padding: 16px 20px !important;
    letter-spacing: 0.01em !important;
}
[data-testid="stExpander"] summary:hover {
    color: #93c5fd !important;
}
[data-testid="stExpander"] > div > div > div {
    padding: 0 20px 20px !important;
}

/* ── Metrics ───────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(79, 142, 247, 0.05) !important;
    border: 1px solid rgba(79, 142, 247, 0.1) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
}
[data-testid="stMetricValue"] {
    color: #93c5fd !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* ── Progress bar ──────────────────────────────────────── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #4f8ef7, #7c3aed, #06b6d4) !important;
    border-radius: 6px !important;
    box-shadow: 0 0 10px rgba(79,142,247,0.4) !important;
}
.stProgress > div > div {
    background: rgba(79,142,247,0.08) !important;
    border-radius: 6px !important;
    height: 6px !important;
}

/* ── Divider ───────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid rgba(79,142,247,0.1) !important;
    margin: 2rem 0 !important;
}

/* ── Caption ───────────────────────────────────────────── */
[data-testid="stCaptionContainer"] {
    color: var(--text-muted) !important;
    font-size: 11.5px !important;
}

/* ── Spinner ───────────────────────────────────────────── */
[data-testid="stSpinner"] { color: #4f8ef7 !important; }

/* ── Alert / Info ──────────────────────────────────────── */
[data-testid="stAlert"] {
    background: rgba(79,142,247,0.07) !important;
    border: 1px solid rgba(79,142,247,0.18) !important;
    border-radius: 14px !important;
    color: #93c5fd !important;
}

/* ── Selectbox / Number input ──────────────────────────── */
[data-testid="stSelectbox"] > div,
[data-testid="stNumberInput"] > div > div {
    background: rgba(13, 21, 38, 0.8) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}

/* ── Scrollbar ─────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(79,142,247,0.2);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(79,142,247,0.4); }

/* ── Rank badge colors ─────────────────────────────────── */
.rank-1 { color: #fbbf24 !important; }
.rank-2 { color: #94a3b8 !important; }
.rank-3 { color: #cd7c2f !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
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
        raise FileNotFoundError("No build subfolders found.")
    return sorted(candidates, key=lambda p: p.name)[-1]


def load_backbone(ckpt_dir: Path) -> str:
    cfg_path = ckpt_dir / "config.json"
    if not cfg_path.exists():
        return "vinai/phobert-base"
    cfg = load_json(cfg_path)
    return cfg.get("backbone") or "vinai/phobert-base"


@st.cache_resource(show_spinner="⚡ Đang khởi tạo mô hình AI…")
def init_all():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    passages = load_passages(PASSAGES_PATH)
    passage_by_id = {p["passage_id"]: p for p in passages}

    backbone = load_backbone(CKPT_DIR)
    ce_tok = AutoTokenizer.from_pretrained(str(CKPT_DIR), use_fast=False)
    ce_model = CrossEncoderModel(backbone).to(device)
    ce_state = torch.load(str(CKPT_DIR / "model.pt"), map_location=device)
    ce_model.load_state_dict(ce_state, strict=True)
    ce_model.eval()

    build_dir = get_latest_build_dir(BUILD_ROOT)
    index = faiss.read_index(str(build_dir / FAISS_NAME))
    id_map = load_json(build_dir / IDMAP_NAME)
    dpr_tok = AutoTokenizer.from_pretrained(str(DPR_CKPT_DIR), use_fast=False)
    dpr_enc = DualEncoder(DPR_BASE_MODEL).to(device)
    dpr_state = torch.load(str(DPR_CKPT_DIR / DPR_WEIGHTS), map_location=device)
    dpr_enc.load_state_dict(dpr_state, strict=True)
    dpr_enc.eval()

    return {
        "passages": passages,
        "passage_by_id": passage_by_id,
        "ce_tok": ce_tok,
        "ce_model": ce_model,
        "dpr_tok": dpr_tok,
        "dpr_enc": dpr_enc,
        "index": index,
        "id_map": id_map,
        "device": device,
    }


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def do_search(question: str, ctx: dict, top_k: int = DEFAULT_TOP_K) -> List[dict]:
    device = ctx["device"]
    search_k = min(max(DEFAULT_DPR_TOP_K, top_k), ctx["index"].ntotal)
    toks = ctx["dpr_tok"](
        [question], padding=True, truncation=True, max_length=MAX_Q_LEN, return_tensors="pt"
    )
    toks = {k: v.to(device) for k, v in toks.items()}
    with torch.no_grad():
        q_vec = ctx["dpr_enc"].encode(toks["input_ids"], toks["attention_mask"])
    scores, indices = ctx["index"].search(q_vec.cpu().numpy().astype("float32"), search_k)

    candidates = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        if idx < 0:
            continue
        pid = ctx["id_map"][idx]
        p = ctx["passage_by_id"].get(pid)
        if not p:
            continue
        candidates.append({**p, "dpr_rank": rank, "dpr_score": float(score)})

    scored = []
    with torch.no_grad():
        for i in range(0, len(candidates), 8):
            batch = candidates[i : i + 8]
            qs = [question] * len(batch)
            texts = [p["text"] for p in batch]
            toks = ctx["ce_tok"](
                qs, texts, padding=True, truncation=True, max_length=MAX_CE_LEN, return_tensors="pt"
            )
            toks = {k: v.to(device) for k, v in toks.items()}
            logits = ctx["ce_model"](toks["input_ids"], toks["attention_mask"])
            probs = torch.sigmoid(logits).detach().cpu().tolist()
            raw_logits = logits.detach().cpu().tolist()
            for p, logit, prob in zip(batch, raw_logits, probs):
                scored.append({**p, "logit": float(logit), "prob": float(prob)})

    scored.sort(key=lambda r: r["logit"], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------
CTX = init_all()

# Session state
if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.query = ""
    st.session_state.elapsed = 0.0
    st.session_state.search_triggered = False


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------
def trigger_search():
    st.session_state.search_triggered = True


# ---------------------------------------------------------------------------
# HEADER – Hero Section
# ---------------------------------------------------------------------------
st.markdown(
    """
<div style="text-align:center; padding: 3.5rem 0 2rem;">
    <!-- Logo badge -->
    <div style="
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(79,142,247,0.1);
        border: 1px solid rgba(79,142,247,0.25);
        border-radius: 100px;
        padding: 6px 18px;
        margin-bottom: 24px;
        font-size: 12px;
        font-weight: 600;
        color: #93c5fd;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-family: 'Inter', sans-serif;
    ">
        <span style="width:7px;height:7px;border-radius:50%;background:#4f8ef7;box-shadow:0 0 8px #4f8ef7;display:inline-block;"></span>
        AI · DPR + Cross-Encoder · HUMG
    </div>
    <!-- Main title -->
    <h1 style="
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(28px, 5vw, 46px);
        font-weight: 800;
        line-height: 1.15;
        margin: 0 0 16px;
        background: linear-gradient(135deg, #f1f5f9 0%, #93c5fd 45%, #c4b5fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
    ">🎓 HUMG</h1>
    <!-- Subtitle -->
    <p style="
        font-family: 'Inter', sans-serif;
        color: #64748b;
        font-size: 15px;
        font-weight: 400;
        margin: 0 auto;
        max-width: 500px;
        line-height: 1.65;
    ">Tìm kiếm thông minh trong <strong style='color:#7c9fc9;font-weight:600;'>Quy chế Đào tạo</strong> bằng công nghệ<br>Dense Passage Retrieval &amp; Cross-Encoder Reranking</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# SEARCH BAR
# ---------------------------------------------------------------------------
with st.form("search_form", clear_on_submit=False, border=False):
    col_input, col_btn = st.columns([6, 1])
    with col_input:
        query = st.text_input(
            "search",
            value=st.session_state.query,
            placeholder="Nhập câu hỏi về Quy chế Đào tạo...",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("🔍  Tìm", use_container_width=True)
    if submitted:
        st.session_state.search_triggered = True


# ---------------------------------------------------------------------------
# EXECUTE SEARCH
# ---------------------------------------------------------------------------
if st.session_state.search_triggered and query.strip():
    st.session_state.search_triggered = False
    st.session_state.query = query.strip()

    t0 = time.perf_counter()
    with st.spinner("Đang phân tích câu hỏi và xếp hạng kết quả…"):
        results = do_search(query.strip(), CTX, top_k=DEFAULT_TOP_K)
    elapsed = time.perf_counter() - t0

    st.session_state.results = results
    st.session_state.elapsed = elapsed
    st.rerun()

# ---------------------------------------------------------------------------
# RESULTS
# ---------------------------------------------------------------------------
if st.session_state.results is not None:
    results = st.session_state.results
    elapsed = st.session_state.elapsed

    # ── Divider
    st.markdown(
        """<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(79,142,247,0.2),transparent);margin:1.5rem 0;"></div>""",
        unsafe_allow_html=True,
    )

    # ── Query recap + stats bar
    query_disp = st.session_state.query
    st.markdown(
        f"""
<div style="
    display:flex;
    align-items:center;
    justify-content:space-between;
    flex-wrap:wrap;
    gap:10px;
    margin-bottom:20px;
">
    <div style="
        background: rgba(79,142,247,0.07);
        border: 1px solid rgba(79,142,247,0.18);
        border-radius:10px;
        padding: 8px 16px;
        font-family:'Inter',sans-serif;
        font-size:13px;
        color:#93c5fd;
        max-width:65%;
        white-space:nowrap;
        overflow:hidden;
        text-overflow:ellipsis;
    ">🔍 <strong style="color:#bfdbfe;font-weight:600;">{query_disp}</strong></div>
    <div style="
        font-family:'Inter',sans-serif;
        font-size:12px;
        color:#475569;
        display:flex;
        align-items:center;
        gap:6px;
    ">
        <span style="color:#10b981;font-weight:700;">{len(results)}</span> kết quả &nbsp;·&nbsp;
        <span style="color:#10b981;font-weight:700;">{elapsed:.2f}s</span> ⚡
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if not results:
        st.markdown(
            """
<div style="
    text-align:center;
    padding: 3rem 2rem;
    background: rgba(13,21,38,0.6);
    border: 1px dashed rgba(79,142,247,0.2);
    border-radius: 16px;
    color: #475569;
    font-family:'Inter',sans-serif;
">
    <div style="font-size:40px;margin-bottom:12px;">🔭</div>
    <p style="font-size:15px;color:#64748b;margin:0;">Không tìm thấy kết quả phù hợp.</p>
    <p style="font-size:13px;color:#475569;margin:6px 0 0;">Hãy thử diễn đạt lại câu hỏi theo cách khác nhé!</p>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        # Rank badge mapping
        rank_icons = {1: "🥇", 2: "🥈", 3: "🥉"}
        score_colors = {
            "high":   ("#10b981", "#d1fae5", "rgba(16,185,129,0.12)", "Cao"),
            "medium": ("#f59e0b", "#fef3c7", "rgba(245,158,11,0.12)",  "Trung bình"),
            "low":    ("#ef4444", "#fee2e2", "rgba(239,68,68,0.12)",    "Thấp"),
        }

        for i, item in enumerate(results, start=1):
            prob_pct = item["prob"] * 100

            if prob_pct > 60:
                lvl = "high"
            elif prob_pct > 30:
                lvl = "medium"
            else:
                lvl = "low"

            sc_color, sc_bg_text, sc_bg, sc_label = score_colors[lvl]
            rank_icon = rank_icons.get(i, f"#{i}")

            clause_badge = ""
            if item.get("clause_ref"):
                clause_badge = f"  ·  📌 {item['clause_ref']}"

            title_text = f"{rank_icon} Kết quả #{i}  —  Độ phù hợp: {prob_pct:.1f}%{clause_badge}"

            with st.expander(title_text, expanded=(i <= 2)):
                # Metrics row
                m1, m2, m3 = st.columns(3)
                m1.metric("🎯 Độ phù hợp", f"{prob_pct:.1f}%")
                m2.metric("📊 DPR Rank", f"#{item['dpr_rank']}")
                m3.metric("🔑 Passage ID", item["passage_id"])

                st.markdown(
                    "<div style='height:1px;background:rgba(79,142,247,0.1);margin:12px 0;'></div>",
                    unsafe_allow_html=True,
                )

                # Passage text
                st.markdown(
                    f"""
<div style="
    background: rgba(6,10,24,0.5);
    border-left: 3px solid {sc_color};
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 4px 0 16px;
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    line-height: 1.8;
    color: #cbd5e1;
">{item['text']}</div>
""",
                    unsafe_allow_html=True,
                )

                # Relevance bar
                st.progress(min(prob_pct / 100, 1.0))
                st.caption(
                    f"Relevance: **{prob_pct:.1f}%**  ·  "
                    f"DPR score: {item['dpr_score']:.4f}  ·  "
                    f"Logit: {item['logit']:.3f}"
                )

    # ── New search button
    st.markdown("<div style='margin-top:2rem;'>", unsafe_allow_html=True)
    st.markdown("<div class='new-search-btn'>", unsafe_allow_html=True)
    if st.button("↺  Tìm kiếm mới", use_container_width=True):
        st.session_state.results = None
        st.session_state.query = ""
        st.session_state.elapsed = 0.0
        st.rerun()
    st.markdown("</div></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown(
    """
<div style="
    text-align:center;
    padding: 3rem 0 1rem;
    font-family: 'Inter', sans-serif;
">
    <div style="
        display:inline-flex;
        align-items:center;
        gap:20px;
        font-size:11.5px;
        color:#334155;
    ">
        <span>🎓 HUMG · NLP Project</span>
        <span style="color:#1e293b;">|</span>
        <span>DPR + Cross-Encoder Reranking</span>
        <span style="color:#1e293b;">|</span>
        <span>PhoBERT backbone</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)
