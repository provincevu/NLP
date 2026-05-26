# evaluate_baselines.py
# ========================================================
# Đánh giá và so sánh tất cả phương pháp truy hồi:
# 1. BM25 (Okapi BM25)
# 2. TF-IDF + Cosine Similarity
# 3. DPR (Dense Passage Retrieval) — mô hình đã huấn luyện
#
# Metrics: Recall@K (K=1,5,10,20), MRR, Precision@K
# Output: Bảng so sánh + xuất kết quả JSON
# ========================================================

# ⚠️ FIX: Đặt biến môi trường TRƯỚC khi import để tránh segfault
# do xung đột OpenMP giữa FAISS và PyTorch trên macOS
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import time
import sys
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.bm25_retrieval import BM25Retriever
from baselines.tfidf_retrieval import TFIDFRetriever

# ====== CONFIG ======
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASSAGES_PATH = os.path.join(BASE_DIR, "data", "passages.json")
TEST_PATH = os.path.join(BASE_DIR, "data", "splits", "test.json")
VAL_PATH = os.path.join(BASE_DIR, "data", "splits", "val.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "baselines", "results")

# DPR config (optional — chỉ chạy nếu có checkpoint)
DPR_CKPT = os.path.join(BASE_DIR, "inferences", "dpr_ckpt_1")
FAISS_INDEX = os.path.join(BASE_DIR, "data", "faiss_index_last.ivf")
IDMAP_PATH = os.path.join(BASE_DIR, "data", "passage_id_map.json")

TOPK_VALUES = [1, 5, 10, 20]
MAX_K = max(TOPK_VALUES)


# ====== UTILITY FUNCTIONS ======
def read_json_list(path: str) -> list:
    """Đọc JSON file, hỗ trợ cả list và {data: [...]}."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return data


def compute_metrics(
    all_retrieved_pids: List[List[str]],
    all_gold_pids: List[str],
    topk_values: List[int] = None
) -> Dict[str, float]:
    """Tính Recall@K, MRR, Precision@K.
    
    Args:
        all_retrieved_pids: Cho mỗi query, danh sách passage_ids đã retrieve (theo thứ tự rank)
        all_gold_pids: Cho mỗi query, passage_id đúng (gold)
        topk_values: Các giá trị K cần tính
    
    Returns:
        Dict of metric_name → value
    """
    if topk_values is None:
        topk_values = TOPK_VALUES

    n = len(all_gold_pids)
    metrics = {}

    # Recall@K
    for k in topk_values:
        hits = 0
        for retrieved, gold in zip(all_retrieved_pids, all_gold_pids):
            if gold in retrieved[:k]:
                hits += 1
        metrics[f"Recall@{k}"] = hits / max(n, 1)

    # MRR (Mean Reciprocal Rank)
    mrr_sum = 0.0
    for retrieved, gold in zip(all_retrieved_pids, all_gold_pids):
        for rank, pid in enumerate(retrieved, start=1):
            if pid == gold:
                mrr_sum += 1.0 / rank
                break
    metrics["MRR"] = mrr_sum / max(n, 1)

    # Precision@K (= Recall@K / K, nhưng tính riêng cho rõ)
    for k in topk_values:
        hits = 0
        for retrieved, gold in zip(all_retrieved_pids, all_gold_pids):
            if gold in retrieved[:k]:
                hits += 1
        metrics[f"P@{k}"] = hits / max(n, 1)

    return metrics


# ====== EVALUATE BM25 ======
def evaluate_bm25(passages, eval_data, **bm25_kwargs) -> Dict:
    """Đánh giá BM25 retriever."""
    print("\n" + "="*60)
    print("📊 ĐÁNH GIÁ: BM25")
    print("="*60)

    retriever = BM25Retriever(**bm25_kwargs)
    retriever.fit(passages)

    queries = [ex["question"] for ex in eval_data]
    gold_pids = [ex["positive_passage_id"] for ex in eval_data]

    start = time.time()
    batch_results = retriever.batch_retrieve(queries, top_k=MAX_K)
    elapsed = time.time() - start

    all_retrieved = [[pid for pid, _ in results] for results in batch_results]
    metrics = compute_metrics(all_retrieved, gold_pids)
    metrics["latency_total_s"] = elapsed
    metrics["latency_per_query_ms"] = (elapsed / len(queries)) * 1000
    metrics["method"] = "BM25"
    metrics["params"] = {
        "k1": bm25_kwargs.get("k1", 1.5),
        "b": bm25_kwargs.get("b", 0.75),
    }

    return metrics


# ====== EVALUATE TF-IDF ======
def evaluate_tfidf(passages, eval_data, **tfidf_kwargs) -> Dict:
    """Đánh giá TF-IDF retriever."""
    print("\n" + "="*60)
    print("📊 ĐÁNH GIÁ: TF-IDF")
    print("="*60)

    retriever = TFIDFRetriever(**tfidf_kwargs)
    retriever.fit(passages)

    queries = [ex["question"] for ex in eval_data]
    gold_pids = [ex["positive_passage_id"] for ex in eval_data]

    start = time.time()
    batch_results = retriever.batch_retrieve(queries, top_k=MAX_K)
    elapsed = time.time() - start

    all_retrieved = [[pid for pid, _ in results] for results in batch_results]
    metrics = compute_metrics(all_retrieved, gold_pids)
    metrics["latency_total_s"] = elapsed
    metrics["latency_per_query_ms"] = (elapsed / len(queries)) * 1000
    metrics["method"] = "TF-IDF"
    metrics["params"] = {
        "sublinear_tf": tfidf_kwargs.get("sublinear_tf", True),
    }

    return metrics


# ====== EVALUATE DPR ======
def evaluate_dpr(eval_data) -> Optional[Dict]:
    """Đánh giá DPR retriever (nếu có checkpoint + FAISS index)."""
    # Kiểm tra files cần thiết
    if not all(os.path.exists(p) for p in [DPR_CKPT, FAISS_INDEX, IDMAP_PATH]):
        print("\n⚠️  DPR: Không tìm thấy checkpoint/FAISS index. Bỏ qua.")
        print(f"   Cần: {DPR_CKPT}, {FAISS_INDEX}, {IDMAP_PATH}")
        return None

    print("\n" + "="*60)
    print("📊 ĐÁNH GIÁ: DPR (Dense Passage Retrieval)")
    print("="*60)

    try:
        import faiss  # ⚠️ Import faiss TRƯỚC torch để tránh OpenMP conflict
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer, AutoModel

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        MAX_Q_LEN = 64

        # Load model
        print("   Loading DPR model...")
        tokenizer = AutoTokenizer.from_pretrained(DPR_CKPT, use_fast=False)
        encoder = AutoModel.from_pretrained(DPR_CKPT).to(DEVICE)
        encoder.eval()

        # Load FAISS index + id_map
        index = faiss.read_index(FAISS_INDEX)
        with open(IDMAP_PATH, "r", encoding="utf-8") as f:
            id_map = json.load(f)

        queries = [ex["question"] for ex in eval_data]
        gold_pids = [ex["positive_passage_id"] for ex in eval_data]

        # Encode queries in batches
        BATCH_Q = 32
        all_retrieved = []

        start = time.time()
        for i in range(0, len(queries), BATCH_Q):
            batch_queries = queries[i:i + BATCH_Q]
            toks = tokenizer(
                batch_queries, padding=True, truncation=True,
                max_length=MAX_Q_LEN, return_tensors="pt"
            )
            toks = {k: v.to(DEVICE) for k, v in toks.items()}

            with torch.no_grad():
                out = encoder(**toks)
                mask = toks["attention_mask"].unsqueeze(-1).float()
                vec = (out.last_hidden_state * mask).sum(dim=1) / \
                      mask.sum(dim=1).clamp(min=1e-9)
                vec = F.normalize(vec, p=2, dim=1)
                q_embs = vec.cpu().numpy()

            D, I = index.search(q_embs.astype("float32"), MAX_K)
            for qi in range(len(batch_queries)):
                retrieved_pids = [id_map[idx] for idx in I[qi]]
                all_retrieved.append(retrieved_pids)

        elapsed = time.time() - start

        metrics = compute_metrics(all_retrieved, gold_pids)
        metrics["latency_total_s"] = elapsed
        metrics["latency_per_query_ms"] = (elapsed / len(queries)) * 1000
        metrics["method"] = "DPR"
        metrics["params"] = {"model": DPR_CKPT, "device": DEVICE}

        return metrics

    except Exception as e:
        print(f"❌ DPR evaluation failed: {e}")
        return None


# ====== PRINT COMPARISON TABLE ======
def print_comparison_table(all_metrics: List[Dict]):
    """In bảng so sánh đẹp."""
    print("\n")
    print("╔" + "═"*72 + "╗")
    print("║" + " 📊 BẢNG SO SÁNH CÁC PHƯƠNG PHÁP TRUY HỒI".center(72) + "║")
    print("╠" + "═"*72 + "╣")

    # Header
    methods = [m["method"] for m in all_metrics]
    header = f"{'Metric':<20}"
    for method in methods:
        header += f"{'│ ' + method:>18}"
    print("║ " + header.ljust(71) + "║")
    print("╠" + "═"*72 + "╣")

    # Metrics rows
    metric_keys = [f"Recall@{k}" for k in TOPK_VALUES] + ["MRR"]
    for key in metric_keys:
        row = f"{key:<20}"
        values = [m.get(key, 0) for m in all_metrics]
        best_val = max(values)
        for val in values:
            marker = " 🏆" if val == best_val and len(values) > 1 else "   "
            row += f"│ {val:.4f}{marker}  "
        print("║ " + row.ljust(71) + "║")

    print("╠" + "═"*72 + "╣")

    # Latency
    row = f"{'Latency (ms/query)':<20}"
    for m in all_metrics:
        lat = m.get("latency_per_query_ms", 0)
        row += f"│ {lat:.1f} ms      "
    print("║ " + row.ljust(71) + "║")

    print("╚" + "═"*72 + "╝")


def print_detailed_analysis(all_metrics: List[Dict], eval_data: list):
    """Phân tích chi tiết sự khác biệt giữa các phương pháp."""
    print("\n📋 PHÂN TÍCH CHI TIẾT")
    print("-"*60)

    n_queries = len(eval_data)
    print(f"   Số câu hỏi đánh giá: {n_queries}")
    print(f"   Các K đánh giá: {TOPK_VALUES}")

    for m in all_metrics:
        method = m["method"]
        print(f"\n   🔹 {method}:")
        print(f"      Recall@1:  {m.get('Recall@1', 0):.4f} "
              f"({int(m.get('Recall@1', 0) * n_queries)}/{n_queries} câu)")
        print(f"      Recall@5:  {m.get('Recall@5', 0):.4f} "
              f"({int(m.get('Recall@5', 0) * n_queries)}/{n_queries} câu)")
        print(f"      Recall@10: {m.get('Recall@10', 0):.4f}")
        print(f"      Recall@20: {m.get('Recall@20', 0):.4f}")
        print(f"      MRR:       {m.get('MRR', 0):.4f}")
        print(f"      Speed:     {m.get('latency_per_query_ms', 0):.1f} ms/query")

    # So sánh DPR vs Baselines
    dpr = next((m for m in all_metrics if m["method"] == "DPR"), None)
    if dpr:
        print("\n   📈 IMPROVEMENT CỦA DPR SO VỚI BASELINES:")
        for m in all_metrics:
            if m["method"] == "DPR":
                continue
            name = m["method"]
            for k in [1, 5, 10]:
                dpr_val = dpr.get(f"Recall@{k}", 0)
                base_val = m.get(f"Recall@{k}", 0)
                if base_val > 0:
                    improvement = ((dpr_val - base_val) / base_val) * 100
                    arrow = "↑" if improvement > 0 else "↓"
                    print(f"      DPR vs {name} Recall@{k}: "
                          f"{arrow} {abs(improvement):.1f}%")


# ====== MAIN ======
def main():
    """Chạy đánh giá toàn bộ và so sánh."""
    print("🚀 BẮT ĐẦU ĐÁNH GIÁ BASELINES")
    print("="*60)

    # Load data
    passages = read_json_list(PASSAGES_PATH)
    print(f"📂 Loaded {len(passages)} passages")

    # Load evaluation data (dùng val set vì test set quá ít)
    eval_data = read_json_list(VAL_PATH)
    print(f"📂 Loaded {len(eval_data)} evaluation queries (val set)")

    # Nếu muốn dùng cả test set, uncomment dòng dưới:
    # test_data = read_json_list(TEST_PATH)
    # eval_data = eval_data + test_data
    # print(f"📂 Total: {len(eval_data)} queries (val + test)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_metrics = []

    # ====== 1. BM25 ======
    bm25_metrics = evaluate_bm25(
        passages, eval_data,
        k1=1.5, b=0.75,
        use_underthesea=False,
        remove_stopwords=True
    )
    all_metrics.append(bm25_metrics)

    # ====== 2. TF-IDF ======
    tfidf_metrics = evaluate_tfidf(
        passages, eval_data,
        use_underthesea=False,
        remove_stopwords=True,
        sublinear_tf=True
    )
    all_metrics.append(tfidf_metrics)

    # ====== 3. DPR (optional) ======
    dpr_metrics = evaluate_dpr(eval_data)
    if dpr_metrics:
        all_metrics.append(dpr_metrics)

    # ====== IN KẾT QUẢ ======
    print_comparison_table(all_metrics)
    print_detailed_analysis(all_metrics, eval_data)

    # ====== LƯU KẾT QUẢ ======
    output_path = os.path.join(OUTPUT_DIR, "comparison_results.json")
    # Chuyển metrics thành serializable
    serializable = []
    for m in all_metrics:
        sm = {}
        for k, v in m.items():
            if isinstance(v, (np.float32, np.float64)):
                sm[k] = float(v)
            else:
                sm[k] = v
        serializable.append(sm)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Kết quả đã lưu tại: {output_path}")

    return all_metrics


if __name__ == "__main__":
    main()
