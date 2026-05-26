# tune_bm25.py
# ========================================================
# Grid Search tối ưu siêu tham số BM25 (k1, b)
# trên tập validation.
#
# k1: điều chỉnh mức bão hòa term frequency
#   - k1 nhỏ → ít phân biệt giữa tf=1 và tf=10
#   - k1 lớn → phân biệt rõ hơn
#
# b: điều chỉnh ảnh hưởng của document length
#   - b=0 → không normalize theo độ dài
#   - b=1 → normalize hoàn toàn theo độ dài
# ========================================================

import json
import os
import sys
import itertools
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.bm25_retrieval import BM25Retriever
from baselines.evaluate_baselines import read_json_list, compute_metrics

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASSAGES_PATH = os.path.join(BASE_DIR, "data", "passages.json")
VAL_PATH = os.path.join(BASE_DIR, "data", "splits", "val.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "baselines", "results")

# Lưới siêu tham số
K1_VALUES = [0.5, 1.0, 1.2, 1.5, 2.0, 2.5]
B_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
MAX_K = 20


def main():
    print("🔍 GRID SEARCH: Tối ưu siêu tham số BM25")
    print("="*60)

    passages = read_json_list(PASSAGES_PATH)
    val_data = read_json_list(VAL_PATH)
    queries = [ex["question"] for ex in val_data]
    gold_pids = [ex["positive_passage_id"] for ex in val_data]

    print(f"   Passages: {len(passages)}")
    print(f"   Queries:  {len(queries)}")
    print(f"   k1 values: {K1_VALUES}")
    print(f"   b values:  {B_VALUES}")
    print(f"   Total combinations: {len(K1_VALUES) * len(B_VALUES)}")

    best_mrr = -1
    best_params = {}
    all_results = []

    for k1, b in itertools.product(K1_VALUES, B_VALUES):
        retriever = BM25Retriever(k1=k1, b=b, use_underthesea=False)
        retriever.fit(passages)

        batch_results = retriever.batch_retrieve(queries, top_k=MAX_K)
        all_retrieved = [[pid for pid, _ in res] for res in batch_results]
        metrics = compute_metrics(all_retrieved, gold_pids)

        mrr = metrics["MRR"]
        r1 = metrics["Recall@1"]
        r5 = metrics["Recall@5"]

        result = {"k1": k1, "b": b, **metrics}
        all_results.append(result)

        marker = ""
        if mrr > best_mrr:
            best_mrr = mrr
            best_params = {"k1": k1, "b": b}
            marker = " ⭐ BEST"

        print(f"   k1={k1:.1f}, b={b:.2f} → "
              f"R@1={r1:.4f} R@5={r5:.4f} MRR={mrr:.4f}{marker}")

    print("\n" + "="*60)
    print(f"🏆 BEST PARAMS: k1={best_params['k1']}, b={best_params['b']}")
    print(f"   MRR = {best_mrr:.4f}")
    print("="*60)

    # Lưu kết quả
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "bm25_tuning_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_params": best_params,
            "best_mrr": best_mrr,
            "all_results": all_results
        }, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Kết quả đã lưu: {output_path}")


if __name__ == "__main__":
    main()
