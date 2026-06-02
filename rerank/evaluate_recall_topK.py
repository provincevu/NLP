import json
import argparse
from collections import defaultdict

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", required=True)
    p.add_argument("--score_key", default="ce_logit")
    p.add_argument("--max_k", type=int, default=10)
    args = p.parse_args()
    
    # Đọc dữ liệu với encoding='utf-8'
    rows = []
    with open(args.input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    
    # Nhóm theo qid
    grouped = defaultdict(list)
    for r in rows:
        if "qid" in r and "label" in r:
            grouped[r["qid"]].append(r)
    
    # Tính recall@k và MRR
    ks = list(range(1, args.max_k + 1))
    recall_sum = {k: 0.0 for k in ks}
    mrr_sum = 0.0
    total_queries = 0
    
    for qid, items in grouped.items():
        labels = [int(r.get("label", 0)) for r in items]
        total_relevant = sum(labels)
        
        if total_relevant == 0:
            continue
            
        total_queries += 1
        
        # Sắp xếp theo score giảm dần
        ranked = sorted(items, key=lambda r: float(r.get(args.score_key, 0.0)), reverse=True)
        ranked_labels = [int(r.get("label", 0)) for r in ranked]
        
        # Tính MRR: vị trí đầu tiên có label=1
        rr = 0.0
        for i, label in enumerate(ranked_labels, start=1):
            if label == 1:
                rr = 1.0 / i
                break
        mrr_sum += rr
        
        # Tính recall@k
        for k in ks:
            hits = sum(ranked_labels[:k])
            recall_sum[k] += hits / total_relevant
    
    # MRR
    print(f"Tổng số queries = {total_queries}")
    mrr = mrr_sum / total_queries if total_queries > 0 else 0
    print(f"MRR: {mrr:.4f}")
    
    # Recall@k
    for k in ks:
        recall = recall_sum[k] / total_queries if total_queries > 0 else 0
        print(f"recall@{k} = {recall:.6f}")

if __name__ == "__main__":
    main()