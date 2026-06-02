import argparse
import json
import math
from collections import defaultdict
from typing import Dict, List, Tuple


def read_json_list(path: str) -> List[dict]:
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return data


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def rankdata(scores: List[float]) -> List[float]:
    # Average rank for ties, ranks start at 1
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def roc_auc(scores: List[float], labels: List[int]) -> float:
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")

    ranks = rankdata(scores)
    n_pos = len(pos)
    n_neg = len(neg)
    sum_ranks_pos = sum(r for r, l in zip(ranks, labels) if l == 1)
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc


def classification_metrics(scores: List[float], labels: List[int], threshold: float) -> Dict[str, float]:
    preds = [1 if s >= threshold else 0 for s in scores]
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(1, len(labels))
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def dcg(labels: List[int]) -> float:
    score = 0.0
    for i, rel in enumerate(labels, start=1):
        score += rel / math.log2(i + 1)
    return score


def ranking_metrics(grouped: Dict[str, List[dict]], score_key: str, ks: List[int]) -> Dict[str, float]:
    mrr_sum = 0.0
    map_sum = {k: 0.0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}
    recall_sum = {k: 0.0 for k in ks}
    precision_sum = {k: 0.0 for k in ks}
    total = 0

    for _, items in grouped.items():
        labels = [int(r.get("label", 0)) for r in items]
        if sum(labels) == 0:
            continue

        total += 1
        ranked = sorted(items, key=lambda r: r.get(score_key, 0.0), reverse=True)
        ranked_labels = [int(r.get("label", 0)) for r in ranked]

        # MRR
        rr = 0.0
        for i, rel in enumerate(ranked_labels, start=1):
            if rel > 0:
                rr = 1.0 / i
                break
        mrr_sum += rr

        total_relevant = sum(ranked_labels)
        for k in ks:
            topk = ranked_labels[:k]
            hits = sum(topk)
            recall_sum[k] += hits / total_relevant
            precision_sum[k] += hits / k

            # AP@k
            ap = 0.0
            hit_count = 0
            for i, rel in enumerate(topk, start=1):
                if rel > 0:
                    hit_count += 1
                    ap += hit_count / i
            ap = ap / min(total_relevant, k)
            map_sum[k] += ap

            # NDCG@k
            dcg_k = dcg(topk)
            ideal = sorted(ranked_labels, reverse=True)[:k]
            idcg_k = dcg(ideal)
            ndcg_sum[k] += (dcg_k / idcg_k) if idcg_k > 0 else 0.0

    if total == 0:
        return {}

    out = {"mrr": mrr_sum / total}
    for k in ks:
        out[f"map@{k}"] = map_sum[k] / total
        out[f"ndcg@{k}"] = ndcg_sum[k] / total
        out[f"recall@{k}"] = recall_sum[k] / total
        out[f"precision@{k}"] = precision_sum[k] / total
    return out


def main():
    p = argparse.ArgumentParser(description="Evaluate cross-encoder reranker")
    p.add_argument("--input_path", required=True)
    p.add_argument("--score_key", default="ce_logit")
    p.add_argument("--score_type", choices=["logit", "prob"], default="logit")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--ks", default="1,3,5,10")
    args = p.parse_args()

    rows = read_json_list(args.input_path)
    ks = [int(k.strip()) for k in args.ks.split(",") if k.strip()]

    # classification metrics
    has_label = all("label" in r for r in rows)
    if has_label:
        raw_scores = [float(r.get(args.score_key, 0.0)) for r in rows]
        if args.score_type == "logit":
            scores = [sigmoid(s) for s in raw_scores]
        else:
            scores = raw_scores
        labels = [int(r.get("label", 0)) for r in rows]
        cls = classification_metrics(scores, labels, args.threshold)
        auc = roc_auc(scores, labels)
        print("Classification metrics:")
        print(f"  accuracy={cls['accuracy']:.4f}")
        print(f"  precision={cls['precision']:.4f}")
        print(f"  recall={cls['recall']:.4f}")
        print(f"  f1={cls['f1']:.4f}")
        if not math.isnan(auc):
            print(f"  roc_auc={auc:.4f}")

    # ranking metrics
    if any("qid" in r for r in rows) and has_label:
        grouped = defaultdict(list)
        for r in rows:
            grouped[r.get("qid")].append(r)
        rank_scores = ranking_metrics(grouped, args.score_key, ks)
        if rank_scores:
            print("Ranking metrics:")
            print(f"  mrr={rank_scores['mrr']:.4f}")
            for k in ks:
                print(f"  map@{k}={rank_scores[f'map@{k}']:.4f}")
                print(f"  ndcg@{k}={rank_scores[f'ndcg@{k}']:.4f}")
                print(f"  recall@{k}={rank_scores[f'recall@{k}']:.4f}")
                print(f"  precision@{k}={rank_scores[f'precision@{k}']:.4f}")

if __name__ == "__main__":
    main()