import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

BASE_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

DEFAULT_INPUT_DIR = os.path.join(PROJECT_DIR, "rerank", "data")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_DIR, "rerank", "cross_encoder")


def read_json_list(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return data


def write_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sort_negatives(rows: List[dict]) -> List[dict]:
    if not rows:
        return rows
    if "rank" in rows[0] and rows[0]["rank"] is not None:
        return sorted(rows, key=lambda r: r.get("rank", 10**9))
    return rows


def build_split(
    in_path: str,
    out_path: str,
    max_negatives: int,
    output_format: str,
) -> Dict[str, int]:
    rows = read_json_list(in_path)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("qid")].append(row)

    out_rows = []
    total_questions = len(grouped)
    kept_questions = 0
    skipped_questions = 0
    total_pos = 0
    total_neg = 0

    for _, items in grouped.items():
        positives = [r for r in items if r.get("label") == 1]
        negatives = [r for r in items if r.get("label") == 0]
        if not positives:
            skipped_questions += 1
            continue

        kept_questions += 1
        total_pos += len(positives)

        negatives = sort_negatives(negatives)
        if max_negatives >= 0:
            negatives = negatives[:max_negatives]
        total_neg += len(negatives)

        for r in positives + negatives:
            out_rows.append(
                {
                    "qid": r.get("qid"),
                    "question": r.get("question"),
                    "passage_id": r.get("passage_id"),
                    "passage_text": r.get("passage_text"),
                    "label": int(r.get("label", 0)),
                    "rank": r.get("rank"),
                    "score": r.get("score"),
                }
            )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if output_format == "jsonl":
        write_jsonl(out_path, out_rows)
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_rows, f, ensure_ascii=False, indent=2)

    return {
        "rows": len(out_rows),
        "total_questions": total_questions,
        "kept_questions": kept_questions,
        "skipped_questions": skipped_questions,
        "total_pos": total_pos,
        "total_neg": total_neg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cross-encoder rerank dataset")
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_negatives", type=int, default=-1)
    parser.add_argument("--output_format", choices=["jsonl", "json"], default="jsonl")
    args = parser.parse_args()

    splits = ["train", "val", "test"]
    for split in splits:
        in_path = os.path.join(args.input_dir, f"{split}.json")
        if not os.path.exists(in_path):
            continue
        out_ext = "jsonl" if args.output_format == "jsonl" else "json"
        out_path = os.path.join(args.output_dir, f"{split}.{out_ext}")
        stats = build_split(in_path, out_path, args.max_negatives, args.output_format)

        total_questions = stats["total_questions"]
        kept_questions = stats["kept_questions"]
        skipped_questions = stats["skipped_questions"]
        total_pos = stats["total_pos"]
        total_neg = stats["total_neg"]
        rows = stats["rows"]
        keep_ratio = (kept_questions / total_questions) if total_questions else 0.0

        print(f"Da luu {rows} dong vao: {out_path}")
        print(
            "Thong ke split {split}: Tong cau hoi={total}, Giu lai={kept}, Bi loai={skipped}, "
            "Ti le giu={keep:.2%}, Pos={pos}, Neg={neg}".format(
                split=split,
                total=total_questions,
                kept=kept_questions,
                skipped=skipped_questions,
                keep=keep_ratio,
                pos=total_pos,
                neg=total_neg,
            )
        )


if __name__ == "__main__":
    main()
