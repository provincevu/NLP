import argparse
import json
import os
from typing import Dict, List


def read_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_rows(path: str) -> List[dict]:
    if path.endswith(".jsonl"):
        return read_jsonl(path)
    return read_json(path)


def summarize(rows: List[dict]) -> Dict[str, int]:
    pos = sum(1 for r in rows if r.get("label") == 1)
    neg = sum(1 for r in rows if r.get("label") == 0)
    return {"rows": len(rows), "pos": pos, "neg": neg}


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview cross-encoder dataset")
    parser.add_argument("path")
    parser.add_argument("--sample", type=int, default=3)
    args = parser.parse_args()

    rows = load_rows(args.path)
    stats = summarize(rows)
    print(f"Tong dong: {stats['rows']}, Pos={stats['pos']}, Neg={stats['neg']}")
    for row in rows[: args.sample]:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
