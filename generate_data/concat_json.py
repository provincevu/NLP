"""
Concat two JSON files intelligently.

Usage examples:
python generate_data/concat_json.py --in1 a.json --in2 b.json --out merged.json
python generate_data/concat_json.py --in1 a.json --in2 b.json --out merged.json --dedupe-by qid

Behavior:
- If both inputs are JSON arrays: concatenates them. If `--dedupe-by` supplied, will deduplicate by that key.
- If both inputs are JSON objects: merges keys (second overrides first by default or use --prefer first).
- If types mismatch, script will exit with error unless `--force-array` is used (wraps non-array into array).
"""
import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def dedupe_list(items: list, key: str, prefer: str = "first") -> list:
    seen = {}
    order = []
    for item in items:
        if not isinstance(item, dict) or key not in item:
            # item has no key -> include with unique id
            uid = id(item)
            if uid not in seen:
                seen[uid] = item
                order.append(uid)
        else:
            k = item[key]
            if k in seen:
                if prefer == "second":
                    seen[k] = item
            else:
                seen[k] = item
                order.append(k)
    return [seen[k] for k in order]


def _extract_qid_number(qid: str) -> int:
    if not isinstance(qid, str):
        return -1
    if not qid.startswith("q_"):
        return -1
    tail = qid[2:]
    if not tail.isdigit():
        return -1
    return int(tail)


def ensure_unique_qid(items: list, key: str = "qid", prefix: str = "q_") -> list:
    seen = set()
    max_id = -1
    for item in items:
        if isinstance(item, dict) and key in item:
            qid = item[key]
            seen.add(qid)
            n = _extract_qid_number(qid)
            if n > max_id:
                max_id = n

    next_id = max_id + 1
    for item in items:
        if not isinstance(item, dict) or key not in item:
            continue
        qid = item[key]
        if qid in seen:
            seen.remove(qid)
            continue
        while True:
            new_qid = f"{prefix}{next_id:04d}"
            next_id += 1
            if new_qid not in seen:
                item[key] = new_qid
                seen.add(new_qid)
                break
    return items


def merge_objects(a: dict, b: dict, prefer: str = "second") -> dict:
    result = dict(a)
    if prefer == "second":
        result.update(b)
    else:
        # prefer first: only add keys from b that are not present
        for k, v in b.items():
            if k not in result:
                result[k] = v
    return result


def main():
    p = argparse.ArgumentParser(description="Concatenate/merge two JSON files")
    p.add_argument("--in1", required=True, help="First input JSON file")
    p.add_argument("--in2", required=True, help="Second input JSON file")
    p.add_argument("--out", default="generate_data/concat.json", help="Output JSON file")
    p.add_argument("--dedupe-by", default="", help="If concat arrays, dedupe by this dict key (e.g., qid)")
    p.add_argument("--prefer", choices=["first", "second"], default="second", help="When merging objects or deduping, which side to prefer")
    p.add_argument("--force-array", action="store_true", help="If types mismatch, coerce values into arrays and concat")
    p.add_argument("--unique-qid", action="store_true", help="If concat arrays, rewrite duplicate qid values to unique new ids")

    args = p.parse_args()

    in1 = Path(args.in1)
    in2 = Path(args.in2)
    out = Path(args.out)

    a = load_json(in1)
    b = load_json(in2)

    if isinstance(a, list) and isinstance(b, list):
        merged = a + b
        if args.dedupe_by:
            merged = dedupe_list(merged, args.dedupe_by, prefer=args.prefer)
        if args.unique_qid:
            merged = ensure_unique_qid(merged, key="qid")
    elif isinstance(a, dict) and isinstance(b, dict):
        merged = merge_objects(a, b, prefer=args.prefer)
    else:
        if args.force_array:
            A = a if isinstance(a, list) else [a]
            B = b if isinstance(b, list) else [b]
            merged = A + B
            if args.dedupe_by:
                merged = dedupe_list(merged, args.dedupe_by, prefer=args.prefer)
            if args.unique_qid:
                merged = ensure_unique_qid(merged, key="qid")
        else:
            raise SystemExit("Input JSON types do not match (list vs dict). Use --force-array to coerce.")

    write_json(out, merged)
    print(f"Wrote merged JSON to {out}")


if __name__ == "__main__":
    main()
