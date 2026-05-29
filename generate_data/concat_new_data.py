import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_start(value: Optional[str]) -> int:
    if value is None:
        raw = input("Enter start qid (number or q_<number>): ").strip()
    else:
        raw = value.strip()

    if raw.startswith("q_"):
        raw = raw[2:]
    if not raw.isdigit():
        raise ValueError("start must be a number or q_<number>")
    return int(raw)


def normalize_items(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = item.get("question") or item.get("q")
        passage_id = item.get("positive_passage_id")
        if not question or not passage_id:
            continue
        normalized.append(
            {
                "question": str(question).strip(),
                "positive_passage_id": str(passage_id).strip(),
            }
        )
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concat two JSON files and reassign qid sequentially"
    )
    parser.add_argument(
        "--in1",
        required=True,
        help="First input JSON file",
    )
    parser.add_argument(
        "--in2",
        required=True,
        help="Second input JSON file",
    )
    parser.add_argument(
        "--out",
        default="generate_data/NEW_DATA/concat.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Starting qid number (e.g., 2555 or q_2555)",
    )

    args = parser.parse_args()

    in1 = Path(args.in1)
    in2 = Path(args.in2)
    out = Path(args.out)

    data1 = load_json(in1)
    data2 = load_json(in2)

    if not isinstance(data1, list) or not isinstance(data2, list):
        raise SystemExit("Both inputs must be JSON arrays.")

    items = normalize_items(data1) + normalize_items(data2)

    start_id = parse_start(args.start)
    output: List[Dict[str, str]] = []
    current = start_id
    for item in items:
        output.append(
            {
                "qid": f"q_{current}",
                "question": item["question"],
                "positive_passage_id": item["positive_passage_id"],
            }
        )
        current += 1

    write_json(out, output)
    print(f"Wrote {len(output)} items to {out}")


if __name__ == "__main__":
    main()
