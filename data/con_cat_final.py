import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Concat nhiều file JSON và đánh lại qid"
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Thư mục chứa các file json"
    )

    parser.add_argument(
        "--out",
        default="merged.json",
        help="File output"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    json_files = sorted(input_dir.glob("*.json"))

    print(f"Found {len(json_files)} json files")

    merged = []

    for file_path in json_files:
        print(f"Reading: {file_path.name}")

        data = load_json(file_path)

        if not isinstance(data, list):
            print(f"Skip {file_path.name}: JSON root is not a list")
            continue

        merged.extend(data)

    print(f"Total records before renumber: {len(merged)}")

    for idx, item in enumerate(merged, start=1):
        if isinstance(item, dict):
            item["qid"] = f"q_{idx:04d}"

    save_json(Path(args.out), merged)

    print(f"Saved {len(merged)} records to {args.out}")


if __name__ == "__main__":
    main()
