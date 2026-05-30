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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder chứa các file json"
    )

    parser.add_argument(
        "--out",
        default="merged_questions.json",
        help="File output"
    )

    parser.add_argument(
        "--start-qid",
        type=int,
        default=4000,
        help="Số bắt đầu để đánh lại qid"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    merged = []
    

    json_files = sorted(input_dir.glob("*.json"))

    print(f"Found {len(json_files)} json files")

    next_qid = args.start_qid
    for file_path in json_files:
        data = load_json(file_path)

        if not isinstance(data, list):
            print(f"Skip {file_path.name}: not a JSON array")
            continue

        for item in data:
            merged.append(
                {
                    "qid": f"q_{next_qid}",
                    "question": item["question"],
                    "positive_passage_id": item["positive_passage_id"]
                }
            )
            next_qid += 1

    save_json(Path(args.out), merged)

    print(f"Total questions: {len(merged)}")
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()