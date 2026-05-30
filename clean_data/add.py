import json
from pathlib import Path


INPUT_FILE = "abc.json"
OUTPUT_FILE = "bcd.json"


def main():
    positive_passage_id = input(
        "Nhập positive_passage_id: "
    ).strip()

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON phải là một list.")

    for item in data:
        item["positive_passage_id"] = positive_passage_id

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Đã xử lý {len(data)} records")
    print(f"Đã lưu vào: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()