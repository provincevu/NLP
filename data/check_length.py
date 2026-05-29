import argparse
import json
from pathlib import Path
from typing import List, Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_long_entries(file_path: str, mode: str = "passages", max_words: int = 100, text_field: str = None, id_field: str = None) -> List[str]:
    """Tìm các entry (passage hoặc question) vượt quá số từ cho phép.

    mode: 'passages' or 'questions'
    text_field: tên trường chứa văn bản (mặc định 'text' cho passages, 'question' cho questions)
    id_field: tên trường id (mặc định 'passage_id' cho passages, 'qid' cho questions)
    """
    path = Path(file_path)
    data = load_json(path)

    if isinstance(data, dict) and "data" in data:
        items = data["data"]
    else:
        items = data

    if text_field is None:
        text_field = "text" if mode == "passages" else "question"
    if id_field is None:
        id_field = "passage_id" if mode == "passages" else "qid"

    print(f"Kiểm tra file: {file_path}")
    print(f"Chế độ: {mode} — trường văn bản: {text_field} — id field: {id_field}")
    print(f"Giới hạn: {max_words} từ")
    print("=" * 50)

    long_ids = []

    if not isinstance(items, list):
        raise ValueError("Input JSON phải là mảng (list) hoặc object chứa key 'data' là mảng")

    for entry in items:
        if not isinstance(entry, dict):
            continue
        eid = entry.get(id_field, "unknown")
        text = entry.get(text_field, "") or ""
        word_count = len(str(text).split())

        if word_count > max_words:
            long_ids.append({"id": eid, "words": word_count, "over": word_count - max_words, text_field: text})
            print(f"⚠️ {eid}: {word_count} từ (vượt {word_count - max_words} từ)")

    if not long_ids:
        print("✅ Không có entry nào vượt quá giới hạn")
    else:
        print(f"\n📋 Tổng {len(long_ids)} entry cần xử lý")

    return long_ids


def main():
    parser = argparse.ArgumentParser(description="Check max word length for passages or questions JSON")
    parser.add_argument("--file", required=True, help="Path to JSON file (array of objects or object{'data': [...]})")
    parser.add_argument("--mode", choices=["passages", "questions"], default="passages", help="Type of entries to check")
    parser.add_argument("--max-words", type=int, default=100, help="Maximum allowed words")
    parser.add_argument("--text-field", default=None, help="Field name that contains the text (overrides default)")
    parser.add_argument("--id-field", default=None, help="Field name for id (overrides default)")
    parser.add_argument("--output", default="", help="Optional output JSON file with entries exceeding limit")

    args = parser.parse_args()

    results = find_long_entries(args.file, mode=args.mode, max_words=args.max_words, text_field=args.text_field, id_field=args.id_field)
    if args.output and results:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(results)} entries to {args.output}")


if __name__ == "__main__":
    main()