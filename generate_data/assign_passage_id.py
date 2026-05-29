"""
Assign positive_passage_id for LLM output JSON by matching evidence strings to passages.

Usage:
python generate_data/assign_passage_id.py --input generate_data/output_llm.json --passages generate_data/process_pass.json --output generate_data/output_llm_with_ids.json

Notes:
- Skips entries that already have positive_passage_id.
- If evidence matches multiple passages, it is left unchanged and reported.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_passage_index(passages: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    index = []
    for item in passages:
        passage_id = item.get("passage_id")
        text = item.get("text")
        if isinstance(passage_id, str) and isinstance(text, str):
            index.append((passage_id, text))
    return index


def find_passage_id_by_evidence(evidence: str, index: List[Tuple[str, str]]) -> List[str]:
    matches = []
    for passage_id, text in index:
        if evidence and evidence in text:
            matches.append(passage_id)
    return matches


def main() -> None:
    p = argparse.ArgumentParser(description="Assign positive_passage_id from evidence substrings")
    p.add_argument("--input", required=True, help="LLM output JSON file")
    p.add_argument("--passages", required=True, help="Passages JSON file")
    p.add_argument("--output", required=True, help="Output JSON file")
    args = p.parse_args()

    data = load_json(Path(args.input))
    passages = load_json(Path(args.passages))

    if isinstance(data, dict) and "questions" in data:
        questions = data["questions"]
        is_wrapped = True
    elif isinstance(data, list):
        questions = data
        is_wrapped = False
    else:
        raise SystemExit("Input JSON must be a list or an object with 'questions'.")

    if not isinstance(questions, list):
        raise SystemExit("'questions' must be a list.")

    index = build_passage_index(passages)

    assigned = 0
    skipped = 0
    ambiguous = 0
    missing = 0

    for obj in questions:
        if not isinstance(obj, dict):
            continue
        if obj.get("positive_passage_id"):
            skipped += 1
            continue

        evidence = obj.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            missing += 1
            continue

        matches = find_passage_id_by_evidence(evidence.strip(), index)
        if len(matches) == 1:
            obj["positive_passage_id"] = matches[0]
            assigned += 1
        elif len(matches) == 0:
            missing += 1
        else:
            ambiguous += 1

    output_data = data if is_wrapped else questions
    write_json(Path(args.output), output_data)

    print(f"Assigned: {assigned}")
    print(f"Skipped (already set): {skipped}")
    print(f"Missing evidence/no match: {missing}")
    print(f"Ambiguous matches: {ambiguous}")


if __name__ == "__main__":
    main()
