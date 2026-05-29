import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple

from openai import OpenAI

SECTION_RE = re.compile(r"^\s*\d+(?:\.\d+)*[\.)]\s+\S")

SYSTEM_PROMPT = """
Bạn là AI chuyên tóm tắt văn bản.

Yêu cầu:
- Tóm tắt ngắn gọn, tự nhiên, không cộc lốc.
- Chỉ trả về 1 đoạn tóm tắt, không được giải thích hay thêm thông tin gì khác.
- Trả lời bằng tiếng Việt.
""".strip()

SUMMARIZE_PROMPT = """
Hãy tóm tắt đoạn sau. sao cho đoạn tóm tắt ngắn gọn, tự nhiên, không cộc lốc nhưng vẫn giữ được ý chính của đoạn văn bản gốc.

Đoạn cần tóm tắt:
"{text}"
""".strip()


def read_text(path: Path, encoding: str) -> str:
    return path.read_text(encoding=encoding)


def normalize_lines(text: str) -> List[str]:
    return [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]


def split_by_numbered_sections(lines: List[str]) -> List[str]:
    sections: List[List[str]] = []
    current: List[str] = []

    def flush() -> None:
        nonlocal current
        if any(line.strip() for line in current):
            sections.append(current)
        current = []

    for line in lines:
        if SECTION_RE.match(line) and current:
            flush()
        current.append(line)

    flush()
    return ["\n".join(block).strip() for block in sections if "\n".join(block).strip()]


def split_by_paragraphs(lines: List[str]) -> List[str]:
    blocks: List[List[str]] = []
    current: List[str] = []

    def flush() -> None:
        nonlocal current
        if any(line.strip() for line in current):
            blocks.append(current)
        current = []

    for line in lines:
        if not line.strip():
            flush()
        else:
            current.append(line)

    flush()
    return ["\n".join(block).strip() for block in blocks if "\n".join(block).strip()]


def build_passages_for_file(path: Path, mode: str, encoding: str) -> List[Tuple[str, str]]:
    text = read_text(path, encoding=encoding)
    lines = normalize_lines(text)

    if mode == "whole":
        chunks = ["\n".join(lines).strip()] if any(line.strip() for line in lines) else []
    elif mode == "paragraphs":
        chunks = split_by_paragraphs(lines)
    elif mode == "numbered":
        chunks = split_by_numbered_sections(lines)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    results: List[Tuple[str, str]] = []
    stem = path.stem

    if len(chunks) == 1:
        results.append((stem, chunks[0]))
    else:
        for idx, chunk in enumerate(chunks, start=1):
            results.append((f"{stem}__{idx:02d}", chunk))

    return results


def iter_text_files(input_dir: Path, glob_pattern: str) -> Iterable[Path]:
    for path in sorted(input_dir.glob(glob_pattern)):
        if path.is_file():
            yield path


def write_json(path: Path, data: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def call_model(client, model: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def parse_passages(text: str) -> List[str]:
    results: List[str] = []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    blocks: List[List[str]] = []
    current: List[str] = []

    def flush() -> None:
        nonlocal current
        if any(line.strip() for line in current):
            blocks.append(current)
        current = []

    for line in lines:
        stripped = line.strip()
        if stripped == "---":
            flush()
            continue
        if not stripped:
            flush()
            continue
        if stripped.startswith("- "):
            stripped = stripped[2:].strip()
        current.append(stripped)

    flush()

    for block in blocks:
        text_block = " ".join(block).strip()
        if text_block:
            results.append(text_block)

    return results


def word_count(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


def build_llm_passages(text: str, client, model: str, temperature: float, max_words: int) -> List[str]:
    system_prompt = SYSTEM_PROMPT.format(max_words=max_words)
    user_prompt = f"Văn bản nguồn:\n\"{text}\""
    raw = call_model(client, model, system_prompt, user_prompt, temperature)
    passages = parse_passages(raw)

    trimmed: List[str] = []
    for passage in passages:
        if word_count(passage) <= max_words:
            trimmed.append(passage)
            continue

        summarize_prompt = SUMMARIZE_PROMPT.format(max_words=max_words, text=passage)
        summarized = call_model(
            client,
            model,
            "Bạn là AI chuyên tóm tắt ngắn gọn.",
            summarize_prompt,
            temperature,
        ).strip()
        if summarized:
            trimmed.append(summarized)
        else:
            trimmed.append(passage)

    return trimmed


def summarize_text(text: str, client, model: str, temperature: float, max_words: int) -> str:
    system_prompt = SYSTEM_PROMPT.format(max_words=max_words)
    user_prompt = f"Văn bản nguồn:\n\"{text}\""
    raw = call_model(client, model, system_prompt, user_prompt, temperature).strip()
    if not raw:
        return ""

    if word_count(raw) <= max_words:
        return raw

    summarize_prompt = SUMMARIZE_PROMPT.format(max_words=max_words, text=raw)
    summarized = call_model(
        client,
        model,
        "Bạn là AI chuyên tóm tắt ngắn gọn.",
        summarize_prompt,
        temperature,
    ).strip()
    return summarized or raw


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build passages from text files for retrieval pipelines."
    )
    ap.add_argument(
        "--input-dir",
        default=str(Path(__file__).resolve().parent / "raw_data"),
        help="Directory containing input .txt files",
    )
    ap.add_argument(
        "--glob",
        default="*.txt",
        help="Glob pattern to select files",
    )
    ap.add_argument(
        "--mode",
        choices=["whole", "paragraphs", "numbered"],
        default="numbered",
        help="How to split passages: whole | paragraphs | numbered",
    )
    ap.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding",
    )
    ap.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "passages.json"),
        help="Output JSON path",
    )
    ap.add_argument(
        "--use-llm",
        action="store_true",
        help="Use local LLM to summarize and overwrite raw_data files",
    )
    ap.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:5001/v1"),
        help="OpenAI-compatible server base URL",
    )
    ap.add_argument(
        "--api-key",
        default="sk-6af49da8d0b54065a2f7929ac7a43512",
        help="API key for the server",
    )
    ap.add_argument(
        "--model",
        default="deepseek-v4-flash",
        help="Model name",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature",
    )
    ap.add_argument(
        "--max-words",
        type=int,
        default=120,
        help="Maximum words per passage",
    )

    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    passages = []

    client = None
    if args.use_llm:
        if not args.api_key:
            raise SystemExit("Missing API key. Set --api-key or OPENAI_API_KEY.")
        client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    for path in iter_text_files(input_dir, args.glob):
        text = read_text(path, encoding=args.encoding)

        if args.use_llm:
            summary = summarize_text(
                text,
                client=client,
                model=args.model,
                temperature=args.temperature,
                max_words=args.max_words,
            ).strip()
            if summary:
                path.write_text(summary, encoding=args.encoding)
        else:
            for passage_id, passage in build_passages_for_file(path, args.mode, args.encoding):
                passages.append(
                    {
                        "passage_id": passage_id,
                        "text": passage,
                    }
                )

    if not args.use_llm:
        write_json(output_path, passages)
        print(f"đã ghi {len(passages)} passages to {output_path}")
    else:
        print(f"Overwrote summaries for .txt files in {input_dir}")


if __name__ == "__main__":
    main()
