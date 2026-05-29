import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any, Tuple

from openai import OpenAI


# ----------------------------
# Heuristics (VN legal/guideline)
# ----------------------------

HEADING_PATTERNS = [
    r"^hướng dẫn\b.*",                            # HƯỚNG DẪN...
    r"^thông báo\b.*",                            # THÔNG BÁO...
    r"^chương\s+[ivxlcdm0-9]+[\s:.\-–—]+",        # CHƯƠNG I:
    r"^mục\s+\d+[\s:.\-–—]+",                     # Mục 1:
    r"^phần\s+[ivxlcdm0-9]+[\s:.\-–—]+",          # PHẦN I:
    r"^điều\s+\d+[a-zđ]?\b[\s:.\-–—]+",           # Điều 1.
    r"^[IVXLCDM]+\.\s+",                          # II. III.
    r"^\d+(?:\.\d+)*[\.)]\s+\S",                  # 1. 1) 1.1.
    r"^[a-zđ]\)\s+\S",                            # a) ...
    r"^[-–—]\s+\S",                               # - item
]
HEADING_RE = re.compile("|".join(f"(?:{p})" for p in HEADING_PATTERNS), re.IGNORECASE)

SENT_SPLIT_RE = re.compile(r"(?<=[\.\?\!…;:])\s+|\n+")

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    # Keep blank lines, but trim outer whitespace
    return "\n".join(lines).strip()

def safe_strip(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def word_count(text: str) -> int:
    return len([w for w in text.split() if w.strip()])

def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        k = safe_strip(x)
        if not k:
            continue
        key = k.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(k)
    return out

def iter_text_files(input_dir: Path, glob_pattern: str) -> Iterable[Path]:
    for path in sorted(input_dir.glob(glob_pattern)):
        if path.is_file():
            yield path

def write_json(path: Path, data: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ----------------------------
# Chunking that PRESERVES CONTEXT
# ----------------------------

@dataclass
class Chunk:
    context_path: str   # heading chain
    text: str           # raw chunk body

def is_heading(line: str) -> bool:
    return bool(HEADING_RE.match(line.strip()))

def build_context_path(stack: List[str]) -> str:
    cleaned = [safe_strip(s) for s in stack if safe_strip(s)]
    if not cleaned:
        return ""
    # keep last 3 headings to avoid too long prefixes
    return " > ".join(cleaned[-3:])

def split_to_context_chunks(text: str, max_chunk_words: int) -> List[Chunk]:
    """
    Split the document into chunks using headings as context anchors,
    then pack them to max_chunk_words to keep LLM focused.

    IMPORTANT: This is NOT the final passage split.
    It's just to keep context and reduce LLM failure.
    """
    text = normalize_text(text)
    if not text:
        return []

    lines = text.split("\n")
    heading_stack: List[str] = []
    blocks: List[Tuple[str, List[str]]] = []

    cur_lines: List[str] = []
    cur_context = ""

    def flush():
        nonlocal cur_lines, cur_context
        if any(ln.strip() for ln in cur_lines):
            blocks.append((cur_context, cur_lines))
        cur_lines = []

    for ln in lines:
        stripped = ln.strip()
        if is_heading(stripped):
            # start a new block at heading boundary
            flush()
            heading_stack.append(stripped)
            cur_context = build_context_path(heading_stack)
            # keep heading line inside block too (helps LLM produce self-contained passages)
            cur_lines.append(stripped)
        else:
            cur_lines.append(ln)

    flush()

    # Pack blocks into chunks
    chunks: List[Chunk] = []
    buf_lines: List[str] = []
    buf_context = ""

    def flush_buf():
        nonlocal buf_lines, buf_context
        if any(ln.strip() for ln in buf_lines):
            chunks.append(Chunk(context_path=buf_context, text="\n".join(buf_lines).strip()))
        buf_lines = []

    for context, blines in blocks:
        block_text = "\n".join(blines).strip()
        if not block_text:
            continue

        if context:
            buf_context = context

        candidate = ("\n".join(buf_lines + ["", block_text])).strip() if buf_lines else block_text
        if buf_lines and word_count(candidate) > max_chunk_words:
            flush_buf()
            buf_lines = [block_text]
        else:
            if buf_lines:
                buf_lines.append("")
            buf_lines.append(block_text)

        # If one block itself too large, split it by sentences with overlap
        if word_count(block_text) > max_chunk_words:
            flush_buf()
            chunks.extend(split_long_to_chunks(block_text, buf_context, max_chunk_words, overlap_words=50))

    flush_buf()
    return chunks

def split_long_to_chunks(text: str, context_path: str, max_words: int, overlap_words: int) -> List[Chunk]:
    sents = [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]
    if not sents:
        return []

    out: List[Chunk] = []
    i = 0
    while i < len(sents):
        wc = 0
        buf: List[str] = []
        start_i = i

        while i < len(sents):
            s = sents[i]
            s_wc = word_count(s)
            if buf and wc + s_wc > max_words:
                break
            buf.append(s)
            wc += s_wc
            i += 1

        chunk_text = " ".join(buf).strip()
        if chunk_text:
            out.append(Chunk(context_path=context_path, text=chunk_text))

        if i >= len(sents):
            break

        # overlap by approx words
        back_wc = 0
        j = i - 1
        while j > start_i and back_wc < overlap_words:
            back_wc += word_count(sents[j])
            j -= 1
        i = max(j + 1, start_i + 1)

    return out


# ----------------------------
# LLM prompts: semantic packing + context retention
# ----------------------------

REPAIR_PROMPT = """
Sửa output sau thành JSON hợp lệ theo schema và KHÔNG thêm chữ nào khác:

Schema:
{{
    "passages": [
        {{
            "text": "...",
            "needs_prefix": true/false
        }}
    ]
}}

Output cần sửa:
"""{bad}"""
""".strip()

SUMMARY_SYSTEM_PROMPT = """
Bạn là AI chuyên tóm tắt văn bản.

Yêu cầu:
- Tóm tắt ngắn gọn, tự nhiên, không cộc lốc.
- Tối đa {max_words} từ.
- Chỉ trả về 1 đoạn tóm tắt.
""".strip()

SUMMARY_USER_PROMPT = """
Hãy tóm tắt đoạn sau sao cho không quá {max_words} từ.
Tóm tắt ngắn gọn nhưng không cộc lốc và không làm mất ý chính.

Đoạn cần tóm tắt:
"{text}"
""".strip()


def call_chat(client: OpenAI, model: str, system: str, user: str, temperature: float, max_tokens: Optional[int] = None) -> str:
  "passages": [
    {{
      "text": "...",
      "needs_prefix": true/false
    }}
  ]
}}
Trong đó:
- text: passage (tối đa {max_words} từ)
- needs_prefix: đặt true nếu passage cần thêm context_path (ở ngoài) để tự chứa ngữ cảnh.
""".strip()

USER_PROMPT = """
context_path (ngữ cảnh chương/mục/tiêu đề gần nhất): "{context_path}"

Văn bản nguồn:
\"\"\"{text}\"\"\"

Hãy tạo các passage theo yêu cầu và trả về JSON.
""".strip()

REPAIR_PROMPT = """
Sửa output sau thành JSON hợp lệ theo schema và KHÔNG thêm chữ nào khác:

Schema:
{{
  "passages": [
    {{
      "text": "...",
      "needs_prefix": true/false
    }}
  ]
}}

Output cần sửa:
\"\"\"{bad}\"\"\"
""".strip()


def call_chat(client: OpenAI, model: str, system: str, user: str, temperature: float, max_tokens: Optional[int] = None) -> str:
    kwargs = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        **kwargs,
    )
    return resp.choices[0].message.content or ""

def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return None
    return None

def enforce_max_words(text: str, max_words: int) -> str:
    words = [w for w in text.split() if w.strip()]
    if len(words) <= max_words:
        return safe_strip(text)
    return safe_strip(" ".join(words[:max_words]))


def summarize_if_needed(
    text: str,
    client: OpenAI,
    model: str,
    temperature: float,
    max_words: int,
) -> str:
    if word_count(text) <= max_words:
        return safe_strip(text)

    system = SUMMARY_SYSTEM_PROMPT.format(max_words=max_words)
    user = SUMMARY_USER_PROMPT.format(max_words=max_words, text=text)
    raw = call_chat(client, model, system, user, temperature).strip()
    if raw and word_count(raw) <= max_words:
        return raw

    return enforce_max_words(raw or text, max_words)

def maybe_prefix_context(passage: str, context_path: str, max_words: int) -> str:
    """
    Prefix context in a compact way to prevent context loss.
    Must still respect max_words.
    """
    context_path = safe_strip(context_path)
    if not context_path:
        return enforce_max_words(passage, max_words)

    # Compact prefix; avoid overly long chain
    prefix = f"{context_path}: "
    combined = safe_strip(prefix + passage)

    if word_count(combined) <= max_words:
        return combined

    # If too long, shorten prefix aggressively (keep last segment only)
    short_ctx = context_path.split(">")[-1].strip()
    prefix2 = f"{short_ctx}: "
    combined2 = safe_strip(prefix2 + passage)
    if word_count(combined2) <= max_words:
        return combined2

    # Otherwise truncate the full combined
    return enforce_max_words(combined2, max_words)

def llm_passages_for_chunk(
    chunk: Chunk,
    client: OpenAI,
    model: str,
    temperature: float,
    max_words: int,
    repair_rounds: int = 1,
) -> List[str]:
    system = SYSTEM_PROMPT.format(max_words=max_words)
    user = USER_PROMPT.format(context_path=chunk.context_path, text=chunk.text)

    raw = call_chat(client, model, system, user, temperature)
    data = extract_json_object(raw)

    if data is None and repair_rounds > 0:
        raw2 = call_chat(client, model, "Bạn là AI chuyên sửa JSON.", REPAIR_PROMPT.format(bad=raw), 0.0)
        data = extract_json_object(raw2)

    if not isinstance(data, dict):
        return []

    passages = data.get("passages", [])
    if not isinstance(passages, list):
        return []

    out: List[str] = []
    for item in passages:
        if not isinstance(item, dict):
            continue
        txt = safe_strip(item.get("text", ""))
        needs_prefix = bool(item.get("needs_prefix", False))

        if not txt:
            continue

        # Ensure max words with summarization when needed
        txt = summarize_if_needed(txt, client, model, temperature, max_words)

        # Add context prefix if model indicates it lacks context
        if needs_prefix:
            txt = maybe_prefix_context(txt, chunk.context_path, max_words)

        # Filter very short fragments (often low-quality)
        if word_count(txt) < 10:
            continue

        out.append(txt)

    return dedupe_keep_order(out)


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate DPR passages with local LLM, minimizing context loss.")
    ap.add_argument("--input-dir", default=str(Path(__file__).resolve().parent / "raw_data"))
    ap.add_argument("--glob", default="*.txt")
    ap.add_argument("--encoding", default="utf-8")
    ap.add_argument("--output", default=str(Path(__file__).resolve().parent / "process_pass.json"))

    ap.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:5001/v1"))
    ap.add_argument("--api-key", default="sk-6af49da8d0b54065a2f7929ac7a43512")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--temperature", type=float, default=0.05)

    ap.add_argument("--max-words", type=int, default=120)
    ap.add_argument("--max-chunk-words", type=int, default=650, help="Bigger chunk to help LLM keep context.")
    ap.add_argument("--min-passage-words", type=int, default=15)

    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Set --api-key or OPENAI_API_KEY.")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    all_rows: List[Dict[str, Any]] = []

    for path in iter_text_files(Path(args.input_dir), args.glob):
        text = normalize_text(path.read_text(encoding=args.encoding))
        if not text:
            continue

        chunks = split_to_context_chunks(text, max_chunk_words=args.max_chunk_words)

        passages: List[str] = []
        for ch in chunks:
            passages.extend(
                llm_passages_for_chunk(
                    ch,
                    client=client,
                    model=args.model,
                    temperature=args.temperature,
                    max_words=args.max_words,
                )
            )

        passages = [p for p in dedupe_keep_order(passages) if word_count(p) >= args.min_passage_words]

        for i, p in enumerate(passages, start=1):
            all_rows.append(
                {
                    "passage_id": f"{path.stem}__{i:03d}",
                    "ref": path.name,
                    "text": p,
                }
            )

    write_json(Path(args.output), all_rows)
    print(f"Wrote {len(all_rows)} passages to {args.output}")


if __name__ == "__main__":
    main()