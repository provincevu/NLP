import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from openai import OpenAI

# ---- Prompt ----

SYSTEM_PROMPT = """
Bạn là AI chuyên tạo synthetic queries để huấn luyện Dense Passage Retrieval (DPR) cho văn bản tiếng Việt.

Nguyên tắc:
- Mỗi câu hỏi PHẢI được trả lời trực tiếp từ passage.
- Không thêm thông tin ngoài passage. Không suy đoán. Không hỏi "tại sao" nếu passage không nêu lý do.
- Tránh tạo nhiều câu hỏi chỉ khác nhau vài từ (paraphrase spam).
- Ưu tiên đa dạng intent và cách diễn đạt tự nhiên như người dùng thật.

Yêu cầu DPR-friendly:
- Giảm bám từ khóa: tránh copy nguyên cụm dài từ passage (>= 4 từ liên tiếp),
  TRỪ các thực thể bắt buộc như URL, tên hệ thống, mã văn bản, con số.
- Tạo câu hỏi theo tình huống/mục tiêu (ví dụ: "mình làm X rồi mà..."), nhưng vẫn answerable từ passage.

Đầu ra:
- CHỈ trả về JSON hợp lệ theo schema:
{
  "questions": [
    {
      "q": "câu hỏi",
      "evidence": "trích nguyên văn 8-20 từ liên tiếp trong passage chứng minh câu hỏi answerable",
      "intent": "howto|where|what|check|requirement|time|eligibility|procedure|definition"
    }
  ]
}
""".strip()

USER_PROMPT = """
Passage:
\"\"\"{passage}\"\"\"

Hãy tạo {n} câu hỏi KHÁC NHAU về intent (không paraphrase gần giống).
Nếu passage không có thông tin về hậu quả/chế tài/lý do thì KHÔNG tạo câu hỏi dạng đó.

Chỉ trả JSON.
""".strip()

# ---- Utilities ----

BAD_Q_RE = re.compile(
    r"\b(tại sao|vì sao|có sao không|hợp lệ không|bị xử lý|bị phạt|có bị)\b",
    re.IGNORECASE
)

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def normalize_question(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            return None
    return None

def call_model(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float = 1.0,
    max_tokens: int = 1600,
    frequency_penalty: float = 0.2,
    presence_penalty: float = 0.0,
    ):
    import time

    last_err = None
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                max_tokens=max_tokens,
                # stream=False  # default
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))

    raise last_err

def passes_basic_filters(q: str, passage: str) -> bool:
    if not q or len(q) < 6:
        return False
    # avoid speculative questions unless passage supports (simple heuristic)
    if BAD_Q_RE.search(q):
        return False
    return True

def evidence_is_valid(evidence: str, passage: str) -> bool:
    ev = evidence.strip()
    if not ev:
        return False
    # must be verbatim substring (hard check)
    if ev not in passage:
        return False
    # length constraint
    wc = len(ev.split())
    return 8 <= wc <= 24

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", default=str(Path(__file__).resolve().parents[1] / "data" / "passages4gen.json"))
    ap.add_argument("--output-questions", default=str(Path(__file__).resolve().parent / "new_questions.json"))
    ap.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:5001/v1"))
    ap.add_argument("--api-key", default="sk-5999e2cfca9a433bb9534e18c1a83350")
    ap.add_argument("--model", default="deepseek-v4-flash")

    ap.add_argument("--n", type=int, default=18, help="Questions per passage before filtering")
    ap.add_argument("--temperature", type=float, default=0.35)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=1600)

    ap.add_argument("--start-qid", type=int, default=5000)
    ap.add_argument("--max-new", type=int, default=0)

    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Set OPENAI_API_KEY or --api-key")

    passages = load_json(Path(args.passages))
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    out_path = Path(args.output_questions)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    next_id = args.start_qid
    total_written = 0

    with out_path.open("w", encoding="utf-8") as f:
        f.write("[\n")
        first = True

        for item in sorted(passages, key=lambda p: p["passage_id"]):
            passage_id = item["passage_id"]
            passage_text = item["text"]

            raw = call_model(
                client, args.model,
                SYSTEM_PROMPT,
                USER_PROMPT.format(passage=passage_text, n=args.n),
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )

            data = extract_json(raw)
            if not data or "questions" not in data or not isinstance(data["questions"], list):
                continue

            # hard dedupe (string-level)
            seen = set()
            added = 0

            for obj in data["questions"]:
                if not isinstance(obj, dict):
                    continue
                q = (obj.get("q") or "").strip()
                ev = (obj.get("evidence") or "").strip()

                if not passes_basic_filters(q, passage_text):
                    continue
                if not evidence_is_valid(ev, passage_text):
                    continue

                nq = normalize_question(q)
                if nq in seen:
                    continue
                seen.add(nq)

                payload = {
                    "qid": f"q_{next_id:04d}",
                    "question": q,
                    "positive_passage_id": passage_id,
                    "evidence": ev,
                    "intent": obj.get("intent", ""),
                }

                if not first:
                    f.write(",\n")
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.flush()
                first = False

                next_id += 1
                total_written += 1
                added += 1

                if args.max_new and added >= args.max_new:
                    break

            # (optional) you can print per passage stats here

        f.write("\n]\n")

    print(f"Wrote {total_written} questions to {args.output_questions}")

if __name__ == "__main__":
    main()