import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

SYSTEM_PROMPT = """
Bạn là AI chuyên tạo synthetic query cho hệ thống retrieval/RAG.

Nhiệm vụ:
- Dựa chủ yếu vào passage để sinh thêm nhiều câu hỏi mới phù hợp.
- Các câu hỏi phải có thể được trả lời trực tiếp từ passage.
- Không thêm thông tin ngoài passage.
- Không thay đổi ý nghĩa của passage.

Danh sách câu hỏi hiện có chỉ dùng để tham khảo nhằm:
- tránh lặp lại
- tránh tạo câu hỏi quá giống nhau
- tăng độ đa dạng cách hỏi

Ưu tiên:
- đa dạng cách diễn đạt
- nhiều kiểu diễn đạt tự nhiên
- câu hỏi ngắn
- câu hỏi đời thường
- câu hỏi kiểu sinh viên
- câu hỏi dùng từ đồng nghĩa
- câu hỏi viết tắt tự nhiên nếu phù hợp

Yêu cầu:
- Không sinh câu hỏi trùng hoặc gần giống với câu hỏi đã có.
- Mỗi câu hỏi phải khác biệt rõ ràng về wording.
- Chỉ trả về danh sách câu hỏi.
- Trả lời bằng tiếng Việt.
- Không đánh số.
- Mỗi dòng là một câu hỏi.
- Không giải thích.
""".strip()

PROMPT_TAIL = """
Hãy tạo tối thiểu 20 câu hỏi khác nhau dựa CHỈ trên passage.

Ràng buộc:
- Không thêm thông tin ngoài passage, không suy diễn vượt quá nội dung passage.
- Mọi câu hỏi phải được passage hỗ trợ rõ ràng.
- Câu hỏi ngắn gọn, tự nhiên, tiếng Việt.
- Không đánh số, mỗi dòng một câu hỏi, không giải thích.

Yêu cầu đa dạng bắt buộc (phải có đủ nhiều kiểu):
- Câu hỏi điều kiện (ví dụ: nếu ... thì ...?)
- Câu hỏi hậu quả/kết quả (ví dụ: không ... thì thế nào?)
- Câu hỏi về trách nhiệm/nghĩa vụ
- Câu hỏi về thời gian/thời hạn
- Câu hỏi ngắn kiểu sinh viên hỏi nhanh
- Câu hỏi dùng từ đồng nghĩa hoặc viết tắt tự nhiên
- Câu hỏi đảo cấu trúc câu
- Câu hỏi hỏi lại quy định theo cách tự nhiên
- Câu hỏi suy ra trực tiếp từ quy định nhưng vẫn được passage hỗ trợ rõ ràng
""".strip()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_question(text: str) -> str:
    return " ".join(text.strip().lower().split())


def build_prompts(passages):
    prompts = {}
    for item in passages:
        passage_id = item["passage_id"]
        passage_text = item["text"]
        prompt = build_prompt(passage_text)
        prompts[passage_id] = prompt

    return prompts


def write_prompts(prompts, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for idx, (passage_id, prompt) in enumerate(sorted(prompts.items())):
            if idx:
                f.write("\n\n---\n\n")
            f.write(f"# {passage_id}\n\n{prompt}")


def build_prompt(passage_text: str, existing_questions=None):
    if existing_questions:
        existing_lines = "\n".join(f"- {q}" for q in existing_questions)
        existing_section = f"Existing questions:\n{existing_lines}\n\n"
    else:
        existing_section = ""

    return (
        "Passage:\n"
        f"\"{passage_text}\"\n\n"
        f"{existing_section}"
        f"{PROMPT_TAIL}"
    )


def call_model(client, model: str, system_prompt: str, user_prompt: str, temperature: float):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        stream=True,
    )

    chunks = []
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            chunks.append(delta.content)

    return "".join(chunks)


def parse_questions(text: str):
    results = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        results.append(line)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate new questions per passage and export to JSON."
    )
    parser.add_argument(
        "--passages",
        default=str(Path(__file__).resolve().parents[1] / "data" / "passages.json"),
        help="Path to passages.json",
    )
    parser.add_argument(
        "--prompts-output",
        default="",
        help="Optional output file to write prompts",
    )
    parser.add_argument(
        "--output-questions",
        default=str(Path(__file__).resolve().parent / "new_questions.json"),
        help="Output file for generated questions",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:5001/v1"),
        help="OpenAI-compatible server base URL",
    )
    parser.add_argument(
        "--api-key",
        default= "sk-6af49da8d0b54065a2f7929ac7a43512",
        help="API key for the server",
    )
    parser.add_argument(
        "--model",
        default="deepseek-v4-flash",
        help="Model name",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--start-qid",
        type=int,
        default=1000,
        help="Start index for qid numbering",
    )
    parser.add_argument(
        "--max-new",
        type=int,
        default=0,
        help="Max new questions per passage (0 = no limit)",
    )

    args = parser.parse_args()

    passages = load_json(Path(args.passages))

    if not args.api_key:
        raise SystemExit("Missing API key. Set --api-key or OPENAI_API_KEY.")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    if args.prompts_output:
        prompts = build_prompts(passages)
        write_prompts(prompts, Path(args.prompts_output))

    output_counts = {}
    next_id = args.start_qid

    output_path = Path(args.output_questions)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    with output_path.open("w", encoding="utf-8") as f:
        f.write("[\n")
        first = True

        for item in sorted(passages, key=lambda p: p["passage_id"]):
            passage_id = item["passage_id"]
            passage_text = item["text"]

            user_prompt = build_prompt(passage_text)
            raw_text = call_model(
                client=client,
                model=args.model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=args.temperature,
            )

            added = 0
            for q in parse_questions(raw_text):
                if not q:
                    continue
                payload = {
                    "qid": f"q_{next_id:04d}",
                    "question": q,
                    "positive_passage_id": passage_id,
                }
                if not first:
                    f.write(",\n")
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.flush()
                first = False
                next_id += 1
                added += 1
                total_written += 1
                if args.max_new and added >= args.max_new:
                    break

            output_counts[passage_id] = added

        f.write("\n]\n")

    print(f"Wrote {total_written} questions to {args.output_questions}")
    for passage_id, count in sorted(output_counts.items()):
        print(f"{passage_id}: {count}")


if __name__ == "__main__":
    main()
