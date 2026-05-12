import re
import json
import unicodedata
from docx import Document


# ===== Normalize text (fix tiếng Việt + ký tự lỗi) =====
def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace("Ð", "Đ")  # fix lỗi phổ biến
    text = text.replace("\ufeff", "")  # BOM
    return text.strip()


def split_docx(file_path):
    doc = Document(file_path)

    results = []
    passage_id = 1

    current_dieu = None
    current_khoan = None
    buffer = []

    for para in doc.paragraphs:
        raw_text = para.text
        text = normalize_text(raw_text)

        if not text:
            continue

        # ===== DEBUG (bật nếu cần) =====
        # print(repr(text))

        # ===== Detect Điều =====
        dieu_match = re.search(r"[ĐÐ]iều\s+(\d+)", text, re.IGNORECASE)
        if dieu_match:
            # flush khoản trước
            if current_khoan and buffer:
                results.append({
                    "passage_id": f"p_{passage_id:04d}",
                    "clause_ref": f"Điều {current_dieu} Khoản {current_khoan}",
                    "text": " ".join(buffer).strip()
                })
                passage_id += 1

            current_dieu = dieu_match.group(1)
            current_khoan = None
            buffer = []
            continue

        # ===== Detect Khoản =====

        # Case 1: có "1." hoặc "1)"
        khoan_match = re.match(r"(\d+)[\.\)]\s*(.*)", text)

        # Case 2: có "Khoản 1"
        if not khoan_match:
            alt = re.search(r"Khoản\s+(\d+)", text, re.IGNORECASE)
            if alt:
                khoan_match = alt

        # Case 3: Word numbering (không có số trong text)
        is_list_para = para.style.name.startswith("List")

        if khoan_match or is_list_para:

            # flush khoản cũ
            if current_khoan and buffer:
                results.append({
                    "passage_id": f"p_{passage_id:04d}",
                    "clause_ref": f"Điều {current_dieu} Khoản {current_khoan}",
                    "text": " ".join(buffer).strip()
                })
                passage_id += 1

            # xác định số khoản
            if khoan_match:
                current_khoan = khoan_match.group(1)
                content = text.split(".", 1)[-1].strip() if "." in text else text
            else:
                # fallback nếu không có số → tự tăng
                current_khoan = str((int(current_khoan) + 1) if current_khoan else 1)
                content = text

            buffer = [content]

        else:
            # dòng tiếp theo của khoản hiện tại
            if current_khoan:
                buffer.append(text)

    # ===== flush cuối =====
    if current_khoan and buffer:
        results.append({
            "passage_id": f"p_{passage_id:04d}",
            "clause_ref": f"Điều {current_dieu} Khoản {current_khoan}",
            "text": " ".join(buffer).strip()
        })

    return results


# ===== RUN =====
input_file = "E:/A_Project/NLP/document/word/quyche_768.docx"

data = split_docx(input_file)

with open("output.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Done:", len(data))