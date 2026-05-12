import re
import json
import unicodedata
from docx import Document


def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace("Ð", "Đ")
    text = text.replace("\ufeff", "")
    return text.strip()


def get_list_level(para):
    try:
        return para._element.pPr.numPr.ilvl.val
    except:
        return None


def split_docx(file_path):
    doc = Document(file_path)

    results = []
    passage_id = 1

    current_dieu = None
    current_khoan = None
    buffer = []

    for para in doc.paragraphs:
        text = normalize_text(para.text)
        if not text:
            continue

        # ===== FIX 1: Detect Điều CHỈ ở đầu dòng =====
        dieu_match = re.match(r"^\s*[ĐÐ]iều\s+(\d+)", text, re.IGNORECASE)
        if dieu_match:
            if current_khoan and buffer:
                results.append({
                    "passage_id": f"p_{passage_id:04d}",
                    "clause_ref": f"Điều {current_dieu} Khoản {current_khoan}",
                    "text": "\n".join(buffer)
                })
                passage_id += 1

            current_dieu = dieu_match.group(1)
            current_khoan = None
            buffer = []
            continue

        # ===== FIX 2: Khoản 1 implicit =====
        if current_dieu and current_khoan is None:
            current_khoan = "1"
            buffer = [text]
            continue

        # ===== Detect list level =====
        level = get_list_level(para)

        # ===== Detect Khoản =====
        khoan_match = re.match(r"(\d+)[\.\)]\s*(.*)", text)

        if not khoan_match:
            alt = re.match(r"^\s*Khoản\s+(\d+)", text, re.IGNORECASE)
            if alt:
                khoan_match = alt

        is_bullet = text.startswith("-")
        is_alpha = re.match(r"[a-zA-Z]\)", text)
        is_khoan_list = (level == 0)

        if khoan_match or (is_khoan_list and not is_bullet and not is_alpha):

            if current_khoan and buffer:
                results.append({
                    "passage_id": f"p_{passage_id:04d}",
                    "clause_ref": f"Điều {current_dieu} Khoản {current_khoan}",
                    "text": "\n".join(buffer)
                })
                passage_id += 1

            if khoan_match:
                current_khoan = khoan_match.group(1)
                content = text.split(".", 1)[-1].strip() if "." in text else text
            else:
                current_khoan = str((int(current_khoan) + 1) if current_khoan else 1)
                content = text

            buffer = [content]

        else:
            # ===== FIX 3: merge dòng =====
            if current_khoan:
                if buffer:
                    if not buffer[-1].endswith((".", ":", ";")):
                        buffer[-1] += " " + text
                    else:
                        buffer.append(text)
                else:
                    buffer.append(text)

    # ===== flush cuối =====
    if current_khoan and buffer:
        results.append({
            "passage_id": f"p_{passage_id:04d}",
            "clause_ref": f"Điều {current_dieu} Khoản {current_khoan}",
            "text": "\n".join(buffer)
        })

    return results


# ===== RUN =====
input_file = "E:/A_Project/NLP/document/word/quyche_768.docx"

data = split_docx(input_file)

with open("output.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Done:", len(data))