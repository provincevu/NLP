import json
from typing import Dict, List

# === HÀM ĐỌC DANH SÁCH CÂU HỎI CẦN SỬA ===
def load_edited_questions(file_path: str) -> Dict[str, str]:
    """Đọc file JSON chứa danh sách câu hỏi cần sửa, trả về dict {qid: question}"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Nếu là list các object có qid và question
    if isinstance(data, list):
        return {item["qid"]: item["question"] for item in data if "qid" in item and "question" in item}
    # Nếu đã là dict {qid: question}
    return data


def load_questions(file_path: str) -> List[Dict]:
    """Đọc file JSON chứa danh sách câu hỏi gốc"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_questions(file_path: str, questions: List[Dict]) -> None:
    """Ghi danh sách câu hỏi đã cập nhật ra file JSON"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)


def update_questions(input_file: str, edited_file: str, output_file: str = None) -> List[Dict]:
    """
    Cập nhật câu hỏi trong input_file dựa trên edited_file.
    
    Args:
        input_file: Đường dẫn file questions.json (câu hỏi gốc)
        edited_file: Đường dẫn file questions_edited.json (chứa các câu hỏi đã sửa theo qid)
        output_file: Đường dẫn file đầu ra (nếu None thì ghi đè lên input_file)
    """
    questions = load_questions(input_file)
    edited_map = load_edited_questions(edited_file)
    
    updated_count = 0
    for q in questions:
        qid = q.get("qid")
        if qid in edited_map:
            q["question"] = edited_map[qid]
            updated_count += 1
    
    output_path = output_file if output_file else input_file
    save_questions(output_path, questions)
    
    print(f"✅ Đã cập nhật {updated_count} câu hỏi. Kết quả lưu tại: {output_path}")
    return questions


# === SỬ DỤNG ===
if __name__ == "__main__":
    input_file = "questions.json"           # File câu hỏi gốc
    edited_file = "questions_edited.json"   # File chứa câu hỏi đã sửa
    
    # edited_file có cấu trúc:
    # [
    #     {"qid": "q_0002", "question": "Tổ chức đào tạo theo tín chỉ gồm những hoạt động nào?"},
    #     {"qid": "q_0005", "question": "Sinh viên năm nhất cần tích lũy bao nhiêu tín chỉ?"}
    # ]
    
    update_questions(input_file, edited_file, output_file="questions_updated.json")