import json

# === DANH SÁCH PASSAGE_ID ĐÃ ĐƯỢC GIỮ LẠI (TỔNG HỢP TỪ PHẦN TRƯỚC) ===
expected_passage_ids = {
    "p_0001", "p_0003", "p_0004", "p_0005", "p_0006", "p_0007", "p_0008", "p_0009",
    "p_0010", "p_0011", "p_0012", "p_0013", "p_0016", "p_0017", "p_0018", "p_0022",
    "p_0023", "p_0024", "p_0026", "p_0027", "p_0032", "p_0033", "p_0036", "p_0037",
    "p_0039", "p_0041", "p_0042", "p_0043", "p_0044", "p_0045", "p_0047", "p_0048",
    "p_0049", "p_0050", "p_0051", "p_0053", "p_0055", "p_0057", "p_0058", "p_0060",
    "p_0061", "p_0062", "p_0063", "p_0066", "p_0067", "p_0070", "p_0071", "p_0072",
    "p_0075", "p_0076", "p_0077", "p_0078", "p_0093", "p_0094", "p_0095", "p_0096",
    "p_0097", "p_0098", "p_0099", "p_0100", "p_0101", "p_0102", "p_0103", "p_0110",
    "p_0111", "p_0115", "p_0116", "p_0122", "p_0125", "p_0128", "p_0129", "p_0136",
    "p_0139", "p_0142"
}

def load_json_data(file_path):
    """Đọc file JSON và trả về danh sách các passage"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_passage_ids_from_data(data):
    """Lấy danh sách passage_id từ dữ liệu JSON"""
    return {item["passage_id"] for item in data if "passage_id" in item}

def check_mismatch(file_path):
    """Kiểm tra sự khác biệt giữa passage_id trong file và danh sách kỳ vọng"""
    data = load_json_data(file_path)
    actual_ids = get_passage_ids_from_data(data)
    
    missing_in_file = expected_passage_ids - actual_ids      # Có trong danh sách giữ lại nhưng file thiếu
    extra_in_file = actual_ids - expected_passage_ids        # Có trong file nhưng không nằm trong danh sách giữ lại
    
    return {
        "actual_ids": actual_ids,
        "total_actual": len(actual_ids),
        "missing": missing_in_file,
        "extra": extra_in_file,
        "total_missing": len(missing_in_file),
        "total_extra": len(extra_in_file)
    }

# === SỬ DỤNG ===
if __name__ == "__main__":
    json_file_path = "new_passages.json"
    
    result = check_mismatch(json_file_path)
    
    print(f"Tổng số passage_id trong file: {result['total_actual']}")
    print(f"Tổng số passage_id kỳ vọng (giữ lại): {len(expected_passage_ids)}")
    print("-" * 50)
    
    if result["missing"]:
        print(f"THIẾU {result['total_missing']} passage_id (có trong danh sách giữ lại nhưng file không có):")
        for pid in sorted(result["missing"]):
            print(f"   - {pid}")
    else:
        print("Không thiếu passage_id nào so với danh sách giữ lại.")
    
    if result["extra"]:
        print(f"THỪA {result['total_extra']} passage_id (có trong file nhưng không nằm trong danh sách giữ lại):")
        for pid in sorted(result["extra"]):
            print(f"   - {pid}")
    else:
        print("Không có passage_id thừa nào so với danh sách giữ lại.")