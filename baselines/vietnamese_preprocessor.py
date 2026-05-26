# vietnamese_preprocessor.py
# Tiền xử lý văn bản tiếng Việt cho BM25/TF-IDF
# Hỗ trợ: tách từ (word segmentation), loại bỏ stopwords, chuẩn hóa

import re
import unicodedata
from typing import List

# ====== VIETNAMESE STOPWORDS ======
# Danh sách stopwords phổ biến trong tiếng Việt
VIETNAMESE_STOPWORDS = {
    # Đại từ, liên từ, giới từ
    "và", "của", "các", "có", "được", "cho", "là", "với", "này",
    "trong", "đã", "không", "những", "để", "một", "từ", "theo",
    "về", "tại", "do", "khi", "như", "cũng", "hay", "hoặc",
    "nếu", "thì", "mà", "bị", "đến", "ra", "vào", "lên",
    "trên", "dưới", "sau", "trước", "giữa", "ngoài", "bên",
    # Trợ từ, phụ từ
    "đó", "đây", "ấy", "nào", "gì", "sao", "thế", "rằng",
    "rất", "quá", "lắm", "hơn", "nhất", "chỉ", "còn", "đang",
    "sẽ", "vẫn", "lại", "đều", "luôn", "mới", "chưa", "vừa",
    # Số đếm, lượng từ
    "mỗi", "tất", "cả", "mọi", "nhiều", "ít", "vài",
    # Phổ biến khác
    "thì", "mà", "nhưng", "tuy", "dù", "nên", "vì", "bởi",
    "cùng", "cũng", "việc", "sự", "cách", "điều", "người",
    "thể", "phải", "nào", "đó", "ai", "đâu",
}


def normalize_text(text: str) -> str:
    """Chuẩn hóa văn bản tiếng Việt:
    - Unicode NFC normalization
    - Chuyển về chữ thường
    - Loại bỏ ký tự đặc biệt thừa, giữ lại dấu tiếng Việt
    - Chuẩn hóa khoảng trắng
    """
    # Unicode normalization (NFC)
    text = unicodedata.normalize("NFC", text)
    # Chữ thường
    text = text.lower()
    # Loại bỏ ký tự đặc biệt, giữ chữ cái (kể cả có dấu), số, khoảng trắng
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simple_tokenize(text: str) -> List[str]:
    """Tokenize đơn giản: tách theo khoảng trắng sau khi normalize.
    Phù hợp khi không có underthesea/pyvi.
    """
    text = normalize_text(text)
    tokens = text.split()
    return tokens


def tokenize_with_underthesea(text: str) -> List[str]:
    """Tokenize bằng underthesea (tách từ tiếng Việt).
    underthesea xử lý tốt từ ghép: 'sinh viên', 'tín chỉ', 'học phần'...
    """
    from underthesea import word_tokenize
    text = normalize_text(text)
    segmented = word_tokenize(text, format="text")
    # underthesea dùng "_" nối các từ ghép → thay bằng space rồi split
    tokens = segmented.split()
    return tokens


def remove_stopwords(tokens: List[str], stopwords: set = None) -> List[str]:
    """Loại bỏ stopwords khỏi danh sách tokens."""
    if stopwords is None:
        stopwords = VIETNAMESE_STOPWORDS
    return [t for t in tokens if t not in stopwords and len(t) > 1]


def preprocess(text: str, use_underthesea: bool = False,
               remove_sw: bool = True) -> List[str]:
    """Pipeline tiền xử lý hoàn chỉnh cho 1 đoạn văn bản.
    
    Args:
        text: Văn bản đầu vào
        use_underthesea: Có dùng underthesea để tách từ hay không
        remove_sw: Có loại bỏ stopwords hay không
    
    Returns:
        Danh sách tokens đã tiền xử lý
    """
    if use_underthesea:
        try:
            tokens = tokenize_with_underthesea(text)
        except ImportError:
            print("⚠️  underthesea chưa được cài. Dùng simple tokenize.")
            tokens = simple_tokenize(text)
    else:
        tokens = simple_tokenize(text)

    if remove_sw:
        tokens = remove_stopwords(tokens)

    return tokens


def preprocess_corpus(texts: List[str], **kwargs) -> List[List[str]]:
    """Tiền xử lý toàn bộ corpus (danh sách văn bản)."""
    return [preprocess(t, **kwargs) for t in texts]
