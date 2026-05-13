<h1 align="center">
  🔍 Dense Passage Retrieval cho Quy chế Đào tạo HUMG
</h1>

<p align="center">
  <em>Hệ thống truy hồi ngữ nghĩa (Semantic Retrieval) các quy định đào tạo tín chỉ — Trường Đại học Mỏ - Địa chất</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg" alt="HuggingFace">
  <img src="https://img.shields.io/badge/FAISS-Facebook-green.svg" alt="FAISS">
  <img src="https://img.shields.io/badge/PhoBERT-VinAI-red.svg" alt="PhoBERT">
</p>

<p align="center">
  <a href="README.md">🇻🇳 Tiếng Việt</a> · <a href="README_EN.md">🇬🇧 English</a>
</p>

---

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Cài đặt](#-cài-đặt)
- [Pipeline hoạt động](#-pipeline-hoạt-động)
  - [Bước 1 — Chuẩn bị dữ liệu](#bước-1--chuẩn-bị-dữ-liệu)
  - [Bước 2 — Sinh câu hỏi huấn luyện](#bước-2--sinh-câu-hỏi-huấn-luyện-synthetic-data)
  - [Bước 3 — Huấn luyện mô hình DPR](#bước-3--huấn-luyện-mô-hình-dpr)
  - [Bước 4 — Xây dựng chỉ mục FAISS](#bước-4--xây-dựng-chỉ-mục-faiss)
  - [Bước 5 — Truy vấn & Đánh giá](#bước-5--truy-vấn--đánh-giá)
- [Định dạng dữ liệu](#-định-dạng-dữ-liệu)
- [Kết quả & Đánh giá](#-kết-quả--đánh-giá)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
- [Tham khảo](#-tham-khảo)
- [Tác giả](#-tác-giả)

---

## 🎯 Giới thiệu

### Bài toán

Sinh viên thường gặp khó khăn khi tìm kiếm thông tin trong quy chế đào tạo tín chỉ của trường (hàng trăm điều khoản, khoản, điểm). Việc tìm kiếm truyền thống dựa trên từ khóa (keyword-based) thường cho kết quả kém khi câu hỏi không chứa đúng từ khóa trong văn bản quy chế.

### Giải pháp

Dự án xây dựng hệ thống **Dense Passage Retrieval (DPR)** — một phương pháp truy hồi ngữ nghĩa sử dụng mô hình ngôn ngữ **PhoBERT** (VinAI) để mã hoá câu hỏi và đoạn văn thành vector embedding, cho phép tìm kiếm dựa trên **ý nghĩa ngữ nghĩa** thay vì chỉ khớp từ khóa.

### Ý tưởng cốt lõi

```
Câu hỏi sinh viên ──► Encoder ──► Vector q
                                      │
                                      ▼
                              Cosine Similarity / Inner Product
                                      │
Các đoạn quy chế ──► Encoder ──► Vector p₁, p₂, ..., pₙ
                                      │
                                      ▼
                              Top-K đoạn văn liên quan nhất
```

> **Ví dụ:** Sinh viên hỏi *"Nghỉ bao nhiêu buổi thì bị cấm thi?"* → Hệ thống trả về đoạn quy chế về việc vắng quá 20% thời gian học, dù câu hỏi không chứa từ "20%" hay "vắng".

---

## 🏗 Kiến trúc hệ thống

```
┌──────────────────────────────────────────────────────────────────┐
│                        TRAINING PHASE                           │
│                                                                  │
│  passages.json ──► LLM (Synthetic QA) ──► questions.json        │
│                                              │                   │
│                                    split (train/val/test)        │
│                                              │                   │
│                                              ▼                   │
│                                   train_dpr_inbatch.py           │
│                                   (PhoBERT + In-Batch Negatives) │
│                                              │                   │
│                                              ▼                   │
│                                       dpr_ckpt/ (checkpoint)     │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                       INFERENCE PHASE                            │
│                                                                  │
│  passages.json ──► encode_passages.py ──► passage_embeddings.npy │
│                                              │                   │
│                                              ▼                   │
│                                        build_faiss.py            │
│                                     (FAISS IndexFlatIP)          │
│                                              │                   │
│                                              ▼                   │
│  User query ──► encode query ──► FAISS search ──► Top-K passages │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📁 Cấu trúc thư mục

```
NLP/
├── README.md
│
├── train_dpr_inbatch.py          # 🧠 Huấn luyện DPR (Dual Encoder + In-Batch Negatives)
├── split_questions.py            # ✂️  Chia dữ liệu train/val/test
├── DPR.ipynb                     # 📓 Notebook tham khảo (Haystack DPR tutorial)
│
├── generate_data/                # 🔄 Sinh dữ liệu huấn luyện bằng LLM
│   ├── build_prompts.py          #     Tạo prompt & gọi LLM sinh câu hỏi
│   ├── concat_json.py            #     Gộp/merge nhiều file JSON
│   ├── test.py                   #     Script test thử gọi LLM API
│   ├── new_questions.json        #     Câu hỏi đã sinh từ LLM
│   └── long_questions.json       #     Câu hỏi dài cần xử lý
│
├── json_data/                    # 🔧 Công cụ xử lý & làm sạch dữ liệu
│   ├── check_data.py             #     Kiểm tra tính nhất quán passage_id
│   ├── edit_quest.py             #     Cập nhật/sửa câu hỏi theo qid
│   ├── questions.json            #     Tập câu hỏi gốc
│   └── questions_edited.json     #     Câu hỏi đã chỉnh sửa
│
├── inferences/                   # 🚀 Pipeline suy luận (inference)
│   ├── encode_passages.py        #     Mã hoá toàn bộ passages → embeddings
│   ├── build_faiss.py            #     Xây chỉ mục FAISS từ embeddings
│   ├── retrieve.py               #     Truy vấn tương tác (interactive)
│   └── evaluate_retrieval.py     #     Đánh giá Recall@K, MRR
│
├── data/                         # 📊 Dữ liệu (trong .gitignore)
│   ├── passages.json             #     Các đoạn quy chế đào tạo
│   ├── questions.json            #     Câu hỏi huấn luyện
│   ├── splits/                   #     Dữ liệu đã chia train/val/test
│   ├── passage_embeddings.npy    #     Vector embeddings đã mã hoá
│   ├── passage_id_map.json       #     Ánh xạ index → passage_id
│   └── faiss_index_last.ivf      #     Chỉ mục FAISS
│
└── .gitignore
```

---

## ⚙️ Cài đặt

### Yêu cầu hệ thống

- Python ≥ 3.9
- GPU (khuyến nghị) — có thể chạy trên CPU nhưng sẽ chậm hơn đáng kể

### Cài đặt các thư viện

```bash
pip install torch transformers tqdm faiss-cpu numpy
```

> 💡 Nếu có GPU NVIDIA, cài `faiss-gpu` thay vì `faiss-cpu` để tăng tốc tìm kiếm.

```bash
# Tuỳ chọn: nếu muốn sinh dữ liệu bằng LLM
pip install openai
```

---

## 🔄 Pipeline hoạt động

### Bước 1 — Chuẩn bị dữ liệu

Dữ liệu đầu vào là file `data/passages.json` chứa các đoạn văn bản được trích xuất từ quy chế đào tạo tín chỉ của trường HUMG. Mỗi passage bao gồm:

| Trường | Mô tả | Ví dụ |
|--------|--------|-------|
| `passage_id` | Mã định danh duy nhất | `"p_0017"` |
| `clause_ref` | Tham chiếu điều khoản | `"Điều 5 Khoản 1"` |
| `text` | Nội dung đoạn quy chế | `"Tín chỉ (TC) là đơn vị tính..."` |

### Bước 2 — Sinh câu hỏi huấn luyện (Synthetic Data)

Do không có sẵn tập câu hỏi-đoạn văn (question-passage pairs), dự án sử dụng **LLM** (Large Language Model) để tự động sinh câu hỏi huấn luyện từ mỗi passage.

```bash
python generate_data/build_prompts.py \
  --passages data/passages.json \
  --output-questions generate_data/new_questions.json \
  --api-key YOUR_API_KEY \
  --model deepseek-v4-flash \
  --temperature 0.7
```

**Đặc điểm của bộ câu hỏi được sinh:**
- Đa dạng cách diễn đạt (câu hỏi điều kiện, hậu quả, thời hạn, trách nhiệm...)
- Phong cách tự nhiên, kiểu sinh viên hỏi nhanh
- Sử dụng từ đồng nghĩa và viết tắt tự nhiên
- Tối thiểu 20 câu hỏi khác nhau cho mỗi passage

Sau khi sinh, chia dữ liệu thành 3 tập:

```bash
python split_questions.py
# Kết quả: data/splits/train.json (90%), val.json (9.9%), test.json (0.1%)
```

### Bước 3 — Huấn luyện mô hình DPR

Mô hình sử dụng kiến trúc **Dual Encoder** dựa trên **PhoBERT** (`vinai/phobert-base`) với kỹ thuật **In-Batch Negatives**:

```bash
python train_dpr_inbatch.py
```

**Cấu hình mặc định:**

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `MODEL_NAME` | `vinai/phobert-base` | Mô hình ngôn ngữ tiếng Việt |
| `batch_size` | 64 | Kích thước batch |
| `epochs` | 10 | Số epoch huấn luyện |
| `lr` | 1e-4 | Tốc độ học |
| `max_q_len` | 64 | Độ dài tối đa câu hỏi (tokens) |
| `max_p_len` | 256 | Độ dài tối đa passage (tokens) |
| `warmup_ratio` | 0.1 | Tỷ lệ warmup cho learning rate scheduler |

**Cơ chế In-Batch Negatives:**
- Trong mỗi batch kích thước B, mỗi câu hỏi có 1 passage dương (positive) và (B-1) passage âm (negative) lấy từ các cặp khác trong cùng batch
- Loss function: Cross-Entropy trên ma trận similarity `(B × B)`
- Checkpoint tốt nhất (val_loss thấp nhất) được tự động lưu lại

### Bước 4 — Xây dựng chỉ mục FAISS

Sau khi huấn luyện xong, mã hoá toàn bộ passages thành vector embeddings và xây dựng chỉ mục FAISS:

```bash
# 1. Mã hoá passages
cd inferences
python encode_passages.py

# 2. Xây chỉ mục FAISS (Inner Product - Flat)
python build_faiss.py
```

**Output:**
- `data/passage_embeddings.npy` — Ma trận embeddings `(N × d)`
- `data/passage_id_map.json` — Ánh xạ từ chỉ số FAISS → passage_id
- `data/faiss_index_last.ivf` — Chỉ mục FAISS đã xây dựng

### Bước 5 — Truy vấn & Đánh giá

**Truy vấn tương tác:**

```bash
cd inferences
python retrieve.py
```

```
Query: nghỉ bao nhiêu buổi thì bị cấm thi?

Rank 1 - score=0.8234 - passage_id=p_0052
Điều 14 Khoản 1
Sinh viên bị ốm hoặc tai nạn trong thời gian học...

Rank 2 - score=0.7891 - passage_id=p_0053
Điều 14 Khoản 2
Sinh viên bị ốm hoặc tai nạn phải vắng thi cuối kỳ...
```

**Đánh giá định lượng:**

```bash
python evaluate_retrieval.py
```

Các chỉ số đánh giá:
- **Recall@K** (K = 1, 5, 10, 20) — Tỷ lệ câu hỏi tìm đúng passage trong top K
- **MRR** (Mean Reciprocal Rank) — Trung bình nghịch đảo thứ hạng

---

## 📊 Định dạng dữ liệu

### passages.json

```json
[
  {
    "passage_id": "p_0017",
    "clause_ref": "Điều 5 Khoản 1",
    "text": "Tín chỉ (TC) là đơn vị tính khối lượng học tập của sinh viên..."
  }
]
```

### questions.json (dữ liệu huấn luyện)

```json
[
  {
    "qid": "q_0001",
    "question": "Tín chỉ được tính như thế nào?",
    "positive_passage_id": "p_0017"
  }
]
```

---

## 📈 Kết quả & Đánh giá

| Chỉ số | Giá trị |
|--------|---------|
| Recall@1 | — |
| Recall@5 | — |
| Recall@10 | — |
| Recall@20 | — |
| MRR | — |

> 📝 *Chạy `python inferences/evaluate_retrieval.py` để điền kết quả cụ thể.*

---

## 🛠 Công nghệ sử dụng

| Công nghệ | Mục đích |
|------------|----------|
| **[PhoBERT](https://github.com/VinAIResearch/PhoBERT)** | Mô hình ngôn ngữ tiền huấn luyện cho tiếng Việt (VinAI) |
| **[PyTorch](https://pytorch.org/)** | Framework deep learning |
| **[HuggingFace Transformers](https://huggingface.co/docs/transformers)** | Tải và fine-tune mô hình ngôn ngữ |
| **[FAISS](https://github.com/facebookresearch/faiss)** | Thư viện tìm kiếm tương đồng vector tốc độ cao (Facebook AI) |
| **[OpenAI API](https://platform.openai.com/)** | Giao diện gọi LLM để sinh dữ liệu huấn luyện |

---

## 📚 Tham khảo

1. Karpukhin, V. et al. (2020). *"Dense Passage Retrieval for Open-Domain Question Answering"*. [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)
2. Nguyen, D.Q. & Nguyen, A.T. (2020). *"PhoBERT: Pre-trained language models for Vietnamese"*. [arXiv:2003.00744](https://arxiv.org/abs/2003.00744)
3. Johnson, J. et al. (2019). *"Billion-scale similarity search with GPUs"*. IEEE Transactions on Big Data.

---

## 👤 Tác giả

**Dự án môn Xử lý Ngôn ngữ Tự nhiên (NLP)**  
Trường Đại học Mỏ - Địa chất Hà Nội (HUMG)

---

<p align="center">
  <em>Được xây dựng với ❤️ bằng PhoBERT + FAISS</em>
</p>
