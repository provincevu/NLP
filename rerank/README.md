# Chuẩn bị dữ liệu để huấn luyện reranker

Thư mục này chứa các script để chuẩn bị dữ liệu huấn luyện cho mô hình cross-encoder (reranker).

## Các script

- `build_rerank_dataset.py`: dùng DPR để sinh top‑K candidates và gán nhãn.
- `prepare_cross_encoder_dataset.py`: chuyển các file candidate thành định dạng phù hợp cho cross‑encoder.
- `preview_cross_encoder_dataset.py`: kiểm tra nhanh các file đầu ra.

## Bắt đầu nhanh

```bash
python rerank/prepare_cross_encoder_dataset.py
python rerank/preview_cross_encoder_dataset.py rerank/cross_encoder/train.jsonl
```

## Ghi chú

- Định dạng đầu ra mặc định là `jsonl` để dễ stream khi huấn luyện.
- Dùng `--max_negatives` để giới hạn số negative mỗi câu và giảm mất cân bằng.
