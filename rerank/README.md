# Chuẩn bị dữ liệu để huấn luyện reranker

Thư mục này chứa các script để chuẩn bị dữ liệu huấn luyện cho mô hình cross-encoder (reranker).

## Các script

- `build_rerank_dataset.py`: dùng DPR để sinh top‑K candidates và gán nhãn.
- `prepare_cross_encoder_dataset.py`: chuyển các file candidate thành định dạng phù hợp cho cross‑encoder.
- `preview_cross_encoder_dataset.py`: kiểm tra nhanh các file đầu ra.
- `train_cross_encoder.py`: huấn luyện cross‑encoder trên dữ liệu đã chuẩn bị.
- `infer_cross_encoder.py`: chạy inference và ghi điểm cho từng cặp question‑passage.
- `evaluate_cross_encoder.py`: đánh giá nhiều chỉ số phân loại và xếp hạng.
- `run_infer_eval_demo.py`: chạy nhanh inference + đánh giá (demo).

## Bắt đầu nhanh

```bash
python rerank/prepare_cross_encoder_dataset.py
python rerank/preview_cross_encoder_dataset.py rerank/cross_encoder/train.jsonl
python rerank/train_cross_encoder.py --data_dir rerank/cross_encoder --input_ext jsonl --out_dir rerank/ckpt
python rerank/infer_cross_encoder.py --input_path rerank/cross_encoder/test.jsonl --output_path rerank/cross_encoder/test.scored.jsonl --ckpt_dir rerank/ckpt/last --save_logit --save_prob
python rerank/evaluate_cross_encoder.py --input_path rerank/cross_encoder/test.scored.jsonl --score_key ce_logit --score_type logit
```

## Ghi chú

- Định dạng đầu ra mặc định là `jsonl` để dễ stream khi huấn luyện.
- Dùng `--max_negatives` để giới hạn số negative mỗi câu và giảm mất cân bằng.
