#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chuyển tất cả file .docx trong thư mục input thành .txt trong thư mục output.

Tính năng:
- Duyệt đệ quy thư mục input và giữ cấu trúc thư mục trong output
- Ghi file .txt với encoding UTF-8 (đọc được tiếng Việt)
- Bỏ qua file đã tồn tại trừ khi dùng --overwrite

Sử dụng:
	python docx2txt.py --input INPUT_FOLDER --output OUTPUT_FOLDER

Yêu cầu:
	pip install python-docx
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import logging

from docx import Document


def docx_to_text(path: Path) -> str:
	doc = Document(path)
	parts: list[str] = []

	# Lấy tất cả paragraph
	for p in doc.paragraphs:
		text = p.text
		if text:
			parts.append(text)
		else:
			# Giữ nguyên khoảng xuống dòng nếu paragraph rỗng
			parts.append("")

	# Thêm text từ các bảng (nếu có)
	for table in doc.tables:
		for row in table.rows:
			row_text = "\t".join(cell.text for cell in row.cells)
			parts.append(row_text)

	# Ghép các đoạn với một dòng trống giữa các paragraph
	return "\n\n".join(parts)


def convert_folder(input_dir: Path, output_dir: Path, overwrite: bool = False) -> int:
	input_dir = input_dir.resolve()
	output_dir = output_dir.resolve()
	count = 0

	for root, _dirs, files in os.walk(input_dir):
		rel_root = os.path.relpath(root, input_dir)
		target_root = output_dir.joinpath(rel_root) if rel_root != os.curdir else output_dir
		os.makedirs(target_root, exist_ok=True)

		for fname in files:
			if not fname.lower().endswith(".docx"):
				continue

			src_path = Path(root) / fname
			out_name = Path(fname).stem + ".txt"
			out_path = target_root / out_name

			if out_path.exists() and not overwrite:
				logging.info("Skip existing: %s", out_path)
				continue

			try:
				text = docx_to_text(src_path)
				with open(out_path, "w", encoding="utf-8") as f:
					f.write(text)
				logging.info("Wrote: %s", out_path)
				count += 1
			except Exception as e:
				logging.error("Failed to convert %s: %s", src_path, e)

	return count


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Convert DOCX folder to TXT (UTF-8) — supports Vietnamese")
	parser.add_argument("--input", "-i", required=True, help="Input folder containing .docx files")
	parser.add_argument("--output", "-o", required=True, help="Output folder for .txt files")
	parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt files in output")
	parser.add_argument("--quiet", action="store_true", help="Less verbose logging")

	args = parser.parse_args(argv)

	logging.basicConfig(level=(logging.WARNING if args.quiet else logging.INFO), format="%(levelname)s: %(message)s")

	input_dir = Path(args.input)
	output_dir = Path(args.output)

	if not input_dir.is_dir():
		logging.error("Input folder does not exist: %s", input_dir)
		return 2

	output_dir.mkdir(parents=True, exist_ok=True)

	converted = convert_folder(input_dir, output_dir, overwrite=args.overwrite)
	logging.info("Done — converted %d files.", converted)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
