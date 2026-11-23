"""Chuyển các file model.json (mỗi file gộp mọi sách, dạng {"id", "zh"}) sang
định dạng ocr_raw/<sách>/<engine>_shard0.jsonl mà 1_voting_ocr.ipynb đọc được.

Ví dụ:
    python scripts/convert_model_json_to_ocr_raw.py data/hunyuan.json --engine hunyuan
    python scripts/convert_model_json_to_ocr_raw.py data/paddle_vl.json --engine paddle
    python scripts/convert_model_json_to_ocr_raw.py data/ppocr.json --engine ppocr
    python scripts/convert_model_json_to_ocr_raw.py data/tonggu.json --engine tonggu
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# ID sách trong model.json -> tên thư mục sách dùng chung với ocr_raw của các model khác.
BOOK_MAP = {
    "kinh": "HDTTK",   # Hoa Đà trung tàng kinh
    "bien": "ONDK",    # Ôn bệnh điều biện
    "1": "VSYC_1",
    "2": "VSYC_2",
    "3": "VSYC_3",
    "4": "VSYC_4",
    "5": "VSYC_5",
    "6": "VSYC_6",
}


def read_concat_json(path: Path):
    """Đọc file gồm nhiều object JSON nối tiếp (không có dấu phẩy, không phải array)."""
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    i = 0
    while True:
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            return
        obj, i = decoder.raw_decode(text, i)
        yield obj


def convert(src: Path, engine: str, raw_dir: Path, book_map: dict[str, str]) -> dict[str, int]:
    pages: dict[str, dict[int, str]] = {}
    for rec in read_concat_json(src):
        book_id, sep, page_id = rec["id"].rpartition("__p")
        if not sep or not page_id.isdigit():
            raise ValueError(f"{src}: id không đúng dạng <sách>__p<trang>: {rec['id']!r}")
        book = book_map.get(book_id, book_id)
        page = int(page_id)
        if page < 1:
            raise ValueError(f"{src}: số trang phải >= 1: {rec['id']!r}")
        book_pages = pages.setdefault(book, {})
        if page in book_pages:
            raise ValueError(f"{src}: trùng trang {rec['id']!r}")
        book_pages[page] = rec["zh"]

    written = {}
    for book, book_pages in sorted(pages.items()):
        missing = sorted(set(range(1, max(book_pages) + 1)) - set(book_pages))
        if missing:
            raise ValueError(f"{src}: sách {book} thiếu trang {missing[:10]}")
        out = raw_dir / book / f"{engine}_shard0.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for page in sorted(book_pages):
                # page trong ocr_raw đếm từ 0, giống các notebook OCR của nhóm.
                f.write(json.dumps({"page": page - 1, "text": book_pages[page]},
                                   ensure_ascii=False) + "\n")
        written[book] = len(book_pages)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="file model.json gộp mọi sách")
    ap.add_argument("--engine", required=True, help="tên model, cũng là tiền tố file shard")
    ap.add_argument("--raw-dir", type=Path, default=Path("data/ocr_raw"),
                    help="thư mục ocr_raw đích (mặc định: data/ocr_raw)")
    ap.add_argument("--copy-clc", action="store_true",
                    help="chép luôn data/results/ocr_raw/*/clc_shard*.jsonl vào ocr_raw đích")
    ap.add_argument("--book-map", type=Path,
                    help='file JSON ánh xạ tên sách, mặc định dùng BOOK_MAP ở đầu script')
    args = ap.parse_args()

    book_map = dict(BOOK_MAP)
    if args.book_map:
        book_map.update(json.loads(args.book_map.read_text(encoding="utf-8")))

    if args.copy_clc:
        clc_root = Path("data/results/ocr_raw")
        shards = sorted(clc_root.glob("*/clc_shard*.jsonl"))
        if not shards:
            raise SystemExit(f"Không thấy shard CLC trong {clc_root}")
        for shard in shards:
            dst = args.raw_dir / shard.parent.name / shard.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(shard, dst)
        print(f"Đã chép {len(shards)} shard CLC vào {args.raw_dir}")

    written = convert(args.src, args.engine, args.raw_dir, book_map)
    total = sum(written.values())
    print(f"{args.src} -> {args.engine}: {total} trang / {len(written)} sách")
    for book, n in written.items():
        print(f"  {book:<10} {n:>4} trang  -> {args.raw_dir / book / (args.engine + '_shard0.jsonl')}")


if __name__ == "__main__":
    main()
