"""Xuất OCR theo schema golden, không cần tải model hay thư viện GPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ocr_worker import page_lines


def page_record(book: str, page: int, text: str) -> dict:
    """Adapter cho model bất kỳ; page là số trang PDF bắt đầu từ 1."""
    if not isinstance(book, str) or not book or book.strip() != book:
        raise ValueError("book phải là tên sách không rỗng, không có khoảng trắng ở hai đầu")
    if type(page) is not int or page < 1:
        raise ValueError("page phải là số nguyên >= 1 (theo PDF)")
    if not isinstance(text, str):
        raise ValueError("text phải là chuỗi; không dùng None thay cho lỗi OCR")
    return {"id": f"{book}__p{page}", "zh": text}


def export_ocr(raw_dir: str | Path, name: str, reference: str | Path,
               output: str | Path, book_map: dict[str, str] | None = None,
               allow_missing: bool = False) -> dict:
    """Gộp shard của một engine; chỉ xuất các ID thuộc reference.

    book_map ánh xạ tên thư mục OCR sang đúng tên sách trong reference.
    Mặc định thiếu trang sẽ thất bại trước khi ghi output.
    """
    raw_dir, output = Path(raw_dir), Path(output)
    if output.resolve() == Path(reference).resolve():
        raise ValueError("Không được ghi đè dữ liệu tham chiếu")
    expected = []
    with open(reference, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if not isinstance(rec.get("id"), str) or not isinstance(rec.get("zh"), str):
                    raise ValueError("Reference cần các trường chuỗi id, zh")
                expected.append(rec["id"])
    if len(set(expected)) != len(expected):
        raise ValueError("Reference có ID trùng")
    records = {}
    files = sorted(raw_dir.glob(f"*/{name}_shard*.jsonl"))
    if not files:
        raise ValueError(f"Không tìm thấy shard cho {name} trong {raw_dir}")
    for path in files:
        book = (book_map or {}).get(path.parent.name, path.parent.name)
        with path.open(encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    page = raw["page"]
                    if type(page) is not int or page < 0:
                        raise ValueError("page trong shard phải là số nguyên >= 0")
                    rec = page_record(book, page + 1, "\n".join(page_lines(raw)))
                    if rec["id"] in records:
                        raise ValueError(f"ID trùng: {rec['id']}")
                    records[rec["id"]] = rec
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"{path}:{n}: {exc}") from exc
    wanted = set(expected)
    missing = [key for key in expected if key not in records]
    extra = sorted(set(records) - wanted)
    if missing and not allow_missing:
        raise ValueError(f"Thiếu {len(missing)} trang; kiểm tra tên sách/book-map hoặc OCR lại. "
                         f"Ví dụ: {missing[:5]}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for key in expected:
            if key in records:
                f.write(json.dumps(records[key], ensure_ascii=False) + "\n")
    report = {"written": len(wanted & records.keys()), "missing": missing, "extra": extra}
    output.with_suffix(output.suffix + ".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", required=True, help="thư mục ocr_raw của run")
    ap.add_argument("--name", required=True, help="tiền tố shard, thường hunyuan hoặc paddle")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--book-map", help='JSON ánh xạ {"tên thư mục OCR": "tên sách golden"}')
    ap.add_argument("--allow-missing", action="store_true", help="xuất phần đã có và báo ID thiếu")
    a = ap.parse_args(argv)
    mapping = json.loads(Path(a.book_map).read_text(encoding="utf-8")) if a.book_map else None
    try:
        report = export_ocr(a.raw_dir, a.name, a.reference, a.output, mapping, a.allow_missing)
    except ValueError as exc:
        ap.error(str(exc))
    print(f"Đã xuất {report['written']} trang; thiếu {len(report['missing'])}; "
          f"ngoài reference {len(report['extra'])}")


if __name__ == "__main__":
    main()
