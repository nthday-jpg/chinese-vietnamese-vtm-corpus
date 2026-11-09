"""Recompare downloaded raw OCR books without loading or installing a model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .corpus_ocr import write_json
from .ocr_records import read_json, read_jsonl
from .pipeline import parse_pages


def import_ocr_logs(log_dir, work_dir, *, input_dir=None, books=None, max_books=1, pages="", limit=None):
    """Build a separate checkpoint copy; never rewrite the supplied logs/PDFs."""
    source_root = Path(log_dir).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Không có thư mục log: {source_root}")
    paths = ([source_root / "manifest.json"] if (source_root / "raw").is_dir()
             else sorted(source_root.glob("*/manifest.json")))
    selected = []
    wanted = {Path(str(book)).stem for book in books or []}
    for path in paths:
        manifest = read_json(path)
        if manifest.get("format") != "vtm-ocr-book-v1":
            continue
        book = str(manifest["book"])
        if book in {".", ".."} or "/" in book or "\\" in book:
            raise ValueError("Invalid book name in exported manifest")
        if not wanted or book in wanted or path.parent.name in wanted:
            selected.append((path, manifest))
    if max_books is not None:
        if max_books < 1:
            raise ValueError("max_books must be positive or None")
        selected = selected[:max_books]
    if not selected:
        raise FileNotFoundError(f"Không có log sách đã chọn trong {source_root}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive or None")
    root = Path(work_dir).resolve() / "imported_logs"
    if root == source_root or root.is_relative_to(source_root) or source_root.is_relative_to(root):
        raise ValueError("Imported checkpoint must be separate from source logs")
    pairs, models, total, seen = [], None, 0, set()
    for path, manifest in selected:
        book = str(manifest["book"])
        if book in seen:
            raise ValueError(f"Duplicate exported book: {book}")
        seen.add(book)
        names = list(manifest["models"])
        if models is not None and set(names) != set(models):
            raise ValueError("Exported books must contain the same models")
        models = names
        exported_pages = manifest["page_numbers"]
        allowed = set(parse_pages(pages, max(exported_pages)))
        chosen = [number - 1 for number in exported_pages if number - 1 in allowed]
        if limit is not None:
            chosen = chosen[:max(0, limit - total)]
        if not chosen:
            continue
        pair = {"book": book}
        book_key = hashlib.sha256(book.encode()).hexdigest()[:16]
        for name in names:
            if name not in {"ppocr", "tonggu", "paddle", "hunyuan"}:
                raise ValueError(f"Unknown OCR model: {name}")
            info = manifest["models"][name]
            original = dict(info.get("source") or {})
            candidate = Path(input_dir) / f"{book}.pdf" if input_dir else None
            if candidate and candidate.is_file():
                actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
                if original.get("sha256") and original["sha256"] != actual:
                    raise ValueError(f"Different source PDF for exported book {book}")
                # No original digest means source identity remains unverified.
                if original.get("sha256"):
                    original["src"] = str(candidate.resolve())
            original["pages"] = chosen
            cfg = {**info.get("config", {}), "engine": name, "run_name": name}
            directory = root / book_key / name
            raw_dir = directory / "ocr_raw" / book
            raw_dir.mkdir(parents=True, exist_ok=True)
            records, record_pages = [], set()
            for exported in read_jsonl(path.parent / "raw" / f"{name}.jsonl"):
                page = int(exported["page"]) - 1
                if page not in chosen or exported.get("raw") is None:
                    continue
                record = exported["raw"]
                if record.get("page") != page or page in record_pages:
                    raise ValueError(f"Invalid/duplicate exported page: {book}/{name}/{page + 1}")
                records.append(record)
                record_pages.add(page)
            (raw_dir / f"{name}_shard0.jsonl").write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
            (directory / "tasks.jsonl").write_text("".join(
                json.dumps({"book": book, "page": page, "src": original.get("src")}) + "\n" for page in chosen), encoding="utf-8")
            write_json(directory / "run_manifest.json", {"kind": "ocr", "status": "complete", "config": cfg,
                "books": {book: original}, "imported_from": str(path.parent)})
            pair[name] = str(directory)
        pairs.append(pair)
        total += len(chosen)
    if not pairs:
        raise ValueError("Không có trang log nào khớp PAGES/LIMIT")
    output = root / "manifest.json"
    write_json(output, {"models": models, "pairs": pairs, "total_books": len(pairs), "total_pages": total,
                        "imported_from": str(source_root)})
    return output
