"""CLI: vtm ocr | translate | report | info.

Mọi trường cấu hình có thể đặt qua file JSON (--config) và/hoặc --set key=value.
Ví dụ:
    vtm ocr --config configs/ocr_hunyuan.json --set limit=2
    vtm translate --config configs/dich.json --set device=cpu
    vtm report --run work/runs/ocr_hunyuan_abc1234_deadbeef
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from .config import OCRConfig, TranslateConfig
from .pipeline import find_run, list_pdfs, list_texts, run_ocr, run_translate
from .report import show_layouts, show_ocr_report, show_pages, show_translate_report


def _apply(cls, config_path: str | None, sets: list[str]):
    data = {}
    if config_path:
        data.update(json.loads(Path(config_path).read_text(encoding="utf-8")))
    for item in sets or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set phải có dạng key=value: {item}")
        try:
            data[key] = json.loads(value)
        except json.JSONDecodeError:
            data[key] = value
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise SystemExit(f"khoá cấu hình không hợp lệ cho {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="file JSON cấu hình")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override một trường, có thể lặp lại")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="vtm", description="Vietnamese Traditional Medicine corpus pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ocr", help="OCR PDF -> txt chữ Hán")
    _common(p)

    p = sub.add_parser("translate", help="dịch txt chữ Hán -> txt tiếng Việt")
    _common(p)

    p = sub.add_parser("report", help="hiển thị/tổng hợp kết quả của một run")
    p.add_argument("--run", help="đường dẫn run dir")
    p.add_argument("--workdir", default="work")
    p.add_argument("--kind", choices=["ocr", "dich"], default="ocr")
    p.add_argument("--name", help="lọc theo tên run")
    p.add_argument("--book", help="chỉ xem một sách")
    p.add_argument("--pages", help="vd 1-5,10 (chỉ cho OCR)")
    p.add_argument("--layouts", action="store_true", help="hiển thị ảnh layout")

    p = sub.add_parser("list", help="liệt kê PDF hoặc txt đầu vào")
    p.add_argument("--input-dir", default="data")
    p.add_argument("--kind", choices=["pdf", "txt"], default="pdf")
    p.add_argument("--min-han", type=int, default=50)

    p = sub.add_parser("info", help="in các run hiện có")
    p.add_argument("--workdir", default="work")

    a = ap.parse_args(argv)
    if a.cmd == "ocr":
        cfg = _apply(OCRConfig, a.config, a.set)
        try:
            res = run_ocr(cfg)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            raise SystemExit(str(exc)) from exc
        print(f"\nXong. Kết quả: {res.output_dir}\nManifest: {res.run_dir / 'run_manifest.json'}")
    elif a.cmd == "translate":
        cfg = _apply(TranslateConfig, a.config, a.set)
        try:
            res = run_translate(cfg)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            raise SystemExit(str(exc)) from exc
        print(f"\nXong. Kết quả: {res.output_dir}\nManifest: {res.run_dir / 'run_manifest.json'}")
    elif a.cmd == "list":
        if a.kind == "pdf":
            entries = list_pdfs(a.input_dir)
            print(f"{len(entries)} PDF trong {a.input_dir}:")
            for i, e in enumerate(entries, 1):
                print(f"{i:3}. {e['name']:<50} {e['n_pages']:>5} trang  {e['path']}")
        else:
            entries = list_texts(a.input_dir, a.min_han)
            print(f"{len(entries)} txt (>= {a.min_han} chữ Hán) trong {a.input_dir}:")
            for i, e in enumerate(entries, 1):
                print(f"{i:3}. {e['name']:<50} {e['han']:>7} chữ Hán  {e['path']}")
        print('\nChọn tập con: books=["tên hoặc glob", ...]  ([] = tất cả)')
    elif a.cmd == "report":
        run = Path(a.run) if a.run else find_run(a.workdir, a.kind, a.name)
        print(f"Run: {run}")
        if a.kind == "ocr":
            show_ocr_report(run)
            if a.pages:
                from .pipeline import parse_pages
                pages = parse_pages(a.pages, 10_000)
                show_pages(run, book=a.book, pages=pages)
            if a.layouts:
                show_layouts(run, book=a.book)
        else:
            show_translate_report(run)
    else:
        root = Path(a.workdir) / "runs"
        for d in sorted(root.glob("*")) if root.exists() else []:
            print(d)


if __name__ == "__main__":
    main()
