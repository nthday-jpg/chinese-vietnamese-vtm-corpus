"""Hiển thị kết quả ngay trong notebook: bảng tổng hợp, text, layout, cặp ZH-VI."""
from __future__ import annotations

import json
from pathlib import Path

from .config import ReportConfig
from .pipeline import load_dich


def _rows_table(rows: list[dict], title: str = "") -> None:
    if not rows:
        print("(không có dữ liệu)")
        return
    try:
        import pandas as pd
        from IPython.display import Markdown, display
        if title:
            display(Markdown(f"### {title}"))
        display(pd.DataFrame(rows))
    except Exception:  # noqa: BLE001 - notebook không có pandas/IPython
        if title:
            print(f"### {title}")
        cols = list(rows[0])
        print(" | ".join(cols))
        for r in rows:
            print(" | ".join(str(r.get(c, "")) for c in cols))


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in open(path, encoding="utf-8"):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def _tasks(run_dir: Path) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for t in _read_jsonl(run_dir / "tasks.jsonl"):
        out.setdefault(t["book"], []).append(int(t["page"]))
    return {b: sorted(set(v)) for b, v in out.items()}


def ocr_summary(run_dir: str | Path) -> list[dict]:
    run_dir = Path(run_dir)
    expected = _tasks(run_dir)
    grouped: dict[tuple[str, str], dict[int, dict]] = {}
    for f in sorted((run_dir / "ocr_raw").glob("*/*.jsonl")):
        name = f.name.rsplit("_shard", 1)[0]
        if name.endswith("_layout"):
            continue
        book = f.parent.name
        recs = grouped.setdefault((book, name), {})
        for r in _read_jsonl(f):
            recs[int(r["page"])] = r
    rows = []
    for (book, name), recs in grouped.items():
        texts = [r.get("text", "") for r in recs.values()]
        rows.append({
            "book": book,
            "engine": name,
            "expected": len(expected.get(book, recs)),
            "done": len(recs),
            "missing": max(0, len(expected.get(book, recs)) - len(recs)),
            "empty": sum(not t.strip() for t in texts),
            "hit_max": sum(bool(r.get("hit_max")) for r in recs.values()),
            "chars": sum(len(t) for t in texts),
            "avg_chars": round(sum(len(t) for t in texts) / max(1, len(texts))),
            "avg_sec": round(sum(float(r.get("sec", 0)) for r in recs.values()) / max(1, len(recs)), 1),
        })
    return sorted(rows, key=lambda r: (r["book"], r["engine"]))


def translate_summary(run_dir: str | Path) -> list[dict]:
    run_dir = Path(run_dir)
    chunks = _read_jsonl(run_dir / "chunks.jsonl")
    done = load_dich(run_dir)
    per_book: dict[str, list[dict]] = {}
    for c in chunks:
        book = c["id"].split("|", 1)[0]
        per_book.setdefault(book, []).append(c)
    rows = []
    for book, cs in per_book.items():
        got = [done[c["id"]] for c in cs if c["id"] in done]
        rows.append({
            "book": book,
            "chunks": len(cs),
            "done": len(got),
            "missing": len(cs) - len(got),
            "empty": sum(not r.get("vi", "").strip() for r in got),
            "hit_max": sum(bool(r.get("hit_max")) for r in got),
            "chars_vi": sum(len(r.get("vi", "")) for r in got),
        })
    return sorted(rows, key=lambda r: r["book"])


def show_ocr_report(run_dir: str | Path, cfg: ReportConfig | None = None) -> None:
    cfg = cfg or ReportConfig()
    if cfg.show_tables:
        _rows_table(ocr_summary(run_dir), "OCR summary")


def show_translate_report(run_dir: str | Path, cfg: ReportConfig | None = None) -> None:
    cfg = cfg or ReportConfig()
    if cfg.show_tables:
        _rows_table(translate_summary(run_dir), "Translation summary")


def show_layouts(run_dir: str | Path, cfg: ReportConfig | None = None, book: str | None = None) -> None:
    cfg = cfg or ReportConfig()
    layout_dir = Path(run_dir) / "layout"
    files = sorted(layout_dir.glob("*.png"))
    if book:
        files = [f for f in files if f"_{book}_" in f.name]
    files = files[:cfg.layout_images]
    if not files:
        print("(không có ảnh layout)")
        return
    try:
        from IPython.display import Image, display
    except ImportError:
        for f in files:
            print(f)
        return
    for f in files:
        print(f.name)
        display(Image(filename=str(f), width=cfg.image_width))


def _raw_pages(run_dir: Path, book: str | None = None, engine: str | None = None) -> dict[tuple[str, int], dict]:
    out = {}
    for f in sorted((run_dir / "ocr_raw").glob("*/*.jsonl")):
        name = f.name.rsplit("_shard", 1)[0]
        if engine and name != engine:
            continue
        if not engine and name.endswith("_layout"):
            continue
        b = f.parent.name
        if book and b != book:
            continue
        for r in _read_jsonl(f):
            out[(b, int(r["page"]))] = r
    return out


def show_pages(run_dir: str | Path, cfg: ReportConfig | None = None, book: str | None = None,
               engine: str | None = None, pages: list[int] | None = None) -> None:
    cfg = cfg or ReportConfig()
    try:
        from IPython.display import Markdown, display
    except ImportError:
        Markdown = display = None
    recs = _raw_pages(Path(run_dir), book, engine)
    items = sorted(recs.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    if pages is not None:
        wanted = set(pages)
        items = [(k, v) for k, v in items if k[1] in wanted]
    if book is None:
        # mặc định chỉ lấy vài trang đầu của mỗi sách
        per: dict[str, int] = {}
        keep = []
        for k, v in items:
            if per.get(k[0], 0) < cfg.preview_pages:
                keep.append((k, v))
                per[k[0]] = per.get(k[0], 0) + 1
        items = keep
    for (b, p), r in items:
        text = r.get("text", "")[:cfg.preview_chars]
        title = f"{b} — trang {p + 1} ({r.get('sec', '?')}s)"
        if display is None:
            print(f"\n### {title}\n{text}")
        else:
            display(Markdown(f"### {title}\n\n```\n{text}\n```"))


def show_translation_pairs(run_dir: str | Path, cfg: ReportConfig | None = None,
                           n: int = 6) -> None:
    cfg = cfg or ReportConfig()
    try:
        from IPython.display import Markdown, display
    except ImportError:
        Markdown = display = None
    chunks = _read_jsonl(Path(run_dir) / "chunks.jsonl")
    done = load_dich(run_dir)
    shown = 0
    for c in chunks:
        r = done.get(c["id"])
        if not r:
            continue
        title = c["id"]
        zh, vi = c["zh"][:cfg.preview_chars], r.get("vi", "")[:cfg.preview_chars]
        if display is None:
            print(f"\n### {title}\nZH: {zh}\nVI: {vi}")
        else:
            display(Markdown(f"#### {title}\n\n**ZH:** {zh}\n\n**VI:** {vi}"))
        shown += 1
        if shown >= n:
            break
    if not shown:
        print("(chưa có bản dịch)")
