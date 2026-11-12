"""Read selected OCR checkpoints and verify source identity for comparison."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path

from .ocr_worker import page_lines
from .reading_order import order_ppocr_columns
from .ocr_quality import ocr_quality_issues


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def checkpoint_reference(anchor, reference):
    """Resolve a copied campaign tree without changing its stored provenance."""
    anchor = Path(anchor).resolve().parent
    normalized = str(reference).replace('\\', '/')
    path = Path(normalized)
    if not path.is_absolute() and ':' not in normalized:
        candidate = (anchor/path).resolve()
        if not candidate.is_relative_to(anchor):
            raise ValueError('Relative checkpoint reference escapes campaign root')
        return candidate
    # Prefer the mounted copy even when the previous session still exists.
    parts = normalized.split('/')
    suffixes = [parts[i:] for i, value in enumerate(parts) if value in {'variant_runs', 'runs'}]
    suffixes.append(parts[-1:])
    for suffix in suffixes:
        local = anchor.joinpath(*suffix)
        if local.exists():
            return local
    return path


def read_checkpoint_campaign(path):
    path = Path(path)
    campaign = read_json(path)
    if campaign.get('variant_manifests'):
        campaign['variant_manifests'] = {v: str(checkpoint_reference(path, ref))
            for v, ref in campaign['variant_manifests'].items()}
    for pair in campaign.get('pairs', []):
        for engine in campaign['models']:
            pair[engine] = str(checkpoint_reference(path, pair[engine]))
    return campaign


def read_jsonl(path: Path, *, allow_incomplete_tail: bool = False) -> list[dict]:
    rows = []
    lines = path.read_text(encoding="utf-8-sig").splitlines(keepends=True)
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if allow_incomplete_tail and number == len(lines) and not line.endswith("\n"):
                break  # A running worker has not finished writing this record yet.
            raise ValueError(f"Invalid JSON: {path}:{number}: {exc.msg}") from exc
    return rows


def count_text(text: str) -> tuple[int, int]:
    """Return (Han ideographs, non-whitespace characters), without normalizing glyphs."""
    visible = sum(not c.isspace() for c in text)
    han = sum(
        ord(c) >= 0x3400 and unicodedata.name(c, "").startswith(
            ("CJK UNIFIED IDEOGRAPH-", "CJK COMPATIBILITY IDEOGRAPH-")
        )
        for c in text
    )
    return han, visible


def prepare_ocr_record(record, engine, *, reading_order="auto"):
    """Normalize a comparison copy, keeping the exact checkpoint in raw."""
    # Region partitions already establish page order, including mixed row/column
    # splits. A later global column sort could interleave independently read bands.
    geometry, order = (order_ppocr_columns(record, mode=reading_order) if engine == "ppocr" and not record.get("region_order_applied")
                       else (record, {"mode": "model", "changed": False}))
    text = geometry["text"] if engine in {"ppocr", "tonggu"} else "\n".join(page_lines(record))
    han, visible = count_text(text)
    return {"han": han, "visible": visible, "text": text, "raw": record,
            "geometry": geometry, "reading_order": order,
            "quality_issues": ocr_quality_issues(text, record),
            "hit_max": record.get("hit_max"), "sec": record.get("sec")}


def load_run(directory: Path, engine: str, warnings: list[str], *, reading_order="auto") -> dict:
    manifest = read_json(directory / "run_manifest.json")
    cfg = manifest.get("config", {})
    if manifest.get("kind") != "ocr" or cfg.get("engine") != engine:
        raise ValueError(f"Expected an OCR {engine} run: {directory}")
    name = cfg.get("run_name") or engine
    records = {}
    for path in sorted((directory / "ocr_raw").glob("*/*.jsonl")):
        # Exact worker name: never count the independent Hunyuan layout pass.
        if not path.name.startswith(f"{name}_shard"):
            continue
        for record in read_jsonl(path, allow_incomplete_tail=manifest.get("status") == "running"):
            key = (path.parent.name, int(record["page"]))
            if key[1] < 0:
                raise ValueError(f"Negative page in {path}")
            if key in records:
                warnings.append(f"{engine}: duplicate {key}; last record in sorted file order retained.")
            records[key] = prepare_ocr_record(record, engine, reading_order=reading_order)
    tasks = directory / "tasks.jsonl"
    if tasks.exists():
        selected = {(str(t["book"]), int(t["page"])) for t in read_jsonl(tasks)}
        extras = records.keys() - selected
        if extras:
            warnings.append(f"{engine}: {len(extras)} cached pages outside tasks.jsonl excluded.")
    else:
        selected = set(records)
        warnings.append(f"{engine}: tasks.jsonl absent; cannot detect pages with no output in either model.")
    if not records:
        warnings.append(f"{engine}: no OCR page records found.")
    if manifest.get("resumed_files") or cfg.get("resume_from"):
        warnings.append(f"{engine}: resumed outputs; manifest config may not describe every copied record.")
    return {"manifest": manifest, "selected": selected, "records": records}


def check_sources(left: dict, right: dict, warnings: list[str], *, digest_cache=None) -> list[dict]:
    """Hash available PDFs so equal book names cannot silently pair different files."""
    books_h = left["manifest"].get("books", {})
    books_p = right["manifest"].get("books", {})
    cache = digest_cache if digest_cache is not None else {}

    def digest(source: str | None) -> str | None:
        if not source:
            return None
        path = Path(source)
        if not path.is_file():
            return None
        key = str(path.resolve())
        if key not in cache:
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
            cache[key] = hasher.hexdigest()
        return cache[key]

    checks = []
    for book in sorted({b for b, _ in left["selected"]} & {b for b, _ in right["selected"]}):
        h = digest(books_h.get(book, {}).get("src"))
        p = digest(books_p.get(book, {}).get("src"))
        if h and p and h != p:
            raise ValueError(f"Different source PDFs for book {book!r}; do not compare these runs.")
        verified = bool(h and p)
        for info, actual in ((books_h.get(book, {}), h), (books_p.get(book, {}), p)):
            if actual and info.get('sha256') and info['sha256'] != actual:
                raise ValueError(f"Source PDF changed since OCR for book {book!r}")
            if info.get('legacy_source_unverified'):
                warnings.append(f'{book}: legacy cached pages have no original source hash; check source provenance.')
        checks.append({"book": book, "source_verified": verified, "sha256": h if verified else None})
        if not verified:
            warnings.append(f"{book}: original PDFs unavailable; source identity unverified (matching book/page only).")
    return checks
