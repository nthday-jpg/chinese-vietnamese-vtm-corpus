"""Compare OCR output lengths by page; count agreement is not OCR accuracy."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from .ocr_worker import page_lines


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _jsonl(path: Path, *, allow_incomplete_tail: bool = False) -> list[dict]:
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


def discover_runs(roots: list[str | Path]) -> list[dict]:
    """Find manifest-backed OCR runs, including outputs mounted under Kaggle input."""
    runs = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        paths = [root / "run_manifest.json"] if (root / "run_manifest.json").exists() else root.rglob("run_manifest.json")
        for path in paths:
            manifest = _json(path)
            cfg = manifest.get("config", {})
            if manifest.get("kind") != "ocr" or cfg.get("engine") not in {"hunyuan", "paddle"}:
                continue
            directory = str(path.parent.resolve())
            runs[directory] = {
                "engine": cfg["engine"], "run_dir": directory,
                "updated": manifest.get("updated", manifest.get("created", "")),
                "books": ", ".join(sorted(manifest.get("books", {}))),
                "selected": manifest.get("tasks_selected"),
                "done": manifest.get("counts", {}).get("done"),
                "dpi": cfg.get("dpi"), "max_side": cfg.get("max_side"),
            }
    return sorted(runs.values(), key=lambda r: (r["updated"], r["run_dir"]), reverse=True)


def choose_run(runs: list[dict], engine: str, explicit: str | Path | None = None) -> Path:
    if explicit:
        path = Path(explicit).resolve()
        manifest = _json(path / "run_manifest.json")
        if manifest.get("kind") != "ocr" or manifest.get("config", {}).get("engine") != engine:
            raise ValueError(f"Expected an OCR {engine} run: {path}")
        return path
    choices = [r for r in runs if r["engine"] == engine]
    if not choices:
        raise ValueError(f"No {engine} run found. Run its OCR notebook first, or set its run path.")
    return Path(choices[0]["run_dir"])


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


def _load_run(directory: Path, engine: str, warnings: list[str]) -> dict:
    manifest = _json(directory / "run_manifest.json")
    cfg = manifest.get("config", {})
    if manifest.get("kind") != "ocr" or cfg.get("engine") != engine:
        raise ValueError(f"Expected an OCR {engine} run: {directory}")
    name = cfg.get("run_name") or engine
    records = {}
    for path in sorted((directory / "ocr_raw").glob("*/*.jsonl")):
        # Exact worker name: never count the independent Hunyuan layout pass.
        if not path.name.startswith(f"{name}_shard"):
            continue
        for record in _jsonl(path, allow_incomplete_tail=manifest.get("status") == "running"):
            key = (path.parent.name, int(record["page"]))
            if key[1] < 0:
                raise ValueError(f"Negative page in {path}")
            if key in records:
                warnings.append(f"{engine}: duplicate {key}; last record in sorted file order retained.")
            text = "\n".join(page_lines(record))
            han, visible = count_text(text)
            records[key] = {
                "han": han, "visible": visible, "text": text,
                "hit_max": record.get("hit_max"), "sec": record.get("sec"),
            }
    tasks = directory / "tasks.jsonl"
    if tasks.exists():
        selected = {(str(t["book"]), int(t["page"])) for t in _jsonl(tasks)}
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


def _source_checks(left: dict, right: dict, warnings: list[str]) -> list[dict]:
    """Hash available PDFs so equal book names cannot silently pair different files."""
    books_h = left["manifest"].get("books", {})
    books_p = right["manifest"].get("books", {})
    cache = {}

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
        checks.append({"book": book, "source_verified": verified, "sha256": h if verified else None})
        if not verified:
            warnings.append(f"{book}: original PDFs unavailable; source identity unverified (matching book/page only).")
    return checks


def compare_runs(hunyuan_run: str | Path, paddle_run: str | Path, *,
                 mode: str = "han", gap_threshold: float = 0.10) -> dict:
    """Outer join selected pages; missing results stay None, never become zero."""
    if mode not in {"han", "visible"} or not 0 <= gap_threshold <= 1:
        raise ValueError("mode must be han/visible; gap_threshold must be between 0 and 1")
    warnings = []
    left = _load_run(Path(hunyuan_run), "hunyuan", warnings)
    right = _load_run(Path(paddle_run), "paddle", warnings)
    checks = _source_checks(left, right, warnings)
    rows = []
    for key in sorted(left["selected"] | right["selected"]):
        selected_h, selected_p = key in left["selected"], key in right["selected"]
        h = left["records"].get(key) if selected_h else None
        p = right["records"].get(key) if selected_p else None
        paired = h is not None and p is not None
        n_h, n_p = (h[mode] if h else None), (p[mode] if p else None)
        maximum = max(n_h, n_p) if paired else None
        agreement = min(n_h, n_p) / maximum if maximum else None
        status = (
            "not_selected_hunyuan" if not selected_h else
            "not_selected_paddle" if not selected_p else
            "missing_both" if h is None and p is None else
            "missing_hunyuan" if h is None else "missing_paddle" if p is None else
            "both_zero" if maximum == 0 else "one_zero" if min(n_h, n_p) == 0 else
            "large_gap" if 1 - agreement > gap_threshold + 1e-12 else "within_threshold"
        )
        rows.append({
            "book": key[0], "page": key[1] + 1,
            "selected_hunyuan": selected_h, "selected_paddle": selected_p,
            "has_hunyuan": h is not None, "has_paddle": p is not None,
            "han_hunyuan": h["han"] if h else None, "han_paddle": p["han"] if p else None,
            "visible_hunyuan": h["visible"] if h else None, "visible_paddle": p["visible"] if p else None,
            "count_hunyuan": n_h, "count_paddle": n_p,
            "delta_h_minus_p": n_h - n_p if paired else None,
            "abs_gap": abs(n_h - n_p) if paired else None,
            "agreement_pct": 100 * agreement if agreement is not None else None,
            "gap_pct": 100 * (1 - agreement) if agreement is not None else None,
            "status": status,
            "hit_max_hunyuan": h["hit_max"] if h else None,
            "hit_max_paddle": p["hit_max"] if p else None,
            "sec_hunyuan": h["sec"] if h else None, "sec_paddle": p["sec"] if p else None,
        })
    if not any(r["has_hunyuan"] and r["has_paddle"] for r in rows):
        warnings.append("No paired OCR pages: agreement cannot be computed.")
    return {
        "rows": rows, "warnings": warnings, "source_checks": checks,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "book_states": {book: {"hunyuan": left["manifest"].get("status", "unknown"),
                               "paddle": right["manifest"].get("status", "unknown")}
                        for book in sorted({b for b, _ in left["selected"] | right["selected"]})},
        "mode": mode, "gap_threshold": gap_threshold,
        "hunyuan_run": str(Path(hunyuan_run).resolve()), "paddle_run": str(Path(paddle_run).resolve()),
        "hunyuan_manifest": left["manifest"], "paddle_manifest": right["manifest"],
    }


def compare_corpus(manifest_path: str | Path, *, mode: str = "han",
                   gap_threshold: float = 0.10) -> dict:
    """Combine explicit per-book run pairs while preserving each book's config."""
    campaign = _json(Path(manifest_path))
    comparisons = [compare_runs(pair["hunyuan"], pair["paddle"], mode=mode,
                               gap_threshold=gap_threshold) for pair in campaign["pairs"]]
    if not comparisons:
        raise ValueError("Corpus manifest has no run pairs")
    rows = [row for result in comparisons for row in result["rows"]]
    keys = [(row["book"], row["page"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Corpus pairs contain duplicate book/page keys")
    return {
        "rows": sorted(rows, key=lambda row: (row["book"], row["page"])),
        "warnings": sorted({warning for r in comparisons for warning in r["warnings"]}),
        "source_checks": [check for r in comparisons for check in r["source_checks"]],
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "book_states": {book: states for r in comparisons for book, states in r["book_states"].items()},
        "mode": mode, "gap_threshold": gap_threshold,
        "hunyuan_run": f"{len(comparisons)} runs; {manifest_path}",
        "paddle_run": f"{len(comparisons)} runs; {manifest_path}",
        "hunyuan_manifest": {"config": {"engine": "hunyuan", "per_book_configs": {
            pair["book"]: r["hunyuan_manifest"]["config"]
            for pair, r in zip(campaign["pairs"], comparisons)}}},
        "paddle_manifest": {"config": {"engine": "paddle", "per_book_configs": {
            pair["book"]: r["paddle_manifest"]["config"]
            for pair, r in zip(campaign["pairs"], comparisons)}}},
        "corpus_manifest": str(Path(manifest_path).resolve()),
    }
