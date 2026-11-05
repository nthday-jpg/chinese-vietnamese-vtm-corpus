"""Resumable on-disk SwinIR page cache, shared by preprocessing and OCR."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile
import sys


@lru_cache(maxsize=128)
def _digest(path: str, size: int, modified: int):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _hash_stream(stream)


def _hash_stream(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _digest(str(path), stat.st_size, stat.st_mtime_ns)


def cached_page(src, page, dpi, max_side, cache):
    import fcntl
    from PIL import Image
    from .restoration import resolve_assets, restore_image
    from .ocr_worker import render

    source, weights = resolve_assets()
    settings = {"version": 1, "pdf_sha256": fingerprint(src), "dpi": dpi,
                "max_side": max_side, "architecture_sha256": fingerprint(source),
                "weights_sha256": fingerprint(weights), "tile": 128, "overlap": 32}
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    folder = Path(cache) / key
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"page_{page + 1:05d}.png"
    with open(folder / ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            try:
                with Image.open(target) as image:
                    image.load()
                    return image.convert("RGB")
            except (OSError, ValueError):
                pass  # interrupted/corrupt cache entry: regenerate this page
        image = render(src, page, dpi, max_side)
        image = restore_image(image)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=folder, suffix=".png", delete=False) as file:
                temporary = Path(file.name)
            image.save(temporary, format="PNG")
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        # Write alongside each completed page; the PNG itself is the resume marker.
        manifest = folder / "manifest.json"
        if not manifest.exists():
            with tempfile.NamedTemporaryFile(dir=folder, mode="w", delete=False) as file:
                json.dump({"source": str(Path(src).resolve()), **settings}, file, indent=2)
                name = file.name
            os.replace(name, manifest)
        return image


def run_preprocess(cfg):
    """Process selected PDF pages sequentially; model is loaded once per process."""
    from tqdm import tqdm
    from .pipeline import build_ocr_tasks, _demo_pages
    from .ocr_worker import _DOCS
    from .restoration import _load, resolve_assets

    books, tasks = build_ocr_tasks(cfg)
    if cfg.limit is not None:
        tasks = tasks[:cfg.limit]
    # Layout samples can lie outside the OCR limit. Prepare them too so a
    # Paddle-only worker never needs PyTorch to render those cached pages.
    if cfg.n_layout > 0 and cfg.engine != "dummy":
        seen = {(task["src"], task["page"]) for task in tasks}
        for task in _demo_pages(books, cfg.n_layout):
            if (task["src"], task["page"]) not in seen:
                tasks.append(task)
                seen.add((task["src"], task["page"]))
    resolve_assets()
    try:
        for task in tqdm(tasks, desc="SwinIR preprocessing", unit="page"):
            image = cached_page(task["src"], task["page"], cfg.dpi, cfg.max_side,
                                cfg.swinir_cache)
            image.close()
    finally:
        for doc in _DOCS.values():
            doc.close()
        _DOCS.clear()
        _load.cache_clear()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"Preprocessed {len(tasks)} pages (including cache hits): {cfg.swinir_cache}")
    return Path(cfg.swinir_cache)
