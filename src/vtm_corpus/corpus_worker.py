"""A single model instance OCRs all selected books in one isolated process."""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
import traceback
from types import SimpleNamespace
import uuid

from .config import HUNYUAN_PROMPT, OCRConfig
from . import ocr_worker
from .corpus_ocr import now, write_json
from .pipeline import export_ocr
from .worker_paths import runtime_path
from .image_preparation import run_prepared_page


def create_engine(engine: str, cfg: OCRConfig):
    args = SimpleNamespace(model=cfg.model, dtype=cfg.dtype, device=cfg.device,
                           model_revision=cfg.model_revision, tonggu_repo=cfg.tonggu_repo,
                           tonggu_config=cfg.tonggu_config,
                           max_new=cfg.max_new_tokens, rep_penalty=cfg.repetition_penalty,
                           prompt=cfg.prompt, paddle_version=cfg.paddle_version,
                           model_min_pixels=cfg.model_min_pixels, model_max_pixels=cfg.model_max_pixels,
                           paddle_args=json.dumps(cfg.paddle_args))
    return ocr_worker.ENGINES[engine](args)


def append_record(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve a partial last write from an interrupted worker before retrying it.
    if path.exists() and path.stat().st_size:
        with path.open("r+b") as stream:
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b"\n":
                stream.seek(0)
                contents = stream.read()
                boundary = contents.rfind(b"\n") + 1
                tail = contents[boundary:]
                try:
                    json.loads(tail)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    path.with_suffix(f".interrupted-{time.time_ns()}.log").write_bytes(tail)
                    stream.truncate(boundary)
                else:
                    stream.seek(0, os.SEEK_END)
                    stream.write(b"\n")
    with path.open("ab") as stream:
        stream.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def corpus_book_plan(pair, engine, attempt):
    directory = runtime_path(pair[engine])
    manifest_file = directory / 'run_manifest.json'
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    cfg = OCRConfig(**manifest['config'])
    name = cfg.run_name or engine
    raw = directory / 'ocr_raw'
    raw_book = raw / pair['book']
    raw_book.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(line) for line in (directory / 'tasks.jsonl').read_text(encoding='utf-8').splitlines() if line]
    done = ocr_worker.load_results(raw, pair['book'], name)
    failed = set()
    errors = raw_book / f'{name}_errors.jsonl'
    if errors.exists():
        for line in errors.read_text(encoding='utf-8').splitlines():
            try:
                error = json.loads(line)
                if error.get('attempt') == attempt:
                    failed.add(int(error['page']))
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
    todo = [task for task in tasks if task['page'] not in done and task['page'] not in failed]
    return dict(pair=pair, directory=directory, manifest_file=manifest_file, manifest=manifest,
                cfg=cfg, name=name, raw=raw, errors=errors, tasks=tasks, done=done, todo=todo)


def corpus_page_image(campaign, manifest, book, task, cfg, render, temporary):
    """Share lossless renders across models; a damaged cache is rendered again."""
    from PIL import Image
    source_sha = manifest.get('books', {}).get(book, {}).get('sha256')
    key = {'source_sha256': source_sha, 'page': task['page'], 'dpi': cfg.dpi, 'max_side': cfg.max_side,
           'renderer': 'pdfium-rgb-lanczos-v2'}
    cached = bool(campaign.get('cache_renders') and source_sha)
    if cached:
        cache_root = runtime_path(campaign['render_cache']).resolve()
        if not cache_root.is_relative_to(runtime_path(cfg.workdir).resolve()):
            raise ValueError('Render cache must stay within the campaign workdir')
        path = cache_root / source_sha / f'{cfg.dpi}_{cfg.max_side}' / f"p{task['page'] + 1:05d}.png"
        metadata = path.with_suffix('.json')
        try:
            if path.is_symlink() or metadata.is_symlink():
                raise ValueError('Cache symlink')
            info = json.loads(metadata.read_text(encoding='utf-8'))
            data = path.read_bytes()
            if info['key'] != key or hashlib.sha256(data).hexdigest() != info['sha256']:
                raise ValueError('Render cache mismatch')
            with Image.open(io.BytesIO(data)) as loaded:
                image = loaded.convert('RGB')
            if list(image.size) != info['size']:
                raise ValueError('Render cache size mismatch')
            image.info['render'] = info.get('render', {})
            return image, path, True
        except (OSError, ValueError, KeyError):
            pass
    else:
        path = temporary
    image = render(str(runtime_path(task['src'])), task['page'], cfg.dpi, cfg.max_side)
    path.parent.mkdir(parents=True, exist_ok=True)
    destination = path.with_suffix('.png.tmp')
    image.save(destination, format='PNG', compress_level=1)
    destination.replace(path)
    if cached:
        write_json(metadata, {'key': key, 'size': list(image.size), 'render': image.info.get('render', {}),
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return image, path, False


def process_corpus(manifest_path: str | Path, engine: str, *, engine_factory=None, renderer=None, attempt_id=None):
    factory = engine_factory or create_engine
    render = renderer or ocr_worker.render
    campaign = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    # One attempt spans supervisor restarts, but a new notebook call retries
    # missing pages even if an older attempt recorded an error or skip marker.
    attempt = attempt_id or os.environ.get("VTM_OCR_ATTEMPT") or uuid.uuid4().hex
    plans = [corpus_book_plan(pair, engine, attempt) for pair in campaign['pairs']]
    completed = sum(len({task['page'] for task in plan['tasks']} & plan['done'].keys()) for plan in plans)
    progress_file = runtime_path(Path(manifest_path).parent) / f'{engine}_progress.json'
    def update_progress():
        write_json(progress_file, {'engine': engine, 'attempt': attempt, 'pages': campaign['total_pages'],
            'done': completed, 'missing': campaign['total_pages'] - completed, 'updated': now()})
    update_progress()
    backend = None
    try:
        with tempfile.TemporaryDirectory(prefix="vtm-corpus-") as temporary:
            image_path = Path(temporary) / "page.png"
            for plan in plans:
                pair, directory, manifest_file, manifest, cfg, name, raw, errors, tasks, done, todo = (
                    plan[key] for key in ('pair', 'directory', 'manifest_file', 'manifest', 'cfg', 'name',
                                          'raw', 'errors', 'tasks', 'done', 'todo'))
                if todo and backend is None:
                    backend = factory(engine, cfg)
                    print(f"[corpus] {engine}: model đã tải; dùng chung cho toàn bộ sách đã chọn.", flush=True)
                if backend is not None:
                    backend.max_new = cfg.max_new_tokens
                    backend.model_min_pixels, backend.model_max_pixels = cfg.model_min_pixels, cfg.model_max_pixels
                    backend.preserve_input_resolution = cfg.segmentation != "none"
                    if engine == "hunyuan":
                        backend.rep, backend.prompt = cfg.repetition_penalty, cfg.prompt or HUNYUAN_PROMPT
                manifest.update(status="running", updated=now())
                write_json(manifest_file, manifest)
                print(f"[corpus] {engine} / {pair['book']}: {len(tasks)} trang được chọn, còn {len(todo)} trang cần OCR.", flush=True)
                for task in todo:
                    page, started = task["page"], time.monotonic()
                    print(f"[corpus] {engine} / {pair['book']} / trang {page + 1}: "
                          f"bắt đầu (timeout={cfg.page_timeout}s).", flush=True)

                    def on_timeout():
                        append_record(errors, {"page": page, "error": "TimeoutError",
                            "message": f"quá {cfg.page_timeout}s; worker phải khởi động lại",
                            "attempt": attempt, "timeout_seconds": cfg.page_timeout,
                            "seconds": round(time.monotonic() - started, 1), "updated": now()})

                    try:
                        with ocr_worker.page_deadline(cfg.page_timeout, on_timeout):
                            image, rendered_path, cache_hit = corpus_page_image(
                                campaign, manifest, pair['book'], task, cfg, render, image_path)
                            rendered_at = time.monotonic()
                            text, extra = run_prepared_page(backend, image, rendered_path, cfg, page=page)
                        record = {"page": page, "text": text, "w": image.width, "h": image.height,
                                  "sec": round(time.monotonic() - started, 1), **extra,
                                  'render_cache_hit': cache_hit, 'render_sec': round(rendered_at - started, 3),
                                  'inference_sec': round(time.monotonic() - rendered_at, 3)}
                        preparation = extra.get('image_preparation')
                        if preparation:
                            regions = preparation['regions']
                            dimensions = ', '.join('x'.join(map(str, r['input_size'])) for r in regions)
                            observed = [r.get('processor_size') for r in regions]
                            processor = (', '.join('x'.join(map(str, size)) for size in observed)
                                         if all(observed) else 'unobserved native tensors')
                            print(f"[image] {image.width}x{image.height} -> {len(regions)} input(s): "
                                  f"{dimensions}; processor: {processor}", flush=True)
                        append_record(ocr_worker.result_file(raw, pair["book"], name, 0), record)
                        done[page] = record
                        completed += 1
                        update_progress()
                        print(f"[corpus] {engine} / {pair['book']} / trang {page + 1}: "
                              f"xong sau {record['sec']}s.", flush=True)
                    except Exception as exc:
                        append_record(errors, {"page": page, "error": type(exc).__name__,
                            "message": str(exc), "traceback": traceback.format_exc(), "attempt": attempt,
                            "timeout_seconds": cfg.page_timeout,
                            "seconds": round(time.monotonic() - started, 1), "updated": now()})
                        detail = str(exc).replace("\n", " ")[:300]
                        print(f"[corpus] {engine} / {pair['book']} / trang {page + 1}: "
                              f"lỗi {type(exc).__name__}: {detail}; log: {errors}", flush=True)
                wanted = {t["page"] for t in tasks}
                got = len(wanted & done.keys())
                counts = {"pages": len(tasks), "done": got, "missing": len(tasks) - got}
                if cfg.export:
                    export_ocr(directory, {pair["book"]: (tasks[0]["src"], sorted(wanted))}, name)
                manifest.update(status="done" if got == len(tasks) else "partial", counts=counts, updated=now())
                write_json(manifest_file, manifest)
                print(f"[corpus] {engine} / {pair['book']}: xong {got}/{len(tasks)} trang.", flush=True)
    finally:
        if backend is not None:
            del backend
        for document in ocr_worker._DOCS.values():
            document.close()
        ocr_worker._DOCS.clear()
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--engine", choices=["hunyuan", "paddle", "ppocr", "tonggu"], required=True)
    parser.add_argument("--attempt")
    args = parser.parse_args()
    process_corpus(args.manifest, args.engine, attempt_id=args.attempt)
