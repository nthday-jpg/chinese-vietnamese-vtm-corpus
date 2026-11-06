"""A single model instance OCRs all selected books in one isolated process."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

from .config import HUNYUAN_PROMPT, OCRConfig
from . import ocr_worker
from .corpus_ocr import now, write_json
from .pipeline import export_ocr


def create_engine(engine: str, cfg: OCRConfig):
    args = SimpleNamespace(model=cfg.model, dtype=cfg.dtype, device=cfg.device,
                           max_new=cfg.max_new_tokens, rep_penalty=cfg.repetition_penalty,
                           prompt=cfg.prompt, paddle_version=cfg.paddle_version,
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


def process_corpus(manifest_path: str | Path, engine: str, *, engine_factory=None, renderer=None):
    factory = engine_factory or create_engine
    render = renderer or ocr_worker.render
    campaign = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    backend = None
    try:
        with tempfile.TemporaryDirectory(prefix="vtm-corpus-") as temporary:
            image_path = Path(temporary) / "page.png"
            for pair in campaign["pairs"]:
                directory = Path(pair[engine])
                manifest_file = directory / "run_manifest.json"
                manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
                cfg = OCRConfig(**manifest["config"])
                name = cfg.run_name or engine
                raw = directory / "ocr_raw"
                raw_book = raw / pair["book"]
                raw_book.mkdir(parents=True, exist_ok=True)
                tasks = [json.loads(line) for line in (directory / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if line]
                done = ocr_worker.load_results(raw, pair["book"], name)
                failed = set()
                for log in raw_book.glob(f"{name}_skip*.log"):
                    failed.update(int(p) for p in log.read_text(encoding="utf-8").split() if p.isdigit())
                errors = raw_book / f"{name}_errors.jsonl"
                if errors.exists():
                    for line in errors.read_text(encoding="utf-8").splitlines():
                        try:
                            failed.add(int(json.loads(line)["page"]))
                        except (json.JSONDecodeError, KeyError, ValueError):
                            pass
                todo = [t for t in tasks if t["page"] not in done and t["page"] not in failed]
                if todo and backend is None:
                    backend = factory(engine, cfg)
                    print(f"[corpus] {engine}: model đã tải; dùng chung cho toàn bộ sách đã chọn.", flush=True)
                if backend is not None:
                    backend.max_new = cfg.max_new_tokens
                    if engine == "hunyuan":
                        backend.rep, backend.prompt = cfg.repetition_penalty, cfg.prompt or HUNYUAN_PROMPT
                manifest.update(status="running", updated=now())
                write_json(manifest_file, manifest)
                print(f"[corpus] {engine} / {pair['book']}: {len(tasks)} trang được chọn, còn {len(todo)} trang cần OCR.", flush=True)
                for task in todo:
                    page, started = task["page"], time.monotonic()

                    def on_timeout():
                        with (raw_book / f"{name}_skip.log").open("a", encoding="utf-8") as stream:
                            stream.write(f"{page}\n")

                    try:
                        with ocr_worker.page_deadline(cfg.page_timeout, on_timeout):
                            image = render(task["src"], page, cfg.dpi, cfg.max_side)
                            image.save(image_path)
                            text, extra = backend.run(str(image_path))
                        record = {"page": page, "text": text, "w": image.width, "h": image.height,
                                  "sec": round(time.monotonic() - started, 1), **extra}
                        append_record(ocr_worker.result_file(raw, pair["book"], name, 0), record)
                        done[page] = record
                    except Exception as exc:
                        append_record(errors, {"page": page, "error": type(exc).__name__, "message": str(exc), "updated": now()})
                        print(f"[corpus] {engine} / {pair['book']}: một trang lỗi {type(exc).__name__}; đã ghi log, tiếp tục.", flush=True)
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
    parser.add_argument("--engine", choices=["hunyuan", "paddle"], required=True)
    args = parser.parse_args()
    process_corpus(args.manifest, args.engine)
