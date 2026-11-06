"""Notebook-driven OCR: one worker/model for all books, then release the process."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tqdm.auto import tqdm

from .config import OCRConfig, load_ocr_config
from .ocr_worker import load_results
from .pipeline import _run_dir, list_pdfs, parse_pages, select_entries


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def gpu_lock(directory: Path):
    """Prevent separate notebook kernels from using this campaign's GPU at once."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "notebook_gpu.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("Một kernel khác đang chạy OCR cho tập dữ liệu này. Chờ lượt đó kết thúc trước khi chạy lại.") from None
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def prepare_corpus(hunyuan: OCRConfig, paddle: OCRConfig, profile_dir: str | Path | None = None,
                   *, max_books: int | None = None) -> Path:
    """Build identical page selections and per-book profiles for the two engines."""
    if hunyuan.engine != "hunyuan" or paddle.engine != "paddle":
        raise ValueError("Cần cấu hình Hunyuan và Paddle đúng engine.")
    if Path(hunyuan.input_dir).resolve() != Path(paddle.input_dir).resolve():
        raise ValueError("Hai model phải dùng cùng thư mục PDF.")
    if Path(hunyuan.workdir).resolve() != Path(paddle.workdir).resolve():
        raise ValueError("Hai model phải dùng cùng workdir.")
    if hunyuan.limit != paddle.limit:
        raise ValueError("Hai model phải dùng cùng limit.")
    if hunyuan.limit is not None and hunyuan.limit < 1:
        raise ValueError("limit phải >= 1 hoặc None.")
    if max_books is not None and max_books < 1:
        raise ValueError("max_books phải >= 1 hoặc None.")
    for base in (hunyuan, paddle):
        if base.fresh or base.resume_from:
            raise ValueError("Đổi workdir để chạy thử nghiệm mới; không dùng fresh/resume_from khi so hai model.")
        if base.n_gpu not in (0, 1):
            raise ValueError("Pipeline này giữ một model trên một GPU; n_gpu phải là 0 hoặc 1.")
    entries = list_pdfs(hunyuan.input_dir)
    selected = {}
    for base in (hunyuan, paddle):
        chosen = select_entries(entries, base.books)
        if max_books is not None:
            chosen = chosen[:max_books]
        selected[base.engine] = {
            e["name"]: (e, parse_pages(base.pages, e["n_pages"]))
            for e in chosen
        }
        remaining = base.limit
        trimmed = {}
        for book, (entry, pages) in selected[base.engine].items():
            if remaining is not None:
                pages = pages[:remaining]
                remaining -= len(pages)
            if pages:
                trimmed[book] = (entry, pages)
        selected[base.engine] = trimmed
    pages_h = {b: p for b, (_, p) in selected["hunyuan"].items()}
    pages_p = {b: p for b, (_, p) in selected["paddle"].items()}
    if not pages_h or pages_h != pages_p or any(not p for p in pages_h.values()):
        raise ValueError("Hai model phải OCR đúng cùng sách và cùng tập trang không rỗng.")
    workdir = Path(hunyuan.workdir).resolve()
    pairs, startup = [], {}
    with gpu_lock(workdir):
        for book, (entry, pages) in selected["hunyuan"].items():
            pair = {"book": book}
            for base in (hunyuan, paddle):
                cfg = (load_ocr_config(Path(profile_dir) / f"{entry['stem']}.json", base=base)
                       if profile_dir is not None else OCRConfig(**base.to_dict()))
                # Book selection belongs to this common plan; profiles control OCR settings.
                cfg.engine, cfg.books, cfg.pages = base.engine, [Path(entry["path"]).name], base.pages
                cfg.input_dir, cfg.workdir = str(Path(base.input_dir).resolve()), str(workdir)
                cfg.limit, cfg.n_layout, cfg.fresh, cfg.resume_from = base.limit, 0, False, []
                signature = (cfg.model, cfg.dtype, cfg.device, cfg.paddle_version,
                             json.dumps(cfg.paddle_args, sort_keys=True))
                if cfg.engine in startup and signature != startup[cfg.engine]:
                    raise ValueError("Các sách phải dùng cùng model/dtype/device để chỉ tải model một lần.")
                startup[cfg.engine] = signature
                name = cfg.run_name or cfg.engine
                directory = _run_dir(workdir, "ocr", name, cfg)
                directory.mkdir(parents=True, exist_ok=True)
                tasks = [{"book": book, "src": str(Path(entry["path"]).resolve()), "page": p} for p in pages]
                (directory / "tasks.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks), encoding="utf-8")
                manifest_path = directory / "run_manifest.json"
                old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
                existing = load_results(directory / "ocr_raw", book, name)
                count = len(set(existing) & set(pages))
                write_json(manifest_path, {**old, "kind": "ocr", "config": cfg.to_dict(),
                           "created": old.get("created", now()), "updated": now(),
                           "status": "done" if count == len(pages) else "pending",
                           "books": {book: {"src": tasks[0]["src"], "pages": pages}},
                           "tasks_total": len(pages), "tasks_selected": len(pages),
                           "counts": {"pages": len(pages), "done": count, "missing": len(pages) - count}})
                pair[cfg.engine] = str(directory)
            pairs.append(pair)
        manifest = workdir / "manifest.json"
        write_json(manifest, {"workflow": "notebook_sequential_models", "created": now(),
                   "total_books": len(pairs), "total_pages": sum(map(len, pages_h.values())), "pairs": pairs})
    return manifest


def engine_progress(manifest_path: str | Path, engine: str) -> dict:
    campaign = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    done = 0
    for pair in campaign["pairs"]:
        directory = Path(pair[engine])
        tasks = [json.loads(line) for line in (directory / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if line]
        pages = {t["page"] for t in tasks}
        cfg = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))["config"]
        done += len(pages & load_results(directory / "ocr_raw", pair["book"], cfg.get("run_name") or engine).keys())
    return {"pages": campaign["total_pages"], "done": done, "missing": campaign["total_pages"] - done}


def _terminate_worker(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
    else:
        os.killpg(process.pid, signal.SIGTERM)
    process.wait(timeout=15)


def run_corpus_engine(manifest_path: str | Path, engine: str) -> dict:
    """Blocking notebook call. The child process owns and releases the GPU model."""
    if engine not in {"hunyuan", "paddle"}:
        raise ValueError("engine phải là hunyuan hoặc paddle")
    manifest_path = Path(manifest_path).resolve()
    campaign = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = json.loads((Path(campaign["pairs"][0][engine]) / "run_manifest.json").read_text(encoding="utf-8"))["config"]
    budget = float(first.get("budget_h", 24)) * 3600
    max_restarts = int(first.get("max_restarts", 5))
    progress = engine_progress(manifest_path, engine)
    if progress["missing"] == 0:
        print(f"{engine}: đã có đủ {progress['done']:,} trang cùng cấu hình; dùng lại kết quả.")
        return progress
    with gpu_lock(manifest_path.parent):
        log = manifest_path.parent / f"{engine}_notebook.log"
        state_path = manifest_path.parent / "status.json"
        started = time.monotonic()
        environment = os.environ.copy()
        environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + environment.get("PYTHONPATH", "")
        python = first.get("worker_python") or sys.executable
        restarts = 0
        with tqdm(total=progress["pages"], initial=progress["done"], desc=f"OCR {engine}", unit="trang") as bar:
            while True:
                offset = log.stat().st_size if log.exists() else 0
                with log.open("a", encoding="utf-8") as output:
                    process = subprocess.Popen([python, "-u", "-m", "vtm_corpus.corpus_worker",
                                                "--manifest", str(manifest_path), "--engine", engine],
                                               stdout=output, stderr=subprocess.STDOUT, env=environment,
                                               start_new_session=os.name != "nt")
                try:
                    while process.poll() is None:
                        time.sleep(1)
                        with log.open(encoding="utf-8", errors="replace") as stream:
                            stream.seek(offset)
                            while True:
                                position = stream.tell()
                                line = stream.readline()
                                if not line or not line.endswith("\n"):
                                    stream.seek(position)
                                    break
                                if line.startswith(("[model]", "[corpus]")):
                                    bar.write(line.rstrip())
                            offset = stream.tell()
                        progress = engine_progress(manifest_path, engine)
                        bar.n = progress["done"]
                        bar.refresh()
                        write_json(state_path, {"status": "running", "active_engine": engine,
                                   "workflow": "notebook_sequential_models", "pid": os.getpid(),
                                   "worker_pid": process.pid, "updated": now(), **progress})
                        if time.monotonic() - started > budget:
                            raise TimeoutError(f"{engine}: vượt ngân sách {budget / 3600:g} giờ; kết quả đã xong được giữ lại.")
                    code = process.wait()
                except BaseException:
                    _terminate_worker(process)
                    write_json(state_path, {"status": "interrupted", "active_engine": None, "updated": now()})
                    raise
                if code == 124 and restarts < max_restarts:
                    restarts += 1
                    bar.write(f"{engine}: khởi động lại sau timeout ({restarts}/{max_restarts}).")
                    continue
                if code != 0:
                    tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-16:])
                    write_json(state_path, {"status": "error", "active_engine": None, "updated": now()})
                    raise RuntimeError(f"{engine} worker lỗi (exit={code}). Log: {log}\n{tail}")
                progress = engine_progress(manifest_path, engine)
                bar.n = progress["done"]
                bar.refresh()
                break
        write_json(state_path, {"status": "stage_complete" if not progress["missing"] else "partial",
                   "active_engine": None, "completed_engine": engine, "updated": now(), **progress})
    print(f"{engine}: {progress['done']:,}/{progress['pages']:,} trang có output; worker đã thoát và giải phóng model.")
    return progress
