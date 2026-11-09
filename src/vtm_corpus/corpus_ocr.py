"""Notebook-driven OCR: one worker/model for all books, then release the process."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from tqdm.auto import tqdm

from .config import OCRConfig, load_ocr_config
from .ocr_worker import load_results
from .pipeline import _run_dir, list_pdfs, parse_pages, select_entries
from .worker_paths import to_wsl_path


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def ocr_source_sha256(path):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def corpus_model_key(config):
    """Selection changes do not invalidate already completed OCR pages."""
    config = OCRConfig(**config.to_dict()) if isinstance(config, OCRConfig) else OCRConfig(**config)
    config.books, config.pages = [], ''
    return config.cache_key()


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


def prepare_multi_corpus(models: dict[str, OCRConfig], profile_dir: str | Path | None = None,
                         *, max_books: int | None = None, cache_renders=True) -> Path:
    """Prepare identical selections for all configured engines; preserve model identities."""
    if len(models) < 2 or any(key != cfg.engine for key, cfg in models.items()):
        raise ValueError("Cần >= 2 model với key trùng engine, mỗi engine một cấu hình.")
    base_configs = list(models.values())
    first = base_configs[0]
    if len({str(Path(cfg.input_dir).resolve()) for cfg in base_configs}) != 1:
        raise ValueError("Các model phải dùng cùng thư mục PDF.")
    if len({str(Path(cfg.workdir).resolve()) for cfg in base_configs}) != 1:
        raise ValueError("Các model phải dùng cùng workdir.")
    if len({cfg.limit for cfg in base_configs}) != 1:
        raise ValueError("Các model phải dùng cùng limit.")
    if first.limit is not None and first.limit < 1:
        raise ValueError("limit phải >= 1 hoặc None.")
    if max_books is not None and max_books < 1:
        raise ValueError("max_books phải >= 1 hoặc None.")
    for base in base_configs:
        if base.fresh or base.resume_from:
            raise ValueError("Đổi workdir để chạy thử nghiệm mới; không dùng fresh/resume_from khi so nhiều model.")
        if base.n_gpu not in (0, 1):
            raise ValueError("Pipeline này giữ một model trên một GPU; n_gpu phải là 0 hoặc 1.")
    entries = list_pdfs(first.input_dir)
    selected = {}
    for base in base_configs:
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
    pages_h = {b: p for b, (_, p) in selected[first.engine].items()}
    if not pages_h or any({b: p for b, (_, p) in chosen.items()} != pages_h for chosen in selected.values()):
        raise ValueError("Các model phải OCR đúng cùng sách và cùng tập trang không rỗng.")
    workdir = Path(first.workdir).resolve()
    pairs, startup = [], {}
    with gpu_lock(workdir):
        prior_path = workdir / 'manifest.json'
        prior = json.loads(prior_path.read_text(encoding='utf-8')) if prior_path.is_file() else {}
        prior_pairs = {pair['book']: pair for pair in prior.get('pairs', [])}
        for book, (entry, pages) in selected[first.engine].items():
            pair = {"book": book}
            source_sha = ocr_source_sha256(entry['path'])
            for base in base_configs:
                cfg = (load_ocr_config(Path(profile_dir) / f"{entry['stem']}.json", base=base)
                       if profile_dir is not None else OCRConfig(**base.to_dict()))
                # Book selection belongs to this common plan; profiles control OCR settings.
                cfg.engine, cfg.books, cfg.pages = base.engine, [Path(entry["path"]).name], base.pages
                # Book profiles tune image/token settings, not which model is compared.
                for identity in ("model", "model_revision", "run_name", "paddle_version", "paddle_args",
                                 "tonggu_repo", "tonggu_config", "worker_python", "worker_wsl_distro", "worker_pythonpath"):
                    setattr(cfg, identity, getattr(base, identity))
                cfg.input_dir, cfg.workdir = str(Path(base.input_dir).resolve()), str(workdir)
                cfg.limit, cfg.n_layout, cfg.fresh, cfg.resume_from = base.limit, 0, False, []
                signature = (cfg.model, cfg.model_revision, cfg.dtype, cfg.device, cfg.paddle_version,
                             json.dumps(cfg.paddle_args, sort_keys=True), cfg.tonggu_repo, cfg.tonggu_config)
                if cfg.engine in startup and signature != startup[cfg.engine]:
                    raise ValueError("Các sách phải dùng cùng model/dtype/device để chỉ tải model một lần.")
                startup[cfg.engine] = signature
                name = cfg.run_name or cfg.engine
                directory = _run_dir(workdir, "ocr", name, cfg)
                if source_sha:
                    directory = directory.with_name(directory.name + '_' + source_sha[:12])
                prior_dir = Path(prior_pairs.get(book, {}).get(cfg.engine, directory))
                prior_manifest = prior_dir / 'run_manifest.json'
                if prior_manifest.is_file() and prior_dir.resolve().is_relative_to(workdir):
                    cached = json.loads(prior_manifest.read_text(encoding='utf-8'))
                    source_info = cached.get('books', {}).get(book, {})
                    if (corpus_model_key(cached['config']) == corpus_model_key(cfg)
                            and source_info.get('src') == str(Path(entry['path']).resolve())
                            and source_info.get('sha256', source_sha) == source_sha):
                        directory = prior_dir
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
                           "books": {book: {"src": tasks[0]["src"], "pages": pages, "sha256": source_sha,
                               "legacy_source_unverified": bool(existing) and (
                                   old.get('books', {}).get(book, {}).get('legacy_source_unverified', False)
                                   or not old.get('books', {}).get(book, {}).get('sha256'))}},
                           "tasks_total": len(pages), "tasks_selected": len(pages),
                           "counts": {"pages": len(pages), "done": count, "missing": len(pages) - count}})
                pair[cfg.engine] = str(directory)
            pairs.append(pair)
        manifest = workdir / "manifest.json"
        write_json(manifest, {"workflow": "notebook_sequential_models", "created": now(),
                   "models": list(models), "total_books": len(pairs),
                   "total_pages": sum(map(len, pages_h.values())), "pairs": pairs,
                   "cache_renders": bool(cache_renders), "render_cache": str(workdir / 'render_cache')})
    return manifest


def engine_progress(manifest_path: str | Path, engine: str, *, attempt_id=None) -> dict:
    campaign = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if attempt_id:
        snapshot = Path(manifest_path).parent / f'{engine}_progress.json'
        try:
            state = json.loads(snapshot.read_text(encoding='utf-8'))
            if (state.get('attempt') == attempt_id and state.get('engine') == engine
                    and state.get('pages') == campaign['total_pages']
                    and type(state.get('done')) is int and 0 <= state['done'] <= state['pages']
                    and state.get('missing') == state['pages'] - state['done']):
                return {key: state[key] for key in ('pages', 'done', 'missing')}
        except (OSError, ValueError, KeyError):
            pass
        # No matching snapshot yet: retain the count from before worker startup.
        return None
    done = 0
    for pair in campaign["pairs"]:
        directory = Path(pair[engine])
        tasks = [json.loads(line) for line in (directory / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if line]
        pages = {t["page"] for t in tasks}
        cfg = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))["config"]
        done += len(pages & load_results(directory / "ocr_raw", pair["book"], cfg.get("run_name") or engine).keys())
    return {"pages": campaign["total_pages"], "done": done, "missing": campaign["total_pages"] - done}


def _terminate_worker(process: subprocess.Popen, *, wsl_distro="", pid_file=None, token="") -> None:
    if wsl_distro and pid_file and Path(pid_file).exists():
        try:
            marker = json.loads(Path(pid_file).read_text(encoding="utf-8"))
            if marker.get("token") == token and isinstance(marker.get("pgid"), int) and marker["pgid"] > 1:
                subprocess.run(["wsl.exe", "--distribution", wsl_distro, "--exec", "kill", "-KILL", "--",
                                f"-{marker['pgid']}"], capture_output=True, timeout=15, check=True)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            print(f"Không xác nhận được việc dừng worker WSL: {exc}")
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
    else:
        os.killpg(process.pid, signal.SIGTERM)
    process.wait(timeout=15)


def corpus_worker_command(first: dict, manifest_path: Path, engine: str, token: str) -> tuple[list[str], Path | None]:
    """Argument lists only: no shell interpolation of paths, model names or prompts."""
    distro = first.get("worker_wsl_distro", "")
    if distro:
        if os.name != "nt":
            raise ValueError("worker_wsl_distro is for Windows; clear it when running on Linux")
        python = first.get("worker_python", "")
        if not python.startswith("/"):
            raise ValueError("Set worker_python to an absolute Linux Python path for the WSL worker")
        pid_file = manifest_path.parent / f"{engine}_wsl_pid.json"
        command = ["wsl.exe", "--distribution", distro, "--exec", python, "-u",
                   to_wsl_path(Path(__file__).with_name("wsl_worker_launcher.py").resolve()),
                   "--manifest", to_wsl_path(manifest_path), "--engine", engine,
                   "--token", token, "--pid-file", to_wsl_path(pid_file)]
        return command, pid_file
    return [first.get("worker_python") or sys.executable, "-u", "-m", "vtm_corpus.corpus_worker",
            "--manifest", str(manifest_path), "--engine", engine], None


def run_corpus_engine(manifest_path: str | Path, engine: str) -> dict:
    """Blocking notebook call. The child process owns and releases the GPU model."""
    if engine not in {"hunyuan", "paddle", "ppocr", "tonggu"}:
        raise ValueError("engine phải là hunyuan, paddle, ppocr hoặc tonggu")
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
        environment["VTM_OCR_ATTEMPT"] = uuid.uuid4().hex
        environment["PYTHONPATH"] = os.pathsep.join([
            str(Path(__file__).resolve().parents[1]), *first.get("worker_pythonpath", []),
            environment.get("PYTHONPATH", ""),
        ])
        restarts = 0
        with tqdm(total=progress["pages"], initial=progress["done"], desc=f"OCR {engine}", unit="trang") as bar:
            while True:
                worker_token = uuid.uuid4().hex
                command, pid_file = corpus_worker_command(first, manifest_path, engine, worker_token)
                # Explicit argument also works across the Windows -> WSL boundary.
                command.extend(["--attempt", environment["VTM_OCR_ATTEMPT"]])
                offset = log.stat().st_size if log.exists() else 0
                with log.open("a", encoding="utf-8") as output:
                    process = subprocess.Popen(command,
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
                        progress = engine_progress(manifest_path, engine, attempt_id=environment['VTM_OCR_ATTEMPT']) or progress
                        bar.n = progress["done"]
                        bar.refresh()
                        write_json(state_path, {"status": "running", "active_engine": engine,
                                   "workflow": "notebook_sequential_models", "pid": os.getpid(),
                                   "worker_pid": process.pid, "updated": now(), **progress})
                        if time.monotonic() - started > budget:
                            raise TimeoutError(f"{engine}: vượt ngân sách {budget / 3600:g} giờ; kết quả đã xong được giữ lại.")
                    code = process.wait()
                    # A fast worker can finish between polls; include its final
                    # page/error lines instead of leaving them only in the log.
                    with log.open(encoding="utf-8", errors="replace") as stream:
                        stream.seek(offset)
                        for line in stream:
                            if line.startswith(("[model]", "[corpus]")):
                                bar.write(line.rstrip())
                except BaseException:
                    _terminate_worker(process, wsl_distro=first.get("worker_wsl_distro", ""),
                                      pid_file=pid_file, token=worker_token)
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
