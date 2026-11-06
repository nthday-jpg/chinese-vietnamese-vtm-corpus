"""Điều phối pipeline: tìm input, chia việc, chạy worker, ghép output, ghi manifest."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from fnmatch import fnmatch
from datetime import datetime, timezone
from pathlib import Path

from tqdm.auto import tqdm

from . import ocr_worker
from .config import OCRConfig, TranslateConfig

HAN = re.compile(r"[㐀-鿿\U00020000-\U0002ffff]")
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#00a0b0",
          "#f032e6", "#9a6324", "#808000", "#000075"]


@dataclass
class RunResult:
    run_dir: Path
    raw_dir: Path
    output_dir: Path
    layout_dir: Path | None = None
    counts: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return str(self.run_dir)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sanitize(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name).strip("_") or "run"


def _detect_gpus() -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL)
        return max(1, len([ln for ln in out.splitlines() if ln.strip()]))
    except Exception:  # noqa: BLE001
        return 1


def _src_dir() -> str:
    return str(Path(__file__).resolve().parents[1])


def _run_dir(workdir: str | Path, kind: str, name: str, cfg) -> Path:
    digest = cfg.cache_key()[:8]
    return Path(workdir) / "runs" / f"{kind}_{_sanitize(name)}_{digest}"


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _manifest(run_dir: Path, cfg, kind: str, **extra) -> None:
    path = run_dir / "run_manifest.json"
    data = {
        "kind": kind,
        "created": _now(),
        "config": cfg.to_dict(),
        **extra,
    }
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            old.update(data)
            data = old
        except Exception:  # noqa: BLE001
            pass
    data["updated"] = _now()
    _write_json(path, data)


def parse_pages(spec: str | int, n: int) -> list[int]:
    """'1-10,15' -> [0..9, 14] (đếm từ 1); '' -> mọi trang."""
    spec = str(spec)
    if not spec.strip():
        return list(range(n))
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        out += range(int(a) - 1, min(int(b or a), n))
    return sorted(set(p for p in out if 0 <= p < n))


def _pdf_pages(path: Path) -> int:
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(path))
    try:
        return len(doc)
    finally:
        doc.close()


def _entry_matches(entry: dict, pattern: str) -> bool:
    return (fnmatch(entry["name"], pattern) or fnmatch(entry["stem"], pattern)
            or fnmatch(Path(entry["path"]).name, pattern))


def select_entries(entries: list[dict], patterns: list[str]) -> list[dict]:
    """Chọn sách theo tên/glob. [] = tất cả. Pattern không khớp -> báo lỗi."""
    if not patterns:
        return entries
    chosen: list[dict] = []
    seen: set[str] = set()
    missing: list[str] = []
    for pat in patterns:
        hits = [e for e in entries if _entry_matches(e, pat)]
        if not hits:
            missing.append(pat)
        for e in hits:
            if e["path"] not in seen:
                chosen.append(e)
                seen.add(e["path"])
    if missing:
        names = ", ".join(e["name"] for e in entries)
        raise ValueError(f"books không khớp: {missing}. Có sẵn: {names}")
    return chosen


def list_pdfs(input_dir: str | Path) -> list[dict]:
    """Liệt kê PDF trong input_dir: name (đã làm sạch), stem, path, pages."""
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"không thấy input_dir: {root}")
    files = [p for p in sorted(root.rglob("*.pdf")) if "__MACOSX" not in str(p)]
    if not files:
        raise FileNotFoundError(f"không thấy PDF nào trong {root}")
    entries: list[dict] = []
    used: set[str] = set()
    for p in files:
        name = _sanitize(p.stem)
        while name in used:
            name += "_"
        used.add(name)
        entries.append({"name": name, "stem": p.stem, "path": str(p)})
    # PDFium is not thread-safe: isolate concurrent document access in processes.
    workers = min(len(files), os.cpu_count() or 1, 8)
    if workers == 1:
        page_counts = map(_pdf_pages, files)
        for entry, count in zip(entries, page_counts):
            entry["n_pages"] = count
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for entry, count in zip(entries, pool.map(_pdf_pages, files)):
                entry["n_pages"] = count
    return entries


def list_texts(input_dir: str | Path, min_han: int = 0) -> list[dict]:
    """Liệt kê file .txt chữ Hán: name, stem, path, han."""
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"không thấy input_dir: {root}")
    files = sorted(root.rglob("*.txt"))
    entries: list[dict] = []
    used: set[str] = set()
    for p in files:
        text = p.read_text(encoding="utf-8", errors="ignore")
        n_han = len(HAN.findall(text))
        if n_han < min_han:
            continue
        name = _sanitize(p.stem)
        while name in used:
            name += "_"
        used.add(name)
        entries.append({"name": name, "stem": p.stem, "path": str(p), "han": n_han})
    return entries


def build_ocr_tasks(cfg: OCRConfig) -> tuple[dict[str, tuple[str, list[int]]], list[dict]]:
    entries = select_entries(list_pdfs(cfg.input_dir), cfg.books)
    if not entries:
        raise RuntimeError(f"không có sách nào được chọn (books={cfg.books})")
    books: dict[str, tuple[str, list[int]]] = {
        e["name"]: (e["path"], parse_pages(cfg.pages, e["n_pages"])) for e in entries
    }
    tasks = [{"book": b, "src": src, "page": pg}
             for b, (src, sel) in books.items() for pg in sel]
    if not tasks:
        raise RuntimeError("không có trang nào được chọn (kiểm tra pages)")
    return books, tasks


def _demo_pages(books: dict, n: int) -> list[dict]:
    """n trang ở giữa các sách (bìa ít khung, xem không thú vị)."""
    out: list[dict] = []
    for frac in (1 / 3, 1 / 2, 2 / 3, 1 / 4, 3 / 4):
        for b, (src, sel) in books.items():
            if not sel:
                continue
            t = {"book": b, "src": src, "page": sel[int(len(sel) * frac)]}
            if t not in out and len(out) < n:
                out.append(t)
    return out


def _copy_resume(sources: list[str], dest: Path, kind: str) -> int:
    """Copy kết quả cũ (jsonl/skip log) từ run dir hoặc thư mục ocr_raw/dich_raw."""
    if not sources:
        return 0
    raw_name = "ocr_raw" if kind == "ocr" else "dich_raw"
    raw_dirs: list[Path] = []
    for s in sources:
        p = Path(s)
        if p.name == raw_name:
            raw_dirs.append(p)
        elif (p / raw_name).is_dir():
            raw_dirs.append(p / raw_name)
        elif p.is_dir():
            raw_dirs.extend(d for d in p.rglob(raw_name) if d.is_dir())
    n = 0
    dest.mkdir(parents=True, exist_ok=True)
    for src in raw_dirs:
        for f in src.rglob("*"):
            if not f.is_file():
                continue
            if kind == "ocr":
                ok = (f.suffix == ".jsonl" and "_shard" in f.name) or "_skip" in f.name
            else:
                ok = f.suffix == ".jsonl" and f.name.startswith("dich_shard")
            if not ok:
                continue
            dst = dest / f.relative_to(src)
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
                n += 1
    return n


def _supervise(jobs: list[dict], budget_h: float, stall_min: int, max_restarts: int,
               *, progress: tqdm | None = None, progress_count=None) -> None:
    """Chạy các worker, tự khởi động lại nếu log đứng yên quá lâu."""
    if not jobs:
        return

    def log(msg: str) -> None:
        if progress is not None:
            progress.write(msg)
        else:
            print(msg, flush=True)

    def refresh_progress() -> None:
        if progress is not None and progress_count is not None:
            progress.n = min(progress_count(), progress.total or 0)
            progress.refresh()

    def start(j: dict) -> None:
        with open(j["log"], "a", encoding="utf-8") as stream:
            j["p"] = subprocess.Popen(
                j["cmd"], stdout=stream, stderr=subprocess.STDOUT,
                env=j["env"], cwd=j.get("cwd"),
            )

    for j in jobs:
        j["restart"] = 0
        j["model_log_offset"] = Path(j["log"]).stat().st_size if Path(j["log"]).exists() else 0
        start(j)

    def show_model_logs() -> None:
        for j in jobs:
            with open(j["log"], encoding="utf-8", errors="replace") as stream:
                stream.seek(j["model_log_offset"])
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line or not line.endswith("\n"):
                        stream.seek(offset)
                        break
                    if line.startswith("[model]"):
                        log(f"{Path(j['log']).name}: {line.rstrip()}")
                j["model_log_offset"] = stream.tell()

    refresh_progress()
    t0 = t_print = t_prog = time.time()
    while True:
        for j in jobs:
            if j["p"].poll() == ocr_worker.PAGE_TIMEOUT_EXIT and j["restart"] < max_restarts:
                j["restart"] += 1
                start(j)
                log(f"!!! {Path(j['log']).name}: quá thời gian/trang -> khởi động lại lần {j['restart']}")
        if not any(j["p"].poll() is None for j in jobs):
            break
        time.sleep(5)
        show_model_logs()
        if time.time() - t_prog >= 10:
            refresh_progress()
            t_prog = time.time()
        if time.time() - t0 > budget_h * 3600:
            log(f"!!! Hết {budget_h} giờ -> dừng, phần đã xong vẫn được giữ")
            for j in jobs:
                if j["p"].poll() is None:
                    j["p"].terminate()
            break
        for j in jobs:
            if (j["p"].poll() is None and time.time() - os.path.getmtime(j["log"]) > stall_min * 60
                    and j["restart"] < max_restarts):
                j["p"].kill()
                j["p"].wait()
                j["restart"] += 1
                start(j)
                log(f"!!! {Path(j['log']).name}: treo -> khởi động lại lần {j['restart']}")
        if time.time() - t_print > 120:
            t_print = time.time()
            last = []
            for j in jobs:
                try:
                    lines = Path(j["log"]).read_text(errors="ignore").strip().splitlines()
                except OSError:
                    lines = []
                last.append((lines or ["..."])[-1][-120:])
            log(f"[{(time.time() - t0) / 60:5.1f} phút] " + " | ".join(last))
    show_model_logs()
    refresh_progress()

    for j in jobs:
        try:
            code = j["p"].wait()
        except Exception:  # noqa: BLE001
            code = 1
        if code not in (0, -15):
            log(f"\n!!! {Path(j['log']).name} lỗi, 20 dòng cuối:")
            try:
                lines = Path(j["log"]).read_text(errors="ignore").splitlines()[-20:]
                log("\n".join(lines))
            except OSError:
                pass


def _ocr_cmd(cfg: OCRConfig, raw_dir: Path, tasks: Path, shard: int, nshard: int,
             name: str, extra: list[str], gpu: int) -> tuple[list[str], dict]:
    python = cfg.worker_python or sys.executable
    cmd = [
        python, "-m", "vtm_corpus.ocr_worker",
        "--engine", cfg.engine, "--name", name,
        "--tasks", str(tasks), "--out", str(raw_dir),
        "--shard", str(shard), "--nshard", str(nshard),
        "--dpi", str(cfg.dpi), "--max-side", str(cfg.max_side),
        "--max-new", str(cfg.max_new_tokens), "--dtype", cfg.dtype,
        "--model", cfg.model, "--prompt", cfg.prompt,
        "--rep-penalty", str(cfg.repetition_penalty),
        "--device", cfg.device,
        "--paddle-version", cfg.paddle_version,
        "--paddle-args", json.dumps(cfg.paddle_args, ensure_ascii=False),
        "--page-timeout", str(cfg.page_timeout),
        *extra,
    ]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = _src_dir() + os.pathsep + env.get("PYTHONPATH", "")
    if cfg.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return cmd, env


def _parse_layout(text: str, w: int, h: int) -> list[tuple[str, list[float]]]:
    """Hunyuan layout JSON -> [(nhãn, box pixel)]."""
    out = []
    for obj in re.findall(r"\{[^{}]*\}", text):
        label = re.search(r'"layout_type"\s*:\s*"([^"]*)"', obj)
        pts = [(int(x), int(y)) for x, y in re.findall(r"\((\d+),\s*(\d+)\)", obj)]
        if pts:
            xs = [x * w / 1000 for x, _ in pts]
            ys = [y * h / 1000 for _, y in pts]
            out.append((label.group(1) if label else "box", [min(xs), min(ys), max(xs), max(ys)]))
    return out


def _draw_boxes(t: dict, boxes: list[tuple[str, list[float]]], out_dir: Path,
                engine: str, cfg: OCRConfig) -> Path:
    from PIL import ImageDraw, ImageFont
    img = ocr_worker.render(t["src"], t["page"], cfg.dpi, cfg.max_side)
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=max(14, img.width // 60))
    labels = sorted({l for l, _ in boxes})
    for i, (label, box) in enumerate(boxes, 1):
        c = COLORS[labels.index(label) % len(COLORS)]
        d.rectangle(box, outline=c, width=3)
        d.text((box[0] + 4, box[1] + 2), f"{i}.{label}", fill=c, font=font)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{engine}_{t['book']}_tr{t['page'] + 1}.png"
    img.save(path)
    return path


def _run_layout(cfg: OCRConfig, run_dir: Path, raw_dir: Path, books: dict,
                name: str, n_gpu: int) -> list[Path]:
    if cfg.n_layout <= 0 or cfg.engine == "dummy":
        return []
    demo = _demo_pages(books, cfg.n_layout)
    if not demo:
        return []
    demo_path = run_dir / "demo.jsonl"
    demo_path.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in demo),
                         encoding="utf-8")
    if cfg.engine == "paddle":
        layout_name, extra = name, []
    else:
        layout_name = f"{name}_layout"
        extra = ["--prompt", cfg.layout_prompt, "--max-new", "3000"]
    n = max(1, min(n_gpu, len(demo)))
    jobs = []
    for k in range(n):
        cmd, env = _ocr_cmd(cfg, raw_dir, demo_path, k, n, layout_name, extra, k)
        jobs.append({"cmd": cmd, "env": env, "log": raw_dir / f"_{layout_name}_{k}.log",
                     "label": layout_name})
    _supervise(jobs, min(cfg.budget_h, cfg.layout_budget_h), cfg.stall_min, cfg.max_restarts)

    layout_dir = run_dir / "layout"
    paths: list[Path] = []
    for t in demo:
        rec = ocr_worker.load_results(raw_dir, t["book"], layout_name).get(t["page"])
        if not rec:
            continue
        if cfg.engine == "paddle":
            boxes = [(b["label"], b["box"]) for b in sorted(
                rec.get("boxes", []), key=lambda b: (b.get("order") is None, b.get("order") or 0))]
        else:
            boxes = _parse_layout(rec["text"], rec["w"], rec["h"])
        if boxes:
            paths.append(_draw_boxes(t, boxes, layout_dir, cfg.engine, cfg))
    return paths


def export_ocr(run_dir: str | Path, books: dict, name: str) -> dict:
    run_dir = Path(run_dir)
    raw_dir, txt_dir = run_dir / "ocr_raw", run_dir / "txt"
    txt_dir.mkdir(parents=True, exist_ok=True)
    total_ok = total = total_missing = 0
    for b, (_, sel) in books.items():
        recs = ocr_worker.load_results(raw_dir, b, name)
        pages = ["\n".join(ocr_worker.page_lines(recs[p])) for p in sel if p in recs]
        (txt_dir / f"{b}.txt").write_text("\n\n".join(p for p in pages if p) + "\n", encoding="utf-8")
        miss = len(sel) - len(pages)
        total_ok += len(pages)
        total += len(sel)
        total_missing += miss
        print(f"{b:<50} {len(pages)}/{len(sel)} trang" + (f"  -- THIẾU {miss} trang" if miss else ""))
    return {"pages": total, "done": total_ok, "missing": total_missing}


def _ocr_progress_count(raw_dir: Path, books: dict, name: str,
                        wanted: set[tuple[str, int]]) -> int:
    """Số trang đã xong (kể cả trang bị bỏ qua) trong tập trang đang chọn."""
    done: set[tuple[str, int]] = set()
    for b in books:
        done.update((b, p) for p in ocr_worker.load_results(raw_dir, b, name) if (b, p) in wanted)
        for f in (raw_dir / b).glob(f"{name}_skip*.log"):
            for token in f.read_text(encoding="utf-8", errors="ignore").split():
                try:
                    page = int(token)
                except ValueError:
                    continue
                if (b, page) in wanted:
                    done.add((b, page))
    return len(done)


def _dich_progress_count(raw_dir: Path, wanted: set[str]) -> int:
    """Số đoạn đã dịch trong tập đoạn đang chọn."""
    done: set[str] = set()
    for f in raw_dir.glob("dich_shard*.jsonl"):
        try:
            lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                chunk_id = json.loads(line)["id"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            if chunk_id in wanted:
                done.add(chunk_id)
    return len(done)


def run_ocr(cfg: OCRConfig, *, books: dict | None = None, tasks: list[dict] | None = None) -> RunResult:
    name = cfg.run_name or cfg.engine
    run_dir = _run_dir(cfg.workdir, "ocr", name, cfg)
    if cfg.fresh and run_dir.exists():
        shutil.rmtree(run_dir)
    raw_dir, txt_dir = run_dir / "ocr_raw", run_dir / "txt"
    raw_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    if books is None or tasks is None:
        books, tasks = build_ocr_tasks(cfg)
    n_total = len(tasks)
    if cfg.limit is not None:
        tasks = tasks[:cfg.limit]
    tasks_path = run_dir / "tasks.jsonl"
    tasks_path.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks),
                          encoding="utf-8")
    n_resume = _copy_resume(cfg.resume_from, raw_dir, "ocr")
    n_gpu = max(1, cfg.n_gpu or _detect_gpus())
    _manifest(run_dir, cfg, "ocr", status="running", state="ocr", books={
        b: {"src": src, "pages": sel} for b, (src, sel) in books.items()},
        tasks_total=n_total, tasks_selected=len(tasks), resumed_files=n_resume, n_gpu=n_gpu)
    print(f"Run: {run_dir}\n{len(books)} sách, {n_total} trang, chọn {len(tasks)}, "
          f"{n_gpu} GPU/worker, resume {n_resume} file", flush=True)

    layout_paths = _run_layout(cfg, run_dir, raw_dir, books, name, n_gpu)
    if layout_paths:
        print(f"Layout: {len(layout_paths)} ảnh -> {run_dir / 'layout'}", flush=True)

    jobs = []
    for k in range(n_gpu):
        cmd, env = _ocr_cmd(cfg, raw_dir, tasks_path, k, n_gpu, name, [], k)
        jobs.append({"cmd": cmd, "env": env, "log": raw_dir / f"_{name}_{k}.log", "label": name})
    wanted_pages = {(t["book"], t["page"]) for t in tasks}

    def count_done() -> int:
        return _ocr_progress_count(raw_dir, books, name, wanted_pages)

    with tqdm(total=len(tasks), desc=f"OCR {name}", unit="trang", dynamic_ncols=True) as bar:
        bar.n = min(count_done(), len(tasks))
        bar.refresh()
        _supervise(jobs, cfg.budget_h, cfg.stall_min, cfg.max_restarts,
                   progress=bar, progress_count=count_done)

    counts = export_ocr(run_dir, books, name) if cfg.export else {}
    _manifest(run_dir, cfg, "ocr", status="done", counts=counts,
              layout_images=len(layout_paths))
    return RunResult(run_dir, raw_dir, txt_dir, run_dir / "layout", counts)


def _split_long(line: str, n: int) -> list[str]:
    parts, cur = [], ""
    for s in re.split(r"(?<=[。！？；!?;])", line):
        while len(s) > n:
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(s[:n])
            s = s[n:]
        if cur and len(cur) + len(s) > n:
            parts.append(cur)
            cur = ""
        cur += s
    return parts + ([cur] if cur else [])


def make_chunks(text: str, n: int) -> list[str]:
    chunks, cur = [], []
    for line in (ln.strip() for ln in text.splitlines()):
        for piece in _split_long(line, n) if line else []:
            if cur and sum(map(len, cur)) + len(piece) > n:
                chunks.append("\n".join(cur))
                cur = []
            cur.append(piece)
    return chunks + (["\n".join(cur)] if cur else [])


def build_translate_chunks(cfg: TranslateConfig) -> tuple[dict[str, list[dict]], list[dict]]:
    entries = select_entries(list_texts(cfg.input_dir, cfg.min_han), cfg.books)
    if not entries:
        raise RuntimeError(f"không có sách nào được chọn (books={cfg.books})")
    books: dict[str, list[dict]] = {}
    for e in entries:
        b = e["name"]
        text = Path(e["path"]).read_text(encoding="utf-8", errors="ignore")
        books[b] = [
            {"id": f"{b}|{i}|{hashlib.md5(c.encode()).hexdigest()[:8]}", "zh": c}
            for i, c in enumerate(make_chunks(text, cfg.chunk_chars))
        ]
    chunks = [c for cs in books.values() for c in cs]
    return books, chunks


def load_dich(run_dir: str | Path) -> dict[str, dict]:
    out = {}
    for f in (Path(run_dir) / "dich_raw").glob("dich_shard*.jsonl"):
        with open(f, encoding="utf-8") as stream:
            for line in stream:
                try:
                    r = json.loads(line)
                    out[r["id"]] = r
                except (json.JSONDecodeError, KeyError):
                    pass
    return out


def export_translate(run_dir: str | Path, books: dict) -> dict:
    run_dir = Path(run_dir)
    out_dir = run_dir / "dich"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = load_dich(run_dir)
    n_ok = n_total = n_missing = n_max = 0
    for b, cs in books.items():
        vi = [done[c["id"]]["vi"] if c["id"] in done else "[chưa dịch]" for c in cs]
        (out_dir / f"{b}.txt").write_text("\n\n".join(vi) + "\n", encoding="utf-8")
        ok = sum(c["id"] in done for c in cs)
        mx = sum(done[c["id"]].get("hit_max", False) for c in cs if c["id"] in done)
        n_ok += ok
        n_total += len(cs)
        n_missing += len(cs) - ok
        n_max += mx
        print(f"{b:<50} {ok}/{len(cs)} đoạn" + (f", {mx} đoạn chạm MAX_NEW_TOKENS" if mx else ""))
    return {"chunks": n_total, "done": n_ok, "missing": n_missing, "hit_max": n_max}


def run_translate(cfg: TranslateConfig, *, books: dict | None = None,
                  chunks: list[dict] | None = None) -> RunResult:
    name = cfg.run_name or cfg.model.split("/")[-1]
    run_dir = _run_dir(cfg.workdir, "dich", name, cfg)
    if cfg.fresh and run_dir.exists():
        shutil.rmtree(run_dir)
    raw_dir, out_dir = run_dir / "dich_raw", run_dir / "dich"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if books is None or chunks is None:
        books, chunks = build_translate_chunks(cfg)
    n_total = len(chunks)
    if cfg.limit is not None:
        chunks = chunks[:cfg.limit]
    chunks_path = run_dir / "chunks.jsonl"
    chunks_path.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks),
                           encoding="utf-8")
    prompt_path = run_dir / "prompt.json"
    prompt_path.write_text(json.dumps({"system": cfg.system, "user": cfg.user},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    n_resume = _copy_resume(cfg.resume_from, raw_dir, "dich")
    n_gpu = max(1, cfg.n_gpu or _detect_gpus())
    n_models = max(1, n_gpu // cfg.gpu_per_model) if cfg.engine == "hf" else 1
    _manifest(run_dir, cfg, "dich", status="running", state="dich",
              books={b: len(cs) for b, cs in books.items()},
              chunks_total=n_total, chunks_selected=len(chunks), resumed_files=n_resume,
              n_gpu=n_gpu, n_models=n_models)
    print(f"Run: {run_dir}\n{len(books)} sách, {n_total} đoạn, chọn {len(chunks)}, "
          f"{n_models} model, resume {n_resume} file", flush=True)

    jobs = []
    for k in range(n_models):
        gpus = ",".join(str(g) for g in range(k * cfg.gpu_per_model, (k + 1) * cfg.gpu_per_model))
        cmd = [
            cfg.worker_python or sys.executable, "-m", "vtm_corpus.dich_worker",
            "--engine", cfg.engine, "--model", cfg.model,
            "--chunks", str(chunks_path), "--prompt", str(prompt_path),
            "--out", str(raw_dir), "--shard", str(k), "--nshard", str(n_models),
            "--batch", str(cfg.batch), "--max-new", str(cfg.max_new_tokens),
            "--temperature", str(cfg.temperature), "--rep-penalty", str(cfg.repetition_penalty),
            "--device", cfg.device,
        ]
        if cfg.load_4bit:
            cmd.append("--load-4bit")
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONPATH"] = _src_dir() + os.pathsep + env.get("PYTHONPATH", "")
        env["CUDA_VISIBLE_DEVICES"] = "" if cfg.device == "cpu" or cfg.engine == "dummy" else gpus
        jobs.append({"cmd": cmd, "env": env, "log": raw_dir / f"_dich_{k}.log", "label": "dich"})
    wanted_chunks = {c["id"] for c in chunks}

    def count_done() -> int:
        return _dich_progress_count(raw_dir, wanted_chunks)

    with tqdm(total=len(chunks), desc=f"Dịch {name}", unit="đoạn", dynamic_ncols=True) as bar:
        bar.n = min(count_done(), len(chunks))
        bar.refresh()
        _supervise(jobs, cfg.budget_h, cfg.stall_min, cfg.max_restarts,
                   progress=bar, progress_count=count_done)

    counts = export_translate(run_dir, books) if cfg.export else {}
    _manifest(run_dir, cfg, "dich", status="done", counts=counts)
    return RunResult(run_dir, raw_dir, out_dir, None, counts)


def find_run(workdir: str | Path, kind: str, name: str | None = None) -> Path:
    """Run mới nhất trong <workdir>/runs (theo mtime)."""
    root = Path(workdir) / "runs"
    pat = f"{kind}_*" if not name else f"{kind}_{_sanitize(name)}_*"
    runs = [p for p in root.glob(pat) if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"không thấy run {kind} trong {root}")
    return max(runs, key=lambda p: p.stat().st_mtime)
