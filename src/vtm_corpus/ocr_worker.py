"""Worker OCR từng trang PDF; bỏ qua trang đã xong khi chạy lại.

Các backend được dùng chung bởi pipeline OCR đơn và so sánh nhiều model.
Chỉ import thư viện model khi khởi tạo backend.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from .config import HUNYUAN_PROMPT, OCRConfig
from .ocr_quality import ocr_quality_issues
from .image_preparation import run_prepared_page
from .image_dataset import dataset_ocr_page, dataset_file_hash

PADDLE_IGNORE = {"header", "header_image", "footer", "footer_image", "number", "footnote", "aside_text",
                 "image", "chart", "seal"}


def clean_degenerate(s: str, max_unit: int = 40, min_chars: int = 60, min_cycles: int = 4) -> str:
    """Cắt đuôi lặp vô hạn (VLM kẹt vào 1 chu kỳ ngắn rồi lặp tới hết token)."""
    st = s.rstrip()
    for u in range(1, max_unit + 1):
        unit = st[-u:]
        if len(st) < min_chars or not unit.strip():
            continue
        n, i = 0, len(st)
        while i >= u and st[i - u:i] == unit:
            n, i = n + 1, i - u
        if n * u >= min_chars and n >= min_cycles:
            return st[:i].rstrip()
    return s


def md_to_lines(md: str) -> list[str]:
    """Markdown/HTML của OCR -> các dòng text thuần (bỏ #, **, ảnh, thẻ bảng, toạ độ hình)."""
    md = re.sub(r"</t[dh]>", " ", md)
    md = re.sub(r"</tr>|<br\s*/?>", "\n", md)
    md = re.sub(r"<[^>]+>", "", md)
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)
    out = []
    for ln in md.splitlines():
        ln = re.sub(r"^\s{0,3}#{1,6}\s*", "", ln)
        ln = re.sub(r"^\s*[-*+]\s+(?=\S)", "", ln)
        ln = re.sub(r"(?:[ \t,，]*\(\d{1,4},\d{1,4}\),\(\d{1,4},\d{1,4}\))+\s*$", "", ln)  # toạ độ hình của Hunyuan
        ln = ln.replace("**", "").replace("__", "").strip()
        if ln.startswith("|") and ln.endswith("|"):
            ln = " ".join(c.strip() for c in ln.strip("|").split("|") if c.strip())
        if ln and not re.fullmatch(r"[-|:\s]+", ln):
            out.append(ln)
    return out


def page_lines(rec: dict) -> list[str]:
    """1 trang OCR -> các dòng chữ theo thứ tự đọc."""
    if "blocks" not in rec:  # hunyuan: text markdown của cả trang
        return md_to_lines(rec["text"])
    return [ln for b in ordered_page_blocks(rec["blocks"]) for ln in md_to_lines(b["text"])]


def ordered_page_blocks(blocks):
    """Use the same block sequence for exported text and segmented geometry."""
    ordered = sorted((b for b in blocks if b["order"] is not None), key=lambda b: b["order"])
    center = lambda b: ((b["box"][0] + b["box"][2]) / 2, (b["box"][1] + b["box"][3]) / 2)  # noqa: E731
    for b in (b for b in blocks if b["order"] is None):  # bảng, chú thích hình: paddle không xếp thứ tự đọc
        if not ordered:
            ordered.append(b)
            continue
        cx, cy = center(b)
        k = min(range(len(ordered)), key=lambda i: (center(ordered[i])[0] - cx) ** 2 + (center(ordered[i])[1] - cy) ** 2)
        ordered.insert(k + (cy > center(ordered[k])[1]), b)  # đặt cạnh vùng gần nhất
    return ordered


_DOCS = {}


def render(src: str, page: int, dpi: int = 200, max_side: int = 1600):
    """Trang PDF -> ảnh PIL (cạnh dài tối đa max_side)."""
    import pypdfium2 as pdfium
    if src not in _DOCS:
        _DOCS[src] = pdfium.PdfDocument(src)
    pdf_page = _DOCS[src][page]
    bitmap = None
    try:
        bitmap = pdf_page.render(scale=dpi / 72)
        img = bitmap.to_pil().convert("RGB")
    finally:
        if bitmap is not None:
            bitmap.close()
        pdf_page.close()
    original_size = img.size
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        from PIL import Image
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.Resampling.LANCZOS)
    img.info["render"] = {"dpi": dpi, "uncapped_size": list(original_size), "size": list(img.size),
                          "downsampled": img.size != original_size, "max_side": max_side}
    return img


class Dummy:
    """Engine không cần model, dùng để test toàn bộ pipeline trên CPU.

    Trả về đủ chữ Hán để bước dịch không lọc mất file.
    """

    def __init__(self, a):
        self.a = a

    def run(self, img_path):
        from PIL import Image
        with Image.open(img_path) as im:
            w, h = im.size
        text = f"測試頁面 {w}x{h}\n" + "黃帝內經素問" * 10
        return text, {"hit_max": False, "dummy": True}


class Hunyuan:
    def __init__(self, a):
        started = time.monotonic()
        print("[model] Hunyuan: bắt đầu import torch/transformers", flush=True)
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import HunYuanVLForConditionalGeneration as M
        except ImportError:
            from transformers import AutoModelForImageTextToText as M
        self.torch, self.max_new, self.rep = torch, a.max_new, a.rep_penalty
        self.prompt = a.prompt or HUNYUAN_PROMPT
        self.model_min_pixels = getattr(a, "model_min_pixels", None)
        self.model_max_pixels = getattr(a, "model_max_pixels", None)
        self.dev = self._device(a.device)
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[a.dtype]
        print(f"[model] Hunyuan: bắt đầu tải/đọc model {a.model} (dùng cache nếu có)", flush=True)
        revision = {"revision": a.model_revision} if getattr(a, "model_revision", "") else {}
        self.proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True, use_fast=False, **revision)
        self.processor_default_size = dict(self.proc.image_processor.size)
        self.model = M.from_pretrained(a.model, dtype=dtype, trust_remote_code=True, **revision).to(self.dev).eval()
        print(f"[model] Hunyuan: model sẵn sàng trên {self.dev}, sau {time.monotonic() - started:.1f}s", flush=True)

    @staticmethod
    def _device(requested: str) -> str:
        import torch
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return requested

    def run(self, img_path):
        torch = self.torch
        # Read the effective processor settings, rather than guessing from the
        # model card (released configs may contain conflicting size/max_pixels).
        processor = self.proc.image_processor
        size = dict(self.processor_default_size)
        if self.model_min_pixels is not None:
            size["shortest_edge"] = self.model_min_pixels
        if self.model_max_pixels is not None:
            size["longest_edge"] = self.model_max_pixels
        processor.size = size
        msgs = [{"role": "user", "content": [{"type": "image", "image": img_path},
                                             {"type": "text", "text": self.prompt}]}]
        inputs = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt").to(self.dev)
        grid = inputs.get("image_grid_thw")
        grids = grid.detach().cpu().tolist() if grid is not None else []
        patch = getattr(processor, "patch_size", 16)
        model_input = {"processor": type(processor).__name__, "effective_size": size,
                       "image_grid_thw": grids,
                       "processed_sizes_wh": [[int(row[2])*patch, int(row[1])*patch] for row in grids],
                       "pixel_values_shape": list(inputs["pixel_values"].shape) if "pixel_values" in inputs else None,
                       "observation": "processor_tensor" if grids else "unavailable"}
        for k, v in list(inputs.items()):
            if torch.is_tensor(v) and v.is_floating_point():
                inputs[k] = v.to(self.model.dtype)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new, do_sample=False,
                                      repetition_penalty=self.rep)
        gen = out[:, inputs["input_ids"].shape[1]:]
        text = self.proc.batch_decode(gen, skip_special_tokens=True)[0]
        hit_max = int(gen.shape[1]) >= self.max_new
        return text, {"hit_max": hit_max, "model_input": model_input,
                      "quality_issues": ocr_quality_issues(text, {"hit_max": hit_max})}


class Paddle:
    def __init__(self, a):
        started = time.monotonic()
        print("[model] Paddle: bắt đầu import paddleocr", flush=True)
        from paddleocr import PaddleOCRVL
        print(f"[model] Paddle: bắt đầu tải/khởi tạo PaddleOCR-VL {a.paddle_version} (dùng cache nếu có)", flush=True)
        options = json.loads(a.paddle_args)
        self.spatial_frame_safe = all(options.get(k) is False for k in ('use_doc_orientation_classify', 'use_doc_unwarping'))
        self.pipe = PaddleOCRVL(pipeline_version=a.paddle_version, **options)
        self.max_new = a.max_new
        self.model_min_pixels = getattr(a, "model_min_pixels", None)
        self.model_max_pixels = getattr(a, "model_max_pixels", None)
        print(f"[model] Paddle: model sẵn sàng, sau {time.monotonic() - started:.1f}s", flush=True)

    def run(self, img_path):
        kwargs = {"max_new_tokens": self.max_new}
        for key, value in (("min_pixels", self.model_min_pixels), ("max_pixels", self.model_max_pixels)):
            if value is not None:
                kwargs[key] = value
        # Unsupported APIs must fail visibly; never silently drop quality limits.
        res = list(self.pipe.predict(img_path, **kwargs))[0]
        text, extra = self.parse(res.json)
        return text, {**extra, "coordinate_frame": "input_image" if self.spatial_frame_safe else "unverified_native_transform",
                      "model_input": {"requested": kwargs, "observation": "native_internal_unavailable"}}

    @staticmethod
    def parse(j):
        """res.json của PaddleOCR-VL -> (text, {"blocks": vùng đã đọc chữ, "boxes": mọi khung layout})."""
        j = j.get("res", j) if isinstance(j, dict) else {}
        blocks = [{"label": b.get("block_label"), "box": [round(x) for x in b.get("block_bbox", [0, 0, 0, 0])],
                   "order": b.get("block_order"), "text": b.get("block_content") or ""}
                  for b in j.get("parsing_res_list", []) if b.get("block_label") not in PADDLE_IGNORE]
        blocks = [b for b in blocks if b["text"]]
        text = "\n".join(page_lines({"blocks": blocks}))
        boxes = [{"label": b["label"], "box": [round(x) for x in b["coordinate"]], "order": b.get("order"),
                  "score": round(b.get("score", 0), 2)}
                 for b in (j.get("layout_det_res") or {}).get("boxes", [])]
        return text, {"blocks": blocks, "boxes": boxes, "quality_issues": ocr_quality_issues(text, {})}


class PPOCR:
    """PP-OCRv6 medium, retaining genuine single-character word boxes only."""

    def __init__(self, a):
        from paddleocr import PaddleOCR
        options = dict(ocr_version="PP-OCRv6", text_detection_model_name="PP-OCRv6_medium_det",
                       text_recognition_model_name="PP-OCRv6_medium_rec",
                       use_doc_orientation_classify=False, use_doc_unwarping=False,
                       use_textline_orientation=False)
        options.update(json.loads(a.paddle_args))
        # Coordinates must remain in the original input-image frame.
        for key in ("use_doc_orientation_classify", "use_doc_unwarping", "use_textline_orientation"):
            if options.get(key):
                raise ValueError(f"{key} must be False for source-image character coordinates")
        self.pipe = PaddleOCR(**options)

    def run(self, img_path):
        kwargs = {"return_word_box": True}
        if getattr(self, "preserve_input_resolution", False):
            from PIL import Image
            with Image.open(img_path) as image:
                # Request detector input at the crop resolution, aligned to its
                # stride. Recognition still has its own trained line geometry.
                kwargs.update(text_det_limit_side_len=((max(image.size)+31)//32)*32,
                              text_det_limit_type="max")
        results = list(self.pipe.predict(img_path, **kwargs))
        if len(results) != 1:
            raise ValueError("Expected exactly one PP-OCR page result")
        text, extra = self.parse(results[0].json)
        return text, {**extra, "model_input": {"requested": kwargs,
                       "observation": "native_internal_unavailable"}}

    @staticmethod
    def parse(j):
        from .text_alignment import comparison_text
        j = j.get("res", j)
        texts = j.get("rec_texts", [])
        polygons = j.get("rec_polys", [])
        words = j.get("text_word", [])
        regions = j.get("text_word_region", [])
        rectangles = j.get("text_word_boxes", [])
        characters, lines, offset = [], [], 0
        for index, text in enumerate(texts):
            filtered = comparison_text(text)
            line_words = words[index] if index < len(words) else []
            line_boxes = (regions[index] if index < len(regions) else
                          rectangles[index] if index < len(rectangles) else [])
            # Never divide a line/word rectangle evenly to invent character boxes.
            if len(line_words) == len(line_boxes) and comparison_text("".join(line_words)) == filtered:
                local_offset = 0
                for word, polygon in zip(line_words, line_boxes):
                    word = comparison_text(word)
                    if len(word) == 1:
                        characters.append({"offset": offset + local_offset, "text": word,
                                           "polygon": polygon, "granularity": "character",
                                           "method": "ppocr_word_box", "estimated": True})
                    local_offset += len(word)
            lines.append({"text": text, "polygon": polygons[index] if index < len(polygons) else None})
            offset += len(filtered)
        return "\n".join(texts), {"character_boxes": characters, "text_lines": lines,
                                   "coordinate_frame": "input_image", "hit_max": False}


class TongGu:
    """Use the official TongGu layout/block pipeline in its configured environment."""

    def __init__(self, a):
        import sys
        root = Path(a.tonggu_repo).expanduser().resolve() if a.tonggu_repo else None
        if root is None or not (root / "tongguocr" / "pipeline.py").is_file():
            raise ValueError("Set tonggu_repo to the official TongGuOCR checkout with installed dependencies/weights")
        sys.path.insert(0, str(root))
        from tongguocr.config import load_config
        from tongguocr.pipeline import run_pipeline_on_image
        self.predict = run_pipeline_on_image
        self.config = load_config(a.tonggu_config or None)
        self.default_image_limits = {key: self.config["ocr"]["inference"].get(key, default)
                                     for key, default in (("image_min_pixels", 1024), ("image_max_pixels", 1048576))}
        self.model_min_pixels = getattr(a, "model_min_pixels", None)
        self.model_max_pixels = getattr(a, "model_max_pixels", None)
        for group, key in (("layout_inference", "detector_checkpoint_path"),
                           ("layout_inference", "detector_config_path")):
            path = Path(self.config[group][key])
            if not path.is_absolute():
                self.config[group][key] = str(root / path)
        model_path = Path(self.config["ocr"]["transformers"]["model_path"])
        if not model_path.is_absolute():
            self.config["ocr"]["transformers"]["model_path"] = str(root / model_path)
        self.max_new = a.max_new
        self.temporary = tempfile.TemporaryDirectory(prefix="vtm-tonggu-")
        self.prepare_models()

    def prepare_models(self):
        """Fill the pinned upstream caches before the per-page deadline starts."""
        from tongguocr.layout.extractor import LineBBoxExtractor, _EXTRACTOR_CACHE
        started = time.monotonic()
        print("[model] TongGu: bắt đầu nạp detector layout", flush=True)
        layout = self.config["layout_inference"]
        key = "|".join(str(layout[field]) for field in (
            "detector_config_path", "detector_checkpoint_path", "device", "detector_score_threshold"))
        if key not in _EXTRACTOR_CACHE:
            _EXTRACTOR_CACHE[key] = LineBBoxExtractor(self.config)
        if self.config["ocr"]["backend"] == "transformers":
            from tongguocr.ocr.transformers_backend import _load_transformers_runtime
            print("[model] TongGu: bắt đầu nạp model OCR", flush=True)
            backend = self.config["ocr"]["transformers"]
            _load_transformers_runtime(backend["model_path"], backend.get("device", "cuda:0"))
        print(f"[model] TongGu: model sẵn sàng, sau {time.monotonic() - started:.1f}s; "
              "thời gian nạp không tính vào timeout trang.", flush=True)

    def run(self, img_path):
        self.config["ocr"]["inference"]["max_tokens"] = self.max_new
        for key, value in (("image_min_pixels", getattr(self, "model_min_pixels", None)),
                           ("image_max_pixels", getattr(self, "model_max_pixels", None))):
            if value is not None or hasattr(self, "default_image_limits"):
                self.config["ocr"]["inference"][key] = value if value is not None else self.default_image_limits[key]
        # Unique page directory prevents stale upstream layout/crop artifacts.
        output = Path(self.temporary.name) / str(time.time_ns())
        result = self.predict(img_path, output, self.config, visualize=False, crop_blocks=True)
        if result.status != "ok":
            raise RuntimeError(f"TongGuOCR returned status={result.status}")
        # Upstream joins recognition by block_id. These are coarse block contexts;
        # sparse canvases do not establish positions of individual recognized glyphs.
        recognized = {b.block_id: b.ocr_text for b in getattr(result, 'ocr_blocks', [])}
        spatial = [{"box": b['bbox'], "text": recognized[b['block_id']],
                    "granularity": "block", "method": "tonggu_recognized_block",
                    "sparse_canvas_applied": b.get('sparse_canvas_applied', False)}
                   for b in result.blocks if b.get('block_id') in recognized and recognized[b['block_id']]]
        return result.ocr_text, {"tonggu_layout": result.line_bboxes, "tonggu_blocks": result.blocks,
                                 "spatial_regions": spatial, "coordinate_frame": "input_image",
                                 "hit_max": None, "character_boxes": [],
                                 "model_input": {"requested": {key: self.config["ocr"]["inference"].get(key)
                                     for key in ("image_min_pixels", "image_max_pixels")},
                                     "observation": "native_internal_unavailable"},
                                 "quality_issues": ocr_quality_issues(result.ocr_text, {})}


ENGINES = {"hunyuan": Hunyuan, "paddle": Paddle, "ppocr": PPOCR, "tonggu": TongGu, "dummy": Dummy}

# The supervisor restarts a Windows worker after a hard page timeout.
PAGE_TIMEOUT_EXIT = 124


@contextmanager
def page_deadline(seconds: int, on_hard_timeout):
    """Unix: interrupt the page; Windows: stop the worker so CUDA cannot keep running."""
    if seconds <= 0:
        yield
        return
    if hasattr(signal, "SIGALRM"):
        def expire(signum, frame):
            raise TimeoutError(f"quá {seconds}s")

        previous = signal.signal(signal.SIGALRM, expire)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
    else:
        def expire():
            try:
                on_hard_timeout()
            finally:
                os._exit(PAGE_TIMEOUT_EXIT)

        timer = threading.Timer(seconds, expire)
        timer.daemon = True
        timer.start()
        try:
            yield
        finally:
            timer.cancel()
            timer.join()


def result_file(out: Path, book: str, name: str, shard: int) -> Path:
    return out / book / f"{name}_shard{shard}.jsonl"


def load_results(out: Path, book: str, name: str) -> dict[int, dict]:
    """{trang: bản ghi} của 1 sách, gộp mọi shard (dòng hỏng thì bỏ)."""
    recs = {}
    for f in sorted((out / book).glob(f"{name}_shard*.jsonl")):
        with open(f, encoding="utf-8") as stream:
            for line in stream:
                try:
                    r = json.loads(line)
                    recs[r["page"]] = r
                except (json.JSONDecodeError, KeyError):
                    pass
    return recs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=list(ENGINES))
    ap.add_argument("--name", default=None, help="tên file kết quả (mặc định = engine)")
    ap.add_argument("--tasks", required=True, help='jsonl, mỗi dòng {"book", "src", "page"}')
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--image-config", default="{}", help="JSON segmentation/resolution settings from OCRConfig")
    ap.add_argument("--model-min-pixels", type=int)
    ap.add_argument("--model-max-pixels", type=int)
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--dtype", default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--model", default="tencent/HunyuanOCR")
    ap.add_argument("--model-revision", default="")
    ap.add_argument("--tonggu-repo", default="")
    ap.add_argument("--tonggu-config", default="")
    ap.add_argument("--prompt", default="", help="hunyuan: prompt OCR thay cho prompt mặc định")
    ap.add_argument("--rep-penalty", type=float, default=1.08)
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:1 (hunyuan)")
    ap.add_argument("--paddle-version", default="v1.6")
    ap.add_argument("--paddle-args", default="{}", help="json tham số thêm cho PaddleOCRVL")
    ap.add_argument("--limit", type=int, default=None, help="chỉ chạy N trang (chạy thử)")
    ap.add_argument("--page-timeout", type=int, default=300, help="giây tối đa/trang, quá thì bỏ trang đó")
    a = ap.parse_args(argv)
    image_cfg = OCRConfig(engine=a.engine, dpi=a.dpi, max_side=a.max_side,
                          model_min_pixels=a.model_min_pixels, model_max_pixels=a.model_max_pixels,
                          **json.loads(a.image_config))
    name, out = a.name or a.engine, Path(a.out)

    with open(a.tasks, encoding="utf-8") as stream:
        tasks = [json.loads(l) for l in stream if l.strip()]
    mine = [t for i, t in enumerate(tasks) if i % a.nshard == a.shard]
    done, skip = set(), set()
    for book in {t["book"] for t in mine}:
        done |= {(book, p) for p in load_results(out, book, name)}
        for f in (out / book).glob(f"{name}_skip*.log"):
            skip |= {(book, int(p)) for p in f.read_text().split()}
    # trang đang chạy dở khi worker bị giết vì treo -> ghi vào skip, không thử lại
    inflight = out / f"_inflight_{name}_{a.shard}"
    if inflight.exists():
        book, _, p = inflight.read_text(encoding="utf-8").rpartition("|")
        if book and (book, int(p)) not in done:
            with open(out / book / f"{name}_skip.log", "a") as f:
                f.write(p + "\n")
            skip.add((book, int(p)))
            print(f"[{name}#{a.shard}] {book} trang {int(p) + 1}: bị treo ở lần trước -> bỏ qua", flush=True)
        inflight.unlink()
    seen = done | skip
    todo = [t for t in mine if (t["book"], t["page"]) not in seen][:a.limit]
    print(f"[{name}#{a.shard}] {len(mine)} trang, đã xong {len(mine) - len(todo)}, chạy {len(todo)}", flush=True)
    if not todo:
        return

    engine = ENGINES[a.engine](a)
    engine.preserve_input_resolution = image_cfg.segmentation != "none" or bool(image_cfg.image_dataset)
    prepared_source_hashes = {}
    tmp = Path(tempfile.mkdtemp()) / "page.png"

    t_start = time.time()
    for n, t in enumerate(todo, 1):
        book, page, t0 = t["book"], t["page"], time.time()
        (out / book).mkdir(parents=True, exist_ok=True)
        inflight.write_text(f"{book}|{page}", encoding="utf-8")

        def hard_timeout():
            with open(out / book / f"{name}_skip.log", "a", encoding="utf-8") as f:
                f.write(f"{page}\n")
            inflight.unlink(missing_ok=True)
            print(f"[{name}#{a.shard}] {book} tr.{page + 1}: quá {a.page_timeout}s -> bỏ qua, khởi động lại worker",
                  flush=True)

        try:
            with page_deadline(a.page_timeout, hard_timeout):
                prepared = None
                if image_cfg.image_dataset:
                    if t["src"] not in prepared_source_hashes:
                        prepared_source_hashes[t["src"]] = dataset_file_hash(t["src"])
                    prepared = dataset_ocr_page(image_cfg, prepared_source_hashes[t["src"]], page)
                if prepared is None:
                    img = render(t["src"], page, a.dpi, a.max_side)
                    img.save(tmp)
                    input_path = tmp
                else:
                    img, input_path = prepared
                text, extra = run_prepared_page(engine, img, input_path, image_cfg, page=page)
            rec = {"page": page, "text": text, "w": img.width, "h": img.height,
                   "sec": round(time.time() - t0, 1), **extra,
                   "image_input": "prepared" if prepared else "legacy_missing_page_fallback" if image_cfg.image_dataset else "legacy"}
            with open(result_file(out, book, name, a.shard), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            eta = (time.time() - t_start) / n * (len(todo) - n) / 60
            print(f"[{name}#{a.shard}] {n}/{len(todo)} {book} tr.{page + 1}: {rec['sec']}s, "
                  f"{len(text)} ký tự | còn ~{eta:.0f} phút", flush=True)
        except Exception as e:  # noqa: BLE001 - 1 trang lỗi không làm dừng cả lượt
            if isinstance(e, TimeoutError):
                with open(out / book / f"{name}_skip.log", "a") as f:
                    f.write(f"{page}\n")
            print(f"[{name}#{a.shard}] {book} tr.{page + 1}: LỖI {type(e).__name__}: {str(e)[:200]}", flush=True)
    inflight.unlink(missing_ok=True)
    for d in _DOCS.values():  # đóng PDF, khỏi in cảnh báo pypdfium2 vào log
        d.close()


if __name__ == "__main__":
    main()
