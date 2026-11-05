"""OCR từng trang PDF (HunyuanOCR, PaddleOCR-VL hoặc dummy) trên 1 tiến trình.

Chạy lại thì bỏ qua trang đã xong. File này được nhúng nguyên văn vào notebook
Kaggle qua %%writefile, nên chỉ dùng thư viện chuẩn ở cấp module và import thư
viện nặng một cách lazy.
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import tempfile
import time
from pathlib import Path

HUNYUAN_PROMPT = ("提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
                  "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。")
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
    ordered = sorted((b for b in rec["blocks"] if b["order"] is not None), key=lambda b: b["order"])
    center = lambda b: ((b["box"][0] + b["box"][2]) / 2, (b["box"][1] + b["box"][3]) / 2)  # noqa: E731
    for b in (b for b in rec["blocks"] if b["order"] is None):  # bảng, chú thích hình: paddle không xếp thứ tự đọc
        if not ordered:
            ordered.append(b)
            continue
        cx, cy = center(b)
        k = min(range(len(ordered)), key=lambda i: (center(ordered[i])[0] - cx) ** 2 + (center(ordered[i])[1] - cy) ** 2)
        ordered.insert(k + (cy > center(ordered[k])[1]), b)  # đặt cạnh vùng gần nhất
    return [ln for b in ordered for ln in md_to_lines(b["text"])]


_DOCS = {}


def render(src: str, page: int, dpi: int = 200, max_side: int = 1600,
           swinir: bool = False, swinir_cache: str = "work/preprocessed"):
    """Trang PDF -> ảnh PIL (cạnh dài tối đa max_side)."""
    import pypdfium2 as pdfium
    if swinir:
        from .preprocess import cached_page
        return cached_page(src, page, dpi, max_side, swinir_cache)
    if src not in _DOCS:
        _DOCS[src] = pdfium.PdfDocument(src)
    img = _DOCS[src][page].render(scale=dpi / 72).to_pil().convert("RGB")
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)))
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
        self.dev = self._device(a.device)
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[a.dtype]
        print(f"[model] Hunyuan: bắt đầu tải/đọc model {a.model} (dùng cache nếu có)", flush=True)
        self.proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True, use_fast=False)
        self.model = M.from_pretrained(a.model, dtype=dtype, trust_remote_code=True).to(self.dev).eval()
        print(f"[model] Hunyuan: model sẵn sàng trên {self.dev}, sau {time.monotonic() - started:.1f}s", flush=True)

    @staticmethod
    def _device(requested: str) -> str:
        import torch
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return requested

    def run(self, img_path):
        torch = self.torch
        msgs = [{"role": "user", "content": [{"type": "image", "image": img_path},
                                             {"type": "text", "text": self.prompt}]}]
        inputs = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt").to(self.dev)
        for k, v in list(inputs.items()):
            if torch.is_tensor(v) and v.is_floating_point():
                inputs[k] = v.to(self.model.dtype)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new, do_sample=False,
                                      repetition_penalty=self.rep)
        gen = out[:, inputs["input_ids"].shape[1]:]
        text = self.proc.batch_decode(gen, skip_special_tokens=True)[0]
        return clean_degenerate(text), {"hit_max": int(gen.shape[1]) >= self.max_new}


class Paddle:
    def __init__(self, a):
        started = time.monotonic()
        print("[model] Paddle: bắt đầu import paddleocr", flush=True)
        from paddleocr import PaddleOCRVL
        print(f"[model] Paddle: bắt đầu tải/khởi tạo PaddleOCR-VL {a.paddle_version} (dùng cache nếu có)", flush=True)
        self.pipe = PaddleOCRVL(pipeline_version=a.paddle_version, **json.loads(a.paddle_args))
        self.max_new = a.max_new
        print(f"[model] Paddle: model sẵn sàng, sau {time.monotonic() - started:.1f}s", flush=True)

    def run(self, img_path):
        try:
            res = list(self.pipe.predict(img_path, max_new_tokens=self.max_new))[0]
        except TypeError:  # bản paddleocr cũ chưa nhận max_new_tokens
            res = list(self.pipe.predict(img_path))[0]
        return self.parse(res.json)

    @staticmethod
    def parse(j):
        """res.json của PaddleOCR-VL -> (text, {"blocks": vùng đã đọc chữ, "boxes": mọi khung layout})."""
        j = j.get("res", j) if isinstance(j, dict) else {}
        blocks = [{"label": b.get("block_label"), "box": [round(x) for x in b.get("block_bbox", [0, 0, 0, 0])],
                   "order": b.get("block_order"), "text": clean_degenerate(b.get("block_content") or "")}
                  for b in j.get("parsing_res_list", []) if b.get("block_label") not in PADDLE_IGNORE]
        blocks = [b for b in blocks if b["text"]]
        text = "\n".join(page_lines({"blocks": blocks}))
        boxes = [{"label": b["label"], "box": [round(x) for x in b["coordinate"]], "order": b.get("order"),
                  "score": round(b.get("score", 0), 2)}
                 for b in (j.get("layout_det_res") or {}).get("boxes", [])]
        return text, {"blocks": blocks, "boxes": boxes}


ENGINES = {"hunyuan": Hunyuan, "paddle": Paddle, "dummy": Dummy}


def result_file(out: Path, book: str, name: str, shard: int) -> Path:
    return out / book / f"{name}_shard{shard}.jsonl"


def load_results(out: Path, book: str, name: str) -> dict[int, dict]:
    """{trang: bản ghi} của 1 sách, gộp mọi shard (dòng hỏng thì bỏ)."""
    recs = {}
    for f in sorted((out / book).glob(f"{name}_shard*.jsonl")):
        for line in open(f, encoding="utf-8"):
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
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--swinir", action="store_true")
    ap.add_argument("--swinir-cache", default="work/preprocessed")
    ap.add_argument("--max-new", type=int, default=6000)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--model", default="tencent/HunyuanOCR")
    ap.add_argument("--prompt", default="", help="hunyuan: prompt OCR thay cho prompt mặc định")
    ap.add_argument("--rep-penalty", type=float, default=1.08)
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:1 (hunyuan)")
    ap.add_argument("--paddle-version", default="v1.6")
    ap.add_argument("--paddle-args", default="{}", help="json tham số thêm cho PaddleOCRVL")
    ap.add_argument("--limit", type=int, default=None, help="chỉ chạy N trang (chạy thử)")
    ap.add_argument("--page-timeout", type=int, default=300, help="giây tối đa/trang, quá thì bỏ trang đó")
    a = ap.parse_args(argv)
    name, out = a.name or a.engine, Path(a.out)

    tasks = [json.loads(l) for l in open(a.tasks, encoding="utf-8") if l.strip()]
    mine = [t for i, t in enumerate(tasks) if i % a.nshard == a.shard]
    done, skip = set(), set()
    for book in {t["book"] for t in mine}:
        done |= {(book, p) for p in load_results(out, book, name)}
        for f in (out / book).glob(f"{name}_skip*.log"):
            skip |= {(book, int(p)) for p in f.read_text().split()}
    # trang đang chạy dở khi worker bị giết vì treo -> ghi vào skip, không thử lại
    inflight = out / f"_inflight_{name}_{a.shard}"
    if inflight.exists():
        book, _, p = inflight.read_text().rpartition("|")
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

    if a.swinir:
        from .restoration import resolve_assets
        resolve_assets()
    engine = ENGINES[a.engine](a)
    tmp = Path(tempfile.mkdtemp()) / "page.png"

    def on_timeout(signum, frame):
        raise TimeoutError(f"quá {a.page_timeout}s")
    signal.signal(signal.SIGALRM, on_timeout)

    t_start = time.time()
    for n, t in enumerate(todo, 1):
        book, page, t0 = t["book"], t["page"], time.time()
        (out / book).mkdir(parents=True, exist_ok=True)
        inflight.write_text(f"{book}|{page}")
        signal.alarm(a.page_timeout)
        try:
            img = render(t["src"], page, a.dpi, a.max_side, a.swinir,
                         a.swinir_cache)
            img.save(tmp)
            text, extra = engine.run(str(tmp))
            signal.alarm(0)
            rec = {"page": page, "text": text, "w": img.width, "h": img.height,
                   "sec": round(time.time() - t0, 1), **extra}
            with open(result_file(out, book, name, a.shard), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            eta = (time.time() - t_start) / n * (len(todo) - n) / 60
            print(f"[{name}#{a.shard}] {n}/{len(todo)} {book} tr.{page + 1}: {rec['sec']}s, "
                  f"{len(text)} ký tự | còn ~{eta:.0f} phút", flush=True)
        except Exception as e:  # noqa: BLE001 - 1 trang lỗi không làm dừng cả lượt
            signal.alarm(0)
            if isinstance(e, TimeoutError):
                with open(out / book / f"{name}_skip.log", "a") as f:
                    f.write(f"{page}\n")
            print(f"[{name}#{a.shard}] {book} tr.{page + 1}: LỖI {type(e).__name__}: {str(e)[:200]}", flush=True)
    inflight.unlink(missing_ok=True)
    for d in _DOCS.values():  # đóng PDF, khỏi in cảnh báo pypdfium2 vào log
        d.close()


if __name__ == "__main__":
    main()
