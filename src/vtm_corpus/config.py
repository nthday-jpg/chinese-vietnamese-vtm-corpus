"""Cấu hình cho pipeline OCR và dịch.

Notebook chỉ cần dựng các dataclass này với mọi tham số muốn đổi; phần logic
nằm trong package. Mọi trường đều có thể override từ notebook/CLI.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HUNYUAN_PROMPT = ("提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
                  "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。")
LAYOUT_PROMPT = "按照阅读顺序解析图中的版式信息。"
SYSTEM_PROMPT = ("Bạn là dịch giả chuyên dịch sách Y học cổ truyền từ chữ Hán (văn ngôn và bạch thoại) "
                 "sang tiếng Việt.")
USER_PROMPT = ("Văn bản chữ Hán:\n{text}\n\n"
               "Dịch toàn bộ văn bản trên sang tiếng Việt, đầy đủ ý, văn phong tự nhiên. Tên huyệt, vị thuốc, "
               "bài thuốc, bệnh danh giữ theo âm Hán-Việt (vd: 足三里 -> Túc tam lý). Chỉ viết tiếng Việt: "
               "không chép lại chữ Hán, không dịch sang tiếng Anh, không lời dẫn, không giải thích.")


def _json_default(o: Any):
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def config_hash(cfg: Any, exclude: set[str]) -> str:
    data = {k: v for k, v in dataclasses.asdict(cfg).items() if k not in exclude}
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True, default=_json_default)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


@dataclass
class OCRConfig:
    """Cấu hình bước OCR. Mọi trường đều có thể sửa từ notebook."""

    engine: str = "hunyuan"                 # hunyuan | paddle | dummy
    input_dir: str = "data"                 # thư mục chứa PDF (glob đệ quy)
    books: list[str] = field(default_factory=list)  # [] = tất cả; tên hoặc glob, vd ["*ky_huyet*"]
    workdir: str = "work"                   # thư mục gốc chứa runs/
    run_name: str = ""                      # "" = dùng engine
    pages: str = ""                         # "" = cả sách; "1-10,15" = các trang chỉ định
    dpi: int = 200
    max_side: int = 1280
    max_new_tokens: int = 4096
    dtype: str = "fp16"                     # fp32 | fp16 | bf16 (hunyuan)
    model: str = "tencent/HunyuanOCR"
    prompt: str = ""                        # "" = HUNYUAN_PROMPT
    repetition_penalty: float = 1.08
    device: str = "auto"                    # auto | cpu | cuda | cuda:1
    paddle_version: str = "v1.6"            # v1.6 | v1.5
    paddle_args: dict = field(default_factory=dict)
    limit: int | None = None                # giới hạn tổng số trang (chạy thử)
    page_timeout: int = 300                 # giây/trang
    n_gpu: int = 0                          # 0 = tự phát hiện
    worker_python: str = ""                 # "" = sys.executable; Paddle local có thể là venv
    budget_h: float = 10.5
    stall_min: int = 15
    max_restarts: int = 5
    n_layout: int = 0                       # bật khi cần; layout Hunyuan tải model thêm một lượt
    layout_budget_h: float = 1.0            # ngân sách riêng cho lượt layout
    layout_prompt: str = LAYOUT_PROMPT
    export: bool = True                     # ghi txt/<book>.txt sau khi chạy
    fresh: bool = False                     # xoá run dir trước khi chạy
    resume_from: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.engine not in ("hunyuan", "paddle", "dummy"):
            raise ValueError(f"engine không hợp lệ: {self.engine}")
        if self.dtype not in ("fp32", "fp16", "bf16"):
            raise ValueError(f"dtype không hợp lệ: {self.dtype}")
        if self.dpi <= 0 or self.max_side <= 0 or self.max_new_tokens <= 0:
            raise ValueError("dpi, max_side, max_new_tokens phải > 0")

    def cache_key(self) -> str:
        return config_hash(self, {
            "input_dir", "workdir", "run_name", "limit", "page_timeout", "n_gpu",
            "worker_python", "budget_h", "stall_min", "max_restarts", "n_layout",
            "layout_budget_h", "export", "fresh", "resume_from",
        })

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def load_ocr_config(path: str | Path, *, base: OCRConfig | None = None) -> OCRConfig:
    """Load a JSON OCR config over a copy of the notebook/runtime configuration.

    Fields omitted from the JSON (engine, input_dir, workdir, worker_python,
    pages, etc.) keep the values from base. The original base is not modified.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: OCR config must be a JSON object")
    known = {f.name for f in dataclasses.fields(OCRConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{path}: unknown OCR config fields: {sorted(unknown)}")
    if "books" in data and (not isinstance(data["books"], list) or
                            any(not isinstance(book, str) or not book for book in data["books"])):
        raise ValueError(f"{path}: books must be a list of nonempty strings")
    # Copy nested values too: editing one loaded config must not affect another.
    values = (base or OCRConfig()).to_dict()
    values.update(data)
    return OCRConfig(**values)


@dataclass
class TranslateConfig:
    """Cấu hình bước dịch. Mọi trường đều có thể sửa từ notebook."""

    engine: str = "hf"                      # hf | dummy
    model: str = "Qwen/Qwen2.5-3B-Instruct"
    input_dir: str = "work/runs"            # thư mục chứa .txt (glob đệ quy)
    books: list[str] = field(default_factory=list)  # [] = tất cả; tên hoặc glob
    workdir: str = "work"
    run_name: str = ""                      # "" = tên model
    gpu_per_model: int = 1                  # mặc định cho 1 GPU 8 GB; 2: model 7-8B trên Kaggle
    load_4bit: bool = False
    batch: int = 1
    chunk_chars: int = 300
    max_new_tokens: int = 1200
    temperature: float = 0.0
    repetition_penalty: float = 1.05
    limit: int | None = None                # giới hạn tổng số đoạn (chạy thử)
    budget_h: float = 10.5
    stall_min: int = 15
    max_restarts: int = 5
    system: str = SYSTEM_PROMPT
    user: str = USER_PROMPT
    min_han: int = 50                       # bỏ file có ít hơn N chữ Hán
    device: str = "auto"                    # auto | cpu | cuda
    n_gpu: int = 0                          # 0 = tự phát hiện
    worker_python: str = ""
    resume_from: list[str] = field(default_factory=list)
    export: bool = True
    fresh: bool = False

    def __post_init__(self) -> None:
        if self.engine not in ("hf", "dummy"):
            raise ValueError(f"engine không hợp lệ: {self.engine}")
        if self.gpu_per_model < 1:
            raise ValueError("gpu_per_model phải >= 1")

    def cache_key(self) -> str:
        return config_hash(self, {
            "input_dir", "workdir", "run_name", "limit", "budget_h", "stall_min",
            "max_restarts", "n_gpu", "gpu_per_model", "worker_python", "resume_from",
            "export", "fresh",
        })

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class ReportConfig:
    """Cấu hình hiển thị kết quả ngay trong notebook."""

    preview_pages: int = 3                  # số trang text hiển thị mỗi sách
    preview_chars: int = 1500
    layout_images: int = 6
    show_tables: bool = True
    image_width: int = 700

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
