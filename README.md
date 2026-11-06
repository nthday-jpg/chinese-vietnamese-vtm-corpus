# Đề tài: Dịch sách Y học cổ truyền từ chữ Hán sang tiếng Việt

```
PDF sách Hán (scan) ──► Bước 1: OCR ──► .txt chữ Hán ──► Bước 2: LLM dịch ──► .txt tiếng Việt
```

## Bạn nhận được
| | |
|---|---|
| `data/` | PDF sách Hán (bản scan, không lấy chữ trực tiếp được) |
| `1_ocr_hunyuan.ipynb` | Bước 1 bằng **HunyuanOCR** (Tencent) |
| `1_ocr_paddle.ipynb` | Bước 1 bằng **PaddleOCR-VL** (Baidu) |
| `2_dich_han_viet.ipynb` | Bước 2: dịch bằng một LLM nhỏ tuỳ chọn |
| `3_so_sanh_so_chu_ocr.ipynb` | Chạy lần lượt Hunyuan/Paddle rồi so số chữ bằng một bảng ngắn |
| `PROBLEM.md` | Mô tả bài toán: input / output của từng bước |
| `FLOW.md` | Giải thích từng bước notebook làm gì + gợi ý các cách làm khác |

Bước 1 chọn **1 trong 2** notebook; chạy cả 2 rồi so sánh chất lượng OCR thì càng tốt.
Notebook chỉ là cách làm mẫu — bạn được tự đề xuất cách khác (xem cuối `FLOW.md`).

## Cấu trúc repo

```
src/vtm_corpus/          # toàn bộ logic: OCR, dịch, chia việc, report, CLI
  config.py              # OCRConfig / TranslateConfig / ReportConfig
  pipeline.py            # tìm input, chạy worker, resume, ghép output, manifest
  ocr_worker.py          # worker OCR (1 GPU / 1 tiến trình)
  dich_worker.py         # worker dịch (1 model / 1 tiến trình)
  report.py              # bảng tổng hợp + hiển thị trong notebook
  cli.py                 # lệnh `vtm`
1_ocr_*.ipynb            # notebook mỏng: cấu hình + chạy + xem kết quả
2_dich_han_viet.ipynb
pyproject.toml
```

Notebook **không còn chứa logic**; mọi tham số nằm trong ô cấu hình của notebook và
được truyền vào package. Khi sửa logic, push repo rồi `git pull` trong notebook Kaggle.
Mỗi run ghi `run_manifest.json` (toàn bộ cấu hình đã dùng), và thư mục
`runs/<loại>_<tên>_<config-hash>/` nên không bị lẫn kết quả giữa các cấu hình khác nhau.

Kết quả nằm trong `runs/ocr_*/txt/`, `runs/ocr_*/layout/`, `runs/dich_*/dich/` và
được hiển thị trực tiếp trong notebook ở các ô report — không cần tải file về.

## So sánh tự động số lượng chữ của hai model OCR

Mở [`3_so_sanh_so_chu_ocr.ipynb`](3_so_sanh_so_chu_ocr.ipynb), chọn kernel `.venv`
và **Run All**. Notebook tự tải Hunyuan, OCR phần dữ liệu đã chọn, giải phóng model,
rồi tải Paddle và OCR đúng cùng dữ liệu, sau đó mới so sánh. Cần GPU/model như các
notebook OCR gốc. Nếu thiếu dependency, chạy `uv sync --all-extras` rồi restart kernel.

- Mặc định chỉ thử `1.pdf`, trang 1–2. Chỉnh `BOOKS`, `MAX_BOOKS`, `PAGES`, `LIMIT`
  trong ô cấu hình: `MAX_BOOKS` giới hạn số file sau khi lọc tên, `LIMIT` giới hạn
  tổng số trang trên toàn bộ file được chọn. Muốn toàn bộ data: `BOOKS=[]`,
  `MAX_BOOKS=None`, `PAGES=""`, `LIMIT=None`.
- Các argument riêng nằm trong hai `OCRConfig` là `HUNYUAN` và `PADDLE`.
  `USE_BOOK_PROFILES=True` nạp JSON riêng của từng PDF; đặt `False` để dùng trực tiếp
  DPI/`max_side`/giới hạn token trong notebook. Trước khi chạy, notebook in tổng số
  file và số trang sẽ xử lý.
- Mỗi model được tải một lần để xử lý mọi sách của stage, trừ khi phải khởi động
  lại sau timeout. Hai stage chạy chờ hoàn tất ngay trong notebook; không có runner
  nền riêng. Kết quả lưu sau từng trang, chạy lại cùng cấu hình sẽ tiếp tục phần còn thiếu.
- Model chạy trong tiến trình riêng để giải phóng VRAM khi kết thúc. Interrupt ô
  đang chạy sẽ dừng worker của ô đó. Đổi `WORK_DIR` để làm thử nghiệm mới.
- Ghép theo tên sách và số trang PDF; phân biệt trang thiếu kết quả, trang không được
  chọn chạy và trang có 0 chữ. Loại lượt layout riêng và cache ngoài tập trang chọn.
- Mặc định đếm chữ Hán sau khi bỏ định dạng Markdown/HTML. Có thể đổi sang mọi ký tự
  không trắng bằng `COUNT_MODE="visible"`.
- Chỉ hiển thị một bảng theo sách: số trang đã OCR của hai model, số trang ghép cặp,
  tổng chữ trên phần ghép cặp và độ khớp số lượng. Có dòng tổng nếu chọn nhiều sách.
  Không vẽ biểu đồ hoặc xuất báo cáo HTML/CSV/PNG. Checkpoint OCR vẫn được lưu để tiếp tục.

CLI dùng cùng pipeline, mặc định cũng chỉ 1 PDF/2 trang:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/run_full_corpus_ocr.py
# Chỉ lập kế hoạch, không tải model:
.\.venv\Scripts\python.exe -X utf8 scripts/run_full_corpus_ocr.py --books 1 2 --pages 1-2 --limit 2 --prepare-only
# Chạy mọi PDF/mọi trang khi chủ động chọn:
.\.venv\Scripts\python.exe -X utf8 scripts/run_full_corpus_ocr.py --all
```

Độ khớp số lượng mỗi trang là `min(H,P)/max(H,P)`; điểm tổng là
`Σmin(Hᵢ,Pᵢ)/Σmax(Hᵢ,Pᵢ)` trên các trang có output cả hai. Hai bên cùng 0 chữ không
có điểm; thiếu output không được thay bằng 0. Nếu còn PDF nguồn thì so SHA-256 để
tránh ghép nhầm file; khi không truy cập được PDF, notebook cảnh báo chưa xác minh nguồn.
**Đây không phải độ chính xác OCR**: hai bản có cùng số chữ vẫn có thể sai nội dung.
Các bộ lọc bỏ tiêu đề/chú thích khác nhau cũng có thể gây chênh lệch số lượng.

## Chọn tập sách (mỗi sách có thể cần cấu hình khác nhau)

Mỗi sách có thể có vấn đề riêng (chữ dọc, trang nghiêng, bảng, mực mờ…), nên pipeline
cho phép chọn một tập con thay vì luôn chạy tất cả. Ô đầu của notebook khai báo
`INPUT_DIR = "/kaggle/input"`; ô liệt kê và ô cấu hình đều dùng chung biến này, nên
chỉ cần sửa một chỗ khi input nằm ở đường dẫn khác. Ô liệt kê input:

```python
from vtm_corpus import list_pdfs, list_texts

list_pdfs(INPUT_DIR)                  # OCR: tên sách + số trang
list_texts(INPUT_DIR, min_han=50)     # dịch: tên sách + số chữ Hán
```

Rồi đặt `books` trong `OCRConfig` / `TranslateConfig`:

```python
books=[]                               # tất cả sách
books=["Vệ_sinh_yếu_chỉ_-_tập_4"]      # đúng 1 sách
books=["*ky_huyet*", "*nhi*"]          # nhiều sách theo glob
```

Tên dùng trong `books` là `name` do `list_pdfs`/`list_texts` in ra (đã làm sạch dấu
cách/ký tự đặc biệt). Pattern không khớp sách nào sẽ báo lỗi kèm danh sách tên có sẵn.
`books` nằm trong config hash, nên chạy sách khác sẽ tạo run riêng.

Trên CLI:

```bash
vtm list --input-dir data --kind pdf
vtm list --input-dir work/runs/ocr_xxx/txt --kind txt --min-han 50
vtm ocr --set input_dir=data --set 'books=["*ky_huyet*"]'
```

## Chạy local Windows (Python 3.13, RTX 4060 Laptop 8 GB)

Cấu hình local được đặt cho máy i7-13650HX, khoảng 24 GB RAM và 1 GPU RTX 4060
Laptop 8 GB. `.python-version` chọn Python 3.13. Dùng chung **một `.venv`** cho
CLI và cả ba notebook, kể cả Paddle GPU:

```powershell
uv sync --python 3.13 --all-extras
.\.venv\Scripts\Activate.ps1
vtm list --input-dir data --kind pdf

# OCR thật 2 trang đầu của data/1.pdf
vtm ocr --config configs/ocr_local.json
```

TOML khai báo các kho wheel chính thức CUDA 12.6 cho PyTorch và Paddle GPU;
dùng `uv sync` để áp dụng cấu hình kho này. Không cài thêm `paddlepaddle` bản CPU.

| | Mặc định local |
|---|---|
| OCR | HunyuanOCR, FP16, DPI 200, cạnh dài tối đa 1280, tối đa 4096 token/trang |
| Dịch | Qwen2.5-3B-Instruct, FP16 trên GPU, batch 1, 1 GPU/model |
| Layout | Tắt; bật `n_layout` khi cần xem riêng |
| Đầu vào / kết quả | `data/` / `work/runs/` |

Đây là cấu hình khởi đầu để giảm nhu cầu VRAM, chưa phải kết quả benchmark GPU.
OCR và dịch chạy lần lượt trong các tiến trình riêng để giải phóng model giữa hai bước.
Nếu OCR báo chạm giới hạn token, kiểm tra trang đó và tăng `max_new_tokens`; trang
nhiều chữ nhỏ có thể cần tăng `max_side` lên 1600. Cấu hình thay đổi tạo run mới.

Sau OCR, lấy đường dẫn `output_dir` được in ra làm đầu vào dịch:

```powershell
# Thay <hash> bằng thư mục thực tế vừa được tạo
vtm translate --set input_dir=work/runs/ocr_hunyuan_<hash>/txt
vtm report --kind dich --workdir work
```

Trong VS Code, chọn `.venv\Scripts\python.exe` bằng **Python: Select Interpreter**
và **Select Kernel** cho mỗi notebook. Chạy notebook từ thư mục gốc repo. Khi chạy
local, notebook không tự `pip install`, không tạo môi trường Paddle riêng, và dùng
cấu hình 1 GPU. Notebook OCR mặc định chọn profile `"1"` và 2 trang của `data/1.pdf`;
đổi `BOOK_CONFIG` để chọn sách khác, rồi đổi `pages=""` khi muốn OCR cả sách. Notebook dịch lấy thư mục
`txt/` của run OCR mới nhất; có thể sửa `INPUT_DIR` trong ô liệt kê để chọn run khác.
Trên Kaggle, notebook giữ đường dẫn và cấu hình nhiều GPU của Kaggle.

Worker OCR dùng bộ đếm thời gian riêng trên Windows: trang quá `page_timeout` được
ghi vào skip log, worker dừng rồi khởi động lại để xử lý trang tiếp theo, trong giới
hạn `max_restarts`. Trên Unix vẫn dùng `SIGALRM`.

Có thể kiểm tra pipeline mà không tải model bằng `--set engine=dummy`. Bộ kiểm tra
local bao gồm OCR, resume, dịch, đường dẫn Unicode và timeout/restart trên Windows:

```powershell
uv run --all-extras python -m unittest discover -s tests -v
```

## Kiểm tra độ phân giải ảnh trước khi chọn `max_side`

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/check_max_side.py
```

Script quét mọi trang trong `data/`, tính mức `max_side` nhỏ nhất không thu nhỏ ảnh
sau render, kiểm tra DPI so với ảnh scan nhúng và tạo ảnh so sánh trên các trang
mẫu. Kết quả và cấu hình theo từng PDF nằm trong `work/max_side_check/`.
Không sửa cấu hình pipeline hiện tại. Xem [hướng dẫn và cách đọc kết quả](scripts/README.md).

## OCR theo cấu hình riêng của từng PDF

Đã có 8 profile trong [`configs/ocr_books/`](configs/ocr_books/README.md), dựa trên
kích thước trang và độ phân giải scan thực tế. Trong ô cấu hình notebook OCR:

```python
BOOK_CONFIG = "1"  # 1, 2, 3, 4, 5, 6, bien, kinh; None = sửa cấu hình thủ công
```

Hoặc chạy CLI:

```powershell
.\.venv\Scripts\vtm.exe ocr --config configs/ocr_books/1.json --set pages=50-51
```

JSON có thể đặt riêng `dpi`, `max_side`, `max_new_tokens`, `prompt` và các trường
của `OCRConfig`. Giá trị JSON ghi đè cấu hình chung của notebook; cấu hình có hiệu
lực được in ra để kiểm tra. Các profile hiện tại giữ độ chính xác số và engine
đã chọn trong notebook, chỉ chọn sách và chỉnh tham số ảnh/giới hạn sinh.
Đây là điểm xuất phát để thử OCR, chưa phải cấu hình đã xác nhận tối ưu chất lượng.

## Chuẩn bị trên Kaggle
1. Upload `data/` thành một Dataset (*Datasets → New Dataset*, kéo thả file zip).
2. *Create → New Notebook → File → Import Notebook* → chọn file `.ipynb`.
3. *Settings*: **Accelerator = GPU T4 x2** (2 GPU, mỗi GPU 16 GB VRAM), **Internet = On**.
4. Trong ô đầu notebook, điền `REPO_URL` (repo công khai). Notebook tự clone/pull rồi `pip install -e`.
5. *Add Input* → dataset ở bước 1 → *Save Version* → **Save & Run All (Commit)**: chạy nền, tắt máy được;
   xong thì xem kết quả ngay trong các ô report của notebook, hoặc tab *Output*.

Kaggle cho 30 giờ GPU/tuần, mỗi phiên tối đa 12 giờ.

## Bước 1 — OCR
- Notebook tự tìm mọi PDF trong Input, OCR từng trang trên cả 2 GPU, ghi ra
  `runs/ocr_<tên>_<hash>/txt/<tên sách>.txt`.
- Layout vẽ các khung (tiêu đề, đoạn văn, bảng, hình…) trên vài trang mẫu, lưu ở
  `runs/ocr_*/layout/<engine>_<tên sách>_tr<trang>.png`. Chạy cả 2 notebook rồi đặt ảnh cạnh nhau
  để so cách 2 model "nhìn" trang.
- Chạy thử nhanh: đặt `pages="1-10"`, `limit=10`, `n_layout=0` trong ô `OCRConfig`.
- Tốc độ khoảng 10–20 giây/trang/GPU.

## Bước 2 — Dịch
- *Add Input* → *Notebooks* → chọn notebook bước 1 (lấy Output của nó), hoặc một dataset chứa các file `.txt`.
- Sửa ô `TranslateConfig` rồi chạy như trên. Kết quả nằm ở `runs/dich_*/dich/<tên sách>.txt`.
- `model`: model chat/instruct bất kỳ trên Hugging Face, miễn vừa VRAM. Ví dụ:

| Model | `gpu_per_model` | `load_4bit` |
|---|---|---|
| `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen3-4B` | 1 (2 bản chạy song song) | False |
| `Qwen/Qwen2.5-7B-Instruct` (mặc định), `Qwen/Qwen3-8B`, `SeaLLMs/SeaLLMs-v3-7B-Chat`, `sail/Sailor2-8B-Chat` | 2 | False |
| `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen3-14B` | 2 | True |

  Model cần đăng nhập (Llama, Gemma…): *Add-ons → Secrets*, thêm `HF_TOKEN`.
- Chạy thử nhanh: đặt `limit=20` trong `TranslateConfig`.

## Hết 12 giờ mà chưa xong
Cả 3 notebook đều lưu kết quả sau từng trang hoặc từng đoạn. Khi bị dừng: *Add Input* → *Notebooks* → chọn chính notebook đó (lấy Output phiên vừa chạy), đặt `resume_from=["/kaggle/input/<output>"]` →
*Save & Run All* lại. Phần đã xong sẽ không phải chạy lại.

## Gợi ý thử nghiệm
Ngoài đổi model, có thể chỉnh các tham số trong ô cấu hình rồi so kết quả. Mỗi lần chỉ đổi 1 tham số, chạy trên cùng
vài trang/đoạn (`pages`, `limit`), đổi `run_name` (hoặc `fresh=True`) để không dùng lại cache cũ.

**OCR**
| Tham số | Thử | Xem gì |
|---|---|---|
| `dpi`, `max_side` | 150 / 200 / 300; 1280 / 1600 / 2048 | chữ nhỏ, nét mờ đọc đúng hơn không; chậm hơn bao nhiêu |
| `prompt` (Hunyuan) | prompt cho văn bản cổ `提取图中的文字。`, hoặc tự viết | sách chữ dọc, văn ngôn |
| `dtype` (Hunyuan) | `"fp16"` | nhanh gấp đôi, chữ có sai khác không |
| `repetition_penalty` (Hunyuan) | 1.0 / 1.08 / 1.2 | trang bị lặp chữ vô hạn, hoặc mất chữ lặp thật (vd 一一) |
| `max_new_tokens` | 2000 / 6000 | trang dày chữ có bị cụt không |
| `paddle_version`, `paddle_args` (Paddle) | `"v1.5"`; `{"use_doc_orientation_classify": True}`, `{"use_doc_unwarping": True}` | trang xoay, trang scan cong; tên tham số xem tài liệu PaddleOCR-VL |

Ô **Layout detection** cho thấy model chia vùng ra sao. Với Paddle, vùng nào bị bỏ (chú thích, số trang) hay xếp sai
thứ tự thì text cũng thiếu hoặc lộn xộn theo. Với Hunyuan, khung chỉ để xem: lượt OCR chính đọc nguyên trang.

**Dịch**
| Tham số | Thử | Xem gì |
|---|---|---|
| `system`, `user` (prompt) | thêm vài cặp câu mẫu Hán–Việt, thêm bảng thuật ngữ (tên huyệt, vị thuốc), yêu cầu giữ xuống dòng | thuật ngữ đúng và nhất quán hơn không |
| `chunk_chars` | 150 / 300 / 600 | đoạn dài: đủ ngữ cảnh hơn nhưng dễ dịch sót |
| `temperature` | 0 / 0.3 / 0.7 | văn tự nhiên hơn hay bịa thêm |
| `repetition_penalty` | 1.0 / 1.05 / 1.15 | lặp câu, hoặc né lặp cả thuật ngữ phải lặp |
| `load_4bit` | cùng model, bật/tắt | nén 4-bit kém đi bao nhiêu, đổi lại được model lớn hơn |
| `max_new_tokens` | tăng nếu báo "chạm max_new_tokens" | đoạn dịch bị cụt |
