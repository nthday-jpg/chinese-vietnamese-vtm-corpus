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

## Chạy local (CPU)

Dùng `uv` để tạo venv nhẹ; chỉ cần `pypdfium2`, `pillow` cho pipeline và engine
`dummy` để test/resume/report mà không tải model:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e .
source .venv/bin/activate

# OCR giả lập 2 trang, không cần GPU/model
vtm ocr --set engine=dummy --set input_dir=data --set pages=1-2 --set n_layout=0

# Dịch giả lập từ output OCR vừa tạo (trỏ vào đúng run/txt)
vtm translate --set engine=dummy --set input_dir=work/runs/ocr_smoke_.../txt --set limit=2

# Xem tổng hợp
vtm report --kind ocr --workdir work
```

Muốn chạy model thật trên CPU: cài thêm extra tương ứng và đặt `engine`, `device="cpu"`.
Hunyuan/Paddle 1B trên CPU rất chậm; nên chỉ chạy vài trang để đánh giá, còn full run
vẫn trên Kaggle. Paddle cần venv riêng (`paddle-cpu` hoặc `paddle-gpu`).

## Chuẩn bị trên Kaggle
1. Upload `data/` thành một Dataset (*Datasets → New Dataset*, kéo thả file zip).
2. *Create → New Notebook → File → Import Notebook* → chọn file `.ipynb`.
3. *Settings*: **Accelerator = GPU T4 x2** (2 GPU, mỗi GPU 16 GB VRAM), **Internet = On**.
4. Trong ô đầu notebook, điền `REPO_URL` (repo công khai). Notebook tự clone/pull rồi `pip install -e`.
5. *Add Input* → dataset ở bước 1 → *Save Version* → **Save & Run All (Commit)**: chạy nền, tắt máy được;
   xong thì xem kết quả ngay trong các ô report của notebook, hoặc tab *Output*.

Kaggle cho 30 giờ GPU/tuần, mỗi phiên tối đa 12 giờ.

## Bước 1 — OCR

### Tiền xử lý đã thử và loại bỏ

Hai cách tiền xử lý **xoá mực đỏ** bằng ngưỡng màu và **xoá nền** bằng ngưỡng
xám cục bộ cho kết quả kém trên các trang scan đã thử: còn sót mực đỏ, tạo
mảng trắng, làm mất nét chữ mờ và vẫn giữ viền scan tối. Các bộ tiền xử lý này
đã bị loại bỏ, không tích hợp vào pipeline. Không nên dùng lại nếu chưa có
phương pháp tốt hơn và kiểm tra chất lượng trên từng sách.

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
