# Đề tài: Dịch sách Y học cổ truyền từ chữ Hán sang tiếng Việt

```
PDF sách Hán (scan) ──► Bước 1: OCR ──► .txt chữ Hán ──► Bước 2: LLM dịch ──► .txt tiếng Việt
```

## Bạn nhận được
| | |
|---|---|
| `data/` | PDF sách Hán (bản scan, không lấy chữ trực tiếp được) |
| `0_kiem_tra_moi_truong_kaggle_offline.ipynb` | Kiểm tra lại thư viện, CPU/RAM/GPU và ABI trên Kaggle; không cài package hoặc OCR |
| `2_dich_han_viet.ipynb` | Bước 2: dịch bằng một LLM nhỏ tuỳ chọn |
| `3_so_sanh_so_chu_ocr.ipynb` | So bốn model, căn chỉnh ký tự, xem bốn dự đoán và crop từng chữ bất đồng |
| `PROBLEM.md` | Mô tả bài toán: input / output của từng bước |
| `FLOW.md` | Giải thích từng bước notebook làm gì + gợi ý các cách làm khác |

Chạy notebook 3 để OCR và so sánh, sau đó dùng notebook 2 để dịch bản `.txt` đã chọn.
Workflow dịch trong notebook 2 được giữ nguyên.
Notebook chỉ là cách làm mẫu — bạn được tự đề xuất cách khác (xem cuối `FLOW.md`).

## Cấu trúc repo

```
src/vtm_corpus/          # toàn bộ logic: OCR, dịch, chia việc, report, CLI
  config.py              # OCRConfig / TranslateConfig / ReportConfig
  pipeline.py            # tìm input, chạy worker, resume, ghép output, manifest
  ocr_worker.py          # worker OCR (1 GPU / 1 tiến trình)
  dich_worker.py         # worker dịch (1 model / 1 tiến trình)
  report.py              # bảng tổng hợp + hiển thị OCR/dịch
  corpus_ocr.py          # chọn trang và chạy tuần tự các model
  ocr_records.py         # đọc checkpoint và xác minh PDF nguồn
  text_alignment.py      # căn chỉnh chuỗi ký tự
  multi_consensus.py     # phiếu và entropy của nhiều model
  multi_compare.py       # tổng hợp kết quả và xem từng chữ
  character_regions.py   # tọa độ chung và crop ký tự
  cli.py                 # lệnh `vtm`
3_so_sanh_so_chu_ocr.ipynb # cài offline + OCR bốn model + xem bất đồng
2_dich_han_viet.ipynb     # workflow dịch thuật hiện tại
pyproject.toml
```

Mã nguồn trong repo dùng cho CLI/local. Notebook 3 offline có 8 ô code: cấu hình,
các hàm xử lý, setup dependency, một worker, gọi setup, chạy OCR, xem kết quả và xuất ZIP.
Hàm setup/điều phối/so sánh viết trực tiếp; chỉ worker dùng `%%writefile`.
Không đóng wheel dự án vào packages.
Khi sửa mã trong repo, chạy `tools/update_kaggle_notebook_bootstrap.py` để cập nhật
các ô nguồn; sửa logic không cần upload lại thư viện.
Mỗi run ghi `run_manifest.json` (toàn bộ cấu hình đã dùng), và thư mục
`runs/<loại>_<tên>_<config-hash>/` nên không bị lẫn kết quả giữa các cấu hình khác nhau.

Kết quả nằm trong `runs/ocr_*/txt/`, `runs/ocr_*/layout/`, `runs/dich_*/dich/` và
được hiển thị trực tiếp trong notebook ở các ô report — không cần tải file về.

## Chạy OCR trên Kaggle offline bằng notebook 3

Môi trường đích đã được ghi trong [hướng dẫn packages offline](docs/kaggle_offline_packages.md).
Notebook 3 kiểm tra runtime/package trước khi chạy các model.
Đã dựng lại bộ packages theo báo cáo Kaggle mới: khoảng 3,93 GiB,
thay cho bộ đầy đủ 8,79 GiB; dùng lại các thư viện kernel đáp ứng phiên bản/ABI.
Upload `kaggle_upload/packages_upload.zip` đã tạo sẵn thành Dataset packages riêng.
ZIP này dùng `ZIP_STORED`, không nén thêm; giữ nguyên cây `packages/wheels`,
`packages/sources`, `packages/requirements` và tên wheel gốc. `sources` đã giải nén
thành thư mục thường, không đưa tar/ZIP hay symlink vào Dataset. Packages khoảng 3,93 GiB;
Dataset models/PDF đã upload trước đó tiếp tục dùng lại. Gắn các Dataset vào
[`3_so_sanh_so_chu_ocr.ipynb`](3_so_sanh_so_chu_ocr.ipynb). Điền các đường dẫn còn trống
ở bước 1: `PACKAGES_DIR`, `MODELS_DIR`, `DATA_DIR` cho ba Dataset riêng;
đường dẫn từng model được tự ghép từ `MODELS_DIR`. Mỗi phiên chỉ mở notebook 3
và **Run All**: tự cài/kiểm tra packages offline → nạp runtime → OCR → so sánh.
Nếu gặp checksum `nvidia-nccl-cu13/.../nccl.h`, cập nhật Dataset packages bằng bộ mới
đã chứa NCCL 2.28.9 riêng cho TongGu rồi gắn phiên bản mới vào notebook.
Bước cài offline nằm ngay trong notebook 3. `PACKAGES_DIR` trỏ tới thư mục chứa
`paddle_manifest.json` trong Dataset mới; cần import lại notebook 3 để có các ô mã nguồn.
Có `kaggle_upload/sources_upload_test.zip` khoảng 178 MiB để thử riêng thư mục sources
trước khi upload lại bộ đầy đủ. ZIP thử không thay thế Dataset packages cho OCR.
Xem [hướng dẫn, cấu trúc bundle và phần TongGu cần build](docs/kaggle_offline_packages.md).
Dataset packages chưa chứa trọng số model/PDF và chưa thay thế việc xác nhận GPU thực trên Kaggle.
Khi kernel thay đổi, chạy lại [`0_kiem_tra_moi_truong_kaggle_offline.ipynb`](0_kiem_tra_moi_truong_kaggle_offline.ipynb)
để kiểm tra packages và dựng lại bundle nếu cần. Không cần chạy notebook 0 mỗi phiên.

## Character-Level Consensus trong notebook 3: bốn model

[`3_so_sanh_so_chu_ocr.ipynb`](3_so_sanh_so_chu_ocr.ipynb) chạy lần lượt
**TongGuOCR, PP-OCRv6-medium, PaddleOCR-VL-1.6 và HunyuanOCR-1.5** trên cùng
tập sách/trang. Giữ bảng đếm chữ và bổ sung căn chỉnh từng ký tự, bốn dự đoán,
phiếu 3/4 hoặc hòa 2/2, entropy và ảnh từng ký tự bất đồng.

- Điền đường dẫn và tham số ở bước 1 rồi Run All. `setup_ocr` chuẩn bị phiên,
  `run_ocr_models` chạy bốn model và `compare_ocr_results` tổng hợp kết quả.
  Chỉ một ô `%%writefile` ghi worker độc lập ra `/tmp/vtm_notebook_worker/ocr_worker.py`
  trên Kaggle; logic còn lại chạy trực tiếp trong notebook.
- Windows dùng kernel `.venv`; TongGuOCR dùng Python riêng trong WSL. Điền
  `tonggu_repo`, `tonggu_python`, `tonggu_wsl_distro` trong `ADVANCED` theo môi trường thực tế.
  Notebook kiểm tra môi trường trước khi OCR; không tự cài WSL/dependency.
- Kaggle mặc định `RUN_OCR=True`, tự chuẩn bị cả Paddle và TongGu từ Dataset ở đầu
  notebook. Runtime cùng phiên còn tốt được dùng lại;
  phiên mới mất `/tmp` sẽ tự cài lại. Windows giữ mặc định `RUN_OCR=False`.
  Đặt `RUN_OCR=False` để đọc checkpoint; OCR mới mặc định toàn bộ `1.pdf`, `MAX_BOOKS=1`, `PAGES=""`,
  `LIMIT=None`. Điền `MODELS_DIR` ở bước 1; setup báo thiếu trước
  khi cài môi trường. Chế độ đọc checkpoint không cài/build Paddle hoặc TongGu.
  Các model giải phóng tiến trình giữa stage.
- Render PNG được cache chung giữa các model; các trang đã xong được dùng lại khi
  mở rộng phạm vi. Ô cuối xuất HTML offline, JSONL, ảnh bất đồng và ZIP nén nhẹ tại
  `WORK_DIR/output/<tên_sách>_results.zip`. Giải nén rồi mở `index.html` trong thư mục sách.
- Khung chung lấy từ `return_word_box=True` của PP-OCRv6, chỉ giữ phần tử có đúng
  một ký tự và geometry hợp lệ. Mỗi bất đồng in dự đoán của cả bốn model, crop chữ
  và ảnh ngữ cảnh đánh dấu khung trên pixel trang PDF gốc. Không chia đều khung dòng
  để bịa vị trí; chỗ không định vị được ghi “chưa xác định”.
- `REVIEW_BOOK/REVIEW_PAGE` chọn trang PDF cần xem; `MAX_REVIEW_ROWS` giới hạn số
  ảnh hiển thị. Chỉ render trang được chọn một lần, không render toàn bộ corpus.
- Model chưa có output khác với gap `∅`. Đa số nghiêm ngặt cần 3/4; hòa 2/2 hay
  2/1/1 không tự chọn chữ. Khung là ước lượng của PP-OCR, các dự đoán khác được
  liên kết bằng căn chỉnh; vẫn cần đối chiếu ảnh khi chữ lặp hoặc khác thứ tự cột.
- `ADVANCED["review_threshold"]=0.05` và `ADVANCED["max_alignment_cells"]=8_000_000`
  điều khiển rà soát và ngân sách căn chỉnh. `ADVANCED["export_consensus"]=True` tùy chọn lưu JSON bằng
  chứng/config; không xuất ảnh/báo cáo riêng mặc định.

Đồng thuận không phải độ chính xác/CER: cả bốn model có thể cùng sai hoặc bỏ sót
dòng; hai pipeline Paddle có thể có lỗi tương quan. Không tự chốt bản OCR hay đưa
chữ còn tranh chấp sang dịch. SHA-256 PDF được kiểm tra trước khi ghép nguồn;
nguồn chưa xác minh không được dùng để hiển thị crop.

Xem **[hướng dẫn cài WSL/TongGu, cấu hình và cách đọc kết quả](docs/four_model_consensus.md)**.
Ý tưởng agreement/routing tham khảo [Consensus Entropy](https://arxiv.org/html/2504.11101v3);
entropy theo cột ở đây là thống kê bổ sung, không tái lập đầy đủ CE của bài báo.

## Chọn tập sách (mỗi sách có thể cần cấu hình khác nhau)

Mỗi sách có thể có vấn đề riêng (chữ dọc, trang nghiêng, bảng, mực mờ…), nên pipeline
cho phép chọn một tập con thay vì luôn chạy tất cả. Điền `DATA_DIR` bằng thư mục PDF
trong Dataset đã gắn ở bước 1; notebook không điền sẵn tên Dataset trên Kaggle.
Các lệnh liệt kê và cấu hình dùng chung biến này. Ví dụ liệt kê input:

```python
from vtm_corpus import list_pdfs, list_texts

list_pdfs(DATA_DIR)                   # notebook 3, OCR: tên sách + số trang
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
CLI và hai notebook, kể cả Paddle GPU:

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
cấu hình 1 GPU. Notebook 3 mặc định chọn `BOOKS=["1"]`, `MAX_BOOKS=1`, `PAGES=""`, `LIMIT=None`;
đổi các biến này để chọn sách/trang khác. Notebook dịch lấy thư mục
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
kích thước trang và độ phân giải scan thực tế. Trong ô cấu hình notebook 3:

```python
BOOKS = ["1"]  # 1, 2, 3, 4, 5, 6, bien, kinh
ADVANCED = {"use_book_profiles": True}
# Có thể thêm "profile_dir": Path("configs/ocr_books") nếu cần chọn thư mục khác.
```

Hoặc chạy CLI:

```powershell
.\.venv\Scripts\vtm.exe ocr --config configs/ocr_books/1.json --set pages=50-51
```

JSON có thể đặt riêng `dpi`, `max_side`, `max_new_tokens`, `prompt` và các trường
của `OCRConfig`. Giá trị JSON ghi đè cấu hình chung của notebook; cấu hình có hiệu
lực được in ra để kiểm tra. Các profile hiện tại giữ độ chính xác số và engine
đã chọn trong notebook, chỉ chỉnh tham số ảnh/giới hạn sinh cho sách đã chọn.
Đây là điểm xuất phát để thử OCR, chưa phải cấu hình đã xác nhận tối ưu chất lượng.

## Chuẩn bị notebook 2 trên Kaggle (workflow dịch hiện tại)
1. Upload `data/` thành một Dataset (*Datasets → New Dataset*, kéo thả file zip).
2. *Create → New Notebook → File → Import Notebook* → chọn file `.ipynb`.
3. *Settings*: **Accelerator = GPU T4 x2** (2 GPU, mỗi GPU 16 GB VRAM), **Internet = On**.
4. Trong ô đầu notebook, điền `REPO_URL` (repo công khai). Notebook tự clone/pull rồi `pip install -e`.
5. *Add Input* → dataset ở bước 1 → *Save Version* → **Save & Run All (Commit)**: chạy nền, tắt máy được;
   xong thì xem kết quả ngay trong các ô report của notebook, hoặc tab *Output*.

Kaggle cho 30 giờ GPU/tuần, mỗi phiên tối đa 12 giờ.

## Bước 1 — OCR
- Dùng notebook 3: bốn model chạy lần lượt trên cùng tập PDF/trang, lưu checkpoint
  và xuất `.txt` theo từng model/sách. Đường dẫn output được ghi trong manifest.
- Xem bốn dự đoán và khung ký tự bất đồng trước khi chọn bản OCR đưa sang dịch.
- Mặc định xử lý hết một PDF; dùng `PAGES="1-2"`, `LIMIT=2` để thử nhanh trước lượt dài.

## Bước 2 — Dịch
- *Add Input* → *Notebooks* → chọn Output notebook 3, hoặc một dataset chứa các file `.txt` đã chọn sau rà soát OCR.
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
Notebook 2 lưu kết quả sau từng đoạn. Khi bị dừng: *Add Input* → *Notebooks* → chọn chính notebook đó (lấy Output phiên vừa chạy), đặt `resume_from=["/kaggle/input/<output>"]` → *Save & Run All* lại.
Notebook 3 lưu sau từng trang; giữ cùng `WORK_DIR` hoặc khôi phục đầy đủ checkpoint vào đó trước khi chạy lại. Phần đã xong được bỏ qua.

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

Ô xem bất đồng trong notebook 3 hiển thị bốn dự đoán cùng khung ký tự từ PP-OCR.
Khung này dùng để đối chiếu ảnh; chất lượng OCR vẫn cần kiểm tra theo nội dung gốc.

**Dịch**
| Tham số | Thử | Xem gì |
|---|---|---|
| `system`, `user` (prompt) | thêm vài cặp câu mẫu Hán–Việt, thêm bảng thuật ngữ (tên huyệt, vị thuốc), yêu cầu giữ xuống dòng | thuật ngữ đúng và nhất quán hơn không |
| `chunk_chars` | 150 / 300 / 600 | đoạn dài: đủ ngữ cảnh hơn nhưng dễ dịch sót |
| `temperature` | 0 / 0.3 / 0.7 | văn tự nhiên hơn hay bịa thêm |
| `repetition_penalty` | 1.0 / 1.05 / 1.15 | lặp câu, hoặc né lặp cả thuật ngữ phải lặp |
| `load_4bit` | cùng model, bật/tắt | nén 4-bit kém đi bao nhiêu, đổi lại được model lớn hơn |
| `max_new_tokens` | tăng nếu báo "chạm max_new_tokens" | đoạn dịch bị cụt |
