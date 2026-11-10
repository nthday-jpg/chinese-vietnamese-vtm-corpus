# Notebook 3: bốn model và ảnh từng ký tự bất đồng

Notebook có 8 ô code: **cấu hình → hàm xử lý → setup offline → worker → gọi setup → OCR → kết quả → xuất ZIP**.
Điền đường dẫn ở bước 1 rồi Run All. Mã setup/điều phối/so sánh là hàm Python trực tiếp,
có thể sửa ngay trong notebook. Chỉ một ô `%%writefile` ghi worker độc lập để chạy riêng.
Nếu chạy **Kaggle offline**, dùng lại Dataset packages đã chuẩn bị; chỉ cần import bản notebook 3 mới.
Xem [Dataset packages và cấu hình khởi động](kaggle_offline_packages.md).
Notebook tự đọc runtime vừa chuẩn bị từ `runtime_config.json`; phần WSL bên dưới chỉ dành cho Windows.

Các lệnh thực thi cuối notebook:

```python
session = setup_ocr(...)  # 3 đường dẫn Dataset và BOOKS/PAGES/LIMIT ở ô cấu hình đầu
stage_results = run_ocr_models(session)
result = compare_ocr_results(session)
review_character_conflicts(session, book=REVIEW_BOOK, page=REVIEW_PAGE, max_characters=MAX_REVIEW_ROWS, view=REVIEW_VIEW)
exports = export_ocr_results(session, output_dir=OUTPUT_DIR, include_crops=EXPORT_CROPS, create_zip=CREATE_ZIP)
```

## So sánh lại log đã tải về

Notebook tự chạy **Page-level consensus → Character-level consensus**.
Lớp cấp trang đối chiếu sự có mặt, số lần xuất hiện và thứ tự của vùng chữ giữa
các model. Bảng có `Cấu trúc trang`, `Dấu hiệu cấp trang`; HTML có bảng model/vùng/
đối chứng và ảnh tổng quan khi có tọa độ. `comparison.jsonl` giữ `page_consensus`.
Xem [thuật toán, các nhãn và giới hạn bằng chứng](page_level_consensus.md).

Đặt `RUN_OCR=False`, `EXPORTED_LOG_DIR="data/log_notebook_3"`, `DATA_DIR="data"`.
Thư mục log có thể là thư mục chứa các sách hoặc một sách có `manifest.json` và `raw/`.
Notebook áp dụng `BOOKS`, `MAX_BOOKS`, `PAGES`, `LIMIT`, tạo checkpoint riêng trong
`WORK_DIR/imported_logs` và xác minh PDF địa phương bằng SHA-256 đã lưu khi OCR.
Log và PDF đầu vào được giữ nguyên. Để `EXPORTED_LOG_DIR=""` khi dùng checkpoint sẵn ở
`WORK_DIR` hoặc chạy OCR mới. `EXPORT_RESULTS=True` xuất lại vào `OUTPUT_DIR`.

So sánh mặc định dùng `ADVANCED.get("reading_order", "auto")`: PP-OCR có phần lớn chữ
trong các cột dọc được xếp từ phải sang trái, các mảnh trong cùng cột từ trên xuống.
Offset ký tự, khung chữ và dòng được chuyển cùng nhau; `raw` giữ nguyên bản gốc.
Có thể ép `"vertical_rtl"` hoặc giữ `"detection"` cho bố cục khác. Mốc chữ khớp duy nhất
trong từng dòng giới hạn phạm vi căn chỉnh khi đủ bằng chứng; nếu thiếu mốc hoặc mốc
đảo thứ tự thì dùng căn chỉnh chuỗi toàn trang. `line_alignment` ghi rõ từng trường hợp.
Khoảng cách từng cặp vẫn là Levenshtein toàn trang; mốc chữ không xác nhận tọa độ giữa model.

Output chạm giới hạn token, lặp dài, lặp hàng đánh số, phần lớn là ký hiệu hoặc có độ dài
bất thường so với ít nhất hai model còn lại được giữ để kiểm tra và **không bỏ phiếu**.
`excluded_models` tách biệt với `missing_models`; đa số vẫn cần hơn nửa số model cấu hình
(bốn model cần ba phiếu). HTML/JSONL/bảng notebook hiển thị model và lý do bị loại khỏi phiếu.
`status="complete"` chỉ nói đã có đủ output, còn `quality_review_pages` cho biết số trang
có vấn đề chất lượng. Không coi số cột bất đồng là số chữ sai hay độ chính xác OCR.

PDF 1 hiện có ảnh nhúng 600 × 450 pixel. Render DPI lớn hơn chỉ phóng lớn ảnh này;
muốn phục hồi chi tiết cần bản quét nguồn có độ phân giải cao hơn.

Notebook [`3_so_sanh_so_chu_ocr.ipynb`](../3_so_sanh_so_chu_ocr.ipynb) so sánh:

| Key | Model | Môi trường mặc định | Tọa độ dùng trong báo cáo |
|---|---|---|---|
| `ppocr` | PP-OCRv6-medium, detector + recognizer medium | `.venv` Windows | `return_word_box=True`, chỉ giữ phần tử có đúng một ký tự |
| `tonggu` | TongGuOCR Stage 3, pipeline layout/block chính thức | Python riêng trong WSL | Khung chung PP-OCR, không nhận khung dòng TongGu là khung chữ |
| `paddle` | PaddleOCR-VL-1.6 | `.venv` Windows | Khung chung PP-OCR |
| `hunyuan` | HunyuanOCR-1.5 | `.venv` Windows, Transformers | Khung chung PP-OCR |

Mỗi stage dùng cùng tập sách/trang và chạy chờ hoàn tất. Worker kết thúc trước stage tiếp theo để giải phóng model. Khi cùng cấu hình đã có đủ checkpoint, stage không tải model lại.

## Đường dẫn weights ở bước 1

Hai bộ upload tách riêng models/packages. Với packages, dùng file có sẵn
`kaggle_upload/packages_upload.zip` (ZIP_STORED, không nén thêm, giữ nguyên cây thư mục)
hoặc folder `kaggle_upload/packages/`. Packages chỉ chứa dependency/runtime;
mã điều phối dự án nằm trong notebook. `PACKAGES_DIR` trỏ tới thư mục chứa
`paddle_manifest.json` sau khi gắn Dataset.
Notebook có ba đường dẫn đầu vào riêng: `PACKAGES_DIR`, `MODELS_DIR`, `DATA_DIR`.
Điền theo ba Dataset thực tế; Dataset PDF đã upload trước đó được dùng lại.
`WEIGHTS` được tự ghép từ `MODELS_DIR` theo cây thư mục dưới đây, không cần điền từng đường dẫn.

| Key trong `WEIGHTS` | Nội dung |
|---|---|
| `hunyuan` | Snapshot HunyuanOCR-1.5 cục bộ |
| `ppocr_det`, `ppocr_rec` | Hai thư mục inference PP-OCRv6-medium |
| `paddle_vl`, `paddle_layout` | Thư mục PaddleOCR-VL-1.6 và model layout tương ứng |
| `tonggu_ocr` | Thư mục `tonggu/ocr` |
| `tonggu_layout` | File `tonggu/layout/finetune_infer.pth` |

Trên Kaggle, notebook tạo `WORK_DIR/tonggu_paths.json` theo hai đường dẫn TongGu đã điền.
Nếu có config riêng (ví dụ cho WSL), thêm `WEIGHTS["tonggu_config"]` để dùng config đó.

Setup báo tất cả đường dẫn thiếu trước khi cài/build runtime. Không còn cờ
`OFFLINE_WEIGHTS_CONFIGURED` phải bật thủ công. Việc thư mục tồn tại chưa xác nhận
nội dung snapshot hoặc GPU inference thành công.
Các tham số đường dẫn Paddle theo [PaddleOCR API](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html)
và [PaddleOCR-VL API](https://github.com/PaddlePaddle/PaddleOCR/blob/main/paddleocr/_pipelines/paddleocr_vl.py).

## Chuẩn bị Windows và TongGuOCR trong WSL

Môi trường Windows dùng chung `.venv` sẵn có, chọn nó làm kernel notebook. PaddleOCR cần phiên bản ít nhất 3.7 cho PP-OCRv6; không cài thêm Paddle CPU vào môi trường GPU.

```powershell
uv sync --all-extras
```

TongGuOCR cần môi trường riêng vì upstream công bố Linux/Python 3.10, CUDA, bf16 và `mmcv-full` biên dịch. Không cài bộ này vào `.venv` Windows. Trước khi dùng, cần WSL2 có GPU passthrough và NVIDIA driver hỗ trợ CUDA của môi trường TongGu.

Nếu chưa có WSL, người dùng cài từ PowerShell với quyền quản trị và hoàn tất khởi tạo Ubuntu (có thể cần khởi động lại Windows):

```powershell
wsl --install -d Ubuntu
wsl --list --verbose
```

Trong Ubuntu, sau khi đã có Conda và GPU hoạt động, theo setup chính thức:

```bash
git clone https://github.com/jzzh2004/TongGuOCR.git "$HOME/TongGuOCR"
cd "$HOME/TongGuOCR"
bash tools/setup_env.sh
conda activate tongguocr
bash tools/download_weights.sh huggingface
which python
pwd
```

Không cần chuyển PDF sang Linux: worker ánh xạ đường dẫn ổ đĩa Windows, ví dụ `C:\books\1.pdf` sang `/mnt/c/books/1.pdf`, và ghi checkpoint vào cùng thư mục `work/` Windows. Đường dẫn UNC/network share chưa được hỗ trợ bởi ánh xạ này; đặt repo/data/work trên ổ đĩa Windows thông thường.

Điền vào ô cấu hình notebook theo output `which python` và `pwd`:

```python
ADVANCED = {
    "tonggu_wsl_distro": "Ubuntu",
    "tonggu_repo": "/home/<linux-user>/TongGuOCR",
    "tonggu_python": "/home/<linux-user>/miniconda3/envs/tongguocr/bin/python",
}
WEIGHTS["tonggu_config"] = "/home/<linux-user>/tonggu.yaml"
```

Tên `<linux-user>` và vị trí Conda là giá trị thực tế của máy, không chạy nguyên placeholder. Nếu dùng config TongGu riêng, model/layout weights theo config đó; notebook truyền `max_new_tokens` làm giới hạn sinh mỗi block.

Preflight chỉ kiểm tra interpreter, package có mặt và đường dẫn weights mặc định/config. Nó chưa chứng minh mmcv/CUDA ABI, tải model hoặc inference thành công. Nếu preflight thất bại, notebook dừng trước khi OCR để không bỏ một model rồi báo đã so đủ bốn.

Worker WSL đăng ký process group kèm token riêng của lượt chạy. Interrupt/timeout sẽ nhắm process group của đúng lượt đó trước khi dừng launcher Windows. Log ghi trong workdir campaign. Không dùng `wsl --shutdown` vì có thể ảnh hưởng công việc WSL khác.

## Phạm vi và cách chạy

Mặc định:

```python
BOOKS = ["1"]
PAGES = "1-2"
LIMIT = 2
RUN_OCR = ON_KAGGLE  # True trên Kaggle; False trên Windows
WORK_DIR = Path("work/four_model_consensus")  # Kaggle dùng /kaggle/working/four_model_consensus
```

Đây là cấu hình thử nhanh. Mặc định mới là `BOOKS=["1"]`, `MAX_BOOKS=1`, `PAGES=""`,
`LIMIT=None`: toàn bộ một PDF, bốn model chạy lần lượt. Chỉ thay phạm vi trang không
làm mất các trang đã OCR với cùng model/cấu hình/PDF. Nguồn có SHA-256 khác sẽ tách
run mới. Checkpoint cũ chưa lưu hash nguồn được giữ với cảnh báo provenance.

## Xuất trọn sách và tăng tốc

Kaggle mặc định `EXPORT_RESULTS=True`, `EXPORT_CROPS=True`, `CREATE_ZIP=True` và
`PREVIEW_CONFLICTS=False`. Sau OCR, notebook so sánh rồi xuất vào `OUTPUT_DIR=WORK_DIR/"output"`:

```text
output/
  <ten_sach>/
    index.html
    pages/p00001.html
    raw/ppocr.jsonl
    raw/tonggu.jsonl
    raw/paddle.jsonl
    raw/hunyuan.jsonl
    comparison.jsonl
    crops/p00001/region_0001.png
    crops/p00001/region_0001_context.png
    errors.jsonl
    logs/<model>.log
    manifest.json
    style.css
    reader.js
  <ten_sach>_results.zip
```

Giải nén ZIP và mở `<ten_sach>/index.html`; không cần Internet hoặc server. HTML có
mục lục/lọc trang, bốn tab model, điều hướng trước/sau, ảnh cắt và ngữ cảnh có khung đỏ.
Văn bản OCR được escape khi đưa vào HTML, giữ nguyên trong UTF-8 JSONL. Số trang ngoài
gói bắt đầu từ 1; trường `raw.page` giữ chỉ số worker bắt đầu từ 0, ghi rõ trong manifest.
Trang thiếu output ghi `missing`, toàn sách ghi `partial`; không coi chạy hết lượt thử
là tất cả trang đã có OCR. `errors.jsonl` giữ lịch sử lỗi, kể cả lỗi đã thử lại thành công.

Các thay đổi tốc độ giữ nguyên DPI, kích thước ảnh và giới hạn sinh chữ:

- Cache PNG lossless dùng chung giữa các model, khóa theo SHA-256 PDF, trang, DPI và
  `max_side`. Cache hỏng được render lại; đặt `ADVANCED["cache_renders"]=False` nếu cần tắt.
- Worker giữ một model cho toàn bộ trang; ghi/fsync JSONL sau mỗi trang, snapshot tiến
  độ nhỏ để tránh đọc lại toàn bộ JSONL mỗi giây. Lượt chạy mới chỉ bổ sung trang thiếu.
- Tái sử dụng căn chỉnh của model tham chiếu thay vì tính lại cùng cặp chuỗi; giữ nguyên
  quy tắc xử lý tie và giới hạn bộ nhớ.
- HTML/crop dựng sau lượt model. Mỗi trang có tọa độ đủ tin cậy chỉ render một lần khi
  xuất; ảnh tải chậm trong bộ đọc. ZIP được tạo một lần cuối, văn bản nén mức 1, PNG giữ nguyên.

Cache ảnh và checkpoint nằm ngoài ZIP kết quả để tránh tải thừa. TXT tương thích với
workflow dịch vẫn được xuất tại run OCR; có thể tắt bằng `ADVANCED["ocr"]={"export":False}`.
Nếu một stage lỗi thì các model tiếp theo vẫn được thử, gói cuối ghi rõ phần thiếu.
Interrupt/dừng phiên giữ checkpoint; chạy lại với cùng `WORK_DIR`, hoặc dùng
`RUN_OCR=False`, `EXPORT_RESULTS=True` để xuất phần đã có mà không tải model.
Khi đọc checkpoint trên Windows, `EXPORT_RESULTS` mặc định False để không tự ghi gói mới.

Kiểm tra CPU dùng PDF ba trang đã xác nhận ba ảnh render dùng chung cho bốn model,
mở rộng từ hai trang sang cả sách, resume, cache hỏng, nguồn đổi, ZIP CRC và các liên
kết offline. Chưa benchmark GPU/inference toàn sách hoặc kiểm tra UI bằng browser.

`RUN_OCR=False` đọc `WORK_DIR / "manifest.json"`, không chạy model hay thay tập trang.
Manifest phải khai báo đúng bốn model. Chế độ này trên Kaggle chỉ chuẩn bị PDFium
nếu cần; không cài/build Paddle hoặc TongGu và không thử GPU.
`LIMIT` giới hạn tổng số trang trên mọi sách; `MAX_BOOKS` giới hạn số PDF sau lọc `BOOKS`.

Các thiết lập ít dùng nằm trong `ADVANCED`, mặc định để `{}`:

```python
ADVANCED = {
    "review_threshold": 0.05,
    "max_alignment_cells": 8_000_000,
    "use_book_profiles": True,
    "reuse_runtime": True,
    "export_consensus": False,
    "ocr": {"dpi": 200, "max_side": 1280, "max_new_tokens": 6000},
}
```

`existing_manifest` đổi đường dẫn checkpoint; `profile_dir` chọn thư mục profile khác.
`runtime_dir` và `log_dir` đổi vị trí runtime/log trên Kaggle.
`model_overrides` nhận dictionary theo key model, ví dụ `{"hunyuan": {"dtype": "bf16"}}`.

### Trang lỗi và timeout khi chạy lại

Mặc định TongGu có `page_timeout=1800` giây/trang; ba model còn lại là 600 giây/trang.
TongGu nạp detector layout và model OCR vào cache của pipeline trước khi bắt đầu đếm
timeout trang. Giới hạn sinh token vẫn giữ nguyên: TongGu nhận giới hạn cho từng block,
nên trang có nhiều block có thể chạy lâu hơn model đọc một ảnh toàn trang.
Đổi riêng timeout của TongGu trước Setup bằng:

```python
ADVANCED = {"model_overrides": {"tonggu": {"page_timeout": 2400}}}
```

`1/2 trang có output` kèm `TimeoutError` nghĩa là hai trang đã được chọn nhưng chỉ một
trang OCR thành công. Chạy lại ô OCR với cùng `WORK_DIR` sẽ thử lại trang thiếu, kể cả
trang từng có `NameError`, timeout hoặc marker skip từ phiên bản cũ. Trang có output
được giữ nguyên; stage đã đủ trang không tải model lại. Lần khởi động lại worker trong
cùng lượt chạy vẫn bỏ qua trang vừa lỗi để tránh lặp timeout vô hạn.

Log hiển thị số trang PDF bắt đầu từ 1, tên lỗi, thông báo và đường dẫn file lỗi.
File `*_errors.jsonl` lưu traceback, thời gian chạy, timeout và mã lượt thử; trường
`page` trong JSON bắt đầu từ 0. Lịch sử lỗi được giữ lại sau khi thử thành công.
Output thiếu vẫn được ghi `partial` và không được coi là đủ bốn model.

Bản notebook trước thiếu `comparison_text` trong worker độc lập của PPOCR, dù hàm
này có trong kernel. Worker mới chứa trực tiếp hàm đó và đã được thử parser trong
tiến trình riêng. Khi cập nhật bản sửa này trên Kaggle, chạy lại cả ô định nghĩa,
ô setup offline và ô worker `%%writefile`, sau đó Setup và OCR. Giữ nguyên session,
ba đường dẫn Dataset và `WORK_DIR` để tiếp tục checkpoint hiện tại.

Profile theo sách chỉ điều chỉnh ảnh/token/dtype…; danh tính model, revision, phiên bản Paddle và interpreter giữ theo cấu hình notebook. Hunyuan ghim commit `47644ecc4fc854efa4f505155158831f36773ee4` của repository `tencent/HunyuanOCR`, nơi bản 1.5 nằm ở root và bản 1.0 nằm trong `v1.0/`. Run name và config hash tách cache mới khỏi lần OCR cũ.

## Đọc bảng và ảnh

1. Bảng trang ghi số chữ Hán của từng model, số cột bất đồng và số cột có khung ký tự.
2. Bảng bất đồng ghi một hàng mỗi cột căn chỉnh, dự đoán của đủ bốn model, số phiếu và trạng thái định vị.
3. Chọn `REVIEW_BOOK`/`REVIEW_PAGE` ở bước 1 rồi chạy ô xem kết quả để in bốn dự đoán, ảnh cắt và vùng ngữ cảnh có khung đỏ. Một lần gọi chỉ render một trang PDF, một lần; không render toàn bộ corpus.

Mặc định `REVIEW_VIEW="regions"` gom các ký tự bất đồng liền nhau thành một vùng:

- Có khung ký tự hợp lệ: cắt theo khung chữ/cụm chữ của PP-OCR, giữ nguyên pixel ảnh nguồn.
- Chỉ có khung dòng hợp lệ và căn chỉnh có điểm neo rõ: cắt ngữ cảnh dòng, ghi rõ chưa định vị được chữ bất đồng.
- Thiếu tọa độ hoặc căn chỉnh mơ hồ: vẫn hiện dự đoán của bốn model và trạng thái chưa xác định.

Mỗi mục hiển thị dự đoán của từng model, ảnh cắt và ảnh ngữ cảnh có khung đỏ.
`REVIEW_VIEW="characters"` xem từng ký tự như trước. `MAX_REVIEW_ROWS` giới hạn số vùng
hoặc số ký tự theo chế độ đã chọn. Khi chưa chọn sách/trang cụ thể, notebook ưu tiên
trang có bất đồng thật giữa các output hiện có, thay vì dừng ở một trang chỉ thiếu model.

Nếu đã có `session` và checkpoint trên Kaggle, cập nhật/chạy lại các ô `definitions`,
`offline-setup` (chỉ định nghĩa hàm) và `results` để xem ảnh mới. Việc xem ảnh không cần
chạy lại Setup/OCR hay upload Dataset packages/models. PDF gốc và checkpoint cần còn
truy cập được. Ảnh hiển thị trực tiếp trong notebook; mặc định không xuất file ảnh.

Đây là căn chỉnh Levenshtein kiểu star, lấy PP-OCR làm chuỗi neo. Phần chèn được căn chỉnh riêng theo chuỗi chèn dài nhất. Đây không phải tối ưu căn chỉnh đồng thời bốn chuỗi. Chữ lặp, khác thứ tự cột/dòng hay mất cả đoạn có thể làm sai liên kết; cần kiểm tra ảnh.

Phiếu `∅` là gap của model có output; “chưa có output” không phải phiếu gap. Đa số nghiêm ngặt cần > một nửa tổng model cấu hình, tức 3/4. Hòa 2/2 và 2/1/1 không có lựa chọn. Nếu một model thiếu, không giảm mẫu số từ bốn xuống ba để tự tạo đa số 2/3. Chưa tự chốt chữ, sửa OCR, hoặc đưa bản có tranh chấp sang dịch.

Entropy là Shannon entropy của phiếu theo cột; điểm CE khoảng cách của một candidate là trung bình edit distance chuẩn hóa tới các candidate khác. Cả hai là tín hiệu đồng thuận, không phải xác suất đúng hoặc CER so với nhãn chuẩn. Hai pipeline Paddle có thể có lỗi tương quan; đa số 3/4 vẫn có thể sai. Ngưỡng review 5% là tham số thử nghiệm, chưa hiệu chỉnh trên corpus.

## Vì sao có dòng “chưa xác định”

Khung từng chữ chỉ dùng các phần tử `text_word` có đúng một ký tự, khớp chuỗi `rec_texts`
và có geometry hợp lệ. Từ Latin nhiều ký tự, output thiếu word boxes, ký tự chèn không
có trong PP-OCR, hoặc geometry không hợp lệ đều không được chia khung giả. Ảnh ngữ cảnh
dòng không thay đổi trạng thái định vị từng chữ. Khung PP-OCR là ước lượng từ model/CTC,
không phải bounding box chuẩn do người gán nhãn.

Khung được lưu theo tọa độ chuẩn hóa ảnh input PP-OCR, rồi ánh xạ sang kích thước trang PDF render để crop pixel gốc, thay vì crop thumbnail hiển thị. Các phép xoay/unwarp trong PP-OCR bị tắt để giữ hệ tọa độ. Model khác chỉ cung cấp dự đoán được căn chỉnh vào khung chung, không có nghĩa chính model đó trả được tọa độ ký tự này.

SHA-256 nguồn phải khớp giữa các run trước khi ghép; PDF khác nhau bị từ chối. Nguồn không truy cập/xác minh được vẫn có bảng text nhưng không hiển thị crop có thể sai nguồn.

`MAX_REVIEW_ROWS` giới hạn số hàng/ảnh hiển thị, không xóa bất đồng khỏi kết quả.
`ADVANCED["max_alignment_cells"]` giới hạn từng phép căn chỉnh; quá ngân sách ghi
`alignment_limit`, không gán điểm gần đúng. Đặt `ADVANCED["export_consensus"]=True`
trước setup nếu cần JSON bằng chứng/config; mặc định không ghi báo cáo hoặc ảnh rời.

## Nguồn và kiểm tra

- [TongGuOCR, setup và API pipeline](https://github.com/jzzh2004/TongGuOCR)
- [PaddleOCR, PP-OCRv6 và return_word_box](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html)
- [PaddleOCR-VL-1.6](https://www.paddleocr.ai/main/en/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.6.html)
- [HunyuanOCR-1.5, Transformers inference](https://github.com/Tencent-Hunyuan/HunyuanOCR/tree/main/inference/transformers)

Kiểm tra tự động không tải model:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_four_model_consensus.py -v
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_character_consensus.py -v
```

Các kiểm tra dùng campaign nhỏ hai/ba trang, xác minh ghép bốn model, 3/4–2/2, thiếu
output, resume, cache, source hash, crop, HTML/ZIP và cách dựng lệnh WSL. Chúng không
xác nhận chất lượng OCR thực, CUDA/mmcv hoặc inference toàn sách trên GPU.
