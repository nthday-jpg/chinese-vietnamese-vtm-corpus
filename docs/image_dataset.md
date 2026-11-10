# Chuẩn bị ảnh một lần, dùng lại cho các model OCR

Workflow này dùng PDFium/Pillow có sẵn, không cần trọng số OCR, GPU, tài khoản,
kết nối mạng hoặc API trả phí. Chạy bằng `.venv` ở gốc repo. Đọc
[báo cáo rà soát và quyết định](image_dataset_audit.md) cùng
[kết quả kiểm chứng](image_dataset_validation.md) trước khi chọn cấu hình.

## 1. Chạy mẫu nhỏ và xem đường cắt

Từ thư mục gốc repo, chạy bằng PowerShell:

```powershell
.\.venv\Scripts\vtm.exe images prepare --input-dir data --output-dir work/image_dataset --books 1 --pages 2,50 --limit 2 --max-books 1 --profile-dir configs/ocr_books --config configs/image_preparation_smoke.json
.\.venv\Scripts\vtm.exe images validate work/image_dataset
```

Mặc định CLI chỉ chọn PDF đầu tiên và hai trang đầu. Ví dụ trên chủ động chọn
trang 2 và 50 của sách 1. Profile theo sách chỉ đóng góp DPI và hướng đọc; token,
cap cạnh dài và ngân sách pixel riêng của model không đi vào dataset chuẩn.
PDF và kết quả OCR hiện có không bị thay đổi.

`dataset.json` trỏ tới nội dung của snapshot hoàn chỉnh mới nhất. Các file
`snapshots/<sha256>.json` cũ vẫn được giữ khi chuẩn bị thêm một tập trang khác.

Xem preview trong Python/notebook mà không nạp model:

```python
import json
from vtm_corpus.image_dataset import dataset_open, dataset_read, dataset_load_image

root, dataset = dataset_open("work/image_dataset")
for page in dataset["pages"]:
    record = json.loads(dataset_read(root, page["record"]))
    layout = json.loads(dataset_read(root, record["layout"]))
    display(dataset_load_image(root, layout["preview"]))
    print(page["book"], page["page"] + 1, layout["cuts"], layout["warnings"])
```

Đường đỏ biểu diễn các vùng cột đề xuất và số thứ tự đọc. Đường cam đánh dấu
rủi ro về lượng mực, độ ổn định theo ngưỡng hoặc hỗ trợ chữ ở hai phía. Tất cả
layout đều bắt đầu ở trạng thái `needs_visual_review`, kể cả khi độ phủ hình học
bằng 1,0. Đường cắt có thể vướng đường kẻ, chú thích hoặc nét chữ. Preview là công
cụ kiểm tra, không phải chứng nhận tách cột đúng hoặc OCR đã phủ hết văn bản.

Các vùng được gọi là `column` là **ứng viên hình học**, chưa chắc tương ứng từng
cột chữ thật. Phép chiếu có thể bỏ sót nhiều khe trên trang chữ dày; không dùng số
crop để kết luận số cột thật. Khi nội dung nối qua đường cắt, ưu tiên đối chứng
bằng ảnh toàn trang hoặc sửa đường cắt thủ công.

## 2. Cấu hình đường cắt và loại crop

Ví dụ file JSON cấu hình `ImageDatasetConfig`:

```json
{"column_cuts": null, "page_column_cuts": {"50": [0.5]}, "variants": ["original"]}
```

Khóa trang bắt đầu từ 1; vị trí cắt là tỷ lệ chiều ngang trong khoảng `(0, 1)`.
Đặt `[]` cho một trang để chủ động giữ toàn trang. Nếu mỗi sách cần đường cắt
riêng, chuẩn bị từng sách bằng cấu hình và thư mục dataset riêng.

Mỗi nhóm `page`, `group`, `column` tự tạo một phân hoạch đầy đủ của ảnh nguồn,
không bỏ lề hoặc xóa vùng bị cho là phi văn bản. `group` gộp tối đa ba vùng cột
ứng viên liền nhau, từ phải sang trái đối với chữ dọc. Trang đọc ngang mặc định
giữ nguyên. Không thêm overlap, không padding ra ngoài trang và không khử lặp
bằng cách so chuỗi chữ. Ảnh toàn trang luôn sẵn để đối chiếu các đường cắt chưa
chắc chắn.

`analysis_max_side` mặc định 1200 và nhận giá trị từ 32 đến 1600. Chỉ ảnh phân
tích bị thu nhỏ. Các đường cắt được đổi về pixel ảnh nguồn; crop chuẩn lấy trực
tiếp từ nguồn và không resize theo giới hạn của bất kỳ model OCR nào.

## 3. Chuẩn bị toàn bộ tập và tiếp tục khi gián đoạn

Sau khi xem mẫu nhỏ và chấp nhận cấu hình, có thể chủ động chạy toàn bộ:

```powershell
.\.venv\Scripts\vtm.exe images prepare --input-dir data --output-dir work/image_dataset --pages "" --limit 0 --max-books 0 --profile-dir configs/ocr_books --config configs/image_preparation_smoke.json
```

`--limit 0`, `--max-books 0` nghĩa là không giới hạn; `--pages ""` chọn toàn bộ
trang. Một số phiên bản PowerShell làm mất đối số rỗng; có thể dùng Python để
giữ lựa chọn này chính xác:

```python
from vtm_corpus.image_dataset import ImageDatasetConfig, prepare_image_dataset

result = prepare_image_dataset("data", "work/image_dataset",
    config=ImageDatasetConfig(), books=[], pages="", limit=None, max_books=None,
    profile_dir="configs/ocr_books")
print(result)  # Đường dẫn snapshot cố định và số bước thật sự được dùng lại.
```

Lặp lại đúng lệnh để resume. Nguồn, layout và biến thể có cache riêng theo hash
nội dung, cấu hình và phiên bản phần mềm; các file được kiểm tra trước khi dùng
lại. Thêm biến thể không bắt buộc lấy lại nguồn hoặc tìm lại layout. Thay mã/
thư viện sẽ chủ động làm mất hiệu lực các bước liên quan.

Nếu một bước lỗi, hệ thống không công bố snapshot thiếu trang. Những bước đã
xong vẫn dùng lại được ở lượt sau. Chỉ object, receipt và chỉ mục do workflow
sinh được cập nhật bằng ghi tạm rồi thay thế. Không ghi đè archive đã có và
không tự upload bất kỳ dữ liệu nào.

## 4. Biến thể ảnh

| Biến thể | Trạng thái và mục đích |
|---|---|
| `original` | Bắt buộc, mặc định; crop giữ nguyên pixel nguồn. |
| `bicubic_x2` | Thử nghiệm nội suy ×2; phép đổi về nguồn có tỷ lệ 0,5. Không phục hồi chi tiết đã mất. |
| `clahe` | Thử nghiệm tăng tương phản trên kênh độ sáng; clip 1,5, lưới 8 × 8. Cần OpenCV/NumPy đã cài sẵn. |

Dùng `configs/image_preparation_experimental.json` để tạo ba biến thể trên cùng
tập trang. Workflow không tự cài OpenCV hoặc tải model. Cần xem nét nhạt và chạy
đối chứng OCR với model/cấu hình cố định trước khi kết luận về chất lượng. Hiện
chưa có bằng chứng OCR để bật CLAHE hoặc nội suy mặc định.

Khử nhiễu, tăng nét, neural SR, deskew và dewarp chưa được triển khai: chúng có
thể làm mất nét yếu hoặc sinh chi tiết không có trong nguồn; phiên này chưa có
trọng số và bằng chứng OCR/ranh giới phù hợp. Cache nguồn/layout/biến thể được
tách để sau này bổ sung phương pháp phục hồi mà không phải đổi hình học crop
chuẩn. Không thêm PP-DocLayout thành model bắt buộc của bước chuẩn bị; layout
native của Paddle/TongGu vẫn có thể dùng ở bước OCR.

## 5. Cấu trúc lưu trữ và truy vết

```text
dataset.json                   # mô tả snapshot hoàn chỉnh mới nhất
snapshots/<hash>.json           # mô tả cố định để tái lập đầu vào
manifests/<hash>.jsonl          # một dòng cho mỗi trang/vùng/biến thể
pages/<hash>.json               # tham chiếu nguồn, layout và các ảnh vùng
sources/<input-hash>.json       # receipt dùng để tiếp tục lấy ảnh nguồn
sources/<input-hash>.acquisition.json
layouts/<input-hash>.json       # hình học, bằng chứng đường cắt, mask/preview
variants/<input-hash>.json      # receipt biến đổi ảnh
objects/<prefix>/<sha256>.png   # ảnh lossless, chỉ lưu một lần theo hash
```

Các vùng logic có ảnh giống hệt nhau dùng chung một PNG; không cần symlink hoặc
hardlink trên Kaggle. Tham chiếu là đường dẫn POSIX tương đối, có kiểm tra hash
và ngăn thoát khỏi thư mục dataset. JSONL tránh thêm Parquet/Arrow vào mỗi runtime
OCR; có thể đọc thành dataframe sau. Dataset không chứa dự đoán OCR, giới hạn
inference hoặc thiết lập riêng của model.

Manifest và metadata lưu danh tính PDF tương đối/SHA-256, chỉ số trang bắt đầu
từ 0 và số trang hiển thị bắt đầu từ 1, cách lấy nguồn, kích thước ảnh nhúng và
ma trận đặt ảnh, kích thước nguồn/phân tích/biến thể, loại vùng, thứ tự đọc, hộp
nguồn dạng nửa mở, tham số xử lý, phép biến đổi affine, cảnh báo QC, phiên bản
pipeline, hash mã và phiên bản thư viện.

### Giới hạn của việc bảo toàn nguồn

PDF gốc vẫn là nguồn có thẩm quyền; hãy giữ chúng cùng dataset ảnh. PNG RGB chuẩn
không được coi là bản sao stream nén, metadata màu hoặc độ sâu bit ban đầu trong
PDF. Nó bảo toàn pixel hiển thị đã kiểm tra. Trích trực tiếp chỉ được chấp nhận
khi bằng pixel với render; nếu không, dùng PDFium để giữ rotation, crop, clip,
mask và lớp phủ.

Trang chỉ chứa ảnh dùng mật độ pixel gốc quan sát được. Trang có vector/nội dung
hỗn hợp render ít nhất bằng độ phân giải yêu cầu và mật độ của các ảnh quan sát
được, không cap cạnh dài. `max_source_pixels` là giới hạn bộ nhớ chủ động: vượt
giới hạn thì báo lỗi, không âm thầm thu nhỏ. PDF có nhiều ảnh mật độ khác nhau
có thể cần xem lại giới hạn này. DPI vật lý chưa xác định nếu PDF dùng `/UserUnit`;
các phép đổi tọa độ sử dụng đơn vị canvas PDF.

## 6. Dùng dataset trong notebook 3

Tiếp tục gắn dataset PDF để xác minh SHA-256 và dùng luồng HTML/crop rà soát hiện
có. Trong ô cấu hình:

```python
PREPARED_IMAGES = "/kaggle/input/YOUR_IMAGE_DATASET"  # thư mục có dataset.json
PREPARED_VARIANT = "original"
PREPARED_CROP_TYPE = "group"  # cùng vùng chuẩn cho cả bốn model
PREPARED_MISSING_FALLBACK = False
PAGES = "2,50"
LIMIT = 2
```

`page`, `group`, `column` dùng chung phân hoạch để so sánh có kiểm soát. `native`
đưa toàn bộ trang đã chuẩn bị vào mỗi engine; Paddle/TongGu tự điều khiển layout
và ngân sách block. Processor vẫn có thể resize, kể cả khi đã dùng dataset.
Chỉ ghi kích thước tensor thật ở backend có hỗ trợ quan sát; crop trong dataset
không bị thay đổi theo inference.

Biến thể đã chọn phải tồn tại ở mọi vùng cần OCR. Với code local, đặt các trường
tương đương trong `OCRConfig(image_dataset=..., image_variant=...,
image_crop_type=...)`.

Trước khi tìm checkpoint, pipeline kiểm tra SHA-256 nguồn, trang, biến thể,
hình học và pixel. Cấu hình được gắn với một snapshot cố định; hash checkpoint
bao gồm snapshot/biến thể/loại crop/model. Khi không cấu hình dataset, hash của
workflow cũ được giữ nguyên.

Danh tính checkpoint hiện gắn với cả snapshot. Chuẩn bị một snapshot mới, kể cả
chỉ mở rộng tập trang hoặc thêm biến thể, tạo danh tính OCR mới; các checkpoint
cũ vẫn tồn tại ở run cũ. Đây là cách vô hiệu hóa bảo thủ để tránh trộn đầu vào.
Cache chuẩn bị nguồn/layout/biến thể vẫn được dùng lại độc lập. Muốn OCR bổ sung
trang mà giữ cùng snapshot, nên chuẩn bị trước tập trang đầy đủ rồi chỉ thay
`PAGES`/`LIMIT` ở notebook.

Trang chuẩn bị hợp lệ không render lại PDF hoặc phân tích lại layout. Trang
thiếu báo lỗi, trừ khi bật `PREPARED_MISSING_FALLBACK=True`; trang hỏng luôn báo
lỗi. Fallback cho trang thiếu được ghi rõ là `legacy_missing_page_fallback`.

Thứ tự crop chung xác định cách nối văn bản vùng ở **tất cả** model. Khung thật
do model cung cấp và offset ký tự được chuyển về pixel trang nguồn; raw theo vùng,
hash biến thể và metadata nằm trong `image_preparation`. Cách nối này có thể khác
thứ tự layout native, vì vậy cần đối chứng `group` với `native`. Bộ nạp model,
giao dịch theo trang, timeout/restart, so sánh, raw output và HTML vẫn được giữ.
Notebook dịch số 2 không thay đổi.

Sau khi sửa mã nguồn, sinh lại các bản mã trong notebook:

```powershell
.\.venv\Scripts\python.exe -X utf8 tools/update_kaggle_notebook_bootstrap.py
```

## 7. Đóng gói để đưa lên Kaggle

```powershell
.\.venv\Scripts\vtm.exe images package work/image_dataset work/image_dataset_kaggle.zip
```

Lệnh kiểm tra snapshot rồi chỉ đưa các file được tham chiếu cùng descriptor cố
định vào ZIP. Thứ tự file và timestamp được cố định để tái lập; PNG không nén lại.
Không đóng gói snapshot cũ, biến thể không dùng, receipt, khóa, PDF gốc hoặc OCR
output. Archive đã tồn tại sẽ bị từ chối ghi đè.

Giải nén ZIP để tạo Kaggle Dataset, gắn vào notebook 3 và trỏ `PREPARED_IMAGES`
đến thư mục gốc vừa giải nén. **Upload là bước thủ công, chưa được thực hiện.**

Các crop/biến thể làm tăng dung lượng và số file dù đã loại bỏ ảnh trùng. Với
corpus lớn, có thể chia dataset/package theo sách hoặc batch. Manifest vẫn là
JSONL đọc offline thông thường.

## 8. Tái lập phép thử chuẩn bị ảnh

Phép thử giới hạn 12 trang, không chạy OCR:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/benchmark_image_preparation.py
# Thử thêm biến thể trong thư mục kết quả riêng:
.\.venv\Scripts\python.exe -X utf8 scripts/benchmark_image_preparation.py --experimental --output-dir work/image_dataset_variants
```

Số đo JSON và contact sheet được lưu trong `work/`, ngoài Git. Các kết quả tách
biệt bảo toàn pixel nguồn, kích thước đầu vào, rủi ro đường cắt, thời gian, resume
và lưu trữ khỏi độ phủ văn bản OCR. Chúng không đo CER và không chứng minh OCR
chính xác hơn. Đối chứng bốn model trên GPU là một bước kiểm chứng riêng.
