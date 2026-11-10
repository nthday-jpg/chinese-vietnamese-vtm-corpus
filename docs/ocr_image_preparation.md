# Tách layout/cột và kiểm soát độ phân giải OCR

Ngày kiểm tra: 2026-11-09. Chưa có điểm CER/ground truth hoặc inference bốn model
để khẳng định cấu hình tối ưu về độ chính xác. Các thay đổi kiểm soát mất chi tiết
do resize, bảo toàn nguồn và làm rõ dữ liệu mỗi model nhận.

## Kết quả kiểm tra PDF

Đã quét hình học và ảnh scan toàn trang của **1.133 trang / 8 PDF** bằng PDFium.
Các profile hiện tại không cap thấp hơn ảnh render tại DPI đã chọn và không render
dưới độ phân giải scan toàn trang được phát hiện. Không cần tăng đồng loạt DPI.

| PDF | Trang | DPI / cap cạnh dài giữ lại | Cạnh dài scan nhúng lớn nhất | Hướng đọc mặc định |
|---|---:|---:|---:|---|
| 1 | 100 | 225 / 750 | 600 | Cột dọc, phải sang trái |
| 2 | 103 | 226 / 754 | 600 | Cột dọc, phải sang trái |
| 3 | 103 | 226 / 754 | 600 | Cột dọc, phải sang trái |
| 4 | 70 | 181 / 704 | 700 | Cột dọc, phải sang trái |
| 5 | 70 | 181 / 704 | 700 | Cột dọc, phải sang trái |
| 6 | 96 | 226 / 653 | 650 | Cột dọc, phải sang trái |
| bien | 422 | 200 / 2333 | 1120, 421/422 trang có scan toàn trang | Ngang, trái sang phải |
| kinh | 169 | 45 / 2206 | 2185 | Cột dọc, phải sang trái |

Đây là số đo hình học/ảnh nhúng; không chứng minh chữ đọc được hoặc đúng thứ tự
trên mọi trang. `kinh` khai báo khổ trang rất lớn nên DPI 45 vẫn cho ảnh 1595×2206.
`bien` giữ DPI 200 vì một trang không có bằng chứng scan phủ toàn trang; giảm theo
scan của các trang khác có thể ảnh hưởng nội dung vector/chữ ở trang đó.

PDF 1–6 vốn ít pixel: nội suy lớn hơn có thể phù hợp processor nhưng không phục
hồi nét. Không tự tăng nét, nhị phân hóa, khử nhiễu, xoay hoặc xóa lề.

## Ba lớp độ phân giải

1. `dpi`, `max_side`: ảnh trang được render và cache PNG. Crop luôn lấy từ ảnh này.
2. `region_max_pixels`, `region_max_side`, `region_upscale`: kích thước PNG gửi backend.
3. `model_min_pixels`, `model_max_pixels`: ngân sách processor/recognizer hỗ trợ chúng.

Tăng lớp 1 không ngăn lớp 3 thu nhỏ. Ngược lại, `min_pixels` có thể phóng lớn crop.
Code ghi cả kích thước crop lẫn ảnh backend nhận. Hunyuan đọc `image_grid_thw`
và `pixel_values.shape` của tensor thật để ghi kích thước sau processor.

Ví dụ chạy **processor PIL Transformers 5.22.1 trên CPU, không tải weights**, dùng
ngân sách min 262144 / max 4194304: trang 1/p50 600×450 → 608×448; crop 93×450
→ 256×1152; kinh/p50 1595×2206 → 1600×2208. Các trường hợp này không cho thấy
thu nhỏ mạnh do vượt max_pixels; sai khác gồm làm tròn patch và phóng lớn crop.

[Config Hunyuan ở revision notebook dùng](https://huggingface.co/tencent/HunyuanOCR/blob/47644ecc4fc854efa4f505155158831f36773ee4/preprocessor_config.json)
chứa cả `max_pixels=16777216` và `size.longest_edge=4194304`.
Processor đã cài ưu tiên trường tương thích cũ khi nạp và cho effective max 16777216.
Không suy diễn từ tên `longest_edge` rằng đây là chiều dài pixel: trong processor này
nó là ngân sách diện tích. Notebook đặt rõ min/max và log effective size/tensor.
Khác phiên bản Transformers cần kiểm tra lại log này.

## Model và chế độ tự động

- **Hunyuan**: `auto` tách nhóm cột dọc theo khe ít mực, mặc định 3 cột mỗi vùng.
  Không đủ bằng chứng khe thì giữ nguyên. Trang ngang chỉ tách thêm nếu vượt budget.
  Ngân sách khởi đầu min 262144 / max 4194304; không phải tuyên bố giới hạn kiến trúc.
- **PP-OCR**: `auto` chia vùng vượt budget 4194304 pixel. Khi bật chuẩn bị ảnh,
  yêu cầu detector dùng cap cạnh dài bằng cạnh crop, làm tròn lên bội 32, loại `max`.
  Recognition dòng vẫn có kích thước nội bộ theo model đã huấn luyện.
- **PaddleOCR-VL**: `auto` giữ ảnh trang để pipeline phát hiện layout rồi nhận dạng
  block. Gửi rõ `min_pixels=200704`, `max_pixels=1048576` vào `predict`.
- **TongGu**: `auto` giữ detector/block segmentation gốc; truyền rõ
  `image_min_pixels=1024`, `image_max_pixels=1048576` vào inference upstream.

Paddle/TongGu đã có layout nội bộ, vì vậy không ép tách hai lần trong `auto`.
Muốn kiểm tra external segmentation cho chúng, đặt `segmentation="columns"`
qua `ADVANCED["model_overrides"]`. Khi đó ngân sách PNG mặc định là 1048576.
Ngân sách model là lựa chọn thực nghiệm có thể thay đổi, chưa đo VRAM/accuracy.

Native engine chưa lộ tensor resize nội bộ qua adapter: log ghi
`native_internal_unavailable`, chỉ xác nhận PNG gửi backend và các tham số yêu cầu.
Không diễn giải chúng thành kích thước tensor đã đo. Native detector/recognizer
vẫn có thể resize block; kiểm tra output thật trước khi chọn cấu hình tốt nhất.
Đối chiếu API chính thức: [PaddleOCR-VL](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md),
[TongGu preprocessing](https://github.com/jzzh2004/TongGuOCR/blob/main/tongguocr/ocr/preprocess.py),
[TongGu backend](https://github.com/jzzh2004/TongGuOCR/blob/main/tongguocr/ocr/transformers_backend.py).

## Preview ngay trong notebook, chưa chạy OCR

Sau các ô định nghĩa hàm, chạy riêng ví dụ sau; không cần setup model hoặc weights:

```python
cfg = OCRConfig(engine="hunyuan", **BOOK_PROFILES["1.json"], **IMAGE_PREPARATION)
preview, audit = preview_preparation("data/1.pdf", 50, cfg)  # trang PDF bắt đầu từ 1
display(preview)  # đường cắt + số thứ tự vùng
display(audit)    # crop_size, input_size, scale, warnings
```

Trên Kaggle thay đường dẫn bằng PDF trong `DATA_DIR`. Preview chỉ biết PNG/crop;
chỉ lượt inference mới quan sát tensor Hunyuan. Xem cả trang chữ nhỏ, chữ mờ,
chú thích hai hàng, bảng, bìa và trang ngang. Hai trang đầu thường chưa đại diện.

Tùy chỉnh trong `IMAGE_PREPARATION`:

```python
IMAGE_PREPARATION["segmentation"] = "auto"  # none: baseline cả trang; columns: ép dò cột
IMAGE_PREPARATION["segmentation_args"] = {
    "columns_per_region": 3,
    "max_regions": 64,
    "page_column_cuts": {"50": [0.45, 0.70]},  # phần trăm chiều ngang, trang 1-based
}
IMAGE_PREPARATION["region_upscale"] = 1.0
IMAGE_PREPARATION["resolution_strict"] = False
ADVANCED["model_overrides"] = {
    "hunyuan": {"model_min_pixels": 262144, "model_max_pixels": 4194304},
    "ppocr": {"region_max_side": 2048},
}
```

`column_cuts` áp dụng mọi trang, `page_column_cuts` ưu tiên theo số trang.
Danh sách `[]` giữ nguyên trang cụ thể; ranh giới là tỷ lệ 0–1 tăng dần.
Thủ công có thể cắt qua chữ, nên luôn preview. Có thể đổi `reading_direction`
qua model override khi bỏ profile, hoặc sửa riêng `BOOK_PROFILES` tương ứng.
Profile vẫn ưu tiên các trường của sách như DPI/hướng đọc; bỏ bằng
`ADVANCED["use_book_profiles"]=False` nếu muốn toàn bộ cấu hình thủ công.

## Bảo toàn nguồn và giới hạn

Bộ tách là **heuristic hình học bằng phép chiếu**, chưa phải detector layout ngữ
nghĩa được huấn luyện. Chỉ ảnh phân tích được chuẩn hóa nền/che ảnh hưởng đường
kẻ; ảnh OCR vẫn là crop màu gốc. Trang có chữ xiên, tiêu đề vắt nhiều cột, chú
thích xen kẽ hoặc đường kẻ cần xem trước và có thể tắt tách theo trang.

Các vùng phủ đủ ảnh trang, không chồng lặp, không bỏ lề và không tự deduplicate chữ.
Nhóm cột theo phải→trái, trong cột trên→dưới. Không chia ngang nhóm nhiều cột
dọc lớn khi thiếu khe dọc, vì có thể phá thứ tự đọc. Hướng đọc là giả định theo
profile, chưa phải chứng nhận thứ tự đúng trên mọi trang.

Nếu không thể chia đủ nhỏ ở khe đáng tin cậy: mặc định resize bằng LANCZOS và
ghi `unsplittable_region_exceeds_budget`, `downsampled` cùng cảnh báo. Muốn không
chấp nhận điều đó, bật `resolution_strict=True`: trang được ghi lỗi thay vì output
thành công. Strict cũng phát hiện cap render và Hunyuan processor thu nhỏ, kể cả
làm tròn patch; không xác nhận nội bộ native engine chưa quan sát được.

`region_upscale` từ 1 đến 4 chỉ là nội suy có chủ đích; mặc định 1. Tăng nó không
tạo nét thật và có thể làm processor resize thêm. `max_new_tokens` vẫn là trần
theo lần nhận dạng vùng; trang có bất kỳ vùng nào chạm trần được gắn `hit_max`.

Mọi raw prediction/text theo vùng nằm trong `image_preparation.regions`.
Khung chữ/dòng do model cung cấp được scale/offset về pixel trang gốc và chuyển
offset ký tự cùng văn bản. Không chia đều khung dòng để bịa tọa độ chữ.
Một vùng lỗi làm cả trang thất bại, không ghi phần đầu như trang hoàn chỉnh.
Timeout vẫn tính trên cả trang; tách nhiều vùng có thể cần tăng `page_timeout`.

Thay layout/ngân sách/min-max processor tạo cache identity khác để không trộn với
checkpoint cũ. PNG trang vẫn dùng chung; crop chạy trong thư mục tạm riêng.
CLI một model dùng `--set segmentation=auto`; `OCRConfig` mặc định `none` để
giữ tương thích. Notebook bốn model bật `auto` qua `IMAGE_PREPARATION`.

## Kiểm chứng

Đã kiểm tra hình học đủ 1.133 trang, render/preview hai trang mỗi PDF (2 và 50),
và xem trực quan mẫu manuscript, trang có khung/cột, trang ngang. Có trang được
giữ nguyên khi không có khe rõ; không khẳng định tách đúng mọi trang của corpus.
Kiểm thử bao gồm tái ghép crop bằng pixel, ngân sách, thứ tự/offset, khung trang,
thất bại giữa trang, tensor processor Hunyuan thật với model giả, và worker tự chứa
mã chạy PDF/crop/checkpoint/resume ở tiến trình con. Không tải weights hoặc chạy
inference OCR thật, không chạy corpus trên GPU. Notebook 2 được giữ nguyên.

So chất lượng bằng cùng tập trang và cùng ground truth, lần lượt `none` và `auto`;
đọc raw theo vùng, `hit_max`, vùng thiếu và cảnh báo. Thống nhất giữa các model
không thay thế điểm chính xác. Chỉ chạy cả sách sau smoke trên trang đại diện.

Sau OCR, [Page-level consensus](page_level_consensus.md) đối chiếu cấu trúc đầu ra
trước khi bỏ phiếu ký tự. Lớp này dùng log đã có; không phải detector ảnh mới và
không cần chạy lại model. Nó giúp phát hiện các dấu hiệu mà số ký tự đơn thuần
không thể phân biệt, như đảo cột với tổng số chữ không đổi.
