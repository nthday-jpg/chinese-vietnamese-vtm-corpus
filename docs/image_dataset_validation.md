# Kết quả kiểm chứng dataset ảnh

Ngày thực hiện: **10/11/2026**, Windows, `.venv` dùng chung của repo.
Python 3.13.14, Pillow 13.0.0, pypdfium2 5.14.0, PDFium 156.0.8076.0.
Không tải trọng số hoặc chạy inference OCR trên GPU.

## Phạm vi đã chạy

Chọn **12 trang**: trang 2 và 50 của `1`, `2`, `4`, `6`, `bien`, `kinh`.
Tập này có scan ít pixel/nét nhạt, chữ dọc dày, ghi chú màu đỏ xen kẽ, trang có
con dấu/trang trí và trang đọc ngang. `kinh/p50` tương đối rõ hơn các bản scan
600–650 pixel; `4/p50` mờ và có nhiều nét yếu. Đây là lựa chọn đại diện bằng
quan sát, không phải các nhãn chất lượng được đo bằng ground truth.

Chạy riêng cấu hình chỉ `original`, rồi cấu hình thử nghiệm gồm `original`,
`clahe`, `bicubic_x2`; mỗi cấu hình có thêm một lượt resume. Đã xem contact sheet
của 12 trang và crop chi tiết đối chiếu biến thể trên `4/p50`.

## Kết quả lấy ảnh và kích thước

| Sách, trang 2/50 | Render cũ | Ảnh nguồn mới | Cách lấy nguồn |
|---|---|---|---|
| 1 | 600 × 450 | 600 × 450 | Native đã kiểm tra bằng pixel với trang hiển thị |
| 2 | 603 × 461 | 600 × 458 | Native đã kiểm tra bằng pixel |
| 4 | 604 × 535 | 600 × 532 | Native đã kiểm tra bằng pixel |
| 6 | 653 × 423 / 653 × 404 | 650 × 421 / 650 × 402 | Native đã kiểm tra bằng pixel |
| bien | 1534 × 2333 | 737 × 1120 | Render để giữ cách PDF hiển thị ảnh |
| kinh | 1595 × 2206 | 1595 × 2206 | Render để giữ object đường vẽ và scan |

Tổng cộng **8 trang dùng native đã xác minh, 4 trang dùng render**.
Hai trang sách 1 và hai trang `kinh` có ảnh mới bằng pixel hoàn toàn với render
cũ cùng kích thước. Các sách 2/4/6 tránh nội suy nhẹ ở DPI profile. `bien` tránh
phóng lớn ảnh nhúng 736 × 1120 lên gần 2 lần mỗi chiều; render cần thêm một cột
pixel do khổ PDF thập phân và tỷ lệ đặt ảnh. Việc này không phải phục hồi nét.
Không dùng giải mã thô của `bien` vì nó không khớp hình PDF hiển thị.

Chỉ ảnh phân tích của `kinh` được giảm còn 868 × 1200. Crop của nó vẫn lấy từ
nguồn 1595 × 2206; không lấy từ thumbnail và không cap theo model khi lưu dataset.

## Layout: những gì đúng và những gì chưa xác nhận

Mẫu chỉ `original` có 12 vùng toàn trang, 22 nhóm và 51 vùng cột ứng viên,
tổng 85 dòng manifest. QC xác nhận mỗi họ crop có hộp hợp lệ, thứ tự nhất quán,
phân hoạch đủ trang và các crop `original` bằng pixel với phần tương ứng của
nguồn. Không xóa lề, chú thích hoặc vùng không chắc chắn.

**Không thể coi 51 vùng ứng viên là 51 cột chữ thật.** Preview cho thấy:

- `1/p50`, `6/p50` có nhiều khe phù hợp với cột dọc, nhưng vẫn cần rà soát nét
  nhạt/chú thích. Đường cắt x=82 của `1/p50` được gắn cờ hỗ trợ yếu ở một phía.
- `2/p2`, `2/p50` chỉ tìm được một phần khe; từng vùng ứng viên vẫn chứa nhiều
  cột thật. Không có pixel bị bỏ, nhưng mục tiêu “mỗi crop một cột chữ” chưa đạt
  trên các trang này.
- `4/p50`, `6/p2` giữ nguyên trang khi không có khe rõ. Đây là phương án bảo toàn
  nội dung, chưa phải bằng chứng layout đơn giản.
- `kinh/p50` chỉ phát hiện khe phía mép thay vì các cột trong khung. Đường cắt
  x=1411 được gắn cờ có mực ở đường cắt; chưa phân biệt đó là khung hay nét chữ.
- `kinh/p2` có tiêu đề và cụm trang trí; nhóm hình học không được xem là layout
  ngữ nghĩa đã xác nhận. `bien` giữ toàn trang theo hướng đọc ngang.

Do đó chưa có cơ sở đổi detector mặc định hoặc tự chấp nhận mọi đường cắt.
Tất cả layout vẫn là `needs_visual_review`; có thể chọn `page`/`native` hoặc sửa
`page_column_cuts`. Phủ đủ pixel là **độ phủ hình học**, không phải độ phủ văn bản
OCR hay chứng nhận thứ tự đọc đúng trên trang có bố cục phức tạp.

## Thời gian và lưu trữ

Các số dưới đây là thời gian thực của một lượt trên máy local, không phải
benchmark thống kê hoặc dự báo tốc độ Kaggle:

| Cấu hình | Chuẩn bị lần đầu | Resume | Dòng manifest | PNG duy nhất | Tổng thư mục dataset |
|---|---:|---:|---:|---:|---:|
| Chỉ `original` | 7,106 giây | 1,405 giây | 85 | 96 | 12.756.241 byte, khoảng 12,17 MiB |
| Ba biến thể | 16,889 giây | 6,814 giây | 255 | 240 | 51.177.227 byte, khoảng 48,81 MiB |

Ở lượt resume, cả 12 nguồn và 12 layout được dùng lại, tương ứng 85 hoặc 255
ảnh vùng/biến thể. Snapshot trước và sau resume giống nhau trong từng cấu hình.
Thời gian vẫn gồm đọc/giải mã, kiểm tra hash, hình học và pixel crop.

Luồng cũ render + lập vùng mẫu Hunyuan mất tổng khoảng 0,927 giây trên cùng 12
trang, nhưng không tạo/lưu dataset, các biến thể và toàn bộ QC ở trên. Vì khối
lượng công việc khác nhau, **không kết luận bước chuẩn bị mới nhanh hơn lần
render cũ**. Lợi ích đã xác minh là các model có thể đọc lại đầu vào cố định mà
không render/phân tích lại; chi phí tạo bộ ảnh được trả một lần.

Thử nghiệm tăng số ảnh và dung lượng đáng kể. Loại trùng theo hash tránh lưu lặp
ảnh toàn trang/group/column giống nhau; vẫn còn nhiều file nhỏ. Chưa đo corpus
đầy đủ hoặc giới hạn file/dung lượng thực tế của Kaggle.

## Kích thước sau processor: kiểm tra thật, chưa chạy OCR

Đã chạy `HunYuanVLImageProcessorPil` trên CPU, không tải weights, với
`shortest_edge=262144`, `longest_edge=4194304`. Kích thước sau processor được lấy
từ `image_grid_thw` của tensor thật:

| Vùng | Biến thể lưu trong dataset | PNG đưa vào processor sau ngân sách backend | Kích thước sau processor |
|---|---|---|---|
| 1/p50, toàn trang | `original`: 600 × 450 | 600 × 450 | 608 × 448 |
| 1/p50, toàn trang | `bicubic_x2`: 1200 × 900 | 1200 × 900 | 1216 × 896 |
| 1/p50, nhóm đầu | `original`: 140 × 450 | 140 × 450 | 288 × 928 |
| 1/p50, nhóm đầu | `bicubic_x2`: 280 × 900 | 280 × 900 | 288 × 928 |
| kinh/p50, toàn trang | `original`: 1595 × 2206 | 1595 × 2206 | 1600 × 2208 |
| kinh/p50, toàn trang | `bicubic_x2`: 3190 × 4412 | 1741 × 2408 | 1728 × 2400 |

Nội suy ×2 không bảo đảm tăng kích thước cuối: nhóm đầu sách 1 nhận cùng kích
thước tensor vì ngân sách tối thiểu; toàn trang `kinh` ×2 lại bị giảm vì ngân sách
tối đa. Dataset vẫn giữ nguyên các ảnh lớn tương ứng. Bảng này chỉ xác nhận
hành vi tiền xử lý, chưa đo văn bản nhận dạng hoặc khả năng chạy model trên GPU.
Tensor nội bộ của ba backend còn lại chưa được quan sát trong phiên này.

CLAHE làm tương phản nét/nền mạnh hơn trên crop đã xem; bicubic thay đổi cách
nội suy. Chưa có ground truth để kết luận chúng giúp đọc đúng nét yếu. Cả hai
tiếp tục là thử nghiệm, không thay `original` làm tham chiếu.

## Kiểm thử tự động và tích hợp

Lệnh kiểm thử:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -q
```

Kết quả cuối: **279 test được báo cáo, `OK (skipped=1)`**, thời gian 58,180 giây.
Có 19 test mới cho dataset; các kiểm thử chuẩn bị ảnh, notebook, checkpoint,
timeout/restart và luồng dummy hiện có tiếp tục đạt. Nhánh Unix alarm được bỏ qua
trên máy Windows.

Các tình huống mới đã chạy bằng PDF fixture thật và ảnh có nội dung:

- Native bằng pixel; nguồn ít pixel không bị giảm theo DPI yêu cầu thấp.
- Rotation 90/180/270, phản chiếu, crop, clip, nhiều ảnh, lớp phủ vector, annotation,
  alpha mask và `/Decode`; giữ đúng hình PDF hiển thị hoặc từ chối trích thô.
- Giới hạn bộ nhớ báo lỗi thay vì thu nhỏ nguồn; ma trận pixel → PDF cho các hướng.
- Đổi tọa độ qua thumbnail có tỷ lệ lẻ, tái ghép crop bằng pixel, nhóm từ phải
  sang trái, giữ toàn trang ngang và kiểm tra đường cắt xuyên glyph tổng hợp.
- Thêm biến thể dùng lại nguồn/layout; kiểm tra kích thước và affine của biến thể.
- Resume không gọi lại lấy nguồn/layout/biến đổi; khôi phục PNG cache bị hỏng;
  gián đoạn không công bố snapshot thiếu trang; phát hiện hash/hình học sai.
- Đóng ZIP hai lần cho byte giống nhau, kiểm CRC, giải nén ở vị trí mới rồi QC;
  ngăn đường dẫn thoát root và từ chối ghi đè archive.
- Worker tự chứa chạy ở tiến trình con, đọc các cột chuẩn mà renderer bị đặt để
  báo lỗi nếu được gọi; lần resume không nạp lại model giả. Khung/offset từ model
  giả được chuyển đúng về pixel nguồn ngay cả khi biến thể và input bị resize.
- Chế độ native gửi một ảnh toàn trang, không áp dụng cap ngoài cho block native;
  fallback chỉ cho trang thiếu và không che lỗi file hỏng; hash workflow cũ giữ
  nguyên, snapshot/variant/crop/model mới tạo danh tính checkpoint khác.
- Mã notebook được sinh khớp nguồn và có đúng các thiết lập dataset đã công bố.

CLI thật đã chạy prepare/validate hai trang `1/p2,p50`. Gói thử nghiệm Kaggle đã
tạo thành công: **279 file, 51.129.392 byte**, không có PDF, trọng số hoặc OCR output.
Gói đã được giải nén và kiểm tra lại toàn bộ 12 trang.

SHA-256 notebook 2 vẫn là:
`c3cbef76e0eafb8010b8f0a407099b1b1dbd49850a0da2519432d8bc6b35954f`.

## Thay đổi mã và tương thích

- `src/vtm_corpus/pdf_images.py`: lấy nguồn theo trang, kiểm tra native và giữ
  hình PDF khi phải render; ghi bằng chứng hình học và quyết định.
- `src/vtm_corpus/image_dataset.py`: cấu hình, cache riêng từng bước, nguồn/layout/
  crop/biến thể, QC, snapshot, manifest, đọc đầu vào OCR và đóng gói.
- `src/vtm_corpus/image_preparation.py`: dùng lại mask phép chiếu, đọc crop/biến
  thể đã chuẩn bị và giữ phép đổi tọa độ/raw của luồng OCR hiện có.
- `config.py`, `corpus_ocr.py`, `corpus_worker.py`, `ocr_worker.py`, `pipeline.py`,
  `cli.py`: thiết lập dataset, gắn snapshot trước checkpoint, đọc đầu vào đã chuẩn
  bị, ghi fallback rõ ràng và bổ sung các lệnh `vtm images`.
- `tools/notebook_source_builder.py`, `tools/rebuild_plain_ocr_notebook.py` và
  notebook 3: đưa cùng mã nguồn vào hàm notebook/worker; thêm lựa chọn dataset,
  biến thể và loại crop mà vẫn giữ luồng render cũ.
- `tests/test_image_dataset.py`, `scripts/benchmark_image_preparation.py`, hai
  cấu hình `configs/image_preparation_*.json`, tài liệu tiếng Việt và liên kết
  README: kiểm chứng và cách tái lập. `.gitignore` cho phép đưa các file cần
  review vào Git, vẫn bỏ qua ảnh/PDF/weights/archive và output trong `work/`.

Không thay bộ nạp bốn model, notebook dịch hoặc triển khai downstream VLM/ngữ
nghĩa. Các thay đổi Page-level Consensus có sẵn được giữ lại. Danh tính OCR gắn
với cả snapshot theo cách bảo thủ: snapshot mới tạo run mới; cache chuẩn bị vẫn
độc lập và dùng lại các bước chưa đổi. Nên chuẩn bị tập trang đầy đủ trước rồi
thay `PAGES`/`LIMIT` khi muốn OCR bổ sung trong cùng snapshot.

## Bằng chứng và phần chưa thực hiện

Các artifact local nằm ngoài Git:

- `work/image_dataset_final_smoke/measurements.json` và `previews.png`.
- `work/image_dataset_variants/measurements.json`, `previews.png`, `variant_detail.png`
  và `processor_observations.json`.
- `work/image_dataset_variants/kaggle_input.zip`.

Có thể tái lập bằng script/lệnh trong [hướng dẫn sử dụng](image_dataset.md).
**Chưa chạy OCR thật bốn model, chưa đo CER/độ chính xác, chưa chạy toàn corpus,
chưa xác nhận inference trên RTX PRO 6000/Kaggle và chưa upload.**
Không có kết luận chất lượng OCR tăng lên từ số ký tự, độ sắc hoặc độ phủ pixel.
