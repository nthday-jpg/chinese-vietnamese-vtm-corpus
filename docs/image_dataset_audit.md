# Báo cáo rà soát pipeline chuẩn bị ảnh PDF

Ngày kiểm tra: **10/11/2026**. Nhánh: `mduy`.

Các phát hiện và quyết định dưới đây được ghi nhận trước khi sửa phần lấy ảnh,
quy hoạch vùng và tích hợp worker. Những thay đổi Page-level Consensus có sẵn
trong workspace được giữ lại.

## Những gì đã xác nhận

### Render và độ phân giải nguồn

`ocr_worker.render()` raster hóa theo DPI, sau đó áp dụng `max_side` **trước khi
tách cột**. Hàm đã ghi kích thước trước/sau và cờ thu nhỏ. Tuy nhiên, không thể
kết luận cấu hình hiện tại luôn làm mất chi tiết: `1.pdf`, DPI 225 cho ảnh
600 × 450; cap 750 không làm ảnh nhỏ đi. Tăng DPI chỉ nội suy thêm từ scan này.

Đã trực tiếp kiểm tra trang 2 và 50 của các sách sau:

| Sách | Kích thước ảnh nhúng quan sát được |
|---|---|
| 1 | 600 × 450 |
| 2 | 600 × 458 |
| 4 | 600 × 532 |
| 6 | 650 × 421 và 650 × 402 |
| bien | 736 × 1120 |
| kinh | 1504 × 2176 và 1504 × 2185 |

Đây là kích thước nguồn, chưa phải kích thước tensor sau processor hoặc bằng
chứng OCR chính xác. Các profile không cần tăng DPI đồng loạt.

`bien.pdf` chứng minh đếm object và kiểm tra vị trí vẫn chưa đủ để trích ảnh:
hai trang mẫu chỉ có một ảnh phủ trang, nhưng pixel giải mã thô khác rất lớn so
với hình thực tế PDF hiển thị; sai khác tuyệt đối trung bình khoảng 249–253/255.
Cần giữ đúng mask và cách diễn giải màu. `kinh.pdf` có thêm một object đường vẽ
bên cạnh scan; bỏ qua object này có thể thay đổi trang. Kích thước PDF dạng số
thập phân còn có thể làm render lệch một hàng pixel.

### Tách cột và ngân sách model

`projection_gaps()` dùng ảnh xám thu nhỏ, chuẩn hóa nền cục bộ và giảm ảnh hưởng
đường kẻ trong phép chiếu. Pixel gửi OCR vẫn lấy từ ảnh màu gốc. `plan_regions()`
giữ lề, tạo phân hoạch đầy đủ không chồng lặp, nhóm từ phía bắt đầu đọc, hỗ trợ
đường cắt thủ công và giữ layout riêng của Paddle/TongGu trong chế độ `auto`.
Đây là những hành vi phù hợp cần bảo toàn. **15 kiểm thử chuẩn bị ảnh hiện có
đều đạt trước khi sửa mã.**

Độ phủ hình học không xác nhận ranh giới chữ. Trước thay đổi này chưa có báo cáo
độ ổn định đường cắt theo ngưỡng hoặc lượng mực tại đường cắt. Trang có đường kẻ,
chú thích, chữ nghiêng và nét nhạt vẫn cần kiểm tra trực quan. Chưa có bằng chứng
đủ để thay toàn bộ phép chiếu bằng một thuật toán khác.

`plan_regions()` gọi `image_budget()`, vì vậy cách chia vùng có thể phụ thuộc
engine và giới hạn processor. `prepare_region()` tiếp tục resize nếu cần. Cách
này phù hợp khi inference, nhưng không phù hợp để tạo bộ crop chuẩn dùng chung
cho cả bốn model.

`run_prepared_page()` đã chuyển tọa độ thật do model cung cấp và offset ký tự về
pixel trang, giữ raw theo vùng và hủy thành công toàn trang nếu một vùng bị lỗi.
Cần giữ các đặc tính này. Hunyuan có quan sát tensor sau processor; các pipeline
native ghi rõ khi không quan sát được kích thước tensor nội bộ.

### Cache, checkpoint và mã notebook

`corpus_page_image()` đã có cache PNG dùng chung giữa model, kiểm tra hash và
cấu hình render. OCR đã hỗ trợ tiếp tục các trang hoàn tất theo danh tính nguồn
và cấu hình. Các crop tạm bị xóa sau inference. Phần còn thiếu là dataset bền
vững gồm nguồn, layout và biến thể ảnh để gắn vào Kaggle như một input độc lập.
Luồng render/checkpoint hiện tại cần được giữ làm phương án tương thích.

`src/` là nguồn mã chính. `tools/notebook_source_builder.py` dùng AST để ghép các
module thành hàm trong notebook và một worker độc lập. Các ô notebook là bản mã
được sinh, không phải triển khai có thẩm quyền riêng. Phải sửa nguồn/builder rồi
sinh lại notebook 3; notebook 2 không thuộc phạm vi công việc.

## Quyết định kỹ thuật

1. **Dùng PDFium/Pillow hiện có.** Chỉ chấp nhận pixel trích trực tiếp khi trang
   là một ảnh đơn giản, không xoay, phủ đúng trang, vượt qua kiểm tra hình học
   và **bằng pixel hoàn toàn với một render của trang hiển thị**. Nếu không,
   render không cap cạnh dài để giữ mask, lớp phủ, chú thích, crop và rotation.
   Ghi lại kích thước, lý do lựa chọn, phiên bản thư viện và phép đổi tọa độ PDF.
2. **Giữ phép chiếu làm đường cơ sở.** Phân tích trên thumbnail riêng; đổi tọa độ
   page/group/column về ảnh nguồn và không dùng giới hạn model để tạo crop chuẩn.
   Bổ sung lượng mực, độ rộng khe, hỗ trợ hai phía và độ ổn định theo ngưỡng.
   Đường cắt chưa chắc chắn được gắn cờ; ảnh toàn trang luôn được giữ.
3. **Mặc định chỉ tạo `original`.** CLAHE và bicubic ×2 chỉ là thử nghiệm có chủ
   đích. Không bật khử nhiễu, SR, tăng nét, deskew, dewarp hoặc detector mới khi
   chưa có bằng chứng, trọng số và kiểm chứng ranh giới phù hợp.
4. **Tách cache nguồn, layout và biến thể.** PNG được lưu theo hash để tránh lặp;
   JSON/JSONL đọc được trong các runtime Kaggle mà không thêm Parquet/Arrow.
   Chỉ công bố snapshot sau khi toàn bộ trang dự kiến vượt qua QC. Resume kiểm
   tra các file và hash; bổ sung biến thể dùng lại nguồn và hình học đã có.
5. **Tích hợp dataset theo lựa chọn rõ ràng.** Kiểm tra nguồn/trang/biến thể trước
   khi dùng checkpoint; đưa snapshot, biến thể và loại crop vào danh tính OCR.
   Giữ chế độ native toàn trang. Trang hỏng phải báo lỗi; chỉ fallback cho trang
   chưa chuẩn bị khi người dùng bật tùy chọn tương ứng.

## Những hướng chưa đủ bằng chứng

PP-DocLayout là một hướng thử nghiệm có cơ sở: tài liệu upstream có nhắc sách cổ
trong dữ liệu huấn luyện và chữ dọc ở các phiên bản layout mới. Tuy nhiên, điều
đó chưa chứng minh detector tìm đúng ranh giới và thứ tự đọc trên các scan này.
Không có lý do để thêm nó thành dependency bắt buộc của bước chuẩn bị ảnh.

Chưa có kiểm chứng CER/ground truth để khẳng định CLAHE, nội suy hoặc SR cải thiện
OCR. Chữ nhìn sắc hơn không đồng nghĩa nhận dạng đúng hơn. Mẫu trực quan sau triển
khai và những giới hạn thực tế được ghi ở
[báo cáo kiểm chứng](image_dataset_validation.md).

## Tài liệu đối chiếu

- [API PDFium: trích ảnh, mask và các giới hạn](https://pypdfium2.readthedocs.io/en/stable/python_api.html).
- [PaddleOCR: phát hiện layout](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/module_usage/layout_detection.en.md).
- [PaddleOCR: phân tích layout và loại vùng chữ dọc](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/module_usage/layout_analysis.en.md).
