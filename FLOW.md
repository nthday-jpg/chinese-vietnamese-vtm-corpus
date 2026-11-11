# Luồng xử lý

Xem [tài liệu đầy đủ](docs/README.md) để biết cấu hình, hợp đồng dữ liệu và giới hạn.

1. **Kiểm tra môi trường** bằng notebook 0 khi runtime Kaggle thay đổi. Gắn các
   Dataset packages/models/PDF thực tế; xác minh ABI và GPU trước inference.
2. **Chuẩn bị ảnh tùy chọn** bằng notebook 1 trong phiên riêng: lấy ảnh nguồn đã
   kiểm chứng, canonical page/group/column, biến thể và provenance. Giữ PDF gốc,
   immutable snapshot, manifest và object; không cần chạy notebook 1 cùng phiên OCR.
3. **OCR** bằng notebook 3: chọn cùng sách/trang/crop; một engine trong một worker
   xử lý các trang/biến thể hợp lệ rồi thoát. Từng trang thành công có checkpoint.
   Luồng render PDF cũ vẫn dùng được khi không cấu hình prepared images.
4. **Thu thập bằng chứng**: Spatial Coverage tạo ứng viên từ ảnh gốc độc lập;
   Page-Level Consensus đối chiếu cấu trúc đầu ra; Character-Level Consensus giữ
   đa số nghiêm ngặt 3/4; Variant Stability so trong từng engine. Biến thể không
   được cộng thành phiếu model độc lập. Thiếu geometry hoặc alignment mơ hồ giữ
   chưa xác định. Không tự sửa phồn thể/dị thể hay dùng LLM viết lại OCR.
5. **Rà soát** HTML offline, raw text, source-coordinate crop và các phương án.
   Người đọc chọn bản Hán để dịch; đồng thuận/overlap/stability không chứng minh
   reading đúng và không thay ground truth.
6. **Dịch** bằng notebook 2: TXT đã chọn → chia đoạn → prompt/model dịch → ghép
   TXT Việt. Đoạn thiếu/token-limit giữ cảnh báo. Chưa xác nhận notebook 2 offline.

Bắt đầu một sách/một trang. Giữ `WORK_DIR` và cấu hình để resume, hoặc dùng
`RUN_OCR=False` phân tích checkpoint. Không cần nạp model khi chỉ xem bằng chứng.
