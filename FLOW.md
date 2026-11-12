# Luồng xử lý

PDF gốc → ảnh canonical tùy chọn → từng OCR chạy tuần tự → checkpoint →
đối chiếu/báo cáo → người đọc chọn bản Hán → dịch tiếng Việt.

Xem [sơ đồ và cơ chế đã triển khai](docs/pipeline.md). Coverage tìm ứng viên từ
ảnh nguồn; consensus so các OCR; stability so biến thể trong từng engine.
Chúng hỗ trợ rà soát, không chứng minh bản chép đúng.

[Hướng dẫn vận hành](docs/usage.md) mô tả thứ tự notebook 0 → 1 (tùy chọn, phiên
riêng) → 3 → 2, chạy mẫu nhỏ, đọc checkpoint và runtime offline. Mặc định notebook
3 chọn một PDF/một trang để smoke. Giữ nguồn,
manifest, checkpoint và các phương án chưa phân định; người đọc chuẩn bị TXT
đã chọn cho dịch. Workflow dịch hiện chưa xác nhận hoàn toàn offline.

Các hướng chưa triển khai được tách trong [giả thuyết nghiên cứu](docs/hypotheses.md).

Kiến trúc hiện tại: Notebook 1 tạo immutable Kaggle Dataset; Notebook 3 dùng lại
ảnh và raw checkpoint trong phiên độc lập. Restoration Ablation và Visual
Fidelity đã có mã và test CPU, xem [pipeline](docs/pipeline.md).
