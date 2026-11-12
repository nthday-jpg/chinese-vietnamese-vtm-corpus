# OCR chữ Hán cổ và dịch tiếng Việt

PDF scan → ảnh nguồn có provenance → OCR → rà soát bản Hán → dịch tiếng Việt.
Dự án chưa có ground truth trên corpus mục tiêu; đồng thuận, độ ổn định và độ
phủ là tín hiệu rà soát, không phải OCR accuracy.

Bắt đầu tại [tổng quan tài liệu](docs/README.md). Xem [bài toán](docs/problem.md),
[pipeline hiện tại](docs/pipeline.md), [giả thuyết nghiên cứu](docs/hypotheses.md)
và [hướng dẫn vận hành](docs/usage.md).

Luồng chính: notebook 0 kiểm tra môi trường → notebook 1 chuẩn bị dataset ảnh
trong phiên riêng → notebook 3 OCR/rà soát → notebook 2 dịch TXT đã chọn.
Nhánh render PDF trực tiếp được giữ để tương thích; H10 cần dataset ảnh.
Đường dẫn notebook, runtime, tài sản offline và giấy phép nằm trong hướng dẫn
vận hành. Local Windows dùng `.venv` tại root, Python 3.13.

Trước khi chạy notebook 3, điền `PACKAGES_DIR`, `MODELS_DIR`, `DATA_DIR` rồi đặt
`BOOKS=["1"]`, `MAX_BOOKS=1`, `PAGES="1"`, `LIMIT=1`. Mặc định notebook là một PDF/một trang. `RUN_OCR=False` đọc checkpoint không nạp model.

Có mã và kiểm thử CPU cho các lớp bằng chứng; chưa xác nhận inference mới bốn
model/multi-variant trên Kaggle trong đợt đối chiếu tài liệu 12/11/2026.
Notebook 2 hiện vẫn cần mạng trên Kaggle. Xem [mức kiểm chứng](docs/usage.md#mức-kiểm-chứng).

Kiến trúc hiện tại: Notebook 1 tạo immutable Kaggle Dataset; Notebook 3 dùng lại
ảnh và raw checkpoint trong phiên độc lập. Restoration Ablation và Visual
Fidelity đã có mã và test CPU, xem [pipeline](docs/pipeline.md).
