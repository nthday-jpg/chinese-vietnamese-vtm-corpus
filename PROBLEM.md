# Bài toán OCR và dịch sách Hán

Chuyển PDF scan sách chữ Hán, đặc biệt sách Y học cổ truyền, thành bản chép Hán
có thể rà soát rồi dịch sang tiếng Việt. Mục tiêu là trung thành với nguồn,
truy xuất được bằng chứng và giữ rõ chỗ chưa đọc được.

Corpus chưa có ground truth nên không thể trực tiếp xác định accuracy hay CER.
Một câu hợp nghĩa không nhất thiết là bản chép đúng. Phải tách nhận dạng chữ,
xác minh bằng ảnh, suy luận ngữ nghĩa và dịch; bảo tồn phồn thể/dị thể cùng các
phương án chưa phân định. Ảnh phục hồi và bản dịch không thay bằng chứng nguồn.

Xem [phân tích dữ liệu và giới hạn](docs/problem.md), [pipeline hiện tại](docs/pipeline.md)
và [cách kiểm chứng các giả thuyết](docs/hypotheses.md). Hợp đồng checkpoint,
đầu ra TXT/JSONL/HTML cùng điều kiện Kaggle/local nằm trong [vận hành](docs/usage.md).

Notebook 1 chuẩn bị ảnh bất biến; Notebook 3 dùng lại ảnh và raw checkpoint để
đối chứng restoration, cảnh báo thị giác và rà soát thủ công trong phiên riêng.
