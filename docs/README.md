# OCR Hán cổ và dịch tiếng Việt

Luồng hiện tại: notebook 0 → [1a OCR gốc](../1a_ocr_anh_goc.ipynb) →
[1 đếm/phục dựng](../1_ocr_va_so_sanh.ipynb) → notebook 2 dịch TXT.
Đặt `original_ocr_handoff.zip` từ 1a cùng PDF trong DATA_DIR của 1; mặc định chỉ đếm,
chưa phục dựng khi `RESTORATION_THRESHOLDS=None`. Corpus chưa có ground truth; đồng thuận và stability
không xác nhận accuracy.

| Đọc tiếp | Nội dung |
|---|---|
| [Bài toán](problem.md) | Ba nhóm ảnh đã quan sát, nguồn và giới hạn bằng chứng |
| [Giả thuyết](hypotheses.md) | Cơ chế, inductive bias và những điều cần kiểm chứng |
| [Pipeline](pipeline.md) | Trạng thái thực, manifest/cache và hai trục đánh giá |
| [Vận hành](usage.md) | Cấu hình notebook, asset offline, scripts và đọc kết quả |
| [4 GPU L4](multi_gpu_l4.md) | Cấu hình mới cho notebook 0 v? notebook 1 g?p; chia model và chạy worker đồng thời |
| [Báo cáo khả thi](../reports/restoration_feasibility.md) | Mẫu thật, mọi phương pháp cũ/mới, ma trận và quyết định |

Notebook chạy thực tế trên Kaggle. Kiểm tra local chỉ là CPU nhỏ/static, không
thay xác nhận Linux/GPU/Internet Off. Giữ original và raw checkpoint để rà soát.
