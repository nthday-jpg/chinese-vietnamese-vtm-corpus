# Luồng xử lý

Notebook 0 kiểm tra môi trường → notebook 1a OCR gốc → notebook 1 đếm/phục dựng → người đọc chọn bản Hán → notebook 2 dịch.

[Notebook 1a](1a_ocr_anh_goc.ipynb) và [notebook 1](1_ocr_va_so_sanh.ipynb) chạy Kaggle offline, có thể khác phiên:

1. Notebook 1a lấy ảnh nguồn, crop và chạy bốn OCR trên original, giữ raw/checkpoint để resume.
   Chỉ xuất gói handoff, không tính consensus/coverage hoặc tạo ZIP báo cáo riêng.
2. Tải `original_ocr_handoff.zip`, đặt cùng PDF trong `DATA_DIR` của notebook 1.
3. Notebook 1 kiểm gói dữ liệu, dùng lại ảnh/raw, đếm theo nhiều ngưỡng và xuất `original_ocr_analysis.zip`.
4. Mặc định `RESTORATION_THRESHOLDS=None`, chưa chạy restoration/OCR. Sau khi xem số liệu mới điền ngưỡng.
5. Khi đã cấu hình, phục dựng vùng mà cả 6 cặp đều qua ngưỡng, OCR phần phục dựng và xuất HTML/JSONL/ZIP.

Các nhánh chọn lọc giữ original ở vùng không phục dựng và ghi rõ provenance;
trang không có vùng chọn không OCR lặp. TADiSR, DiffTSR và UniHIR giữ tích hợp nhưng
chưa thực thi khi ngưỡng là None. DiffTSR chỉ dành cho dòng ngang đã duyệt.

Bước ảnh chưa hoàn tất ở notebook 1a dừng trước OCR. Chạy lại dùng cache/checkpoint đã có.
`RUN_OCR=False` bỏ qua ảnh/model, chỉ đọc và xuất lại checkpoint tại `WORK_DIR`.
Notebook chuẩn bị ảnh cũ, notebook 3 và các notebook 1b đã được thay thế.

Xem [pipeline](docs/pipeline.md) và [vận hành](docs/usage.md). Giữ nguồn,
provenance, raw checkpoint và mọi phương án chưa phân định. Consensus, coverage,
stability và proxy chọn vùng đều không chứng minh độ chính xác OCR.
