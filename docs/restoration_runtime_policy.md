# Chọn restoration sau OCR gốc — 17/11/2026

Notebook **1a** hiện chỉ chạy original bằng cả bốn model rồi xuất `original_ocr_handoff.zip`.
Đặt gói cùng PDF trong DATA_DIR của notebook **1**. Notebook 1 mặc định
`RESTORATION_THRESHOLDS=None`: chỉ đếm theo ngưỡng, chưa chạy restoration; xuất
`original_ocr_analysis.zip` để xem phân bố rồi mới cấu hình. Không dùng mặc định 35%/20%.
Khi đã chọn ngưỡng, **cả 6 cặp** phải đạt; các phần phân tích và ZIP của mode full vẫn bật.

| Phương pháp | Chi phí đã xác minh từ mã | Ngưỡng mỗi cặp | Trần mỗi lượt |
|---|---|---:|---|
| UniHIR | BAGEL + định vị/phục dựng, steps=10/ref_count=1; giữ mức đã dùng, không tăng refinement | Chọn sau thống kê | 12 vùng, 2 vùng/PDF, 12 MP nguồn |
| TADiSR | CogView4-6B, một timestep, mỗi tile vẫn cần transformer/VAE/decoder; ảnh x4 có số pixel x16 | Chọn sau thống kê | 256 vùng, 32 MP nguồn |
| DiffTSR | DDIM 20 bước thay vì mặc định upstream 200; chỉ dòng ngang đã duyệt 1–24 chữ, gần 4:1 | Chọn sau thống kê | 512 vùng, 16 MP nguồn |

Nguồn tác giả: [TADiSR](https://github.com/mingcv/TADiSR),
[DiffTSR Q&A về tốc độ và T=20](https://github.com/YuzheZhang-1999/DiffTSR#-qa),
[UniHIR](https://github.com/ZZXF11/UniHIR).
Không thay model/weights, số bước hay package trong thay đổi bộ chọn này.

Chưa đo inference GPU Kaggle cho TADiSR/DiffTSR ở cấu hình này, nên không gán số
giây/ảnh từ GPU, kích thước hoặc checkpoint khác. “Một bước” của TADiSR không có
nghĩa một lần forward cho cả trang: tile=256/overlap=64 chạy trên canvas x4.

Audit metadata snapshot local (không render hoặc inference): 960 trang nguồn
khác nhau; 1.133 bản ghi trang gồm các alias PDF, tổng 7.787 crop page/group/column.
Theo kích thước đã lưu và công thức chia tile hiện tại, trung vị một page cần
368 tile TADiSR, một column cần 18 tile. Đây là số lượt tile dự kiến, **không phải
thời gian benchmark**. Vì vậy TADiSR/DiffTSR chọn column nếu có, rồi group/page;
UniHIR ưu tiên page để giữ ngữ cảnh của phương pháp. Không chạy lặp cả ba mức.

Điểm mỗi cặp là Levenshtein / max(độ dài hai chuỗi), chỉ bỏ khoảng trắng, giữ nguyên
dạng chữ và dấu câu. Mỗi model cần ít nhất 8 ký tự, không thiếu hoặc bị cắt token.
PDF/PNG hash, canonical region và source box phải khớp; không lấy văn bản cả trang
gán cho crop. Một model khác ba model đồng ý không đủ để kích hoạt. Bất đồng do
thứ tự đọc hay hạn chế model vẫn có thể qua ngưỡng, nên đây không phải nhãn ảnh hỏng.

`RESTORATION_THRESHOLDS` và `UNIHIR_MAX_REGIONS` ở ô chính. Tăng ngưỡng để chạy ít
hơn; tăng trần chỉ cho phép thêm ứng viên đã qua ngưỡng. Quota và pixel trong
`SELECTIVE_RESTORATION` ở ô nâng cao; `None` bỏ riêng chặn đó. Các trần không phải
timeout tổng lượt. Alias cùng PDF SHA dùng chung quyết định/quota; nhãn loại trừ
và hợp đồng đầu vào DiffTSR vẫn có hiệu lực. Không có vùng hợp lệ thì bỏ model đó
trước setup/probe/nạp weights. Không tự tạo nhãn duyệt để chạy DiffTSR.

Original và vùng giữ nguyên dùng lại OCR khi khớp ảnh/config/processor/kích thước.
Các giai đoạn dùng GPU nối tiếp, kết thúc worker trước khi chuyển giai đoạn.
Không dùng lại thời gian OCR cũ làm thời gian inference mới; raw đánh dấu
`ocr_reused_from_original` và giữ `original_ocr_seconds` riêng.

Hai tệp để điều chỉnh sau lượt Kaggle:

- `WORK_DIR/original_ocr_disagreement.json`: điểm đủ 6 cặp hoặc lý do không đủ dữ liệu.
- `_dataset/selective_restoration_plan.json`: ứng viên, ngưỡng, quota, vùng chọn/bỏ theo phương pháp.

ZIP vẫn chứa HTML, raw OCR, ảnh từng phương pháp, TXT/JSONL khớp từng ảnh. Vùng giữ
nguyên được ghi `original_retained`, liên kết PNG original và có provenance nguồn;
lỗi inference không bị thay bằng original. Không có vùng được phục dựng trên trang
thì không chạy nhánh OCR lặp cho trang đó.
