# OCR chữ Hán cổ và dịch tiếng Việt

PDF scan → ảnh nguồn có provenance → bốn OCR → rà soát bản Hán → dịch tiếng Việt.
Đồng thuận, độ ổn định và độ phủ là tín hiệu rà soát; corpus chưa có ground truth.

Luồng hiện tại: notebook 0 kiểm tra môi trường →
[1a — OCR gốc](1a_ocr_anh_goc.ipynb) →
[1 — đếm/nghiên cứu và tiếp tục](1_ocr_va_so_sanh.ipynb) → notebook 2 dịch TXT đã chọn.
Hai notebook OCR/restoration chạy Kaggle offline và có thể ở hai phiên khác nhau.
Hai file được giữ dưới **1.000.000 byte**. Các ô runtime nội bộ nén lossless, kiểm SHA-256
trước khi nạp; ô cấu hình và worker OCR vẫn là mã Python thường, không cần tải source ngoài.
Notebook chuẩn bị ảnh cũ, notebook 3 và các notebook 1b đã được thay thế.

Ở notebook 1a, điền `PACKAGES_DIR`, `MODELS_DIR`, `DATA_DIR` chứa PDF rồi Run All.
Chỉ chạy bốn OCR trên original, lưu ngay **original_ocr_handoff.zip** gồm PNG nguyên byte,
crop/toạ độ, cấu hình và raw checkpoint của bốn model. Đây là gói chuyển tiếp duy nhất;
không tính consensus/coverage, không tạo HTML/TXT/ZIP báo cáo theo sách ở notebook 1a.
Ảnh/raw chưa đóng gói được giữ trên ổ làm cache để resume; chỉ cần chuyển ZIP sang notebook 1.
Đặt gói handoff (ZIP hoặc thư mục đã giải nén) cùng PDF vào `DATA_DIR` của notebook 1.
Notebook 1 tự nhập bản sao writable, không OCR lại original. Mặc định
`RESTORATION_THRESHOLDS=None`: đếm vùng theo ngưỡng, xuất **original_ocr_analysis.zip**
gồm JSON/CSV/HTML và báo cáo đầy đủ của bốn OCR gốc, gồm crop bất đồng khi định vị được.
Không setup/nạp restoration hay OCR trong lượt này. Sau khi xem số liệu mới chọn ngưỡng.
Khi phục dựng, báo cáo bổ sung ảnh từng phương pháp và lượt OCR của cả bốn model trên ảnh đó.
Đối chiếu gốc dùng cache kiểm theo raw/config/hash PDF; crop báo cáo dùng PNG đã chuẩn bị,
không render PDF lại. Chưa có số đo runtime GPU mới để khẳng định tốc độ thực tế.

`BOOKS`, `MAX_BOOKS`,
`PAGES`, `LIMIT`, `CROP_TYPES` dùng chung. Theo yêu cầu mới nhất, notebook đã đặt
sẵn **full**: mọi sách/trang, cả page/group/column, bốn OCR và mọi phương pháp còn giữ.
Ô chính giữ 14 biến có chú thích lựa chọn/tác động; profile, GPU và tham số ít
dùng được chuyển vào ô nâng cao thu gọn. `UNIHIR_MAX_REGIONS` chỉnh ngay ở ô chính.

`VARIANT_MODE='full'` chọn original, TADiSR, DiffTSR và UniHIR; bật phân tích,
kiểm tra, preview, ảnh ghép, xuất ảnh riêng và ZIP sau khi có ngưỡng đã chọn.
Restoration chỉ nhận vùng có **cả 6 cặp model** bất đồng trên ngưỡng đã cấu hình.
`RESTORATION_THRESHOLDS` ban đầu là `None`, không mặc định 35%/20% khi chưa có phân bố thực tế.
Tỷ lệ = Levenshtein / độ dài chuỗi dài hơn sau khi bỏ khoảng trắng. Thiếu model,
OCR bị cắt hoặc dưới 8 ký tự/model không được coi là bất đồng cao.
DiffTSR chỉ thực thi trên dòng ngang đã duyệt đúng geometry; vùng khác ghi
`not_applicable`. Không tự tạo nhãn duyệt để ép chạy trên cột dọc.

UniHIR ưu tiên page, tối đa 12 vùng/lượt, 2 vùng/PDF, 12 MP nguồn; giới hạn có hiệu lực
cả trong full. TADiSR/DiffTSR ưu tiên column rồi group/page, không phục dựng lặp cả ba
crop; trần lần lượt 256/512 vùng, 32/16 MP nguồn trong ô nâng cao. PDF trùng dùng chung
quyết định. Không có vùng phù hợp thì không kiểm asset/cài runtime/nạp model đó.
`selective_restoration_plan.json` ghi lý do và 6 tỷ lệ; bất đồng không chứng minh accuracy.
`selective` chỉ thử UniHIR với cùng gate; `baseline` tắt phục dựng. Các phương pháp
đã bỏ không được thêm lại. OCR gốc và vùng giữ nguyên được dùng lại checkpoint.

UniHIR giữ FlashAttention/SDPA upstream. Hunyuan dùng `attention_backend='sdpa'`,
ghi cấu hình backend thực tế vào log/raw. Không tải hoặc build FlashAttention
riêng cho Hunyuan. SDPA tự chọn kernel phù hợp; cấu hình SDPA chưa chứng minh
kernel Flash đã chạy trên GPU. Xem [vận hành](docs/usage.md).

Ảnh/cache ở `/kaggle/working/image_dataset`; HTML/JSONL/ZIP ở
`/kaggle/working/four_model_consensus/output`. Gói handoff tự mang `_dataset` và raw OCR,
không cần ghép dữ liệu thủ công. `RUN_OCR=False` không chạy inference;
giữ cấu hình crop/variant của checkpoint. Notebook 2 và Dataset models/packages
đang dùng được giữ nguyên.

ZIP mặc định giữ báo cáo như notebook 3 cũ và thêm PNG theo phương pháp,
TXT/JSONL tương ứng từng ảnh/model. Giải nén, mở `index.html` → **Ảnh đã tạo và
OCR trên từng ảnh**. `EXPORT_IMAGES_AND_OCR=True` bật phần này; xuất lại từ
checkpoint cần giữ `_dataset` để lấy đúng PNG, không chạy lại OCR.

Xem [tài liệu](docs/README.md), [pipeline](docs/pipeline.md) và
[GPU](docs/multi_gpu_l4.md). Kiểm thử local chỉ dùng fixture CPU và kiểm tra
cấu trúc; chưa đo tốc độ hay xác minh inference GPU mới trên Kaggle.
Notebook 2 hiện vẫn cần mạng trên Kaggle.
