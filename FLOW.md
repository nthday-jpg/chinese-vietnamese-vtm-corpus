# Flow thực hiện

> Notebook 3 chứa trực tiếp các hàm setup, điều phối và so sánh; chỉ worker OCR chạy ở file riêng.
> `src/vtm_corpus/` vẫn dùng cho CLI/local và làm nguồn cập nhật notebook. Notebook 2 dùng package như trước.
> Output của mỗi lần chạy nằm trong `work/runs/ocr_*/` hoặc `work/runs/dich_*/`
> (trên Kaggle là `/kaggle/working/runs/...`), không ghi đè lẫn nhau.

```
PDF scan ─► ảnh từng trang ─► OCR ─► làm sạch ─► txt chữ Hán ─► chia đoạn ─► LLM dịch ─► ghép ─► txt tiếng Việt
            └──── Bước 1: 3_so_sanh_so_chu_ocr.ipynb ──────────────┘   └────── Bước 2: 2_dich_han_viet.ipynb ──────┘
```

## Bước 1 — OCR (PDF scan → txt chữ Hán)

1. **Tìm PDF:** lấy file `.pdf` trong Input; mỗi PDF là 1 cuốn, tên file là tên sách.
   `BOOKS=[]` chọn tất cả; `BOOKS=["tên", "*glob*"]` chỉ chọn tập con (xem `vtm list`).
2. **Chọn tập trang chung:** `BOOKS`, `MAX_BOOKS`, `PAGES`, `LIMIT` áp dụng cho cả bốn model; mặc định hai trang của một PDF.
3. **Render:** mỗi trang PDF được vẽ thành ảnh ở `DPI`, thu nhỏ để cạnh dài không quá `MAX_SIDE`.
4. **OCR:** PP-OCRv6-medium, TongGuOCR, PaddleOCR-VL-1.6 và HunyuanOCR-1.5 chạy lần lượt trong tiến trình riêng; model được giải phóng sau mỗi stage.
5. **Làm sạch:** cắt đoạn lặp vô hạn ở cuối (lỗi hay gặp của model sinh chữ), bỏ ký hiệu markdown/HTML, bảng thành
   dòng chữ.
6. **Ghi file:** mỗi model lưu checkpoint và `.txt` theo sách; đường dẫn nằm trong manifest campaign.
7. **So sánh và rà soát:** căn chỉnh ký tự, đếm phiếu/entropy, hiển thị bốn dự đoán và crop chữ bất đồng khi có tọa độ đáng tin cậy. Chọn bản `.txt` đã rà soát làm đầu vào notebook 2; không tự chuyển chữ còn tranh chấp sang dịch.

Mỗi trang xong được lưu ngay. Timeout và số lần khởi động lại theo cấu hình; chạy lại cùng checkpoint sẽ bỏ qua trang đã xong. Notebook 3 chuẩn bị packages offline trước OCR; notebook 2 giữ workflow dịch hiện tại.

## Bước 2 — Dịch (txt chữ Hán → txt tiếng Việt)

1. **Đọc txt:** lấy file `.txt` có chữ Hán trong Input; `books` lọc tập con như bước 1.
2. **Chia đoạn:** gom các dòng liên tiếp thành đoạn không quá `CHUNK_CHARS` chữ; dòng quá dài thì cắt ở dấu 。！？；.
   LLM chỉ đọc được giới hạn độ dài, và đoạn ngắn thì ít bị dịch sót hơn.
3. **Ghép prompt:** mỗi đoạn được đặt vào `USER` (chỗ `{text}`), kèm `SYSTEM`, theo đúng định dạng hội thoại của model.
4. **Sinh bản dịch:** nhiều đoạn được dịch cùng lúc (`BATCH`); model ≤ 4B thì mỗi GPU 1 bản, model lớn hơn thì trải
   trên cả 2 GPU (`GPU_PER_MODEL`).
5. **Ghép file:** bản dịch các đoạn được nối theo thứ tự gốc thành `runs/dich_*/dich/<tên sách>.txt`. Đoạn chưa dịch được ghi
   `[chưa dịch]`; đoạn chạm `MAX_NEW_TOKENS` được báo vì có thể bị cụt.

## Gợi ý

Notebook chỉ là **một cách làm mẫu**. Bạn được đề xuất cách khác, thay một bước hoặc cả quy trình, miễn là:
đầu vào vẫn là PDF scan, đầu ra vẫn là `.txt` bản dịch cho từng PDF, và bạn **giải thích vì sao** cách của mình tốt
hơn (so trên cùng vài trang/đoạn, bằng mắt hoặc bằng số đo). Vài hướng để tham khảo:

**OCR**
- Model OCR/VLM khác: Qwen2.5-VL, GOT-OCR 2.0, PP-OCRv5 (OCR truyền thống, nhanh), Tesseract (`chi_tra_vert` cho chữ dọc)…
- Chạy 2 model rồi kết hợp: trang nào model A lặp/thiếu chữ thì lấy model B.
- Xử lý ảnh trước khi OCR: tăng tương phản, xoá watermark, nắn trang cong, cắt lề.
- Sửa lỗi sau OCR bằng LLM; đổi giản thể/phồn thể cho thống nhất (OpenCC); thêm dấu câu cho văn ngôn không có dấu.

**Dịch**
- Dịch 2 chặng: văn ngôn → tiếng Trung hiện đại → tiếng Việt.
- Đưa bảng thuật ngữ (tên huyệt, vị thuốc, bài thuốc) vào prompt, hoặc tự tìm thuật ngữ liên quan cho từng đoạn.
- Cho LLM tự soát lại bản dịch của chính nó (dịch → kiểm tra thiếu/sai → sửa).
- Dùng vLLM hoặc model lượng tử hoá để dịch nhanh hơn, từ đó dùng được model lớn hơn.
- Bỏ qua bước OCR riêng: đưa thẳng ảnh trang cho VLM dịch.

**Đánh giá**
- Dịch tay vài đoạn làm bản chuẩn, rồi so các cách bằng BLEU / chrF / COMET hoặc nhờ người đọc chấm.
