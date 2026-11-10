# Rà soát trước triển khai notebook chuẩn bị ảnh

Ngày 10/11/2026, nhánh `mduy`; cây Git sạch trước khi làm. Không có notebook
`1_chuan_bi_anh_ocr.ipynb`. Đã kiểm tra README, FLOW, PROBLEM, ba tài liệu dataset,
tài liệu packages, notebook 0/3, các module dataset/PDF/layout/config/worker.

## Nguồn có thẩm quyền và ranh giới giữ nguyên

- `pdf_images.py`: PDFium; chỉ trích native khi placement, clip, rotation,
  annotation và so sánh toàn pixel với trang hiển thị đều đạt. Giữ nguyên thuật
  toán này; không đổi sang PyMuPDF, không cap nguồn theo processor OCR.
- `image_preparation.py`: background subtraction và projection chỉ dùng trên
  ảnh phân tích. `image_dataset.py` tạo phân hoạch page/group/column, thứ tự RTL,
  manual cuts, object PNG theo hash, receipt riêng cho nguồn/layout/variant và
  snapshot atomic. Nhúng trực tiếp các hàm cần dùng từ nguồn này vào notebook.
- Notebook 3 nhúng hàm, bootstrap offline và worker; chạy từng model trong
  subprocess. Dataset reader xác minh hash PDF/snapshot/pixel trước khi dùng.
  Notebook 2/3 và các runtime OCR hiện có không cần sửa.
- `BOOKS` là tên/stem/glob, `MAX_BOOKS` áp dụng sau chọn sách; `PAGES` đếm từ 1
  cho từng PDF; `LIMIT` là giới hạn trang toàn bộ tập đã chọn.

## Thiếu sót phải xử lý

- Dataset v1 chỉ cho original/CLAHE/bicubic x2. Validator cũ mặc định mọi tên
  khác có scale 1; đưa ảnh x4 vào `regions.images` sẽ làm notebook 3 từ chối cả
  trang. Dùng extension manifest có version riêng cho phục hồi; core v1 chỉ
  chứa original. Notebook 3 tiếp tục đọc original bằng reader hiện hành.
- `.gitignore` thực tế bỏ qua toàn bộ tools/tests/scripts/configs/kaggle_upload;
  câu nói ngược lại trong tài liệu kiểm chứng cũ không đúng ở HEAD này. Runtime
  mới không phụ thuộc những thư mục đó; kiểm thử và worker phải nằm trong notebook.
- Phải tách cache canonical khỏi model/config restoration; ghi trạng thái mọi
  lựa chọn, preflight trước xử lý đắt tiền, chỉ công bố dataset cuối khi đầy đủ
  hoặc khi người dùng bật partial rõ ràng.
- Tiling TADiSR upstream đặt lại *cả hai* trục về `[0]` khi một trục ngắn hơn
  tile: crop hẹp nhưng cao có thể bị bỏ phần dưới. Wrapper nhúng sẽ tính trục
  độc lập sau padding, kiểm tra mọi pixel có trọng số và test crop hẹp/dài.
- Không cài chung các stack xung đột, không hạ Torch/NumPy của kernel. Chỉ dùng
  wheel Linux đúng tag; probe import/NumPy bridge/CUDA/kiến trúc trong tiến trình
  thật trước khi gọi model. Trọng số tồn tại không có nghĩa runtime đã chạy.

## Hợp đồng model và khả năng thực tế

| Model | Hợp đồng đã đọc từ upstream | Quyết định |
|---|---|---|
| Real-ESRGAN | RRDBNet 64 feature, 23 block, growth 32; RealESRGAN_x4plus, BSD-3-Clause | Nhúng kiến trúc inference tối thiểu từ BasicSR, strict state dict, tiled x4; không kéo GFPGAN/training |
| TADiSR | CogView4-RealCE; adapter 785453495 byte, SHA256 `7cec8d0917305037cbafcb1b1bf81062add1613b561f5d7e2d1e599fe0baf5e0`; CogView4 khoảng 31 GB, prompt embeddings, scheduler, tokenizer; MIT code, adapter research, base Apache-2.0 | Ưu tiên; nhúng code inference đúng revision và license; lưu ảnh/mask; float32 theo upstream, chưa hứa vừa GPU local 8 GB |
| DiffTSR | Một dòng ngang, <=24 ký tự, W/H 1..8; processor 512x128; DDIM 200 (có thể thử 20); Apache-2.0 | page/group/column chưa xác minh dòng ngang => not_applicable. Chỉ chạy vùng được người dùng xác nhận hợp đồng; không xoay glyph/reflow tự động |
| UniHIR | BAGEL/MLLM, font, ae.safetensors + model.safetensors + config/tokenizer; Python 3.10, Torch 2.5.1/cu124, flash-attn 2.5.8; khuyến nghị 80 GB; CC BY-NC-ND 4.0 | Ordinary scan => not_applicable. Có adapter cho vùng inscription xác nhận; bản pin CUDA/flash-attn cũ chưa xác minh Blackwell sm120. Không nhúng/phân phối code có sửa dưới ND; source nguyên trạng ở asset riêng khi điều kiện cho phép |

Nguồn chính thức: [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN),
[TADiSR](https://github.com/mingcv/TADiSR),
[DiffTSR](https://github.com/YuzheZhang-1999/DiffTSR),
[UniHIR](https://github.com/ZZXF11/UniHIR).

## Asset còn cần chuẩn bị và giới hạn

Đã thấy bundle OCR có PDFium Linux, OpenCV và Torch cp310/cu130; không giả định
extension cp310 dùng được cho kernel cp313. Có thể dùng lại artifact đúng tag,
không sửa manifest/weights OCR. Tải mới vào nhánh `image_restoration` dưới
`kaggle_upload`, ghi URL/revision/hash/license; không đưa binary vào Git.
TADiSR base/adapter công khai, không gated theo API hiện tại. DiffTSR phát hành
qua Google Drive/Baidu, cần kiểm tra truy cập thực. UniHIR trên ModelScope,
phải giữ hạn chế nghiên cứu phi thương mại/ND trong hướng dẫn phân phối.

Máy hiện tại là Windows/Python 3.13, RTX 4060 Laptop, khoảng 8 GB VRAM; không
phải runtime Kaggle 96 GB. Sẽ đo smoke thật cho phương pháp chạy được và ghi rõ
các blocker còn lại. Không chạy toàn corpus. Không có ground truth để chứng minh
nét sinh thêm là đúng hoặc OCR tốt hơn. Deskew/dewarp mặc định tắt vì chưa có
đo góc/cong trang đủ tin cậy; giữ canonical làm đối chứng, manual cuts để rà soát.

## Điều chỉnh trong quá trình triển khai

Code upstream nguyên trạng được nhúng vào notebook và cũng lưu bản nguồn trong
asset; wrapper ở file riêng do notebook sinh ra. Không cần đọc source ngoài
notebook khi process. TADiSR có thể tạo prompt embeddings offline từ base đã
kiểm checksum trong worker riêng, lưu vào working, không sửa Input.

Người dùng sau đó yêu cầu dừng các lượt thử local và chỉ kiểm tra nhanh khi cần.
Đã áp dụng: không chạy tiếp inference/test suite; các sửa cuối chỉ kiểm tra tĩnh.
Không dùng kết quả local để xác nhận runtime Kaggle. Trạng thái asset/model thực
tế và các phần chưa hoàn tất nằm trong [báo cáo bàn giao](offline_image_notebook_report.md).
