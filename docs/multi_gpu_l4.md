# Chạy notebook 0 và notebook 1 gộp trên 4 GPU L4

Cấu hình này giả định **cả bốn L4 xuất hiện trong cùng một phiên Linux/Kaggle**.
Nếu bốn card nằm trong bốn phiên độc lập, không thể dùng device map này để chia
một model qua các phiên. Notebook 0 ghi số GPU, VRAM tổng/còn trống từng card,
compute capability, tensor test từng card và ma trận peer access. L4 có 24 GB
theo thông số NVIDIA; dung lượng runtime báo theo GiB có thể nhỏ hơn. Bốn vùng
nhớ tách biệt; không khai báo một GPU có 96 GB.

| Notebook/giai đoạn | Triển khai |
|---|---|
| 0 | Inventory GPU; tuỳ chọn build/probe và smoke UniHIR |
| 1: ảnh | Original; UniHIR chọn lọc trong ngân sách; TADiSR/DiffTSR chỉ khi bật riêng |
| 1: OCR | PP-OCR → 0, TongGu → 1, Paddle-VL → 2, Hunyuan → 3; card xong sớm nhận việc tiếp |

## Trình tự sử dụng

1. Gắn models/packages/PDF, tắt Internet, chạy notebook 0 nếu cần kiểm môi trường.
2. Mở `1_ocr_va_so_sanh.ipynb`, điền ba đường dẫn; giữ một trang, crop group cho smoke.
3. Run All: bước ảnh hoàn tất và nhả GPU rồi OCR chạy trong cùng phiên.
   Dataset nối tự động; không chuyển `_dataset` sang notebook khác.
4. Mở rộng phạm vi sau khi smoke Kaggle đạt. `RUN_OCR=False` chỉ đọc checkpoint.

## Điều phối theo trang

`ADAPTIVE_GPU_SCHEDULING=True` là mặc định của notebook 1 gộp.
Một worker giữ model và nhận từng trang đến khi hàng đợi model đó hết việc.
GPU vừa hoàn tất sẽ ưu tiên phương pháp chưa bắt đầu, rồi giúp phương pháp
có thời gian còn lại ước tính lớn nhất. Chi phí tải model và thời gian mỗi trang
được dùng để tránh tải thêm bản model khi lợi ích dự kiến quá nhỏ.
Trang đang chạy không bị ngắt hay chuyển GPU. Phạm vi sách/trang/crop giữ theo
ô cấu hình; cùng tập trang được xử lý cho đủ các phương pháp đã chọn.

Giai đoạn ảnh ghi shard theo phương pháp/GPU rồi ghép theo thứ tự trang canonical;
các crop trùng nội dung dùng chung khóa receipt để tránh inference trùng.
UniHIR mặc định có `UNIHIR_GPU_IDS=[0,1,2,3]`, VAE ở card dành riêng theo
`vae_gpu_id='auto'`; không thêm OCR vào các card đang phục dựng. TADiSR cũng
giữ reservation nhiều GPU khi bật riêng. Phục dựng kết thúc trước bước OCR.

Giai đoạn OCR dùng hàng đợi SQLite cục bộ trong `WORK_DIR/page_queues/`.
Nhận trang là giao dịch độc quyền; mỗi worker ghi checkpoint shard và log riêng.
Coordinator cập nhật manifest/export sau khi mọi worker thoát. Trang lỗi được
giữ trạng thái lỗi trong lượt hiện tại; lượt chạy mới tiếp tục thử trang còn thiếu.
`gpu_assignments.json` ghi các lần GPU đổi model; log có dạng
`<engine>_gpu<id>_notebook.log`. Resume đọc mọi shard theo format cũ.

Giai đoạn ảnh mặc định `RUN_SELF_TESTS=False`, `VALIDATE_OUTPUT=False`,
`verify_vae=False` để bỏ kiểm tra lặp lại. Kiểm tra asset/runtime, QC và manifest
trước publish vẫn chạy. `PNG_COMPRESSION_LEVEL=1` lưu PNG lossless nhanh hơn;
đặt 6 để ưu tiên dung lượng. `original` dùng trực tiếp object canonical.
Đặt `ADAPTIVE_GPU_SCHEDULING=False` để quay lại phân công model cố định.

## Model lớn và giới hạn bộ nhớ

UniHIR giữ BF16 và chia các decoder layer qua Accelerate. Adapter dùng ngân sách
weights tối đa 18 GiB/card, đồng thời trừ phần VRAM còn trống cần dành cho
activation/KV và VAE. Không chia nhỏ một decoder layer, không tự spill sang CPU
hoặc disk. Nếu device map không phù hợp, dừng và giữ log; tổng VRAM đủ không
chứng minh inference vừa bộ nhớ. `max_memory` không giới hạn toàn bộ activation.

TADiSR giữ float32, nạp checkpoint trên CPU rồi chia transformer qua GPU;
VAE/segmentation decoder ở GPU đầu. Tile mặc định 256, overlap 64. Không nạp
toàn bộ transformer lên L4 đầu trước khi chia. Nếu thiếu prompt embedding,
text encoder tạo prompt trên CPU một lần và lưu cache; cần đủ **RAM hệ thống**
cho model float32, độc lập với VRAM. Nếu RAM không đủ, chuẩn bị embedding từ
trước và gắn asset theo manifest. Không tự đổi dtype, lượng tử hóa hoặc giảm ảnh
canonical để né lỗi.

Các worker nhận `CUDA_VISIBLE_DEVICES` riêng trước khi import Torch/Paddle.
GPU ID trong cấu hình là chỉ số logic dưới mask của nền tảng; mỗi OCR worker
nhìn thấy card của mình là `cuda:0`. Không dùng DDP/torchrun để nhân bản model.
MMCV build dò compute capability trong Python đích (L4 là 8.9); không còn cố
định ở kiến trúc RTX PRO 6000. Giữ các môi trường Python/native ABI riêng.

## Đọc kết quả kiểm tra

- Notebook 0: `kaggle_environment_logs.zip`; khi bật kiểm GPU, thêm
  `restoration_check_*_logs.zip`. Build, CUDA probe, load và inference là các trạng thái riêng.
- Giai đoạn ảnh: `preflight.json`, worker `ready.json`/`response.json`, device map,
  peak allocated/reserved từng GPU. Các model ghi JSONL riêng; coordinator ghép
  theo thứ tự cố định, kiểm hash/QC rồi mới công bố snapshot. Ghi object và kiểm
  ngân sách ổ đĩa được khóa để tránh xung đột giữa các worker.
- Giai đoạn OCR: `gpu_inventory.json`, `<engine>_gpu<id>_notebook.log`, `<engine>_status.json`,
  `stage_errors.json` và `status.json`. Một engine lỗi không xóa kết quả engine
  khác. Timeout exit 124/restart và resume theo trang tiếp tục được sử dụng.
  Ngắt notebook sẽ báo hủy cho các supervisor để dừng child process.

Đổi sang `EXECUTION_MODE='sequential'` khi cần chẩn đoán. Giai đoạn ảnh vẫn giữ
placement sharded của model lớn; đổi execution mode không biến L4 thành GPU 96 GB.
Giai đoạn OCR có thể đặt `ADVANCED['gpu']=0` cho lượt tuần tự. Khi sửa ô cấu hình,
chạy lại ô đó và các ô setup liên quan.

Các kiểm thử local dùng fixture nhỏ và subprocess CPU, không nạp weights và
không chứng minh inference trên bốn L4. Cần chạy notebook 0 và smoke một trang
trên runtime đích để xác minh load/inference. Chưa có số đo tốc độ, peak VRAM
thật hoặc accuracy Hán cổ cho lịch điều phối mới này.

Nguồn: [NVIDIA L4](https://www.nvidia.com/en-us/data-center/l4/),
[Accelerate Big Model Inference](https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference).
