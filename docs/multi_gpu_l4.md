# Chạy notebook 0, 1, 3 trên 4 GPU L4

Cấu hình này giả định **cả bốn L4 xuất hiện trong cùng một phiên Linux/Kaggle**.
Nếu bốn card nằm trong bốn phiên độc lập, không thể dùng device map này để chia
một model qua các phiên. Notebook 0 ghi số GPU, VRAM tổng/còn trống từng card,
compute capability, tensor test từng card và ma trận peer access. L4 có 24 GB
theo thông số NVIDIA; dung lượng runtime báo theo GiB có thể nhỏ hơn. Bốn vùng
nhớ tách biệt; không khai báo một GPU có 96 GB.

| Notebook | Triển khai mặc định |
|---|---|
| 0 | Inventory mọi GPU; tùy chọn build/probe FlashAttention trên mọi card hoặc smoke UniHIR chia model |
| 1 | CPU chạy song song GPU; model nhỏ dùng hàng đợi trang chung; UniHIR dùng 0/1/3 và model nhỏ dùng GPU 2; TADiSR giữ bốn card |
| 3 | Ban đầu PP-OCR → 0, TongGu → 1, Paddle-VL → 2, Hunyuan → 3; GPU xong sớm nhận trang chưa chạy của model chậm |

## Trình tự sử dụng

1. Mở phiên mới có bốn L4, gắn models/packages/PDF, tắt Internet; chạy notebook 0.
   `RESTORATION_GPU_CHECK='off'` chỉ inventory. Chọn `build_and_probe` để kiểm
   native build và kernel CUDA; chọn `unihir_smoke` để thử một PDF/một trang thật.
   Default `gpu_ids=[0,1,2,3]`, `placement='sharded'`, `max_memory='18GiB'`.
2. Chạy notebook 1 trong phiên riêng. Điền ba đường dẫn, giữ `BOOKS=['1']`,
   `MAX_BOOKS=1`, `PAGES='1'`, `LIMIT=1`, `BATCH_SIZE=1` cho smoke đầu.
   `GPU_IDS=[0,1,2,3]`, `EXECUTION_MODE='parallel_models'`. Không chạy notebook
   3 cùng lúc trên bốn card đang được notebook 1 sử dụng.
3. Chuyển **toàn bộ thư mục `_dataset`** đã xác minh sang Dataset Input bất biến
   của notebook 3. ZIP chỉ PNG dùng để xem ảnh, không thay thế dataset manifest.
4. Trong notebook 3, điền `PREPARED_IMAGES`, chọn cùng variant/crop cho bốn
   engine, giữ phạm vi một trang. `GPU_MAP` sửa được ở ô cấu hình.
   `RUN_OCR=False` vẫn chỉ đọc checkpoint, không cần bốn GPU hoặc cài runtime model.
5. Chỉ mở rộng `BOOKS`, `MAX_BOOKS`, `PAGES`, `LIMIT` sau khi smoke runtime đích
   thành công. UniHIR mặc định 10 steps/1 refinement để thử; tự tăng thông số
   trong `MODEL_OPTIONS` sau khi đo VRAM/thời gian.

## Điều phối theo trang

`ADAPTIVE_GPU_SCHEDULING=True` là mặc định của notebook 1 và 3.
Một worker giữ model và nhận từng trang đến khi hàng đợi model đó hết việc.
GPU vừa hoàn tất sẽ ưu tiên phương pháp chưa bắt đầu, rồi giúp phương pháp
có thời gian còn lại ước tính lớn nhất. Chi phí tải model và thời gian mỗi trang
được dùng để tránh tải thêm bản model khi lợi ích dự kiến quá nhỏ.
Trang đang chạy không bị ngắt hay chuyển GPU. Phạm vi sách/trang/crop giữ theo
ô cấu hình; cùng tập trang được xử lý cho đủ các phương pháp đã chọn.

Notebook 1 ghi shard theo phương pháp/GPU rồi ghép theo thứ tự trang canonical;
các crop trùng nội dung dùng chung khóa receipt để tránh inference trùng.
CPU chạy từng phương pháp trong một luồng riêng, đồng thời với phần GPU.
UniHIR dùng `UNIHIR_GPU_IDS=[0,1,3]`: BAGEL trên hai GPU đầu, VAE trên GPU cuối
trong mask này. Bố trí này dựa trên report thành công trước đó: BAGEL dùng 0/1,
VAE dùng 3, GPU 2 chưa dùng. Scheduler cho GPU 2 chạy model nhỏ cùng lúc, rồi
giải phóng GPU 0/1/3 về hàng đợi khi UniHIR xong. TADiSR giữ reservation bốn GPU.
Một bản model sharded cần nhiều GPU, nên không nhân thêm bản trên một L4 khi
thấy một card tạm thời ít hoạt động. `UNIHIR_GPU_IDS=GPU_IDS.copy()` khôi phục
mask bốn card nếu cấu hình/checkpoint khác cần thêm bộ nhớ.

Notebook 3 dùng hàng đợi SQLite cục bộ trong `WORK_DIR/page_queues/`.
Nhận trang là giao dịch độc quyền; mỗi worker ghi checkpoint shard và log riêng.
Coordinator cập nhật manifest/export sau khi mọi worker thoát. Trang lỗi được
giữ trạng thái lỗi trong lượt hiện tại; lượt chạy mới tiếp tục thử trang còn thiếu.
`gpu_assignments.json` ghi các lần GPU đổi model; log có dạng
`<engine>_gpu<id>_notebook.log`. Resume đọc mọi shard theo format cũ.

Notebook 1 mặc định `RUN_SELF_TESTS=False`, `VALIDATE_OUTPUT=False`,
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
- Notebook 1: `preflight.json`, worker `ready.json`/`response.json`, device map,
  peak allocated/reserved từng GPU. Các model ghi JSONL riêng; coordinator ghép
  theo thứ tự cố định, kiểm hash/QC rồi mới công bố snapshot. Ghi object và kiểm
  ngân sách ổ đĩa được khóa để tránh xung đột giữa các worker.
- Notebook 3: `gpu_inventory.json`, `<engine>_gpu<id>_notebook.log`, `<engine>_status.json`,
  `stage_errors.json` và `status.json`. Một engine lỗi không xóa kết quả engine
  khác. Timeout exit 124/restart và resume theo trang tiếp tục được sử dụng.
  Ngắt notebook sẽ báo hủy cho các supervisor để dừng child process.

Đổi sang `EXECUTION_MODE='sequential'` khi cần chẩn đoán. Notebook 1 vẫn giữ
placement sharded của model lớn; đổi execution mode không biến L4 thành GPU 96 GB.
Notebook 3 có thể đặt `ADVANCED['gpu']=0` cho lượt tuần tự. Khi sửa ô cấu hình,
chạy lại ô đó và các ô setup liên quan.

Các kiểm thử local dùng fixture nhỏ và subprocess CPU, không nạp weights và
không chứng minh inference trên bốn L4. Cần chạy notebook 0 và smoke một trang
trên runtime đích để xác minh load/inference. Chưa có số đo tốc độ, peak VRAM
thật hoặc accuracy Hán cổ cho lịch điều phối mới này.

Nguồn: [NVIDIA L4](https://www.nvidia.com/en-us/data-center/l4/),
[Accelerate Big Model Inference](https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference).
