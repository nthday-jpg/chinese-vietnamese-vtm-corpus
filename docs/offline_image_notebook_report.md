# Bàn giao notebook chuẩn bị ảnh offline

Ngày 10/11/2026, nhánh `mduy`. Notebook mới:
[`1_chuan_bi_anh_ocr.ipynb`](../1_chuan_bi_anh_ocr.ipynb).
**Chưa xác nhận chạy được trên Kaggle.** Kiểm tra local đã có chỉ mô tả hành vi
local; sau chỉ đạo của người dùng, không chạy thêm inference hoặc bộ test local.
Notebook cuối chỉ được kiểm tra nhanh cấu trúc/cú pháp. Không chạy toàn corpus,
OCR, dịch, upload hoặc sửa notebook 2/3.

Kiểm tra tĩnh cuối: schema notebook, 9 ô code, worker và 67 file Python upstream
nhúng đều biên dịch cú pháp được; không thực thi model. Notebook 0/2/3 khớp HEAD
theo byte. Mặc định mới giảm còn một PDF/một trang để thử nhanh trên Kaggle.

## Phạm vi đã triển khai

Notebook có 18 ô, gồm 9 ô code, chứa cấu hình, PDFium acquisition, layout/crop,
dataset canonical, các phép biến đổi, worker/source upstream, staging asset,
preflight, 15 kiểm thử nhúng, xử lý, QC, preview và báo cáo. Không import
`vtm_corpus`, không cần `src/`, `tools/`, `tests/`, `configs/` hoặc notebook 3
trong runtime. Source worker và code model được sinh từ notebook vào working.

Giữ thuật toán PDFium đã có: native chỉ khi bằng pixel với visible render;
rotation/crop/mask/annotation/overlay không bị bỏ qua. Ảnh phân tích tách khỏi
nguồn; canonical không bị cap theo model. Page/group/column ứng viên giữ đủ
phân hoạch, tọa độ và RTL; không tuyên bố đó luôn là cột văn bản thật. Manual
cuts và profile nguồn của các sách hiện có được nhúng. Deskew/dewarp tắt vì chưa
có bằng chứng cần thiết và chưa có phép đổi hình học được xác minh cho corpus.

| Phương pháp | Tài sản đã có | Trạng thái và giới hạn |
|---|---|---|
| Original, CLAHE, bicubic ×2/×4, NLM | PDFium Linux dùng lại; Pillow/NumPy/OpenCV từ runtime | Đã sinh ảnh local. Original bắt buộc, NLM opt in. Mỗi biến thể lấy độc lập từ canonical |
| Real-ESRGAN ×4 | `RealESRGAN_x4plus.pth`, 67.040.989 byte | Đã có inference local trước chỉ đạo dừng. RRDBNet 64/23/32, strict weights, tiling; chưa thử Kaggle |
| TADiSR ×4 + mask | CogView4 đầy đủ và adapter: 21 file, 31.914.476.368 byte | Nhúng API CogView4-RealCE upstream. Embedding có thể tạo offline trong worker riêng từ base đã xác minh. Chưa chạy inference; GPU local 8 GB không đủ cho stack upstream |
| DiffTSR | `DiffTSR.ckpt` + `transocr.pth`: 4.937.344.761 byte | Chưa có runtime legacy được xác minh hoặc vùng corpus phù hợp đã rà soát. Dòng ngang, <=24 ký tự, W/H 1..8; wrapper còn yêu cầu gần 4:1 để tránh biến dạng. Không xoay cột dọc |
| UniHIR | AE, config/tokenizer và 6 font cần thiết: 17 file, 469.638.789 byte | Thiếu `model.safetensors` chính 29.302.764.504 byte; lượt tải chậm chưa hoàn tất đã dừng và bỏ file range không hoàn chỉnh. Chưa có runtime/input inscription phù hợp hoặc inference 80 GB-class |

Source và hợp đồng chính thức:
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN),
[TADiSR](https://github.com/mingcv/TADiSR),
[DiffTSR](https://github.com/YuzheZhang-1999/DiffTSR),
[UniHIR](https://github.com/ZZXF11/UniHIR).
DiffTSR có text prior `zhconv(..., 'zh-cn')` nội bộ: notebook yêu cầu opt in rõ,
không dùng dự đoán đó làm bản chép chữ. UniHIR là tái dựng sinh nội dung, không
phải SR thông thường; lưu ảnh, intermediate text/boxes và diagnostic riêng khi
chạy được. Không cung cấp confidence giả hoặc coi chữ sinh ra là ground truth.

## Asset và package để cập nhật Dataset thủ công

Tất cả tài sản bàn giao nằm dưới:

```text
kaggle_upload/models/image_restoration/{realesrgan,tadisr,difftsr,unihir}/
kaggle_upload/packages/image_restoration/{wheels,requirements,manifests,sources}/
```

Mỗi model có `assets.json` với danh sách byte/SHA-256. Inventory tổng hợp:
`kaggle_upload/packages/image_restoration/manifests/final_inventory.json`.
URL từng thành phần CogView4 và đối chiếu SHA-256 LFS chính thức nằm trong
`tadisr_publisher_checksums.json`; metadata ModelScope và blocker UniHIR cũng
được giữ trong manifests. Hash của weights Drive là hash local của tệp tải từ
liên kết upstream, không được gọi là checksum do tác giả công bố.

Các checkpoint chính:

- Real-ESRGAN: `4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1`.
- Adapter TADiSR: `7cec8d0917305037cbafcb1b1bf81062add1613b561f5d7e2d1e599fe0baf5e0`
  — đã khớp manifest phát hành chính thức.
- DiffTSR: `7e260843f19063a420f7e27ef1a0ed39111d864c92090ac94aa3d5bcfc515740`.
- TransOCR prior của DiffTSR: `3620c92893a2d9b71bd99479a1547aaf5043cf5033548f656cc4699aa2cfd828`.

Có **13 wheel Linux/pure Python**, tổng **23.494.899 byte**:
diffusers 0.35.2, peft 0.17.1, transformers 4.56.2, tokenizers 0.22.1,
accelerate 1.10.1, huggingface-hub 0.35.3, safetensors 0.6.2,
sentencepiece 0.2.1, importlib-metadata 8.7.0, zipp 3.23.0, packaging 25.0,
PyYAML 6.0.3, psutil 7.1.0. Đã đối chiếu SHA-256 với đúng wheel trên PyPI và
kiểm CRC; URL, wheel tags và license được ghi trong `tadisr_runtime.json`.
Không tải wheel Windows cho bộ Kaggle.

Dùng lại artifact PDFium 5.15.0 Linux hiện có, 4.025.240 byte; không ghi đè
Torch/CUDA/NumPy hay các manifest/weights OCR. Real-ESRGAN dùng kernel đã probe.
TADiSR có venv overlay riêng, chỉ cài wheel local với `--no-index --no-deps`,
kiểm tra dependency closure, version module/metadata, wheel ABI, RECORD của
module/native extension thực sự import, NumPy bridge và phép tính CUDA. Torch,
torchvision và NumPy kernel chỉ được chấp nhận sau các probe đó trên runtime
thực tế. **Chưa thực hiện bước cài/probe này trên Kaggle.**

DiffTSR/UniHIR có requirements upstream và manifest blocker, chưa có môi trường
Linux hoàn chỉnh được xác minh. Cần Python/runtime riêng qua `RUNTIME_PYTHONS`;
không tự hạ kernel để cài Torch/CUDA/flash-attn cũ. Các phiên bản Kaggle trong
tài liệu repo chỉ là runtime tham chiếu, không phải inventory của một phiên mới.

License: Real-ESRGAN BSD-3-Clause, kiến trúc BasicSR Apache-2.0; code TADiSR MIT,
adapter research use và base CogView4 Apache-2.0; DiffTSR Apache-2.0; UniHIR
CC BY-NC-ND 4.0 cho nghiên cứu phi thương mại. Source UniHIR được nhúng nguyên
trạng, wrapper riêng. Metadata sáu font chỉ nêu tác giả/copyright, chưa thiết lập
được quyền phân phối lại: xem `font_license_metadata.json`, không coi license
repo là bằng chứng tất cả font đều được phân phối không hạn chế. Font thứ bảy
không được code upstream sử dụng đã bị loại khỏi thư mục upload.

**Các ZIP models/packages cũ không được cập nhật.** Khi upload thủ công, dùng
nhánh `image_restoration/` mới cùng cấu trúc Dataset đang dùng; không giả định
archive cũ đã chứa tài sản mới. Không có thao tác upload tự động.

## Dataset và resume

Core giữ `vtm-image-dataset` v1, original-only, với `dataset.json`, immutable
snapshots, object PNG theo nội dung, source/acquisition/layout/page records và
manifest. Reader hiện hành của notebook 3 có thể tiếp tục dùng **original** theo
hợp đồng v1; chưa chạy notebook 3 trên Kaggle để xác nhận tích hợp thực tế.

Phục hồi nằm trong extension `vtm-image-restoration` v1, đọc bằng `prep_open` và
`prep_validate` trong notebook. Đây là quyết định cần thiết vì validator cũ coi
mọi tên ngoài bicubic ×2 là scale 1. Không đưa x4 vào `regions.images` cũ, không
sửa notebook 2/3 và không dùng ZIP packager v1 cũ cho extension.

Mỗi dòng phục hồi có book/page/region, bounding box nguồn, crop type/thứ tự đọc,
hash source/canonical, phương pháp, tham số, architecture/checkpoint identity,
kích thước, affine về source, QC, runtime và lỗi. Khi chạy được, mask và
diagnostic/prediction cũng có hash/reference riêng. UniHIR ghi phép ánh xạ canvas
chỉ mang tính danh nghĩa; không tuyên bố alignment từng glyph sinh ra.

Phân biệt bốn loại provenance: `source_observation`,
`deterministic_transformation`, `neural_restoration`, `generative_reconstruction`.
Mọi phương pháp được chọn đều có trạng thái. `not_applicable` không phải kết
quả giả. Thiếu asset/model phù hợp dừng ở preflight trừ khi bật partial rõ ràng.

Cache nguồn/layout và từng variant độc lập. Đổi tham số/checkpoint/code làm đổi
cache tương ứng; thêm model không đổi nguồn/crop. Ghi atomic, phát hiện ảnh hỏng,
khóa tiến trình, per-region timeout, pixel/storage budget, một heavy model mỗi
tiến trình. TADiSR bắt buộc có cả SR/mask; tiling xử lý độc lập hai chiều và kiểm
trọng số phủ đủ cả crop hẹp/dài. `dataset.json` chỉ công bố sau QC; lỗi/ngắt giữ
snapshot trước và các receipt hoàn tất. `canonical.json` chỉ là stage nguồn.

## Bằng chứng đã có trước khi dừng thử local

- 15 kiểm thử nhúng đạt trong **8,482 giây**, chặn mạng và import `vtm_corpus`:
  PDF native/fallback/rotation/crop/overlay/mask/annotation, phân hoạch/RTL,
  variant geometry, missing assets, worker isolation và chặn kết nối, resume
  sau ngắt, cache/ảnh hỏng, checkpoint identity, storage budget và manifest.
  Test tiling dùng tensor identity; test thiếu mask dùng backend giả có chủ đích
  để kiểm tra failure. **Không phải inference TADiSR/DiffTSR/UniHIR thật.**
- Smoke thật trước chỉ đạo dừng: `4.pdf`, trang 50; original 600×532, x2
  1200×1064, x4 2400×2128. Ba region page/group/column cùng box trong layout
  chưa tìm được gutter, nên deduplicate; 18 kết quả logic cho sáu phương pháp.
  Tổng lượt **28,913 giây**, đã dùng lại canonical từ lượt trước.
- Real-ESRGAN thật, tile 128/padding 24, FP32: worker xử lý region **6,051 giây**,
  model load **0,507 giây**; outer region gồm startup/encode/QC **14,309 giây**.
  Peak allocated **355.271.680 byte (338,8 MiB)**; peak reserved **901.775.360 byte
  (860 MiB)**. Đây là số đo RTX 4060 Laptop local, không dự báo tốc độ/VRAM Kaggle.
- Resume dùng lại đủ 18/18 variant receipt, snapshot giữ nguyên, **11,073 giây**;
  vẫn gồm import/GPU preflight và kiểm integrity. Thư mục lúc đo 13.228.627 byte,
  gồm cả metadata/cache phát triển trước đó; không ngoại suy dung lượng corpus.
- Có contact sheet đã xem ở `work/offline_image_smoke/previews/`; không thấy
  thay đổi hình học lớn ở thumbnail. Chưa xác minh fidelity từng nét/character
  hoặc seam ở mọi vùng, không kết luận restoration cải thiện OCR.
- Lượt kernel Jupyter local đã sinh hai trang `1.pdf`, 125 dòng classical, trong
  15,722 giây. Harness dọn kernel gặp WinError32 nên không có artifact Run All
  được chốt; **không tính là phép kiểm chứng Run All hoàn tất** và không chạy lại
  sau chỉ đạo của người dùng.

Artifact local: `work/offline_notebook_test_results.json`,
`work/offline_image_smoke_results.json`, `work/offline_image_smoke/last_run.json`,
`work/offline_run_all_dataset/last_run.json`. Những file này ở `work/` ngoài Git;
notebook nhúng code kiểm thử để người dùng có thể thực hiện lại trên Kaggle.
Các sửa cuối về provenance/checkpoint metadata, font cache và báo lỗi đường dẫn
chỉ được kiểm tra cú pháp/cấu trúc, không có lượt thực thi local mới.

## Chạy thử ngắn trực tiếp trên Kaggle

1. Cập nhật hai Dataset models/packages thủ công từ thư mục mới; gắn thêm PDF.
   Tắt Internet. Không dùng các ZIP cũ như bản bundle đã cập nhật.
2. Import notebook mới, điền ba đường dẫn. Giữ `BOOKS=['1']`, `MAX_BOOKS=1`;
   thử nhanh nhất với `PAGES='1'`, `LIMIT=1`, `CROP_TYPES=['page']` và một phương
   pháp cùng original. Chưa chọn full corpus.
3. Thử classical/Real-ESRGAN trước. Đối với TADiSR, chọn
   `VARIANTS=['original','tadisr_x4']`; SR và mask luôn đi cùng. Nếu không upload
   embedding, giữ `GENERATE_TADISR_PROMPT_OFFLINE=True`: worker tạo embedding
   bằng đúng script upstream từ base local đã xác minh rồi thoát trước model SR.
4. Xem `preflight.json`, log worker, `last_run.json`, contact sheet toàn vùng và
   detail; ghi GPU/ABI/runtime thực của Kaggle. Chỉ sau khi một model qua lượt
   này mới tăng phạm vi hoặc cấu hình lưu trữ.
5. Resume bằng output ghi được đã giữ lại. Khi mở phiên Kaggle mới, chép output
   trước sang working; không đặt OUTPUT/RUNTIME dưới input chỉ đọc.
6. Full corpus: `BOOKS=[]`, `PAGES=''`, `LIMIT=None`, `MAX_BOOKS=None`. Đây chỉ là
   cấu hình hỗ trợ, chưa chạy. Output ở `/kaggle/working/image_dataset/`.

Thiếu runtime/weights hoặc input không đúng hợp đồng phải được giữ là blocker,
không tạo ảnh thay thế và không chuyển nội dung dự đoán vào canonical.
