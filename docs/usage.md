# Vận hành và kiểm chứng

Hướng dẫn đối chiếu mã ngày **12/11/2026**. Bắt đầu bằng một sách/một trang;
chỉ mở rộng sau khi xem ảnh, đường cắt, output và trạng thái lỗi. Các thuật ngữ
và hợp đồng tọa độ nằm trong [pipeline](pipeline.md); mục tiêu đánh giá nằm trong
[bài toán](problem.md). Không cần inference mới khi chỉ đọc checkpoint.

## Thứ tự notebook

| Notebook | Cách dùng |
|---|---|
| [0 — môi trường](../0_kiem_tra_moi_truong_kaggle_offline.ipynb) | Kiểm kê Python, package, ABI, GPU và phép thử nhỏ; xuất JSON/TXT. Không cài package hay OCR. Chạy lại khi kernel đổi. |
| [1 — chuẩn bị ảnh](../1_chuan_bi_anh_ocr.ipynb) | Phiên riêng: PDF → canonical, biến thể, manifest, QC (kiểm tra chất lượng) và preview; chuyển output thành Dataset. Không bắt buộc cho OCR PDF trực tiếp. |
| [3 — OCR và rà soát](../3_so_sanh_so_chu_ocr.ipynb) | Chạy từng engine, lưu checkpoint, so sánh và xuất báo cáo offline. |
| [2 — dịch](../2_dich_han_viet.ipynb) | Dịch TXT Hán đã chọn; kiểm tra `INPUT_DIR` vì local mặc định tìm run OCR mới nhất, chưa chắc là bản đã duyệt. Kaggle vẫn cần mạng để clone/cài repo và tải model. |

## Môi trường và tài sản offline

Local Windows chọn interpreter/kernel `.venv/Scripts/python.exe` tại root,
Python 3.13. [pyproject.toml](../pyproject.toml) cho phép Python `>=3.10,<3.14`,
có extra `notebook`, `paddle-gpu`, `hunyuan`, `translate`, `report`; không có yêu
cầu thêm Paddle CPU. Cấu hình local hai trang có ở
[ocr_local.json](../configs/ocr_local.json). Dùng môi trường hiện có; không hạ
Torch/NumPy để chữa xung đột của runtime riêng.

Kaggle: gắn packages, models và PDF thành các Input riêng rồi điền
`PACKAGES_DIR`, `MODELS_DIR`, `DATA_DIR` trong notebook 3. Có thể có folder bọc
`packages/` hoặc `models/`; không dùng `Path("")` làm đường dẫn chưa cấu hình.
`PREPARED_IMAGES` tùy chọn, chỉ tới dataset hoặc snapshot cố định. PDF vẫn cần
để xác minh SHA-256 và rà soát nguồn.

```text
models/
  hunyuan/
  ppocr_det/, ppocr_rec/
  paddle_vl/, paddle_layout/
  tonggu/ocr/, tonggu/layout/finetune_infer.pth
  models_manifest.json
  image_restoration/            # chỉ notebook 1 cần
packages/
  wheels/, sources/, requirements/
  paddle_manifest.json, tonggu_manifest.json
  system_assets_manifest.json, bundle_manifest.json
  image_restoration/            # tài sản riêng cho notebook 1
pdfs/*.pdf
image_dataset/                 # bắt buộc khi OCR nhiều biến thể
  dataset.json, snapshots/, objects/
```

Runtime tham chiếu lịch sử là Linux x86_64/glibc 2.39, CPython 3.13.16,
CUDA 13.0, GCC 13.3, RTX PRO 6000 Blackwell `sm_120`, khoảng 95 GiB VRAM.
Đây không phải cam kết cấu hình Kaggle hiện tại; chạy notebook 0 để xác nhận.
Không áp cứng hạn mức phiên/GPU từ tài liệu cũ cho mọi tài khoản.

| Thành phần | Điều kiện của bộ cài tham chiếu |
|---|---|
| HunyuanOCR-1.5 | Dùng kernel sau API/NumPy/CUDA probe; revision mặc định `47644ecc4fc854efa4f505155158831f36773ee4`. |
| PP-OCRv6-medium, PaddleOCR-VL-1.6 | Python 3.13 riêng, Paddle GPU 3.2.1, PaddleOCR/PaddleX 3.7.x. |
| TongGuOCR Stage 3 | Python 3.10 riêng, Torch 2.10.0+cu130, torchvision 0.25.0+cu130, Transformers 5.3.0; build/probe OpenMMLab/MMCV riêng. |

Installer dùng wheel/source local với `pip --no-index --no-deps`, xác minh
version, dependency closure, ABI và checksum `RECORD`. Không đưa extension cp313
vào cp310. Chỉ một provider OpenCV trong mỗi runtime. Source upload là file
thường, không symlink/archive lồng; installer phục hồi tên Linux/quyền thực thi
và build trong bản sao ghi được dưới `/tmp`. Chỉ bỏ qua `.pyc` thiếu nếu `.py`
nguồn tương ứng đã được xác minh; thiếu payload thật phải dừng.

Probe compiler/C extension, NumPy bridge, CUDA và `mmcv.ops.nms_rotated` trên GPU
phải đạt trước OCR. Cache native wheel gắn identity source/Python/NumPy/Torch/
CUDA/compiler/cờ build và hash; chạy lại Setup có thể dùng component đã xác minh
trong cùng phiên. `TORCH_CUDA_ARCH_LIST=12.0` chỉ phù hợp GPU đích đã xác nhận.
TongGu local cần WSL/Linux Python riêng qua `ADVANCED['tonggu_repo']`,
`tonggu_python`, `tonggu_wsl_distro`; ánh xạ `C:` → `/mnt/c`, chưa hỗ trợ UNC.

Giữ `runtime_config.json`, `notebook_setup_receipt.json`, `build_logs/`,
`build_probe/toolchain.json`, `.vtm_kernel_plan.json`, `.vtm_opencv_payload.json`
và `.vtm_opencv_probe.json` để chẩn đoán. Runtime `/tmp` cần tạo lại ở phiên mới;
Input chỉ đọc. Upload models/packages/PDF/prepared images là các bước riêng,
chưa được xác nhận bởi việc kiểm hash local. ZIP cũ không tự có asset mới; một
bản vá NCCL hay `sources_upload_test.zip` không thay bộ packages đầy đủ.

## Chuẩn bị ảnh

Trong notebook 1, dùng `RUN_MODE='process'`, điền ba đường dẫn tài sản và giữ
`BOOKS=['1']`, `PAGES='1'`, `LIMIT=1`, `MAX_BOOKS=1`. Bắt đầu chỉ với
`VARIANTS=['original']`, sau đó thêm classical/biến thể cần so. Chế độ process
chỉ dùng tài sản local; tải tài sản là chế độ `prepare_assets` riêng trên máy
local với các cờ cho phép mạng. `RUN_SELF_TESTS`, `VALIDATE_OUTPUT` và
`EXPORT_PREVIEWS` hỗ trợ kiểm tra trước khi chuyển dataset.

Mặc định có `CROP_TYPES=['page','group','column']`, `RESUME=True`,
`ALLOW_PARTIAL=False`, nguồn/đầu ra tối đa 100 triệu pixel, storage 15 GiB,
timeout vùng 600 giây, load model 900 giây, batch 1. Thiếu asset hoặc lỗi không
được công bố như dataset đủ; `ALLOW_PARTIAL` phải bật có chủ đích. TADiSR tile
hai trục và kiểm phủ crop hẹp/dài; mỗi model có seed/cache/preflight riêng.

Kiểm tra preview đường cắt: `vertical_rtl` đọc phải → trái, trên → dưới;
`horizontal_ltr` dùng cho bố cục ngang. `column_cuts` là tỷ lệ ngang tăng dần
trong `(0,1)`; `page_column_cuts` đếm trang từ 1, `[]` giữ nguyên trang. Phép
chiếu không hiểu chú thích; không ép cắt nếu chưa có khe đáng tin cậy.

Trong notebook 3, `IMAGE_PREPARATION` mặc định segmentation `auto`: Hunyuan chia
nhóm khoảng ba cột; PP-OCR chia thêm khi vượt ngân sách; PaddleOCR-VL/TongGu giữ
layout native khi không ép crop chung. `resolution_strict=True` làm trang lỗi
khi không thể chia mà vẫn giữ độ phân giải, thay vì chấp nhận thu nhỏ.

Các [profile sách](../configs/ocr_books/README.md) là điểm khởi đầu hình học:
`1`:225/750, `2–3`:226/754, `4–5`:181/704, `6`:226/653,
`bien`:200/2333, `kinh`:45/2206 (DPI/`max_side`). Chúng chưa tối ưu theo accuracy.
DPI phụ thuộc kích thước trang PDF; scan 1–6 vốn ít pixel, DPI lớn hơn không tạo
nét thật. `ADVANCED['use_book_profiles']=False` để dùng cấu hình thủ công;
`ADVANCED['ocr']` chỉnh chung, `model_overrides` chỉnh từng engine. Token cap
profile là 6000, không ép model sinh đủ; `hit_max` cần rà soát. Xem thêm
[script kiểm kích thước](../scripts/README.md) và cấu hình
[smoke](../configs/image_preparation_smoke.json)/[experimental](../configs/image_preparation_experimental.json).

### Biến thể và giấy phép

| Biến thể | Điều kiện và giới hạn cần giữ |
|---|---|
| `original` | Đối chứng `source_observation`. |
| `clahe`, `denoised` | Biến đổi deterministic scale 1; có thể xóa nét yếu. |
| `bicubic_x2`, `bicubic_x4` | Nội suy, affine 0,5/0,25 về nguồn; không tái tạo nét thật. |
| `realesrgan_x4` | Neural restoration ×4; Real-ESRGAN BSD-3-Clause, BasicSR Apache-2.0; RRDBNet 64/23/32, weights ×4plus. |
| `tadisr_x4` | Neural restoration ×4; code MIT, adapter phạm vi research, base CogView4 Apache-2.0. Mask đi kèm không dùng làm ảnh OCR. |
| `difftsr` | Apache-2.0; cần vùng đã duyệt: dòng ngang thẳng, 1–24 chữ, gần tỷ lệ 4:1, cho phép prior giản thể upstream; output 512×128. Không xoay/reflow cột dọc để ép chạy. |
| `unihir_restoration` | CC BY-NC-ND 4.0, nghiên cứu phi thương mại; cần khai báo vùng inscription và opt in. Canvas danh nghĩa không xác nhận glyph nguồn. Quyền phân phối lại font chưa xác lập chỉ từ metadata tác giả. |

Notebook 1 dùng `REVIEWED_REGIONS` để khai báo điều kiện sau khi xem original;
UniHIR cần `ACCEPT_UNIHIR_NONCOMMERCIAL`. Notebook 3 chỉ nhận DiffTSR/UniHIR khi
`ALLOW_GENERATED_VARIANTS=True`. Không tự chuyển phồn thể thành giản thể do
prior `zhconv(...,'zh-cn')`. Giữ license/revision/URL/checksum trong `assets.json`
và manifest tài sản; không suy giấy phép weights/PDF từ giấy phép wrapper.
Source upstream nhúng giữ nguyên license; không đưa binary/font vào Git.

Notebook 1 nén mã upstream bằng zlib/Base85 và kiểm SHA-256 khi giải nén.
`RUNTIME_PYTHONS={}` cho phép bộ cài chuẩn bị overlay riêng: Real-ESRGAN dùng
kernel, TADiSR có runtime riêng; DiffTSR/UniHIR dùng staged overlay kế thừa Torch
phù hợp. UniHIR cần build FlashAttention offline từ source local bằng CUDA/GCC
cho GPU thực tế; log dưới `/tmp/vtm_image_runtime/environments/.../native_build/build.log`.
Chưa xác nhận build/inference này trên Kaggle trong phạm vi tài liệu.

Với bộ tài sản được ghi nhận ngày 11/11/2026, DiffTSR cần `zhconv`, UniHIR dùng
Transformers 4.49.0 theo upstream và source FlashAttention 2.8.3.post2 có CUTLASS;
không thay bằng Transformers 5 của kernel. Bộ staging cũ ghi 21 wheel restoration
và 13 wheel TADiSR riêng. Phải kiểm lại manifest/closure theo kernel thực tế;
đủ metadata không bảo đảm import, native build hay inference thành công.

Công cụ [prepare_kaggle_offline.py](../tools/prepare_kaggle_offline.py) mặc định
chỉ in kế hoạch; `--download-small` tải vào `kaggle_upload/`, không cài `.venv`.
Nó hoãn file trên 5 GB hoặc ước tính quá 300 giây, không tự tải UniHIR 29,3 GB.
[download_unihir_checkpoint.py](../tools/download_unihir_checkpoint.py) cũng
mặc định kiểm tra; chỉ tải với `--download --accept-unihir-noncommercial`.
Có thể thêm `--connections 4` (hỗ trợ 1–8, mặc định 1). Dừng tiến trình cũ bằng
Ctrl+C trước khi tiếp tục; giữ `.part`, receipt và khóa. Script resume bằng HTTP
Range, kiểm đủ kích thước/SHA-256 trước khi công bố; file sai được giữ để xem.
Cập nhật Dataset models/packages và manifest sau khi staging; ZIP cũ không tự đổi.

## Chạy OCR từ mẫu nhỏ

Trong ô cấu hình notebook 3, sau khi điền Input:

```python
BOOKS = ["1"]
MAX_BOOKS = 1
PAGES = "1"
LIMIT = 1
RUN_OCR = True
PREPARED_IMAGES = ""
PREPARED_VARIANT = "original"
ENABLE_MULTI_VARIANT_OCR = False
ENABLE_SPATIAL_COVERAGE = False
```

**Notebook 3 mặc định `PAGES="1"`, `LIMIT=1`: một PDF/một trang.** `BOOKS` nhận stem/tên/glob; `MAX_BOOKS` giới hạn PDF sau
lọc. `PAGES` đếm từ 1 trong mỗi PDF (ví dụ `"2,50"`); `LIMIT` chặn tổng trang sau
chọn trên mọi PDF. Nên thử thêm trang nội dung có chữ nhỏ/mờ/chú thích, không
chỉ bìa. Sau khi xem mẫu mới cân nhắc `BOOKS=[]`, `MAX_BOOKS=None`, `PAGES=""`,
`LIMIT=None`; đây là cấu hình mở rộng, không phải lượt chạy đã thực hiện.

Muốn dùng ảnh chuẩn bị sẵn, thay các trường sau bằng dataset thật:

```python
PREPARED_IMAGES = "/kaggle/input/YOUR_IMAGES"
PREPARED_VARIANT = "original"
PREPARED_CROP_TYPE = "group"  # page | group | column | native
PREPARED_MISSING_FALLBACK = False
```

`native` gửi nguyên trang nhưng layout bên trong vẫn có thể khác. Multi-variant
bắt buộc prepared images, không bật missing fallback và phải có original:

```python
ENABLE_MULTI_VARIANT_OCR = True
OCR_VARIANTS = ["original", "bicubic_x4"]  # phải chuẩn bị trước
ALLOW_GENERATED_VARIANTS = False
VARIANT_STABILITY_OPTIONS = {
    "reference_variant": "original",
    "compare_by_engine": True,
    "compare_across_engines": True,
    "save_difference_crops": True,
}
```

Mask/tên sai bị từ chối; biến thể thiếu giữ trạng thái, không làm prediction giả.
Một model xử lý các trang/biến thể hợp lệ rồi thoát trước model kế tiếp. TongGu
mặc định timeout 1800 giây/trang, ba engine còn lại 600; token cap theo lần đọc
vùng/block. Một vùng lỗi làm trang chưa hoàn tất.

Để phân tích ảnh nguồn độc lập với output OCR:

```python
ENABLE_SPATIAL_COVERAGE = True
SPATIAL_COVERAGE_OPTIONS = {
    "analysis_max_side": 1600,
    "save_overlays": True,
    "save_candidate_masks": False,
    "max_detail_crops": 12,
}
```

Coverage dùng được cả PDF trực tiếp hoặc log cũ nếu xác minh nguồn; không cần
chạy lại model. CLI một engine vẫn dùng được, ví dụ từ root:

```powershell
.venv/Scripts/vtm.exe ocr --config configs/ocr_books/1.json --set pages=1-2
```

CLI không chọn trang thì có thể chạy cả sách; segmentation mặc định của CLI là
`none`, khác notebook. `vtm images prepare`, `validate`, `package` là luồng core;
packager core không đóng extension restoration. Chuyển dataset notebook 1 phải
giữ đủ object/report/manifest/snapshot được tham chiếu.

## Checkpoint và tái lập

Đọc lại: `RUN_OCR=False`, `EXPORTED_LOG_DIR=""`, giữ `WORK_DIR` và cờ
`ENABLE_MULTI_VARIANT_OCR` đúng loại campaign. Single dùng `manifest.json`, multi
dùng `multi_variant_manifest.json`; `ADVANCED['existing_manifest']` chỉ định
manifest khác. Phân tích danh sách biến thể đã lưu, không tự thêm biến thể mới.
`EXPORT_RESULTS=True` tạo lại báo cáo mà không tải model.

Import log cũ: đặt `RUN_OCR=False`, `EXPORTED_LOG_DIR='data/log_notebook_3'`,
`DATA_DIR='data'`, `PAGES='2'`, `LIMIT=1` nếu các đường dẫn này tồn tại. Import
tạo checkpoint riêng, giữ log/PDF gốc. Source chưa xác minh phải giữ cảnh báo.

```text
WORK_DIR/
  manifest.json, multi_variant_manifest.json
  runs/ocr_<name>_<config>_<source>/
    run_manifest.json, tasks.jsonl
    ocr_raw/<book>/<name>_shard0.jsonl
    ocr_raw/<book>/<name>_errors.jsonl
    txt/<book>.txt
  variant_runs/<variant>/
  render_cache/, evidence/, *_notebook.log
  output/<book>/
    index.html, pages/p00001.html, style.css, reader.js
    raw/<engine>.jsonl, comparison.jsonl, errors.jsonl, manifest.json
    variants/<variant>/<engine>.jsonl
    crops/, evidence/
  output/<book>_results.zip
```

Các manifest trong sơ đồ là theo chế độ, không phải run nào cũng có đủ mọi file.
Dataset ảnh giữ `dataset.json`, `snapshots/`, `manifests/`, `pages/`, `sources/`,
`layouts/`, `variants/`, `objects/` và extension `restoration_manifests/`,
`restoration_cache/`. Ảnh lưu theo hash, JSON giữ reference thay base64.
Chuỗi `source → crop → variant → engine → raw → finding` liên kết bằng source
SHA-256, canonical region ID, snapshot, cấu hình và hash ảnh.

Giữ snapshot bất biến và manifest kèm model/revision, prompt, decoding, pixel
budget, seed, môi trường. Hash cấu hình core dùng SHA-1 cho tên run, khác SHA-256
kiểm nội dung. Restoration có `image_content_key` theo nguồn/vùng/thứ tự/tài sản
của riêng biến thể; thêm biến thể khác không vô hiệu run hợp lệ. Đổi nguồn,
pixel/model/preprocessing có thể tách run. Core snapshot mới vẫn có thể tách
original theo quy tắc bảo thủ. `resume_from` chủ động chép kết quả cũ: không dùng
nó để tạo thí nghiệm “mới” khi đã đổi cấu hình.

JSONL được fsync theo trang; đuôi ghi dở được giữ riêng trước khi sửa append.
Trang lỗi không thành prediction. Cùng attempt bỏ trang vừa lỗi để tránh lặp
vô hạn; lượt notebook mới thử lại trang thiếu. Windows dùng timer/hard exit
`PAGE_TIMEOUT_EXIT=124` và supervisor restart; Unix dùng alarm. Giữ checkpoint
và config để resume. Phiên Kaggle mới chép output cũ sang working ghi được,
không ghi resume vào Input. Không bật `fresh` nếu muốn giữ run.

Schema evidence gồm `vtm-multi-variant-campaign-v1`, `vtm-spatial-coverage-v1`,
`vtm-variant-stability-v1`, `vtm-evidence-v1`; export `vtm-ocr-book-v1` giữ trường
cũ, thêm evidence khi bật.

## Đọc báo cáo

Giải nén giữ nguyên cây rồi mở `index.html`; không chép riêng HTML. Xem raw từng
engine, source crop, quality flags và các phương án trước khi chọn TXT dịch.
Trang có các tab OCR, đồng thuận, coverage và stability; có lọc engine/variant/
region. Màu xanh dương là ứng viên nguồn, xanh ngọc là geometry OCR, đỏ là nghi
chưa phủ, cam là chưa chắc chắn. `complete` chỉ là đủ output đã chọn.

| Nhãn | Diễn giải |
|---|---|
| `suspected_missing_region`, `suspected_duplicate_region`, `suspected_reading_order`, `suspected_extra_region` | Nghi vấn cấu trúc từ mốc text; xem raw/ảnh trước khi kết luận. |
| `region_text_mismatch`, `unmatched_region_text` | Văn bản không khớp hoặc thiếu đối ứng đủ rõ. |
| `potential_uncovered_text_region` | Ứng viên ảnh ít overlap geometry đo được; chưa chứng minh bỏ chữ. |
| `partially_covered_region` | Chỉ một phần mực ứng viên giao vùng OCR. |
| `covered_by_character_evidence` | Đủ overlap khung ký tự; không xác nhận reading. |
| `covered_by_coarse_ocr_region` | Chỉ có khung dòng/khối, có thể bỏ sót bên trong. |
| `ambiguous_visual_candidate`, `geometry_unavailable` | Ứng viên chưa chắc là chữ hoặc không đủ geometry. |
| `source_geometry_mismatch`, `analysis_incomplete` | Nguồn/mapping không khớp hoặc phân tích chưa đủ ngân sách. |
| `ambiguous_alignment` | Không xác định chắc vị trí thay/chèn/xóa; số thao tác có thể `null`. |
| `failed`, `missing_assets`, `not_applicable`, `skipped_by_configuration`, `missing_variant` | Trạng thái ảnh/biến thể, không phải output trống hợp lệ. |

Ngân sách page consensus mặc định: mốc 4 chữ, 10 mốc/vùng, vùng tối thiểu 12
chữ, đơn vị tối đa 80 chữ, 20000 chữ/model, 256 vùng, 8 lần xuất hiện/mốc,
64 mốc toàn trang. Vượt giới hạn giữ `analysis_limit`; chỉnh bằng
`ADVANCED['page_consensus_options']`. Character alignment giới hạn 8 triệu ô.
Stability giới hạn 20000 chữ, 256 vùng/thay đổi, 12 difference crop và 8 triệu ô;
raw đầy đủ vẫn giữ. Chữ `未` đổi thành `末` sau SR chỉ chứng minh reading thay đổi,
không xác định chữ nào đúng hay SR đã tạo nét giả.

Coverage mặc định cạnh phân tích 1600, 30000 component, 256 ứng viên, 12 crop;
contrast tối thiểu 10, window 31, ít nhất 3 component/40 pixel mực, mật độ
0,08–0,70, mép 2,5%. Ngưỡng overlap 0,85 là phủ, 0,10 là một phần. Đây là
heuristic chưa hiệu chuẩn, không phải xác suất đúng. Stability ghi cả kích thước
prepared → worker → processor; không quan sát được thì `null`. Khác pixel
budget/layout là yếu tố gây nhiễu khi so restoration.

## Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| Thiếu model/package/PDF | Kiểm ba Input và manifest; folder tồn tại chưa chứng minh weights đủ. |
| ABI/CUDA sai | Chạy notebook 0, đối chiếu runtime riêng và wheel tag; không trộn cp313/cp310. |
| NCCL checksum sai | Kiểm wheel `nvidia-nccl-cu13==2.28.9` đúng pin TongGu trong bundle; không bỏ qua checksum hoặc lấy pin Paddle thay thế. |
| OpenCV overlap/thiếu `resize` | Giữ một provider, xem payload/RECORD audit và reinstall đúng wheel local. |
| Source/native build lỗi | Xem `source_tree_report.json`, toolchain/build log, header/prefix và cache receipt; không tạo file giả. Probe GPU ops trước inference. |
| `sitecustomize`/`wrapt`/inline backend | Launcher private loại biến Python host và dùng `MPLBACKEND=Agg`; kiểm log runtime riêng. |
| Snapshot/PDF/object hash sai | Gắn đúng phiên bản hoặc kiểm file hỏng; không sửa hash để ép QC qua. |
| DiffTSR/UniHIR `not_applicable` | Kiểm metadata vùng, opt in và license; không đổi nhãn để ép chạy. |
| Geometry thiếu | Giữ text-only/coarse; không suy tọa độ bằng chia đều dòng. |
| Hết VRAM/storage hoặc timeout | Giảm trang/biến thể, kiểm crop/processor, xem `*_errors.jsonl` và stage log; không chạy bốn model cùng lúc. |
| Chọn sai checkpoint | Kiểm `existing_manifest`, single/multi và `WORK_DIR`; chạy lại cùng cấu hình để bổ sung thiếu. |
| HTML thiếu ảnh | Giữ nguyên cây export, kiểm đường dẫn tương đối và hash evidence. |

## Mức kiểm chứng

Đợt triển khai H10 ngày 12/11/2026 chạy **361 test CPU: 360 đạt, 1 bỏ qua**
(nhánh Unix alarm trên Windows). Trong đó 12 test H10 dùng PDF/ảnh tổng hợp,
checkpoint và worker giả: ablation, engine thiếu, config/geometry/pixel/runtime
mismatch, fidelity warnings, cache/resume, snapshot partial, archive extension,
chuyển cả dataset/checkpoint/PDF sang root mới, export và nhãn thủ công.
JavaScript lưu/đọc nhãn được kiểm bằng DOM fixture trong Node, không phải browser QA.

Kiểm tra tĩnh: 33 module nguồn, 4 notebook/29 code cell, 63 liên kết Markdown local;
JSON/schema, cú pháp và source nhúng khớp builder. Notebook 1 giữ checksum mã
upstream và dung lượng dưới 1 MB. Log hồi quy mới nằm ở
`work/h10_regression_tests.log` trong workspace này.

Không chạy phiên notebook/Run All, inference model thật, tải weights, cài môi
trường GPU hoặc upload/thực nghiệm Kaggle. Kiểm thử worker giả chỉ xác minh
điều phối và hợp đồng, không chứng minh OCR đúng hay model thật hoạt động.
Browser không khả dụng trong phiên kiểm tra này, nên chưa xác minh layout và
thao tác HTML trên trình duyệt thật. Các giới hạn về model bên dưới vẫn giữ nguyên.

Tài liệu trước ghi các bằng chứng lịch sử sau; chưa chạy lại toàn bộ artifact:

| Mốc và môi trường | Phạm vi và giới hạn |
|---|---|
| 11/11/2026, local Windows/CPU | Ghi nhận 312 test, 311 đạt, 1 bỏ qua nhánh Unix alarm. Đây là số của lần chạy cũ, không phải kết quả mới. |
| 11/11/2026, log PDF 1 trang 2 | Smoke export/coverage: 177 ứng viên, 169 cần rà soát do coarse/không chắc chắn; không chứng minh OCR đầy đủ. Cột cùng bị bỏ được kiểm bằng fixture tổng hợp. |
| 11/11/2026, ảnh lưu PDF 4 trang 50 | Đọc original/CLAHE/×2/×4/denoised/Real-ESRGAN và mapping, không inference mới. Tài liệu cũ ghi Real-ESRGAN từng chạy local. |
| 11/11/2026, kiểm kê local | Báo cáo 6090 file/~74,76 GiB khớp manifest; UniHIR chính 29.302.764.504 byte đủ hash. Đây không xác nhận asset hiện còn đủ hoặc runtime đã chạy. |
| 11/11/2026, báo cáo Kaggle do người dùng cung cấp | Python 3.13.16, Torch 2.11.0+cu130, CUDA 13, Blackwell; thiếu `zhconv`/`flash-attn` ở kernel. Chưa gắn models/packages/prepared nên không kiểm được asset upload hay private runtime. |

Báo cáo lịch sử có thể nằm tại `work/offline_assets_latest_audit.json`,
`work/evidence_delivery_smoke_20261111/` và các manifest
`offline_completion.json`, `ocr_report_compatibility.json` của tài sản restoration;
chúng là artifact local, không phải điều kiện luôn có trong checkout sạch.
TADiSR/DiffTSR/UniHIR có adapter nhưng chưa xác nhận inference thật trong bằng
chứng đã khảo sát. Log người dùng cũ về build MMCV không thay một lượt OCR bốn
model thành công. Chưa xác nhận multi-variant GPU/Kaggle, upload, toàn corpus,
QA HTML tương tác trong browser hoặc dịch hoàn toàn offline.

Muốn tái kiểm tra, dùng đúng phạm vi nhỏ và môi trường rồi lưu log mới. Ví dụ
smoke chỉ đọc log/ảnh có sẵn, từ chối ghi đè output, không inference/download:

```powershell
.venv/Scripts/python.exe -X utf8 tools/smoke_evidence_pipeline.py --logs data/log_notebook_3 --data-dir data --book 1 --page 2 --output work/my_new_evidence_smoke
```

Có thể thêm `--prepared-dataset` với thư mục thật. Sau khi sửa mã OCR, công cụ
[update_kaggle_notebook_bootstrap.py](../tools/update_kaggle_notebook_bootstrap.py)
sinh lại hàm/worker nhúng từ nguồn; notebook 3 chạy độc lập trên Kaggle.
Notebook 1 dùng [compact_image_notebook.py](../tools/compact_image_notebook.py),
`--check` kiểm dung lượng/checksum không ghi. Những bước sinh notebook này không
cần cho thay đổi Markdown. Bộ kiểm thử phần mềm hiện có:

```powershell
.venv/Scripts/python.exe -X utf8 -m unittest discover -s tests -q
```

Test CPU cần Pillow/PDFium/NumPy/OpenCV; test notebook cần IPython, bảng cần
pandas. Chọn provider OpenCV phù hợp môi trường. Không ngoại suy thời gian/VRAM
từ CPU hay GPU local 8 GB sang Kaggle, và không dùng proxy làm kết luận accuracy.

Một số công cụ local trong `configs/`, `scripts/`, `tools/`, `tests/` vẫn bị ignore;
liên kết vận hành lịch sử được đối chiếu với workspace hiện có. Đợt H10 cho phép
Git theo dõi các builder notebook, installer dependency và các test cần cho thay
đổi này. Các tài sản models/packages không được đóng gói hay upload lại.

## Chuyển dataset giữa hai phiên Kaggle

Notebook 1 và 3 độc lập phiên. Sau khi QC, tải toàn bộ cây dataset hoặc ZIP đã
kiểm tra rồi tự upload thành Kaggle Dataset; gắn vào Notebook 3 ở một phiên mới.
Đặt `PREPARED_IMAGES` tới `dataset.json` hoặc immutable snapshot dưới Input,
`PREPARED_MISSING_FALLBACK=False` và chọn cùng `PREPARED_CROP_TYPE`.
PDF gốc vẫn cần trong `DATA_DIR` để lập tác vụ và xác minh SHA-256.

Trong Notebook 1, `VARIANT_MODE='selected'` giữ danh sách `VARIANTS` (mặc định
original/CLAHE/×2/×4). `classical` chọn cả năm phương pháp cổ điển;
`all_eligible` yêu cầu cả chín phương pháp. Điều kiện vùng, asset, runtime và
license vẫn được giữ; không ép DiffTSR/UniHIR chạy trên scan không phù hợp.

```python
VARIANT_MODE = 'all_eligible'
ALLOW_PARTIAL = True  # lưu phần hợp lệ và trạng thái thiếu/lỗi; không gọi là complete
CROP_TYPES = ['group', 'column']
REGION_IDS = []       # hoặc ['column_000']; canonical vẫn giữ đầy đủ
RESUME = True
EXPORT_DATASET_ZIP = True
```

Các trạng thái manifest: `ok`, `not_applicable`, `missing_assets`,
`runtime_incompatible`, `inference_failed`, `timeout`, `resource_exceeded`,
`skipped_by_configuration`; reader vẫn chấp nhận `failed` của dataset cũ.
Thiếu phương pháp bắt buộc thì `partial=True`. Với `ALLOW_PARTIAL=False`, lỗi
preflight dừng trước PDF; lỗi lúc xử lý giữ cache/`last_attempt.json` và marker
snapshot cũ, không công bố canonical trung gian như đã hoàn tất restoration.

`EXPORT_DATASET_ZIP=True` dùng `DATASET_ZIP` và từ chối ghi đè archive. Gói chứa
snapshot hiện hành, canonical images/metadata/layout, restoration report/manifest,
ảnh và extras được tham chiếu, kể cả trạng thái chưa xong. Ảnh trùng chỉ lưu một
lần theo hash. Không kèm weights, PDF, cache cũ hoặc preview không được tham chiếu.
Di chuyển cả cây dataset giữ được tham chiếu tương đối. Đường dẫn asset/PDF cũ
trong report là lịch sử provenance, không dùng để tìm ảnh khi đọc lại.

## Đánh giá restoration trong Notebook 3

Ví dụ cấu hình sau khi đã có dataset và muốn tạo raw OCR cho các biến thể:

```python
PREPARED_IMAGES = '/kaggle/input/YOUR_IMAGES/image_dataset'
PREPARED_CROP_TYPE = 'column'
PREPARED_MISSING_FALLBACK = False
ENABLE_MULTI_VARIANT_OCR = True
OCR_VARIANTS = ['original', 'clahe', 'bicubic_x4', 'realesrgan_x4']
ENABLE_RESTORATION_ABLATION = True
ENABLE_VISUAL_FIDELITY = True
RESTORATION_EVALUATION_OPTIONS = {
    'max_regions': 256, 'max_review_cases': 48,
    'analysis_max_side': 1024, 'blind_review': False,
}
BOOKS = ['1']
MAX_BOOKS = 1
PAGES = '1'
LIMIT = 1
RUN_OCR = True
```

Chỉ chọn những variants bạn muốn so; thiếu/không áp dụng được ghi rõ, không đổi
sang original. Neural ×4 cần `bicubic_x4`. DiffTSR chỉ có đối chứng OCR cùng kích
thước khi dataset thật chứa bicubic đúng canvas; không tự tạo OCR cho một ảnh
thay thế. Cảnh báo visual có đối chứng nội suy sinh từ original để đo pixel,
độc lập với điều kiện phải có raw OCR bicubic trong ablation.

Ba tình huống sử dụng:

- **A — OCR mới:** `RUN_OCR=True`; từng worker xử lý các sách/trang/variants,
  một model/lượt rồi giải phóng. Không nạp model lại cho từng variant.
- **B — Đánh giá lại:** giữ `ENABLE_MULTI_VARIANT_OCR=True`, bật hai cờ H10,
  đặt `RUN_OCR=False`, `EXPORT_RESULTS=True` và
  `ADVANCED['existing_manifest']='/đường/dẫn/multi_variant_manifest.json'`.
  Chọn đúng campaign nhiều biến thể, không dùng manifest single-variant.
  Dataset ở `PREPARED_IMAGES` có thể đổi vị trí; reader kiểm pixel/provenance
  trước khi dùng. Giữ nguyên cây `runs/`, `variant_runs/` dưới checkpoint root; reader ưu tiên
  bản mounted hiện tại và tự đổi các tham chiếu campaign/run trên bản đọc.
  Điền `DATA_DIR` mới để tìm PDF và kiểm hash; raw manifest không bị ghi lại.
  Artifact bằng chứng được ghi dưới `WORK_DIR/evidence`, kể cả checkpoint ở Input;
  `EXPORTED_LOG_DIR` hiện chỉ nhập raw single-variant, không khôi phục H10 đầy đủ.
  Phạm vi đọc là các tasks trong manifest đã lưu; sửa `PAGES` không tự lọc campaign cũ.
- **C — Thêm variant:** giữ cùng workdir và cấu hình inference; thêm variant đã
  chuẩn bị vào `OCR_VARIANTS`, bật `RUN_OCR=True`. Worker chỉ chạy tasks thiếu
  checkpoint. Không thay weights tại cùng path mà giữ nguyên revision/identity.

Cache raw dùng source hash, content/geometry/variant identity, engine/model/revision,
prompt và cấu hình inference. Content key không phụ thuộc vị trí mount hoặc
những variants khác trong dataset. Snapshot/core v1 và extension v1 vẫn được
đọc, `failed` cũ vẫn hợp lệ. Checkpoint prepared cũ chưa có content key được nhận
lại khi snapshot/cấu hình cũ vẫn khớp chính xác; không đoán danh tính cho cache
thiếu provenance. Đổi pixel/geometry/inference làm mất tính tái sử dụng của phần
liên quan. `ocr_algorithm_version` chỉ đổi khi thuật toán tạo raw thay đổi;
logic H10/HTML không nằm trong cache key raw. Raw mới ghi processor/runtime
identity khi quan sát được; khác identity làm cặp ablation không hợp lệ. Raw cũ
không có trường này vẫn đọc được, nhưng báo rõ runtime/processor chưa quan sát.
Thay phiên bản package hoặc weights tại cùng path cần dùng identity/revision
mới hoặc tăng `ocr_algorithm_version` trước OCR mới.

HTML trang có mục Restoration Ablation, ảnh original/bicubic/neural cùng crop,
text từng engine và warning. Chọn `rescued`, `regressed`, `unchanged`, `uncertain`
rồi **Tải nhãn JSON của trang**; nút đọc lại chỉ nhận đúng evidence identity của
trang. Nhãn không tự ghi xuống ổ đĩa khi đóng trang. Điền `MANUAL_LABELS_FILE`
để notebook đọc/tổng hợp nhãn không inference. Mẫu số là số cặp đã gán nhãn,
bao gồm `uncertain`; sampling và số cặp xuất được báo riêng. Không suy thành
accuracy toàn corpus. `blind_review=True` giấu tên phương pháp trong mục H10;
cần tránh mở các mục raw/stability nếu muốn đánh giá mù.

H04–H09 vẫn theo roadmap trong [hypotheses](hypotheses.md); không có stub được
trình bày như một phương pháp đã hoạt động.

## Phạm vi thay đổi

| Nhóm | File |
|---|---|
| Tài liệu | `docs/README.md`, `problem.md`, `pipeline.md`, `hypotheses.md`, `usage.md`; `README.md`, `FLOW.md`, `PROBLEM.md` tại root. |
| Notebook | `1_chuan_bi_anh_ocr.ipynb`, `3_so_sanh_so_chu_ocr.ipynb`; giữ notebook 0 và 2. |
| Module mới | `src/vtm_corpus/restoration_evaluation.py`, `restoration_status.py`. |
| Module được sửa | Trong `src/vtm_corpus/`: `image_dataset.py`, `prepared_variants.py`, `config.py`, `corpus_ocr.py`, `image_preparation.py`, `ocr_records.py`, `multi_compare.py`, `evidence_pipeline.py`, `evidence_report.py`, `book_output.py`. |
| Bảo trì notebook | Trong `tools/`: `notebook_source_builder.py`, `notebook_ocr_workflow.py`, `rebuild_plain_ocr_notebook.py`; cho phép theo dõi cả `update_kaggle_notebook_bootstrap.py`, `install_kaggle_packages.py` để builder không thiếu dependency trong checkout sạch. |
| Kiểm thử và Git | `.gitignore`; trong `tests/`: thêm `test_restoration_evaluation.py`, cập nhật `test_image_dataset.py`, `test_character_consensus.py`, `test_notebook_bootstrap.py`, cho phép theo dõi helper `test_notebook_sources.py`. |
