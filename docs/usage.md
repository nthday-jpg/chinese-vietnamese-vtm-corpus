# Vận hành và kiểm chứng

## Toàn bộ phương pháp đã tích hợp — 13/11/2026

Notebook 1 mặc định `VARIANT_MODE='all_eligible'`, chạy độc lập mọi CPU/neural
đã tích hợp từ original. `VARIANTS` chỉ dùng khi chọn `selected`. Phạm vi dữ
liệu vẫn mặc định một sách/một trang; không tự downscale, deskew hoặc dewarp.

Đã bổ sung `docentr_binarization`, `degan_binarization`, `degan_deblurring`,
`degan_watermark_removal`; source/checkpoint có revision và SHA-256:

```text
kaggle_upload/models/image_restoration/docentr/model_8_2018_base.pt
kaggle_upload/models/image_restoration/degan/degan_weights.bundle
kaggle_upload/packages/image_restoration/sources/docentr/
kaggle_upload/packages/image_restoration/sources/degan/
kaggle_upload/packages/image_restoration/wheels/
kaggle_upload/packages/image_restoration/manifests/learned_documents_runtime.json
```

DE-GAN `.bundle` là ZIP chứa ba weights HDF5 gốc, giữ một bản duy nhất. Worker
kiểm hash và giải nén riêng task vào thư mục tạm writable. Keras 3 dùng backend
Torch trong worker riêng; không cài TensorFlow 1.x hoặc hạ Torch/NumPy. DocEnTR
dùng BinModel/vit-pytorch 0.24.3, base/patch 8. Cả hai dùng patch 256 không resize/
rotate; bỏ padding về kích thước nguồn. Chưa xác minh inference GPU trên Kaggle.

Watermark removal cần `REVIEWED_REGIONS['1:1:column_000'] =
{'watermark_removal_approved': True}` vì có thể xóa mực đỏ thật. Chưa duyệt thì
ghi `not_applicable`. DiffTSR tiếp tục chỉ chọn tường minh theo điều kiện dòng.

Script tự tìm root, stage đúng chỗ, resume `.part` và không tải lại file hợp lệ:

```powershell
.venv/Scripts/python.exe -X utf8 tools/prepare_learned_restoration_assets.py --download
```

Nếu chưa có kernel snapshot, thêm `--environment-report <JSON từ Notebook 0>`.
Nhóm tổng >5 GB hoặc ước tính >5 phút được hoãn để bạn tự chạy:

```powershell
.venv/Scripts/python.exe -X utf8 tools/prepare_learned_restoration_assets.py --download --include-large
.venv/Scripts/python.exe -X utf8 tools/download_large_restoration_assets.py --all
.venv/Scripts/python.exe -X utf8 tools/download_unihir_checkpoint.py --download --accept-unihir-noncommercial --connections 4
```

Hai checkpoint mới và ba wheel Linux cp313 đã đủ trong workspace. Upload lại
**models và packages hiện tại**, không dùng ZIP upload cũ. Kernel dependency
được tái dùng theo inventory đã đo; khác ABI/version phải chuẩn bị lại, không
fallback tải mạng. Notebook 3 đã có reader; thêm các tên mới vào `OCR_VARIANTS`
để đọc ảnh đã lưu, không chạy lại restoration.

Để gửi cấu hình phiên Kaggle mới, chạy Notebook 0 với models/packages hiện tại,
Internet Off và `RESTORATION_GPU_CHECK='off'` (mặc định mới). Kiểm môi trường toàn
diện với `quick_restoration_checks=False`, `check_existing_runtimes=True`, ngân
sách tối đa 900 giây; không build hoặc nạp weights. Có inventory mọi distribution,
dependency, wheel METADATA/tags, nguồn/license, CUDA/native API nhỏ và đối chiếu
kernel/overlay với từng runtime plan. Gửi lại **`kaggle_environment_logs.zip`**
trong Output, kể cả báo cáo lỗi hoặc partial. Có thể chọn chế độ probe nhanh
qua `quick_restoration_checks=True` nếu chỉ cần restoration.

LWTformer/GLADUnet chưa có checkpoint/inference upstream được xác minh; Gatos/
energy và bleed-through chuyên biệt chưa có implementation đã pin. Notebook
công khai `not_integrated_methods`; không có adapter giả cho các tên này.

## Chạy nhiều cơ chế sau khảo sát dữ liệu thật

[Báo cáo khả thi](../reports/restoration_feasibility.md) mô tả 13 trang nguồn
độc lập, các profile phương pháp và giới hạn. Các phương pháp mới chỉ dùng
NumPy/OpenCV/Pillow đã có trong payload; không thêm checkpoint hoặc cài Torch.

Trong ô cấu hình Notebook 1, giữ giới hạn smoke hiện có rồi chọn:

```python
VARIANT_MODE = 'selected'  # hoặc 'classical' để chạy mọi phương pháp CPU
VARIANTS = ['original', 'clahe', 'bicubic_x2', 'bicubic_x4',
            'retinex_msr', 'retinex_msrcp', 'wolf', 'nick',
            'color_pca', 'color_gmm', 'mean_shift', 'anisotropic_diffusion',
            'background_morphological', 'background_guided', 'background_otsu']
APPLICABILITY = {'color_gmm': {'books': ['1', '2', '4', '6']}}
# Ví dụ tham số; không phải giá trị tốt nhất đã xác minh:
VARIANT_OPTIONS['anisotropic_diffusion'] = {'iterations': 5, 'kappa': 12., 'timestep': .15}
```

Chạy lại ô cấu hình sau khi sửa để cập nhật `CONFIG`. Mỗi variant nhận cùng
original, không nối chuỗi mặc định. `background_otsu` là hypothesis kết hợp có
tên và provenance riêng. `APPLICABILITY` nhận `books`, `crop_types`,
`require_review`; khi yêu cầu review, thêm `approved_variants` vào
`REVIEWED_REGIONS['book:page:region']`. `excluded_variants` vẫn dùng được.
Tên có/không `.pdf` đều được nhận. Đây là điều kiện thủ công, không phải detector.

Plan in ra và lưu `_dataset/last_plan.json`: số region/ảnh, scaled pixels,
estimate storage trước cache/dedup, giới hạn pixel. Estimate có thể dư khi ảnh
trùng; actual byte cap vẫn được kiểm ở từng lần ghi. Một số method có cap riêng:
Mean Shift 4 triệu, GMM/diffusion 8 triệu, Retinex/guided 12 triệu pixel input;
vượt thì chọn region nhỏ hơn tường minh, không tự giảm ảnh. `ALLOW_PARTIAL=True`
lưu trạng thái lỗi/thiếu có giải thích; không được xem partial là đủ phương pháp.

Để tránh OCR lặp nguồn, với bộ PDF hiện tại có thể chọn `BOOKS=['1','2','4','6',
'bien','kinh']`, vì `3` trùng `2`, `5` trùng `4`. Không xóa file nguồn. Nếu chọn
cả alias, reader kiểm kết quả giống nhau; nếu applicability tạo kết quả khác
nhau trên cùng source thì báo xung đột thay vì chọn ngầm một bản.

Trong Notebook 3, bật `ENABLE_MULTI_VARIANT_OCR`, trỏ `PREPARED_IMAGES` tới
`_dataset/dataset.json` hoặc immutable snapshot, đặt `PREPARED_MISSING_FALLBACK=False`
và chọn `OCR_VARIANTS` chỉ gồm ảnh đã sinh. Chọn cùng `PREPARED_CROP_TYPE` cho mọi
engine. Bật `ENABLE_RESTORATION_ABLATION` để xem Utility A và
`ENABLE_VISUAL_FIDELITY` nếu cần chẩn đoán pixel; stability B tách sáu cơ chế,
từng engine, và canonical region. `summary.exact_agreement_rate` tổng là `null`
có chủ đích; đọc `groups`, `region_groups`, `engine_sensitivity_profiles`,
`joint_changes`, `consensus_stability_axes`, và `stability_summary.by_book_group`.
G1/G2/G3 được tra bằng SHA đã xem, file mới giữ `unclassified`; có thể khai báo
`ADVANCED['book_groups']={'my_book.pdf':'G1'}` sau khi khảo sát.

Các projection Lab a/b, red-only, PCA/GMM được đánh dấu mất thông tin có chủ đích;
không đọc tỷ lệ của chúng như invariance bảo toàn nội dung. `uncertain`, thiếu
processor/config/geometry và alignment mơ hồ cần được giữ. Bốn engine không
thành 4×số variant phiếu. Khi chỉ đánh giá lại, dùng `RUN_OCR=False` và manifest
multi-variant cũ; không nạp OCR hoặc restoration. Khi thêm variant, scheduler
chỉ làm các tác vụ chưa có checkpoint hợp lệ với cùng source/engine/config.

Source scripts và tests liên quan nay có exception `.gitignore`; weights, PDFs,
preview PNG không vào Git. Muốn dựng lại source notebook sau sửa module, chạy
`tools/update_restoration_notebook.py`, rồi `tools/compact_image_notebook.py --check`;
đây chỉ là build/static, không Run All.

Audit ngắn không tải/cài/chạy model:

```powershell
.venv/Scripts/python.exe -X utf8 tools/audit_restoration_assets_quick.py
```

Kiểm mọi file được liệt kê có mặt/đúng size, hash tối đa 256 MiB; file lớn dùng
digest manifest cũ, không gọi là hash mới. Full checksum tùy chọn bằng verifier
hiện có ở phần dưới. Nếu thiếu tài sản nhỏ, dùng preparation script; nhóm >5 GB
hoặc ước tính >5 phút dùng dedicated downloader, tự stage đúng `kaggle_upload/`.
Khi chuẩn bị runtime mới, truyền `--environment-report <file JSON từ Notebook 0>`
để khóa phiên bản Linux/ABI theo Kaggle thật, không giả định từ máy Windows.
Không chạy kiểm tra workflow local như bằng chứng Kaggle; chỉ kiểm static/CPU
ngắn. Chưa xác nhận runtime-ready hay inference GPU cho các model trong đợt này.

## Tài sản restoration mở rộng — bàn giao ngày 12/11/2026

Implementation đã có trong Notebook 1 và reader Notebook 3. Chưa chạy notebook,
inference thật hoặc upload Kaggle. Bảng dưới phân biệt mã tích hợp với asset
thực tế; “offline” luôn có điều kiện runtime Linux cp313 phù hợp báo cáo môi
trường và probe target thành công.

Sau khi người dùng chạy scripts, đã kiểm lại: cả sáu nhóm trong catalog đều đủ
và qua checksum, không có dependency metadata lỗi; verifier trả mã 0. UniHIR
được kiểm riêng: 18 file model/font (29.772.403.293 byte), 27 file source, bốn
wheel overlay và source FlashAttention có CUTLASS đều đạt. Báo cáo nằm trong
`manifests/restoration_verification.json` và `manifests/unihir_current_verification.json`.
Native FlashAttention build, GPU inference và Kaggle runtime vẫn chưa kiểm.

Sửa phân loại UniHIR: việc gộp nó với DiffTSR rồi bỏ khỏi `all_eligible` là quá
rộng. UniHIR phục hồi toàn trang; prior inscription, rủi ro sinh nét và yêu cầu
tài nguyên không phải ràng buộc một dòng ngang. Đã đưa `unihir_restoration` trở
lại `all_eligible`, giữ preflight và nhãn generative; không đổi `VARIANTS` mặc
định. Notebook 3 vẫn cần `ALLOW_GENERATED_VARIANTS=True` khi so ảnh UniHIR.
Kiểm tra sau sửa: 29/29 test CPU/mock đạt, gồm UniHIR trên mọi crop, preflight
thiếu asset, publish/reader/resume và loại DiffTSR khỏi tự chọn; log tại
`work/restoration_unihir_followup_tests.log`. Không inference model thật.

| Phương pháp / variant | Phạm vi | Checkpoint | Asset hiện tại / offline |
|---|---|---|---|
| Giữ `original`, `clahe`, `denoised`, `bicubic_x2`, `bicubic_x4` | Dọc/ngang/nhiều cột | Không | CPU; không tải model |
| `grayscale_luma`, `grayscale_red`, `grayscale_green`, `grayscale_blue` | Chọn tín hiệu sáng/kênh màu | Không | CPU; mỗi kênh là thí nghiệm riêng, có thể làm mờ mực màu |
| `lab_l`, `lab_a`, `lab_b` | Luminance/chroma; a/b là ảnh phân tích màu | Không | CPU; giữ original RGB |
| `red_ink` | Phân tích riêng mực đỏ, giữ màu đỏ | Không | CPU; cố ý chỉ giữ ứng viên đỏ trong variant này, không thay original |
| `background_normalized` | Gaussian nền + sửa Lab L, giữ chroma | Không | CPU; sigma/strength/floor riêng |
| `sauvola`, `niblack` | Ngưỡng cục bộ độc lập hướng viết | Không | CPU; window_size/k, Sauvola thêm r; cảnh báo xóa/nối nét |
| `bilateral`, `unsharp` | Khử nhiễu/cạnh nhẹ trên Lab L | Không | CPU; tham số bảo thủ, giữ chroma |
| `realesrgan_x4` | Trang/vùng ×4 | `RealESRGAN_x4plus.pth` | Có sẵn, SHA-256 đạt; không tải lại |
| `tadisr_x4` (+ mask chẩn đoán) | Trang/vùng, tile hai trục | Adapter + đầy đủ CogView4 shard/config/tokenizer | Có sẵn 31,914 GB, SHA-256 đạt; prompt cố định được tạo offline trong worker nếu thiếu |
| `docres_appearance` | Appearance trên trang/vùng | `docres.pkl` | Đã tải, SHA-256 upstream đạt; runtime metadata đủ |
| `docres_deblurring` | Deblur trên trang/vùng | `docres.pkl` | Dùng cùng checkpoint, task/prompt/cache riêng |
| `docres_deshadowing` | Deshadow trên trang/vùng | `docres.pkl` | Task độc lập; không nối chuỗi |
| `docres_binarization` | Binarization trên trang/vùng | `docres.pkl` | Cảnh báo mất màu/nét; không tự thay original |
| `docdiff_deblurring` | Document deblur/enhancement | `init.pth` + `denoiser.pth`, schedule linear 100 | Đã tải cả hai, đối chiếu Git blob + SHA-256; không dùng seal-removal weights |
| `restormer_motion_deblurring` | Motion blur | `motion_deblurring.pth` | Người dùng đã tải; checksum/metadata đạt |
| `restormer_defocus_deblurring` | Single-image defocus | `single_image_defocus_deblurring.pth` | Đã tải/kiểm; không yêu cầu ảnh dual-pixel |
| `restormer_real_denoising` | Real noise | `real_denoising.pth` | Đã tải/kiểm; normalization đúng checkpoint |
| `restormer_gaussian_denoising` | Blind Gaussian color noise | `gaussian_color_denoising_blind.pth` | Đã tải/kiểm; không mô tả là real denoising |
| `unihir_restoration` | Phục hồi sinh nội dung toàn trang/vùng, có trong `all_eligible` | `model.safetensors`, VAE/config/tokenizer/font | Đủ và kiểm hash; native build và GPU chưa xác minh |

Model tổng quát nhận được bố cục chữ dọc/nhiều cột, chưa chứng minh hiệu quả
trên Hán cổ. DocRes giữ nguyên định nghĩa DTSPrompt upstream, bỏ import MBD không
được dùng ở bốn task. Adapter dùng padding và tile, không ép ảnh thành vuông như
nhánh appearance/deshadow kích thước lớn của upstream; ghi khác biệt này trong
diagnostics. DocDiff dùng cả coarse predictor/residual sampler. Bỏ toàn bộ padding
trước khi lưu; output cùng canvas nguồn, không xoay/reflow chữ.

Lệnh thực tế trên Windows, chạy từ repo hoặc dùng đường dẫn tuyệt đối tới script:

```powershell
.venv/Scripts/python.exe -X utf8 tools/prepare_restoration_assets.py
.venv/Scripts/python.exe -X utf8 tools/download_large_restoration_assets.py --list --all
.venv/Scripts/python.exe -X utf8 tools/download_large_restoration_assets.py --all
.venv/Scripts/python.exe -X utf8 tools/verify_restoration_assets.py --all
```

Linux dùng `python` trong môi trường có `packaging`; không cài lại Torch vào
kernel đang hoạt động. `--model restormer`, `--model docres_mbd`, `--model tadisr`
chọn riêng nhóm; có thể lặp `--model`. Tải tiếp mặc định, không cần cờ resume.
`--all` ở script lớn lấy toàn bộ nhóm trong catalog, gồm MBD phụ trợ; không lấy
DiffTSR bị loại khỏi tự động; UniHIR dùng script tải riêng và asset hiện có.
`--verify` chỉ kiểm tra/báo cáo. Thiếu asset
hoặc dependency trả mã 2; không diễn giải mã này thành inference thất bại.

| Nhóm lớn/chậm | Dung lượng đã xác minh từ metadata nguồn | Đích trong `kaggle_upload/` | Nguồn / license / trạng thái |
|---|---|---|---|
| TADiSR adapter + CogView4 | 31.914.476.368 byte | `models/image_restoration/tadisr/` | HF upstream pin trong catalog; MIT source, adapter research use, CogView4 Apache-2.0; đã có đủ local |
| Restormer, 4 checkpoint | 418.624.772 byte (~419 MB); cả source/config 418.700.756 byte | `models/image_restoration/restormer/` | [Official v1.0 release](https://github.com/swz30/Restormer/releases/tag/v1.0); MIT; người dùng đã tải, checksum đạt |
| DocRes MBD phụ trợ | 712.981.585 byte (~713 MB) | `models/image_restoration/docres/mbd.pkl` | [Space được README chính thức liên kết](https://huggingface.co/spaces/qubvel-hf/documents-restoration); SHA-256 LFS đạt; đã tải, chỉ cần nếu nghiên cứu dewarp sau này |

DocRes chính 183.260.711 byte; DocDiff hai checkpoint 32.915.186 byte đã tải mới.
Không có GPU/native build hoặc tải nhóm lớn trong đợt này. Restormer release
không công bố SHA-256; lần tải đầu kiểm URL release/size, khóa SHA-256 vào receipt
và mọi lượt kiểm sau dùng đúng hash đó. Phân biệt checksum publisher (DocRes,
DocDiff/Git, TADiSR) với checksum lần tải đầu, không tự rebaseline file đã hỏng.

Catalog pin nguồn/revision ở [restoration_assets_catalog.json](../tools/restoration_assets_catalog.json).
Verifier xuất `packages/image_restoration/manifests/restoration_verification.json`
và `.txt`: download, integrity, runtime, thiếu file, lỗi dependency/license và
trạng thái Kaggle/GPU riêng. `document_runtime.json` hiện có một PDFium wheel và
reuse dependency closure từ `kernel_snapshot.json`; TADiSR có overlay riêng.
Wheels đã kiểm metadata/ABI Linux cp313, hash và dependency closure; chưa cài/probe
trên Kaggle. Khi kernel đổi, dùng `--environment-report <JSON notebook 0 mới>`;
không dùng báo cáo Windows hoặc hạ Torch/NumPy để ép qua.

Các file chính thay đổi: code `restoration_classical.py`, `document_restoration.py`,
`image_dataset.py`, `prepared_variants.py`, `config.py`; hai notebook 1/3; tools
prepare/download/verify/catalog/runtime và `update_restoration_notebook.py`;
năm file docs hiện có. Notebook 2, raw OCR checkpoints và canonical nguồn không đổi.
Builder cập nhật source cells, giữ ô cấu hình hiện có của Notebook 3. Notebook 1
vẫn nén lossless upstream cũ và dưới 1 MB.

Cấu trúc thực tế sau khi kiểm lại lượt tải của người dùng:

```text
kaggle_upload/
  models/                         # Dataset models
    image_restoration/
      realesrgan/, tadisr/         # đã có và kiểm hash
      docres/docres.pkl            # đã tải
      docres/mbd.pkl               # đã tải/kiểm; MBD phụ trợ, không bật dewarp
      docdiff/init.pth, denoiser.pth
      restormer/                  # đủ 4 checkpoint và assets.json
      unihir/                     # đủ model/font; ứng viên all_eligible
      difftsr/                    # tài sản cũ, không chọn/tải tự động
  packages/                       # Dataset packages
    image_restoration/
      sources/docres/, docdiff/, restormer/  # source/config/license đã pin
      sources/TADiSR/, BasicSR/, Real-ESRGAN/
      wheels/                     # overlay + PDFium Linux cp313
      requirements/document.txt, tadisr.txt
      runtime/                    # helper versioned, giữ helper legacy nguyên byte
      manifests/                  # catalog, receipts, source/runtime inventory
```

Runtime dùng trực tiếp các thư mục source đã giải nén. Script chuẩn bị không
tạo thêm ZIP trùng của DocRes/DocDiff/Restormer. Để dọn các ZIP từ lần chuẩn bị
cũ, chạy `./tools/remove_redundant_source_zips.ps1`; script kiểm hash source và
đối chiếu toàn bộ nội dung ZIP trước khi xoá. Giữ wheel `.whl` và source
`flash_attn-*.tar.gz` vì chúng cần cho cài đặt/build offline.

Tài sản local đã xác minh không đồng nghĩa Dataset trên Kaggle đã đầy đủ. ZIP
models/packages cũ chưa cập nhật tự động. Upload nguyên hai thư mục gốc sau khi
script lớn + verifier đạt theo các model bạn chọn. MBD thiếu không cản bốn task
DocRes; nếu không cần dewarp, dùng verifier mặc định thay `--all`.

Kiểm chứng phần mềm của đợt này: bộ hồi quy **417 test: OK, 1 skipped**
(`work/restoration_regression_tests.log`); bộ kiểm thử mở rộng cuối cùng **24/24
đạt** (`work/restoration_extension_tests.log`). Có fixture CPU/mock cho geometry,
ảnh dọc/mực đỏ, checkpoint loading contracts, resume/cache, chuyển root, tải
tiếp/HTTP range/hash và script chạy từ cwd khác. Notebook JSON/Python syntax,
checksum payload upstream, giới hạn 1 MB và builder chạy lại byte-identical đạt.
Không có Run All, GPU inference, build native nặng hoặc upload Kaggle. Kết quả
mock không xác nhận model thật hoạt động hoặc tăng accuracy OCR.

Các script trong `tools/` tự tìm repository root, đặt model vào
`kaggle_upload/models/image_restoration/`, source/wheels/requirements/manifests
vào `kaggle_upload/packages/image_restoration/`. Hai gốc `models/` và `packages/`
được upload thành hai Dataset riêng. Không dùng cache Hugging Face ngoài staging
làm vị trí cuối. Giữ file tải dở dưới `.part`, chỉ công bố sau kiểm checksum.

Chuẩn bị nhẹ và tải lớn là hai bước riêng. Nhóm trên 5 GB hoặc ước tính quá
5 phút (mặc định 1 MB/s), nhóm chưa biết kích thước và native build dài được
hoãn. Người dùng tự chạy script tải lớn. Verifier phải báo thiếu/corrupt/license
blocker và không nâng trạng thái thành GPU verified chỉ vì đủ file.

Với bố cục cột lớn kèm cột nhỏ: giữ `page` hoặc `group` cho phục hồi tổng quát;
xem preview rồi mới chọn `column`. Không cần biết ranh giới từng chữ. DiffTSR
không còn trong tự chọn/tải; không xoay cả cột. UniHIR có trong `all_eligible`.
Không bật dewarp/end2end DocRes vì không có mapping phi tuyến.

Hướng dẫn đối chiếu mã ngày **12/11/2026**. Bắt đầu bằng một sách/một trang;
chỉ mở rộng sau khi xem ảnh, đường cắt, output và trạng thái lỗi. Các thuật ngữ
và hợp đồng tọa độ nằm trong [pipeline](pipeline.md); mục tiêu đánh giá nằm trong
[bài toán](problem.md). Không cần inference mới khi chỉ đọc checkpoint.

## Thứ tự notebook

| Notebook | Cách dùng |
|---|---|
| [0 — môi trường](../0_kiem_tra_moi_truong_kaggle_offline.ipynb) | Inventory + chế độ Kaggle-only build/probe FlashAttention và UniHIR thật trên một trang; xuất ZIP log kể cả khi lỗi. Không tải mạng, không OCR; local không build/inference. |
| [1 — chuẩn bị ảnh](../1_chuan_bi_anh_ocr.ipynb) | Phiên riêng: PDF → canonical, biến thể, manifest, QC (kiểm tra chất lượng) và preview; chuyển output thành Dataset. Không bắt buộc cho OCR PDF trực tiếp. |
| [3 — OCR và rà soát](../3_so_sanh_so_chu_ocr.ipynb) | Chạy từng engine, lưu checkpoint, so sánh và xuất báo cáo offline. |
| [2 — dịch](../2_dich_han_viet.ipynb) | Dịch TXT Hán đã chọn; kiểm tra `INPUT_DIR` vì local mặc định tìm run OCR mới nhất, chưa chắc là bản đã duyệt. Kaggle vẫn cần mạng để clone/cài repo và tải model. |

## Môi trường và tài sản offline

### Chạy Notebook 0 và gửi log GPU

Upload bản Notebook 0 mới, gắn models/packages/PDF và giữ Internet Off. Điền
`PACKAGES_DIR`, `MODELS_DIR`, `DATA_DIR` (thư mục trực tiếp chứa PDF), rồi Run All.
Mặc định là `RESTORATION_GPU_CHECK='off'`: kiểm kê và probe môi trường toàn diện,
ngân sách tối đa 900 giây, không build/inference. Nếu chủ động bật `'unihir_smoke'`: một PDF, một
trang, crop page, `steps=10`, `ref_count=1`, không dùng cache ảnh restoration.
`books=[]` chọn PDF đầu; `pages='1'` có thể đổi trang nhưng tổng vẫn giới hạn một.

`'off'` chỉ inventory; `'build_and_probe'` chỉ build wheel và chạy cả fixed/varlen
FlashAttention, kiểm finite và đối chiếu SDPA, không nạp weights. GPU smoke UniHIR
kiểm GPU đầu >=75 GiB trước khi đọc weights/build. Kernel Python/ABI và package
versions phải khớp plan; lỗi khớp môi trường được lưu để điều chỉnh bundle.
Overlay kế thừa Torch hiện có, không hạ package kernel. Cache native hợp lệ được
dùng lại với wheel checksum và CUDA probe mới; inference một trang luôn mới.

Mỗi lần kiểm có phase report riêng: hardware, asset integrity, runtime build,
FlashAttention CUDA, worker preflight, model load, inference, output validation.
`native_build_verified`, `flash_attention_cuda_verified`, `model_inference_verified`
được ghi độc lập. Import thành công không được nâng thành inference thành công.
Compile hoặc inference lỗi/timeout vẫn giữ traceback/log; supervisor dừng cả
nhóm tiến trình trên Linux. Tổng timeout mặc định 5400 giây, chỉnh được trong ô
cấu hình; phase load/inference có timeout riêng. Đây không phải ước tính thời gian chạy.

Sau cùng gửi file `restoration_check_*_logs.zip` trong Output: gồm JSON/TXT,
build log, receipt, worker log và hai ảnh cùng trang nếu thành công. ZIP không
chứa weights hoặc venv. Chỉ inventory thì tên `kaggle_environment_logs.zip`.
Nếu tự ngắt trước ô cuối, phase JSON và log vẫn còn trong thư mục
`restoration_check_*`; chạy ô lưu cuối để gom ZIP.

Kiểm local bổ sung: 53/53 file model OCR và 74 artifact package/source OCR
(~3,93 GiB) qua hash/source-tree audit; sáu nhóm catalog restoration và UniHIR
cũng đủ, hash và wheel metadata đạt. Báo cáo:
`packages/image_restoration/manifests/full_local_inventory_verification.json`
và `notebook0_preparation_audit.json`. Điều này chưa xác minh runtime/native ops
hoặc inference Kaggle cho toàn bộ OCR/restoration.

### Trạng thái asset DiffTSR

DiffTSR đã bị loại khỏi `all_eligible`; weights đã retired khỏi staging hiện
dùng. Notebook 0 ghi `excluded_by_configuration`, không coi đây là tài nguyên
đủ để inference. Adapter legacy, line gate và reader kết quả cũ vẫn được giữ.
Nếu một bản staging cũ còn asset riêng của DiffTSR, script dọn có kiểm phạm vi
đường dẫn và tham chiếu dependency dùng chung:

```powershell
.\tools\remove_difftsr_assets.ps1 -WhatIf
.\tools\remove_difftsr_assets.ps1
```

Script không chạm `denoised`, Real-ESRGAN, UniHIR, TADiSR, DocRes, DocDiff,
Restormer hoặc source `zhconv` của OCR. Nó xác minh đích vắng mặt sau xoá và ghi
`manifests/difftsr_retired.json`. ZIP upload cũ vẫn giữ nội dung cũ; upload các
thư mục staging hiện tại để dùng bộ đã thay đổi.

### Môi trường và cấu trúc upload

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
| `unihir_restoration` | Có trong `all_eligible`; CC BY-NC-ND 4.0, nghiên cứu phi thương mại. Chọn được page/group/column, chưa chứng minh phù hợp Hán cổ. Canvas danh nghĩa không xác nhận glyph nguồn; quyền phân phối lại font chưa xác lập chỉ từ metadata tác giả. |

Notebook 1 dùng `REVIEWED_REGIONS` để khai báo điều kiện sau khi xem original;
Công cụ tải UniHIR cũ giữ điều kiện phi thương mại. Notebook 3 chỉ nhận DiffTSR/UniHIR khi
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
packager core hiện đóng đủ extension restoration. Chuyển dataset notebook 1 phải
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

Riêng đợt H10 trước đợt ảnh mở rộng không chạy notebook/Run All, inference thật,
tải weights, cài môi trường GPU hoặc upload Kaggle. Kiểm thử worker giả chỉ xác minh
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
đổi này. Đợt ảnh mở rộng đã thêm source, weights DocRes/DocDiff và
runtime metadata; chưa upload hoặc dựng lại ZIP models/packages lớn.

## Chuyển dataset giữa hai phiên Kaggle

Notebook 1 và 3 độc lập phiên. Sau khi QC, tải toàn bộ cây dataset hoặc ZIP đã
kiểm tra rồi tự upload thành Kaggle Dataset; gắn vào Notebook 3 ở một phiên mới.
Đặt `PREPARED_IMAGES` tới `dataset.json` hoặc immutable snapshot dưới Input,
`PREPARED_MISSING_FALLBACK=False` và chọn cùng `PREPARED_CROP_TYPE`.
PDF gốc vẫn cần trong `DATA_DIR` để lập tác vụ và xác minh SHA-256.

Trong Notebook 1, `VARIANT_MODE='selected'` giữ danh sách `VARIANTS` (mặc định
original/CLAHE/×2/×4). `classical` chọn 18 phương pháp cổ điển;
`all_eligible` chọn thêm 11 neural variant tổng quát và UniHIR generative
(TADiSR kèm mask chẩn đoán).
Điều kiện vùng, asset, runtime và
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
