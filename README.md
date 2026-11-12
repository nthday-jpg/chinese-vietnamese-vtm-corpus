# OCR sách Hán lịch sử và dịch tiếng Việt

PDF scan → ảnh nguồn có provenance → bốn OCR → rà soát bằng chứng → TXT Hán đã
chọn → workflow dịch Việt. Không có ground truth: đồng thuận, ổn định và overlap
ảnh đều là chẩn đoán, không phải độ chính xác OCR.

**[Tài liệu pipeline và vận hành đầy đủ bằng tiếng Việt](docs/README.md)**:
tài sản Kaggle offline, dataset canonical/restoration, checkpoint, cấu hình,
coverage, stability, giấy phép, kiểm chứng và xử lý sự cố.

| Notebook | Trách nhiệm |
|---|---|
| [0 — kiểm tra môi trường](0_kiem_tra_moi_truong_kaggle_offline.ipynb) | Kiểm kê runtime/package/ABI/GPU, không cài hoặc OCR. |
| [1 — chuẩn bị ảnh](1_chuan_bi_anh_ocr.ipynb) | Workflow riêng: canonical page/group/column, biến thể, manifest và QC. |
| [3 — OCR và rà soát](3_so_sanh_so_chu_ocr.ipynb) | TongGuOCR, PP-OCRv6-medium, PaddleOCR-VL-1.6, HunyuanOCR-1.5; đồng thuận và báo cáo offline. |
| [2 — dịch Hán–Việt](2_dich_han_viet.ipynb) | Dịch TXT đã rà soát; workflow hiện tại vẫn cần chuẩn bị repo/model qua mạng. |

Notebook 1 nén dữ liệu mã upstream bằng zlib/Base85 để giữ file dưới 1 MB.
Ô mã upstream tự giải nén và kiểm SHA-256, giữ nguyên từng file và license;
chạy offline không cần thêm Dataset mã nguồn. Sau khi cập nhật phần mã nhúng,
chạy `uv run --no-project python tools/compact_image_notebook.py`;
thêm `--check` để kiểm dung lượng và checksum mà không ghi file.

Trong notebook 3, điền `PACKAGES_DIR`, `MODELS_DIR`, `DATA_DIR`. Khởi đầu bằng
`BOOKS=["1"]`, `MAX_BOOKS=1`, `PAGES="1"`, `LIMIT=1`. Hai tính năng mới mặc định tắt:

```python
ENABLE_MULTI_VARIANT_OCR = False
ENABLE_SPATIAL_COVERAGE = False
```

Multi-variant tiêu thụ `PREPARED_IMAGES` từ phiên chuẩn bị trước; không chạy model
restoration trong notebook 3. `RUN_OCR=False` đọc checkpoint mà không tải model.
Kết quả HTML/JSONL/crop/ZIP ở `WORK_DIR/output/`. Giữ source và checkpoint để resume.

Local Windows dùng `.venv` gốc, Python 3.13. Mã CLI ở `src/vtm_corpus/`; notebook
3 nhúng hàm và worker tự chứa để chạy Kaggle không cần checkout. Cập nhật mã nhúng:

```powershell
.venv/Scripts/python.exe -X utf8 tools/update_kaggle_notebook_bootstrap.py
.venv/Scripts/python.exe -X utf8 -m unittest discover -s tests -q
```

Đã kiểm thử CPU, worker giả, đọc restoration thật đã lưu và báo cáo offline.
Chưa xác nhận inference multi-variant bốn model trên GPU/Kaggle hoặc QA tương tác
trong browser. [Sơ đồ ngắn](FLOW.md) · [Bài toán gốc](PROBLEM.md).
