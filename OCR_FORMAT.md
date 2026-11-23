# Định dạng OCR dùng cho voting

Đã chuẩn hóa dữ liệu vào **`data/ocr_raw/`**, không ghi đè bản gốc.

```text
ocr_raw/<book>/<engine>_shard0.jsonl
```

Mỗi dòng là một trang: `{"page": 0, "text": "chữ OCR"}`; `page` đếm từ **0**.
Metadata bổ sung của CLC được giữ nguyên. Các file `data/*.json` gốc là nhiều object
JSON nối tiếp, không phải một JSON hợp lệ hay JSONL; không đưa trực tiếp vào notebook.

## Ánh xạ

- `kinh` → `HDTTK`; `bien` → `ONDK`; `1`…`6` → `VSYC_1`…`VSYC_6`.
- `paddle_vl.json` → engine `paddle`; các engine khác: `hunyuan`, `ppocr`, `tonggu`, `clc`.
- Hunyuan, Paddle, PPOCR và CLC: mỗi engine có 1.133 trang/8 sách.
- `tonggu.json`: 711 trang/7 sách, thiếu ONDK. Không tự lấy kết quả từ phiên
  `ocr_tonggu_stage3_*` để bù vì đó là kết quả khác; không tính chúng như model độc lập.
- Tổng cộng 39 shard, ONDK có 4 model, các sách khác có 5 model trước lọc trang hỏng.

## Tạo lại dữ liệu

Chạy từ thư mục gốc repository:

```bash
python scripts/convert_model_json_to_ocr_raw.py data/hunyuan.json --engine hunyuan --copy-clc
python scripts/convert_model_json_to_ocr_raw.py data/paddle_vl.json --engine paddle
python scripts/convert_model_json_to_ocr_raw.py data/ppocr.json --engine ppocr
python scripts/convert_model_json_to_ocr_raw.py data/tonggu.json --engine tonggu
```

## Chạy notebook trên Kaggle

1. Tạo Dataset chứa **chỉ thư mục `data/ocr_raw/` đã chuẩn hóa**, giữ tên `ocr_raw`.
   Không tải nguyên `data/`: notebook tìm mọi `ocr_raw` lồng bên trong, có thể trộn
   bản cũ với bản mới hoặc đếm Tonggu stage3 thành engine khác.
2. Add Input Dataset đó vào `1_voting_ocr.ipynb`; thêm Dataset golden nếu cần CER.
3. Run All. Notebook đã có tqdm cho đọc shard, voting, CER và tìm chỗ kiểm tay.
4. Lấy kết quả tại `/kaggle/working/ocr_raw/`, `txt/`, `metrics/`.

`GOLD_BOOK_MAP` trong notebook ánh xạ tên sách ngắn sang tên trong golden để tính
CER và tạo `kiem_tay.csv`; không sửa golden. Nếu đổi tên sách, cập nhật mapping này.

## Kiểm tra thực tế

Voting đã chạy thành công trên 1.133 trang. Sau lọc trống/lặp, có 2 trang chỉ còn
2 model hợp lệ (mỗi sách VSYC_4 và VSYC_5 một trang), dùng quy tắc ưu tiên thay vì đa số.
VSYC_2/VSYC_3 và VSYC_4/VSYC_5 có kết quả trùng nhau trong dữ liệu đầu vào;
cần kiểm tra nguồn PDF nếu dự kiến đây là các tập khác nhau. Không tự sửa nội dung OCR.

Lưu ý: `data/` đang bị Git ignore; các shard được tạo chỉ nằm ở máy hiện tại.
