# Chạy bổ sung Hunyuan local cho các trang chạm token

Dùng `.venv` tại gốc repo và model đã có ở `kaggle_upload/models/hunyuan`.
Script chỉ chọn `hit_max=True` trong lượt Hunyuan gốc: 54 trang nguồn duy nhất,
tương ứng 99 trang nếu tính cả các sách trùng 2/3 và 4/5. Mỗi nguồn chỉ inference
một lần, giữ danh sách sách/trang tương ứng trong kết quả.

## Cách chạy

Cấu hình đã thử thực tế trên RTX 4060 Laptop 8 GB là chia hai nửa phải → trái,
2.048 token mỗi nửa. Lệnh nhanh:

```powershell
# Chuẩn bị danh sách, không inference
.venv\Scripts\python.exe tools/retry_hunyuan_tokens_local.py --split-halves --max-new-tokens 2048

# Thử một trang chưa chạy
.\scripts\retry_hunyuan_tokens.cmd

# Chạy mọi trang còn lại, giữ model trong một worker
.\scripts\retry_hunyuan_tokens.cmd --all
```

Smoke sách 4 trang 1: 13,1 giây inference, không `hit_max`, không cờ chất lượng;
lượt thử toàn trang 8.192 token trước đó mất 252,4 giây và vẫn bị lặp/chạm token.
Vì vậy không dùng việc tăng trần token đơn thuần làm cấu hình chạy bù mặc định.
Các số này chỉ đo một trang, chưa phải benchmark hoặc xác nhận đủ/đúng chữ.

Chia nửa là thử nghiệm dành cho scan hai lá có gáy gần giữa, không phải dò cột.
Xem `input_crops/` trước khi dùng với trang nghiêng hoặc bố cục khác; thao tác cắt
có thể cắt qua chữ nếu gáy không nằm giữa. Giữ nguyên PNG nguồn, lưu cả hai crop,
tọa độ gốc, output riêng và thứ tự ghép. Không tự nhập kết quả này vào handoff.

Các lệnh Python phía dưới là công cụ chung để thử toàn trang với tham số riêng;
cấu hình toàn trang mặc định chưa sửa được lỗi lặp ở trang smoke trên.

Chuẩn bị danh sách, kiểm SHA-256 từng PNG, không inference:

```powershell
.venv\Scripts\python.exe -X utf8 tools/retry_hunyuan_tokens_local.py
```

Thử một trang chưa có kết quả (ưu tiên sách 4 trang 1):

```powershell
.venv\Scripts\python.exe -X utf8 tools/retry_hunyuan_tokens_local.py --run --limit 1
```

Sau khi xem kết quả thử, chạy toàn bộ trang còn lại:

```powershell
.venv\Scripts\python.exe -X utf8 tools/retry_hunyuan_tokens_local.py --run --all
```

Mặc định chạy tuần tự trên GPU, không tải model qua mạng. Giữ đúng PNG đã đưa
vào OCR gốc; không render lại PDF, phục dựng ảnh hoặc thay pixel.

## Cấu hình lượt bổ sung

| Tham số | Lượt gốc | Lượt bổ sung mặc định |
|---|---:|---:|
| `max_new_tokens` | 6000 | 8192 |
| `repetition_penalty` | 1.08 | 1.12 |
| Attention | đọc từ checkpoint | SDPA |
| dtype | bf16 | bf16 |
| Model min/max pixels | 262144 / 4194304 | giữ nguyên |

Prompt giữ `HUNYUAN_PROMPT` của repo. Đây là thử thay đổi cách sinh văn bản,
không đảm bảo sửa được OCR lặp hoặc tăng độ đúng. Không dùng giới hạn n-gram cứng
để loại các đoạn lặp hợp lệ trong nguyên bản.

Có thể chỉnh `--max-new-tokens`, `--repetition-penalty`, `--model`, `--output`,
`--page-timeout` (mặc định 600 giây), `--load-timeout` (900 giây).
Thay tham số model/sinh văn bản tạo thư mục fingerprint khác, không trộn cache.

## Kết quả và resume

```text
work/hunyuan_token_retry/<fingerprint>/
  plan.json                 # toàn bộ danh sách 54 trang và cấu hình
  pending.json              # phạm vi thực sự của lần gọi hiện tại
  progress.json             # trạng thái load/page/complete
  candidates/
    4__p00001.json          # OCR bổ sung, geometry và provenance mới
    4__p00001.txt           # văn bản dễ xem
  summary.json              # số trang đã có, còn lại và cờ chất lượng
  timeout.json              # chỉ có nếu worker bị dừng do timeout
```

Chạy lại cùng tham số bỏ qua các JSON kết quả đã lưu; `--limit 1` chọn một trang
chưa chạy, không phải bắt buộc chạy lại trang đầu. Timeout/lỗi worker giữ kết quả
các trang hoàn tất; script báo lỗi để kiểm tra trước khi tiếp tục.

Kết quả là **ứng viên Hunyuan thay thế**, không phải model thứ năm. Không tự sửa
`data/original_ocr_readable/`, manifest, raw gốc hay notebook Kaggle. Giữ ứng viên
có cờ lặp/token để xem xét; không tự chọn bản dài hơn hoặc ít bất đồng hơn làm bản
đúng. Sau khi kiểm tra, mới nhập các bản được chấp nhận và tính lại ngưỡng phục dựng.

Notebook 1/1a vẫn chỉ chạy trên Kaggle; script này là lượt bổ sung local riêng.
