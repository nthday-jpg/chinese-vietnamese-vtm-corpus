# Báo cáo OCR

**Sách 4/5 có CER chữ Hán cao nhất ở cả bốn model và API OCR.** Các trang đã soi cho thấy lỗi sinh lặp, đảo thứ tự đọc và bỏ sót chữ. Golden cũng có trang thiếu nội dung.

## 1. Kết quả so sánh

Đo trên **932/960 trang nguồn** có chữ Hán trong Golden và trong mọi đầu ra OCR áp dụng cho sách đó. Loại **24 trang Golden rỗng**, **2 trang Golden không có chữ Hán** và **2 trang `kinh` (123, 169)** có đầu ra model rỗng. Sách 2/3 và 4/5 là hai cặp PDF trùng, mỗi cặp tính một lần. Các trang lặp hoặc chạm trần token vẫn được tính nếu qua điều kiện chữ Hán.

**Cách đọc số.** D và CER cùng đếm số lần thêm, xóa hoặc thay chữ để khớp hai văn bản. D chia cho độ dài bản dài hơn của từng trang. CER chia cho độ dài Golden, nên có thể vượt 100% khi model sinh nhiều chữ dư. Trung vị lấy giá trị giữa các trang. Chỉ số gộp cộng số phép sửa và mẫu số qua các trang trước khi chia. Golden là một lượt OCR khác, nên các điểm dưới đo mức lệch văn bản.

### So bốn model và API trên cùng 510 trang

Tập này bỏ `bien` vì TongGu không áp dụng. Các chỉ số dưới chỉ tính chữ Hán.

| Model | D gộp | CER gộp | Trung vị CER |
|---|---:|---:|---:|
| Hunyuan | 76.12% | 162.85% | 36.53% |
| PaddleOCR-VL | 29.23% | 29.55% | 20.16% |
| PP-OCR | 30.75% | 30.81% | 22.02% |
| TongGu | 44.49% | 57.83% | 18.22% |
| API OCR | 33.27% | 33.32% | 21.36% |

### So từng sách với Golden

Mỗi ô ghi **D gộp / CER gộp** khi chỉ tính chữ Hán. Tất cả model trong một hàng dùng cùng số trang.

| Sách | Trang | Hunyuan | PaddleOCR-VL | PP-OCR | TongGu | API OCR |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 97 | 69.54% / 117.30% | 30.35% / 30.42% | 31.12% / 31.19% | 24.91% / 25.03% | 32.08% / 32.17% |
| 2/3 | 101 | 75.01% / 122.95% | 24.95% / 24.99% | 28.09% / 28.11% | 24.42% / 24.51% | 27.57% / 27.63% |
| 4/5 | 49 | 97.28% / 504.98% | 59.78% / 62.67% | 60.42% / 60.42% | 86.00% / 215.68% | 74.19% / 74.19% |
| 6 | 96 | 47.34% / 53.37% | 21.75% / 21.81% | 22.55% / 22.57% | 19.87% / 19.92% | 23.05% / 23.10% |
| bien | 422 | 0.42% / 0.42% | 0.92% / 0.92% | 1.65% / 1.65% | — | 5.73% / 5.73% |
| kinh | 167 | 0.88% / 0.88% | 4.02% / 4.02% | 8.31% / 8.37% | 2.48% / 2.48% | 5.73% / 5.74% |

Ở sách 4/5, đầu ra lặp dài đẩy CER Hán của Hunyuan lên khoảng **505 phép sửa trên 100 chữ Golden**. Ở `bien`, cả ba model áp dụng có CER Hán dưới **2%**.

API có **1.133 bản ghi trên 8 PDF**, tương ứng **960 trang nguồn** sau khi gộp bản scan trùng. Hai trang API trả rỗng là bìa **4/5 tr. 69–70**, đã nằm ngoài tập đo. Trên cùng **932 trang**, API có **D Hán gộp 17.36%**, **CER Hán gộp 17.38%** (62.281 phép sửa trên 358.389 chữ Golden) và **trung vị CER Hán 8.13%**. CER tính cả dấu, số và ký hiệu là **19.87%**. Sách `bien` dùng `ocr_id=4` và có 422 bản ghi. Các sách còn lại dùng `ocr_id=1` và có 711 bản ghi. Phép đo dùng trường `text` chứa OCR chữ Hán. [Văn bản API](../data/ocr_raw/api_ocr.json) · [Bản ghi API có tọa độ](../data/ocr_raw/api_source/) · [Số liệu theo sách](../reports/ocr_source_page_audit/api_ocr_summary.json).

Ở **4/5 tr. 10**, API trả **398 chữ Hán** so với **699** của Golden, với D **86.70%**. Ở tr. **12** và **37**, D lần lượt là **89.83%** và **89.69%**. Cần soi ảnh để phân biệt thiếu chữ, sai chữ và đảo cột. [Ảnh trang 10](../reports/ocr_source_page_audit/4__p10.png) · [Ảnh trang 37](../reports/ocr_source_page_audit/4__p37.png).

### Trang bất đồng lớn với Golden

Các số dưới là **D chữ Hán của từng model so với Golden**, tính trên từng trang. Chỉ xét trang thuộc tập đã lọc và có ít nhất 80 chữ Hán ở Golden lẫn mỗi model áp dụng.

**Cả bốn model đều lệch từ 60% trở lên.** Có đúng tám trang thỏa điều kiện, đều thuộc sách **4/5**. Bảng xếp giảm dần theo D thấp nhất của bốn model trên mỗi trang.

| Trang PDF sách 4/5 | Hunyuan | PaddleOCR-VL | PP-OCR | TongGu |
|---|---:|---:|---:|---:|
| 68 | 90.17% | 78.87% | 80.96% | 95.73% |
| 10 | 99.70% | 99.13% | 70.96% | 99.82% |
| 12 | 99.10% | 70.45% | 72.81% | 99.62% |
| 37 | 99.70% | 83.67% | 69.75% | 72.97% |
| 19 | 99.17% | 79.90% | 72.34% | 69.59% |
| 8 | 99.58% | 64.28% | 68.01% | 64.99% |
| 17 | 99.05% | 72.22% | 63.57% | 99.82% |
| 57 | 98.61% | 62.18% | 61.33% | 99.18% |

Trang **68** có mức lệch thấp nhất trong bốn model vẫn là **78.87%**. Ở trang **10**, Hunyuan, PaddleOCR-VL và TongGu đều có cờ lặp. [Ảnh trang 10](../reports/ocr_source_page_audit/4__p10.png).

**Một model lệch nổi bật.** Các ví dụ dưới có một model vượt 75% trong khi các model còn lại ở dưới 60%. Dấu “—” nghĩa là TongGu không áp dụng cho `bien`.

| Sách, trang PDF | Hunyuan | PaddleOCR-VL | PP-OCR | TongGu | Dấu hiệu đã kiểm |
|---|---:|---:|---:|---:|---|
| 6, tr. 76 | **99.80%** | 29.78% | 31.46% | 24.44% | Hunyuan có cờ lặp và chạm trần token. |
| 2/3, tr. 23 | **76.97%** | 23.10% | 22.45% | 13.99% | Hunyuan đọc đoạn ở lá trái trước lá phải. |
| bien, tr. 408 | 0.20% | **77.06%** | 3.14% | — | PaddleOCR-VL đưa đoạn dưới lên trước đoạn đầu trang. |
| bien, tr. 418 | 0.86% | 1.15% | **85.01%** | — | PP-OCR ghép xen hai cột mục lục. |

Danh sách đầy đủ theo từng cặp và các trang CER cao nằm ở mục **Tra cứu chi tiết**.

## 2. Các vấn đề cần xử lý

| Vấn đề | Bằng chứng và việc cần kiểm |
|---|---|
| **Thiếu bản tham chiếu** | **24 trang Golden rỗng có chữ trên ảnh.** Cần bổ sung bản chép nếu muốn đưa chúng vào đánh giá. `kinh` tr. 169 cũng cần kiểm lại Golden vì thân trang trống. [Ảnh 24 trang](../reports/ocr_source_page_audit/empty_reference_pages.png) |
| **Sinh lặp và chạm trần token** | Bắt đầu từ **4/5 tr. 10 và 18**. Tr. 10 có vòng lặp ở ba model. Tr. 18 PaddleOCR-VL sinh 4.763 chữ Hán dù thiếu thông tin `hit_max`. Kiểm cả phần chữ còn thiếu sau điểm lặp trước khi chấp nhận đầu ra chạy lại. [Đối chiếu ảnh và OCR](../reports/ocr_source_page_audit/report.md) |
| **Sai thứ tự đọc** | **`bien` tr. 418** có hai cột mục lục. Ghép hết cột trái rồi đến cột phải, giữ nguyên chữ PP-OCR, giảm từ **295 xuống 6 phép sửa**. Sách **2/3 tr. 23** có lỗi đảo hai lá. Có thể kiểm riêng thứ tự bằng khung dòng hoặc bản chép theo cột. [Ảnh trang 418](../data/original_ocr_readable/images/bien/page_00418.png) |
| **Bỏ sót vùng chữ** | **`kinh` tr. 123** có cột `華氏中藏經卷中終`, nhưng PaddleOCR-VL trả rỗng và chỉ lưu nhãn bố cục `chart`, `number`. **`kinh` tr. 100** thiếu khung dòng PP-OCR ở `木通`. Kiểm bố cục và khung trước khi ghép văn bản. [Ảnh trang 123](../data/original_ocr_readable/images/kinh/page_00123.png) · [Khung PP-OCR trang 100](../reports/ocr_source_page_audit/kinh__p100_ppocr_boxes.png) |
| **Nguồn ảnh khó đọc** | **4/5 tr. 26** có raster gốc **600×532** và cả bốn model dùng cùng PNG. Chép vài cột chữ nhỏ để đối chứng trước khi đánh giá tăng độ phân giải hoặc phục dựng. [Ảnh trang 26](../reports/ocr_source_page_audit/4__p26.png) |

### Tóm tắt cờ OCR

Đếm theo trang nguồn trên dữ liệu checkpoint. Một trang có thể mang nhiều cờ.

| Model | Lặp mạnh | Nghi lặp | Chạm trần token đã xác nhận |
|---|---:|---:|---:|
| Hunyuan | 62 | 9 | 54 |
| PaddleOCR-VL | 3 | 0 | Chưa có dữ liệu |
| PP-OCR | 0 | 0 | 0 |
| TongGu | 13 | 0 | Chưa có dữ liệu |
| API OCR | 0 | 0 | Chưa có dữ liệu |

Các số trên đếm tín hiệu do bộ dò phát hiện. Ở `bien` tr. 263, câu bị gắn cờ nghi lặp ở Hunyuan và Golden thực sự được in hai lần trên ảnh. [Ảnh trang 263](../data/original_ocr_readable/images/bien/page_00263.png).

Notebook OCR chạy thử **Hunyuan trên 4 tr. 10**, OCR từng nửa phải → trái và hiển thị bản cũ cùng ứng viên. Notebook UniHIR chạy trên vùng **112×153 px** có chữ mờ và vết đen trên **4 tr. 26**. Cả hai xuất ZIP từ Kaggle. [OCR lại trang lỗi](../1d_ocr_lai_trang_loi.ipynb) · [Thử UniHIR](../1c_chay_thu_unihir_offline.ipynb).

## 3. Tra cứu chi tiết

Các danh sách xếp hạng dùng tập **932 trang**. Xếp hạng D yêu cầu ít nhất 80 chữ Hán ở mỗi bên. Xếp hạng CER yêu cầu Golden có ít nhất 80 chữ Hán.

<details>
<summary>Bổ sung — đủ 10 cặp, điểm theo sách và ký tự hiển thị</summary>

#### Sáu cặp model và bốn cặp model–Golden

D Hán qua lọc cờ bỏ thêm đầu ra có cờ chất lượng, tín hiệu lặp mạnh hoặc nghi lệch độ dài. Số trong ngoặc là số trang còn lại.

| Cặp | Trang D Hán | Trung vị D Hán | D Hán gộp | Trung vị D hiển thị | D Hán qua lọc cờ (n) |
|---|---:|---:|---:|---:|---:|
| Hunyuan ↔ PaddleOCR-VL | 932 | 2.56% | 47.28% | 3.42% | 16.61% (884) |
| Hunyuan ↔ PP-OCR | 932 | 4.78% | 47.86% | 7.51% | 17.45% (884) |
| Hunyuan ↔ TongGu | 510 | 37.90% | 77.33% | 38.33% | 42.44% (461) |
| PaddleOCR-VL ↔ PP-OCR | 932 | 3.85% | 14.24% | 6.34% | 13.74% (931) |
| PaddleOCR-VL ↔ TongGu | 510 | 22.51% | 48.39% | 22.69% | 29.89% (499) |
| PP-OCR ↔ TongGu | 510 | 24.27% | 49.81% | 24.57% | 31.80% (499) |
| Hunyuan ↔ Golden | 932 | 0.85% | 46.58% | 7.71% | 15.64% (884) |
| PaddleOCR-VL ↔ Golden | 932 | 2.57% | 12.94% | 8.62% | 12.44% (931) |
| PP-OCR ↔ Golden | 932 | 4.72% | 13.94% | 10.71% | 13.94% (932) |
| TongGu ↔ Golden | 510 | 18.13% | 44.49% | 18.33% | 24.77% (499) |

#### Model–Golden trên toàn bộ phạm vi áp dụng của từng model

| Model | Trang | Trung vị D Hán | D Hán gộp | CER Hán gộp | D hiển thị gộp | CER hiển thị gộp | CER Hán qua lọc cờ (n) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hunyuan | 932 | 0.85% | 46.58% | 68.98% | 46.52% | 68.10% | 15.72% (884) |
| PaddleOCR-VL | 932 | 2.57% | 12.94% | 13.01% | 15.03% | 15.13% | 12.45% (931) |
| PP-OCR | 932 | 4.72% | 13.94% | 13.96% | 16.68% | 16.79% | 13.96% (932) |
| TongGu | 510 | 18.13% | 44.49% | 57.83% | 46.61% | 62.98% | 24.87% (499) |

#### Trung vị D chữ Hán theo sách

| Cặp | 1 | 2/3 | 4/5 | 6 | bien | kinh |
|---|---:|---:|---:|---:|---:|---:|
| Hunyuan ↔ PaddleOCR-VL | 42.27% | 56.60% | 98.42% | 40.45% | 0.56% | 2.92% |
| Hunyuan ↔ PP-OCR | 41.04% | 58.46% | 98.44% | 42.26% | 0.88% | 6.40% |
| Hunyuan ↔ TongGu | 41.42% | 58.36% | 97.58% | 40.25% | — | 1.84% |
| PaddleOCR-VL ↔ PP-OCR | 25.99% | 26.07% | 63.78% | 21.72% | 0.98% | 4.91% |
| PaddleOCR-VL ↔ TongGu | 26.60% | 27.25% | 67.80% | 23.46% | — | 2.92% |
| PP-OCR ↔ TongGu | 25.27% | 31.01% | 69.86% | 24.85% | — | 6.84% |
| Hunyuan ↔ Golden | 37.91% | 55.94% | 98.29% | 37.76% | 0.23% | 0.65% |
| PaddleOCR-VL ↔ Golden | 25.71% | 23.80% | 56.57% | 21.04% | 0.47% | 3.07% |
| PP-OCR ↔ Golden | 23.08% | 27.16% | 60.08% | 22.75% | 0.79% | 6.37% |
| TongGu ↔ Golden | 19.75% | 24.03% | 58.75% | 19.62% | — | 1.88% |

#### D gộp / CER gộp theo sách, tính cả dấu, số và ký hiệu

| Sách | Trang | Hunyuan | PaddleOCR-VL | PP-OCR | TongGu |
|---|---:|---:|---:|---:|---:|
| 1 | 97 | 70.97% / 125.58% | 30.44% / 30.51% | 31.27% / 31.36% | 31.21% / 34.24% |
| 2/3 | 101 | 75.12% / 123.70% | 25.03% / 25.08% | 28.28% / 28.30% | 24.64% / 24.82% |
| 4/5 | 49 | 97.33% / 518.26% | 60.21% / 63.18% | 60.77% / 60.77% | 86.81% / 231.03% |
| 6 | 96 | 47.48% / 53.54% | 21.84% / 21.89% | 22.64% / 22.66% | 20.01% / 20.08% |
| bien | 422 | 4.70% / 4.72% | 5.80% / 5.81% | 7.44% / 7.50% | — |
| kinh | 167 | 0.89% / 0.89% | 4.16% / 4.17% | 10.47% / 10.76% | 2.51% / 2.51% |

</details>

<details>
<summary>Trang rỗng, thiếu chữ Hán hoặc ngoài phạm vi model</summary>

Tách bản ghi thiếu khỏi chuỗi rỗng. Một trang chỉ có mã, số hoặc dấu được liệt kê ở cột không có chữ Hán. Báo cáo cũng dò chuỗi giống placeholder để tránh bỏ qua trường hợp API trả văn bản như `None`.

Đã đối chiếu ID trang trong checkpoint với chỉ mục ảnh. **Không có trang nào thiếu bản ghi trong phạm vi áp dụng của từng model.** Riêng TongGu trên **422 trang `bien`** nằm ngoài phạm vi theo cấu hình. Ba model còn lại có bản ghi cho đủ 422 trang đó. [Cấu hình loại trừ](../src/vtm_corpus/config.py).

| Nguồn văn bản | Có bản ghi / áp dụng | Thiếu bản ghi | Rỗng sau bỏ khoảng trắng | Có ký tự, không có chữ Hán |
|---|---:|---|---|---|
| Hunyuan | 960 / 960 | 0 | 0 | 3 · **4/5** (69–70) · **kinh** (169) |
| PaddleOCR-VL | 960 / 960 | 0 | 4 · **4/5** (69–70) · **kinh** (123, 169) | 0 |
| PP-OCR | 960 / 960 | 0 | 0 | 2 · **4/5** (69–70) |
| TongGu | 538 / 538 | 0 | 0 | 0 |
| API OCR | 960 / 960 | 0 | 2 · **4/5** (69–70) | 0 |
| Golden | 960 / 960 | 0 | 24 · **1** (53, 80, 82) · **2/3** (36, 44) · **4/5** (2, 6, 9, 18, 22, 24, 29, 35, 38, 41, 46–47, 49–50, 54, 58, 63–64, 66) | 2 · **4/5** (69–70) |

- Chuỗi nghi là placeholder gồm Hunyuan, 4/5 tr. 70 (`None`).
- **24 trang Golden rỗng** đều có chữ trên ảnh và có checkpoint OCR của các model. Các trang này bị loại ở bước chọn tham chiếu để tính CER vì Golden rỗng làm mẫu số bằng 0. Dữ liệu của lượt Golden chỉ có ID và văn bản, nên chưa biết lượt đó đã chạy và trả rỗng hay gặp vấn đề trước khi xuất. [Bảng ảnh của 24 trang](../reports/ocr_source_page_audit/empty_reference_pages.png).
- **4/5 tr. 69–70** là hai ảnh bìa. Trang 69 có mã bìa, trang 70 gần như trắng. Checkpoint PaddleOCR-VL ghi có chạy cả hai trang và lưu kết quả rỗng, với 0 khung bố cục. Golden của hai trang có ký tự nhưng không có chữ Hán, nên cả hai trang bị loại khi lập tập Golden đã lọc. [Bìa trang 69](../reports/ocr_source_page_audit/4__p69.png) · [Bìa trang 70](../reports/ocr_source_page_audit/4__p70.png).
- **`kinh` tr. 123** có một cột chính đọc rõ `華氏中藏經卷中終`. PaddleOCR-VL đã chạy và lưu hai khung bố cục có nhãn `chart` và `number`, nhưng kết quả có 0 khối chữ. Cả hai nhãn nằm trong danh sách bỏ qua lúc lấy văn bản. Trang bị loại khỏi phép đo do PaddleOCR-VL trả rỗng. Bằng chứng nghiêng về lỗi phân loại bố cục. Checkpoint thiếu kết quả phân tích trước lọc để biết có chữ nào từng được nhận dạng. Cả ba model kia đều có chuỗi trên. [Ảnh đầu vào](../data/original_ocr_readable/images/kinh/page_00123.png) · [Mã lấy văn bản](../src/vtm_corpus/ocr_worker.py).
- **`kinh` tr. 169** có số trang, dấu ở lề và khung thân trang trống. PaddleOCR-VL cũng đã chạy, lưu hai khung `chart` và `number` và trả rỗng. Golden trả `口鎮発火`. Trang này bị loại khỏi phép đo do PaddleOCR-VL rỗng và Hunyuan không có chữ Hán. Chưa có bằng chứng ảnh cho bốn chữ Golden trong khung chính. [Ảnh đầu vào](../data/original_ocr_readable/images/kinh/page_00169.png).

</details>

<details>
<summary>Danh sách trang lặp, chạm trần token và các cờ OCR khác</summary>

Tín hiệu lặp mạnh lấy từ cờ checkpoint hoặc dò chuỗi lặp dài/dòng lặp. Tín hiệu nhẹ cần con người xem vì số thứ tự, công thức và mục lục có thể lặp hợp lệ. Không cắt đuôi chuỗi trước khi tính điểm. [Mã dò và ngưỡng](../tools/audit_ocr_repetition.py), [bằng chứng từng đoạn](../reports/ocr_repetition_audit/repetition_evidence.json).

| Model | Lặp mạnh | Nghi lặp, cần soi | `hit_max=True` | Nghi thiếu chữ do lệch độ dài | Nghi dư chữ | Nhiều ký hiệu |
|---|---|---|---|---|---|---|
| Hunyuan | 62 · **1** (26, 42, 87, 91, 97–99) · **2/3** (12, 19, 37, 65) · **4/5** (1–2, 4–17, 19–20, 22–25, 27–29, 33–34, 37–41, 46–48, 50–51, 54–60, 62, 64–66) · **6** (76, 88, 90) | 9 · **1** (100) · **2/3** (5, 45, 77) · **4/5** (44, 49, 53, 61) · **bien** (263) | 54 · **1** (26, 42, 87, 91, 97–99) · **2/3** (12, 19, 37, 65) · **4/5** (1–2, 4, 6–17, 19–20, 22–25, 27–29, 33–34, 37–40, 46, 50, 55–58, 60, 62, 64–66) · **6** (76, 88) | 1 · **4/5** (68) | 0 | 0 |
| PaddleOCR-VL | 3 · **4/5** (10, 18, 46) | 0 | 0 | 0 | 0 | 0 |
| PP-OCR | 0 | 0 | 0 | 0 | 0 | 0 |
| TongGu | 13 · **1** (4) · **4/5** (10–13, 17, 33, 38–39, 57–58, 62, 68) | 0 | 0 | 0 | 0 | 1 · **1** (4) |

### Lặp trong Golden

Dùng cùng bộ dò chuỗi lặp và dòng lặp trên Golden. Kết quả là tín hiệu kiểm tra vì Golden thiếu metadata sinh và không có cờ trần token.
- **Mạnh** · 0.
- **Cần soi** · 1 · **bien** (263).

Ca duy nhất bị nghi là **`bien` tr. 263**. Cụm 56 ký tự xuất hiện hai lần trong Golden và in hai lần trên ảnh, trước và sau tên bài thuốc. Tín hiệu này cũng xuất hiện ở Hunyuan. Đây là lặp có thật trên nguồn. [Ảnh đầu vào](../data/original_ocr_readable/images/bien/page_00263.png).

### Có và thiếu thông tin về trần token

| Nguồn | Chạm trần | Đã ghi `False` | Không có giá trị xác nhận |
|---|---:|---:|---:|
| Hunyuan | 54 | 906 | 0 |
| PaddleOCR-VL | 0 | 0 | 960 |
| PP-OCR | 0 | 960 | 0 |
| TongGu | 0 | 0 | 538 |
| API OCR | 0 | 0 | 960 |
| Golden | 0 | 0 | 960 |

</details>

<details>
<summary>Xếp hạng và danh sách trang lệch lớn theo D, CER</summary>

Mỗi ô có D và số chữ Hán của hai phía theo thứ tự cặp. Cột qua lọc cờ chỉ giữ trang không có dấu hiệu chất lượng đã nêu.

| Cặp | Tất cả output đủ 80 chữ mỗi bên | Qua lọc cờ |
|---|---|---|
| Hunyuan ↔ PaddleOCR-VL | 4/5 tr. 56 · **99.87%** (3,017/374)<br>6 tr. 76 · **99.80%** (1,537/347)<br>4/5 tr. 7 · **99.80%** (4,021/448) | 2/3 tr. 74 · **97.37%** (342/332)<br>2/3 tr. 54 · **97.18%** (294/319)<br>2/3 tr. 75 · **96.33%** (289/300) |
| Hunyuan ↔ PP-OCR | 4/5 tr. 15 · **99.87%** (6,001/666)<br>4/5 tr. 33 · **99.85%** (4,002/351)<br>4/5 tr. 56 · **99.83%** (3,017/344) | 2/3 tr. 54 · **99.32%** (294/292)<br>2/3 tr. 74 · **99.12%** (342/337)<br>2/3 tr. 7 · **95.88%** (388/349) |
| Hunyuan ↔ TongGu | 4/5 tr. 39 · **99.98%** (5,967/5,965)<br>4/5 tr. 10 · **99.95%** (3,050/5,997)<br>4/5 tr. 13 · **99.95%** (4,629/5,993) | 2/3 tr. 74 · **98.83%** (342/332)<br>2/3 tr. 54 · **97.26%** (294/329)<br>2/3 tr. 89 · **97.08%** (342/320) |
| PaddleOCR-VL ↔ PP-OCR | 4/5 tr. 10 · **98.75%** (2,080/581)<br>bien tr. 421 · **87.50%** (144/128)<br>6 tr. 1 · **87.10%** (233/248) | bien tr. 421 · **87.50%** (144/128)<br>6 tr. 1 · **87.10%** (233/248)<br>4/5 tr. 37 · **85.74%** (409/568) |
| PaddleOCR-VL ↔ TongGu | 4/5 tr. 13 · **99.95%** (466/5,993)<br>4/5 tr. 39 · **99.90%** (513/5,965)<br>4/5 tr. 10 · **99.87%** (2,080/5,997) | 4/5 tr. 37 · **87.53%** (409/762)<br>4/5 tr. 19 · **84.52%** (325/549)<br>6 tr. 1 · **84.46%** (233/251) |
| PP-OCR ↔ TongGu | 4/5 tr. 13 · **99.95%** (479/5,993)<br>4/5 tr. 39 · **99.83%** (414/5,965)<br>4/5 tr. 10 · **99.80%** (581/5,997) | 4/5 tr. 19 · **79.05%** (470/549)<br>4/5 tr. 60 · **78.75%** (428/753)<br>4/5 tr. 8 · **78.71%** (417/653) |
| Hunyuan ↔ Golden | 4/5 tr. 7 · **99.83%** (4,021/658)<br>6 tr. 76 · **99.80%** (1,537/356)<br>4/5 tr. 56 · **99.77%** (3,017/417) | 2/3 tr. 74 · **98.54%** (342/343)<br>2/3 tr. 54 · **97.21%** (294/323)<br>2/3 tr. 89 · **96.49%** (342/319) |
| PaddleOCR-VL ↔ Golden | 4/5 tr. 10 · **99.13%** (2,080/699)<br>4/5 tr. 37 · **83.67%** (409/747)<br>6 tr. 1 · **82.54%** (233/252) | 4/5 tr. 37 · **83.67%** (409/747)<br>6 tr. 1 · **82.54%** (233/252)<br>4/5 tr. 19 · **79.90%** (325/582) |
| PP-OCR ↔ Golden | bien tr. 421 · **86.71%** (128/143)<br>bien tr. 418 · **85.01%** (346/347)<br>4/5 tr. 68 · **80.96%** (285/478) | bien tr. 421 · **86.71%** (128/143)<br>bien tr. 418 · **85.01%** (346/347)<br>4/5 tr. 68 · **80.96%** (285/478) |
| TongGu ↔ Golden | 4/5 tr. 39 · **99.95%** (5,965/474)<br>4/5 tr. 13 · **99.83%** (5,993/646)<br>4/5 tr. 10 · **99.82%** (5,997/699) | 4/5 tr. 37 · **72.97%** (762/747)<br>4/5 tr. 19 · **69.59%** (549/582)<br>4/5 tr. 28 · **68.07%** (705/714) |

### Danh sách trang vượt ngưỡng theo cặp

**D ≥ 60%** và hai bên có ít nhất 80 chữ Hán. Các trang có cờ vẫn nằm trong danh sách.

| Cặp | Số trang | Trang theo sách |
|---|---:|---|
| Hunyuan ↔ PaddleOCR-VL | 123 | **1** (5–6, 8, 26, 42, 61, 78–79, 81, 83–90, 92–100) · **2/3** (5–7, 12, 18–21, 23–24, 34, 37–38, 41, 51–52, 54–55, 58–59, 61, 65, 68, 73–75, 77, 81–82, 84, 87–90, 93–95, 98–102) · **4/5** (1, 3–5, 7–8, 10–17, 19–21, 23, 25–28, 30–34, 36–37, 39–40, 43–44, 48, 51–53, 55–57, 59–62, 65, 67–68) · **6** (1, 41, 52, 76, 88–90) · **bien** (408) |
| Hunyuan ↔ PP-OCR | 133 | **1** (5–6, 8, 26, 42, 61, 78–79, 81, 83–90, 92–100) · **2/3** (5–7, 12, 14, 18–21, 23–24, 28, 31, 33–34, 37–38, 41, 45, 51–52, 54–55, 58–59, 61, 65, 68, 72–75, 77, 81–82, 84, 87–90, 93–95, 98–102) · **4/5** (1, 3–5, 7–8, 10–17, 19–21, 23, 25–28, 30–34, 36–37, 39–40, 44, 48, 51–53, 55–57, 59–62, 65, 67–68) · **6** (41, 52, 66, 76, 88–90) · **bien** (417–421) · **kinh** (124) |
| Hunyuan ↔ TongGu | 121 | **1** (5–6, 8, 26, 42, 61, 78–79, 81, 83–89, 92–100) · **2/3** (6–7, 12, 18–21, 23–24, 33–34, 37–38, 41, 49, 51–52, 54–55, 58–59, 61, 65, 68, 73–75, 77, 81–82, 84, 87–90, 93–95, 97–102) · **4/5** (1, 3–5, 7–8, 10–17, 19–21, 23, 25–28, 30–34, 36–37, 39–40, 44, 48, 51–53, 55–57, 59–62, 65, 67–68) · **6** (41, 52, 76, 88–90) |
| PaddleOCR-VL ↔ PP-OCR | 46 | **1** (5–6, 8, 81, 84–87) · **4/5** (5, 7–8, 10–12, 14–15, 17, 19, 21, 23, 26–28, 31, 33–34, 36–37, 40, 44, 48, 51–52, 55, 57, 60, 62, 68) · **6** (1) · **bien** (408, 417–421) · **kinh** (124) |
| PaddleOCR-VL ↔ TongGu | 39 | **1** (5–6, 8, 96, 99) · **4/5** (5, 7–8, 10–15, 17, 19, 21, 23, 26, 28, 30–31, 33–34, 36–37, 39–40, 44, 48, 51–52, 55, 57, 60, 62, 67–68) · **6** (1) |
| PP-OCR ↔ TongGu | 48 | **1** (8, 84–85, 87, 96, 98–99) · **4/5** (1, 4–5, 7–8, 10–15, 17, 19, 21, 23, 25–28, 30–34, 36–37, 39–40, 44, 48, 51, 55, 57, 59–62, 65, 67–68) · **kinh** (124) |
| Hunyuan ↔ Golden | 119 | **1** (5–6, 8, 26, 36, 42, 61, 78–79, 81, 83–89, 92–100) · **2/3** (6–7, 12, 18–21, 23–24, 33–34, 37–38, 41, 51–52, 54, 58–59, 65, 68, 73–75, 77, 81–82, 84, 87–90, 93–95, 97–102) · **4/5** (1, 3–5, 7–8, 10–17, 19–21, 23, 25–28, 30–34, 36–37, 39–40, 44, 48, 51–53, 55–57, 59–62, 65, 67–68) · **6** (41, 52, 76, 88–90) |
| PaddleOCR-VL ↔ Golden | 19 | **1** (5–6, 8) · **4/5** (5, 8, 10, 12, 17, 19, 21, 26, 36–37, 48, 52, 57, 68) · **6** (1) · **bien** (408) |
| PP-OCR ↔ Golden | 34 | **1** (85, 87, 97) · **4/5** (5, 8, 10–15, 17, 19, 23, 26–28, 33–34, 37, 40, 48, 55, 57, 60–62, 68) · **bien** (417–421) · **kinh** (124) |
| TongGu ↔ Golden | 22 | **4/5** (8, 10–15, 17, 19, 21, 23, 28, 33–34, 37, 39–40, 55, 57, 60, 62, 68) |

### Trang CER cao so với Golden

Ngưỡng **CER ≥ 75%**. Cột bên phải lấy ba trang CER cao nhất và ba trang có số phép sửa lớn nhất cho mỗi model. Hai cách xếp bắt được cả trang ngắn lệch nhiều theo tỷ lệ lẫn trang tạo khối lượng sửa lớn.

| Model | Số trang và danh sách CER ≥ 75% | Cao nhất theo CER | Nhiều phép sửa nhất |
|---|---|---|---|
| Hunyuan | 86 · **1** (26, 42, 78–79, 81, 86–87, 89, 91–94, 97–100) · **2/3** (7, 12, 19, 23–24, 37–38, 41, 54, 58, 65, 68, 74–75, 81–82, 84, 87–89, 93–95, 100) · **4/5** (1, 4–5, 7–8, 10–17, 19–21, 23, 25–28, 30–31, 33–34, 36–37, 39–40, 44, 48, 51, 55–57, 59–62, 65, 68) · **6** (52, 76, 88–90) | 2/3 tr. 65 · **1670.21%** (5,662 phép sửa)<br>2/3 tr. 37 · **1664.29%** (5,825 phép sửa)<br>2/3 tr. 12 · **1499.44%** (5,353 phép sửa) | 4/5 tr. 1 · **1267.58%** (6,490 phép sửa)<br>4/5 tr. 55 · **824.23%** (6,396 phép sửa)<br>1 tr. 26 · **1318.70%** (6,277 phép sửa) |
| PaddleOCR-VL | 7 · **1** (8) · **4/5** (10, 19, 37, 68) · **6** (1) · **bien** (408) | 4/5 tr. 10 · **294.99%** (2,062 phép sửa)<br>4/5 tr. 37 · **83.67%** (625 phép sửa)<br>6 tr. 1 · **82.54%** (208 phép sửa) | 4/5 tr. 10 · **294.99%** (2,062 phép sửa)<br>4/5 tr. 37 · **83.67%** (625 phép sửa)<br>4/5 tr. 12 · **70.45%** (596 phép sửa) |
| PP-OCR | 7 · **4/5** (68) · **bien** (417–421) · **kinh** (124) | bien tr. 421 · **86.71%** (124 phép sửa)<br>bien tr. 418 · **85.01%** (295 phép sửa)<br>4/5 tr. 68 · **80.96%** (387 phép sửa) | 4/5 tr. 11 · **73.55%** (684 phép sửa)<br>4/5 tr. 12 · **72.81%** (616 phép sửa)<br>4/5 tr. 62 · **66.91%** (556 phép sửa) |
| TongGu | 10 · **4/5** (10–13, 17, 33, 39, 57, 62, 68) | 4/5 tr. 39 · **1257.81%** (5,962 phép sửa)<br>4/5 tr. 13 · **926.16%** (5,983 phép sửa)<br>4/5 tr. 10 · **856.37%** (5,986 phép sửa) | 4/5 tr. 57 · **733.94%** (6,055 phép sửa)<br>4/5 tr. 10 · **856.37%** (5,986 phép sửa)<br>4/5 tr. 13 · **926.16%** (5,983 phép sửa) |

</details>

<details>
<summary>Dấu hiệu đảo thứ tự và các phép kiểm để phân biệt nguyên nhân</summary>

Đo thêm **D tần suất** bằng cách đếm số lần xuất hiện của từng chữ Hán rồi so phần chung, bỏ qua thứ tự. Chênh lệch lớn giữa D và D tần suất là tín hiệu để soi bố cục. Chữ trùng ngẫu nhiên cũng có thể làm D tần suất thấp. Bảng lấy cặp model–Golden không mang cờ model, hai phía có ít nhất 100 chữ Hán, D từ 45% và độ chênh từ 30 điểm phần trăm.

| Cặp | Trang | D theo thứ tự | D tần suất | Chênh lệch (điểm %) |
|---|---|---:|---:|---:|
| PP-OCR ↔ Golden | bien tr. 418 | 85.01% | 1.73% | 83.29 |
| PP-OCR ↔ Golden | bien tr. 419 | 79.81% | 1.28% | 78.53 |
| PaddleOCR-VL ↔ Golden | bien tr. 408 | 77.06% | 0.59% | 76.47 |
| PP-OCR ↔ Golden | bien tr. 417 | 76.31% | 10.84% | 65.46 |
| PP-OCR ↔ Golden | bien tr. 421 | 86.71% | 21.68% | 65.03 |
| PP-OCR ↔ Golden | kinh tr. 124 | 77.39% | 13.04% | 64.35 |
| PaddleOCR-VL ↔ Golden | 6 tr. 1 | 82.54% | 25.40% | 57.14 |
| Hunyuan ↔ Golden | 2/3 tr. 74 | 98.54% | 41.40% | 57.14 |
| Hunyuan ↔ Golden | 6 tr. 89 | 83.38% | 28.10% | 55.29 |
| Hunyuan ↔ Golden | 2/3 tr. 89 | 96.49% | 41.81% | 54.68 |

Có **31 trường hợp model–trang** thỏa tiêu chí trên. Chỉ dùng tín hiệu này để chọn ảnh kiểm tra.

- **`bien` tr. 418** có mục lục hai cột. PP-OCR ghép xen một dòng trái rồi một dòng phải. Dùng tâm khung dòng để ghép hết cột trái rồi đến cột phải, giữ nguyên chữ của PP-OCR, làm số phép sửa so với Golden giảm từ **295 xuống 6 trên 347 chữ** (85,01% → 1,73%). Đây là bằng chứng trực tiếp cho lỗi thứ tự. [Ảnh đầu vào](../data/original_ocr_readable/images/bien/page_00418.png).
- **`bien` tr. 408** có đoạn văn tiếp nối nằm trên tiêu đề `行浆务令满足论`. PaddleOCR-VL đưa tiêu đề và đoạn bên dưới lên trước đoạn trên. D chữ Hán là **77,06%**, còn chênh lệch tần suất chỉ **0,59%**. [Ảnh đầu vào](../data/original_ocr_readable/images/bien/page_00408.png).
- **`kinh` tr. 124** có cột chữ lớn và các cặp chữ nhỏ ghi lượng thuốc. PP-OCR đưa `萬應圓` lên đầu rồi trộn tiêu đề, tên và lượng thuốc từ nhiều cột. Đây là trường hợp cần kiểm nhóm theo cột và vị trí chữ nhỏ. [Ảnh đầu vào](../data/original_ocr_readable/images/kinh/page_00124.png).

#### Giả thuyết về chênh lệch giữa hai lượt Hunyuan

Pixel đầu vào, cấu hình sinh và hậu xử lý đều có thể ảnh hưởng. Golden thiếu hash ảnh và cấu hình chạy để tách các yếu tố. Lượt thử chia hai lá ở 4/5 tr. 1 hết cờ lặp nhưng vẫn sai tên sách. Cần so với bản chép trước khi dùng làm kết quả thay thế.

[Ảnh và output đối chiếu 36 trang](../reports/ocr_source_page_audit/report.md) · [Kiểm tra setup Hunyuan trước đó](../reports/hunyuan_setup_deep_audit/report.md).

</details>

<details>
<summary>Dữ liệu và xác minh sách trùng</summary>

Đầu vào gồm [checkpoint OCR ảnh gốc](../data/original_ocr_readable/README.md), [Golden gốc](../data/golden/golden.json) và [danh sách trang bị loại khỏi Golden](../data/golden_filtered/excluded_pages.json).

Sách 2/3 trùng đủ **103 trang**, sách 4/5 trùng đủ **70 trang**. SHA-256 PDF và ảnh khớp, văn bản Golden trùng, cả **692 cặp bản ghi OCR** khớp văn bản, tọa độ và cờ chất lượng. JSON thô khác ở bộ đếm thời gian và đường dẫn crop tạm của TongGu.

- Đối chiếu SHA-256 của bản ghi OCR với PDF, kiểm trang theo index ảnh và kiểm hash đoạn lặp với checkpoint trước khi dùng kết quả. SHA-256 Golden `756cc00be77c522b435a2e89a8cba9fbef5287bfc15ac6dbbfc0767fd84db48e`.
- Chạy lại các phép tính trên CPU từ bản gốc. Không sinh OCR mới, không thay checkpoint, Golden hay cấu hình notebook.

</details>
