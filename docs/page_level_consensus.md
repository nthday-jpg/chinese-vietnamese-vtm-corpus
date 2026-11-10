# Page-level consensus trước Character-level consensus

Notebook 3 tự chạy lớp này khi gọi `compare_ocr_results(session)`, kể cả khi chỉ
đọc checkpoint/log cũ. Không cần weights, inference mới hay chỉnh từng trang.
Luồng xử lý: đọc output/kiểm tra chất lượng → đối chiếu cấu trúc trang → giữ
trạng thái không bỏ phiếu khi có lỗi cấu trúc đủ bằng chứng → căn chỉnh ký tự.

## Những gì được đối chiếu

| Nhãn | Bằng chứng cần có | Kết luận được phép |
|---|---|---|
| `suspected_missing_region` | Đa số cùng có vùng chữ; một model không khớp vùng đó; mốc hai đầu ở model đó gần nhau hoặc output rỗng | Nghi bỏ sót vùng/cột |
| `suspected_duplicate_region` | Các nhóm mốc tạo nhiều bản sao nhất quán ở một model, đa số chỉ có ít bản hơn | Nghi đọc lặp vùng/cột |
| `suspected_reading_order` | Các vùng/mốc chữ có mặt nhưng thứ tự đảo so với đa số | Nghi sai thứ tự đọc |
| `suspected_extra_region` | Một model có đoạn riêng, đa số không có; mốc hai đầu gần nhau trong output đối chứng | Nghi có văn bản dư so với đa số |
| `region_text_mismatch` | Có khung ngữ cảnh tương đương trên ảnh nhưng văn bản không khớp | Chưa kết luận vùng bị bỏ sót |
| `unmatched_region_text` | Đoạn không khớp nhưng thiếu bằng chứng phân biệt thay thế với thêm/bớt | Chỉ rà soát, không gọi là hallucination |

Các nhãn đều là **dấu hiệu**, không phải xác nhận lỗi OCR. Vùng có khung dòng
cũng chỉ là ngữ cảnh do model ước lượng, chưa phải cột được người xác nhận.
Vùng chỉ có mốc chữ được ghi `localization_status="unresolved"`.

## Thuật toán

1. Dùng chuỗi so sánh bỏ khoảng trắng; giữ nguyên chữ, dấu, chữ số và biến thể
   ký tự. Không dùng số chữ Hán để suy ra đủ cột.
2. Tạo đơn vị từ dòng/khối OCR. Ưu tiên đơn vị có tọa độ rõ; khi chỉ có văn bản
   thì dùng dòng/đoạn có giới hạn độ dài. Không coi số dòng hay số block của các
   engine khác nhau là số cột thật.
3. Chọn mốc chữ xuất hiện duy nhất trong output nguồn, rải theo chiều dài vùng.
   Tìm mốc độc lập trong toàn bộ output mỗi model, không buộc cùng thứ tự.
   Các mốc có độ dịch gần nhau tạo một bản khớp; nhiều bản rời nhau mới là dấu
   hiệu lặp. Mốc lặp/không nhất quán giữ trạng thái `ambiguous`.
4. Đối chiếu số lần khớp vùng và thứ tự từng cặp. Ngưỡng là hơn nửa **tất cả model
   cấu hình**: bốn model cần ba phiếu. Model chạm giới hạn token/lặp bất thường
   không tạo vùng tham chiếu hoặc bỏ phiếu cấu trúc, nhưng output vẫn được chẩn đoán.
   Model khác có thể bổ sung vùng mà detector tham chiếu bỏ sót; không đặt PP-OCR
   làm đáp án chuẩn.
5. Trước khi gọi một đoạn là thiếu/dư, tìm mốc duy nhất ngoài hai đầu vùng (hoặc
   mép trang). Nếu khoảng giữa hai mốc ở output đối chiếu vẫn có nhiều chữ,
   giữ nhãn chưa phân biệt thay thế hay dư/thiếu. Nếu khung tương đương trên ảnh
   tồn tại, ưu tiên nhãn chữ không khớp trong vùng.
6. Một lượt mốc toàn trang phát hiện đảo thứ tự cả khi tất cả output là một đoạn
   liền. Cần nhiều cặp đảo với ít nhất bốn mốc khác nhau và khoảng cách đủ lớn;
   không dùng một cụm đảo ngắn để kết luận thứ tự toàn trang.

Đây là thuật toán so sánh bằng mốc chữ, chưa phải detector layout AI mới. Mốc
chuỗi chỉ chứng minh quan hệ văn bản. Chuỗi chữ khác nhau nhiều có thể không
ghép được dù cùng vùng ảnh; hệ thống giữ trạng thái chưa xác định.

## Quan hệ với phiếu ký tự

Các phát hiện thiếu/lặp/dư/thứ tự **chưa sửa** được đa số đối chứng sẽ thêm lý do
`page_*` vào `excluded_models`. Model vẫn có text/raw và bằng chứng riêng nhưng
không bỏ phiếu ký tự ở trang đó. Mẫu số vẫn tính toàn bộ model, không hạ ngưỡng
3/4 thành 2/3. Nhãn thay thế chữ/chưa phân biệt dư thiếu chỉ đưa vào review,
không tự loại toàn bộ output khỏi phiếu.

PP-OCR có thể đã được chuẩn hóa thứ tự cột trước khi căn chỉnh. Chẩn đoán giữ
cả luồng gốc `raw_before_normalization` và luồng so sánh `comparison`. Nếu lỗi
chỉ ở luồng gốc, output sau sửa vẫn được bỏ phiếu. Khi đối chiếu luồng gốc,
output đã sửa của chính model không được tính là nhân chứng độc lập; cần đủ
model khác đối chứng. Không sửa thêm, bỏ chữ trùng hay ghép lại output ở lớp này.

## Xem kết quả

Trong bảng notebook, xem:

- **Cấu trúc trang**: `structural_review`, `unresolved_structure`,
  `no_structural_disagreement_detected`, `insufficient_evidence`, `analysis_limit`.
- **Dấu hiệu cấp trang**: tên model cùng các nhãn cần kiểm tra.
- **Không bỏ phiếu**: những model bị loại khỏi phiếu ký tự và lý do.

Trong HTML mỗi trang, mục **Đồng thuận cấp trang** có model, vùng, model đối
chứng và luồng gốc/luồng đã chuẩn hóa. Mở từng vùng để đọc đoạn chữ, số lần khớp,
offset và tình trạng tọa độ. Có tọa độ và PDF đã xác minh thì ảnh tổng quan vẽ
khung ngữ cảnh đánh số; vùng chỉ có mốc chữ không được bịa khung.

`comparison.jsonl` lưu `page_consensus` gồm các vùng, mốc, spans, số bản khớp,
phiếu đa số, dấu hiệu, mốc hai đầu, cặp đảo thứ tự và giới hạn bằng chứng.
Offsets là vị trí bắt đầu từ 0 trong chuỗi bỏ khoảng trắng, không phải pixel.
Mọi text/raw model tiếp tục nằm trong `raw/<model>.jsonl`.

Để xem lại hai trang log hiện có, trong notebook đặt:

```python
RUN_OCR = False
EXPORTED_LOG_DIR = "data/log_notebook_3"
DATA_DIR = "data"
BOOKS = ["1"]
MAX_BOOKS = 1
PAGES = "1-2"
LIMIT = 2
EXPORT_RESULTS = True
```

Chạy các ô định nghĩa/setup/so sánh/xuất; không nạp model OCR. Kết quả được tạo
riêng trong `WORK_DIR`, không sửa log nguồn. Với PDF khác, thay tên và số trang.

## Giới hạn tài nguyên và bằng chứng

Mặc định: mốc 4 ký tự; tối đa 10 mốc/vùng; vùng ít nhất 12 ký tự; đoạn nguồn dài
được chia ở 80 ký tự; tối đa 20000 ký tự/model/trang và 256 vùng; tối đa 8 lần
xuất hiện/mốc và 64 mốc toàn trang. Mỗi bản khớp cần ít nhất hai mốc nhất quán,
trải ít nhất 65% độ dài đoạn. Mốc hai đầu được tìm trong phạm vi 80 ký tự; gap
không quá max(2, 20% độ dài vùng) mới hỗ trợ dấu hiệu chèn/xóa.
Giới hạn giúp chẩn đoán nhẹ hơn căn chỉnh ký tự và tránh xử lý vô hạn.

Có thể điều chỉnh qua `ADVANCED["page_consensus_options"]`; các giá trị là ngưỡng
kỹ thuật, chưa phải xác suất hoặc ngưỡng đã hiệu chuẩn bằng ground truth.
Đặt `ADVANCED["page_consensus"]=False` chỉ khi cần chạy đối chứng phiên bản cũ.
Khi vượt giới hạn, trạng thái là `analysis_limit`, không tự loại model dựa trên
lượt phân tích chưa đầy đủ. Nhiều chữ lặp tự nhiên có thể không tạo được mốc duy
nhất; không tự kết luận model đọc lặp.

Hai phía 2–2 giữ chưa xác định. Thiếu output khác với output rỗng. **Cả bốn model
cùng bỏ sót một cột thì lớp này không thể phát hiện chỉ từ output**. Phải đối
chiếu với ảnh/layout độc lập hoặc ground truth. Tương tự, văn bản chỉ một model
đọc được có thể là phần đúng bị ba model kia bỏ sót; không coi đa số là sự thật.

## Kiểm chứng thực hiện ngày 2026-11-10

- 23 kiểm thử mới gồm thiếu/lặp/đảo/dư vùng, output một đoạn liền, thay thế chữ,
  thế hòa 2–2, thiếu model, lặp tự nhiên, giới hạn phân tích, tọa độ, output gốc
  trước chuẩn hóa, bỏ phiếu ký tự, notebook tự chứa mã và HTML/ảnh/JSONL.
- Toàn bộ suite: 260 tests, 259 passed và 1 skipped.
- Đọc lại log đã có của PDF 1, trang 1–2, xác minh SHA-256 nguồn và giữ nguyên
  các file log. Trang 1 có bằng chứng thứ tự output PP-OCR gốc khác với ba model
  khác; phần chuẩn hóa đã sửa trước phiếu ký tự. Đoạn Hunyuan không khớp được
  giữ ở nhãn chưa phân biệt thay thế với dư/thiếu, không gọi là hallucination.
- Đã kiểm tra liên kết/ảnh trong HTML, xem ảnh tổng quan và kiểm tra notebook
  sinh mã ổn định. Không có Browser khả dụng nên chưa QA hiển thị HTML trong
  trình duyệt. Không tải weights hay chạy inference OCR mới.
- Cấu hình chọn sách/trang, worker OCR, fingerprint cài dependency và notebook 2
  giữ nguyên. Thay đổi nằm ở bước so sánh và phần trình bày/xuất bằng chứng.
