# Hướng dẫn nộp bài giữa kỳ

Mỗi nhóm làm trên bộ sách của nhóm mình, tức thư mục `data/<tên nhóm trưởng>/`. Mỗi nhóm được chấm riêng và **không so điểm số giữa các nhóm**, vì sách của mỗi nhóm khó dễ khác nhau: có sách chép tay, có sách khắc in, có sách sắp chữ hiện đại, bản scan có cuốn nét có cuốn mờ.

Điều quan trọng nhất là nhóm **hiểu kết quả của mình**: chỗ nào tốt, chỗ nào sai và vì sao. Kết quả chưa cao nhưng nhận xét đúng, có ví dụ vẫn được điểm cao. Số đo đẹp mà không giải thích được thì không được điểm cao.

## 1\. Tài liệu gửi thêm

Ba notebook `1_ocr_hunyuan`, `1_ocr_paddle`, `2_dich_han_viet` đã gửi trước đó giữ nguyên, không thay đổi. Đợt này có thêm:

| File | Dùng để |
| :---- | :---- |
| `golden_<tên nhóm trưởng>.zip` | Dữ liệu tham chiếu riêng của nhóm, dùng để tự đo kết quả |
| `3_tao_input_test_chuoi.ipynb` | Lấy OCR của nhóm cho các trang chấm dịch, xem mục 3 |
| `4_danh_gia.ipynb` | Đo kết quả OCR và dịch so với dữ liệu tham chiếu, xem mục 5 |
| `vi_du_nop_bai/Nguyen_Van_A/` | Mẫu cấu trúc thư mục khi nộp |

Bên trong `golden_<tên nhóm trưởng>.zip` có:

```
golden_<tên nhóm trưởng>/
├── golden.jsonl              # chữ Hán tham chiếu của MỌI trang trong các sách của nhóm
├── dich_chuan/<cuốn>.jsonl   # bản dịch tham chiếu của một cuốn, gọi là "cuốn chấm dịch"
├── test_dich_zh/<id>.txt     # chữ Hán của từng trang trong cuốn chấm dịch, dùng làm input để dịch
└── README.md
```

Trong toàn bộ hướng dẫn này, `<id>` có dạng `<tên sách>__p<số trang>`, số trang đếm từ 1 theo file PDF.

Dữ liệu tham chiếu được tạo tự động bằng model lớn nên **không phải đáp án tuyệt đối**. Để dùng trên Kaggle, tạo một Dataset từ file zip này bằng *Create Dataset*, sau đó *Add Input* dataset đó vào notebook khi cần.

## 2\. Bài nộp

Nén thành một file `<tên nhóm trưởng>.zip` với cấu trúc sau:

```
<tên nhóm trưởng>/
├── txt/<tên sách>.txt        # Output của notebook OCR, giữ nguyên
├── ocr_raw/<tên sách>/...    # Output của notebook OCR, giữ nguyên, chứa kết quả từng trang để đo OCR
├── dich/<tên sách>.txt       # Output của notebook dịch: bản dịch từ OCR của nhóm
├── test_dich/<id>.txt        # cuốn chấm dịch, dịch từ chữ Hán tham chiếu, xem mục 3
├── test_chuoi/<id>.txt       # cuốn chấm dịch, dịch từ OCR của nhóm, xem mục 3
├── bao_cao.pdf               # báo cáo, xem mục 4
└── notebooks.txt             # link các notebook Kaggle đã chạy, để chế độ chia sẻ cho GV xem được
```

Lưu ý:

1. `<tên sách>` giữ đúng tên mà notebook đã sinh ra. Dấu cách và ký tự đặc biệt trong tên đã được đổi thành `_`.  
2. Nếu một sách phải chạy qua nhiều phiên Kaggle, gộp các thư mục `ocr_raw/` của các phiên vào chung một thư mục. Các file `*_shard*.jsonl` không trùng tên nên chép chung được.  
3. `dich/` gồm bản dịch của mọi sách, trong đó có cả cuốn chấm dịch.  
4. Nhóm nào tự làm pipeline khác notebook mẫu thì thay `ocr_raw/` bằng file `ocr_pages.jsonl`. Mỗi dòng của file là một trang, theo dạng `{"id": "<tên sách>__p<số trang>", "zh": "<chữ Hán của trang>"}`, giống hệt dạng của `golden.jsonl`.

## 3\. Dịch cuốn chấm dịch

Cuốn chấm dịch được dịch **hai lần**, mỗi trang ra một file. Cả hai lần đều phải dùng **đúng pipeline nhóm dùng cho cả sách**, tức cùng model, cùng prompt, cùng tham số.

| Thư mục nộp | Dịch từ | Cho biết điều gì |
| :---- | :---- | :---- |
| `test_dich/` | `test_dich_zh/`, là chữ Hán tham chiếu có sẵn trong file zip | Chất lượng của riêng bước dịch, khi chữ Hán đầu vào gần đúng |
| `test_chuoi/` | `test_chuoi_zh/`, là OCR của chính nhóm cho đúng các trang đó | Chất lượng của cả quy trình OCR rồi dịch |

So `test_chuoi` với `test_dich` sẽ thấy lỗi OCR làm bản dịch kém đi bao nhiêu.

Các bước thực hiện:

1. **Tạo `test_chuoi_zh/`.** Mở `3_tao_input_test_chuoi.ipynb`, không cần bật GPU. *Add Input* hai thứ: dataset `golden_<tên nhóm trưởng>` và Output của notebook OCR của nhóm. Chọn *Run All*, chạy xong thì *Save Version*.  
2. **Dịch lần thứ nhất.** Mở `2_dich_han_viet`, chỉ *Add Input* dataset `golden_<tên nhóm trưởng>`. Thư mục `dich/` trong Output chính là bản `test_dich`.  
3. **Dịch lần thứ hai.** Mở `2_dich_han_viet`, chỉ *Add Input* Output của bước 1\. Thư mục `dich/` trong Output chính là bản `test_chuoi`.  
4. Khi nộp, đổi tên hai thư mục `dich/` ở bước 2 và bước 3 lần lượt thành `test_dich/` và `test_chuoi/`. Giữ nguyên tên các file `<id>.txt` bên trong.

Mỗi lần dịch chỉ được *Add Input* đúng một nguồn, vì notebook dịch đọc tất cả file `.txt` có trong Input.

Notebook dịch tự bỏ qua những file có dưới 50 chữ Hán. Vì vậy trang nào OCR quá hỏng sẽ không có bản dịch trong `test_chuoi/`, và trang đó được tính chrF++ bằng 0\. Đây cũng là một kết quả đáng nhận xét trong báo cáo.

## 4\. Viết báo cáo

Báo cáo không giới hạn số trang và không cần theo mẫu nào. Mỗi nhận xét cần **kèm ví dụ cụ thể**: ảnh trang hoặc đoạn gốc, kết quả của nhóm, dữ liệu tham chiếu tương ứng, và chỉ rõ chỗ sai. Báo cáo cần có đủ sáu phần sau.

**Phần 1\. Quy trình và lý do lựa chọn.** Nêu model OCR, model dịch, các tham số chính, những chỗ khác với notebook mẫu, thời gian chạy của từng sách và sách nào chưa chạy xong. Với **mỗi lựa chọn**, chẳng hạn chọn model nào, đặt tham số bao nhiêu, thêm bước xử lý nào, hãy nói rõ **vì sao chọn như vậy** và dựa vào bằng chứng gì: số đo, ảnh trang, hay một ví dụ lỗi cụ thể. Lựa chọn nào giải thích được bằng bằng chứng sẽ được cộng điểm, xem mục 6\.

**Phần 2\. Kết quả OCR.**

1. Bảng CER của từng sách và CER tổng, lấy từ `4_danh_gia.ipynb`.  
2. Kiểm chứng số đo: lấy 5 trang có CER cao nhất, mở ảnh trang, xác định lỗi chủ yếu do OCR của nhóm hay do chữ Hán tham chiếu sai, mỗi trang kèm bằng chứng.&nbsp;  
3. Các loại lỗi hay gặp, mỗi loại kèm một ví dụ. Ví dụ: nhầm chữ có hình gần giống, đọc sai thứ tự cột, mất dòng, lọt số trang hay tiêu đề vào nội dung.

**Phần 3\. Kết quả dịch.**

1. Số đo của `test_dich` và `test_chuoi`: chrF++, tỉ lệ rò chữ Hán, tỉ lệ độ dài.  
2. Đọc bản dịch đặt cạnh bản dịch tham chiếu, chỉ ra chỗ dịch tốt, chỗ sai nghĩa, chỗ dịch sót hoặc tự thêm lời.  
3. Nhận xét thuật ngữ: tên huyệt, vị thuốc, bài thuốc, tên bệnh có được giữ đúng âm Hán Việt và dịch nhất quán trong cả cuốn không.

**Phần 4\. Thử nghiệm và cải tiến.**

1. **Mốc ban đầu.** Chạy notebook mẫu, để nguyên cấu hình mặc định, rồi ghi lại CER và chrF++ của `test_dich`. Có thể chạy trên cả cuốn chấm dịch hoặc chỉ 10 đến 20 trang của cuốn đó, miễn là các lần sau đo trên đúng các trang ấy.  
2. **Ít nhất hai lần thử thay đổi.** Ví dụ: đổi DPI, đổi model OCR, sửa prompt, đổi model dịch. Mỗi lần thử ghi rõ: đã đổi **một yếu tố** nào; so với lần chạy nào, tức lần đối chứng; chạy trên những trang nào; số đo trước và sau, có thể ghi thêm thời gian chạy; kết luận giữ hay bỏ thay đổi đó, và **vì sao** kết quả lại như vậy. Lần thử thất bại, tức không tốt hơn hoặc bị lỗi, vẫn được tính điểm nếu giải thích được nguyên nhân.  
3. **Kết quả cuối so với mốc.** Pipeline nhóm chọn để làm bài nộp tốt hơn mốc ban đầu bao nhiêu, đo trên cùng các trang. Càng tiến gần dữ liệu tham chiếu càng tốt. Nhóm giữ nguyên notebook mẫu cũng được, chỉ không có điểm cải tiến.

**Phần 5\. Trả lời bốn câu hỏi.**

1. Trang OCR tệ nhất của nhóm là trang nào? Mở ảnh trang ra thấy gì, vì sao model đọc sai?  
2. Bản dịch có còn sót chữ Hán hoặc bị trôi sang tiếng Trung không? Nếu có thì ở đâu và nhóm đã xử lý thế nào. Nếu không thì nhờ đâu.  
3. So `test_dich` với `test_chuoi`, lỗi OCR làm bản dịch kém đi ở những chỗ nào? Cho ví dụ cụ thể.  
4. Nếu có thêm một tuần GPU, nhóm sẽ thử gì tiếp theo và vì sao?

**Phần 6\. Phân công.** Ai làm phần nào. Mỗi thành viên viết ba đến năm câu về điều mình rút ra được.

## 5\. Tự đo kết quả

Chạy `4_danh_gia.ipynb`, không cần bật GPU. *Add Input* dataset `golden_<tên nhóm trưởng>`, Output của notebook OCR, và hai Output dịch ở mục 3\. Thiếu Input của phần nào thì notebook tự bỏ qua phần đó. Notebook in ra các số đo và ghi kết quả của từng trang vào hai file `cer_trang.csv` và `dich_trang.csv`. **GV chấm bằng đúng notebook này.**

Muốn đo một lần thử chỉ chạy trên vài trang thì chạy notebook với Output của lần thử đó, rồi đọc kết quả từng trang thay vì số tổng, vì các trang chưa chạy đều bị tính là sai hết:

* **OCR:** mở `cer_trang.csv`, lọc những dòng có cột `co_ocr` bằng True.  
* **Dịch:** mở `dich_trang.csv`, lọc những dòng có cột `ti_le_do_dai_test_dich` lớn hơn 0 (tức trang đã dịch), rồi lấy trung bình cột `chrf_test_dich`. Làm tương tự với các cột `..._test_chuoi`.

Mốc ban đầu và các lần thử phải đo trên **cùng các trang** và so bằng cách đọc này.

**Dữ liệu tham chiếu không phải đáp án tuyệt đối.** Chữ Hán tham chiếu do một model OCR lớn đọc, bản dịch tham chiếu do một model ngôn ngữ lớn dịch từ chính chữ Hán đó. Cả hai chạy tự động nên vẫn có lỗi, nhất là ở trang viết thảo, chữ chú thích nhỏ và bản khắc mờ. Khi số đo xấu, hãy mở ảnh trang và đọc lại bản dịch để xem bên nào sai, nhóm mình hay dữ liệu tham chiếu. Chỉ ra được chỗ dữ liệu tham chiếu sai cũng là một kết quả tốt và được cộng điểm.

**Đo OCR**, so với `golden.jsonl` trên mọi trang sách của nhóm:

1. **CER** là số ký tự phải sửa để OCR trở thành chữ tham chiếu, gồm thêm, xoá và thay, chia cho số ký tự của chữ tham chiếu. CER càng thấp càng tốt.  
2. Khi tính không xét khoảng trắng, dấu câu, và không phân biệt chữ giản thể với phồn thể.  
3. Trang nhóm chưa OCR được tính là sai hết. Trang mà chữ tham chiếu để trống thì bỏ qua.  
4. Notebook in CER của từng sách, CER tổng và 10 trang có CER cao nhất.

**Đo dịch**, so `test_dich/` và `test_chuoi/` với `dich_chuan/`:

1. **chrF++** đo mức khớp với bản dịch tham chiếu theo chuỗi ký tự và theo từ, thang từ 0 đến 100, càng cao càng tốt. Chỉ nên dùng chrF++ để so các lần chạy với nhau, vì bản tham chiếu cũng có lỗi và một câu có thể có nhiều cách dịch đúng.  
2. **Tỉ lệ rò chữ Hán** là phần trăm ký tự chữ Hán còn sót trong bản dịch. Mục tiêu là 0\.  
3. **Tỉ lệ độ dài** là số âm tiết của bản dịch chia cho số âm tiết của bản tham chiếu. Dưới 0,7 có thể đã dịch sót, trên 1,5 có thể đã tự thêm lời hoặc bịa.  
4. Notebook in thêm 5 trang có chrF++ thấp nhất và 5 trang mà `test_chuoi` kém `test_dich` nhiều nhất.

Các số đo dùng để đối chiếu với nhận xét trong báo cáo và để xem nhóm tiến bộ bao nhiêu so với mốc ban đầu của chính nhóm. Không so số đo giữa các nhóm.

## 6\. Cách tính điểm

Tổng điểm là 10, gồm ba phần A, B, C và điểm cộng.

**A. Nộp bài: 1 điểm.** Nộp đủ sách, đúng cấu trúc ở mục 2, có `test_dich/` và `test_chuoi/`, chạy được `4_danh_gia.ipynb` trên bài nộp, link notebook mở xem được.

**B. Báo cáo: 5 điểm.** Gồm 10 ý, mỗi ý 0,5 điểm:

| Ý | Nội dung |
| :---- | :---- |
| 1 | Quy trình rõ ràng, mỗi lựa chọn đều có nêu lý do |
| 2 | Bảng CER từng sách khớp với số của `4_danh_gia.ipynb` |
| 3 | Kiểm chứng số đo OCR: lấy 5 trang có CER cao nhất, mở ảnh trang, xác định lỗi chủ yếu do OCR của nhóm hay do chữ Hán tham chiếu sai, mỗi trang kèm bằng chứng.&nbsp; |
| 4 | Phân loại lỗi OCR, mỗi loại có ví dụ |
| 5 | Nhận xét bản dịch khi đặt cạnh bản tham chiếu, có ví dụ |
| 6 | Nhận xét về thuật ngữ: âm Hán Việt và tính nhất quán |
| 7 | Câu hỏi 1: có mở ảnh trang tệ nhất và giải thích đúng nguyên nhân |
| 8 | Câu hỏi 2: rò chữ Hán và cách xử lý |
| 9 | Câu hỏi 3: so `test_dich` với `test_chuoi`, có ví dụ cụ thể |
| 10 | Câu hỏi 4 và phần rút ra của từng thành viên có nội dung thật, không chung chung |

C. Thử nghiệm và cải tiến: 4 điểm.

1. Hai lần thử, mỗi lần 1,5 điểm. Trong đó: đổi đúng một yếu tố được 0,25; có lần đối chứng và số đo trước, sau được 0,5; giải thích được vì sao ra kết quả đó được 0,75. Lần thử thất bại mà giải thích đúng vẫn được đủ điểm.  
2. Cải tiến so với mốc ban đầu, 1 điểm, chia đều cho OCR và dịch, mỗi phần 0,5: OCR tốt hơn mốc được 0,25. CER giảm từ 20% trở lên so với mốc thì được 0,5, ví dụ CER từ 0,10 xuống 0,08 hoặc thấp hơn. Dịch tốt hơn mốc được 0,25. chrF++ của `test_dich` tăng từ 3 điểm trở lên và tỉ lệ rò chữ Hán dưới 1% thì được 0,5.

**Điểm cộng: tối đa 1 điểm.** Tổng điểm không vượt quá 10\.

| Việc làm được | Điểm cộng |
| :---- | :---- |
| Một lựa chọn trong quy trình được giải thích bằng bằng chứng như số đo, ảnh trang, ví dụ lỗi | 0,25 mỗi lựa chọn, tối đa 0,5 |
| Chỉ ra chỗ dữ liệu tham chiếu sai, có ảnh trang chứng minh | 0,25 mỗi mục |
| Phát hiện lỗi trong dữ liệu được giao | 0,25 mỗi mục |
| Làm pipeline khác hẳn notebook mẫu, xem gợi ý ở cuối `FLOW.md` | 0,25 |
| Xây bảng thuật ngữ để dịch nhất quán | 0,25 |

&nbsp;