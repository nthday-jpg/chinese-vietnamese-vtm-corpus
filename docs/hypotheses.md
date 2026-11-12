# Các giả thuyết nghiên cứu

## Các hướng mở rộng xử lý ảnh

Các giả thuyết dưới đây tạo variant độc lập trong Notebook 1, chưa có kết luận
accuracy. Notebook 3 so cùng nguồn/vùng, cùng cấu hình OCR, giữ mọi reading và
đối chiếu thủ công trên original; không chạy lại restoration khi đánh giá.

| Hướng | Giả thuyết và cơ chế | Điều kiện phù hợp | Rủi ro | Kiểm chứng ở Notebook 3 |
|---|---|---|---|---|
| Color-aware Enhancement | Lab L, kênh RGB và tương phản màu có thể tách mực khỏi giấy vàng | Trang có màu/mực đỏ; xem riêng từng kênh | Kênh đỏ làm mực đỏ sáng lên; grayscale mất thông tin màu | So original, Lab và từng kênh; rà cả chữ đỏ lẫn đen |
| Background / Illumination Normalization | Chia luminance cho nền Gaussian để giảm độ sáng biến thiên, giữ chroma | Nền biến thiên chậm hơn bề rộng nét | Halo, khuếch đại nhiễu, mất nét nhạt | So crop có gradient nền, kiểm nét yếu và reading thay đổi |
| Adaptive Binarization | Sauvola/Niblack dùng trung bình và độ lệch chuẩn cục bộ | Độc lập hướng viết; cửa sổ lớn hơn nét nhưng nhỏ hơn biến thiên nền | Xóa nét nhạt, nối nét; mất màu | Giữ cả hai variant, ghi window/k/r; rà riêng dấu đỏ |
| Classical Denoising / Sharpening | NLM/bilateral giảm nhiễu; unsharp nhẹ tăng cạnh | Nhiễu/blur vừa, không yêu cầu crop dòng | Xóa/chẻ/nối nét và halo | So riêng từng phương pháp, original và bicubic |
| Neural Document Restoration | DocRes dùng prompt theo task, DocDiff coarse + residual diffusion | Ảnh trang/vùng, không ràng buộc dòng ngang; Hán cổ chưa được xác minh | Thêm/xóa nét, thay màu, khác miền huấn luyện | Ablation riêng appearance/deblur/deshadow/binarization, không nối chuỗi |
| Deblurring | Restormer motion/defocus và DocRes/DocDiff phục hồi tín hiệu nhòe khác nhau | Chọn đúng loại checkpoint; không coi Gaussian là real noise | Sharpen sai, hallucination, domain shift | Phân tầng loại blur, so original/bicubic và đọc thủ công |
| Bleed-through Reduction | Có thể giảm mực mặt sau bằng phân biệt tín hiệu/nền | Một mặt không đủ phân biệt nét thật; hai mặt cần registration | Xóa chữ thật mờ | Chưa triển khai model chuyên biệt khi chưa đủ source/weights/license; cần nhãn mặt trước/mặt sau |
| Restoration Fidelity | Ảnh rõ hơn có thể giúp OCR nhưng làm sai hình dạng glyph | Mọi restoration; đặc biệt diffusion/generative | Dị thể bị thay, consensus cùng sai | H10: rescue/regression/uncertain trên cùng crop; agreement không là accuracy |

Classical trên không phụ thuộc hướng viết. DocRes/DocDiff/Restormer nhận được
ảnh dọc và nhiều cột nhưng chưa được chứng minh hiệu quả trên Hán cổ. DiffTSR
bị giới hạn dòng/canvas/prior, được bỏ khỏi lựa chọn tự động; xoay cột không giải
quyết vấn đề. UniHIR có prior inscription và canvas danh nghĩa, nhưng không có
ràng buộc một dòng ngang; giữ trong `all_eligible` để thí nghiệm, kèm preflight
và nhãn generative. Không tự tách glyph ở chữ dính hoặc ghép cột nhỏ thành dòng.

Nguồn implementation: [DocRes](https://github.com/ZZZHANG-jx/DocRes),
[DocDiff](https://github.com/Royalvice/DocDiff),
[Restormer](https://github.com/swz30/Restormer),
[Sauvola/Niblack](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_niblack_sauvola.html).

Mỗi thành phần được thử vì một giả thuyết có thể kiểm chứng về nhận dạng hoặc
khả năng xác minh. Có mã chạy được chưa chứng minh giả thuyết đúng. Cần xác định
đối chứng, giữ cùng trang/crop và ghi cả kết quả xấu trước khi xem kết quả tốt.
Các nguyên tắc về nguồn và ground truth được định nghĩa trong [bài toán](problem.md).

Với corpus chưa có bản chép chuẩn, có thể đo thời gian rà soát, tỷ lệ cảnh báo
được người đọc xác nhận và bất đồng, nhưng chưa thể kết luận accuracy. Nên xây
tập nhỏ do hai người đọc độc lập, đối chiếu trên ảnh gốc rồi phân xử; giữ chữ
không đọc được và quy tắc dị thể. Chia theo sách/trang, tránh dùng lại mẫu chọn
ngưỡng làm tập đánh giá; báo mẫu số, loại trang, chi phí và khoảng bất định.
Nếu chưa có người đọc chuyên môn, kết quả chỉ là chẩn đoán hoặc tính đúng của
phần mềm. Các thí nghiệm dưới đây là đề xuất, không phải kết quả đã thực hiện.

## H01 — Multi-model Consensus

**Giả thuyết:** Đối chiếu các model khác kiến trúc có thể tìm nhiều lỗi cần rà
soát hơn việc chỉ dùng một engine ở cùng ngân sách người đọc.

**Cơ sở và ý nghĩa:** Detector, recognizer và VLM có thể bỏ/sai các chữ khác nhau.
Bất đồng cung cấp chỗ để kiểm tra; đồng thuận không chứng minh đúng.

**Cơ chế kỳ vọng:** Bất đồng giữa detector/recognizer và VLM tạo hàng đợi rà soát; đa số chỉ gợi ý cách đọc.

**Phương pháp:** So cấp trang trước để nhận diện thiếu/lặp/đảo khối, rồi căn chỉnh
ký tự và giữ phiếu từng engine. Không coi model cùng họ là nguồn hoàn toàn độc lập.

**Kiểm chứng:** Trên tập chép thủ công, so một engine, bất đồng ký tự và thêm
chẩn đoán cấu trúc; đo lỗi thực tìm được trên mỗi phút rà soát và lỗi cùng sai
còn lọt. Nếu cảnh báo chỉ tăng mà lỗi tìm được không tăng thì giả thuyết chưa
được ủng hộ. Không có bản chuẩn thì chỉ báo tỷ lệ bất đồng, không báo accuracy.

**Giới hạn:** Lỗi tương quan, cùng bỏ cột, alignment sai và thiên lệch neo có thể
tạo đa số sai. Nhiều model hơn có thể chỉ tăng chi phí và phương án nhiễu.

**Trạng thái:** Đã triển khai page/character consensus trong [mã](pipeline.md#các-lớp-đối-chiếu),
có kiểm thử tổng hợp và checkpoint/worker giả; chưa xác nhận inference mới bốn
model trên Kaggle trong lần đối chiếu tài liệu này.

## H02 — Multi-variant Stability

**Giả thuyết:** Thay đổi OCR dưới các biến đổi ảnh có kiểm soát giúp ưu tiên vùng
nhạy với preprocessing, vượt thông tin từ bất đồng giữa model đơn thuần.

**Cơ sở và ý nghĩa:** Nét yếu có thể được tăng tương phản hoặc bị xóa; một engine
đọc đổi khi ảnh đổi là tín hiệu cần xem nguồn, chưa nói lần đọc nào đúng.

**Cơ chế kỳ vọng:** Biến đổi có kiểm soát tác động nét yếu; reading đổi làm lộ độ nhạy của cùng một engine.

**Phương pháp:** So cùng engine trên `original` và từng biến thể của cùng vùng;
ghi pixel budget thực, layout và text. Mỗi biến thể không tạo thêm phiếu độc lập.

**Kiểm chứng:** So danh sách ưu tiên theo stability với consensus trên cùng tập
rà soát, đo lỗi được xác nhận và thời gian. Cố định crop/ngân sách processor hoặc
tách riêng ảnh hưởng resize. Nếu vùng ổn định vẫn sai nhiều, bác bỏ cách dùng
stability như tiêu chí tin cậy. Không có ground truth chỉ đo độ nhạy, chưa đo accuracy.

**Giới hạn:** Mọi biến thể có thể cùng sai; processor thu nhỏ có thể triệt tiêu
khác biệt. Chữ lặp và đổi thứ tự làm căn chỉnh mơ hồ; ảnh sinh có lỗi tương quan.

**Trạng thái:** Đã có multi-variant scheduling, reader extension và stability;
kiểm thử CPU/tổng hợp, worker giả và đọc ảnh đã lưu. Chưa xác nhận nhiều biến thể
với bốn OCR thật trên GPU/Kaggle.

## H03 — Spatial Coverage

**Giả thuyết:** Ứng viên chứa chữ tìm độc lập từ ảnh có thể phát hiện vùng tất cả
engine cùng bỏ, vốn không xuất hiện trong bất đồng text.

**Cơ sở và ý nghĩa:** Nếu không model nào sinh chữ cho một cột, consensus không
có chuỗi để so; tín hiệu mực độc lập bổ sung một kênh phát hiện.

**Cơ chế kỳ vọng:** Tín hiệu mực độc lập với text OCR có thể làm lộ cột mọi model cùng bỏ.

**Phương pháp:** Tạo mask và vùng ứng viên từ original trước, rồi mới đo overlap
với geometry OCR đã quy về nguồn; phân biệt character, coarse và không có geometry.

**Kiểm chứng:** Fixture cố ý bỏ cả cột kiểm tra cơ chế; trên trang thật có nhãn
vùng chữ, đo precision/recall cảnh báo bỏ sót và cảnh báo giả do hình/dấu. Nếu mực
mờ không tạo ứng viên hoặc khung block che mất thiếu sót thì giả thuyết bị giới
hạn tương ứng. Nhãn vùng chỉ đánh giá detection, chưa đủ để tính accuracy ký tự.

**Giới hạn:** Nhiễu, mép giấy, texture và chữ nối làm detector sai. Không có
geometry không được diễn giải là mọi model bỏ chữ. Overlap không xác nhận reading.

**Trạng thái:** Đã triển khai heuristic CPU, kiểm thử tổng hợp có cột cùng bị bỏ
và smoke log/ảnh cũ ngày 11/11/2026; chưa có đánh giá precision/recall trên corpus thật.

## H04 — Hierarchical Layout Analysis

**Giả thuyết:** Mô hình quan hệ trang → vùng chính → chú thích → dòng có thể giảm
đảo thứ tự và mất chú thích so với chỉ cắt theo khe trắng.

**Cơ sở và ý nghĩa:** Cột lớn chứa cột nhỏ không có một tỷ lệ khe cố định; cần
quan hệ bao chứa, vai trò và thứ tự đọc thay vì chỉ tọa độ ngang.

**Cơ chế kỳ vọng:** Quan hệ bao chứa và thứ tự cục bộ có thể giữ được chú thích nhỏ trong vùng chính.

**Phương pháp:** Tạo cây/đồ thị vùng từ ảnh, cho phép chú thích lồng và thứ tự
cục bộ; giữ vùng chưa phân loại, trả về tọa độ nguồn để kiểm tra.

**Kiểm chứng:** Gán nhãn vùng và thứ tự trên trang có chú thích; so phân hoạch
hiện tại với đồ thị đề xuất, đo chú thích bị bỏ và cặp vùng đảo thứ tự. Nếu chỉ
thêm tầng mà thứ tự không tốt hơn thì chưa có lợi ích. Chưa có bản chép chuẩn
thì không suy từ layout tốt hơn ra accuracy OCR cao hơn.

**Giới hạn:** Cây có thể không biểu diễn được chú thích nối nhiều vùng; detector
huấn luyện trên tài liệu hiện đại có thể sai miền chữ cổ, chữ nghiêng và bảng.

**Trạng thái:** Một phần nền tảng: có page/group/column và layout native từng
backend. Chưa có mô hình phân cấp chú thích chung hay đánh giá lợi ích của nó.

## H05 — Overlapping Multi-scale OCR

**Giả thuyết:** Crop chồng lấn ở nhiều mức ngữ cảnh có thể giảm chữ mất nét tại
ranh giới và giảm phụ thuộc vào segmentation chính xác.

**Cơ sở và ý nghĩa:** Chữ dính hoặc nghiêng dễ vượt đường cắt; crop rộng giữ
ngữ cảnh, crop hẹp giữ độ phân giải cho chữ nhỏ.

**Cơ chế kỳ vọng:** Ngữ cảnh rộng và pixel crop nhỏ có thể bù lỗi cắt biên; phải hợp nhất chống lặp.

**Phương pháp:** Chạy cửa sổ chồng lấn và vùng lớn hơn trên cùng ảnh, gộp kết quả
qua tọa độ/mốc; giữ bất đồng thay vì nối chuỗi mù hoặc cộng phiếu cùng engine.

**Kiểm chứng:** Dịch có chủ đích ranh giới crop trên trang đã rà soát, so crop
không chồng với chồng lấn và nhiều scale. Đo chữ biên mất/lặp, sai thứ tự, thời
gian và VRAM. Nếu lỗi lặp tăng vượt số lỗi biên giảm thì giả thuyết không đạt.
Chưa có bản chuẩn chỉ quan sát độ nhạy theo đường cắt, chưa kết luận accuracy.

**Giới hạn:** Gộp chữ lặp khó, ngữ cảnh có thể làm VLM suy diễn; chi phí inference
tăng. Crop nhiều scale không phải nhiều người bỏ phiếu độc lập.

**Trạng thái:** Có page/group/column, nội suy và tile chồng lấn của restoration;
chưa có luồng OCR cửa sổ chồng lấn đa scale cùng cơ chế hợp nhất. Tile restoration
không phải bằng chứng đã triển khai giả thuyết này.

## H06 — Color-aware Processing

**Giả thuyết:** Tách tín hiệu màu có thể phân biệt mực đỏ, mực đen và nền giấy tốt
hơn chỉ dùng grayscale ở các vùng có dấu/chú thích màu.

**Cơ sở và ý nghĩa:** Hai loại mực có thể cùng độ sáng nhưng khác màu; xóa “nhiễu
đỏ” thiếu kiểm tra có thể xóa luôn nội dung thật.

**Cơ chế kỳ vọng:** Khác màu mực bổ sung thông tin phân tách mà grayscale không có.

**Phương pháp:** Phân tích kênh màu hoặc mask mực, tạo biến thể có provenance;
giữ RGB original và cả vùng không phân định, không mặc định dấu đỏ là rác.

**Kiểm chứng:** Chọn vùng có/không có mực màu, gán nhãn lớp mực và chữ cần giữ;
so grayscale với xử lý màu, đo nét bị xóa/giữ nhầm và lỗi OCR xác nhận thủ công.
Không có bản chuẩn chỉ đo phân tách màu và thay đổi OCR, chưa đo accuracy.

**Giới hạn:** Giấy ố, màu scan sai, mực đen bạc màu hoặc dấu đè chữ có thể làm
phân tách không xác định; kết quả có thể tốt ở một sách nhưng sai ở sách khác.

**Trạng thái:** Đã có RGB/Lab channel variants, `red_ink` riêng và normalization
giữ chroma; kiểm thử màu/geometry trên fixture, chưa đánh giá lợi ích OCR trên
Hán cổ. Detector coverage hiện vẫn dùng grayscale.

## H07 — Intra-document Glyph Retrieval

**Giả thuyết:** Tìm hình chữ tương tự trong cùng tài liệu có thể cung cấp bằng
chứng bổ sung cho chữ hỏng hoặc mờ mà không chỉ dựa vào ngữ nghĩa.

**Cơ sở và ý nghĩa:** Cùng phông/người viết có nét lặp lại; một lần xuất hiện rõ
có thể giúp đối chiếu lần khác bị mất nét.

**Cơ chế kỳ vọng:** Hình chữ rõ cùng tay viết/phông cung cấp đối chứng thị giác cho hình chữ hỏng.

**Phương pháp:** Truy hồi glyph, vùng hoặc cửa sổ ảnh theo đặc trưng thị giác;
không yêu cầu cắt ký tự hoàn hảo. Trả top-k ảnh nguồn cùng vị trí, không gán chữ
từ nhãn OCR của láng giềng như sự thật.

**Kiểm chứng:** Che/làm mờ có kiểm soát trên mẫu rõ rồi thử vùng hỏng tự nhiên;
đo tỷ lệ láng giềng thực cùng chữ và thay đổi quyết định của người đọc. Tách mẫu
truy vấn khỏi bản sao cùng ảnh. Thử nghiệm tổng hợp không thay đánh giá chữ hỏng
thật; chưa có nhãn thì chưa thể kết luận accuracy.

**Giới hạn:** Chữ gần hình, dị thể, khác tay viết, nét hỏng và chữ dính có thể
trả về láng giềng sai. Dùng OCR làm nhãn rồi xác nhận bằng chính nhãn đó gây vòng lặp.

**Trạng thái:** Chưa triển khai retrieval/index/ranking; crop/provenance hiện
có chỉ là hạ tầng có thể dùng cho nghiên cứu.

## H08 — Independent Visual Verification

**Giả thuyết:** Model thị giác kiểm tra độc lập có thể phát hiện giả thuyết OCR
không được nét trên ảnh nguồn hỗ trợ.

**Cơ sở và ý nghĩa:** Model nhận dạng có thể ưu tiên chuỗi hợp ngữ cảnh; lần đọc
độc lập giúp kiểm tra xem ứng viên có thực sự quan sát được hay không.

**Cơ chế kỳ vọng:** Lần đọc ảnh không thấy ứng viên có thể hạn chế sự dẫn dắt từ OCR trước đó.

**Phương pháp:** So blind verification (chỉ ảnh, chưa thấy ứng viên) với kiểm tra
có OCR hypotheses. Cho phép “không đọc được”; ghi model, prompt và ảnh đầu vào.
Lần nhìn thấy ứng viên phải được đánh dấu vì có nguy cơ bị gợi ý.

**Kiểm chứng:** Dùng chữ đúng/sai do người đọc xác nhận, chữ rất mờ và ảnh không
chữ; đo chấp nhận ứng viên sai và mức bị dẫn dắt khi đổi hypotheses. Nếu verifier
đồng ý mọi gợi ý hoặc bịa nét thì bác bỏ tính độc lập. Chưa có nhãn chỉ so khác
biệt blind/có gợi ý, chưa xác nhận accuracy.

**Giới hạn:** VLM có thể tự sinh chữ và giải thích thị giác không có căn cứ;
khác prompt không làm cùng model trở thành bằng chứng độc lập. Vẫn cần ảnh gốc.

**Trạng thái:** Chưa có verifier độc lập. Hunyuan/PaddleOCR-VL hiện là OCR engine,
không phải triển khai blind verification chỉ vì có năng lực thị giác.

## H09 — Semantic Validation

**Giả thuyết:** LLM có thể tìm bất thường ngữ nghĩa và đề xuất cách đọc đáng xem,
giúp người đọc tìm lỗi còn lọt qua đối chiếu hình ảnh/văn bản.

**Cơ sở và ý nghĩa:** Sai một chữ có thể làm hỏng liều lượng, phủ định hoặc thuật
ngữ; ngữ cảnh chuyên môn có thể chỉ ra nơi cần kiểm tra.

**Cơ chế kỳ vọng:** Bất thường ngữ nghĩa gợi vị trí cần nhìn lại, không tự xác nhận hay thay ký tự.

**Phương pháp:** Sinh cảnh báo và alternative riêng, liên kết về nguồn; không
tự ghi đè bản Hán. Muốn chấp nhận sửa phải có bằng chứng thị giác và quyết định rà soát.

**Kiểm chứng:** Trộn văn bản đúng có cách dùng cổ/hiếm với lỗi cố ý và lỗi thật;
đo cảnh báo đúng, sửa sai văn bản vốn đúng và thời gian người đọc. Nếu model
hiện đại hóa câu cổ hoặc chỉ làm câu trôi chảy hơn thì chưa đạt. Không có ground
truth chưa thể coi tính hợp nghĩa là accuracy.

**Giới hạn:** Kiến thức hiện đại, thuật ngữ hiếm và lỗi tương quan với OCR VLM
có thể làm model “sửa” chữ đúng. Bản dịch không phải đối chứng độc lập.

**Trạng thái:** Chưa có semantic validator hoặc luồng tự sửa. Model dịch và
prompt dịch hiện có không được tính là triển khai giả thuyết này.

## H10 — Restoration Utility vs. Hallucination

**Giả thuyết có thể bác bỏ:** Với cùng vùng nguồn và cùng cấu hình OCR, một
phương pháp restoration có thể cứu thêm cách đọc sai so với original **và**
bicubic cùng độ phân giải, nhưng có thể gây regression do thêm/xóa/nối nét.
Lợi ích cần được người đọc xác nhận, phân tầng theo phương pháp và loại nguồn.

**Cơ sở và ý nghĩa:** Ảnh sắc hơn có thể kém trung thành hơn. OCR/VLM và model
restoration có prior tương quan, vì vậy nhiều model cùng đổi sang một reading
khác vẫn chưa xác nhận reading đó đúng. Tăng agreement không phải tăng accuracy.

**Cơ chế kỳ vọng:** Tăng tương phản hoặc phục hồi nét yếu có thể giúp detector và
recognizer; generative prior cũng có thể thay dị thể bằng chữ phổ biến hoặc làm
mất nét. Processor thu nhỏ ảnh ×4 có thể xóa lợi ích hoặc tạo yếu tố gây nhiễu.

**Phương pháp triển khai:**

1. Chọn trước sách/trang/region; lưu source hash, geometry, pixel hash, variant,
   model/checkpoint và cấu hình. So trong cùng engine, prompt, processor, decoding
   và reading order; giữ kết quả khác crop/config ở trạng thái không so được.
2. So original, CLAHE, denoised, bicubic ×2/×4 và neural đã lưu. Neural ×4 cần
   đối chứng bicubic ×4; thiếu đối chứng phải ghi rõ, không thay bằng ảnh khác.
   Với DiffTSR có canvas đặc biệt cần đối chứng đúng kích thước; không gọi ×4
   khi kích thước thực không khớp. UniHIR có mapping danh nghĩa không đủ để so nét.
3. Từ checkpoint, ghi thay/chèn/xóa, chuỗi đổi, agreement giữa các engine với mẫu
   số, stability trong từng engine, đồng đổi reading và tăng/giảm bất đồng.
   Ghi prepared/worker/processor size, thời gian OCR/restoration nếu quan sát được.
4. Chỉ so cấu trúc ảnh khi affine hợp lệ, cùng vùng nguồn và bằng chứng alignment
   đủ. Đưa ảnh về lưới nguồn; dùng bicubic để phân biệt tác động nội suy với thay
   đổi đáng nghi. Edge, contrast và stroke warnings là heuristic, không là nhãn đúng/sai.
5. Báo cáo HTML offline đặt original, bicubic và neural của cùng vùng cạnh nhau,
   cùng text từng engine, khác biệt và cảnh báo. Người đọc gán `rescued`,
   `regressed`, `unchanged`, `uncertain`; lưu/đọc lại nhãn không inference.
   Ưu tiên tùy chọn giấu tên phương pháp để giảm thiên lệch.

**Kiểm chứng và điều kiện bác bỏ:** Dùng tập chép thủ công độc lập và tập suy giảm
có kiểm soát, sau đó đối chiếu scan hỏng thật. Giữ cố định quy tắc dị thể và tách
mẫu hiệu chuẩn khỏi mẫu đánh giá. Báo số rescue/regression/uncertain, mẫu số và
cách lấy mẫu; nhóm vùng bất đồng không đại diện toàn corpus. Nếu chỉ tăng độ sắc
hoặc agreement, hoặc regression bù hết rescue, chưa có bằng chứng ủng hộ lợi ích.
Nguồn không đọc được thì giữ `uncertain`, không dùng ảnh sinh để tự chứng minh nét.

**Giới hạn và rủi ro:** Suy giảm tổng hợp không mô phỏng đủ mất giấy/mực;
interpolation, sharpening và khử nhiễu cũng đổi edge. Thước đo không xác nhận
nét chữ tồn tại. Canvas/scale đúng không chứng minh alignment nội dung của ảnh
sinh. Mọi reading thô phải giữ nguyên, không tự sửa theo ngữ nghĩa hay bản dịch.

**Trạng thái và bằng chứng:** Đã có classical, adapter neural/generative,
provenance và variant stability. Đã thêm [Restoration Ablation, Visual Fidelity](../src/vtm_corpus/restoration_evaluation.py)
và nhãn review trong HTML/JSONL hiện có; test CPU kiểm cặp có/thiếu engine,
geometry/config/pixel mismatch, warning tổng hợp, export và đọc nhãn. Đây là
bằng chứng phần mềm, chưa phải kết quả thực nghiệm H10 trên corpus.
TADiSR/DiffTSR/UniHIR chưa xác nhận inference thật; chưa chứng minh tăng accuracy.

## Tài liệu tham khảo đã xác minh nguồn

[LayoutParser (2021)](https://arxiv.org/abs/2103.15348) cung cấp bộ công cụ phân
tích tài liệu; có thể tham khảo cách tổ chức thành phần cho H04. Đây chưa phải
bằng chứng mô hình phân cấp phù hợp chú thích chữ Hán của corpus này.

[The Perception-Distortion Tradeoff (2018)](https://arxiv.org/abs/1711.06077)
phân tích xung đột giữa chất lượng cảm nhận và distortion trong phục hồi ảnh.
Suy luận áp dụng ở H10 là cần kiểm tra độ trung thành riêng; bài báo không đo
accuracy OCR trên dữ liệu của dự án.

[ICDAR2019HDRC](https://tc11.cvc.uab.es/datasets/ICDAR2019HDRC_1) là nguồn dữ liệu
tài liệu gia phả chữ Hán lịch sử cho nghiên cứu detection/recognition. Cần khảo
sát quyền dùng, annotation và khác biệt miền trước thí nghiệm; chưa tải, tích
hợp hoặc chạy benchmark dataset này trong repository ở lần chỉnh tài liệu.

## Ưu tiên phát triển đề xuất

Ưu tiên trước hết là tập rà soát nhỏ có nhãn và đo chi phí người đọc. Bảng dưới
là quyết định nghiên cứu đề xuất, không phải thứ hạng hiệu quả đã đo. “Bằng chứng”
phân biệt mức triển khai cơ chế với bằng chứng lợi ích trên corpus.

| Hướng | Ưu tiên | Lợi ích kỳ vọng | Chi phí tương đối | Độ chắc chắn của bằng chứng hiện có |
|---|---|---|---|---|
| H01 | Cao | Ưu tiên bất đồng | Vừa–cao | Cơ chế đã kiểm thử; lợi ích accuracy chưa rõ |
| H03 | Cao | Tìm vùng cùng bị bỏ | Thấp–vừa | Fixture xác nhận cơ chế; trang thật chưa có precision/recall |
| H02 | Cao | Tìm vùng nhạy xử lý ảnh | Vừa–cao | Có kiểm thử; chưa có thí nghiệm GPU nhiều biến thể |
| H04 | Cao | Giữ chú thích/thứ tự đọc | Cao | Mới có phân hoạch cơ bản |
| H05 | Vừa | Giảm lỗi biên crop | Vừa–cao | Chưa triển khai phép gộp OCR |
| H06 | Vừa | Phân biệt mực màu | Thấp–vừa | Chưa đánh giá |
| H07 | Vừa | Thêm đối chứng trong sách | Vừa–cao | Chưa triển khai |
| H08 | Vừa | Phát hiện reading thiếu hỗ trợ | Cao | Chưa triển khai; rủi ro bị gợi ý |
| H09 | Sau xác minh thị giác | Tìm bất thường nội dung | Vừa | Chưa triển khai; rủi ro sửa quá tay |
| H10 | Cao | Đối chứng lợi ích/rủi ro restoration | Cao–rất cao | Ablation/fidelity/manual review đã có test CPU; lợi ích và hallucination trên corpus chưa được đo |
