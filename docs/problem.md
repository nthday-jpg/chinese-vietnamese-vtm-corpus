# Bài toán và giới hạn

## Ba nhóm đã quan sát trên ảnh thật

Khảo sát 12/11/2026 ở HEAD `6dd370b`: 15 cặp tên sách/trang, 13 trang nguồn độc
lập. G1 (`1`–`6`) là scan màu hai trang viết dọc, nét biến thiên như viết tay,
chữ nhỏ và dấu đỏ, giấy xám vàng/rãnh gáy/viền sờn. G2 (`bien`) là bản in ngang,
nền gần trắng, chữ nhỏ và đốm lề. G3 (`kinh`) là bản in dọc có khung/cột chú giải,
nhiều pixel đen/trắng bão hòa. Không mặc định chúng cần cùng restoration.

`2.pdf`/`3.pdf` trùng SHA-256; `4.pdf`/`5.pdf` cũng trùng. Không tính alias thành
nguồn độc lập. Phân nhóm dựa trên ảnh và hash, không chỉ tên. Chưa xác nhận
bleed-through hoặc phân biệt motion/defocus/JPEG bằng chứng đủ mạnh.
Kích thước scan thật G1 chỉ 600–650 pixel cạnh dài; `bien` scan 736×1120,
`kinh` ảnh nhúng chính 1472×2144, canvas canonical profile hiện có 1595×2206.
Pixel render không đồng nghĩa số pixel chi tiết nguồn.

Xem [báo cáo khả thi](../reports/restoration_feasibility.md) và manifest mẫu
để đọc giới hạn lấy mẫu, ảnh đại diện và cơ sở từng nhận xét. Không có nhãn OCR
chuẩn; A là khảo sát Utility qua output/vùng/cost, B là prediction stability
trong engine theo cơ chế. Cả hai không tự đo accuracy.

Corpus gồm tám bộ sách, bảy bộ chủ yếu chữ dọc. Vùng một cột lớn phía trên và
hai cột chú thích nhỏ phía dưới được giữ nguyên nếu không có đường cắt đáng tin
cậy. Phục hồi ảnh tổng quát không cần tách glyph. Xoay cả cột 90° làm xoay từng
chữ, không tạo dòng ngang đúng phân phối huấn luyện DiffTSR. Model này được loại
khỏi lựa chọn/tải tự động; khả năng đọc dataset cũ vẫn giữ nguyên.

Đầu vào là PDF scan sách chữ Hán; đầu ra mong muốn là bản chép Hán có bằng chứng
để rà soát và bản dịch tiếng Việt theo thứ tự nội dung. Với sách Y học cổ truyền,
một chữ sai có thể đổi tên thuốc, liều lượng hoặc phủ định, nên không thể lấy
văn phong tự nhiên làm tiêu chí thay cho tính trung thành.

## Dữ liệu gây khó ở đâu?

Sách có thể chứa chữ in và viết tay, phồn thể và dị thể. Văn bản dọc thường đọc
cột phải trước, từ trên xuống; trang khác có chữ ngang hoặc bố cục bất quy tắc.
Một cột lớn có thể chứa nhiều cột chú thích nhỏ. Vì vậy, vùng hình học tìm bằng
khoảng trắng chưa chắc tương ứng một đơn vị đọc, còn thứ tự OCR sai có thể làm
câu mất nghĩa dù từng chữ được nhận dạng đúng.

Chữ nghiêng, dính nét và nét giao nhau khiến ranh giới ký tự không rõ. Dấu đỏ,
vết mực, mực xuyên giấy, lem mực, nhiễu nền, chữ mất nét và chữ mờ có thể bị nhận nhầm hoặc bị bỏ qua. Trang
cong, nghiêng, hư hại, thủng giấy hoặc scan mất một phần chữ còn làm thiếu thông
tin thật. Đây là các tình huống phải kiểm tra theo trang; không phải mọi loại
đã được thống kê hay có detector được kiểm chứng trên corpus.

Phục hồi ảnh có thể làm nét dễ nhìn hơn nhưng cũng có thể thêm, xóa hoặc nối nét.
Nếu ảnh nguồn không đủ thông tin, rất khó phân biệt nét thật đã mất với nét do
model sinh ra. Tăng DPI hoặc nội suy cũng không tạo lại chi tiết đã mất trong scan.

## Thiếu ground truth có nghĩa gì?

Ground truth là bản chép tham chiếu đã được kiểm tra đáng tin cậy. Corpus mục
tiêu chưa có bản này nên không thể trực tiếp tính OCR accuracy hay CER
(tỷ lệ lỗi ký tự so với bản đúng). Khoảng cách giữa hai bản OCR chỉ đo bất đồng
giữa chúng. Hai model có thể cùng sai; mọi model có thể cùng bỏ một cột.

Mục tiêu vận hành là **trung thành với nguồn, truy xuất được bằng chứng thị giác
và biểu diễn được sự không chắc chắn**. Các chỉ báo thay thế (proxy) giúp ưu tiên
rà soát, không thay phép đo đúng/sai. Muốn kết luận phương pháp tăng accuracy,
cần tập đánh giá được chép/đối chiếu thủ công, quy tắc xử lý dị thể rõ ràng và
so sánh có kiểm soát. Tập nhỏ chỉ cho kết luận trong phạm vi đã đánh giá.

## Bốn công việc cần tách biệt

1. **Nhận dạng chữ:** OCR tạo giả thuyết văn bản từ ảnh, có thể kèm geometry
   (tọa độ vùng) và thứ tự đọc.
2. **Xác minh bằng hình ảnh:** đối chiếu giả thuyết với nét quan sát được trên
   ảnh nguồn; crop ngữ cảnh hoặc khung dòng không tự xác nhận từng chữ.
3. **Suy luận ngữ nghĩa:** phát hiện câu lạ và đề xuất cách đọc khác. Sự hợp lý
   về nghĩa không cấp quyền sửa bản chép khi thiếu bằng chứng thị giác.
4. **Dịch nội dung:** diễn đạt bản Hán đã chọn sang tiếng Việt; lỗi dịch cần được
   phân biệt với lỗi OCR, layout và xử lý ảnh.

Ảnh/PDF gốc là nguồn có thẩm quyền. Phải giữ phồn thể, dị thể và mọi phương án
chưa phân định cùng provenance (nguồn gốc và lịch sử biến đổi). “Không đủ thông
tin để đọc” là kết quả hợp lệ. Bản dịch hoặc ảnh phục hồi không được quay lại
làm bằng chứng độc lập xác nhận bản chép.

Xem [pipeline](pipeline.md) để biết khả năng hiện có và [giả thuyết](hypotheses.md)
để biết cách kiểm tra những cải tiến chưa được chứng minh.
