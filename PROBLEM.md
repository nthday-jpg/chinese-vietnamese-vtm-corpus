# Bài toán: Dịch sách Y học cổ truyền từ chữ Hán sang tiếng Việt

Nhiều sách Y học cổ truyền chữ Hán chỉ có dưới dạng **bản scan**: mỗi trang là một tấm ảnh, không copy được chữ.
Bài toán là biến một cuốn sách như vậy thành **bản dịch tiếng Việt dạng văn bản**, gồm 2 bước:

```
            Bước 1: OCR                    Bước 2: Dịch
PDF scan ──────────────────► txt chữ Hán ──────────────────► txt tiếng Việt
```

|   | Input | Output |
|---|---|---|
| **Cả bài toán** | 1 hoặc nhiều file PDF scan sách chữ Hán | Mỗi PDF 1 file `.txt` bản dịch tiếng Việt |
| **Bước 1 — OCR** | PDF scan | `.txt` chữ Hán (mỗi PDF 1 file) |
| **Bước 2 — Dịch** | `.txt` chữ Hán của bước 1 | `.txt` tiếng Việt (mỗi file 1 bản dịch) |

## Bước 1 — OCR: PDF scan → txt chữ Hán

**Input**
- Các file `.pdf` trong `data/`, mỗi file là 1 cuốn sách (hoặc 1 phần cuốn sách). Tên file là tên sách, vd `dong_thi_ky_huyet.pdf`.
- Mỗi trang là ảnh scan, PDF không có lớp chữ.
- Trang sách có thể là: chữ phồn thể hoặc giản thể, viết ngang hoặc **viết dọc** (đọc từ phải sang trái), 1 hoặc 2 cột,
  có hình vẽ, bảng, watermark, số trang, tiêu đề chạy ở đầu trang.

**Output**
- `txt/<tên sách>.txt` — mã hoá UTF-8, chỉ chứa chữ Hán của phần nội dung chính:
  - đúng **thứ tự đọc** (chữ dọc: cột phải trước, trên xuống dưới);
  - các trang nối theo thứ tự, giữa 2 trang có 1 dòng trống;
  - không còn số trang, đầu trang/chân trang, chữ nằm trong hình vẽ, ký hiệu định dạng (markdown, HTML).
- Đi kèm (để kiểm tra, không bắt buộc): `layout/` (ảnh trang có vẽ các vùng layout), `ocr_raw/` (kết quả thô từng
  trang, dạng `.jsonl`).

**Ví dụ** — 1 trang sách viết dọc → đoạn đầu trong `txt/dong_thi_ky_huyet.txt`:

```
【後枝穴】
部位：當肩中與肘之直線上，距富頂穴一寸，離肘橫紋八寸。
解剖：心之分支神經。
主治：血壓高、頭量、頭痛、殺菌、皮膚病、血管硬化。
```

## Bước 2 — Dịch: txt chữ Hán → txt tiếng Việt

**Input**
- Các file `txt/<tên sách>.txt` của bước 1 (UTF-8, chữ Hán; văn ngôn hoặc bạch thoại).
- Text có thể còn lỗi OCR (sai chữ, thiếu chữ, ngắt dòng giữa câu).

**Output**
- `dich/<tên sách>.txt` — mã hoá UTF-8, bản dịch tiếng Việt của cả file:
  - **đủ ý, đúng thứ tự** với bản Hán, không thêm lời dẫn hay giải thích của model;
  - thuật ngữ chuyên ngành (tên huyệt, vị thuốc, bài thuốc, bệnh danh) giữ theo âm Hán-Việt, dịch **nhất quán** trong cả sách;
  - các đoạn cách nhau 1 dòng trống.
- Đi kèm (để kiểm tra): `dich_raw/` (bản dịch từng đoạn, dạng `.jsonl`). Đoạn chưa dịch được ghi `[chưa dịch]`.

**Ví dụ** — đoạn Hán ở trên → `dich/dong_thi_ky_huyet.txt`:

```
【Huyệt Hậu Chi】
Vị trí: nằm trên đường thẳng nối huyệt Kiên Trung với khuỷu tay, cách huyệt Phú Đỉnh 1 thốn, cách lằn chỉ khuỷu tay 8 thốn.
Giải phẫu: nhánh thần kinh của Tâm.
Chủ trị: huyết áp cao, chóng mặt, đau đầu, sát trùng, bệnh ngoài da, xơ cứng mạch máu.
```

## Điều kiện
- Chạy trên **Kaggle**, GPU **T4 x2** (2 × 16 GB VRAM); mỗi phiên tối đa 12 giờ, 30 giờ GPU/tuần.
- Model (OCR và LLM) phải tải về và chạy được ngay trên Kaggle, không gọi API trả phí.
- Cách làm tuỳ bạn: notebook kèm theo là cách làm mẫu (xem `FLOW.md`), miễn giữ đúng input/output ở trên.
