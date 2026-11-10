# Hai Dataset models và packages cho Kaggle offline

## Bộ đã chốt theo báo cáo mới

Đã đọc báo cáo 955 distributions: Torch/CUDA, torchvision, NumPy bridge và API Hunyuan đều đạt.
Bộ packages mới khoảng 3,93 GiB, có 5.973 file. Có 74 artifact dependency;
6 archive trong `sources` đã chuyển thành 6 thư mục với 5.899 file thường.
Mã dự án đã chuyển hoàn toàn vào các ô notebook, không có wheel `vtm_corpus` trong bộ upload.
Kernel giữ Torch 2.11.0+cu130, torchvision 0.26.0+cu130 và Transformers 5.20.2 cho Hunyuan.

| Runtime | Gói gửi kèm | Gói dùng lại từ kernel |
|---|---:|---:|
| Paddle / Python 3.13 | 24 | 87 |
| TongGu / Python 3.10 | 49 | 53 |

Một số artifact được hai runtime dùng chung nên số gói gửi kèm không cộng trực tiếp.
Danh sách dùng lại nằm trong từng manifest, kèm phiên bản, dependency và nguồn báo cáo.
Installer kiểm tra ràng buộc phiên bản, Python ABI và RECORD checksum. File Python được sao chép
vào runtime riêng; native library tương thích được liên kết. Không thêm toàn bộ site-packages
Python 3.13 vào Python 3.10; extension cp313 bị từ chối trong runtime cp310.

Các lỗi `pip check` của Colab, Gradio, moviepy và các gói khác trong báo cáo được giữ làm
thông tin đối chiếu; không dùng chúng làm lý do hạ phiên bản hoặc sửa toàn bộ kernel.

### Kernel đổi phiên bản gói phụ: `cryptography expected 41.0.7, got 50.0.2`

Lỗi này xuất hiện sau khi 73 artifact đã được xác minh và 24 wheel Paddle đã cài xong.
Đối chiếu lại báo cáo gốc cho thấy nhiều bản metadata cùng tên: `cryptography` có cả
`50.0.2` và `41.0.7`, NumPy có cả `2.1.3` và `1.26.4`. Bộ chọn cũ lấy bản xuất hiện cuối,
nên metadata cũ của hệ điều hành đã ghi đè bản wheel đang được dùng. Bộ tạo báo cáo và bộ
chọn packages mới giữ bản active/đầu tiên theo thứ tự tìm kiếm, vẫn lưu các bản khác để đối chiếu.
Notebook mới đọc metadata hiện tại và kiểm tra dependency của các wheel offline, các root
đã pin và toàn bộ gói được dùng lại, có xét extras cùng marker Linux/Python đích.
Gói phụ đổi phiên bản chỉ được dùng khi các ràng buộc vẫn thỏa. Dependency mới phát sinh
(ví dụ `cffi`) được lấy từ kernel nếu có và tương thích; thiếu gói hay sai ABI vẫn dừng rõ tên.
Torch, torchvision, Transformers, NumPy và các gói CUDA giữ kiểm tra phiên bản nghiêm ngặt.
Riêng manifest từ báo cáo SHA-256 `1e47c86c9fe706d2e297ea8c0f574716c1d1aadb7366e8de99b195c720c0c746`,
installer sửa tham chiếu NumPy từ metadata cũ `1.26.4` thành bản active `2.1.3` đã được
probe `torch_numpy_bridge` xác nhận trong chính báo cáo đó; không chấp nhận tùy ý bản NumPy khác.
Wheel gửi kèm không bị thay bằng phiên bản kernel; không tải mạng hoặc hạ phiên bản kernel.

Kế hoạch và lỗi dependency được ghi tại
`/tmp/vtm_kaggle_runtime/paddle_py313_cu130/lib/python3.13/site-packages/.vtm_kernel_plan.json`
(TongGu có báo cáo tương tự trong site-packages riêng). Sau khi lắp ráp, installer chạy
`pip check` trong runtime riêng rồi mới chạy probe GPU; TongGu kiểm tra sau bước build mmcv.

Cảnh báo `Error in sitecustomize ... No module named 'wrapt'` có thể do đường dẫn Python
của kernel bị truyền vào interpreter riêng. Installer mới bỏ `PYTHONPATH`, `PYTHONHOME`,
`PYTHONUSERBASE` kế thừa và tắt user site trong các lệnh cài/probe và launcher OCR riêng.
Không cài thêm `wrapt` vào kernel chỉ để che cảnh báo này.

Khi chạy lại Setup, bản sao gói kernel từ lần cài dở được dọn trong runtime riêng theo
RECORD/ledger trước khi cài lại. Dataset Input và file gốc trong kernel không bị sửa.
Chỉ cần thay ô `offline-setup` rồi chạy lại ô đó và Setup; không cần sửa manifest hay upload
lại packages cho lỗi lệch phiên bản gói phụ này. Các bước cài/GPU còn lại cần xác minh trên Kaggle.

### TongGu: `Kernel RECORD checksum differs: nvidia-nccl-cu13/.../nccl.h`

Log đã đi qua bước Paddle và cài wheel TongGu, rồi dừng khi xác minh file kernel.
Báo cáo gốc có cả `nvidia-nccl-cu12==2.32.3` và `nvidia-nccl-cu13==2.28.9`;
các bản NCCL có thể cùng sở hữu đường dẫn `nvidia/nccl`. Metadata đúng phiên bản
không đảm bảo payload ở đường dẫn dùng chung vẫn đúng RECORD.

Torch 2.10.0+cu130 của TongGu pin chính xác `nvidia-nccl-cu13==2.28.9`.
Đã thêm wheel này từ PyPI vào `packages/wheels`, chuyển NCCL từ `reused` sang
`records` trong `tonggu_manifest.json` và thêm pin vào `requirements/tonggu.txt`.
Wheel mới được kiểm tra SHA-256, CRC và dependency; không dùng wheel NCCL 2.28.3
của Paddle để thay cho pin của TongGu. Không bỏ qua checksum của header hay thư viện `.so`.
Mọi lần dựng bundle mới cũng luôn gửi kèm NCCL thay vì dùng lại payload kernel.

**Lỗi này cần cập nhật Dataset packages, chỉ thay ô notebook là chưa đủ.**

- Thư mục đầy đủ `kaggle_upload/packages/` đã được cập nhật; ZIP đầy đủ là
  `kaggle_upload/packages_upload.zip`, vẫn dùng `ZIP_STORED` và giữ nguyên cấu trúc thư mục.
  Dùng thư mục hoặc ZIP này để tạo phiên bản mới của Dataset packages hiện có.
- `kaggle_upload/packages_nccl_fix.zip` khoảng 189 MiB là bản vá **cho bộ packages đã có**.
  Giải nén và ghép đè vào thư mục packages cũ trước khi cập nhật Dataset. Bản vá chỉ chứa
  wheel mới, `tonggu_manifest.json`, `requirements/tonggu.txt` và `bundle_manifest.json`;
  không dùng riêng bản vá làm Dataset packages đầy đủ.
- Thay ô `offline-setup` từ notebook 3 mới, chạy lại ô đó rồi Setup. Bản sửa dọn metadata
  NCCL cài dở trong runtime TongGu để pip không nhầm là wheel đã cài đủ. Giữ nguyên ba
  đường dẫn Dataset, models và data tiếp tục dùng lại.

Gắn **phiên bản mới** của Dataset packages vào notebook trước khi chạy lại. Setup mới
dừng ngay trước bước cài nếu manifest còn yêu cầu dùng lại NCCL kernel, kèm hướng dẫn
cập nhật Dataset; không cài lại toàn bộ rồi mới báo cùng lỗi checksum.

Cảnh báo `sitecustomize` thiếu `wrapt` vẫn xuất hiện trong log người dùng gửi, nhưng
pip đã cài xong các wheel TongGu; điểm dừng của lần này là checksum NCCL.
Chưa chạy lại việc cài, build mmcv hoặc OCR trên GPU Kaggle sau bản vá này.

### TongGu: `Package files overlap with different bytes: cv2/LICENSE-3RD-PARTY.txt`

Dataset đã có NCCL mới nhưng manifest cũ vẫn liệt kê `opencv-python==4.14.0.94`
ở nhóm dùng lại từ kernel, đồng thời gửi kèm `opencv-python-headless==5.0.0.93`.
Hai gói cùng sở hữu `cv2`; [OpenCV yêu cầu chỉ chọn một biến thể cho mỗi môi trường](https://github.com/opencv/opencv-python#installation-and-usage).
Xung đột ở file license chỉ là vị trí đầu tiên bị phát hiện, không nên bỏ qua riêng file đó.

Setup mới chọn đúng một gói OpenCV gửi kèm: headless cho TongGu và contrib cho Paddle.
Các biến thể OpenCV kernel khác bị loại khỏi kế hoạch dùng lại, được ghi ở `excluded_packages`
trong `.vtm_kernel_plan.json`. Nếu một dependency/root thực sự yêu cầu biến thể khác,
setup vẫn dừng thay vì giả metadata hoặc tự bỏ ràng buộc đó. Các checksum/ABI khác giữ nguyên.

Trước khi pip cài, setup dọn payload và metadata OpenCV cài dở trong runtime riêng,
rồi cài lại wheel đã có trong Dataset. Bộ dựng packages cũng đã sửa metadata tĩnh của
MMCV để dùng lựa chọn `opencv-python-headless>=3`, phù hợp với `setup.py` nguồn và
pin headless của TongGu; không kéo thêm `opencv-python` chỉ vì metadata tĩnh cũ.

**Lần sửa này chỉ cần thay ô `offline-setup`, chạy lại ô đó rồi Setup.**
Dataset packages trong log đã chứa NCCL 2.28.9 nên tiếp tục dùng lại, không cần upload mới.
Ba đường dẫn packages/models/data giữ nguyên. Chưa xác minh lại trên GPU Kaggle.

### Polygon3 build lỗi, pip chỉ báo `No available output`

Log lần này đã xác nhận Torch `2.10.0+cu130` chạy phép nhân tensor trên GPU `(12, 0)`.
Lỗi tiếp theo nằm ở build Polygon3; log cũ chưa có thông báo compiler nên chưa thể
kết luận chính xác subprocess nào thất bại. Không hạ NumPy chỉ dựa vào thông báo này.

Đối chiếu Python standalone gửi kèm thấy `CC/CXX/LDSHARED` còn dùng `clang/clang++`
và `INCLUDEPY` còn là `/install/include/python3.10`. Đây là cấu hình cần sửa khi đưa
runtime sang `/tmp`; [python-build-standalone cũng ghi nhận vấn đề tham chiếu compiler](https://github.com/astral-sh/python-build-standalone/issues/209).
Bản setup mới:

- Chọn gcc/g++, sửa sysconfig trong **bản sao Python ở runtime riêng**, chuyển các
  đường dẫn `/install` về prefix hiện tại và xóa cache sysconfig của cả cp310/cp313.
  Không sửa nguồn trong Dataset Input.
- Kiểm tra compiler và header bằng cách build rồi import một C extension nhỏ trước
  khi cài các source dependency. Mặc định build hai job, vẫn tôn trọng `MAX_JOBS` đã đặt.
- Build riêng wheel của jieba, Polygon3, zhconv và mmcv với pip `-vvv`, tắt global pip
  cache, giữ source build và ghi log đầy đủ vào `runtime/build_logs/<package>-<key>.pip.log`.
  Polygon3 có thêm lệnh chẩn đoán compiler trực tiếp nếu pip vẫn thất bại.
- Cache wheel theo hash nguồn, Python ABI, NumPy, Torch, CUDA, compiler và cờ build.
  Chỉ dùng lại khi receipt và SHA-256 khớp; cache hỏng hoặc đổi ABI sẽ build lại.
- Đảm bảo source MMCV dùng C++17 và thay `tensor.type().scalarType()` bằng
  `tensor.scalar_type()` trong `spconv_utils.h`. Chỉ sửa source đã chép vào `/tmp`.
- Sau cài nguồn, chạy thử Polygon với mảng NumPy, jieba, zhconv và resize OpenCV;
  sau build MMCV vẫn bắt buộc thử `mmcv.ops.nms_rotated` trên GPU trước OCR.
- Launcher Python riêng khởi tạo site/venv bình thường nhưng bỏ hook `sitecustomize`
  của host trong tiến trình con, tránh kéo instrumentation thiếu `wrapt` vào runtime.
  Kernel đang chạy giữ nguyên; logic launcher hiển thị trực tiếp trong ô notebook.
- Lưu checkpoint sau mỗi component đã qua probe. Nếu TongGu lỗi sau Paddle, chạy lại
  Setup sẽ kiểm tra và dùng lại Paddle, rồi tiếp tục TongGu. Chỉ component ghi trong
  receipt của cùng chữ ký setup mới được xem là đã hoàn thành.

Thông tin toolchain nằm trong `/tmp/vtm_kaggle_runtime/build_probe/toolchain.json`.
`all.log` giữ thêm 100 dòng cuối khi báo lỗi; log pip/compiler đầy đủ vẫn được lưu riêng.

Kiểm tra cục bộ đã cross-compile thành công **mã Polygon3 gốc** sang Linux x86_64,
dùng đúng header Python 3.10.20 và NumPy 2.2.6 từ bundle, bằng Zig cc chạy trên Windows.
Đã thử build wheel Python thật, offline, qua launcher mới và dùng lại cache đã xác minh.
Đây chưa phải kiểm tra import Polygon3 trên Linux hay build/chạy MMCV trên Kaggle.

**Giữ Dataset packages hiện tại, thay toàn bộ ô `offline-setup` từ notebook 3 mới rồi
chạy lại ô đó và Setup.** Không cần nén hoặc upload lại packages cho bản sửa build này.

### Sau khi build MMCV: lỗi `matplotlib_inline.backend_inline`

Log mới đã xác nhận build/cài `mmcv_full-1.7.2-cp310-cp310-linux_x86_64.whl` thành công
và `pip check` không còn dependency lỗi. Bước tiếp theo import MMRotate kéo theo
Matplotlib, nhưng runtime Python riêng kế thừa `MPLBACKEND=module://matplotlib_inline.backend_inline`
của kernel notebook. Backend inline này không có trong runtime riêng nên import bị chặn
trước khi phép thử `nms_rotated` trên GPU được gọi.

Setup mới đặt `MPLBACKEND=Agg` trong môi trường con và ngay trong launcher `vtm-python`.
Probe kiểm tra runtime có sẵn cũng dùng Agg. Backend của kernel notebook giữ nguyên;
không thêm `matplotlib_inline` hay đổi phiên bản Matplotlib.
[Agg là backend không tương tác của Matplotlib](https://matplotlib.org/3.7.1/users/explain/backends.html).

Đã tái hiện bằng một tiến trình thật với backend inline kế thừa và chặn import
`matplotlib_inline`, rồi kiểm tra cùng tiến trình khởi chạy với Agg có thể import
`pyplot` và vẽ canvas. Test cũng kiểm tra backend của parent không bị thay đổi.

**Giữ phiên Kaggle hiện tại để dùng lại cache wheel MMCV đã build.** Thay ô
`offline-setup`, chạy lại ô đó rồi Setup. Không cần upload lại Dataset. Thay backend
không đổi ABI hay khóa cache wheel, nên wheel có receipt/checksum đúng được dùng lại.
Build MMCV đã được log người dùng xác nhận; phép thử GPU NMS sau bản sửa backend
vẫn cần chạy lại trên Kaggle trước khi coi runtime TongGu hoàn tất.

### Chạy lại Setup: pip báo đã cài nhưng `cv2` thiếu `resize`

Log tiếp theo cho thấy metadata `opencv-python-headless==5.0.0.93` vẫn tồn tại,
pip báo `Requirement already satisfied`, nhưng payload `cv2` không có API `resize`.
Chỉ loại gói OpenCV trùng hoặc kiểm tra tên/version metadata chưa bảo đảm file thực
đã được khôi phục sau một lần cài dở. Bản trước chưa kiểm tra điều kiện này đầy đủ.

Bản sửa mới tách OpenCV khỏi bước dọn gói kernel chung và kiểm tra riêng:

- Đối chiếu kích thước/checksum từng file loader, native extension, thư viện đi kèm
  và metadata với RECORD của wheel đã xác minh. Kiểm tra cả RECORD bị mất, provider
  khác còn sót và loader/binary không thuộc wheel đang chọn.
- Nếu phát hiện hỏng/thiếu, xóa đúng namespace OpenCV và metadata trong private
  site, rồi cài lại **đúng wheel gửi kèm** bằng `--force-reinstall --no-index --no-deps`.
  Đường dẫn xóa phải nằm trong runtime riêng; không đi theo symlink ra kernel/Input.
- Kiểm tra lại toàn bộ payload sau cài và chạy trong tiến trình mới: xác nhận
  `cv2.__file__` nằm trong private site, chỉ có một provider đúng phiên bản và thực
  hiện resize, đổi màu, threshold/contour, box points, warp affine, padding, PNG encode/decode.
- Kiểm tra cuối setup nay import đúng API worker của từng model, import pipeline
  TongGu, đọc config đường dẫn model và thử render PDFium nhỏ; không tải trọng số hay
  tạo model. Các lỗi import được giữ phần cuối traceback, kết quả dùng marker riêng
  để log của framework không làm hỏng việc đọc JSON preflight.

Audit payload và kết quả API được lưu ở `.vtm_opencv_payload.json` và
`.vtm_opencv_probe.json` trong site-packages riêng của từng runtime.

Đã tái hiện bằng **pip thật, private venv thật**, OpenCV headless 5.0.0.93 và NumPy
2.2.6 trên Windows: xóa loader nhưng giữ metadata, xác nhận pip bỏ qua và import thiếu
`resize`, sau đó chạy bản sửa để phục hồi và thử lại setup lần hai. Cả hai lần kiểm tra
native image API đều qua; lần hai không phải cài lại OpenCV. Đây chưa phải phép thử
GPU/model thật trên Linux Kaggle.

Giữ phiên Kaggle để giữ cache MMCV. Thay toàn bộ ô `offline-setup` từ notebook mới rồi
chạy lại ô đó và Setup. Ô này đã chứa cả preflight API mới nên chỉ thay một ô là đủ.
Không cần upload lại Dataset hay đổi ba đường dẫn packages/models/data.

## ZIP giữ nguyên cây thư mục

**Upload file `kaggle_upload/packages_upload.zip` đã tạo sẵn và import lại notebook 3 mới.**
ZIP dùng `ZIP_STORED`: chỉ gom file để upload, không nén lại dữ liệu. Tên wheel gốc
và các đường dẫn `packages/wheels/`, `packages/sources/`, `packages/requirements/`
được giữ nguyên. Không có `.bin` hoặc tên file đánh số.

ZIP chỉ ghi entry file có đường dẫn đầy đủ; thư mục được tạo từ các đường dẫn khi
giải nén. Không ghi entry thư mục có basename rỗng. Wheel giữ nguyên định dạng/bytes
gốc. `sources` chứa cây thư mục đã giải nén, không còn `.tar.gz`, `.zip`, PAX header hay symlink.

### Sửa riêng thư mục sources

Nguồn cũ chứa runtime Python với 1.048 symlink, hai source tar dùng PAX metadata và
các archive lồng trong ZIP upload. [Kaggle tự giải nén archive khi tạo Dataset](https://www.kaggle.com/docs/datasets).
Người dùng đã khoanh vùng lỗi vào `sources`; bản mới loại các thành phần archive/link
khỏi phần upload này. Chưa có log máy chủ để khẳng định chính xác entry gây lỗi.

```text
packages/sources/
  python-3.10.20/
  jieba-0.42.1/
  mmcv-full-1.7.2/
  Polygon3-3.0.9.1/
  TongGuOCR-df2a46eac12a14482609bffc6e14ead06fc57d66/
  zhconv-1.4.3/
```

Symlink được chép thành file thường. Các tên Linux chỉ khác hoa/thường được lưu
bằng tên không va chạm trên Windows; manifest ghi tên đích và quyền Linux cần phục hồi.
Một tài nguyên dữ liệu gzip của Tcl được đưa lên ở dạng dữ liệu đã giải nén và tạo lại
gzip trong runtime riêng. Manifest đối chiếu SHA-256 từng payload và lưu hash archive gốc.

Notebook mới chép Python/source dependency từ Input vào `/tmp`, phục hồi quyền thực thi
và tên Linux, rồi mới cài/build. Không build trực tiếp trong Dataset chỉ đọc.
**Phải import lại notebook 3 mới để dùng bộ sources đã giải nén.**

Để thử riêng phần vừa sửa, upload `kaggle_upload/sources_upload_test.zip` khoảng
178 MiB thành một Dataset thử. File này chỉ chứa `sources`, không phải bộ chạy OCR.
Khi thử thành công, dùng `packages_upload.zip` khoảng 3,93 GiB làm Dataset packages đầy đủ.

### Lỗi `Incomplete source tree` sau khi upload thư mục

Báo cáo mới cho thấy cây Python có 4.943/4.957 file: thiếu 14 cache `.pyc` trong
`__pycache__`. Đây là bytecode Python có thể tạo lại từ nguồn `.py`; không cần upload
lại Dataset để bổ sung cache nếu các file nguồn tương ứng vẫn đúng checksum.

Notebook mới vẫn kiểm tra SHA-256, đồng thời xử lý các trường hợp có thể khôi phục chính xác:

- File rỗng thiếu được tạo lại trong `/tmp` khi manifest xác nhận kích thước 0 và hash của nội dung rỗng.
- Tên file đổi hoa/thường được chấp nhận khi kích thước và SHA-256 khớp.
- File thiếu có payload trùng một file đã xác minh được chép từ payload đó.
- Riêng runtime Python, cache `.pyc` thiếu trong `__pycache__` được bỏ qua khi file `.py`
  tương ứng có trong manifest và đã xác minh SHA-256. Runtime dùng nguồn và tự biên dịch
  cache khi cần; không tạo file `.pyc` rỗng. Bytecode không có nguồn vẫn bắt buộc phải đủ.
- File thừa ngoài manifest không được chép hay dùng trong runtime.
- Ngoài cache có nguồn đã xác minh ở trên, file có dữ liệu bị thiếu hoặc sai checksum
  vẫn làm setup dừng. Cache có sẵn nhưng sai checksum cũng bị từ chối.

Không ghi sửa vào `/kaggle/input`. Báo cáo đầy đủ được lưu ở
`/kaggle/working/offline_install_logs/source_tree_report.json`, gồm `missing_uploaded`,
`recreated_empty`, `omitted_bytecode` (cache thiếu và nguồn đã xác minh), `ignored_extras`,
`unresolved_nonempty` và `changed` của từng cây nguồn.
Thông báo lỗi mới in số lượng và tối đa 12 tên file, thay vì chỉ báo `Incomplete source tree`.

Cập nhật ô `offline-setup` từ notebook mới hoặc import lại notebook, giữ ba đường dẫn
Dataset đã điền, rồi chạy lại ô Setup. Chưa cần upload lại packages nếu các payload cần thiết vẫn đủ.

Đã kiểm tra ZIP, giải nén và đối chiếu cây thư mục/checksum cục bộ. Chưa xác định
chính xác nguyên nhân lỗi tạo Dataset trước đó trên dịch vụ Kaggle.
**Chưa kiểm thử tạo Dataset mới hay GPU inference trên Kaggle.**

## Models và packages tách riêng

Dùng hai thư mục độc lập trong `kaggle_upload/`. Tạo một Dataset cho `models/`,
một Dataset cho `packages/`, rồi gắn cả hai vào notebook 3. Dataset PDF đã upload
trước đó được dùng lại; không đưa PDF vào hai bộ mới.

```text
kaggle_upload/
  models/
    hunyuan/
    ppocr_det/
    ppocr_rec/
    paddle_vl/
    paddle_layout/
    tonggu/
      ocr/
      layout/finetune_infer.pth
    models_manifest.json
  packages/                  # 5.973 file, chỉ dependency/runtime
    wheels/                  # tên wheel gốc
    sources/                 # 6 thư mục nguồn/runtime đã giải nén
    requirements/
    paddle_manifest.json
    tonggu_manifest.json
    system_assets_manifest.json
    bundle_manifest.json
  packages_upload.zip        # ZIP_STORED chứa cây packages/ như trên
  packages_nccl_fix.zip      # bản vá khoảng 189 MiB cho packages đã có
  sources_upload_test.zip    # khoảng 178 MiB, chỉ thử upload sources
```

Tạo Dataset packages bằng **file ZIP đã tạo sẵn** hoặc upload trực tiếp folder `packages/`
mới. Chỉ dùng một trong hai cách. Dataset models và Dataset PDF cũ tiếp tục dùng lại;
không cần upload lại chúng. Không gộp toàn bộ `kaggle_upload/` vào cùng một Dataset.

Khi gắn Dataset mới, điền `PACKAGES_DIR` bằng thư mục `packages` chứa
`paddle_manifest.json` và `system_assets_manifest.json`. Import lại notebook 3 mới
để có mã setup/worker trực tiếp trong notebook. Dataset models và PDF vẫn dùng lại.

Bộ nguồn cũ được giữ tại `work/kaggle_packages_source/`; khi chuẩn bị upload,
tool loại wheel mã dự án và `app_manifest.json` khỏi bộ dependency.
`models/` có 54 file weights/manifest, khoảng 8,68 GiB,
gồm weights cho đủ bốn model và layout/detector/recognizer liên quan.
`models_manifest.json` ghi revision nguồn,
đường dẫn, kích thước và SHA-256 từng file. Các file model tải từ Hugging Face được
đối chiếu LFS SHA-256 hoặc Git blob hash; cache chỉ được dùng lại khi khớp nguồn đã ghim.

## Runtime và phần còn cần kiểm tra

Paddle dùng [wheel CUDA 13.0 chính thức](https://www.paddlepaddle.org.cn/packages/stable/cu130/paddlepaddle-gpu/)
của Paddle GPU 3.2.1, PaddleOCR 3.7.0 và PaddleX 3.7.2, trong
`/tmp/vtm_kaggle_runtime/paddle_py313_cu130`. Những dependency kernel không đáp ứng
pin cụ thể được cài riêng; không thay Torch/NumPy của kernel.

TongGu giữ Python Linux 3.10 portable, Torch 2.10.0+cu130, torchvision 0.25.0+cu130 và
Transformers 5.3.0 trong `/tmp/vtm_kaggle_runtime/tonggu_py310_cu130`.
Bản CUDA 13 của [PyTorch](https://download.pytorch.org/whl/cu130) dùng cùng toolkit
13.0 đã có trên Kaggle; không còn đóng thêm bộ compiler CUDA 12.8.
Chỉ dùng lại package khi tương thích đúng Python đích; không ghép extension cp313 vào cp310.

Các wrapper `vtm-python` ưu tiên thư viện CUDA của đúng runtime cho cả probe và worker.
Python files được sao chép nên patch OpenMMLab không ghi ngược vào kernel.
Runtime nguồn được xác nhận bởi báo cáo: Linux x86_64, glibc 2.39, Python 3.13.16,
RTX PRO 6000 Blackwell sm_120, khoảng 94,97 GiB VRAM, CUDA toolkit 13.0, GCC 13.3.

**Chưa chạy cài/build/inference thật trên Kaggle từ máy Windows chuẩn bị packages.**
Metadata đạt không chứng minh native ABI hoạt động. Installer thử Torch GPU trước khi
build mmcv-full 1.7.2 từ source, dùng CUDA 13.0 và `TORCH_CUDA_ARCH_LIST=12.0`.
Chỉ đánh dấu runtime thành công sau khi Paddle GPU và `mmcv.ops.nms_rotated` chạy được.
Lỗi dừng trước OCR, có log trong `/kaggle/working/offline_install_logs/`.

## Notebook 3 và weights

Notebook có 8 ô code theo thứ tự: cấu hình → hàm xử lý → hàm setup offline → worker
→ gọi setup → chạy OCR → xem kết quả → xuất HTML/JSONL/ảnh/ZIP. Đường dẫn và profile theo sách tập trung ở ô đầu.
Các hàm setup, chọn trang, điều phối và so sánh chạy trực tiếp trong notebook;
không có loader base64/zlib hay lớp `OCRWorkflow`.

Chỉ một ô `%%writefile` ghi worker độc lập ra `/tmp/vtm_notebook_worker/ocr_worker.py`
(Windows dùng `work/notebook_worker/ocr_worker.py`). Worker chứa backend OCR và logic
xử lý trang cần chạy ở tiến trình riêng; không import `vtm_corpus` hay tải repo.
Bạn sửa trực tiếp mã trong notebook rồi chạy lại các ô liên quan. Các lệnh chính là
`setup_ocr(...)`, `run_ocr_models(session)`, `compare_ocr_results(session)` và
`review_character_conflicts(session, ...)`.

Packages chỉ chứa thư viện/runtime bên thứ ba;
source TongGu/mmcv là dependency cần cài/build, không phải mã điều phối của dự án.

Import bản notebook 3 mới. Bước 1 có đúng ba đường dẫn gốc cho ba Dataset riêng:

```python
PACKAGES_DIR = ""  # Dataset packages
MODELS_DIR = ""    # Dataset models
DATA_DIR = ""      # Dataset PDF
```

Điền đường dẫn thực tế trên Kaggle vào ba dòng này. Notebook tự ghép `WEIGHTS`;
không phải điền bảy đường dẫn model riêng. `WORK_DIR` là thư mục ghi kết quả, không phải Dataset đầu vào.

| Biến | Đường dẫn cần điền |
|---|---|
| `PACKAGES_DIR` | Thư mục `packages` chứa `paddle_manifest.json` và `system_assets_manifest.json` |
| `MODELS_DIR` | Dataset chứa các thư mục `hunyuan`, `ppocr_det`, `ppocr_rec`, `paddle_vl`, `paddle_layout`, `tonggu` |
| `DATA_DIR` | Thư mục PDF trong Dataset đã upload trước đó |

Có thể điền thư mục chứa manifest/model trực tiếp hoặc Dataset có thêm folder bọc
`packages/` hay `models/`. Dưới `tonggu`, notebook dùng `ocr/` và
`layout/finetune_infer.pth`. PDF được tìm đệ quy trong `DATA_DIR`.

Không dùng `Path("")` cho đường dẫn chưa điền vì nó trỏ tới thư mục hiện tại.
Notebook báo lỗi khi chuỗi đường dẫn bắt buộc còn rỗng. Trên Kaggle, config TongGu
được tạo tại `WORK_DIR/tonggu_paths.json` bằng các đường dẫn bạn điền, không cần sửa
file trong Dataset chỉ đọc. Config TongGu riêng qua `WEIGHTS["tonggu_config"]` vẫn
được hỗ trợ nếu đã có.

Sau khi điền đường dẫn, Run All.
Mặc định 1 PDF, `MAX_BOOKS=1`, `PAGES=""`, `LIMIT=None`; bốn model chạy lần lượt và giải phóng worker.
Thử nhanh bằng `PAGES="1-2"`, `LIMIT=2`. Kết quả được đóng gói ở
`WORK_DIR/output/<tên_sách>_results.zip`; tải về, giải nén và mở `index.html` trong thư mục sách.
Setup báo đường dẫn thiếu trước khi cài môi trường. Runtime còn tốt được dùng lại.
`RUN_OCR=False` dùng mã notebook và PDFium để xem checkpoint, không build các model.

Notebook 2 và workflow dịch được giữ nguyên.

## Tái tạo và đối chiếu

Khi kernel thay đổi, chạy lại [notebook 0](../0_kiem_tra_moi_truong_kaggle_offline.ipynb)
trong phiên mới và gửi `kaggle_environment_report.json`. Không suy ra khả năng import
hay GPU chỉ từ metadata.

```powershell
.\.venv\Scripts\python.exe -X utf8 tools/prepare_kaggle_folders.py --component packages
# Khi dựng lại từ bộ nguồn cũ, bổ sung wheel NCCL đúng pin của Torch:
.\.venv\Scripts\python.exe -X utf8 tools/repair_kaggle_nccl.py --patch-output kaggle_upload/packages_nccl_fix_next.zip
# Chỉ khi muốn tạo ZIP khác để upload (không ghi đè ZIP đã có):
.\.venv\Scripts\python.exe -X utf8 tools/package_kaggle_upload.py --output kaggle_upload/packages_upload_next.zip
# Tạo ZIP chỉ thử sources nếu cần:
.\.venv\Scripts\python.exe -X utf8 tools/package_kaggle_upload.py --sources-only --output kaggle_upload/sources_upload_test_next.zip
```

Lệnh đầu sao chép wheel và giải nén nguồn/runtime thành thư mục thường; không cài package hay chạy OCR.
Nó dùng `kaggle_offline_packages/` nếu còn, nếu không dùng `work/kaggle_packages_source/`.
Lệnh thứ hai tạo ZIP không nén lại, giữ nguyên các đường dẫn thư mục. Không cần chạy lại
để dùng `packages_upload.zip` đã tạo sẵn.

Để chuẩn bị folder mới và ZIP từ một bộ nguồn khác:

```powershell
.\.venv\Scripts\python.exe -X utf8 tools/prepare_kaggle_folders.py --component packages --bundle <raw-packages> --output <thu-muc-moi>
.\.venv\Scripts\python.exe -X utf8 tools/package_kaggle_upload.py --bundle <thu-muc-moi>/packages --output <file-upload-moi.zip>
```

`--component models` tải/kiểm tra model độc lập nếu cần; không cần chạy khi chỉ sửa packages.

Khi cần dựng packages theo báo cáo kernel khác:

```powershell
.\.venv\Scripts\python.exe -X utf8 tools/rebuild_kaggle_packages.py --report <report.json>
.\.venv\Scripts\python.exe -X utf8 tools/prepare_kaggle_folders.py --component packages --bundle work/kaggle_rebuild/kaggle_offline_packages --output kaggle_upload_next
```

Trình dựng dùng lại cache và ghi bộ đã kiểm tra metadata vào `work/kaggle_rebuild/kaggle_offline_packages/`,
giữ nguyên bộ đang dùng. Kernel reuse plan và dữ liệu resolver ở `work/kaggle_package_build/`, không upload.
Bộ đầy đủ cũ được giữ tại `work/kaggle_package_build/full_bundle_before_kernel_reuse/` để đối chiếu.
Khi sửa mã trong repo, chạy `tools/update_kaggle_notebook_bootstrap.py` để cập nhật
các hàm trực tiếp và worker trong notebook; tool giữ nguyên cấu hình người dùng.
Sau đó import lại notebook 3. Không cần dựng lại hay upload models/packages khi chỉ sửa logic.

Installer chỉ dùng `pip --no-index --no-deps` và file trong Dataset, không tải Internet.
Dependency đến từ wheel gửi kèm hoặc kernel đúng phiên bản/ABI đã kiểm tra.
Runtime/build ở `/tmp` vì `/kaggle/working` chỉ còn khoảng 19,5 GiB trong báo cáo.
