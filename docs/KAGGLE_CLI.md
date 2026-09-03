# Kaggle CLI — Training & Resume Guide

## 1. Yêu cầu local

Kaggle CLI yêu cầu Python 3.11+.

```powershell
py -m pip install -U kaggle
kaggle --version
kaggle auth login
```

Bạn cũng có thể dùng credential/token theo tài liệu Kaggle chính thức
(`~/.kaggle/kaggle.json`, hoặc biến môi trường `KAGGLE_USERNAME`/`KAGGLE_KEY`).

## 2. Quy trình train

Quy trình đầy đủ — đẩy source và corpus thành dataset, push notebook, resume,
cùng các cạm bẫy của Kaggle — nằm ở [`scripts/kaggle/README.md`](../scripts/kaggle/README.md).

Tóm tắt:

```powershell
# source + config (cần --dir-mode zip, nếu không thư mục sẽ bị bỏ qua)
kaggle datasets create -p <thư-mục-source> --dir-mode zip

# corpus đã tokenize, chỉ cần một lần
kaggle datasets create -p artifacts\local

# notebook — phải chỉ định accelerator, xem lưu ý bên dưới
kaggle kernels push -p scripts\kaggle --accelerator NvidiaTeslaT4
kaggle kernels status <user>/minivigpt-train
```

## 3. Theo dõi và tải kết quả

```powershell
.\scripts\kaggle_status.ps1   -KaggleUsername "<user>" -Slug "minivigpt-train"
.\scripts\kaggle_download.ps1 -KaggleUsername "<user>" -Slug "minivigpt-train"
```

Kaggle chỉ cho tải output sau khi kernel kết thúc; trong lúc chạy chỉ xem được
trạng thái. Log của một lần chạy lỗi vẫn tải về được và là nơi đầu tiên nên xem.

## 4. Resume

Kaggle giới hạn 9 giờ mỗi session. Lưu output lần chạy trước thành một private
dataset rồi đính kèm vào lần chạy sau — notebook tự tìm `checkpoint_latest.pt`
dưới `/kaggle/input` (đệ quy) và tiếp tục từ đó. Checkpoint mang theo cả
optimizer state, RNG state và tokenizer nên quỹ đạo train được nối liền mạch.

## 5. Lưu ý quan trọng

**Luôn chỉ định GPU.** Thiếu `machine_shape`/`--accelerator`, Kaggle có thể cấp
P100 (`sm_60`) mà bản PyTorch trong image Kaggle không còn hỗ trợ — mọi phép
tính CUDA sẽ lỗi `no kernel image is available for execution on the device`.

**Dataset được mount lồng nhau** theo dạng `/kaggle/input/datasets/<user>/<slug>/`,
không phải mỗi dataset một thư mục ở cấp một.

**File push phải thuần ASCII** — CLI đọc bằng cp1252 và sẽ lỗi với tiếng Việt.
