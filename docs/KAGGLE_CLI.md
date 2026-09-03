# Kaggle CLI — Training & Resume Guide

## 1. Yêu cầu local

Kaggle CLI hiện yêu cầu Python 3.11+.

```powershell
py --version
py -m pip install -U kaggle
kaggle --version
```

Authentication hiện hỗ trợ OAuth:

```powershell
kaggle auth login
```

Bạn cũng có thể dùng credential/token theo tài liệu Kaggle chính thức.

## 2. Luôn chạy debug trước

```powershell
cd MiniViGPT

.\scripts\kaggle_push.ps1 `
  -KaggleUsername "YOUR_USERNAME" `
  -Slug "minivigpt-debug" `
  -Config "configs/minivigpt_debug.yaml"
```

Script tạo `dist/kaggle/` gồm:

- `train.py`
- `config.yaml`
- package `minivigpt/`
- `kernel-metadata.json`

Metadata bật GPU, Internet và `NvidiaTeslaT4` theo mặc định.

## 3. Check status

```powershell
.\scripts\kaggle_status.ps1 `
  -KaggleUsername "YOUR_USERNAME" `
  -Slug "minivigpt-debug"
```

## 4. Download output

```powershell
.\scripts\kaggle_download.ps1 `
  -KaggleUsername "YOUR_USERNAME" `
  -Slug "minivigpt-debug"
```

Kiểm tra tối thiểu:

- `summary.json` tồn tại
- `checkpoint_best.pt` tồn tại
- `checkpoint_latest.pt` tồn tại
- `tokenizer.json` tồn tại
- `metrics.jsonl` có train + val records
- `sample.txt` generate được

## 5. Train config 20M

```powershell
.\scripts\kaggle_push.ps1 `
  -KaggleUsername "YOUR_USERNAME" `
  -Slug "minivigpt-20m" `
  -Config "configs/minivigpt_20m.yaml"
```

Không thay nhiều hyperparameter cùng lúc ngay run đầu tiên.

## 6. Resume qua Kaggle Dataset source

Kaggle kernel có thể attach dataset sources. Workflow an toàn:

1. Download output của run trước.
2. Tạo một **private Kaggle Dataset** chứa `checkpoint_latest.pt` (và có thể cả tokenizer/config).
3. Attach dataset đó vào kernel tiếp theo.
4. `kaggle_entry.py` tìm `**/checkpoint_latest.pt` dưới `/kaggle/input` và auto-resume.

Khi đã có dataset source `YOUR_USERNAME/minivigpt-resume`, push:

```powershell
.\scripts\kaggle_push.ps1 `
  -KaggleUsername "YOUR_USERNAME" `
  -Slug "minivigpt-20m-resume" `
  -Config "configs/minivigpt_20m.yaml" `
  -DatasetSource "YOUR_USERNAME/minivigpt-resume"
```

Nếu attach nhiều dataset có `checkpoint_latest.pt`, entry script sẽ liệt kê và chọn path lexicographically last. Tốt nhất mỗi resume job chỉ attach một checkpoint dataset.

## 7. Vì sao không hard-code `/kaggle/input/...`?

Slug dataset của bạn chưa biết trước. Auto-discovery giúp repo portable. Nhưng checkpoint vẫn bị kiểm config model trước khi load.

## 8. T4 vs P100

Repo mặc định T4. Kaggle CLI metadata hiện hỗ trợ `NvidiaTeslaT4`; tài liệu hiện tại còn cảnh báo default Kaggle image có thể không chạy CUDA op đúng trên P100 do build compatibility, nên T4 là lựa chọn an toàn cho repo này.

## 9. Internet

`enable_internet=true` vì Hugging Face dataset/tokenizer dependencies được tải trong Kaggle job. Nếu muốn hoàn toàn offline, bạn phải đóng gói dataset snapshot + Python wheels thành Kaggle Dataset; đó là một experiment reproducibility khác.
