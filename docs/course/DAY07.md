# Day 7 — Kaggle CLI: Debug → Main Run → Resume

## Mục tiêu

Không click-run thủ công phụ thuộc UI. Toàn bộ job phải push/status/download được bằng CLI.

Đọc `docs/KAGGLE_CLI.md`.

## 1. Local verification

```powershell
py -m pip install -r requirements.txt
$env:PYTHONPATH="$PWD/src"
pytest -q
python scripts/estimate_training_budget.py
```

## 2. Auth

```powershell
py -m pip install -U kaggle
kaggle auth login
```

## 3. Debug first

Push `minivigpt_debug.yaml`. Nếu data/tokenizer/test/checkpoint/sample chưa pass thì không chạy 20M.

## 4. Read logs

Bạn phải tìm:

- parameter count
- precision
- train loss
- val loss
- gradient norm
- test summary

NaN/Inf hoặc loss không giảm là failure, không phải “train thêm sẽ hết”.

## 5. Main run

Chỉ khi debug pass mới push 20M.

## 6. Resume

Kaggle session có giới hạn. Download checkpoint latest, upload thành private Kaggle Dataset, attach bằng `-DatasetSource`. Entry script tự detect checkpoint trong `/kaggle/input`.

## 7. Điều cần ghi vào experiment log

- Kaggle kernel slug/version
- GPU machine shape
- config file
- dataset revision
- checkpoint source nếu resume

## Checkpoint Day 7

Bạn pass khi có thể xóa browser khỏi workflow và vẫn chạy/debug/download bằng command line.
