# MiniViGPT — Vietnamese Decoder-only LM From Scratch

MiniViGPT là repo **giáo dục + thực nghiệm reproducible** để tự xây và pretrain một decoder-only language model trên dữ liệu tiếng Việt bằng PyTorch và Kaggle CLI.

> Mục tiêu của repo là hiểu/trực tiếp kiểm chứng toàn bộ pipeline LLM pretraining. Repo **không** tuyên bố kiến trúc mới, SOTA, “first Vietnamese LLM”, hay compute-optimal foundation model.

## Pipeline

```text
UVW-2026 Vietnamese Wikipedia (pinned revision)
        ↓
shuffle + quality filtering + exact dedup
        ↓
NFC + byte-level BPE tokenizer trained from scratch
        ↓
EOS-packed uint16 token streams
        ↓
MiniViGPT — 20,306,304 params
        ├── Pre-RMSNorm
        ├── RoPE
        ├── Multi-Head Causal Self-Attention
        ├── SwiGLU with LLaMA-style ~8/3·d width
        ├── Residual connections
        ├── tied input/output embeddings
        └── depth-scaled residual projection init
        ↓
AdamW + warmup/cosine + grad accumulation + AMP
        ↓
deterministic validation → best checkpoint
        ↓
held-out test evaluation
        ↓
greedy / temperature / top-k / top-p generation
```

## Vì sao implementation này không sơ sài?

Các lựa chọn kiến trúc/training đều có provenance rõ ràng:

- scaled dot-product causal attention: Transformer
- RMSNorm: Zhang & Sennrich
- RoPE: RoFormer
- SwiGLU: Shazeer; width rule theo Meta LLaMA reference
- tied embeddings: Press & Wolf
- residual projection scaling + optimizer recipe: GPT/nanoGPT-style engineering baseline
- top-p generation: Holtzman et al.

Đọc **[`docs/THEORY_PROVENANCE.md`](docs/THEORY_PROVENANCE.md)** để xem từng kỹ thuật đến từ paper/repo nào và MiniViGPT khác reference ở đâu.

## Config chính

`configs/minivigpt_20m.yaml`:

| Field | Value |
|---|---:|
| Vocabulary | 16,000 byte-BPE tokens |
| Context | 256 |
| Model dim | 384 |
| Layers | 8 |
| Heads | 6 |
| Head dim | 64 |
| SwiGLU hidden | 1,024 |
| Parameters | **20,306,304** |
| Effective sequences/update | 48 |
| Tokens/update | 12,288 |
| Updates | 8,000 |
| Planned token presentations | 98,304,000 |
| Tokens/parameter | ~4.84 |

Con số parameter/training budget được tính từ code thật:

```powershell
$env:PYTHONPATH="$PWD/src"
python scripts/estimate_training_budget.py --config configs/minivigpt_20m.yaml
```

~4.84 tokens/parameter là **educational Kaggle budget**, không phải compute-optimal claim. Chinchilla-style scaling literature cho thấy large-language-model quality phụ thuộc mạnh vào cả model size và số training tokens.

## Dataset

Dataset mặc định: `undertheseanlp/UVW-2026`.

Repo dùng:

- train / validation / test split có sẵn
- pinned dataset commit
- `quality_score` threshold
- Unicode NFC
- streaming shuffle với seed
- exact duplicate removal trong sampled subset
- loại một số internal/disambiguation-like categories theo config

Dataset preparation và caveat được mô tả trong [`docs/DATASET_TOKENIZER.md`](docs/DATASET_TOKENIZER.md).

## Tokenizer

Tokenizer được **train from scratch** từ train split:

- byte-level BPE
- NFC normalization
- vocabulary 16K
- `<pad>`, `<unk>`, `<eos>`
- byte alphabet cho coverage

Repo có unit test encode/decode round-trip tiếng Việt và tokenizer metrics. Không sử dụng PhoBERT/LLaMA tokenizer có sẵn cho baseline.

## Train / validation / test đúng protocol

- tokenizer merges: train split only
- model updates: train only
- checkpoint selection: deterministic validation windows
- test: chỉ chạy cuối cùng trên best validation checkpoint

Do đó test không bị dùng để tune checkpoint trong training loop.

## Checkpoint v2

Checkpoint lưu:

- model
- optimizer
- AMP GradScaler
- step / best validation
- raw config
- Python / NumPy / Torch / CUDA RNG states
- tokenizer JSON

Tokenizer ID mapping là một phần của model, vì vậy checkpoint tự chứa tokenizer để resume không vô tình train lại một mapping khác.

## 1. Cài đặt local

Kaggle CLI hiện yêu cầu Python 3.11+.

```powershell
cd MiniViGPT
py -m pip install -r requirements.txt
py -m pip install -U kaggle
kaggle auth login
```

Local tests:

```powershell
$env:PYTHONPATH="$PWD/src"
pytest -q
```

Verification:

```powershell
python scripts/verify_repo.py --config configs/minivigpt_20m.yaml
```

## 2. Luôn chạy Kaggle debug trước

```powershell
.\scripts\kaggle_push.ps1 `
  -KaggleUsername "YOUR_KAGGLE_USERNAME" `
  -Slug "minivigpt-debug" `
  -Config "configs/minivigpt_debug.yaml"
```

Check:

```powershell
.\scripts\kaggle_status.ps1 `
  -KaggleUsername "YOUR_KAGGLE_USERNAME" `
  -Slug "minivigpt-debug"
```

Download:

```powershell
.\scripts\kaggle_download.ps1 `
  -KaggleUsername "YOUR_KAGGLE_USERNAME" `
  -Slug "minivigpt-debug"
```

Debug phải tạo được `checkpoint_best.pt`, `checkpoint_latest.pt`, `summary.json`, `metrics.jsonl`, `tokenizer.json`, `sample.txt` trước khi chạy model chính.

## 3. Train 20M trên Kaggle T4

```powershell
.\scripts\kaggle_push.ps1 `
  -KaggleUsername "YOUR_KAGGLE_USERNAME" `
  -Slug "minivigpt-20m" `
  -Config "configs/minivigpt_20m.yaml"
```

Kaggle bundle dùng `NvidiaTeslaT4` mặc định. Config `precision:auto` chọn fp16 + GradScaler trên T4; BF16 chỉ được dùng khi GPU thật sự hỗ trợ.

Chi tiết: [`docs/KAGGLE_CLI.md`](docs/KAGGLE_CLI.md).

## 4. Resume trên Kaggle

Đưa `checkpoint_latest.pt` của run trước vào một private Kaggle Dataset rồi attach khi push:

```powershell
.\scripts\kaggle_push.ps1 `
  -KaggleUsername "YOUR_KAGGLE_USERNAME" `
  -Slug "minivigpt-20m-resume" `
  -Config "configs/minivigpt_20m.yaml" `
  -DatasetSource "YOUR_KAGGLE_USERNAME/minivigpt-resume"
```

Kaggle entry tự tìm `checkpoint_latest.pt` dưới `/kaggle/input` và resume nếu config khớp.

## 5. Generate

```powershell
$env:PYTHONPATH="$PWD/src"
python -m minivigpt.generate `
  --checkpoint artifacts/kaggle/checkpoint_best.pt `
  --tokenizer artifacts/kaggle/tokenizer.json `
  --prompt "Trí tuệ nhân tạo" `
  --max-new-tokens 120 `
  --temperature 0.9 `
  --top-k 50 `
  --top-p 0.95
```

## Cấu trúc repo

```text
MiniViGPT/
├── configs/
│   ├── minivigpt_debug.yaml
│   └── minivigpt_20m.yaml
├── docs/
│   ├── THEORY_PROVENANCE.md
│   ├── ARCHITECTURE.md
│   ├── DATASET_TOKENIZER.md
│   ├── TRAINING_METHODOLOGY.md
│   ├── KAGGLE_CLI.md
│   ├── EVALUATION.md
│   ├── REPRODUCIBILITY.md
│   ├── EXPERIMENT_PROTOCOL.md
│   ├── REFERENCES.md
│   └── course/                  # Day 1 → Day 10
├── notebooks/
│   └── 00_model_smoke_test.ipynb
├── scripts/
│   ├── build_kaggle_bundle.py
│   ├── estimate_training_budget.py
│   ├── verify_repo.py
│   ├── kaggle_push.ps1
│   ├── kaggle_status.ps1
│   └── kaggle_download.ps1
├── src/minivigpt/
│   ├── config.py
│   ├── model.py
│   ├── tokenizer.py
│   ├── tokenizer_eval.py
│   ├── data.py
│   ├── train.py
│   ├── generate.py
│   └── kaggle_entry.py
├── tests/
├── pyproject.toml
└── LICENSE
```

## Khóa học 10 ngày

Bắt đầu tại [`docs/course/README.md`](docs/course/README.md). Mỗi day map trực tiếp vào class/function/test trong repo; không học một notebook riêng rồi bỏ đi.

## Research rules

Nếu muốn làm contribution sau baseline, đọc [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md). Repo yêu cầu parameter/compute/data fairness trước khi claim improvement.

## Limitations

- corpus hiện tại chủ yếu Vietnamese Wikipedia
- model nhỏ ~20M
- context 256
- training budget thấp so với foundation-model scaling
- exact dedup, chưa near-duplicate MinHash/semantic dedup
- chưa instruction-tuned
- generation không phải factual QA system
- chưa có KV cache trong baseline educational generation

## License

Code: MIT.

UVW-2026: CC BY-SA 4.0 theo dataset source. Code license không thay thế license/attribution obligations của dataset. Xem source dataset trước khi redistribute data-derived artifacts.

## References

Xem [`docs/REFERENCES.md`](docs/REFERENCES.md).
