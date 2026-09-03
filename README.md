# MiniViGPT — Vietnamese Decoder-only LM From Scratch

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.3+](https://img.shields.io/badge/pytorch-2.3+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: UVW-2026](https://img.shields.io/badge/Dataset-UVW--2026-green.svg)](https://huggingface.co/datasets/undertheseanlp/UVW-2026)

**MiniViGPT** là repository giáo dục và thực nghiệm reproducible để tự xây dựng, huấn luyện (pretrain) từ đầu một mô hình ngôn ngữ **Decoder-only Language Model (LLM)** trên dữ liệu tiếng Việt sử dụng PyTorch.

Mục tiêu dự án là giúp lập trình viên và nhà nghiên cứu hiểu sâu toàn bộ đường ống (pipeline) huấn luyện LLM pretraining từ con số 0. Dự án không tuyên bố kiến trúc mới hay mô hình SOTA, mà tập trung vào tính minh bạch, chính xác về mặt lý thuyết và khả năng tái lập (reproducibility).

---


## 🏗️ Kiến trúc & Pipeline tổng quan

```text
                       UVW-2026 Vietnamese Wikipedia (Pinned Revision)
                                            ↓
                        Streaming Shuffle + Quality Filtering + Dedup
                                            ↓
                      NFC Normalization + Byte-level BPE Tokenizer (16K)
                                            ↓
                        Binary Packed Token Streams (uint16 + EOS)
                                            ↓
                           MiniViGPT (20,306,304 Parameters)
                            ├── Pre-RMSNorm
                            ├── Rotary Position Embedding (RoPE)
                            ├── Multi-Head Causal Self-Attention (SDPA / Manual)
                            ├── SwiGLU FFN (LLaMA-style ~8/3·d width)
                            ├── Residual Connections & Depth-Scaled Init
                            └── Tied Input / Output Embeddings
                                            ↓
                         AdamW + Cosine Warmup + AMP (fp16/bf16)
                                            ↓
                        Deterministic Validation → Best Checkpoint
                                            ↓
                     Greedy / Temperature / Top-K / Top-P Generation
```

### Provenance Kỹ thuật
* **Causal Attention**: Scaled Dot-Product Attention (*Vaswani et al.*)
* **Pre-RMSNorm**: Root Mean Square Normalization (*Zhang & Sennrich*)
* **RoPE**: Rotary Position Embedding (*Su et al.*)
* **SwiGLU**: Swish-Gated Linear Unit (*Shazeer*)
* **Tied Embeddings**: Weight Tying (*Press & Wolf*)
* **Depth-Scaled Init**: Residual Scaling $0.02 / \sqrt{2 \cdot N_{layers}}$ (*GPT-2 / nanoGPT style*)

👉 *Đọc chi tiết về nguồn gốc lý thuyết tại [`docs/THEORY_PROVENANCE.md`](docs/THEORY_PROVENANCE.md) và kiến trúc tại [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).*

---

## ⚙️ Cấu hình Mô hình & Training Budget

Cấu hình mặc định trong [`configs/minivigpt_20m.yaml`](configs/minivigpt_20m.yaml):

| Tham số | Giá trị |
|---|---:|
| **Vocabulary Size** | 16,000 byte-BPE tokens |
| **Context Length (`max_seq_len`)** | 256 |
| **Model Dimension ($d_{model}$)** | 384 |
| **Num Layers** | 8 |
| **Num Attention Heads** | 6 |
| **Head Dimension** | 64 |
| **SwiGLU Hidden Dimension** | 1,024 |
| **Tổng số tham số (Parameters)** | **20,306,304** |
| **Batch Size per update** | 48 (12 x 4 accum) |
| **Tokens per update** | 12,288 |
| **Total Max Steps** | 8,000 |
| **Planned Tokens Trained** | 98,304,000 |
| **Tokens / Parameter Ratio** | ~4.84 |

Kiểm tra ngân sách huấn luyện bằng script:
```powershell
$env:PYTHONPATH="$PWD/src"
python scripts/estimate_training_budget.py --config configs/minivigpt_20m.yaml
```

---

## 🛠️ Hướng dẫn sử dụng

### 1. Cài đặt Môi trường Local

Yêu cầu **Python 3.11+**:

```powershell
# Cài đặt thư viện phụ thuộc
pip install -r requirements.txt

# Kiểm tra cài đặt và bộ test
$env:PYTHONPATH="$PWD/src"
pytest -q

# Xác minh cấu hình mô hình
python scripts/verify_repo.py --config configs/minivigpt_20m.yaml
```

---

### 2. Tải & Tiền xử lý Dataset (Dataset & Tokenizer)

Để tải dataset `undertheseanlp/UVW-2026` từ Hugging Face, huấn luyện Tokenizer Byte-BPE 16K từ đầu và tạo tập binary token streams (`train.bin`, `validation.bin`, `test.bin`):

```powershell
$env:PYTHONPATH="$PWD/src"
python scripts/prepare_dataset.py --config configs/minivigpt_20m.yaml --output-dir artifacts/local
```

> [!TIP]
> Script sẽ lưu kết quả vào [`artifacts/local/`](artifacts/local/). Bạn cũng có thể dùng `configs/minivigpt_debug.yaml` để chạy thử nghiệm nhanh với dữ liệu nhỏ.

---

### 3. Huấn luyện Mô hình Local

```powershell
$env:PYTHONPATH="$PWD/src"
python -m minivigpt.train --config configs/minivigpt_debug.yaml
```

---

### 4. Huấn luyện GPU trên Kaggle T4

Repository hỗ trợ đẩy tiến trình training lên Kaggle GPU T4 miễn phí.

#### 🔑 Đăng nhập Kaggle API:
```powershell
pip install -U kaggle
kaggle auth login
```

#### 🐞 Bước 1: Run Debug Kiểm thử trước
```powershell
.\scripts\kaggle_push.ps1 `
  -KaggleUsername "YOUR_KAGGLE_USERNAME" `
  -Slug "minivigpt-debug" `
  -Config "configs/minivigpt_debug.yaml"
```

Kiểm tra trạng thái & Tải kết quả Debug:
```powershell
.\scripts\kaggle_status.ps1 -KaggleUsername "YOUR_KAGGLE_USERNAME" -Slug "minivigpt-debug"
.\scripts\kaggle_download.ps1 -KaggleUsername "YOUR_KAGGLE_USERNAME" -Slug "minivigpt-debug"
```

#### 🚀 Bước 2: Train Model 20M chính thức
```powershell
.\scripts\kaggle_push.ps1 `
  -KaggleUsername "YOUR_KAGGLE_USERNAME" `
  -Slug "minivigpt-20m" `
  -Config "configs/minivigpt_20m.yaml"
```

#### 🔄 Bước 3: Resume Training trên Kaggle
Đưa `checkpoint_latest.pt` của lần chạy trước vào private Kaggle Dataset và đính kèm khi push:
```powershell
.\scripts\kaggle_push.ps1 `
  -KaggleUsername "YOUR_KAGGLE_USERNAME" `
  -Slug "minivigpt-20m-resume" `
  -Config "configs/minivigpt_20m.yaml" `
  -DatasetSource "YOUR_KAGGLE_USERNAME/minivigpt-resume"
```

---

### 5. Sinh Văn bản (Text Generation)

Sử dụng checkpoint tốt nhất (`checkpoint_best.pt`) để sinh văn bản tiếng Việt:

```powershell
$env:PYTHONPATH="$PWD/src"
python -m minivigpt.generate `
  --checkpoint artifacts/local/checkpoint_best.pt `
  --tokenizer artifacts/local/tokenizer.json `
  --prompt "Trí tuệ nhân tạo" `
  --max-new-tokens 120 `
  --temperature 0.9 `
  --top-k 50 `
  --top-p 0.95
```

---

## 📂 Cấu trúc Repository

```text
MiniViGPT/
├── configs/                  # Các file cấu hình YAML (20M, Debug)
│   ├── minivigpt_20m.yaml
│   └── minivigpt_debug.yaml
├── docs/                     # Tài liệu lý thuyết, kiến trúc & báo cáo
│   ├── THEORY_PROVENANCE.md
│   ├── ARCHITECTURE.md
│   ├── DATASET_TOKENIZER.md
│   ├── TRAINING_METHODOLOGY.md
│   ├── KAGGLE_CLI.md
│   ├── EVALUATION.md
│   ├── REPRODUCIBILITY.md
│   ├── EXPERIMENT_PROTOCOL.md
│   ├── REFERENCES.md
│   └── course/               # Giáo trình 10 ngày từ Zero đến LLM Pretraining
├── notebooks/                # Jupyter Notebooks kiểm thử nhanh
│   └── 00_model_smoke_test.ipynb
├── scripts/                  # Bộ công cụ tự động hóa & CLI
│   ├── prepare_dataset.py     # Tải & tiền xử lý dataset local
│   ├── build_kaggle_bundle.py# Tạo bundle cho Kaggle
│   ├── estimate_training_budget.py
│   ├── verify_repo.py
│   ├── kaggle_push.ps1
│   ├── kaggle_status.ps1
│   └── kaggle_download.ps1
├── src/minivigpt/            # Mã nguồn chính của mô hình
│   ├── config.py             # Dataclass cấu hình mô hình
│   ├── model.py              # Định nghĩa Transformer Decoder
│   ├── tokenizer.py          # Huấn luyện & Load Byte-BPE Tokenizer
│   ├── tokenizer_eval.py     # Đánh giá tỷ lệ nén tokenizer
│   ├── data.py               # Hugging Face Streaming & Binary Packing
│   ├── train.py              # Vòng lặp huấn luyện, AMP, validation & test
│   ├── generate.py           # Thuật toán lấy mẫu sinh chuỗi văn bản
│   └── kaggle_entry.py       # Entrypoint chạy tự động trên Kaggle GPU
├── tests/                    # Pytest unit tests cho toàn bộ pipeline
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🎓 Khóa học 10 ngày (Zero to LLM)

Repository kèm theo khóa học 10 ngày chi tiết tại [`docs/course/README.md`](docs/course/README.md). Mỗi ngày thực hành tương ứng trực tiếp với code và unit test trong repo:

* **Day 1**: Token, Embedding & Next-Token Objective
* **Day 2**: Q/K/V & Causal Self-Attention
* **Day 3**: Pre-RMSNorm, RoPE & SwiGLU FFN
* **Day 4**: Ghép Cấu trúc Transformer Decoder (MiniViGPT)
* **Day 5**: UVW-2026 Dataset & Vietnamese Byte-BPE Tokenizer
* **Day 6**: Training Methodology, Loss & Deterministic Checkpoint
* **Day 7**: Tích hợp Kaggle & Huấn luyện GPU T4
* **Day 9**: Thiết kế Ablation Experiments & Protocol Research
* **Day 10**: Reproduce, Benchmark & Publish Repository



---

## 📄 License & Attribution

* **Code License**: [MIT License](LICENSE).
* **Dataset License**: Dataset `undertheseanlp/UVW-2026` tuân theo giấy phép **CC BY-SA 4.0** từ tác giả gốc.
