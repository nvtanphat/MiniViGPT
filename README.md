# MiniViGPT — Vietnamese Decoder-only LM From Scratch

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.3+](https://img.shields.io/badge/pytorch-2.3+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: UVW-2026](https://img.shields.io/badge/Dataset-UVW--2026-green.svg)](https://huggingface.co/datasets/undertheseanlp/UVW-2026)

**MiniViGPT** là repository thực nghiệm reproducible để tự xây dựng, huấn luyện (pretrain) từ đầu một mô hình ngôn ngữ **Decoder-only Language Model (LLM)** trên dữ liệu tiếng Việt sử dụng PyTorch.

Mục tiêu dự án là giúp lập trình viên và nhà nghiên cứu hiểu sâu toàn bộ đường ống (pipeline) huấn luyện LLM pretraining từ con số 0. Dự án không tuyên bố kiến trúc mới hay mô hình SOTA, mà tập trung vào tính minh bạch, chính xác về mặt lý thuyết và khả năng tái lập (reproducibility).

---

## 📊 Kết quả

| Model | Tham số | Ngữ cảnh | Token huấn luyện | tok/param | Val loss | **Perplexity** | GPU-giờ |
|---|---|---|---|---|---|---|---|
| `run_20m_8k` | 20,306,304 | 256 | 98.3M | 4.84 | 2.9886 | **19.85** | ~2.0 (T4) |

Test loss 2.9881 / perplexity 19.85 — đo **một lần duy nhất** ở cuối, từ
checkpoint chọn theo validation. Val và test gần như trùng khít (2.9886 vs
2.9881): không overfit.

Chi phí: 2 giờ trên Kaggle T4 miễn phí.

### Model học được gì

Bài kiểm tra ngữ pháp — model chấm điểm câu đúng thấp hơn (tốt hơn) câu bị đảo trật tự:

```
[✓] 2.04 vs 3.88   Hà Nội là thủ đô của Việt Nam.
[✓] 4.35 vs 7.07   Tôi đi học vào buổi sáng.
[✓] 4.55 vs 7.09   Học sinh đang làm bài tập.
                                              → 5/5 đúng
```

Tokenizer byte-level BPE 16K: **0% `<unk>`**, khôi phục chính xác 100%,
~3.6 ký tự/token trên tiếng Việt.

### Sinh văn bản

```
▸ Thành phố Hồ Chí Minh nằm ở
  vùng Đồng bằng sông Hồng trên núi Đồng bằng sông Hồng.

  Lịch sử
  Vị trí địa lý
  Thành phố Hồ Chí Minh thuộc tỉnh Hà Tĩnh. Đây là một trong những
  địa điểm du lịch đặc biệt quan trọng
```

Ngữ pháp và dấu tiếng Việt chuẩn, tự sinh đúng cấu trúc mục của Wikipedia
(`Lịch sử`, `Vị trí địa lý`, `Tham khảo`). Nhưng **địa danh, tên riêng và
sự kiện đều là bịa** — "TP.HCM thuộc tỉnh Hà Tĩnh" là sai hoàn toàn.

Đó là điều phải xảy ra ở 20M tham số với 4.84 token/tham số. Model học được
*hình thức* của tiếng Việt Wikipedia, không học được *nội dung*. Perplexity
thấp đo khả năng đoán token kế tiếp, không đo tính đúng đắn của thông tin.
**Đừng dùng output làm nguồn tin.**

Chạy thử: `python scripts/demo.py` · Chat: `python scripts/chat.py`

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

#### 📦 Bước 1: Đẩy source và corpus thành Kaggle Dataset
Notebook chạy trên Kaggle nạp code từ một dataset, và corpus đã tokenize từ một
dataset khác — nhờ vậy mỗi lần chạy bỏ qua được ~30 phút stream + tokenize.

```powershell
# source + config
New-Item -ItemType Directory -Force $env:TEMP\mvg-src | Out-Null
Copy-Item -Recurse -Force src\minivigpt $env:TEMP\mvg-srcCopy-Item configs\minivigpt_20m.yaml $env:TEMP\mvg-src\config.yaml
Remove-Item -Recurse -Force $env:TEMP\mvg-src\minivigpt\__pycache__ -EA SilentlyContinue
# tạo dataset-metadata.json với id "<user>/minivigpt-src", rồi:
kaggle datasets create -p $env:TEMP\mvg-src --dir-mode zip

# corpus đã tokenize (chỉ cần một lần, ~239 MB)
kaggle datasets create -p artifacts\local
```

`--dir-mode zip` là bắt buộc — thiếu nó CLI in `Skipping folder: minivigpt` và
không upload gì ngoài các file rời.

#### 🚀 Bước 2: Push notebook và train
```powershell
kaggle kernels push -p scripts\kaggle --accelerator NvidiaTeslaT4
kaggle kernels status <user>/minivigpt-train
```

`--accelerator` (và `machine_shape` trong `kernel-metadata.json`) là bắt buộc:
nếu để Kaggle tự chọn, bạn có thể nhận P100 (`sm_60`) mà bản PyTorch trong image
Kaggle không còn hỗ trợ, và mọi phép tính CUDA sẽ lỗi ngay.

#### 🔄 Bước 3: Resume Training
Kaggle giới hạn 9 giờ mỗi session. Lưu output lần chạy trước thành dataset rồi
đính kèm — notebook tự tìm `checkpoint_latest.pt` dưới `/kaggle/input` và tiếp tục.

Chi tiết và các cạm bẫy khác: [`scripts/kaggle/README.md`](scripts/kaggle/README.md).

---

### 5. Sinh Văn bản & Demo

**Chat tương tác** — gõ một câu tiếng Việt, model viết tiếp:

```powershell
python scripts/chat.py
```

Lệnh trong phiên: `/temp 0.9` đổi nhiệt độ, `/len 200` đổi độ dài, `/quit` thoát.
Sinh một lần rồi thoát: `python scripts/chat.py --prompt "Hà Nội là"`.

**Demo tổng hợp** — trình bày năng lực đo được của model:

```powershell
python scripts/demo.py
```

In ra kiến trúc, chất lượng tokenizer, bài kiểm tra phân biệt ngữ pháp
đúng/sai, ví dụ sinh văn bản, và đường cong perplexity.

**Gọi trực tiếp module:**

```powershell
$env:PYTHONPATH="$PWD/src"
python -m minivigpt.generate `
  --checkpoint artifacts/kaggle_run/checkpoint_best.pt `
  --tokenizer artifacts/kaggle_run/tokenizer.json `
  --prompt "Trí tuệ nhân tạo" `
  --max-new-tokens 120 --temperature 0.8 --top-k 50 --top-p 0.95
```

> **Lưu ý:** đây là base model pretrain, không phải trợ lý hỏi-đáp. Nó *viết tiếp*
> văn bản chứ không *trả lời câu hỏi*. Ngữ pháp và văn phong tiếng Việt chuẩn,
> nhưng **tên riêng, ngày tháng và sự kiện đều là bịa** — không dùng làm nguồn tin.

---

## 📂 Cấu trúc Repository

```text
MiniViGPT/
├── configs/                  # Các file cấu hình YAML (20M, Debug)
│   ├── minivigpt_20m.yaml
│   ├── minivigpt_20m_chinchilla.yaml  # 33k bước, ngữ cảnh 512
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
│   └── course/               # Hướng dẫn chi tiết từng phần từ Zero đến LLM Pretraining
├── notebooks/                # Jupyter Notebooks kiểm thử nhanh
│   └── 00_model_smoke_test.ipynb
├── scripts/                  # Bộ công cụ tự động hóa & CLI
│   ├── prepare_dataset.py     # Tải & tiền xử lý dataset local
│   ├── chat.py                # Chat tương tác với checkpoint đã train
│   ├── demo.py                # Demo năng lực model (ngữ pháp, tokenizer, ppl)
│   ├── estimate_training_budget.py
│   ├── verify_repo.py
│   ├── kaggle_status.ps1
│   ├── kaggle_download.ps1
│   └── kaggle/                # Notebook + metadata để train trên Kaggle GPU
│       ├── minivigpt_train.ipynb
│       ├── kernel-metadata.json
│       └── README.md
├── src/minivigpt/            # Mã nguồn chính của mô hình
│   ├── config.py             # Dataclass cấu hình mô hình
│   ├── model.py              # Định nghĩa Transformer Decoder
│   ├── tokenizer.py          # Huấn luyện & Load Byte-BPE Tokenizer
│   ├── tokenizer_eval.py     # Đánh giá tỷ lệ nén tokenizer
│   ├── data.py               # Hugging Face Streaming & Binary Packing
│   ├── train.py              # Vòng lặp huấn luyện, AMP, validation & test
│   └── generate.py           # Thuật toán lấy mẫu sinh chuỗi văn bản
├── results/                  # Số liệu các lần train (commit được)
│   └── run_20m_8k/
├── tests/                    # Pytest unit tests cho toàn bộ pipeline
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📖 Hướng dẫn chi tiết (Zero to LLM)

Repository kèm theo hướng dẫn thực hành chi tiết tại [`docs/course/README.md`](docs/course/README.md). Mỗi phần thực hành tương ứng trực tiếp với code và unit test trong repo:

* **Day 1**: Token, Embedding & Next-Token Objective
* **Day 2**: Q/K/V & Causal Self-Attention
* **Day 3**: Pre-RMSNorm, RoPE & SwiGLU FFN
* **Day 4**: Ghép Cấu trúc Transformer Decoder (MiniViGPT)
* **Day 5**: UVW-2026 Dataset & Vietnamese Byte-BPE Tokenizer
* **Day 6**: Training Methodology, Loss & Deterministic Checkpoint
* **Day 7**: Thiết kế Ablation Experiments & Protocol Research
* **Day 8**: Reproduce, Benchmark & Publish Repository



---

## 📄 License & Attribution

* **Code License**: [MIT License](LICENSE).
* **Dataset License**: Dataset `undertheseanlp/UVW-2026` tuân theo giấy phép **CC BY-SA 4.0** từ tác giả gốc.
