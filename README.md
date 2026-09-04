# MiniViGPT — Vietnamese Decoder-only LM From Scratch

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.3+](https://img.shields.io/badge/pytorch-2.3+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: UVW-2026](https://img.shields.io/badge/Dataset-UVW--2026-green.svg)](https://huggingface.co/datasets/undertheseanlp/UVW-2026)

**MiniViGPT** là repository thực nghiệm reproducible để tự xây dựng, huấn luyện (pretrain) từ đầu một mô hình ngôn ngữ **Decoder-only Language Model (LLM)** trên dữ liệu tiếng Việt sử dụng PyTorch.

Mục tiêu dự án là giúp lập trình viên và nhà nghiên cứu hiểu sâu toàn bộ đường ống (pipeline) huấn luyện LLM pretraining từ con số 0. Dự án không tuyên bố kiến trúc mới hay mô hình SOTA, mà tập trung vào tính minh bạch, chính xác về mặt lý thuyết và khả năng tái lập (reproducibility).

---

## 📊 Kết quả

| Run | Tham số | Ngữ cảnh | Token | tok/param | Val ppl | **Test ppl** | GPU-giờ |
|---|---|---|---|---|---|---|---|
| [`run_20m_8k`](results/run_20m_8k/) | 20,306,304 | 256 | 98.3M | 4.84 | 19.86 | **19.85** | ~2.0 |
| [`run_20m_33k`](results/run_20m_33k/) | 20,306,304 | 512 | 405.5M | 19.97 | 12.95 | **13.98** | ~8.8 |

Chinchilla-optimal (≈20 token/tham số) giảm perplexity **30%**, đổi lại 4,4×
GPU-giờ. Test đo **một lần duy nhất** ở cuối, từ checkpoint chọn theo validation.

Val loss vẫn còn giảm khi kết thúc, nhưng rất chậm — ước tính 66.000 bước chỉ được ppl ~11,1, không đáng chi phí.

### Model học được gì

Bài kiểm tra ngữ pháp — model chấm điểm câu đúng thấp hơn (tốt hơn) câu bị đảo trật tự:

```
[✓] 2.03 vs 3.91   Hà Nội là thủ đô của Việt Nam.
[✓] 4.37 vs 7.85   Tôi đi học vào buổi sáng.
[✓] 4.38 vs 6.52   Trời hôm nay mưa rất to.
[✓] 4.02 vs 6.85   Học sinh đang làm bài tập.
                                              → 5/5 đúng
```

Tokenizer byte-level BPE 16K: **0% `<unk>`**, khôi phục chính xác 100%,
~3.6 ký tự/token trên tiếng Việt.

### Sinh văn bản

```
▸ Thành phố Hồ Chí Minh nằm ở
  phía nam của tỉnh, có diện tích 15,96 km², dân số là 1.538 người,
  mật độ dân số đạt 1.087 người/km².

  Lịch sử
  Ngày 9 tháng 1 năm 2003, Chính phủ ban hành Nghị định số 11/2003/NĐ-CP
  về việc thành lập
```

Ngữ pháp và dấu tiếng Việt chuẩn. Model tự sinh đúng cấu trúc mục Wikipedia
(`Lịch sử`, `Địa lý`, `Tham khảo`), đúng quy cách số nghị định, đơn vị đo,
mật độ dân số.

Nhưng **nội dung là bịa** — TP.HCM không phải "phía nam của tỉnh", dân số
không phải 1.538 người, nghị định đó không tồn tại.

Đó là giới hạn cứng ở 20M tham số. Model học được *hình thức* của tiếng Việt
Wikipedia rất tốt, nhưng không đủ dung lượng cho *nội dung*. Perplexity thấp
đo khả năng đoán token kế tiếp, không đo tính đúng đắn của thông tin.
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

Repo có hai cấu hình cho cùng một kiến trúc 20M tham số:

| Tham số | [`minivigpt_20m.yaml`](configs/minivigpt_20m.yaml) | [`minivigpt_20m_chinchilla.yaml`](configs/minivigpt_20m_chinchilla.yaml) |
|---|---:|---:|
| **Vocabulary Size** | 16,000 byte-BPE | 16,000 byte-BPE |
| **Context Length (`max_seq_len`)** | 256 | **512** |
| **Model Dimension ($d_{model}$)** | 384 | 384 |
| **Num Layers** | 8 | 8 |
| **Num Attention Heads** | 6 | 6 |
| **Head Dimension** | 64 | 64 |
| **SwiGLU Hidden Dimension** | 1,024 | 1,024 |
| **Tổng số tham số** | **20,306,304** | **20,306,304** |
| **Batch × grad accum** | 12 × 4 | 6 × 4 |
| **Tokens per update** | 12,288 | 12,288 |
| **Total Max Steps** | 8,000 | **33,000** |
| **Planned Tokens Trained** | 98,304,000 | **405,504,000** |
| **Tokens / Parameter Ratio** | ~4.84 | **~19.97** |
| **GPU-giờ (T4)** | ~2.0 | ~8.8 |
| **Test perplexity** | 19.85 | **13.98** |

`minivigpt_20m.yaml` là ngân sách "chạy nhanh cho biết" — 2 giờ là có kết quả.
`minivigpt_20m_chinchilla.yaml` theo tỉ lệ Chinchilla (≈20 token/tham số) và
cho perplexity thấp hơn 30%, đổi lại 4,4× GPU-giờ.

`tokens per update` giữ nguyên 12,288 ở cả hai để lịch learning rate không đổi
khi ngữ cảnh tăng gấp đôi.

Kiểm tra ngân sách huấn luyện bằng script:
```powershell
$env:PYTHONPATH="$PWD/src"
python scripts/estimate_training_budget.py --config configs/minivigpt_20m_chinchilla.yaml
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

### 4. Huấn luyện Mô hình GPU

```powershell
$env:PYTHONPATH="$PWD/src"
python -m minivigpt.train --config configs/minivigpt_20m_chinchilla.yaml
```

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
  --checkpoint artifacts/kaggle_run_33k/checkpoint_best.pt `
  --tokenizer artifacts/kaggle_run_33k/tokenizer.json `
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
