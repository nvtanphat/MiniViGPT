# Theory Provenance — MiniViGPT không “tự chế” kiến trúc

Tài liệu này trả lời câu hỏi quan trọng nhất của repo: **mỗi kỹ thuật trong MiniViGPT đến từ đâu, đã được dùng ở đâu, và repo này thay đổi gì?**

MiniViGPT là một **educational implementation**, không tuyên bố kiến trúc mới. Mục tiêu là ghép các thành phần đã được công bố và kiểm chứng thành một decoder-only Vietnamese language model đủ nhỏ để một người học có thể đọc hết code và pretrain trên Kaggle T4.

## Bảng nguồn gốc kỹ thuật

| Thành phần | MiniViGPT dùng gì | Nguồn chính / reference implementation | Mức tương đồng |
|---|---|---|---|
| Objective | Causal next-token prediction | GPT family; decoder LM | Cùng objective chuẩn autoregressive LM |
| Attention | Scaled dot-product multi-head causal self-attention | Vaswani et al., 2017, *Attention Is All You Need* | Công thức QK^T/sqrt(d_k), softmax, weighted V |
| Causal masking | Không cho vị trí t nhìn token > t | Decoder self-attention / GPT | Đúng causal dependency |
| Pre-norm | Norm trước attention/FFN | Modern decoder practice; LLaMA reference | `x + sublayer(norm(x))` |
| RMSNorm | RMS normalization, không subtract mean | Zhang & Sennrich, 2019 | Cùng công thức, tính variance ở fp32 |
| RoPE | Rotate Q và K theo position | Su et al., 2021, RoFormer | Pairwise rotation, theta=10000 baseline |
| SwiGLU | `SiLU(W_gate x) * W_up x`, rồi W_down | Shazeer, 2020; LLaMA | Cùng gated FFN form |
| FFN width | round-up `(8/3) * d` | Meta LLaMA reference implementation | Theo cách bù parameter cho 3 projection của SwiGLU |
| Weight tying | Token embedding = LM head weight | Press & Wolf, 2016 | Cùng parameter sharing |
| Residual init | output projections std = `0.02/sqrt(2L)` | GPT-2-style practice, nanoGPT implementation | Dùng để ổn định residual stream theo depth |
| Optimizer | AdamW | Loshchilov & Hutter, 2017 | Weight decay chỉ lên matrix params |
| Betas | `(0.9, 0.95)` | GPT/nanoGPT-style LM pretraining | Baseline phổ biến, không tuyên bố tối ưu cho UVW |
| LR schedule | linear warmup + cosine decay | Common LM training practice; nanoGPT reference | Deterministic closed-form scheduler |
| Grad clipping | global norm 1.0 | Common Transformer training practice | Safety/stability baseline |
| Mixed precision | bf16 nếu GPU hỗ trợ, nếu không fp16 + GradScaler | PyTorch AMP; mixed precision literature | T4 thực tế đi fp16 |
| Tokenizer | Byte-level BPE + Unicode NFC | GPT-2-style byte-level BPE; BPE lineage | Reversible byte coverage, không cần VN word segmenter |
| Packing | Nối article bằng EOS thành token stream | Common autoregressive pretraining | Window có thể đi qua boundary nhưng nhìn thấy EOS |
| Generation | greedy / temperature / top-k / top-p | Top-p: Holtzman et al., 2019 | Sampling tiêu chuẩn |

## Những gì MiniViGPT **không** sao chép nguyên xi

### Không phải bản clone LLaMA

Repo lấy một số lựa chọn kiến trúc của LLaMA như RMSNorm, RoPE, SwiGLU và pre-norm. Nhưng MiniViGPT vẫn dùng **MHA** đầy đủ, không dùng GQA/MQA; context ngắn; tokenizer là byte-level BPE riêng cho corpus tiếng Việt; training budget rất nhỏ so với foundation model.

### Không phải nanoGPT đổi tên

nanoGPT dùng GPT-2-like block với learned positional embedding và GELU. MiniViGPT tự code RoPE, RMSNorm và SwiGLU. Tuy nhiên repo có học theo nanoGPT ở các quyết định engineering đã thành baseline tốt: AdamW grouping, beta2=0.95, cosine schedule, grad accumulation, depth-scaled residual projection init.

### Không tuyên bố 20M model là “LLM mạnh”

20M parameters chỉ là kích thước pedagogical. Nó đủ để học pipeline language-model pretraining, loss curve, tokenization, causal generation và các failure mode. Với config mặc định, tổng số token presentations vẫn thấp hơn nhiều so với compute-optimal scaling rule kiểu Chinchilla. Vì vậy hãy gọi đây là **MiniViGPT pretraining experiment**, không gọi là Vietnamese foundation model cạnh tranh PhoGPT/LLaMA.

## Vì sao dùng Byte-level BPE cho tiếng Việt?

Byte-level BPE có ba lợi ích giáo dục/thực dụng:

1. Vocabulary hữu hạn và bao phủ mọi UTF-8 input nhờ byte alphabet.
2. Không bắt buộc word segmentation tiếng Việt trước khi train.
3. Encode/decode có thể kiểm thử round-trip trực tiếp.

Nhược điểm: byte-level tokenization không bảo đảm token trùng với đơn vị hình thái hoặc “từ tiếng Việt”. Do đó repo có tokenizer metrics (`chars_per_token`, `bytes_per_token`, `unk_rate`) và khuyến khích ablation 8K/16K/32K thay vì mặc định coi 16K là tốt nhất.

## Vì sao validation phải deterministic?

Nếu mỗi lần validation lại sample một tập random khác, hai điểm loss liên tiếp khác nhau vì **cả model thay đổi lẫn sample thay đổi**. Repo tạo `torch.Generator` với `eval_seed` cố định ở mỗi lần eval, vì vậy tập window validation được giữ nhất quán giữa các checkpoint.

Test split không dùng để chọn checkpoint. Repo chọn `checkpoint_best.pt` bằng validation loss và chỉ sau khi training xong mới đánh giá test.

## Vì sao checkpoint phải giữ RNG state và tokenizer?

Resume chỉ model weight + optimizer chưa đủ để tái lập trajectory. Training batch được sample ngẫu nhiên, dropout có thể ngẫu nhiên, AMP scaler có state. Checkpoint v2 lưu:

- model state
- optimizer state
- GradScaler state
- Python RNG
- NumPy RNG
- Torch CPU RNG
- CUDA RNG states
- config
- tokenizer JSON

Tokenizer được nhúng vào checkpoint vì **ID mapping là một phần của model**. Một weight matrix embedding không có nghĩa nếu tokenizer ID mapping bị train lại khác.

## References

Xem `docs/REFERENCES.md` để có DOI/arXiv/GitHub chính thức.
