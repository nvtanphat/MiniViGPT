# Kiến trúc MiniViGPT

## 1. Causal language modeling objective

Cho sequence token `x_1, ..., x_T`, model tối đa hóa:

`log p(x_1, ..., x_T) = Σ_t log p(x_t | x_<t)`.

Trong code, một window token tạo:

- `input = tokens[i : i+T]`
- `target = tokens[i+1 : i+T+1]`

Loss là cross-entropy trên toàn bộ `B*T` vị trí.

## 2. Tensor shape chuẩn

Ký hiệu:

- `B`: batch size
- `T`: sequence length
- `C`: model dimension
- `H`: attention heads
- `D = C/H`: head dimension
- `V`: vocabulary size

Với config 20M:

- `C = 384`
- `H = 6`
- `D = 64`
- `T = 256`
- `V = 16000`

Flow:

```text
input_ids             [B, T]
  ↓ token embedding
x                     [B, T, C]
  ↓ q/k/v projections
q, k, v               [B, H, T, D]
  ↓ RoPE on q,k
q_rot, k_rot          [B, H, T, D]
  ↓ q @ k^T
scores                 [B, H, T, T]
  ↓ causal softmax @ v
attention out          [B, H, T, D]
  ↓ concat heads
attention projection   [B, T, C]
  ↓ residual
x                     [B, T, C]
  ↓ RMSNorm + SwiGLU + residual
x                     [B, T, C]
  ↓ repeat L blocks
x                     [B, T, C]
  ↓ final RMSNorm + LM head
logits                 [B, T, V]
```

## 3. RMSNorm

MiniViGPT dùng:

`RMS(x) = sqrt(mean(x^2) + eps)`

`RMSNorm(x) = g ⊙ x / RMS(x)`

`g` là learnable scale vector shape `[C]`. Implementation chuyển activation sang fp32 khi tính variance, sau đó cast về dtype ban đầu để giảm numerical error trong fp16.

## 4. RoPE

RoPE không cộng position embedding vào `x`. Nó rotate từng pair của Q/K bằng angle phụ thuộc position. Rotation giữ norm của pair, nhưng thay dot product giữa Q và K theo relative position.

MiniViGPT precompute `cos/sin` đến `max_seq_len`. RoPE chỉ áp dụng Q/K, không áp dụng V.

## 5. Attention

Manual reference:

`A = softmax((Q K^T)/sqrt(D) + causal_mask)`

`O = A V`

Repo có hai implementation:

- `manual`: dễ đọc, đúng mục tiêu học.
- `sdpa`: `torch.nn.functional.scaled_dot_product_attention`, có thể dùng fused kernel. Q/K/V/RoPE và block structure vẫn do repo tự xây.

Debug config dùng `manual`; config train 20M dùng `sdpa` để Kaggle hiệu quả hơn.

## 6. SwiGLU

`gate = SiLU(W_gate x)`

`up = W_up x`

`ffn(x) = W_down (gate ⊙ up)`

Vì SwiGLU có 3 linear matrices, nếu giữ hidden size `4C` như FFN ReLU/GELU cổ điển thì parameter count tăng mạnh. LLaMA reference dùng effective width gần `8C/3`, sau đó round lên một multiple. MiniViGPT làm đúng quy tắc đó.

Với `C=384`, `multiple_of=256`:

`8/3 * 384 = 1024`, nên hidden dim = 1024.

## 7. Transformer block

MiniViGPT dùng pre-norm:

```text
h = x + Attention(RMSNorm(x))
out = h + SwiGLU(RMSNorm(h))
```

Không có bias trong Q/K/V/MLP linear layers. Dropout mặc định 0 cho pretraining baseline; có config field để bật khi muốn ablate.

## 8. Embedding và LM head

Token embedding có shape `[V, C]`. LM head về lý thuyết có weight `[V, C]`. Khi `tie_embeddings=true`, cả hai trỏ vào **cùng một parameter**, giảm parameter count và theo weight tying literature.

## 9. Initialization

Base matrices: `Normal(0, 0.02)`.

Hai projection đi trực tiếp vào residual stream:

- attention `out_proj`
- FFN `down_proj`

được init với std:

`0.02 / sqrt(2 * n_layers)`.

Đây là stability choice lấy từ GPT-2/nanoGPT-style practice, không phải một phần bắt buộc của LLaMA.

## 10. Parameter budget

Không đoán bằng tên file. Chạy:

```powershell
$env:PYTHONPATH="$PWD/src"
python scripts/estimate_training_budget.py --config configs/minivigpt_20m.yaml
```

Repo tính parameter count từ model thật, kể cả weight tying.
