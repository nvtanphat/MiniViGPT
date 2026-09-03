# Day 3 — RMSNorm, RoPE, SwiGLU, Residual: Modern Decoder Block

## Mục tiêu

Từ attention đơn lẻ thành block gần LLaMA-style.

## 1. RMSNorm

RMSNorm chuẩn hóa magnitude theo root-mean-square, không subtract mean như LayerNorm. Repo tính variance ở fp32 rồi cast về activation dtype.

Test:

```powershell
pytest -q tests/test_model_theory.py::test_rmsnorm_preserves_shape_and_normalizes_rms
```

## 2. Residual + pre-norm

Block:

```python
x = x + attention(rmsnorm(x))
x = x + ffn(rmsnorm(x))
```

Residual tạo đường identity cho information/gradient. Pre-norm đặt normalization trước sublayer.

## 3. RoPE

Self-attention nếu không có positional information không biết token đứng ở position nào theo cách cần cho sequence modeling. RoPE rotate Q/K theo position thay vì cộng learned position vector vào embedding.

Repo có test norm preservation của rotation. Rotation không giữ từng coordinate, nhưng giữ Euclidean norm của vector pair.

```powershell
pytest -q tests/test_model_theory.py::test_rope_preserves_vector_norm_per_position
```

## 4. SwiGLU

FFN:

`SiLU(W_gate x) * (W_up x) -> W_down`.

Đây không chỉ là `Linear -> activation -> Linear`; gating tạo interaction multiplicative giữa hai projection.

## 5. Vì sao hidden dim không dùng 4C?

Classic Transformer FFN thường có hidden 4C với 2 projection. SwiGLU có 3 projection. LLaMA reference giảm hidden width về gần `8C/3` rồi round lên hardware multiple để parameter budget hợp lý.

Repo function:

```python
llama_swiglu_hidden_dim(dim, multiple_of)
```

Với C=384 -> 1024.

## 6. Scaled residual init

Nếu mọi residual output projection đều std=0.02, nhiều layer có thể làm residual stream variance tăng. Repo init `attn.out_proj` và `ffn.down_proj` nhỏ hơn theo `1/sqrt(2L)` như GPT-2/nanoGPT-style practice.

Đây là engineering stability choice; đừng nhầm nó với RoPE/RMSNorm paper.

## 7. Lab

Chạy toàn bộ theory tests:

```powershell
pytest -q tests/test_model_theory.py
```

Sau đó vẽ block bằng tay và ghi shape `[B,T,C]` ở mọi residual add. Hai tensor cộng nhau bắt buộc cùng shape.

## Checkpoint Day 3

Bạn pass khi phân biệt được:

- attention = communication giữa tokens
- FFN = per-token nonlinear transformation
- RoPE = positional signal cho Q/K
- RMSNorm = activation scale normalization
- residual = identity path
