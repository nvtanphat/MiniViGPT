# Day 2 — Self-Attention, Q/K/V và Causal Mask

## Mục tiêu

Hiểu chính xác vì sao model Day 1 không có context và attention giải quyết điều gì.

## 1. Q, K, V không phải “question/answer” cố định

Q/K/V là ba learned linear projections của cùng hidden states X:

`Q = X W_Q`, `K = X W_K`, `V = X W_V`.

Trực giác “query tìm key để lấy value” hữu ích, nhưng semantic thật được học bằng gradient.

## 2. Shape

Sau reshape:

```text
Q,K,V: [B,H,T,D]
D = C/H
```

Config 20M: C=384, H=6 -> D=64.

Attention score:

`Q @ K^T -> [B,H,T,T]`.

Mỗi row là một query token, mỗi column là một key token.

## 3. Vì sao chia sqrt(D)?

Dot product variance tăng theo dimension. Scaling `1/sqrt(D)` giữ score ở regime softmax không quá bão hòa. Đây là scaled dot-product attention từ Transformer paper.

## 4. Causal mask

Ở position t, decoder LM không được nhìn token > t. Nếu cho nhìn tương lai, model có thể cheat khi training next-token objective.

Mask dạng lower triangle:

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

Future scores bị set `-inf` trước softmax.

## 5. Test tốt hơn việc “nhìn mask”

Repo có test **causal prefix invariance**:

- tạo sequence A và B giống nhau ở prefix 0..7
- thay toàn bộ future 8..11
- logits positions 0..7 phải giống nhau

Đây là behavioral test mạnh hơn chỉ assert buffer mask đúng hình tam giác.

Chạy:

```powershell
pytest -q tests/test_model_theory.py::test_causal_prefix_invariance_to_future_tokens
```

## 6. Manual vs SDPA

`attention_impl=manual` dùng chính công thức bạn học. `sdpa` dùng PyTorch fused operation nhưng input Q/K/V/RoPE và causal semantics không đổi.

Repo còn test hai path gần tương đương numerical khi dropout=0 và weight giống nhau.

## 7. Lab đọc code

Đọc theo thứ tự:

1. `_project_qkv`
2. `_manual_attention`
3. `forward`

Ghi shape bên cạnh từng dòng `.view`, `.transpose`, matmul. Nếu không tự trace shape được thì chưa chuyển Day 3.

## 8. Bài tập

Với B=16,T=256,C=768,H=12:

- D?
- Q sau transpose?
- K^T?
- score?
- weights?
- `weights @ V`?
- merge heads?

Sau đó tính số phần tử attention matrix `B*H*T*T` và giải thích vì sao context length tăng làm attention tốn memory theo T².

## Checkpoint Day 2

Bạn pass khi có thể giải thích **causality bằng dependency**, không chỉ thuộc “GPT dùng mask”.
