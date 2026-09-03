# Day 4 — Ghép Full MiniViGPT và Audit Parameter Count

## Mục tiêu

Hiểu model hoàn chỉnh, không dùng “20M” như một label marketing.

## 1. Stack

Full model:

```text
Token embedding
↓
TransformerBlock × 8
↓
Final RMSNorm
↓
LM head
```

Không có learned absolute position embedding vì position được đưa qua RoPE trong mỗi attention layer.

## 2. Weight tying

`lm_head.weight = token_embedding.weight`.

Điều này giảm một matrix V×C. Unit test kiểm tra hai weight thực sự cùng data pointer, không chỉ có giá trị ban đầu giống nhau.

## 3. Config 20M

Đọc `configs/minivigpt_20m.yaml`. Tự tính:

- head dim = 384/6 = 64
- FFN hidden = 1024
- layers = 8
- vocab = 16K
- context = 256

Sau đó chạy:

```powershell
$env:PYTHONPATH="$PWD/src"
python scripts/estimate_training_budget.py --config configs/minivigpt_20m.yaml
```

Tin con số script, không tin tên config nếu hai thứ lệch nhau.

## 4. Forward trace

Tạo random input:

```python
cfg = MiniViGPTConfig(vocab_size=16000, max_seq_len=256, dim=384, n_layers=8, n_heads=6)
model = MiniViGPT(cfg)
x = torch.randint(0, 16000, (2, 32))
logits, _ = model(x)
print(logits.shape)
```

Expected `[2,32,16000]`.

## 5. Tại sao model nhỏ vẫn hữu ích?

Bạn đang học invariant của architecture/training. Một 20M model vẫn có:

- tokenization problem
- causal objective
- optimizer dynamics
- overfit/underfit
- checkpointing
- generation distribution
- scaling tradeoffs

Những thứ này chuyển được sang model lớn. Điểm không chuyển trực tiếp là quality level và compute system complexity.

## 6. Không gọi model này là “LLaMA 20M”

Nó dùng một số design của LLaMA nhưng không có full LLaMA tokenizer/data/GQA/training recipe. Tên MiniViGPT rõ ràng hơn.

## Checkpoint Day 4

Bạn pass khi có thể ước lượng parameter nằm nhiều ở embedding, attention hay FFN, và giải thích weight tying ảnh hưởng count thế nào.
