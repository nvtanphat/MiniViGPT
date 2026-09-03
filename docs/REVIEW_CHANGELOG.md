# Technical Review — v0.1 → v0.2

## Kết luận review

v0.1 có architecture direction đúng nhưng còn ở mức **good prototype**. v0.2 nâng thành **reference-quality baseline** bằng cách sửa các điểm methodology có thể làm kết quả khó diễn giải hoặc resume không đáng tin.

## Các thay đổi quan trọng

| Vấn đề v0.1 | Vì sao chưa đủ chặt | v0.2 |
|---|---|---|
| SwiGLU hidden = 4d | Không parameter-matched với cách LLaMA dùng SwiGLU | LLaMA-style ~8/3d + rounding |
| 6 layers để đạt ~20M nhờ FFN lớn | Parameter allocation lệch | 8 layers, d=384, FFN=1024 -> 20,306,304 params |
| Init mọi Linear std=0.02 | Residual output không depth-scaled | scaled residual output projections |
| Validation random sample mỗi lần | Curve có sampling noise không kiểm soát | fixed `eval_seed` |
| Không có test split trong trainer | Dễ dùng val như final metric | test split riêng, evaluate best checkpoint cuối |
| Resume thiếu RNG | Không reproduce training sampling trajectory | save/restore Python/NumPy/Torch/CUDA RNG |
| Resume phụ thuộc tokenizer file ngoài | Token-ID mapping có thể mất/sai | tokenizer JSON embedded in checkpoint |
| Dataset first-N | Có ordering bias | streaming shuffle buffer + seed |
| Chỉ `min_chars` | Chưa tận dụng UVW metadata | quality threshold + category exclusions |
| Không dedup | duplicate có thể overweight | exact normalized-text dedup |
| Dataset dùng `main` | upstream drift | pin commit revision |
| Mixed precision bool | không phân biệt T4 fp16 và BF16 GPUs | precision auto resolution |
| AdamW non-fused | bỏ hiệu năng khi CUDA hỗ trợ | auto fused AdamW |
| Chỉ top-k generation | thiếu nucleus sampling | top-p + top-k + temperature |
| 4 unit tests | chưa test behavior/math invariants | causal prefix, RoPE norm, RMSNorm, SDPA equivalence, batch shift, deterministic sampling, LR tests |
| README ngắn | chưa đủ học/research | 10-day course + architecture/data/training/eval/repro docs |

## Những gì vẫn cố ý chưa làm

- GQA/MQA
- KV cache
- FlashAttention implementation riêng
- distributed training
- MinHash/near-duplicate web-scale dedup
- instruction tuning
- RAG/API

Lý do: đây là baseline pretraining from scratch. Các phần trên nên được thêm sau khi baseline được reproduce, không nên làm code phình toàn bộ trước khi hiểu causal LM core.
