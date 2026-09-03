# Lộ trình 8 bước — Build MiniViGPT từ A đến Z

Đây là lộ trình **gắn trực tiếp với repo**. Mỗi phần tương ứng với một mốc kiểm tra bằng code/test:

| Day | Chủ đề | Code chính | Checkpoint |
|---|---|---|---|
| 1 | Token, embedding, next-token objective | `tokenizer.py`, `model.py` | giải thích `[B,T] -> [B,T,C] -> [B,T,V]` |
| 2 | Q/K/V + causal attention | `CausalSelfAttention` | chứng minh future token không ảnh hưởng prefix |
| 3 | RMSNorm + RoPE + SwiGLU + residual | `RMSNorm`, `RotaryEmbedding`, `SwiGLU` | hiểu một modern decoder block |
| 4 | Ghép full MiniViGPT | `MiniViGPT`, config | tính đúng parameter budget |
| 5 | UVW-2026 + Vietnamese BPE | `data.py`, `tokenizer.py` | tạo tokenizer + train/val/test token stream |
| 6 | Training methodology | `train.py` | loss giảm + deterministic val + checkpoint |
| 7 | Ablation như research | configs + `EXPERIMENT_PROTOCOL.md` | thiết kế experiment fair |
| 8 | Reproduce + publish | toàn repo | README/model card/result table có thể audit |

**Phase 2 sau bước này:** SFT/LoRA, RAG, serving/API. Không nhồi các phần đó vào baseline from-scratch trước khi pretraining protocol chạy đúng.
