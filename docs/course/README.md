# Khóa 10 ngày — Build MiniViGPT từ A đến Z

Đây là lộ trình học **gắn trực tiếp với repo**, không phải 10 bài lý thuyết rời rạc. Mỗi ngày bạn phải đạt một checkpoint có thể kiểm tra bằng code/test.

| Day | Chủ đề | Code chính | Checkpoint |
|---|---|---|---|
| 1 | Token, embedding, next-token objective | `tokenizer.py`, `model.py` | giải thích `[B,T] -> [B,T,C] -> [B,T,V]` |
| 2 | Q/K/V + causal attention | `CausalSelfAttention` | chứng minh future token không ảnh hưởng prefix |
| 3 | RMSNorm + RoPE + SwiGLU + residual | `RMSNorm`, `RotaryEmbedding`, `SwiGLU` | hiểu một modern decoder block |
| 4 | Ghép full MiniViGPT | `MiniViGPT`, config | tính đúng parameter budget |
| 5 | UVW-2026 + Vietnamese BPE | `data.py`, `tokenizer.py` | tạo tokenizer + train/val/test token stream |
| 6 | Training methodology | `train.py` | loss giảm + deterministic val + checkpoint |
| 9 | Ablation như research | configs + `EXPERIMENT_PROTOCOL.md` | thiết kế experiment fair |

| 10 | Reproduce + publish | toàn repo | README/model card/result table có thể audit |


**Phase 2 sau khóa này:** SFT/LoRA, RAG, serving/API. Không nhồi các phần đó vào baseline from-scratch trước khi pretraining protocol chạy đúng.
