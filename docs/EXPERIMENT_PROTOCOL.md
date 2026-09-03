# Research / Ablation Protocol

MiniViGPT là baseline. Nếu muốn biến thành đồ án/research nhỏ, contribution nên bắt đầu **sau khi reproduce baseline ổn định**.

## Phase 0 — Reproduce

Pass debug config, sau đó chạy 20M baseline. Không đổi architecture trước khi có:

- loss giảm ổn định
- val loss giảm
- checkpoint/resume chạy
- test summary
- generation có cấu trúc tiếng Việt tốt dần

## Phase 1 — Tokenizer ablation

So 8K / 16K / 32K vocabulary.

Giữ cố định tối đa có thể:

- corpus filter
- model parameter budget (cần adjust embedding/blocks nếu muốn cực kỳ fair)
- token presentation budget hoặc compute budget
- seed protocol

Báo tokenizer compression bên cạnh LM loss.

## Phase 2 — Data quality ablation

So `min_quality_score = 4, 5, 7`.

Cẩn thận: score 7+ giảm corpus mạnh và thiên về article dài/chất lượng theo heuristic. Phải report số article/token thực tế sau filter.

## Phase 3 — Architecture ablation

Một biến mỗi experiment:

- RMSNorm vs LayerNorm
- SwiGLU vs GELU FFN với parameter-matched hidden size
- RoPE vs learned positional embedding
- manual MHA vs SDPA chỉ là systems ablation, output nên gần nhau

Không gọi một ablation “cải tiến” nếu parameter count/compute tăng đáng kể mà không normalize.

## Phase 4 — Context

128 / 256 / 512. Attention complexity theo `T^2`, nên tăng context làm compute/memory tăng mạnh. Đừng chỉ giữ steps giống nhau rồi gọi là fair compute.

## Phase 5 — Scale

10M / 20M / 50M. Report:

- params
- unique corpus tokens
- token presentations
- tokens/param
- wall-clock
- GPU
- val/test loss

## Claim discipline

Có thể claim:

- “X giảm validation loss Y dưới cùng protocol.”
- “Tokenizer A giảm bytes/token...”
- “SDPA giảm wall-clock nhưng giữ loss tương đương...”

Không nên claim:

- “SOTA Vietnamese LLM” từ UVW subset + 20M.
- “first Vietnamese LLM” khi chưa systematic literature/repo search.
- “better tokenizer” chỉ vì PPL thấp hơn trên tokenizer-specific tokens.
