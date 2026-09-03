# Changelog

## 0.2.0

- Reworked model to ~20.306M params with 8 layers and LLaMA-style SwiGLU width.
- Added manual/SDPA attention parity path.
- Added scaled residual projection initialization.
- Added pinned UVW-2026 revision, quality filtering, streaming shuffle, exact dedup, test split.
- Added deterministic validation and held-out test reporting.
- Added fp16/bf16 auto precision and fused AdamW when available.
- Added full-state reproducible checkpoint/resume with tokenizer + RNG state.
- Added top-p sampling.
- Expanded unit tests and GitHub CI.
- Added detailed theory/methodology/reproducibility docs and 10-day learning course.
