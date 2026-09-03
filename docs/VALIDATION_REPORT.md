# Validation report — v0.2

This report records what was actually checked before packaging the reviewed repository.

## Static and unit checks

- `python -m compileall -q src tests`: passed.
- `pytest -q`: **13 passed, 1 skipped** in the current offline container.
- The skipped test is the ByteLevel-BPE integration test because the optional runtime package `tokenizers` is not installed in this offline container. It remains in the test suite and should run after `pip install -e .[dev]` in a normal environment.
- `scripts/verify_repo.py`: passed for `configs/minivigpt_20m.yaml`.
- Exact main-model parameter count: **20,306,304**.
- Main config derived dimensions: `head_dim=64`, LLaMA-style SwiGLU `hidden_dim=1024`.
- Kaggle bundle generation and metadata validation: passed with a dummy owner/slug; GPU, Internet, T4 machine shape, kernel ID and dataset source were checked.

## Training-budget audit

For the default 20M config:

- effective sequences/update: 48
- tokens/update: 12,288
- planned optimizer updates: 8,000
- planned token presentations: 98,304,000
- planned token presentations / parameter: about 4.84

`token presentations` are sampled training windows and may repeat; they are **not** unique corpus tokens. This budget is intentionally an educational Kaggle budget, not a claim of compute-optimal pretraining.

## What was not executed here

This container has no Kaggle account credentials, Internet access, or Kaggle GPU runtime, so a full UVW-2026 download/tokenization/pretraining job was **not** executed here. The repo therefore requires the documented debug Kaggle run before the 20M run. This is an explicit verification gate rather than an untested claim.
