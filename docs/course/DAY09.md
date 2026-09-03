# Day 9 — Thiết kế Ablation như một Research Engineer

## Mục tiêu

Biến repo từ “project chạy được” thành experimental platform.

Đọc `docs/EXPERIMENT_PROTOCOL.md`.

## 1. Baseline trước contribution

Không sửa 5 thứ cùng lúc. Reproduce config 20M trước.

## 2. One-variable ablation

Ví dụ tokenizer 8K vs 16K. Vấn đề: vocab thay đổi embedding parameter count. Vì vậy report parameter count và token length, hoặc adjust architecture để parameter-match.

## 3. Fairness

“Cùng số step” không đồng nghĩa cùng compute nếu context/batch/model size khác.

Report tối thiểu:

- parameters
- context
- batch tokens/update
- total token presentations
- wall time
- GPU

## 4. Validation chọn winner

Không chọn winner theo test. Test chỉ final confirmation.

## 5. Suggested experiments

A. quality threshold 4/5/7

B. byte BPE vocab 8K/16K/32K

C. RoPE vs learned positional embedding

D. RMSNorm vs LayerNorm

E. SwiGLU vs GELU parameter-matched

F. manual attention vs SDPA speed/correctness

## 6. Result table

Tạo table trước khi chạy để tránh cherry-pick metric:

| Run | Change | Params | Train tokens | Val loss | Test loss | Time |
|---|---|---:|---:|---:|---:|---:|

## Checkpoint Day 9

Bạn pass khi có thể phản biện câu: “Model B tốt hơn vì val PPL thấp hơn” bằng cách hỏi tokenizer/compute/params có công bằng không.
