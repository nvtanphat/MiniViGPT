# Training Methodology

## 1. Objective

Model nhận `input_ids [B,T]`, tạo logits `[B,T,V]`, và dùng cross-entropy với target bị shift một token.

Không gọi softmax trước `F.cross_entropy`; PyTorch xử lý log-softmax ổn định số bên trong.

## 2. Effective batch

Config 20M:

- micro batch = 12 sequences
- sequence = 256 tokens
- gradient accumulation = 4

Do đó:

`tokens/update = 12 * 256 * 4 = 12,288`.

Gradient của mỗi microbatch được chia cho 4 trước backward nên tổng gradient tương đương mean của effective batch (bỏ qua khác biệt floating-point).

## 3. AdamW

Parameter grouping:

- matrix params (`ndim >= 2`) -> weight decay 0.1
- vector params như RMSNorm weight -> no decay

Optimizer betas mặc định `(0.9, 0.95)`.

Nếu CUDA/PyTorch hỗ trợ fused AdamW, repo tự bật `fused=True`.

## 4. Learning rate

Schedule:

1. Linear warmup từ nhỏ lên peak LR.
2. Cosine decay về `min_lr`.

Config 20M dùng peak `3e-4`, min `3e-5`, warmup 200 update. Đây là conservative baseline, **không phải hyperparameter đã được chứng minh tối ưu cho UVW-2026**.

## 5. Precision trên Kaggle

`precision: auto`:

- CUDA có BF16 support -> bf16 autocast, không cần loss scaling.
- CUDA không BF16 (ví dụ T4) -> fp16 autocast + GradScaler.
- CPU -> fp32.

Điểm này quan trọng vì hard-code bf16 trên T4 là sai về hardware support.

## 6. Gradient clipping

Sau `scaler.unscale_(optimizer)`, repo clip global gradient norm về threshold 1.0 rồi mới optimizer step. Metrics log `grad_norm_preclip` để bạn phát hiện instability.

## 7. Deterministic validation

Mỗi lần validation tạo generator từ cùng `eval_seed`, nên batch windows giống nhau giữa step 400, 800, ... Điều này làm curve dễ diễn giải hơn.

Training vẫn sample random windows bình thường.

## 8. Checkpoint

`checkpoint_best.pt`: validation loss tốt nhất.

`checkpoint_latest.pt`: trạng thái gần nhất cho resume.

Checkpoint gồm model, optimizer, AMP scaler, RNG states, config và tokenizer JSON. Vì vậy resume không chỉ “load weight”.

## 9. Resume reproducibility

Nếu config model khác checkpoint, repo raise error. Không cho vô tình resume weight 20M vào architecture khác.

Khi resume:

1. restore tokenizer từ checkpoint nếu output folder chưa có tokenizer.
2. rebuild data artifacts từ pinned dataset revision + fixed seeds nếu cần.
3. load model/optimizer/scaler.
4. restore RNG state sau construction.
5. tiếp tục từ `step + 1`.

## 10. Validation vs test

Validation được gọi trong training và dùng để chọn best checkpoint.

Test chỉ chạy sau loop, sau khi load `checkpoint_best.pt`. `summary.json` chứa test loss/PPL.

## 11. Perplexity phải đọc đúng

`PPL = exp(mean token cross-entropy)`.

PPL có ý nghĩa khi so các checkpoint dùng **cùng tokenizer và cùng evaluation tokenization**. Không nên lấy PPL của tokenizer 8K và 32K rồi kết luận tokenizer nào “tốt hơn” chỉ từ số đó, vì token unit khác nhau.

## 12. Training budget và Chinchilla caveat

Chạy script budget. Config mặc định khoảng 20M params và ~98M token presentations. Đây chỉ vài token/parameter, thấp hơn compute-optimal regimes được nghiên cứu cho large models.

Mục tiêu Kaggle baseline là học full pipeline và tạo model có signal, không phải đạt compute-optimal frontier.
