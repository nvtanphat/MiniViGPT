# Day 6 — Training Loop: AdamW, AMP, Grad Accum, LR, Checkpoint

## Mục tiêu

Đọc được `train.py` từ đầu đến cuối và biết thứ tự operation nào không được đảo bừa.

## 1. Thứ tự một optimizer update

```text
sample microbatch
→ autocast forward
→ loss / grad_accum
→ scaled backward
(repeat)
→ unscale gradients
→ clip grad norm
→ optimizer step
→ scaler update
→ zero grad
```

Clip trước unscale là sai khi dùng fp16 GradScaler.

## 2. Tokens/update

Config 20M: 12×4×256 = 12,288. Đây là unit tốt hơn “batch size 12” khi nói language-model training.

## 3. AdamW groups

Matrix params decay, RMSNorm vectors không decay. Đọc `configure_optimizer`.

## 4. LR schedule

Warmup rồi cosine. Unit test check endpoints.

## 5. Precision

Trên T4, auto sẽ chọn fp16 + GradScaler. Nếu GPU BF16 hỗ trợ thì bf16. CPU fp32.

## 6. Validation cố định

`estimate_loss` tạo generator từ fixed eval seed mỗi lần. Hãy giải thích vì sao điều này giúp compare step 400 vs 800.

## 7. Best vs latest

- best: inference/final test
- latest: resume

Không phải lúc nào latest tốt nhất.

## 8. Resume

Checkpoint chứa RNG + tokenizer. Hãy mở `save_checkpoint` và liệt kê từng state. Nếu bạn chỉ save `model.state_dict()`, đó chưa phải full training resume.

## 9. Training budget

Config 20M ~98M token presentations. Đọc caveat Chinchilla trong training methodology: đây là budget học/Kaggle, không phải compute-optimal claim.

## Checkpoint Day 6

Bạn pass khi có thể chỉ ra tại sao `optimizer.zero_grad()` đặt sai chỗ sẽ phá gradient accumulation.
