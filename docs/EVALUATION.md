# Evaluation Protocol

## Metric chính

### Cross-entropy loss

Đây là negative log-likelihood trung bình trên target tokens. Thấp hơn tốt hơn trên cùng tokenizer/data protocol.

### Perplexity

`PPL = exp(loss)`.

Không dùng PPL để so trực tiếp giữa tokenizer khác vocabulary/token granularity mà không có metric normalization bổ sung.

## Validation protocol

Validation dùng fixed random windows được xác định bởi `eval_seed`. Mục tiêu: cùng một evaluation sample ở mọi checkpoint.

Best checkpoint = val loss thấp nhất.

## Test protocol

Test không tham gia:

- chọn LR
- chọn vocab size
- chọn quality threshold
- chọn layer count
- chọn checkpoint

Test được gọi đúng một lần ở cuối training trên best validation checkpoint.

## Generation sample không phải metric

Một đoạn sample “nghe hay” có thể do prompt hoặc sampling seed. Đừng dùng một sample để claim model tốt.

Khi báo cáo generation, giữ:

- checkpoint
- prompt set cố định
- temperature
- top-k
- top-p
- random seed nếu cần reproducibility

## Tokenizer metrics

Tokenizer ablation nên log:

- vocab size
- chars/token
- bytes/token
- unknown rate
- encoded sequence length trên held-out text

Byte BPE baseline kỳ vọng unknown rate gần 0.

## Báo cáo tối thiểu

| Field | Ví dụ |
|---|---|
| Dataset revision | commit hash |
| Train filter | quality >= 5 |
| Vocab | 16K byte BPE |
| Params | script-derived |
| Context | 256 |
| Effective tokens/update | 12,288 |
| Planned token presentations | ~98M |
| Best step | từ summary.json |
| Best val loss/PPL | từ summary.json |
| Test loss/PPL | từ summary.json |
| Precision | fp16 trên T4 |

## Không được cherry-pick test

Nếu bạn chạy 10 config, nhìn test cả 10 rồi chọn cái test tốt nhất để báo cáo, test đã biến thành validation. Với student research nhỏ, hãy quyết định winner từ validation rồi evaluate test một lần cho winner.
