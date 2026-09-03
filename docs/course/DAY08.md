# Day 8 — Generation, Sampling và Evaluation

## Mục tiêu

Không đánh giá LM bằng cảm giác từ một đoạn text.

## 1. Greedy

`temperature <= 0` -> argmax. Deterministic nhưng dễ repetitive.

## 2. Temperature

Chia logits cho temperature trước softmax:

- T < 1: distribution sắc hơn
- T > 1: phẳng hơn

Temperature không làm model “thông minh hơn”; chỉ đổi decoding distribution.

## 3. Top-k

Chỉ giữ k logits lớn nhất. Fixed candidate count.

## 4. Top-p

Nucleus sampling giữ tập token nhỏ nhất có cumulative probability >= p. Candidate count tự thích ứng theo độ peaked của distribution.

Repo có thể combine top-k rồi top-p. Sampling config phải được report cùng sample.

## 5. Perplexity

PPL = exp(loss). Đọc `docs/EVALUATION.md` để hiểu caveat tokenizer.

## 6. Best checkpoint

Generate từ `checkpoint_best.pt`, không mặc định latest.

## 7. Prompt set

Tạo 10 prompt tiếng Việt cố định thuộc nhiều dạng:

- factual opening
- narrative
- definition
- history
- science

Không chỉ dùng “Trí tuệ nhân tạo”. Lưu output theo checkpoint để qualitative comparison có cấu trúc.

## 8. Failure analysis

Quan sát:

- repetition
- broken diacritics (tokenizer bug nếu decode lỗi)
- topic drift
- premature EOS
- memorization-like fragments

## Checkpoint Day 8

Bạn pass khi biết phân biệt **model quality** với **sampling behavior**.
