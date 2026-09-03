# Day 5 — UVW-2026, Data Quality và Vietnamese Byte-BPE

## Mục tiêu

Hiểu rằng data/tokenizer là một phần của model experiment, không phải bước phụ.

## 1. Đọc dataset card trước code

UVW-2026 cung cấp split riêng và metadata quality/category. Repo pin revision để tránh dataset drift.

Đọc `docs/DATASET_TOKENIZER.md` trước.

## 2. Sampling pipeline

Repo không lấy first-N. Nó shuffle streaming buffer với seed, filter quality/category, exact-dedup rồi mới count accepted articles.

Hãy trả lời: nếu quality threshold tăng từ 5 lên 7 thì điều gì thay đổi ngoài “data sạch hơn”? Corpus nhỏ hơn, distribution lệch về article dài/đạt heuristic cao hơn, token diversity có thể thay đổi.

## 3. Train tokenizer chỉ trên train split

Tokenizer vocabulary được học từ train split. Validation/test không tham gia BPE merge learning.

## 4. Byte-level BPE

Học ba khái niệm:

- alphabet ban đầu là byte coverage
- BPE merge frequent pair thành subword units
- vocab size quyết định tradeoff sequence length vs embedding matrix size

16K không phải “magic number”. Nó là baseline để ablate.

## 5. Unicode NFC

Tiếng Việt có thể có Unicode code-point sequence khác nhau nhưng render giống nhau. NFC đưa text về canonical composed form trước BPE, tránh vocabulary bị phân mảnh vô lý.

## 6. EOS

Mỗi article append `<eos>`. BOS không dùng mặc định vì packed causal pretraining không cần một BOS riêng cho mọi random window.

## 7. Data artifacts

Sau prepare:

```text
train.bin
train.bin.json
validation.bin
validation.bin.json
test.bin
test.bin.json
tokenizer.json
data_manifest.json
```

Đọc sidecar để xem token count thật, không suy từ số article.

## 8. Tokenizer quality

Nếu có held-out text file một document/line:

```powershell
python -m minivigpt.tokenizer_eval `
  --tokenizer artifacts/.../tokenizer.json `
  --text-file heldout.txt
```

Quan sát chars/token, bytes/token, unk_rate.

## Checkpoint Day 5

Bạn pass khi có thể giải thích tại sao tokenizer vocabulary thay đổi parameter count và sequence length cùng lúc.
