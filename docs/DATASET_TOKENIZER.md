# Dataset & Tokenizer Methodology

## Dataset: UVW-2026

Repo dùng `undertheseanlp/UVW-2026`, một Vietnamese Wikipedia dataset đã tách train/validation/test và có các trường `content`, `quality_score`, `main_category`, `wikidata_id`.

Config pin một commit revision thay vì `main`. Mục tiêu là một run tháng sau không âm thầm đọc data khác chỉ vì upstream dataset cập nhật.

## Không lấy “first N rows” một cách mù quáng

Iterable dataset được shuffle bằng buffer và seed trước khi lấy subset. Sau đó repo áp dụng:

1. Unicode NFC normalization.
2. `min_chars`.
3. `min_quality_score` nếu bật.
4. loại category nội bộ/định hướng theo config.
5. exact dedup bằng BLAKE2 hash trên normalized whitespace text.
6. dừng khi đủ số article được chấp nhận.

Điều này tốt hơn `for row in dataset: take first 60000`, vì first-N có thể phụ thuộc ordering upstream.

## Quality score không phải “chân lý”

UVW quality score dựa trên length, sentence count/density, markup cleanliness và metadata bonus. Score cao không đồng nghĩa văn phong luôn tốt hơn cho language modeling. Vì vậy `min_quality_score=5` chỉ là một baseline heuristic và phải được ablate nếu làm research.

Không dùng test split để tune threshold.

## Exact dedup khác near-duplicate dedup

Repo chỉ loại **exact duplicate sau normalization whitespace** trong subset đang stream. Nó không làm MinHash/LSH semantic near-dedup. Nếu mở rộng corpus sang web/news rất lớn, exact hash là chưa đủ.

## Byte-level BPE

Pipeline tokenizer:

```text
raw Vietnamese Unicode
  ↓ NFC
UTF-8/byte-level pretokenization
  ↓ BPE merge learning
16K vocabulary
  ↓
token IDs
```

Byte-level alphabet giúp mọi byte sequence có representation, vì vậy `<unk>` về thực tế phải gần zero trên text UTF-8 hợp lệ. Repo có unit test round-trip tiếng Việt và tokenizer evaluation.

## Vì sao không dùng word segmentation trước?

MiniViGPT là project học LM from scratch. Nếu thêm VnCoreNLP/underthesea word segmentation trước BPE, bạn đưa một model/rule preprocessing khác vào pipeline và làm tokenization phụ thuộc segmentation quality. Byte-level BPE giúp baseline đơn giản, language-agnostic và reproducible.

Đó không có nghĩa byte-level BPE luôn tốt nhất cho tiếng Việt. Một experiment tốt có thể so:

- Byte BPE 8K
- Byte BPE 16K
- Byte BPE 32K
- SentencePiece BPE/Unigram

nhưng phải giữ training-token/compute budget công bằng và không so perplexity trực tiếp giữa tokenizer khác nhau mà không giải thích denominator.

## EOS packing

Mỗi article được encode rồi append `<eos>`. Các article được nối thành một binary token stream. Random training window có thể cắt qua boundary:

```text
... cuối bài A <eos> đầu bài B ...
```

Đây là intentional document packing. Model nhìn thấy EOS nên học boundary.

Repo không padding training sequence, vì mọi batch lấy fixed-length packed windows.

## Binary format

Nếu vocab <= 65535, token IDs được lưu `uint16`; lớn hơn dùng `uint32`. Sidecar JSON lưu:

- articles
- tokens
- UTF-8 bytes
- bytes/token
- dtype
- SHA-256 của binary stream

Các field này giúp audit data artifact và tokenizer compression.

## Train / Validation / Test

- tokenizer train: chỉ từ **train split**.
- model train: chỉ `train.bin`.
- checkpoint selection: validation.
- final report: test chỉ sau training.

Tokenizer học vocabulary từ train split được xem là preprocessing của training data, không phải validation/test leakage.
