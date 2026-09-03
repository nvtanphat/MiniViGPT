# Day 1 — Text → Token IDs → Embedding → Next-token Prediction

## Mục tiêu

Kết thúc Day 1 bạn phải giải thích được language model đang tối ưu cái gì, không được dừng ở “GPT dự đoán chữ tiếp theo”.

## 1. Từ text đến token

Neural network không nhận chuỗi Unicode trực tiếp. Tokenizer ánh xạ text thành sequence integer IDs:

```text
"Trí tuệ nhân tạo"
        ↓ tokenizer
[153, 827, 991, ...]
```

ID chỉ là index trong vocabulary. ID 827 không có nghĩa semantic lớn hơn ID 153.

Trong repo, tokenizer production nằm ở `src/minivigpt/tokenizer.py`. Nhưng Day 1 nên tự làm toy word-level tokenizer 10–20 dòng để hiểu `stoi`, `itos`, `encode`, `decode` trước khi đọc BPE.

## 2. Embedding

`nn.Embedding(V, C)` về bản chất là matrix `[V,C]`. Lookup token IDs `[B,T]` tạo activation `[B,T,C]`.

Hãy thuộc ký hiệu:

- B = batch
- T = sequence length
- C = model dimension
- V = vocabulary size

Ví dụ config 20M:

```text
input_ids [12, 256]
embedding [12, 256, 384]
```

## 3. Next-token target

Nếu token stream là:

```text
[10, 20, 30, 40, 50]
```

với T=4:

```text
x = [10, 20, 30, 40]
y = [20, 30, 40, 50]
```

Mỗi position dự đoán token kế tiếp. Repo làm đúng việc này trong `BinaryTokenDataset.get_batch()`.

## 4. Logits

Sau Transformer, hidden state `[B,T,C]` đi qua LM head `[C,V]` thành logits `[B,T,V]`.

Chiều cuối phải là V vì tại mỗi vị trí model cần một score cho **mọi token có thể xuất hiện tiếp theo**.

## 5. Cross-entropy

Training dùng:

```python
F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
```

Không softmax trước. CrossEntropy nhận raw logits để tính log-softmax ổn định hơn.

## 6. Đọc code repo

Mở `src/minivigpt/model.py`, chỉ đọc các phần:

- `MiniViGPT.__init__`: token embedding + lm_head
- `MiniViGPT.forward`: input validation, embedding, logits, loss

Tạm bỏ qua attention/block.

Mở `src/minivigpt/data.py`, đọc dòng tạo x/y. Hãy tự chứng minh `x[:,1:] == y[:,:-1]` — repo có unit test cho invariant này.

## 7. Lab

Chạy:

```powershell
$env:PYTHONPATH="$PWD/src"
pytest -q tests/test_data_training.py
pytest -q tests/test_model.py::test_forward_shape_and_loss
```

Nếu chưa cài dependencies:

```powershell
py -m pip install -r requirements.txt
```

## 8. Bài tập bắt buộc

Cho B=8, T=128, C=512, V=16000:

1. `input_ids.shape`?
2. embedding shape?
3. logits shape?
4. target shape?
5. khi flatten để CE thì logits/target có shape gì?

Đáp án bạn phải tự ra: logits flatten vẫn giữ V ở chiều cuối.

## 9. Câu hỏi phỏng vấn

**Tại sao language model không cần label thủ công?** Vì target được tạo trực tiếp bằng cách shift chính text corpus một token; đây là self-supervised learning.

**Token ID có phải feature numeric không?** Không. ID là categorical index; embedding lookup biến nó thành learnable continuous representation.

## Checkpoint Day 1

Bạn pass Day 1 khi có thể nhìn `[B,T,V]` và giải thích từng chiều mà không đoán.
