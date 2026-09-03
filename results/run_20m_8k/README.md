# Run: MiniViGPT 20M, 8000 steps

Kaggle notebook `phtnguyn1ytj/minivigpt-train`, Tesla T4, bf16, ~2 giờ.

## Kết quả

| Chỉ số | Giá trị |
|---|---|
| Tham số | 20,306,304 |
| Best step | 7,999 |
| **Val loss / perplexity** | **2.9886 / 19.86** |
| **Test loss / perplexity** | **2.9881 / 19.85** |
| Token huấn luyện | 98,304,000 (4.84 token/tham số) |
| Precision | bf16 |
| Thời gian | ~120 phút (0.90 s/step) |

Test được đo đúng một lần, từ `checkpoint_best.pt` chọn theo validation.
Val và test gần như trùng nhau (2.9886 vs 2.9881) — không overfit, và cũng cho
thấy hai tập này cùng phân phối.

## Dữ liệu

`undertheseanlp/UVW-2026`, revision `a0a79294e4568137e25828bb3f2a4cde8546e1fb`,
lọc `quality_score >= 5`. Tokenizer byte-level BPE 16k, 0% `<unk>`,
~3.6 ký tự/token.

| Split | Bài | Token |
|---|---|---|
| train | 150,000 | 112,120,183 |
| validation | 8,000 | 6,088,025 |
| test | 8,000 | 6,368,136 |

Chồng lấn giữa các split: 2 bài train↔val, 1 bài train↔test, 0 bài val↔test
(≈0.02%, có sẵn trong dataset gốc vì dedup chạy riêng từng split). Không đủ
để ảnh hưởng tới số liệu.

## Chất lượng sinh văn bản

Chính tả và dấu tiếng Việt chuẩn, ngữ pháp cấp câu tốt, bắt đúng văn phong
Wikipedia. Mất mạch sau ~20-30 token và bịa tên riêng/sự kiện — đúng giới hạn
của ngân sách 4.84 token/tham số (Chinchilla khuyến nghị ~20).

Ví dụ (`temperature=0.8, top_k=50, top_p=0.95`):

> **Việt Nam là một quốc gia** đa sắc tộc. Dân tộc này là một quốc gia đa sắc
> tộc, đa dạng, là một trong những quốc gia có truyền thống văn hóa, văn hóa,
> truyền thống, văn hóa...

## File

- `summary.json` — số liệu cuối
- `metrics.jsonl` — toàn bộ log train/eval theo step
- `data_manifest.json` — provenance dataset + sha256 từng split

Checkpoint và tokenizer nằm ở `artifacts/kaggle_run/` (gitignored vì ~469 MB).

## Muốn tốt hơn

Nút thắt là compute, không phải code. Tăng `max_steps` lên ~33,000 sẽ đạt
20 token/tham số (Chinchilla-optimal), tốn ~7.5 giờ tức 2 session Kaggle —
resume đã hoạt động sẵn. Kỳ vọng perplexity xuống ~14-16.

## Bước tiếp theo

Đường cong val loss **chưa bão hòa** khi hết 8.000 bước (1.200 bước cuối vẫn
giảm 0.0285), nên compute thêm sẽ có hiệu quả thật. Khớp scaling law
`loss = 23.55 * step^-0.345 + 1.893` trên chính run này cho:

| Steps | tok/param | Dự đoán ppl |
|---|---|---|
| 8.000 (đã chạy) | 4.8 | 19.9 |
| 16.000 | 9.7 | ~15.3 |
| **33.000** | **20.0** | **~12.7** |
| 66.000 | 39.9 | ~11.1 |

`configs/minivigpt_20m_chinchilla.yaml` đặt 33.000 bước (Chinchilla-optimal),
ngữ cảnh 512 thay vì 256 để chữa lỗi mất mạch, và `train_articles` 400k để giữ
run ở ~1.36 epoch thay vì lặp corpus 3,6 lần. Ước tính ~7,5 giờ GPU trên T4,
tức hai session Kaggle — resume đã hoạt động.

Giới hạn không vượt được: ở 20M tham số, model sẽ **không bao giờ nhớ đúng
sự thật**. Perplexity thấp hơn cho văn bản trôi chảy và mạch lạc hơn, không cho
kiến thức chính xác. Muốn có điều đó cần model lớn hơn hẳn.
