# Run: MiniViGPT 20M, 33.000 bước (Chinchilla-optimal)

Kaggle notebook `phtnguyn1ytj/minivigpt-train`, Tesla T4, bf16, ~8,8 giờ.
Chạy trọn 33.000 bước trong một session, không bị cắt.

## Kết quả

| Chỉ số | run 8k | **run 33k** | Thay đổi |
|---|---|---|---|
| Steps | 8.000 | **33.000** | 4,1× |
| Ngữ cảnh | 256 | **512** | 2× |
| token/tham số | 4,84 | **19,97** | Chinchilla-optimal |
| Val loss / ppl | 2,9886 / 19,86 | **2,5609 / 12,95** | **−35%** |
| Test loss / ppl | 2,9881 / 19,85 | **2,6374 / 13,98** | **−30%** |
| GPU-giờ (T4) | 2,0 | 8,8 | 4,4× |

Best step 32.999. Test đo một lần duy nhất từ checkpoint chọn theo validation.

Khoảng cách val↔test rộng hơn run trước (12,95 vs 13,98, chênh 1,03) so với
run 8k (19,86 vs 19,85, gần như trùng). Đây là hệ quả bình thường của việc
train lâu hơn: model bám sát phân phối validation hơn. Test vẫn là con số
đáng tin cậy để báo cáo.

## Không ghi nhớ dù chạy 3,6 epoch

Corpus 112M token với 405M token huấn luyện = 3,6 epoch. Theo dõi khoảng cách
train↔val cho thấy không có ghi nhớ:

| Step | Train loss | Val loss | Gap |
|---|---|---|---|
| 6.000 | 3,097 | 2,998 | −0,099 |
| 12.000 | 2,827 | 2,793 | −0,035 |
| 18.000 | 2,662 | 2,693 | +0,031 |
| 24.000 | 2,547 | 2,614 | +0,067 |
| 32.999 | 2,573 | 2,561 | −0,012 |

Gap dao động quanh 0 (±0,07), không có xu hướng val tách lên trên. Ở 20M tham
số, model quá nhỏ để ghi nhớ 112M token.

Val loss **vẫn còn giảm** khi kết thúc (3.000 bước cuối giảm 0,0076), nhưng
tốc độ đã rất chậm — chạy tiếp lên 66.000 bước chỉ ước được ppl ~11,1, không
đáng chi phí.

## Độ chính xác của dự báo

Scaling law `loss = 23,55 × step^-0,345 + 1,893` khớp từ run 8k dự đoán
ppl 12,7 tại step 33.000. Thực tế: **12,95**. Sai số 2%.

## Chất lượng sinh văn bản

Ngữ pháp: **5/5** trên bài kiểm tra phân biệt câu đúng/câu đảo trật tự.

Cải thiện rõ ở văn bản hành chính — model giờ sinh được số liệu có cấu trúc
nhất quán:

> **Thành phố Hồ Chí Minh nằm ở** phía nam của tỉnh, có diện tích 15,96 km²,
> dân số là 1.538 người, mật độ dân số đạt 1.087 người/km².
>
> Lịch sử
> Ngày 9 tháng 1 năm 2003, Chính phủ ban hành Nghị định số 11/2003/NĐ-CP về
> việc thành lập

Định dạng nghị định, đơn vị đo, mật độ dân số đều đúng quy cách. Nhưng nội
dung vẫn sai (TP.HCM không phải "phía nam của tỉnh", dân số không phải 1.538).

Với prompt trừu tượng thì vẫn lặp:

> **Trong lĩnh vực khoa học máy tính,** máy tính được chia thành hai loại:
> máy tính, máy tính, và máy tính.

Đây là giới hạn không vượt được ở 20M tham số: model học được *hình thức* rất
tốt, nhưng không có đủ dung lượng cho *nội dung*. Perplexity thấp hơn cho văn
bản trôi chảy hơn, không cho kiến thức đúng.

## File

- `summary.json`, `metrics.jsonl`, `data_manifest.json`
- Checkpoint + tokenizer: `artifacts/kaggle_run_33k/` (gitignored, ~469 MB)
