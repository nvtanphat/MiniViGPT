# Day 10 — Reproduce, Audit, Publish

## Mục tiêu

Người khác clone repo phải biết bạn đã làm gì, tại sao và giới hạn ở đâu.

## 1. Full verification

```powershell
$env:PYTHONPATH="$PWD/src"
python scripts/verify_repo.py --config configs/minivigpt_20m.yaml
python scripts/estimate_training_budget.py --config configs/minivigpt_20m.yaml
```

Sau Kaggle run, lưu `summary.json`, `metrics.jsonl`, tokenizer và config.

## 2. README không overclaim

Nên mô tả:

> Educational modern decoder-only LM trained from scratch on Vietnamese Wikipedia, designed for reproducible Kaggle T4 experiments.

Không viết “first Vietnamese LLM”, “SOTA”, “production-ready” nếu chưa có evidence.

## 3. Method section

Method phải có:

- data revision/filter
- tokenizer
- architecture
- objective
- training recipe
- evaluation protocol
- hardware

`docs/THEORY_PROVENANCE.md` là nguồn để viết phần architecture provenance.

## 4. Result section

Report loss/PPL từ machine-readable `summary.json`, không chép tay từ log nếu có thể.

Thêm training curve từ `metrics.jsonl` khi publish.

## 5. Limitations

Bắt buộc ghi:

- Wikipedia-only domain
- ~20M model
- short context
- limited training-token budget
- exact dedup only
- no instruction tuning
- generation không được xem là factual QA system

## 6. License/data attribution

Code MIT. UVW-2026 data CC BY-SA 4.0. Khi redistributing data-derived artifacts/weights, đọc và tuân thủ điều khoản nguồn thay vì giả định code license áp dụng cho dataset.

## 7. Sau Day 10

Khi baseline này ổn mới sang Phase 2:

- SFT/LoRA trên một pretrained model phù hợp
- RAG/retrieval
- KV cache/serving
- FastAPI/UI/Docker

Đó là **LLM Engineering layer**, khác với mục tiêu “pretrain transformer from scratch” của repo này.

## Graduation checklist

Bạn hoàn thành khóa khi tự trả lời được:

1. Vì sao causal mask cần thiết?
2. RoPE nằm ở đâu?
3. Tại sao SwiGLU width ~8/3 C?
4. Tại sao checkpoint cần tokenizer?
5. Vì sao PPL khác tokenizer khó so trực tiếp?
6. Vì sao test không dùng chọn model?
7. Một Kaggle resume đúng cần những state nào?
8. Tại sao 20M/~98M token presentations chưa phải compute-optimal foundation model?
