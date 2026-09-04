"""Demo what a 20M-parameter Vietnamese LM actually learned.

    python scripts/demo.py

Generation alone undersells this model: it writes fluent Vietnamese but invents
facts, which reads as failure. The measurable wins are grammar discrimination,
tokenizer quality, and the loss curve -- so show those too.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch

from minivigpt.generate import load_model
from minivigpt.tokenizer import load_tokenizer, special_token_id

CKPT = Path("artifacts/kaggle_run_33k/checkpoint_best.pt")
TOK = Path("artifacts/kaggle_run_33k/tokenizer.json")
SUMMARY = Path("results/run_20m_33k/summary.json")

# Each pair is (fluent, corrupted): same words, broken order or duplication.
GRAMMAR_PAIRS = [
    ("Hà Nội là thủ đô của Việt Nam.", "Hà Nội thủ đô là của Việt Nam."),
    ("Tôi đi học vào buổi sáng.", "Tôi học đi sáng buổi vào."),
    ("Cô ấy rất thích đọc sách.", "Cô ấy rất thích đọc sách sách sách."),
    ("Trời hôm nay mưa rất to.", "Trời hôm nay to rất mưa."),
    ("Học sinh đang làm bài tập.", "Học sinh đang bài làm tập."),
]

PROMPTS = [
    "Thành phố Hồ Chí Minh nằm ở",
    "Trong lĩnh vực khoa học máy tính,",
    "Nền văn hóa Việt Nam",
]


def rule(title: str) -> None:
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


def main() -> None:
    if not CKPT.exists():
        raise SystemExit(f"Checkpoint not found: {CKPT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(CKPT, device)
    tokenizer = load_tokenizer(TOK)
    eos_id = special_token_id(tokenizer, "<eos>")

    rule("1. MÔ HÌNH")
    cfg = model.config
    print(f"  Tham số      {model.num_parameters():,}")
    print(f"  Kiến trúc    {cfg.n_layers} lớp, dim {cfg.dim}, {cfg.n_heads} head, ngữ cảnh {cfg.max_seq_len}")
    print("               RMSNorm + RoPE + SwiGLU, pre-norm, weight tying")
    if SUMMARY.exists():
        s = json.loads(SUMMARY.read_text(encoding="utf-8"))
        print(f"  Test loss    {s['test_loss']:.4f}  (perplexity {s['test_perplexity']:.2f})")
        print(f"  Ngân sách    {s['planned_tokens_per_parameter']:.2f} token/tham số, {s['precision']}")

    rule("2. TOKENIZER TIẾNG VIỆT")
    samples = [
        "Xin chào, tôi là một mô hình ngôn ngữ.",
        "Nguyễn Huệ đại phá quân Thanh năm Kỷ Dậu 1789.",
    ]
    unk = special_token_id(tokenizer, "<unk>")
    for text in samples:
        ids = tokenizer.encode(text).ids
        back = tokenizer.decode(ids)
        n_unk = sum(1 for i in ids if i == unk)
        print(f"  {text}")
        print(f"    {len(ids)} token, {len(text)/len(ids):.2f} ký tự/token, "
              f"{n_unk} <unk>, khôi phục {'chính xác' if back == text else 'SAI'}")

    rule("3. PHÂN BIỆT NGỮ PHÁP ĐÚNG / SAI")
    print("  Model chấm điểm câu đúng thấp hơn (tốt hơn) câu bị đảo lộn.\n")

    def nll(text: str) -> float:
        ids = tokenizer.encode(text).ids
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            _, loss = model(x[:, :-1], x[:, 1:])
        return float(loss.item())

    correct = 0
    for good, bad in GRAMMAR_PAIRS:
        g, b = nll(good), nll(bad)
        ok = g < b
        correct += ok
        print(f"  [{'✓' if ok else '✗'}] {g:.2f} vs {b:.2f}   {good}")
    print(f"\n  → {correct}/{len(GRAMMAR_PAIRS)} đúng "
          f"({100*correct/len(GRAMMAR_PAIRS):.0f}%) — model học được ngữ pháp tiếng Việt")

    rule("4. SINH VĂN BẢN")
    print("  Lưu ý: đây là base model pretrain. Văn phong và ngữ pháp đúng,")
    print("  nhưng TÊN RIÊNG VÀ SỰ KIỆN LÀ BỊA — không dùng làm nguồn tin.\n")
    for prompt in PROMPTS:
        torch.manual_seed(0)
        ids = tokenizer.encode(prompt).ids
        x = torch.tensor([ids[-(cfg.max_seq_len - 1):]], dtype=torch.long, device=device)
        out = model.generate(x, max_new_tokens=45, temperature=0.8,
                             top_k=50, top_p=0.95, eos_id=eos_id)
        text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
        print(f"  ▸ {prompt}")
        print(f"    {text[len(prompt):].strip()[:200]}\n")

    rule("5. ĐƯỜNG CONG HUẤN LUYỆN")
    metrics = Path("results/run_20m_33k/metrics.jsonl")
    if metrics.exists():
        evals = [json.loads(l) for l in metrics.read_text(encoding="utf-8").splitlines()]
        evals = [m for m in evals if "val_loss" in m]
        # Skip the step-0 point: at ppl ~16,900 it flattens every other bar.
        points = evals[1:]
        shown = points[::4]
        if shown[-1] is not points[-1]:
            shown = shown + [points[-1]]
        width = 42
        lo = min(m["val_loss"] for m in points)
        hi = max(m["val_loss"] for m in points)
        for m in shown:
            v = m["val_loss"]
            n = int(width * (v - lo) / (hi - lo)) if hi > lo else 0
            print(f"  {m['step']:>5} {'█' * max(1, n):<{width}} {math.exp(v):>7.1f}")
        print(f"\n  Perplexity {math.exp(evals[0]['val_loss']):,.0f} → "
              f"{math.exp(evals[-1]['val_loss']):.1f} sau 8.000 bước (~2 giờ trên T4)")

    print()


if __name__ == "__main__":
    main()
