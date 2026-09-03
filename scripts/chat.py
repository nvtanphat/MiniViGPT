"""Interactive sampling from a trained MiniViGPT checkpoint.

    python scripts/chat.py
    python scripts/chat.py --prompt "Hà Nội là" --temperature 0.7

This is a base language model, not an instruction-tuned assistant: it continues
whatever text you give it rather than answering questions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch

from minivigpt.generate import load_model
from minivigpt.tokenizer import load_tokenizer, special_token_id

DEFAULT_DIR = Path("artifacts/kaggle_run")


def resolve(explicit: str | None, name: str) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise SystemExit(f"Not found: {path}")
        return path
    for base in (DEFAULT_DIR, Path("artifacts/local")):
        candidate = base / name
        if candidate.exists():
            return candidate
    raise SystemExit(f"Could not find {name}; pass it explicitly.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--tokenizer")
    parser.add_argument("--prompt", help="Generate once and exit instead of looping")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    checkpoint = resolve(args.checkpoint, "checkpoint_best.pt")
    tokenizer_path = resolve(args.tokenizer, "tokenizer.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(checkpoint, device)
    tokenizer = load_tokenizer(tokenizer_path)
    eos_id = special_token_id(tokenizer, "<eos>")

    def sample(text: str, temperature: float) -> str:
        ids = tokenizer.encode(text).ids or [eos_id]
        # Leave room for the new tokens inside the context window.
        keep = model.config.max_seq_len - 1
        input_ids = torch.tensor([ids[-keep:]], dtype=torch.long, device=device)
        out = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_id=eos_id,
        )
        return tokenizer.decode(out[0].tolist(), skip_special_tokens=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.prompt:
        print(sample(args.prompt, args.temperature))
        return

    print(f"MiniViGPT {model.num_parameters():,} params on {device}")
    print("Type a Vietnamese phrase and the model continues it.")
    print("Commands: /temp <x>  /len <n>  /quit\n")

    temperature = args.temperature
    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            break
        if line.startswith("/temp "):
            try:
                temperature = float(line.split(maxsplit=1)[1])
                print(f"temperature = {temperature}\n")
            except ValueError:
                print("usage: /temp 0.8\n")
            continue
        if line.startswith("/len "):
            try:
                args.max_new_tokens = int(line.split(maxsplit=1)[1])
                print(f"max_new_tokens = {args.max_new_tokens}\n")
            except ValueError:
                print("usage: /len 120\n")
            continue
        print(sample(line, temperature), "\n")


if __name__ == "__main__":
    main()
