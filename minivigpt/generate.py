from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import MiniViGPTConfig
from .model import MiniViGPT
from .tokenizer import load_tokenizer, special_token_id
from .train import load_checkpoint_file


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[MiniViGPT, dict]:
    checkpoint = load_checkpoint_file(checkpoint_path, map_location=device)
    raw_config = checkpoint["config"]
    config = MiniViGPTConfig.from_dict(raw_config["model"])
    model = MiniViGPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, raw_config


def generate_text(
    checkpoint: str | Path,
    tokenizer_path: str | Path,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.9,
    top_k: int | None = 50,
    top_p: float | None = 0.95,
) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(checkpoint, device)
    tokenizer = load_tokenizer(tokenizer_path)
    ids = tokenizer.encode(prompt).ids
    if not ids:
        # EOS is observed during pretraining at article boundaries, unlike an unused BOS token.
        ids = [special_token_id(tokenizer, "<eos>")]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    eos_id = special_token_id(tokenizer, "<eos>")
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=eos_id,
    )
    return tokenizer.decode(output[0].tolist(), skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()
    print(
        generate_text(
            args.checkpoint,
            args.tokenizer,
            args.prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
        )
    )


if __name__ == "__main__":
    main()
