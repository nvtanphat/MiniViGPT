from __future__ import annotations

import argparse
import json
from pathlib import Path

from .tokenizer import load_tokenizer, special_token_id


def evaluate_texts(tokenizer, texts, max_articles: int | None = None) -> dict:
    articles = 0
    characters = 0
    utf8_bytes = 0
    tokens = 0
    unk_tokens = 0
    unk_id = special_token_id(tokenizer, "<unk>")
    for text in texts:
        ids = tokenizer.encode(text).ids
        articles += 1
        characters += len(text)
        utf8_bytes += len(text.encode("utf-8"))
        tokens += len(ids)
        unk_tokens += sum(int(i == unk_id) for i in ids)
        if max_articles is not None and articles >= max_articles:
            break
    return {
        "articles": articles,
        "characters": characters,
        "utf8_bytes": utf8_bytes,
        "tokens": tokens,
        "chars_per_token": characters / tokens if tokens else 0.0,
        "bytes_per_token": utf8_bytes / tokens if tokens else 0.0,
        "unk_rate": unk_tokens / tokens if tokens else 0.0,
        "vocab_size": tokenizer.get_vocab_size(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--text-file", required=True, help="UTF-8 file, one document per line")
    parser.add_argument("--max-articles", type=int, default=None)
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.tokenizer)
    with Path(args.text_file).open("r", encoding="utf-8") as f:
        result = evaluate_texts(tokenizer, (line.rstrip("\n") for line in f), args.max_articles)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
