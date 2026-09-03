from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


DEFAULT_SPECIAL_TOKENS = ["<pad>", "<unk>", "<eos>"]


def train_bpe_tokenizer(
    texts: Iterable[str],
    output_path: str | Path,
    vocab_size: int = 16000,
    min_frequency: int = 2,
    special_tokens: list[str] | None = None,
) -> Tokenizer:
    """Train a reversible byte-level BPE tokenizer on NFC-normalized raw Vietnamese text."""
    special_tokens = special_tokens or DEFAULT_SPECIAL_TOKENS
    if "<unk>" not in special_tokens or "<eos>" not in special_tokens:
        raise ValueError("special_tokens must include <unk> and <eos>")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    return tokenizer


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def special_token_id(tokenizer: Tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise KeyError(f"Tokenizer does not contain special token {token!r}")
    return token_id
