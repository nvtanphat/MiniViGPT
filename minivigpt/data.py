from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from tokenizers import Tokenizer


def _dedup_key(text: str) -> bytes:
    normalized = unicodedata.normalize("NFC", text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).digest()


def iter_hf_texts(
    dataset_name: str,
    split: str,
    text_column: str,
    max_articles: int | None,
    min_chars: int = 100,
    *,
    revision: str | None = None,
    min_quality_score: int | None = None,
    quality_column: str = "quality_score",
    category_column: str = "main_category",
    exclude_categories_contains: Sequence[str] | None = None,
    shuffle_buffer: int = 0,
    seed: int = 42,
    deduplicate_exact: bool = True,
) -> Iterator[str]:
    """Stream a reproducible, filtered subset of a Hugging Face text dataset."""
    kwargs = {"split": split, "streaming": True}
    if revision:
        kwargs["revision"] = revision
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, **kwargs)
    if shuffle_buffer and shuffle_buffer > 1:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)

    excludes = tuple(x.lower() for x in (exclude_categories_contains or ()))
    seen: set[bytes] = set()
    emitted = 0

    for row in dataset:
        text = row.get(text_column)
        if not isinstance(text, str):
            continue
        text = unicodedata.normalize("NFC", text.strip())
        if len(text) < min_chars:
            continue

        if min_quality_score is not None:
            score = row.get(quality_column)
            if not isinstance(score, (int, float)) or score < min_quality_score:
                continue

        if excludes:
            category = row.get(category_column)
            category_text = category.lower() if isinstance(category, str) else ""
            if any(fragment in category_text for fragment in excludes):
                continue

        if deduplicate_exact:
            key = _dedup_key(text)
            if key in seen:
                continue
            seen.add(key)

        yield text
        emitted += 1
        if max_articles is not None and emitted >= max_articles:
            break


def write_token_binary(
    tokenizer: "Tokenizer",
    texts: Iterator[str],
    output_path: str | Path,
    append_eos: bool = True,
    eos_token: str = "<eos>",
) -> dict[str, int | float | str]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_size = tokenizer.get_vocab_size()
    dtype = np.uint16 if vocab_size <= np.iinfo(np.uint16).max else np.uint32
    dtype_max = int(np.iinfo(dtype).max)
    eos_id = tokenizer.token_to_id(eos_token)
    if append_eos and eos_id is None:
        raise ValueError(f"Missing EOS token {eos_token}")

    articles = 0
    tokens = 0
    utf8_bytes = 0
    sha256 = hashlib.sha256()
    with output_path.open("wb") as f:
        for text in tqdm(texts, desc=f"Tokenizing {output_path.name}"):
            ids = tokenizer.encode(text).ids
            if append_eos:
                ids.append(int(eos_id))
            if not ids:
                continue
            if max(ids) > dtype_max:
                raise ValueError(
                    f"Token id {max(ids)} exceeds {np.dtype(dtype).name} range; "
                    "the tokenizer vocabulary is larger than reported."
                )
            array = np.asarray(ids, dtype=dtype)
            raw = array.tobytes()
            f.write(raw)
            sha256.update(raw)
            tokens += len(ids)
            utf8_bytes += len(text.encode("utf-8"))
            articles += 1

    meta: dict[str, int | float | str] = {
        "articles": articles,
        "tokens": tokens,
        "utf8_bytes": utf8_bytes,
        "bytes_per_token": (utf8_bytes / tokens) if tokens else 0.0,
        "vocab_size": vocab_size,
        "dtype": np.dtype(dtype).name,
        "sha256": sha256.hexdigest(),
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


class BinaryTokenDataset:
    def __init__(self, path: str | Path, seq_len: int) -> None:
        self.path = Path(path)
        meta_path = self.path.with_suffix(self.path.suffix + ".json")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.dtype = np.dtype(self.meta["dtype"])
        self.tokens = np.memmap(self.path, dtype=self.dtype, mode="r")
        self.seq_len = seq_len
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        # get_batch draws a start in [0, len - seq_len) and reads tokens[i+1 : i+1+seq_len],
        # so the stream needs at least seq_len + 1 tokens.
        if len(self.tokens) < seq_len + 1:
            raise ValueError(f"Not enough tokens in {path} for seq_len={seq_len}")

    def get_batch(
        self,
        batch_size: int,
        device: torch.device,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.tokens) - self.seq_len
        starts = torch.randint(
            0, max_start, (batch_size,), generator=generator, device="cpu"
        ).tolist()
        x_np = np.stack(
            [np.asarray(self.tokens[i : i + self.seq_len], dtype=np.int64) for i in starts]
        )
        y_np = np.stack(
            [np.asarray(self.tokens[i + 1 : i + 1 + self.seq_len], dtype=np.int64) for i in starts]
        )
        x = torch.from_numpy(x_np).to(device=device, non_blocking=True)
        y = torch.from_numpy(y_np).to(device=device, non_blocking=True)
        return x, y
