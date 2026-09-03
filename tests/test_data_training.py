import json

import numpy as np
import pytest
import torch

from minivigpt.data import BinaryTokenDataset
from minivigpt.train import learning_rate, safe_perplexity


def make_binary(tmp_path):
    path = tmp_path / "tokens.bin"
    tokens = np.arange(1000, dtype=np.uint16) % 127
    tokens.tofile(path)
    meta = {
        "dtype": "uint16",
        "tokens": len(tokens),
        "articles": 1,
        "utf8_bytes": 1000,
        "bytes_per_token": 1.0,
        "vocab_size": 128,
        "sha256": "test",
    }
    (tmp_path / "tokens.bin.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


def test_binary_batch_is_shifted_next_token(tmp_path):
    data = BinaryTokenDataset(make_binary(tmp_path), seq_len=16)
    gen = torch.Generator().manual_seed(7)
    x, y = data.get_batch(4, torch.device("cpu"), generator=gen)
    assert x.shape == y.shape == (4, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_binary_eval_sampling_is_reproducible(tmp_path):
    data = BinaryTokenDataset(make_binary(tmp_path), seq_len=16)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    x1, y1 = data.get_batch(5, torch.device("cpu"), generator=g1)
    x2, y2 = data.get_batch(5, torch.device("cpu"), generator=g2)
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_lr_warmup_and_cosine_endpoints():
    cfg = {
        "learning_rate": 3e-4,
        "min_lr": 3e-5,
        "warmup_steps": 10,
        "max_steps": 100,
    }
    assert abs(learning_rate(0, cfg) - 3e-5) < 1e-15
    assert abs(learning_rate(9, cfg) - 3e-4) < 1e-12
    assert abs(learning_rate(100, cfg) - 3e-5) < 1e-12


def test_dataset_rejects_stream_shorter_than_context(tmp_path):
    """A stream of exactly seq_len tokens cannot yield a shifted target."""
    path = tmp_path / "tiny.bin"
    np.arange(16, dtype=np.uint16).tofile(path)
    (tmp_path / "tiny.bin.json").write_text(json.dumps({"dtype": "uint16"}), encoding="utf-8")
    with pytest.raises(ValueError):
        BinaryTokenDataset(path, seq_len=16)
    # seq_len + 1 tokens is the smallest usable stream.
    BinaryTokenDataset(path, seq_len=15)


def test_safe_perplexity_does_not_overflow_on_diverged_loss():
    assert safe_perplexity(0.0) == 1.0
    assert safe_perplexity(900.0) == float("inf")
