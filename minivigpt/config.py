from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def llama_swiglu_hidden_dim(dim: int, multiple_of: int = 256) -> int:
    """LLaMA-style SwiGLU width: round_up((8/3) * dim, multiple_of).

    Meta's LLaMA reference starts from 4*dim, multiplies by 2/3 to compensate
    for the third SwiGLU projection, then rounds to a hardware-friendly multiple.
    """
    if dim <= 0:
        raise ValueError("dim must be positive")
    if multiple_of <= 0:
        raise ValueError("multiple_of must be positive")
    hidden = int((8 * dim) / 3)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


@dataclass
class MiniViGPTConfig:
    vocab_size: int = 16000
    max_seq_len: int = 256
    dim: int = 384
    n_layers: int = 8
    n_heads: int = 6
    hidden_dim: int | None = None
    ffn_multiple_of: int = 256
    dropout: float = 0.0
    rope_theta: float = 10000.0
    rms_eps: float = 1e-6
    tie_embeddings: bool = True
    init_std: float = 0.02
    scaled_residual_init: bool = True
    attention_impl: str = "manual"  # manual | sdpa

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.max_seq_len <= 0:
            raise ValueError("vocab_size and max_seq_len must be positive")
        if self.dim <= 0 or self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("dim, n_layers, and n_heads must be positive")
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        head_dim = self.dim // self.n_heads
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if self.hidden_dim is None:
            self.hidden_dim = llama_swiglu_hidden_dim(self.dim, self.ffn_multiple_of)
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.rope_theta <= 0 or self.rms_eps <= 0 or self.init_std <= 0:
            raise ValueError("rope_theta, rms_eps, and init_std must be positive")
        if self.attention_impl not in {"manual", "sdpa"}:
            raise ValueError("attention_impl must be 'manual' or 'sdpa'")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MiniViGPTConfig":
        return cls(**value)
