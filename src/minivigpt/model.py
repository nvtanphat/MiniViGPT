from __future__ import annotations

import math
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MiniViGPTConfig


class RMSNorm(nn.Module):
    """RMSNorm as used by LLaMA-style decoder-only Transformers."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(variance + self.eps)
        # Match LLaMA: normalize in fp32, apply gain, then cast back once.
        return (self.weight.float() * x_norm).to(dtype=input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate consecutive feature pairs: (x0, x1) -> (-x1, x0)."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    """RoPE cache for one attention head."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        angles = torch.repeat_interleave(freqs, 2, dim=-1)
        self.register_buffer("cos_cached", angles.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", angles.sin()[None, None, :, :], persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)
        if seq_len > self.cos_cached.size(-2):
            raise ValueError("RoPE cache is shorter than the requested sequence")
        cos = self.cos_cached[:, :, :seq_len].to(dtype=q.dtype, device=q.device)
        sin = self.sin_cached[:, :, :seq_len].to(dtype=q.dtype, device=q.device)
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with a manual reference path.

    `manual` is intentionally kept readable for learning. `sdpa` uses PyTorch's
    fused scaled_dot_product_attention while preserving the same Q/K/V/RoPE logic.
    """

    def __init__(self, config: MiniViGPTConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.dropout = config.dropout
        self.attention_impl = config.attention_impl

        self.q_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.rope = RotaryEmbedding(
            self.head_dim, config.max_seq_len, theta=config.rope_theta
        )

        # Kept even when SDPA is selected so the causal structure is inspectable/testable.
        mask = torch.tril(
            torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", mask[None, None, :, :], persistent=False)

    def _project_qkv(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)
        return q, k, v

    def _manual_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        seq_len = q.size(-2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(~mask, float("-inf"))
        # Compute softmax in fp32 for numerical stability, then cast back.
        weights = F.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        if self.dropout > 0.0:
            weights = F.dropout(weights, p=self.dropout, training=self.training)
        return torch.matmul(weights, v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        if seq_len > self.causal_mask.size(-1):
            raise ValueError("Sequence length exceeds configured max_seq_len")

        q, k, v = self._project_qkv(x)
        if self.attention_impl == "sdpa":
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            out = self._manual_attention(q, k, v)

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        return self.out_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, config: MiniViGPTConfig) -> None:
        super().__init__()
        assert config.hidden_dim is not None
        self.gate_proj = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return F.dropout(x, p=self.dropout, training=self.training) if self.dropout > 0 else x


class TransformerBlock(nn.Module):
    """Pre-norm decoder block: RMSNorm -> Attention -> residual -> RMSNorm -> SwiGLU -> residual."""

    def __init__(self, config: MiniViGPTConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, config.rms_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.dim, config.rms_eps)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MiniViGPT(nn.Module):
    """Educational decoder-only language model for Vietnamese causal LM pretraining."""

    def __init__(self, config: MiniViGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.dim, config.rms_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        if config.scaled_residual_init:
            self._init_residual_projections()
        if config.tie_embeddings:
            # Press & Wolf (2016): share input embedding and output classifier weights.
            self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def _init_residual_projections(self) -> None:
        # GPT-2/nanoGPT-style depth scaling for the two residual-output projections.
        std = self.config.init_std / math.sqrt(2 * self.config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=std)
            nn.init.normal_(block.ffn.down_proj.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")
        if input_ids.size(1) == 0:
            raise ValueError("input_ids cannot be empty")
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError("Input sequence is longer than max_seq_len")
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets must have the same [B, T] shape as input_ids")

        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    @staticmethod
    def _apply_top_k_top_p(
        logits: torch.Tensor,
        top_k: int | None,
        top_p: float | None,
    ) -> torch.Tensor:
        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            threshold = torch.topk(logits, k).values[:, [-1]]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative > top_p
            # Keep the first token whose inclusion crosses p.
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            filtered = torch.full_like(logits, float("-inf"))
            logits = filtered.scatter(1, sorted_indices, sorted_logits)
        return logits

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = 50,
        top_p: float | None = 0.95,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.size(1) == 0:
            raise ValueError("generation input_ids must have shape [B, T] with T > 0")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if top_p is not None and not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")

        was_training = self.training
        self.eval()
        try:
            return self._generate_loop(
                input_ids, max_new_tokens, temperature, top_k, top_p, eos_id
            )
        finally:
            self.train(was_training)

    def _generate_loop(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        eos_id: int | None,
    ) -> torch.Tensor:
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
        for _ in range(max_new_tokens):
            idx = input_ids[:, -self.config.max_seq_len :]
            logits, _ = self(idx)
            logits = logits[:, -1, :]

            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                logits = self._apply_top_k_top_p(logits, top_k=top_k, top_p=top_p)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            if eos_id is not None:
                eos_fill = torch.full_like(next_token, eos_id)
                next_token = torch.where(finished[:, None], eos_fill, next_token)
                finished |= next_token.squeeze(1).eq(eos_id)

            input_ids = torch.cat((input_ids, next_token), dim=1)
            if eos_id is not None and bool(finished.all()):
                break
        return input_ids

    def num_parameters(self, trainable_only: bool = False) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (p for p in parameters if p.requires_grad)
        return sum(p.numel() for p in parameters)

    def config_dict(self) -> dict:
        return asdict(self.config)
