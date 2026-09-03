import copy

import torch

from minivigpt.config import MiniViGPTConfig, llama_swiglu_hidden_dim
from minivigpt.model import MiniViGPT, RMSNorm, RotaryEmbedding


def tiny_config(**overrides):
    values = dict(
        vocab_size=128,
        max_seq_len=32,
        dim=64,
        n_layers=2,
        n_heads=4,
        hidden_dim=128,
        ffn_multiple_of=64,
        dropout=0.0,
        attention_impl="manual",
    )
    values.update(overrides)
    return MiniViGPTConfig(**values)


def test_llama_swiglu_width_rule():
    assert llama_swiglu_hidden_dim(384, 256) == 1024
    assert llama_swiglu_hidden_dim(128, 64) == 384


def test_rmsnorm_preserves_shape_and_normalizes_rms():
    norm = RMSNorm(16, eps=1e-8)
    x = torch.randn(2, 5, 16)
    y = norm(x)
    assert y.shape == x.shape
    rms = y.float().pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=2e-4, rtol=2e-4)


def test_rope_preserves_vector_norm_per_position():
    rope = RotaryEmbedding(head_dim=8, max_seq_len=16)
    q = torch.randn(2, 3, 10, 8)
    k = torch.randn(2, 3, 10, 8)
    qr, kr = rope(q, k)
    assert torch.allclose(q.norm(dim=-1), qr.norm(dim=-1), atol=1e-5, rtol=1e-5)
    assert torch.allclose(k.norm(dim=-1), kr.norm(dim=-1), atol=1e-5, rtol=1e-5)


def test_causal_prefix_invariance_to_future_tokens():
    torch.manual_seed(0)
    model = MiniViGPT(tiny_config()).eval()
    a = torch.randint(0, 128, (1, 12))
    b = a.clone()
    b[:, 8:] = torch.randint(0, 128, (1, 4))
    logits_a, _ = model(a)
    logits_b, _ = model(b)
    # Positions 0..7 cannot depend on tokens 8..11.
    assert torch.allclose(logits_a[:, :8], logits_b[:, :8], atol=1e-6, rtol=1e-6)


def test_sdpa_matches_manual_attention_without_dropout():
    if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        return
    manual = MiniViGPT(tiny_config(attention_impl="manual")).eval()
    sdpa_cfg = tiny_config(attention_impl="sdpa")
    sdpa = MiniViGPT(sdpa_cfg).eval()
    sdpa.load_state_dict(copy.deepcopy(manual.state_dict()))
    x = torch.randint(0, 128, (2, 10))
    a, _ = manual(x)
    b, _ = sdpa(x)
    assert torch.allclose(a, b, atol=2e-5, rtol=2e-5)


def test_scaled_residual_projection_is_smaller_than_base_init():
    torch.manual_seed(0)
    cfg = tiny_config(init_std=0.02, scaled_residual_init=True)
    model = MiniViGPT(cfg)
    q_std = model.blocks[0].attn.q_proj.weight.std().item()
    out_std = model.blocks[0].attn.out_proj.weight.std().item()
    assert out_std < q_std
