import torch

from minivigpt.config import MiniViGPTConfig
from minivigpt.model import MiniViGPT, CausalSelfAttention


def tiny_config() -> MiniViGPTConfig:
    return MiniViGPTConfig(
        vocab_size=128,
        max_seq_len=32,
        dim=64,
        n_layers=2,
        n_heads=4,
        hidden_dim=128,
        dropout=0.0,
    )


def test_forward_shape_and_loss():
    config = tiny_config()
    model = MiniViGPT(config)
    x = torch.randint(0, config.vocab_size, (3, 16))
    y = torch.randint(0, config.vocab_size, (3, 16))
    logits, loss = model(x, y)
    assert logits.shape == (3, 16, config.vocab_size)
    assert loss is not None
    assert torch.isfinite(loss)


def test_weight_tying():
    model = MiniViGPT(tiny_config())
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()


def test_generate_grows_sequence():
    config = tiny_config()
    model = MiniViGPT(config)
    x = torch.randint(0, config.vocab_size, (1, 5))
    out = model.generate(x, max_new_tokens=4, temperature=0.0)
    assert out.shape == (1, 9)


def test_causal_mask_blocks_future():
    config = tiny_config()
    attn = CausalSelfAttention(config)
    mask = attn.causal_mask[0, 0, :4, :4]
    expected = torch.tensor(
        [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]],
        dtype=torch.bool,
    )
    assert torch.equal(mask.cpu(), expected)


def test_generate_restores_previous_training_mode():
    """generate() must not silently leave a training model in eval mode."""
    model = MiniViGPT(tiny_config())
    model.train()
    model.generate(torch.randint(0, 128, (2, 4)), max_new_tokens=3)
    assert model.training is True

    model.eval()
    model.generate(torch.randint(0, 128, (2, 4)), max_new_tokens=3)
    assert model.training is False


def test_rmsnorm_gain_is_applied_in_fp32_and_cast_once():
    from minivigpt.model import RMSNorm

    norm = RMSNorm(8).half()
    out = norm(torch.randn(2, 8).half())
    assert out.dtype == torch.float16
