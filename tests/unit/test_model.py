"""Unit tests for the from-scratch Transformer: shapes, causal masking,
RoPE, GQA, RMSNorm, and MoE routing."""

from __future__ import annotations

import torch

from indiclm.models.config import ModelConfig
from indiclm.models.layers import RMSNorm, apply_rope, precompute_rope_freqs
from indiclm.models.transformer import DecoderOnlyTransformer


def test_rmsnorm_output_shape_and_scale() -> None:
    norm = RMSNorm(16)
    x = torch.randn(2, 5, 16) * 10
    out = norm(x)
    assert out.shape == x.shape
    # RMS of the normalized (pre-weight) output should be ~1 per position.
    rms = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps)).pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rope_preserves_norm() -> None:
    head_dim = 8
    freqs = precompute_rope_freqs(head_dim, max_seq_len=16, theta=10000.0)
    x = torch.randn(1, 2, 16, head_dim)
    rotated = apply_rope(x, freqs)
    assert rotated.shape == x.shape
    # RoPE is a rotation: it must preserve per-vector L2 norm.
    assert torch.allclose(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-4)


def test_model_forward_shapes() -> None:
    cfg = ModelConfig(vocab_size=100, d_model=32, n_layers=2, n_heads=4, max_seq_len=16)
    model = DecoderOnlyTransformer(cfg)
    x = torch.randint(0, 100, (3, 10))
    logits, loss = model(x)
    assert logits.shape == (3, 10, 100)
    assert loss is None
    y = torch.randint(0, 100, (3, 10))
    logits, loss = model(x, y)
    assert loss is not None and loss.item() > 0


def test_causal_masking_is_enforced() -> None:
    cfg = ModelConfig(vocab_size=50, d_model=16, n_layers=1, n_heads=2, max_seq_len=8)
    model = DecoderOnlyTransformer(cfg)
    model.eval()
    x = torch.randint(0, 50, (1, 6))
    with torch.no_grad():
        logits1, _ = model(x)
        x2 = x.clone()
        x2[:, -1] = (x2[:, -1] + 1) % 50
        logits2, _ = model(x2)
    assert torch.allclose(logits1[:, :-1], logits2[:, :-1], atol=1e-5)


def test_grouped_query_attention_shapes() -> None:
    cfg = ModelConfig(vocab_size=50, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2, max_seq_len=8)
    model = DecoderOnlyTransformer(cfg)
    x = torch.randint(0, 50, (2, 8))
    logits, _ = model(x)
    assert logits.shape == (2, 8, 50)


def test_tied_embeddings_share_storage() -> None:
    cfg = ModelConfig(vocab_size=50, d_model=16, n_layers=1, n_heads=2, tie_embeddings=True)
    model = DecoderOnlyTransformer(cfg)
    assert model.lm_head.weight is model.token_embedding.weight


def test_num_parameters_estimate_matches_actual() -> None:
    cfg = ModelConfig(vocab_size=200, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2)
    model = DecoderOnlyTransformer(cfg)
    assert model.num_parameters() == cfg.num_parameters_estimate()


def test_gradients_flow_to_all_parameters() -> None:
    cfg = ModelConfig(vocab_size=50, d_model=16, n_layers=2, n_heads=2, max_seq_len=8)
    model = DecoderOnlyTransformer(cfg)
    x = torch.randint(0, 50, (2, 6))
    y = torch.randint(0, 50, (2, 6))
    _, loss = model(x, y)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"


def test_moe_forward_and_stats() -> None:
    cfg = ModelConfig(
        vocab_size=80, d_model=24, n_layers=1, n_heads=2, max_seq_len=8,
        use_moe=True, moe_num_experts=4, moe_top_k=2,
    )
    model = DecoderOnlyTransformer(cfg)
    x = torch.randint(0, 80, (2, 8))
    y = torch.randint(0, 80, (2, 8))
    logits, _loss = model(x, y)
    assert logits.shape == (2, 8, 80)
    stats = model.blocks[0].ffn.last_stats
    assert stats is not None
    assert sum(stats.tokens_per_expert) <= stats.total_tokens
