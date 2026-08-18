"""From-scratch Transformer building blocks: RMSNorm, RoPE, causal
multi-head/grouped-query attention, and a SwiGLU feed-forward network.

Deliberately implemented directly on `torch.nn` primitives rather than via
a pre-built GPT class, per the project requirement to implement the
architecture rather than import one.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean-centering, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float) -> torch.Tensor:
    """Returns complex-exponential RoPE frequencies of shape (max_seq_len, head_dim/2)."""
    assert head_dim % 2 == 0, "RoPE requires an even head_dim"
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, freqs)  # (max_seq_len, head_dim/2)
    return torch.polar(torch.ones_like(angles), angles)  # complex64


def apply_rope(x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding. x: (batch, heads, seq_len, head_dim)."""
    b, h, t, d = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(b, h, t, d // 2, 2))
    freqs = rope_freqs[:t].view(1, 1, t, d // 2)
    x_rotated = x_complex * freqs
    out = torch.view_as_real(x_rotated).reshape(b, h, t, d)
    return out.type_as(x)


class CausalSelfAttention(nn.Module):
    """Causal multi-head attention with optional grouped-query attention
    (n_kv_heads < n_heads shares each KV head across n_heads/n_kv_heads
    query heads) and RoPE positional encoding."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        rope_theta: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

        rope_freqs = precompute_rope_freqs(self.head_dim, max_seq_len, rope_theta)
        self.register_buffer("rope_freqs", rope_freqs, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, self.rope_freqs)
        k = apply_rope(k, self.rope_freqs)

        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim)
        return self.resid_dropout(self.o_proj(attn_out))


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: down(silu(gate(x)) * up(x))."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))
