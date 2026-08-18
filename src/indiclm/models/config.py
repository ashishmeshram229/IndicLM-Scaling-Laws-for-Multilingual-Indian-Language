"""Configuration for the decoder-only Transformer. Every architectural
knob is here — nothing about model shape is hard-coded in `transformer.py`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int | None = None  # None => n_kv_heads = n_heads (standard MHA); < n_heads => GQA
    d_ff: int | None = None        # None => 8/3 * d_model rounded, per SwiGLU convention
    max_seq_len: int = 256
    dropout: float = 0.0
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    rms_norm_eps: float = 1e-6
    use_moe: bool = False
    moe_num_experts: int = 4
    moe_top_k: int = 2
    moe_capacity_factor: float = 1.25

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        if self.d_ff is None:
            # SwiGLU convention: hidden dim ~ 8/3 * d_model, rounded to nearest 32.
            hidden = int(8 * self.d_model / 3)
            self.d_ff = (hidden + 31) // 32 * 32

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def num_parameters_estimate(self) -> int:
        """Analytic non-embedding + embedding parameter count, used to pick
        model sizes for scaling experiments without instantiating the model."""
        emb = self.vocab_size * self.d_model
        attn_per_layer = (
            self.d_model * self.d_model  # q_proj
            + 2 * self.d_model * (self.n_kv_heads * self.head_dim)  # k_proj, v_proj
            + self.d_model * self.d_model  # o_proj
        )
        ffn_per_layer = 3 * self.d_model * self.d_ff  # SwiGLU: gate, up, down
        norm_per_layer = 2 * self.d_model
        per_layer = attn_per_layer + ffn_per_layer + norm_per_layer
        total = emb + self.n_layers * per_layer + self.d_model  # final norm
        if not self.tie_embeddings:
            total += self.vocab_size * self.d_model
        return total
