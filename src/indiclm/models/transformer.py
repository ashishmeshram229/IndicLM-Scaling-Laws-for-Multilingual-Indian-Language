"""Decoder-only Transformer assembled from `layers.py` / `moe.py`
primitives, per `ModelConfig`. This is the from-scratch model used by the
training engine — not a wrapped Hugging Face `GPT2LMHeadModel` or similar.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from indiclm.models.config import ModelConfig
from indiclm.models.layers import CausalSelfAttention, RMSNorm, SwiGLU
from indiclm.models.moe import MoEFeedForward


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attn = CausalSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads or config.n_heads,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
            dropout=config.dropout,
        )
        self.ffn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        if config.use_moe:
            self.ffn: nn.Module = MoEFeedForward(
                d_model=config.d_model,
                d_ff=config.d_ff or config.d_model * 4,
                num_experts=config.moe_num_experts,
                top_k=config.moe_top_k,
                capacity_factor=config.moe_capacity_factor,
                dropout=config.dropout,
            )
        else:
            self.ffn = SwiGLU(config.d_model, config.d_ff or config.d_model * 4, config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_embedding.weight.numel()
        return n

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.dropout(self.token_embedding(input_ids))
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
            if self.config.use_moe:
                aux_losses = [
                    b.ffn.last_aux_loss
                    for b in self.blocks
                    if isinstance(b.ffn, MoEFeedForward) and b.ffn.last_aux_loss is not None
                ]
                if aux_losses:
                    loss = loss + 0.01 * torch.stack(aux_losses).mean()
        return logits, loss

    @torch.no_grad()
    def generate(
        self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            cond = input_ids[:, -self.config.max_seq_len :]
            logits, _ = self(cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
