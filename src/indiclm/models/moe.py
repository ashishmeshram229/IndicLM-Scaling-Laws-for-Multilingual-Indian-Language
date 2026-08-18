"""Experimental Mixture-of-Experts feed-forward layer, behind
`ModelConfig.use_moe`. Not required for the first working model version;
this exists so scaling/routing experiments can be run later.

Implements: a linear router, top-k routing, per-expert SwiGLU FFNs, a
capacity factor with token dropping when experts overflow, and an
auxiliary load-balancing loss (Switch-Transformer style: encourages the
router's average dispatch probability and average dispatch fraction per
expert to be anti-correlated is not used here; we use the simpler
"importance * load" product formulation).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from indiclm.models.layers import SwiGLU


@dataclass
class MoERoutingStats:
    tokens_per_expert: list[int]
    dropped_tokens: int
    total_tokens: int
    routing_entropy: float
    load_imbalance: float  # max(tokens_per_expert) / mean(tokens_per_expert)

    def to_dict(self) -> dict:
        return self.__dict__


class MoEFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int,
        capacity_factor: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(d_model, d_ff, dropout) for _ in range(num_experts)]
        )
        self.last_stats: MoERoutingStats | None = None
        self.last_aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        flat_x = x.view(-1, d)  # (n_tokens, d_model)
        n_tokens = flat_x.shape[0]

        logits = self.router(flat_x)  # (n_tokens, num_experts)
        probs = F.softmax(logits, dim=-1)
        top_probs, top_idx = probs.topk(self.top_k, dim=-1)  # (n_tokens, top_k)

        capacity = max(1, int(self.capacity_factor * n_tokens * self.top_k / self.num_experts))
        output = torch.zeros_like(flat_x)
        tokens_per_expert = [0] * self.num_experts
        dropped = 0

        for expert_id in range(self.num_experts):
            expert_mask = top_idx == expert_id  # (n_tokens, top_k)
            token_idx, slot_idx = expert_mask.nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            if token_idx.numel() > capacity:
                # Drop lowest-weight tokens beyond capacity (Switch-style).
                weights = top_probs[token_idx, slot_idx]
                keep = torch.topk(weights, capacity).indices
                dropped += token_idx.numel() - capacity
                token_idx, slot_idx = token_idx[keep], slot_idx[keep]
            tokens_per_expert[expert_id] = int(token_idx.numel())
            expert_out = self.experts[expert_id](flat_x[token_idx])
            weight = top_probs[token_idx, slot_idx].unsqueeze(-1)
            output.index_add_(0, token_idx, expert_out * weight)

        # Load-balancing auxiliary loss: num_experts * sum_i (fraction of
        # tokens routed to expert i) * (mean router probability for expert i)
        with torch.no_grad():
            dispatch_fraction = torch.tensor(
                [c / max(n_tokens * self.top_k, 1) for c in tokens_per_expert],
                device=x.device,
            )
        mean_prob = probs.mean(dim=0)
        self.last_aux_loss = self.num_experts * (dispatch_fraction * mean_prob).sum()

        total_routed = sum(tokens_per_expert)
        entropy = 0.0
        if total_routed > 0:
            fracs = [c / total_routed for c in tokens_per_expert if c > 0]
            entropy = -sum(f * torch.log(torch.tensor(f)).item() for f in fracs)
        mean_load = total_routed / self.num_experts if self.num_experts else 0.0
        imbalance = (max(tokens_per_expert) / mean_load) if mean_load > 0 else 0.0

        self.last_stats = MoERoutingStats(
            tokens_per_expert=tokens_per_expert,
            dropped_tokens=dropped,
            total_tokens=n_tokens * self.top_k,
            routing_entropy=round(entropy, 4),
            load_imbalance=round(imbalance, 4),
        )
        return output.view(b, t, d)
