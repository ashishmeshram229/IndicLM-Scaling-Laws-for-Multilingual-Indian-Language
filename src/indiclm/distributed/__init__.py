"""Distributed training helpers: DDP setup, rank/world-size queries, and
model wrapping. Designed to work transparently in single-process (dev/CI)
mode — every function is safe to call even without a distributed backend.

When WORLD_SIZE > 1, call `init_distributed()` before training and
`cleanup_distributed()` after. `wrap_ddp(model)` is a no-op when
world_size == 1.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed(backend: str = "nccl") -> None:
    """Initialize the process group from environment variables set by
    torchrun / mpirun. Falls back to "gloo" when CUDA is unavailable
    (CPU-only runs, CI). Safe to call even if already initialized."""
    if is_distributed():
        return
    if not dist.is_available():
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return
    rank = int(os.environ.get("RANK", "0"))
    effective_backend = backend if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=effective_backend,
        rank=rank,
        world_size=world_size,
    )


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def wrap_ddp(model: torch.nn.Module, device_ids: list[int] | None = None) -> torch.nn.Module:
    """Wrap model in DDP if running distributed, otherwise return as-is."""
    if not is_distributed():
        return model
    return DDP(model, device_ids=device_ids)


__all__ = [
    "cleanup_distributed",
    "get_rank",
    "get_world_size",
    "init_distributed",
    "is_distributed",
    "is_main_process",
    "wrap_ddp",
]
