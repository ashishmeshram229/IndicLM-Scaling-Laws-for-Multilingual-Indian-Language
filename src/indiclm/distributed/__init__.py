"""Distributed training helpers: DDP setup, rank/world-size queries, metrics
all-reduce, and model wrapping. Transparent in single-process mode — every
function is safe to call without a distributed backend.

SLURM usage: launch with `srun --ntasks-per-node=<N> python train.py` or
`torchrun --nproc_per_node=<N> train.py`. Both set RANK, LOCAL_RANK,
WORLD_SIZE, MASTER_ADDR, MASTER_PORT in the environment.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DistributedSampler


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_local_rank() -> int:
    """Local rank within the current node (used for CUDA device assignment)."""
    if is_distributed():
        return int(os.environ.get("LOCAL_RANK", "0"))
    return 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed(backend: str = "nccl") -> None:
    """Initialise the process group from SLURM / torchrun environment variables.

    Falls back to 'gloo' when CUDA is unavailable (CPU-only / CI). Safe to
    call even if already initialised or on a single-process run.
    """
    if is_distributed():
        return
    if not dist.is_available():
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return

    # Provide safe defaults so the user doesn't have to set these manually
    # for single-node jobs. Multi-node jobs must export MASTER_ADDR explicitly.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    effective_backend = backend if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=effective_backend,
        init_method="env://",
        world_size=world_size,
        rank=int(os.environ.get("RANK", "0")),
    )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def barrier() -> None:
    """Block until all ranks reach this point."""
    if is_distributed():
        dist.barrier()


def wrap_ddp(model: torch.nn.Module, device_ids: list[int] | None = None) -> torch.nn.Module:
    """Wrap model in DDP if running distributed, otherwise return as-is."""
    if not is_distributed():
        return model
    return DDP(model, device_ids=device_ids, find_unused_parameters=False)


def all_reduce_dict(metrics: dict[str, float], op: str = "mean") -> dict[str, float]:
    """All-reduce a dict of scalar metrics across all ranks.

    op='mean' averages across ranks (default for loss/perplexity).
    op='sum' sums across ranks (useful for token counts).
    """
    if not is_distributed():
        return metrics

    keys = sorted(metrics.keys())
    tensor = torch.tensor([metrics[k] for k in keys], dtype=torch.float64)
    if op == "sum":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    else:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= get_world_size()

    return {k: float(tensor[i]) for i, k in enumerate(keys)}


def get_distributed_sampler(
    dataset: Dataset,
    shuffle: bool = True,
    seed: int = 0,
) -> DistributedSampler | None:
    """Return a DistributedSampler when running distributed, else None.

    Pass the result to DataLoader as `sampler=` and set `shuffle=False`
    (DistributedSampler handles shuffling internally).
    """
    if not is_distributed():
        return None
    return DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle,
        seed=seed,
    )


__all__ = [
    "all_reduce_dict",
    "barrier",
    "cleanup_distributed",
    "get_distributed_sampler",
    "get_local_rank",
    "get_rank",
    "get_world_size",
    "init_distributed",
    "is_distributed",
    "is_main_process",
    "wrap_ddp",
]
