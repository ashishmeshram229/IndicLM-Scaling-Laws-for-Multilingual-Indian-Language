"""Hardware detection used to recommend a feasible resource profile.

Design note: rather than assume GPU availability, every entry point that
depends on compute (training, benchmarking) should route through
`detect_hardware()` first so the system can pick an honest configuration
profile from `configs/profiles/` (cpu.yaml, 8gb_gpu.yaml, ...). We never
claim distributed or GPU capability that hasn't actually been detected.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareInfo:
    device_type: str  # "cpu" | "cuda" | "mps"
    num_gpus: int
    gpu_names: list[str]
    gpu_memory_gb: list[float]
    cpu_count: int
    total_ram_gb: float
    recommended_profile: str


def detect_hardware() -> HardwareInfo:
    import os

    import psutil  # type: ignore[import-untyped]

    cpu_count = os.cpu_count() or 1
    total_ram_gb = psutil.virtual_memory().total / (1024**3)

    device_type = "cpu"
    num_gpus = 0
    gpu_names: list[str] = []
    gpu_memory_gb: list[float] = []

    try:
        import torch

        if torch.cuda.is_available():
            device_type = "cuda"
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                gpu_names.append(props.name)
                gpu_memory_gb.append(props.total_memory / (1024**3))
        elif torch.backends.mps.is_available():  # Apple Silicon
            device_type = "mps"
    except ImportError:
        pass  # torch not installed yet; report cpu-only honestly

    if device_type == "cpu":
        profile = "cpu"
    elif device_type == "mps":
        profile = "cpu"  # MPS training path not yet validated; fall back honestly
    elif num_gpus > 1:
        profile = "multi_gpu"
    elif gpu_memory_gb and gpu_memory_gb[0] < 10:
        profile = "8gb_gpu"
    elif gpu_memory_gb and gpu_memory_gb[0] < 20:
        profile = "16gb_gpu"
    else:
        profile = "24gb_gpu"

    return HardwareInfo(
        device_type=device_type,
        num_gpus=num_gpus,
        gpu_names=gpu_names,
        gpu_memory_gb=gpu_memory_gb,
        cpu_count=cpu_count,
        total_ram_gb=round(total_ram_gb, 2),
        recommended_profile=profile,
    )
