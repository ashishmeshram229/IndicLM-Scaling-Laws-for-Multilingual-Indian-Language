"""Unit tests for hardware detection.

These tests must pass on CPU-only CI runners, so they assert on structure
and honesty (no GPU claimed when none exists), never on GPU presence.
"""

from indiclm.utils.hardware import HardwareInfo, detect_hardware


def test_detect_hardware_returns_hardware_info() -> None:
    hw = detect_hardware()
    assert isinstance(hw, HardwareInfo)
    assert hw.device_type in {"cpu", "cuda", "mps"}
    assert hw.cpu_count >= 1
    assert hw.total_ram_gb > 0


def test_no_gpu_claimed_without_cuda() -> None:
    hw = detect_hardware()
    if hw.device_type == "cpu":
        assert hw.num_gpus == 0
        assert hw.gpu_names == []
        assert hw.gpu_memory_gb == []


def test_recommended_profile_is_known() -> None:
    hw = detect_hardware()
    assert hw.recommended_profile in {
        "cpu",
        "8gb_gpu",
        "16gb_gpu",
        "24gb_gpu",
        "multi_gpu",
    }
