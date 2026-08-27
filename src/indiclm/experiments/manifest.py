"""Reproducibility manifest: every experiment run gets one, independent of
whether a hosted experiment tracker (MLflow/W&B) is also configured — per
the project principle "never rely solely on a hosted dashboard."
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from indiclm.utils.hardware import detect_hardware


def _git_commit(repo_dir: Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir, capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        return "unknown (not a git repository or git unavailable)"


def _config_hash(config: dict[str, Any]) -> str:
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


@dataclass
class ExperimentManifest:
    experiment_id: str
    git_commit: str
    config_hash: str
    config: dict[str, Any]
    dataset_version: str
    tokenizer_version: str
    seed: int
    hardware: dict[str, Any]
    software_versions: dict[str, str]
    created_at: float
    training_tokens: int | None = None
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    evaluation_metrics: dict[str, Any] = field(default_factory=dict)
    checkpoint_path: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


def build_manifest(
    experiment_id: str,
    config: Any,
    dataset_version: str,
    tokenizer_version: str,
    seed: int,
    repo_dir: Path | None = None,
) -> ExperimentManifest:
    config_dict = _jsonable(config)
    hw = detect_hardware()
    return ExperimentManifest(
        experiment_id=experiment_id,
        git_commit=_git_commit(repo_dir),
        config_hash=_config_hash(config_dict),
        config=config_dict,
        dataset_version=dataset_version,
        tokenizer_version=tokenizer_version,
        seed=seed,
        hardware=_jsonable(hw),
        software_versions={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": _torch_version(),
        },
        created_at=time.time(),
    )


def _torch_version() -> str:
    try:
        import torch

        return torch.__version__
    except ImportError:
        return "not installed"


def write_manifest(manifest: ExperimentManifest, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return path
