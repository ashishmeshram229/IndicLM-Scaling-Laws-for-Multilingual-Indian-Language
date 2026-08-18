"""Experiment-tracking abstraction. `LocalTracker` (JSONL of logged
metrics, always on) is the source of truth; `MLflowTracker` is an optional
adapter behind the same interface, used only if `mlflow` is installed and
a tracking URI is configured. Callers depend on `ExperimentTracker`, never
on MLflow directly — swapping backends (W&B, etc.) means adding one more
adapter class.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Protocol


class ExperimentTracker(Protocol):
    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, float], step: int) -> None: ...
    def log_artifact(self, path: Path) -> None: ...
    def close(self) -> None: ...


class LocalTracker:
    """Always-on local tracker: writes params.json once and appends metrics
    to metrics.jsonl. This is the manifest system's companion, not a
    replacement for `experiments.manifest` (which captures reproducibility
    metadata) — this tracks the metric *time series* during a run."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_file = open(self.run_dir / "tracked_metrics.jsonl", "a", encoding="utf-8")

    def log_params(self, params: dict[str, Any]) -> None:
        (self.run_dir / "params.json").write_text(json.dumps(params, indent=2, default=str))

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        record = {"step": step, "timestamp": time.time(), **metrics}
        self._metrics_file.write(json.dumps(record) + "\n")
        self._metrics_file.flush()

    def log_artifact(self, path: Path) -> None:
        # Local tracker treats "logging" an artifact as recording its path;
        # the file is already on disk under the experiment directory.
        with open(self.run_dir / "artifacts.txt", "a", encoding="utf-8") as f:
            f.write(str(path) + "\n")

    def close(self) -> None:
        self._metrics_file.close()


class MLflowTracker:
    """Optional MLflow-backed tracker. Only usable if `mlflow` is
    installed and `mlflow.set_tracking_uri` succeeds; construction raises
    `RuntimeError` otherwise so callers fail fast with a clear message
    rather than silently no-op logging."""

    def __init__(self, experiment_name: str, tracking_uri: str | None = None) -> None:
        try:
            import mlflow
        except ImportError as e:
            raise RuntimeError(
                "MLflowTracker requires the optional 'mlflow' dependency "
                "(pip install 'indiclm[tracking]')."
            ) from e
        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run = mlflow.start_run()

    def log_params(self, params: dict[str, Any]) -> None:
        flat = {k: v for k, v in params.items() if isinstance(v, (str, int, float, bool))}
        self._mlflow.log_params(flat)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: Path) -> None:
        self._mlflow.log_artifact(str(path))

    def close(self) -> None:
        self._mlflow.end_run()


def get_tracker(run_dir: Path, mlflow_uri: str | None = None, experiment_name: str = "indiclm") -> ExperimentTracker:
    """Returns the local tracker, plus MLflow if configured. Composing two
    trackers behind one call site is handled by `CompositeTracker`."""
    trackers: list[ExperimentTracker] = [LocalTracker(run_dir)]
    if mlflow_uri:
        try:
            trackers.append(MLflowTracker(experiment_name, mlflow_uri))
        except RuntimeError:
            pass  # mlflow not installed; local tracking still runs
    return CompositeTracker(trackers) if len(trackers) > 1 else trackers[0]


class CompositeTracker:
    def __init__(self, trackers: list[ExperimentTracker]) -> None:
        self._trackers = trackers

    def log_params(self, params: dict[str, Any]) -> None:
        for t in self._trackers:
            t.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        for t in self._trackers:
            t.log_metrics(metrics, step)

    def log_artifact(self, path: Path) -> None:
        for t in self._trackers:
            t.log_artifact(path)

    def close(self) -> None:
        for t in self._trackers:
            t.close()
