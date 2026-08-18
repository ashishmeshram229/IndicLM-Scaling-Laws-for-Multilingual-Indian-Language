"""Experiment runner: turns a config dict into a full run — dataset ->
model -> training -> evaluation -> manifest -> markdown report — all
under `experiments/manifests/<experiment_id>/`.

This is the implementation behind `indiclm experiment run`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split

from indiclm.evaluation.perplexity import evaluate_checkpoint
from indiclm.experiments.manifest import build_manifest, write_manifest
from indiclm.experiments.report import render_report
from indiclm.experiments.tracking import get_tracker
from indiclm.models.config import ModelConfig
from indiclm.training.dataset import PackedTokenDataset
from indiclm.training.trainer import TrainingConfig, train
from indiclm.utils.logging import get_logger

log = get_logger(__name__)


def run_experiment(config: dict[str, Any], experiments_root: Path = Path("experiments/manifests")) -> dict[str, Any]:
    experiment_id = config["experiment_id"]
    out_dir = Path(experiments_root) / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = config["data"]
    model_cfg_dict = dict(config["model"])
    train_cfg_dict = dict(config["training"])
    seed = config.get("seed", 0)

    torch.manual_seed(seed)
    dataset = PackedTokenDataset(
        shards_dir=Path(data_cfg["shards_dir"]),
        tokenizer_path=Path(data_cfg["tokenizer_path"]),
        seq_len=data_cfg["seq_len"],
        total_tokens=data_cfg["total_tokens"],
        alpha=data_cfg.get("alpha", 1.0),
        seed=seed,
        languages=data_cfg.get("languages"),
        manual_weights=data_cfg.get("manual_weights"),
    )
    val_frac = data_cfg.get("val_fraction", 0.1)
    n_val = max(1, int(val_frac * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    batch_size = train_cfg_dict.get("micro_batch_size", 4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model_cfg_dict["vocab_size"] = dataset.sp.get_piece_size()
    model_cfg_dict["max_seq_len"] = data_cfg["seq_len"]
    model_config = ModelConfig(**model_cfg_dict)

    train_cfg_dict["output_dir"] = out_dir
    train_cfg_dict["seed"] = seed
    training_config = TrainingConfig(**train_cfg_dict)

    tracker = get_tracker(out_dir)
    tracker.log_params({"model": model_cfg_dict, "training": train_cfg_dict, "data": data_cfg})

    result = train(model_config, training_config, train_loader, val_loader)
    for record in result.history:
        tracker.log_metrics(
            {k: v for k, v in record.items() if isinstance(v, (int, float))}, record["step"]
        )
    tracker.close()

    final_ckpt = out_dir / "checkpoints" / "final.pt"
    eval_report = evaluate_checkpoint(
        checkpoint_path=final_ckpt,
        shards_dir=Path(data_cfg["shards_dir"]),
        tokenizer_path=Path(data_cfg["tokenizer_path"]),
        seq_len=data_cfg["seq_len"],
        batch_size=batch_size,
    )
    (out_dir / "evaluation.json").write_text(json.dumps(eval_report.to_dict(), indent=2))

    manifest = build_manifest(
        experiment_id=experiment_id,
        config=config,
        dataset_version=data_cfg.get("dataset_version", "v1"),
        tokenizer_version=data_cfg.get("tokenizer_version", "v1"),
        seed=seed,
    )
    manifest.training_tokens = result.tokens_seen
    manifest.final_train_loss = result.final_train_loss
    manifest.final_val_loss = result.final_val_loss
    manifest.evaluation_metrics = eval_report.to_dict()
    manifest.checkpoint_path = str(final_ckpt)
    write_manifest(manifest, out_dir)

    render_report(
        experiment_id=experiment_id,
        config=config,
        result=result,
        eval_report=eval_report,
        dataset_stats=dataset.stats,
        manifest=manifest,
        out_path=out_dir / "report.md",
    )

    log.info(
        "experiment_complete",
        experiment_id=experiment_id,
        final_val_loss=result.final_val_loss,
        overall_perplexity=eval_report.overall_perplexity,
    )
    return {
        "experiment_id": experiment_id,
        "result": result.to_dict(),
        "evaluation": eval_report.to_dict(),
        "manifest_path": str(out_dir / "manifest.json"),
    }


def compare_experiments(experiment_ids: list[str], experiments_root: Path = Path("experiments/manifests")) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for exp_id in experiment_ids:
        manifest_path = Path(experiments_root) / exp_id / "manifest.json"
        if not manifest_path.exists():
            comparison[exp_id] = {"status": "not_run", "note": "no manifest.json found"}
            continue
        comparison[exp_id] = json.loads(manifest_path.read_text())
    return comparison
