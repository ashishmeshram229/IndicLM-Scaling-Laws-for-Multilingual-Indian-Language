"""`indiclm train` and `indiclm evaluate` top-level commands (as opposed to
`indiclm experiment run`, which wraps training + evaluation + manifest +
report generation together for registry-tracked experiments)."""

from __future__ import annotations

from pathlib import Path

import torch
import typer
import yaml
from rich.console import Console
from torch.utils.data import DataLoader, random_split

from indiclm.evaluation.perplexity import evaluate_checkpoint
from indiclm.models.config import ModelConfig
from indiclm.training.dataset import PackedTokenDataset
from indiclm.training.trainer import TrainingConfig, train as run_training
from indiclm.utils.logging import configure_logging

app = typer.Typer(help="Train and evaluate models directly (ad-hoc, outside the experiment registry).")
console = Console()


@app.command()
def train(config: Path = typer.Option(..., help="YAML with `model`, `training`, `data` sections.")) -> None:
    configure_logging()
    cfg = yaml.safe_load(Path(config).read_text())
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)

    data_cfg = cfg["data"]
    dataset = PackedTokenDataset(
        shards_dir=Path(data_cfg["shards_dir"]),
        tokenizer_path=Path(data_cfg["tokenizer_path"]),
        seq_len=data_cfg["seq_len"],
        total_tokens=data_cfg["total_tokens"],
        alpha=data_cfg.get("alpha", 1.0),
        seed=seed,
    )
    n_val = max(1, int(data_cfg.get("val_fraction", 0.1) * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
    batch_size = cfg["training"].get("micro_batch_size", 4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model_cfg_dict = dict(cfg["model"])
    model_cfg_dict["vocab_size"] = dataset.sp.get_piece_size()
    model_cfg_dict["max_seq_len"] = data_cfg["seq_len"]
    model_config = ModelConfig(**model_cfg_dict)

    train_cfg_dict = dict(cfg["training"])
    train_cfg_dict["seed"] = seed
    training_config = TrainingConfig(**train_cfg_dict)

    result = run_training(model_config, training_config, train_loader, val_loader)
    console.print(f"[green]Training complete.[/green] final_val_loss={result.final_val_loss}")


@app.command()
def evaluate(
    checkpoint: Path = typer.Option(...),
    shards_dir: Path = typer.Option(Path("data/processed")),
    tokenizer_path: Path = typer.Option(Path("data/tokenizer_v1/indiclm_tokenizer.model")),
    seq_len: int = typer.Option(64),
) -> None:
    configure_logging()
    report = evaluate_checkpoint(checkpoint, shards_dir, tokenizer_path, seq_len)
    console.print(report.to_dict())
