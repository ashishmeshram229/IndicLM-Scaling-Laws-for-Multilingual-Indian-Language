"""Standalone language-modeling evaluation: overall and per-language
validation loss/perplexity, independent of the training loop.

"Independent of training" means: this module loads a checkpoint from
disk and a held-out shard directory, and never touches optimizer/
scheduler state — it can be invoked purely from `indiclm evaluate`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from indiclm.models.config import ModelConfig
from indiclm.models.transformer import DecoderOnlyTransformer
from indiclm.training.checkpoint import load_checkpoint
from indiclm.training.dataset import PackedTokenDataset


@dataclass
class LanguageEvalResult:
    language: str
    loss: float
    perplexity: float
    tokens_evaluated: int

    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class EvaluationReport:
    overall_loss: float
    overall_perplexity: float
    macro_avg_perplexity: float  # unweighted mean across languages
    weighted_avg_perplexity: float  # weighted by tokens evaluated
    per_language: dict[str, LanguageEvalResult]
    checkpoint: str
    tokenizer: str

    def to_dict(self) -> dict:
        return {
            "overall_loss": self.overall_loss,
            "overall_perplexity": self.overall_perplexity,
            "macro_avg_perplexity": self.macro_avg_perplexity,
            "weighted_avg_perplexity": self.weighted_avg_perplexity,
            "per_language": {k: v.to_dict() for k, v in self.per_language.items()},
            "checkpoint": self.checkpoint,
            "tokenizer": self.tokenizer,
        }


EVAL_TOKEN_BUDGET_PER_LANGUAGE = 3000  # small, fixed eval budget; not "as much as available"


@torch.no_grad()
def _eval_language(
    model: DecoderOnlyTransformer, shards_dir: Path, tokenizer_path: Path, language: str,
    seq_len: int, batch_size: int, device: torch.device,
) -> LanguageEvalResult:
    ds = PackedTokenDataset(
        shards_dir=shards_dir,
        tokenizer_path=tokenizer_path,
        seq_len=seq_len,
        total_tokens=EVAL_TOKEN_BUDGET_PER_LANGUAGE,
        alpha=1.0,
        languages=[language],
    )
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=batch_size)
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0
    model.eval()
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        _, loss = model(inputs, targets)
        total_loss += loss.item()
        total_tokens += inputs.numel()
        n_batches += 1
    mean_loss = total_loss / max(n_batches, 1)
    return LanguageEvalResult(
        language=language,
        loss=round(mean_loss, 6),
        perplexity=round(math.exp(min(mean_loss, 20)), 4),
        tokens_evaluated=total_tokens,
    )


def evaluate_checkpoint(
    checkpoint_path: Path,
    shards_dir: Path,
    tokenizer_path: Path,
    seq_len: int = 64,
    batch_size: int = 4,
    device: str = "cpu",
) -> EvaluationReport:
    import glob

    languages = sorted(Path(p).stem for p in glob.glob(str(Path(shards_dir) / "*.jsonl")))

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config_dict = checkpoint["config"]["model_config"]
    model_config = ModelConfig(**model_config_dict)
    model = DecoderOnlyTransformer(model_config).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)

    per_language: dict[str, LanguageEvalResult] = {}
    for lang in languages:
        try:
            per_language[lang] = _eval_language(
                model, shards_dir, tokenizer_path, lang, seq_len, batch_size, torch.device(device)
            )
        except ValueError:
            continue  # language had no usable shard text; skip rather than fail the whole report

    total_tokens = sum(r.tokens_evaluated for r in per_language.values())
    weighted_ppl = (
        sum(r.perplexity * r.tokens_evaluated for r in per_language.values()) / total_tokens
        if total_tokens
        else float("nan")
    )
    macro_ppl = (
        sum(r.perplexity for r in per_language.values()) / len(per_language)
        if per_language
        else float("nan")
    )
    overall_loss = (
        sum(r.loss * r.tokens_evaluated for r in per_language.values()) / total_tokens
        if total_tokens
        else float("nan")
    )

    return EvaluationReport(
        overall_loss=round(overall_loss, 6),
        overall_perplexity=round(math.exp(min(overall_loss, 20)), 4),
        macro_avg_perplexity=round(macro_ppl, 4),
        weighted_avg_perplexity=round(weighted_ppl, 4),
        per_language=per_language,
        checkpoint=str(checkpoint_path),
        tokenizer=str(tokenizer_path),
    )
