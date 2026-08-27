"""Unit tests for zero-shot sentiment classification eval.

Builds a tiny tokenizer + untrained model from scratch (never touches
data/eval/ or a real checkpoint) so these pass in CI without the
project's trained artifacts being present.
"""

from __future__ import annotations

from pathlib import Path

import torch

from indiclm.data.pipeline import DataPipelineConfig, run_pipeline
from indiclm.evaluation.downstream import (
    LABELS,
    evaluate_downstream_sentiment,
    load_sentiment_examples,
)
from indiclm.models.config import ModelConfig
from indiclm.models.transformer import DecoderOnlyTransformer
from indiclm.tokenizer.train import TokenizerTrainConfig, train_tokenizer
from indiclm.training.checkpoint import save_checkpoint

_ENG_LINES = [
    "The sun rises in the east every morning.",
    "Water boils at a high temperature.",
    "Children enjoy playing games after school.",
] * 10


def _build_tiny_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    """Mirrors test_end_to_end_pipeline.py's tokenizer/model setup, sized
    just large enough that "positive"/" negative" tokenize without error."""
    raw_dir = tmp_path / "raw" / "wiki_sample"
    raw_dir.mkdir(parents=True)
    (raw_dir / "eng.txt").write_text("\n".join(_ENG_LINES), encoding="utf-8")

    processed_dir = tmp_path / "processed"
    run_pipeline(DataPipelineConfig(raw_dir=tmp_path / "raw", output_dir=processed_dir))

    tok_cfg = TokenizerTrainConfig(
        input_dir=processed_dir, output_dir=tmp_path / "tokenizer", vocab_size=256, model_type="bpe"
    )
    tokenizer_path = train_tokenizer(tok_cfg)

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    model_config = ModelConfig(vocab_size=sp.get_piece_size(), d_model=16, n_layers=1, n_heads=2, max_seq_len=64)
    model = DecoderOnlyTransformer(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path, model, optimizer, scheduler=None, step=0, tokens_seen=0,
        config={"model_config": vars(model_config)},
    )
    return checkpoint_path, tokenizer_path


def _write_eval_set(eval_dir: Path) -> None:
    eval_dir.mkdir(parents=True)
    (eval_dir / "eng.jsonl").write_text(
        '{"text": "I loved this movie, it was wonderful.", "label": "positive"}\n'
        '{"text": "This was a terrible and boring experience.", "label": "negative"}\n',
        encoding="utf-8",
    )


def test_load_sentiment_examples(tmp_path: Path) -> None:
    _write_eval_set(tmp_path / "eval")
    by_language = load_sentiment_examples(tmp_path / "eval")
    assert set(by_language) == {"eng"}
    assert len(by_language["eng"]) == 2
    assert {ex.label for ex in by_language["eng"]} == {"positive", "negative"}


def test_evaluate_downstream_sentiment_runs_end_to_end(tmp_path: Path) -> None:
    checkpoint_path, tokenizer_path = _build_tiny_checkpoint(tmp_path)
    _write_eval_set(tmp_path / "eval")

    report = evaluate_downstream_sentiment(checkpoint_path, tokenizer_path, eval_dir=tmp_path / "eval")

    assert report.task == "sentiment_classification"
    assert report.n_examples == 2
    assert 0.0 <= report.overall_accuracy <= 1.0
    assert report.chance_accuracy == round(1.0 / len(LABELS), 4)
    assert "eng" in report.per_language
    assert report.per_language["eng"].n_examples == 2
    # Every prediction must be one of the two label words, never something
    # else (e.g. an empty string if scoring degenerated).
    for pred in report.per_language["eng"].predictions:
        assert pred["predicted"] in LABELS


def test_evaluate_downstream_sentiment_raises_on_empty_eval_dir(tmp_path: Path) -> None:
    checkpoint_path, tokenizer_path = _build_tiny_checkpoint(tmp_path)
    empty_dir = tmp_path / "empty_eval"
    empty_dir.mkdir()
    try:
        evaluate_downstream_sentiment(checkpoint_path, tokenizer_path, eval_dir=empty_dir)
        raise AssertionError("expected ValueError for an eval dir with no examples")
    except ValueError:
        pass
