"""Integration test: dataset -> tokenizer -> dataloader -> model ->
training step -> checkpoint, on a tiny synthetic corpus generated in a
pytest tmp_path (never touches data/, so this passes in CI without the
project's hand-authored bootstrap corpus being present).
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from indiclm.data.pipeline import DataPipelineConfig, run_pipeline
from indiclm.models.config import ModelConfig
from indiclm.tokenizer.train import TokenizerTrainConfig, train_tokenizer
from indiclm.training.checkpoint import load_checkpoint
from indiclm.training.dataset import PackedTokenDataset
from indiclm.training.trainer import TrainingConfig, train

_ENG_LINES = [
    "The sun rises in the east every morning.",
    "Water boils at a high temperature.",
    "Children enjoy playing games after school.",
    "Books can take you to imaginary worlds.",
    "The library has many interesting stories.",
] * 5

_HIN_LINES = [
    "सूरज हर सुबह पूर्व में उगता है।",
    "पानी उच्च तापमान पर उबलता है।",
    "बच्चे स्कूल के बाद खेल खेलना पसंद करते हैं।",
    "किताबें आपको कल्पना की दुनिया में ले जा सकती हैं।",
    "पुस्तकालय में कई दिलचस्प कहानियाँ हैं।",
] * 5


def _build_tiny_corpus(root: Path) -> None:
    raw_dir = root / "raw" / "wiki_sample"
    raw_dir.mkdir(parents=True)
    (raw_dir / "eng.txt").write_text("\n".join(_ENG_LINES), encoding="utf-8")
    (raw_dir / "hin.txt").write_text("\n".join(_HIN_LINES), encoding="utf-8")


def test_full_pipeline_data_to_checkpoint(tmp_path: Path) -> None:
    _build_tiny_corpus(tmp_path)

    # 1. Data pipeline
    processed_dir = tmp_path / "processed"
    pipeline_cfg = DataPipelineConfig(
        raw_dir=tmp_path / "raw", output_dir=processed_dir, dataset_version="test_v1"
    )
    stats = run_pipeline(pipeline_cfg)
    assert stats.accepted_documents > 0
    assert (processed_dir / "eng.jsonl").exists()
    assert (processed_dir / "hin.jsonl").exists()

    # 2. Tokenizer
    tokenizer_dir = tmp_path / "tokenizer"
    tok_cfg = TokenizerTrainConfig(
        input_dir=processed_dir, output_dir=tokenizer_dir, vocab_size=128, model_type="bpe"
    )
    model_path = train_tokenizer(tok_cfg)
    assert model_path.exists()

    # 3. Packed dataset + DataLoader
    dataset = PackedTokenDataset(
        shards_dir=processed_dir, tokenizer_path=model_path, seq_len=32, total_tokens=2000, seed=0,
    )
    assert len(dataset) > 0
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    inputs, targets = next(iter(loader))
    assert inputs.shape == targets.shape

    # 4. Model + one training step (via the real trainer, a handful of steps)
    model_config = ModelConfig(
        vocab_size=dataset.sp.get_piece_size(), d_model=16, n_layers=1, n_heads=2, max_seq_len=32
    )
    train_config = TrainingConfig(
        output_dir=tmp_path / "run", max_steps=3, micro_batch_size=2, checkpoint_every=0,
        eval_every=0, log_every=100,
    )
    result = train(model_config, train_config, loader)
    assert result.final_step == 3
    assert result.tokens_seen > 0

    # 5. Checkpoint round-trip
    final_ckpt = tmp_path / "run" / "checkpoints" / "final.pt"
    assert final_ckpt.exists()
    from indiclm.models.transformer import DecoderOnlyTransformer

    reloaded = DecoderOnlyTransformer(model_config)
    state = load_checkpoint(final_ckpt, reloaded)
    assert state["step"] == 3


def test_training_resume_continues_from_checkpoint(tmp_path: Path) -> None:
    _build_tiny_corpus(tmp_path)
    processed_dir = tmp_path / "processed"
    run_pipeline(DataPipelineConfig(raw_dir=tmp_path / "raw", output_dir=processed_dir))
    tok_cfg = TokenizerTrainConfig(input_dir=processed_dir, output_dir=tmp_path / "tok", vocab_size=100)
    model_path = train_tokenizer(tok_cfg)

    dataset = PackedTokenDataset(shards_dir=processed_dir, tokenizer_path=model_path, seq_len=16, total_tokens=1000)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    model_config = ModelConfig(vocab_size=dataset.sp.get_piece_size(), d_model=16, n_layers=1, n_heads=2, max_seq_len=16)

    out_dir = tmp_path / "run"
    train_config = TrainingConfig(output_dir=out_dir, max_steps=2, micro_batch_size=2, checkpoint_every=2, eval_every=0, log_every=100)
    train(model_config, train_config, loader)

    resumed_config = TrainingConfig(
        output_dir=out_dir, max_steps=4, micro_batch_size=2,
        resume_from=out_dir / "checkpoints" / "final.pt", checkpoint_every=0, eval_every=0, log_every=100,
    )
    result = train(model_config, resumed_config, loader)
    assert result.final_step == 4
