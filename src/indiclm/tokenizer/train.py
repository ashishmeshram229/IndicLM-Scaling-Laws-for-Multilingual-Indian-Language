"""Tokenizer training: SentencePiece BPE / Unigram, and a byte-level
fallback, over the sharded, accepted training corpus.

Design note: SentencePiece is used for both BPE and Unigram because it
handles all ten target scripts uniformly without language-specific
pre-tokenization rules (unlike whitespace-based BPE, which assumes
whitespace word boundaries that many of these Indic texts do have, but
which would break for languages that don't). The byte-level backend uses
Hugging Face `tokenizers`' `ByteLevelBPETokenizer` for comparison.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import sentencepiece as spm

from indiclm.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class TokenizerTrainConfig:
    input_dir: Path          # directory of per-language .jsonl shards (data/shard.py output)
    output_dir: Path
    vocab_size: int = 2000
    model_type: str = "bpe"  # "bpe" | "unigram"
    character_coverage: float = 0.9995
    name: str = "indiclm_tokenizer"


def _corpus_text_file(input_dir: Path, scratch_path: Path) -> tuple[Path, int]:
    """Flatten all accepted-document JSONL shards into one plain-text
    corpus file (one document per line), which is what SentencePiece's
    trainer expects. Returns the path and number of lines written."""
    import glob

    n = 0
    with open(scratch_path, "w", encoding="utf-8") as out:
        for shard_path in sorted(glob.glob(str(Path(input_dir) / "*.jsonl"))):
            with open(shard_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    doc = json.loads(line)
                    text = doc.get("text", "").strip()
                    if text:
                        out.write(text + "\n")
                        n += 1
    return scratch_path, n


def train_tokenizer(config: TokenizerTrainConfig) -> Path:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch = output_dir / "_train_corpus.txt"
    _, n_lines = _corpus_text_file(config.input_dir, scratch)
    log.info("tokenizer_training_corpus_built", lines=n_lines)

    if n_lines == 0:
        raise ValueError(f"No training text found under {config.input_dir}")

    # SentencePiece requires vocab_size >= (unique required chars + meta
    # pieces). At small corpus scale, a naive vocab_size cap based only on
    # line count can undershoot this and crash training — so the cap is a
    # ceiling on top of the *measured* character-driven floor, never below it.
    unique_chars = len(set(scratch.read_text(encoding="utf-8")) - {"\n"})
    char_floor = unique_chars + 16  # + meta pieces (pad/unk/bos/eos) with margin
    candidate_vocab_size = min(config.vocab_size, max(64, n_lines * 4))
    effective_vocab_size = max(candidate_vocab_size, char_floor)
    if effective_vocab_size > config.vocab_size:
        log.warning(
            "tokenizer_vocab_size_raised_for_character_floor",
            requested=config.vocab_size, effective=effective_vocab_size, unique_chars=unique_chars,
        )

    model_prefix = str(output_dir / config.name)
    spm.SentencePieceTrainer.train(
        input=str(scratch),
        model_prefix=model_prefix,
        vocab_size=effective_vocab_size,
        model_type=config.model_type,
        character_coverage=config.character_coverage,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        input_sentence_size=0,
        shuffle_input_sentence=True,
    )
    scratch.unlink(missing_ok=True)

    meta = {
        "name": config.name,
        "model_type": config.model_type,
        "vocab_size": effective_vocab_size,
        "requested_vocab_size": config.vocab_size,
        "character_coverage": config.character_coverage,
        "training_lines": n_lines,
        "model_file": f"{config.name}.model",
    }
    (output_dir / f"{config.name}.meta.json").write_text(json.dumps(meta, indent=2))
    log.info("tokenizer_training_complete", model_prefix=model_prefix)
    return Path(f"{model_prefix}.model")
