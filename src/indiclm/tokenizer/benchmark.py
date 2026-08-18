"""Per-language tokenizer benchmarking: tokens/char, tokens/word,
compression ratio, unknown-token rate, sequence-length distribution.
"""

from __future__ import annotations

import glob
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import sentencepiece as spm

from indiclm.data.text_utils import split_words


@dataclass
class LanguageTokenizerReport:
    language: str
    documents: int
    total_chars: int
    total_words: int
    total_tokens: int
    tokens_per_char: float
    tokens_per_word: float
    compression_ratio: float  # chars per token; higher = more compression
    unk_rate: float
    mean_sequence_length: float
    p95_sequence_length: float

    def to_dict(self) -> dict:
        return self.__dict__


def benchmark_tokenizer(
    model_path: Path, shards_dir: Path
) -> dict[str, LanguageTokenizerReport]:
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    unk_id = sp.unk_id()

    reports: dict[str, LanguageTokenizerReport] = {}
    for shard_path in sorted(glob.glob(str(Path(shards_dir) / "*.jsonl"))):
        lang = Path(shard_path).stem
        total_chars = total_words = total_tokens = total_unk = 0
        seq_lengths: list[int] = []
        n_docs = 0
        with open(shard_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                text = json.loads(line).get("text", "")
                if not text:
                    continue
                ids = sp.encode(text, out_type=int)
                total_chars += len(text)
                total_words += len(split_words(text))
                total_tokens += len(ids)
                total_unk += sum(1 for i in ids if i == unk_id)
                seq_lengths.append(len(ids))
                n_docs += 1

        if n_docs == 0:
            continue

        seq_lengths.sort()
        p95_idx = max(0, int(len(seq_lengths) * 0.95) - 1)
        reports[lang] = LanguageTokenizerReport(
            language=lang,
            documents=n_docs,
            total_chars=total_chars,
            total_words=total_words,
            total_tokens=total_tokens,
            tokens_per_char=round(total_tokens / max(total_chars, 1), 4),
            tokens_per_word=round(total_tokens / max(total_words, 1), 4),
            compression_ratio=round(total_chars / max(total_tokens, 1), 4),
            unk_rate=round(total_unk / max(total_tokens, 1), 6),
            mean_sequence_length=round(statistics.mean(seq_lengths), 2),
            p95_sequence_length=float(seq_lengths[p95_idx]),
        )
    return reports


def write_report(reports: dict[str, LanguageTokenizerReport], out_path: Path) -> None:
    payload = {lang: r.to_dict() for lang, r in reports.items()}
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
