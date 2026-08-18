"""Mixture-aware packed dataset: samples documents per language according
to a mixture engine allocation, tokenizes them, concatenates token streams
per language (each document terminated with EOS), and packs the resulting
stream into fixed-length blocks for causal-LM training.

At the corpus scale this milestone operates at (hundreds of documents),
reaching a meaningful token budget requires sampling with replacement
(effectively multiple epochs over small per-language pools) — this is
made explicit via `PackedDatasetStats.epochs_per_language` rather than
hidden, since silently repeating data changes what a loss curve means.
"""

from __future__ import annotations

import glob
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import sentencepiece as spm
import torch
from torch.utils.data import Dataset

from indiclm.data.mixture import static_mixture, temperature_weights, token_budget_allocation
from indiclm.utils.logging import get_logger

log = get_logger(__name__)


def load_shard_texts(shards_dir: Path) -> dict[str, list[str]]:
    texts: dict[str, list[str]] = {}
    for shard_path in sorted(glob.glob(str(Path(shards_dir) / "*.jsonl"))):
        lang = Path(shard_path).stem
        lang_texts = []
        with open(shard_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                text = json.loads(line).get("text", "")
                if text:
                    lang_texts.append(text)
        if lang_texts:
            texts[lang] = lang_texts
    return texts


@dataclass
class PackedDatasetStats:
    seq_len: int
    total_tokens: int
    num_sequences: int
    tokens_per_language: dict[str, int]
    epochs_per_language: dict[str, float]
    padding_ratio: float

    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class PackedTokenDataset(Dataset):
    """Fixed-length causal-LM dataset built from a language mixture.

    `alpha`: temperature-sampling exponent (see `data.mixture`). alpha=1.0
    samples proportional to each language's available token count;
    alpha=0.0 samples uniformly across languages regardless of size
    (oversampling low-resource languages).
    """

    shards_dir: Path
    tokenizer_path: Path
    seq_len: int
    total_tokens: int
    alpha: float = 1.0
    seed: int = 0
    languages: list[str] | None = None  # None = use all languages present
    manual_weights: dict[str, float] | None = None  # overrides alpha with a static mixture
    stats: PackedDatasetStats = field(init=False)

    def __post_init__(self) -> None:
        self.sp = spm.SentencePieceProcessor(model_file=str(self.tokenizer_path))
        self.eos_id = self.sp.eos_id()
        texts_by_lang = load_shard_texts(self.shards_dir)
        if self.languages:
            texts_by_lang = {k: v for k, v in texts_by_lang.items() if k in self.languages}
        if not texts_by_lang:
            raise ValueError(f"No shard text found under {self.shards_dir}")

        rng = random.Random(self.seed)

        # Available token counts (measured, not assumed) per language.
        available_tokens: dict[str, int] = {}
        tokenized: dict[str, list[list[int]]] = {}
        for lang, docs in texts_by_lang.items():
            ids_list = [self.sp.encode(t, out_type=int) for t in docs]
            tokenized[lang] = ids_list
            available_tokens[lang] = sum(len(ids) for ids in ids_list)

        if self.manual_weights is not None:
            # Static, hand-specified mixture (e.g. "English-heavy": {"eng": 0.6, ...}).
            # Any language present in the corpus but omitted from
            # manual_weights gets zero budget, not a silent default.
            full_weights = {lang: self.manual_weights.get(lang, 0.0) for lang in available_tokens}
            weights = static_mixture(full_weights)
        else:
            weights = temperature_weights(available_tokens, self.alpha)
        budget = token_budget_allocation(weights, self.total_tokens)

        all_ids: list[int] = []
        tokens_per_language: dict[str, int] = {}
        epochs_per_language: dict[str, float] = {}
        for lang, lang_budget in budget.items():
            ids_list = tokenized[lang]
            pool = list(range(len(ids_list)))
            stream: list[int] = []
            while len(stream) < lang_budget:
                rng.shuffle(pool)
                for i in pool:
                    stream.extend(ids_list[i])
                    stream.append(self.eos_id)
                    if len(stream) >= lang_budget:
                        break
            stream = stream[:lang_budget]
            all_ids.extend(stream)
            tokens_per_language[lang] = len(stream)
            epochs_per_language[lang] = round(
                len(stream) / max(available_tokens[lang], 1), 3
            )

        rng.shuffle_seed = self.seed  # type: ignore[attr-defined]
        # Pack into fixed-length blocks (input, target) = (block[:-1], block[1:])
        # via a sliding, non-overlapping window over the concatenated stream.
        n_blocks = max(1, len(all_ids) // (self.seq_len + 1))
        usable = n_blocks * (self.seq_len + 1)
        padding = 0
        if usable == 0:
            # Corpus smaller than one block: pad with eos_id rather than
            # fail outright, and record the padding ratio honestly.
            padding = (self.seq_len + 1) - len(all_ids)
            all_ids = all_ids + [self.eos_id] * padding
            n_blocks = 1
            usable = self.seq_len + 1
        blocks = torch.tensor(all_ids[:usable], dtype=torch.long).view(n_blocks, self.seq_len + 1)
        self.inputs = blocks[:, :-1]
        self.targets = blocks[:, 1:]

        self.stats = PackedDatasetStats(
            seq_len=self.seq_len,
            total_tokens=int(blocks.numel()),
            num_sequences=n_blocks,
            tokens_per_language=tokens_per_language,
            epochs_per_language=epochs_per_language,
            padding_ratio=round(padding / max(usable, 1), 4),
        )
        log.info("packed_dataset_built", **self.stats.to_dict())

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]
