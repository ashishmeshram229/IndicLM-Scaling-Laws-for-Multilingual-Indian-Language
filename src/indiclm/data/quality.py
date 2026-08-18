"""Modular quality-scoring pipeline: rule_based | model_based | hybrid.

Only `rule_based` is implemented in this milestone (no model-based scorer
is trained yet — adding one later means implementing `QualityScorer` and
wiring it into `HybridQualityScorer`, not touching this module's callers).
Thresholds are configurable per language/source rather than a single
universal cutoff, per the project's data-quality principles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from indiclm.data.schema import Document
from indiclm.data.text_utils import split_words

_URL_RE = re.compile(r"https?://\S+")
# NOTE: deliberately not `\w` — that excludes Unicode combining marks
# (Mn/Mc), which would undercount "alphabetic" characters in every Indic
# script that uses dependent vowel signs. We instead count anything that
# is not whitespace and not punctuation.
_PUNCT_RE = re.compile(r"[.,!?;:\"'(){}\[\]—–\-।॥]")


def _is_alpha_like(ch: str) -> bool:
    return not ch.isspace() and _PUNCT_RE.fullmatch(ch) is None


@dataclass
class QualitySignals:
    length_chars: int
    alphabetic_ratio: float
    punctuation_ratio: float
    repeated_char_ratio: float
    url_density: float
    unique_word_ratio: float


def compute_signals(text: str) -> QualitySignals:
    n = max(len(text), 1)
    words = split_words(text)
    n_words = max(len(words), 1)

    alpha = sum(1 for ch in text if _is_alpha_like(ch))
    punct = len(_PUNCT_RE.findall(text))
    urls = len(_URL_RE.findall(text))

    # Repeated-character / boilerplate signal: longest run of an identical
    # character, normalized by length (catches "!!!!!!!" / "......" spam).
    longest_run = 1
    run = 1
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            run += 1
            longest_run = max(longest_run, run)
        else:
            run = 1

    unique_words = len(set(w.lower() for w in words))

    return QualitySignals(
        length_chars=len(text),
        alphabetic_ratio=alpha / n,
        punctuation_ratio=punct / n,
        repeated_char_ratio=longest_run / n,
        url_density=urls / n_words,
        unique_word_ratio=unique_words / n_words,
    )


@dataclass
class QualityThresholds:
    min_length_chars: int = 8
    max_length_chars: int = 20_000
    min_alphabetic_ratio: float = 0.4
    max_punctuation_ratio: float = 0.5
    max_repeated_char_ratio: float = 0.3
    max_url_density: float = 0.3
    min_unique_word_ratio: float = 0.4


# Per-source overrides: junk_sample is intentionally noisy for pipeline
# testing, but we still score it with the *same* honest thresholds — the
# point is to demonstrate the filter catches it, not to special-case it.
DEFAULT_THRESHOLDS_BY_LANGUAGE: dict[str, QualityThresholds] = {}


class QualityScorer(Protocol):
    def score(self, doc: Document) -> tuple[float, list[str]]: ...


@dataclass
class RuleBasedQualityScorer:
    thresholds_by_language: dict[str, QualityThresholds] = field(
        default_factory=lambda: dict(DEFAULT_THRESHOLDS_BY_LANGUAGE)
    )
    default_thresholds: QualityThresholds = field(default_factory=QualityThresholds)

    def _thresholds_for(self, language: str) -> QualityThresholds:
        return self.thresholds_by_language.get(language, self.default_thresholds)

    def score(self, doc: Document) -> tuple[float, list[str]]:
        t = self._thresholds_for(doc.language)
        s = compute_signals(doc.text)
        reasons: list[str] = []
        passed = 0
        total = 6

        if t.min_length_chars <= s.length_chars <= t.max_length_chars:
            passed += 1
        else:
            reasons.append("length_out_of_range")

        if s.alphabetic_ratio >= t.min_alphabetic_ratio:
            passed += 1
        else:
            reasons.append("low_alphabetic_ratio")

        if s.punctuation_ratio <= t.max_punctuation_ratio:
            passed += 1
        else:
            reasons.append("high_punctuation_ratio")

        if s.repeated_char_ratio <= t.max_repeated_char_ratio:
            passed += 1
        else:
            reasons.append("high_repeated_char_ratio")

        if s.url_density <= t.max_url_density:
            passed += 1
        else:
            reasons.append("high_url_density")

        if s.unique_word_ratio >= t.min_unique_word_ratio:
            passed += 1
        else:
            reasons.append("low_unique_word_ratio")

        score = passed / total
        return round(score, 3), reasons


@dataclass
class HybridQualityScorer:
    """Combines rule-based signals with an (optional, not-yet-implemented)
    model-based score. Falls back to pure rule-based scoring today so the
    interface is stable when a model scorer is added later."""

    rule_scorer: RuleBasedQualityScorer = field(default_factory=RuleBasedQualityScorer)
    model_scorer: QualityScorer | None = None
    model_weight: float = 0.5

    def score(self, doc: Document) -> tuple[float, list[str]]:
        rule_score, reasons = self.rule_scorer.score(doc)
        if self.model_scorer is None:
            return rule_score, reasons
        model_score, model_reasons = self.model_scorer.score(doc)
        combined = (1 - self.model_weight) * rule_score + self.model_weight * model_score
        return round(combined, 3), reasons + model_reasons
