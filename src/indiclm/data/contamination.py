"""Contamination detection between an evaluation corpus and the training corpus.

Implements exact and n-gram-overlap matching. Approximate (MinHash) and
embedding-similarity matching are architecturally identical to the dedup
module's `MinHashNearDeduplicator` / `SemanticDeduplicator` — call those
across the train/eval boundary rather than duplicating logic here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from indiclm.data.schema import Document
from indiclm.data.text_utils import split_words


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = split_words(text.lower())
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


@dataclass
class ContaminationReport:
    n: int
    overlap_threshold: float
    total_eval_documents: int
    flagged_documents: int
    flagged_examples: list[str] = field(default_factory=list)

    @property
    def flagged_rate(self) -> float:
        if self.total_eval_documents == 0:
            return 0.0
        return self.flagged_documents / self.total_eval_documents

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "overlap_threshold": self.overlap_threshold,
            "total_eval_documents": self.total_eval_documents,
            "flagged_documents": self.flagged_documents,
            "flagged_rate": round(self.flagged_rate, 4),
            "flagged_examples": self.flagged_examples[:10],
        }


def scan_contamination(
    train_docs: list[Document],
    eval_docs: list[Document],
    n: int = 8,
    overlap_threshold: float = 0.5,
) -> ContaminationReport:
    """Flag eval documents whose n-gram overlap with the training corpus
    exceeds `overlap_threshold`. This is exact + n-gram overlap matching;
    it will not catch paraphrased contamination (that needs the embedding
    path described in `dedup.SemanticDeduplicator`)."""
    train_ngrams: set[tuple[str, ...]] = set()
    for doc in train_docs:
        train_ngrams |= _ngrams(doc.text, n)

    flagged = 0
    examples: list[str] = []
    for doc in eval_docs:
        eval_ngrams = _ngrams(doc.text, n)
        if not eval_ngrams:
            # Too short to n-gram at this n; treat as exact-match check.
            if any(doc.text.strip() == t.text.strip() for t in train_docs):
                flagged += 1
                doc.contamination_status = "flagged"
                examples.append(doc.document_id)
            else:
                doc.contamination_status = "clean"
            continue
        overlap = len(eval_ngrams & train_ngrams) / len(eval_ngrams)
        if overlap >= overlap_threshold:
            flagged += 1
            doc.contamination_status = "flagged"
            examples.append(doc.document_id)
        else:
            doc.contamination_status = "clean"

    return ContaminationReport(
        n=n,
        overlap_threshold=overlap_threshold,
        total_eval_documents=len(eval_docs),
        flagged_documents=flagged,
        flagged_examples=examples,
    )
