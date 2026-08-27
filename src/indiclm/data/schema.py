"""Document schema and dataset statistics used throughout the data pipeline.

Design note: every document keeps its full provenance from ingestion through
sharding (document_id, source, language, quality_score, dedup_cluster, ...)
per the "never silently discard data" principle. Rejections are recorded in
`PipelineStats`, not dropped silently.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

PIPELINE_VERSION = "0.1.0"


@dataclass
class Document:
    text: str
    source: str
    language: str = "unknown"
    language_confidence: float = 0.0
    script: str = "unknown"
    is_code_mixed: bool = False
    license: str = "unknown"
    document_id: str = ""
    quality_score: float | None = None
    quality_reasons: list[str] = field(default_factory=list)
    # Exact-dup clusters are keyed by a sequential int index; near-dup
    # clusters are keyed by the document_id of the first-seen match (see
    # MinHashNearDeduplicator) — hence the union rather than int-only.
    dedup_cluster: int | str | None = None
    is_duplicate: bool = False
    is_near_duplicate: bool = False
    contamination_status: str = "not_checked"  # not_checked | clean | flagged
    token_count: int | None = None
    character_count: int = 0
    processing_version: str = PIPELINE_VERSION
    dataset_version: str = "v0"
    ingested_at: float = 0.0
    accepted: bool = True
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        self.character_count = len(self.text)
        if not self.document_id:
            self.document_id = hashlib.sha256(
                f"{self.source}:{self.text}".encode()
            ).hexdigest()[:16]
        if not self.ingested_at:
            self.ingested_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "text": self.text,
            "source": self.source,
            "language": self.language,
            "language_confidence": self.language_confidence,
            "script": self.script,
            "is_code_mixed": self.is_code_mixed,
            "license": self.license,
            "quality_score": self.quality_score,
            "quality_reasons": self.quality_reasons,
            "dedup_cluster": self.dedup_cluster,
            "is_duplicate": self.is_duplicate,
            "is_near_duplicate": self.is_near_duplicate,
            "contamination_status": self.contamination_status,
            "token_count": self.token_count,
            "character_count": self.character_count,
            "processing_version": self.processing_version,
            "dataset_version": self.dataset_version,
            "ingested_at": self.ingested_at,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class PipelineStats:
    """Aggregate, never-silently-discard statistics for a pipeline run."""

    total_documents: int = 0
    accepted_documents: int = 0
    rejected_documents: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    language_distribution: dict[str, int] = field(default_factory=dict)
    source_distribution: dict[str, int] = field(default_factory=dict)
    quality_score_sum: float = 0.0
    quality_score_count: int = 0
    duplicate_count: int = 0
    near_duplicate_count: int = 0

    def record(self, doc: Document) -> None:
        self.total_documents += 1
        self.source_distribution[doc.source] = self.source_distribution.get(doc.source, 0) + 1
        if doc.accepted:
            self.accepted_documents += 1
            self.language_distribution[doc.language] = (
                self.language_distribution.get(doc.language, 0) + 1
            )
            if doc.quality_score is not None:
                self.quality_score_sum += doc.quality_score
                self.quality_score_count += 1
        else:
            self.rejected_documents += 1
            reason = doc.rejection_reason or "unknown"
            self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1
        if doc.is_duplicate:
            self.duplicate_count += 1
        if doc.is_near_duplicate:
            self.near_duplicate_count += 1

    @property
    def mean_quality_score(self) -> float:
        if self.quality_score_count == 0:
            return 0.0
        return self.quality_score_sum / self.quality_score_count

    @property
    def duplicate_rate(self) -> float:
        if self.total_documents == 0:
            return 0.0
        return self.duplicate_count / self.total_documents

    @property
    def near_duplicate_rate(self) -> float:
        if self.total_documents == 0:
            return 0.0
        return self.near_duplicate_count / self.total_documents

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_documents": self.total_documents,
            "accepted_documents": self.accepted_documents,
            "rejected_documents": self.rejected_documents,
            "rejection_reasons": self.rejection_reasons,
            "language_distribution": self.language_distribution,
            "source_distribution": self.source_distribution,
            "mean_quality_score": self.mean_quality_score,
            "duplicate_count": self.duplicate_count,
            "duplicate_rate": self.duplicate_rate,
            "near_duplicate_count": self.near_duplicate_count,
            "near_duplicate_rate": self.near_duplicate_rate,
        }
