"""End-to-end data pipeline orchestration:

Raw -> ingest -> normalize -> langid -> quality filter -> dedup ->
mixture-aware acceptance -> shard -> stats report.

Contamination detection and tokenization/packing are separate stages
(`contamination.py`, `indiclm.tokenizer`) invoked by the evaluation and
training pipelines respectively, not folded into this function, so each
stage can be run, tested, and reasoned about independently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from indiclm.data.dedup import ExactDeduplicator, MinHashNearDeduplicator
from indiclm.data.ingest import ingest_text_directory
from indiclm.data.langid import RuleBasedLanguageIdentifier
from indiclm.data.quality import RuleBasedQualityScorer
from indiclm.data.schema import Document, PipelineStats
from indiclm.data.shard import write_shards
from indiclm.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class DataPipelineConfig:
    raw_dir: Path
    output_dir: Path
    dataset_version: str = "v1"
    min_quality_score: float = 0.5
    min_langid_confidence: float = 0.3
    near_dedup_threshold: float = 0.8
    license_tag: str = "hand-authored-sample; see docs/data_pipeline.md"
    enable_quality_filter: bool = True
    enable_exact_dedup: bool = True
    enable_near_dedup: bool = True


def run_pipeline(config: DataPipelineConfig) -> PipelineStats:
    langid = RuleBasedLanguageIdentifier()
    quality_scorer = RuleBasedQualityScorer()
    stats = PipelineStats()

    docs: list[Document] = list(
        ingest_text_directory(config.raw_dir, license_tag=config.license_tag)
    )
    log.info("ingestion_complete", documents=len(docs))

    for doc in docs:
        result = langid.identify(doc.text)
        doc.language = result.language
        doc.language_confidence = result.confidence
        doc.script = result.script
        doc.is_code_mixed = result.is_code_mixed

        if result.language == "unknown" or result.confidence < config.min_langid_confidence:
            doc.accepted = False
            doc.rejection_reason = "low_langid_confidence"
            continue

        score, reasons = quality_scorer.score(doc)
        doc.quality_score = score
        doc.quality_reasons = reasons
        if config.enable_quality_filter and score < config.min_quality_score:
            doc.accepted = False
            doc.rejection_reason = "low_quality_score"

    # Deduplication runs over all documents (including quality-rejected
    # ones) so duplicate statistics reflect the raw corpus, but only
    # accepted, non-duplicate documents are shipped to shards. Each stage
    # is independently toggleable so EXP-008/EXP-009 can isolate the
    # effect of filtering vs. deduplication on downstream loss.
    if config.enable_exact_dedup:
        docs = ExactDeduplicator().process(docs)
    if config.enable_near_dedup:
        docs = MinHashNearDeduplicator(threshold=config.near_dedup_threshold).process(docs)

    for doc in docs:
        if doc.accepted and (doc.is_duplicate or doc.is_near_duplicate):
            doc.accepted = False
            doc.rejection_reason = "duplicate" if doc.is_duplicate else "near_duplicate"
        stats.record(doc)

    checksums = write_shards(docs, config.output_dir, config.dataset_version)
    log.info("sharding_complete", languages=list(checksums.keys()))

    stats_path = Path(config.output_dir) / "pipeline_stats.json"
    stats_path.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")

    accepted_path = Path(config.output_dir) / "accepted_documents.json"
    accepted_path.write_text(
        json.dumps([d.to_dict() for d in docs if d.accepted], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    log.info(
        "pipeline_complete",
        total=stats.total_documents,
        accepted=stats.accepted_documents,
        rejected=stats.rejected_documents,
        duplicate_rate=stats.duplicate_rate,
        near_duplicate_rate=stats.near_duplicate_rate,
    )
    return stats
