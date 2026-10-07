"""End-to-end data pipeline orchestration.

Two source formats are supported:
  - "txt"                 — plain .txt files under raw_dir (bootstrap/dev)
  - "wikipedia-parquet"   — HuggingFace Wikipedia parquet dumps produced by
                            scripts/download_corpus.py; set raw_wiki_dir and
                            languages in DataPipelineConfig.

Pipeline stages:
  ingest → normalize → langid → quality filter → dedup → mixture-aware
  acceptance → shard → stats report.

Memory strategy: for wikipedia-parquet format, the pipeline runs once per
language so peak RAM is bounded by the largest single-language corpus
(English, capped at 500K docs by default ≈ 4–8 GB). Pass max_docs_per_language
to override the cap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from indiclm.data.dedup import ExactDeduplicator, MinHashNearDeduplicator
from indiclm.data.ingest import ingest_text_directory, ingest_wikipedia_parquet
from indiclm.data.langid import RuleBasedLanguageIdentifier
from indiclm.data.quality import RuleBasedQualityScorer
from indiclm.data.schema import Document, PipelineStats
from indiclm.data.shard import write_shards
from indiclm.utils.logging import get_logger

log = get_logger(__name__)

WIKIPEDIA_LICENSE = (
    "CC BY-SA 3.0 / GFDL (wikimedia/wikipedia dump 20231101); "
    "see https://dumps.wikimedia.org/legal.html"
)


@dataclass
class DataPipelineConfig:
    raw_dir: Path
    output_dir: Path
    dataset_version: str = "v1"
    min_quality_score: float = 0.5
    min_langid_confidence: float = 0.3
    near_dedup_threshold: float = 0.8
    license_tag: str = "hand-authored-sample; see docs/data_pipeline.md"
    license_by_source: dict[str, str] = field(
        default_factory=lambda: {
            "wiki_sample": (
                "CC BY-SA 3.0 / GFDL (Wikipedia, via wikimedia/wikipedia "
                "dump 20231101); see data/raw/wiki_sample/SOURCE.md"
            ),
        }
    )
    enable_quality_filter: bool = True
    enable_exact_dedup: bool = True
    enable_near_dedup: bool = True
    # Wikipedia parquet mode
    source_format: str = "txt"  # "txt" | "wikipedia-parquet"
    raw_wiki_dir: Path | None = None
    languages: list[str] = field(
        default_factory=lambda: ["hi", "mr", "bn", "ta", "te", "kn", "ml", "gu", "pa", "en"]
    )
    max_docs_per_language: int | None = None  # None = use per-language defaults


def _process_docs(docs: list[Document], config: DataPipelineConfig) -> tuple[PipelineStats, list[Document]]:
    """Run filter → dedup → stats on an already-ingested doc list."""
    langid = RuleBasedLanguageIdentifier()
    quality_scorer = RuleBasedQualityScorer()
    stats = PipelineStats()

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

    if config.enable_exact_dedup:
        docs = ExactDeduplicator().process(docs)
    if config.enable_near_dedup:
        docs = MinHashNearDeduplicator(threshold=config.near_dedup_threshold).process(docs)

    for doc in docs:
        if doc.accepted and (doc.is_duplicate or doc.is_near_duplicate):
            doc.accepted = False
            doc.rejection_reason = "duplicate" if doc.is_duplicate else "near_duplicate"
        stats.record(doc)

    return stats, docs


def run_pipeline(config: DataPipelineConfig) -> PipelineStats:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.source_format == "wikipedia-parquet":
        return _run_wikipedia_pipeline(config)
    return _run_txt_pipeline(config)


def _run_txt_pipeline(config: DataPipelineConfig) -> PipelineStats:
    """Original txt-file pipeline (bootstrap / dev corpus)."""
    langid = RuleBasedLanguageIdentifier()
    quality_scorer = RuleBasedQualityScorer()
    stats = PipelineStats()

    docs: list[Document] = list(
        ingest_text_directory(
            config.raw_dir,
            license_tag=config.license_tag,
            license_by_source=config.license_by_source,
        )
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

    _write_stats(stats, config.output_dir)
    return stats


def _run_wikipedia_pipeline(config: DataPipelineConfig) -> PipelineStats:
    """Per-language Wikipedia pipeline — memory-bounded, one language at a time."""
    assert config.raw_wiki_dir is not None, "raw_wiki_dir required for wikipedia-parquet format"
    output_dir = Path(config.output_dir)
    aggregate_stats = PipelineStats()
    all_accepted_docs: list[Document] = []

    for lang in config.languages:
        log.info("wikipedia_pipeline_start", language=lang)
        try:
            docs = list(
                ingest_wikipedia_parquet(
                    config.raw_wiki_dir,
                    language=lang,
                    max_docs=config.max_docs_per_language,
                )
            )
        except FileNotFoundError as e:
            log.warning("wikipedia_parquet_missing", language=lang, error=str(e))
            continue

        log.info("ingestion_complete", language=lang, documents=len(docs))
        stats, docs = _process_docs(docs, config)

        # Merge per-language stats into aggregate
        aggregate_stats.total_documents += stats.total_documents
        aggregate_stats.accepted_documents += stats.accepted_documents
        aggregate_stats.rejected_documents += stats.rejected_documents
        for k, v in stats.rejection_reasons.items():
            aggregate_stats.rejection_reasons[k] = aggregate_stats.rejection_reasons.get(k, 0) + v
        for k, v in stats.language_distribution.items():
            aggregate_stats.language_distribution[k] = (
                aggregate_stats.language_distribution.get(k, 0) + v
            )
        aggregate_stats.duplicate_count += stats.duplicate_count
        aggregate_stats.near_duplicate_count += stats.near_duplicate_count
        aggregate_stats.quality_score_sum += stats.quality_score_sum
        aggregate_stats.quality_score_count += stats.quality_score_count

        accepted = [d for d in docs if d.accepted]
        all_accepted_docs.extend(accepted)
        log.info(
            "wikipedia_pipeline_complete",
            language=lang,
            accepted=len(accepted),
            rejected=stats.rejected_documents,
        )

    checksums = write_shards(all_accepted_docs, output_dir, config.dataset_version)
    log.info("sharding_complete", languages=list(checksums.keys()))

    _write_stats(aggregate_stats, output_dir)
    return aggregate_stats


def _write_stats(stats: PipelineStats, output_dir: Path) -> None:
    stats_path = Path(output_dir) / "pipeline_stats.json"
    stats_path.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")
    log.info(
        "pipeline_complete",
        total=stats.total_documents,
        accepted=stats.accepted_documents,
        rejected=stats.rejected_documents,
        duplicate_rate=stats.duplicate_rate,
        near_duplicate_rate=stats.near_duplicate_rate,
    )
