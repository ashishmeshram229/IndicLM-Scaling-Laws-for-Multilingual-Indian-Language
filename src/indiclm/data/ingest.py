"""Ingestion: read raw text sources into `Document` objects.

Two ingestors are provided:
  - `ingest_text_directory` — plain .txt files (bootstrap / dev corpus)
  - `ingest_wikipedia_parquet` — HuggingFace Wikipedia parquet dumps
    (production corpus; reads the parquet files downloaded by
    scripts/download_corpus.py, streams per-language with an optional cap)
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from indiclm.data.normalize import normalize_text
from indiclm.data.schema import Document
from indiclm.utils.logging import get_logger

log = get_logger(__name__)


def ingest_text_directory(
    root: Path,
    license_tag: str = "unknown",
    license_by_source: dict[str, str] | None = None,
) -> Iterator[Document]:
    """Yield one Document per non-empty line of every .txt file under `root`.

    `source` is set to `<subdirectory>/<filename-without-extension>` so
    provenance (e.g. `wiki_sample/hin`) survives into every later stage.

    A single global `license_tag` is wrong once raw_dir mixes sources
    under genuinely different licenses (e.g. real Wikipedia excerpts
    alongside hand-authored synthetic examples) -- `license_by_source`
    overrides it per immediate subdirectory name (e.g. `"wiki_sample"`),
    falling back to `license_tag` for any subdirectory not listed.
    """
    root = Path(root)
    license_by_source = license_by_source or {}
    for path in sorted(root.rglob("*.txt")):
        subdir = path.parent.name
        source = f"{subdir}/{path.stem}"
        doc_license = license_by_source.get(subdir, license_tag)
        raw = path.read_text(encoding="utf-8")
        n = 0
        for line in raw.splitlines():
            line = normalize_text(line)
            if not line:
                continue
            yield Document(text=line, source=source, license=doc_license)
            n += 1
        log.info("ingested_file", path=str(path), source=source, license=doc_license, documents=n)


WIKI_LICENSE = "CC BY-SA 3.0 / GFDL (wikimedia/wikipedia dump 20231101)"

# Per-language document caps to keep memory bounded in pipeline.py.
# English Wikipedia has 6.4M articles — loading all would require ~60GB RAM.
# Other languages are small enough to process in full.
_WIKI_DEFAULT_CAPS: dict[str, int] = {
    "en": 500_000,
}


def ingest_wikipedia_parquet(
    raw_wiki_dir: Path,
    language: str,
    max_docs: int | None = None,
    min_text_length: int = 100,
) -> Iterator[Document]:
    """Yield Documents from a per-language Wikipedia parquet file.

    Expects the layout written by scripts/download_corpus.py:
        raw_wiki_dir/<lang>/wiki.parquet

    The parquet file has columns: id, url, title, text.
    Each Wikipedia article becomes one Document (not split by paragraph);
    downstream quality / dedup stages handle further cleaning.

    max_docs defaults to _WIKI_DEFAULT_CAPS[language] when set, otherwise
    all documents are yielded. English is capped at 500K by default to
    avoid OOM on the CPU pipeline node (110GB RAM limit).
    """
    import pyarrow.parquet as pq

    parquet_path = Path(raw_wiki_dir) / language / "wiki.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    cap = max_docs if max_docs is not None else _WIKI_DEFAULT_CAPS.get(language)
    source = f"wikipedia/{language}"
    n = 0

    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(columns=["id", "text"], batch_size=10_000):
        for row in zip(batch["id"].to_pylist(), batch["text"].to_pylist()):
            wiki_id, text = row
            text = text.strip() if text else ""
            if len(text) < min_text_length:
                continue
            doc = Document(
                text=text,
                source=source,
                license=WIKI_LICENSE,
                document_id=f"wiki_{language}_{wiki_id}",
            )
            yield doc
            n += 1
            if cap is not None and n >= cap:
                log.info(
                    "wiki_cap_reached",
                    language=language,
                    cap=cap,
                    note="increase max_docs to ingest more",
                )
                return

    log.info("ingested_wikipedia_parquet", language=language, documents=n, path=str(parquet_path))
