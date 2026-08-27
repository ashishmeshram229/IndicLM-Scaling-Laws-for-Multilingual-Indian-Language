"""Ingestion: read raw text sources into `Document` objects.

The reference ingestion here reads plain-text files (one sentence/paragraph
per line) under a source directory, which is what our small bootstrap
corpus under `data/raw/` uses. Real deployments would add ingestors for
WARC/Common Crawl, Parquet dumps, etc. — those slot in alongside this one
without touching downstream pipeline stages, since everything downstream
only depends on the `Document` schema.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from indiclm.data.normalize import normalize_text
from indiclm.data.schema import Document
from indiclm.utils.logging import get_logger

log = get_logger(__name__)


def ingest_text_directory(root: Path, license_tag: str = "unknown") -> Iterator[Document]:
    """Yield one Document per non-empty line of every .txt file under `root`.

    `source` is set to `<subdirectory>/<filename-without-extension>` so
    provenance (e.g. `wiki_sample/hin`) survives into every later stage.
    """
    root = Path(root)
    for path in sorted(root.rglob("*.txt")):
        source = f"{path.parent.name}/{path.stem}"
        raw = path.read_text(encoding="utf-8")
        n = 0
        for line in raw.splitlines():
            line = normalize_text(line)
            if not line:
                continue
            yield Document(text=line, source=source, license=license_tag)
            n += 1
        log.info("ingested_file", path=str(path), source=source, documents=n)
