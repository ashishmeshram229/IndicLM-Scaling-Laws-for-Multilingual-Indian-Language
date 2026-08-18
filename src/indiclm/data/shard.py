"""Sharding: write accepted documents to newline-delimited JSON shards
grouped by language, plus a dataset-level manifest with content hashing
for dataset versioning.

JSONL is used (rather than Parquet/Arrow) at this corpus scale — a few
hundred documents — because the format's simplicity and human-readability
outweigh any I/O benefit Parquet would offer here. `docs/data_pipeline.md`
documents this as a measured tradeoff, not an assumption: Parquet is the
documented upgrade path once corpus size makes row-oriented JSONL a
throughput bottleneck (see Section 28 of the project spec).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from indiclm.data.schema import Document


def write_shards(docs: list[Document], out_dir: Path, dataset_version: str) -> dict[str, str]:
    """Write one JSONL shard per language under `out_dir`. Returns a
    mapping of language -> sha256 of that shard's contents, for dataset
    version manifests."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_language: dict[str, list[Document]] = {}
    for doc in docs:
        if not doc.accepted:
            continue
        doc.dataset_version = dataset_version
        by_language.setdefault(doc.language, []).append(doc)

    checksums: dict[str, str] = {}
    for lang, lang_docs in by_language.items():
        path = out_dir / f"{lang}.jsonl"
        lines = [json.dumps(d.to_dict(), ensure_ascii=False) for d in lang_docs]
        content = "\n".join(lines) + "\n"
        path.write_text(content, encoding="utf-8")
        checksums[lang] = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    manifest = {
        "dataset_version": dataset_version,
        "languages": {lang: len(lang_docs) for lang, lang_docs in by_language.items()},
        "shard_checksums": checksums,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return checksums


def read_shard(path: Path) -> list[dict]:
    path = Path(path)
    docs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            docs.append(json.loads(line))
    return docs
