#!/usr/bin/env python3
"""Fetch a small, real, license-clean text sample per language from
Wikipedia (via the `wikimedia/wikipedia` dataset on Hugging Face,
dump 20231101) and write it into `data/raw/wiki_sample/<lang>.txt` in
the same one-paragraph-per-line format `indiclm.data.ingest` expects.

This REPLACES the earlier hand-authored placeholder text in
`wiki_sample/` with real Wikipedia excerpts. `codemixed_sample/` and
`junk_sample/` are left as deliberately hand-authored synthetic
examples (code-mixing and boilerplate/spam patterns aren't naturally
abundant in Wikipedia prose, and the pipeline needs *some* adversarial
input to exercise its langid/quality-filter code paths at all) -- see
`docs/data_pipeline.md` for why that split is intentional, not an
oversight.

Uses DuckDB's httpfs extension to read only as many Parquet row groups
as needed via HTTP range requests, rather than downloading each
80-140MB language shard in full.

License: Wikipedia text is CC BY-SA 3.0 / GFDL. Attribution is
recorded in `data/raw/wiki_sample/SOURCE.md` and threaded through to
`Document.license` via `data.pipeline`'s per-directory license map
(see `LICENSE_BY_SOURCE` there) -- not the single global tag used for
`codemixed_sample/` and `junk_sample/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

DUMP = "20231101"
BASE_URL = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main"

# indiclm language tag -> Wikipedia language-edition code
LANGUAGES = {
    "eng": "en",
    "hin": "hi",
    "mar": "mr",
    "ben": "bn",
    "tam": "ta",
    "tel": "te",
    "kan": "kn",
    "mal": "ml",
    "guj": "gu",
    "pan": "pa",
}

# Shard lists per language. Larger languages (en) have multiple shards;
# we cap at the first 5 English shards to stay below ~100MB download.
# Single-shard languages are fetched in full.
SHARDS_BY_LANG: dict[str, list[str]] = {
    "en": [f"20231101.en/train-{i:05d}-of-00041.parquet" for i in range(5)],
    "hi": [
        "20231101.hi/train-00000-of-00002.parquet",
        "20231101.hi/train-00001-of-00002.parquet",
    ],
    "mr": ["20231101.mr/train-00000-of-00001.parquet"],
    "bn": [
        "20231101.bn/train-00000-of-00002.parquet",
        "20231101.bn/train-00001-of-00002.parquet",
    ],
    "ta": [
        "20231101.ta/train-00000-of-00002.parquet",
        "20231101.ta/train-00001-of-00002.parquet",
    ],
    "te": [
        "20231101.te/train-00000-of-00002.parquet",
        "20231101.te/train-00001-of-00002.parquet",
    ],
    "kn": ["20231101.kn/train-00000-of-00001.parquet"],
    "ml": ["20231101.ml/train-00000-of-00001.parquet"],
    "gu": ["20231101.gu/train-00000-of-00001.parquet"],
    "pa": ["20231101.pa/train-00000-of-00001.parquet"],
}

ARTICLES_PER_LANG = 500
MIN_PARAGRAPH_CHARS = 80
MAX_PARAGRAPHS_PER_ARTICLE = 8
MIN_ARTICLE_CHARS = 500


def fetch_language(con: duckdb.DuckDBPyConnection, lang_tag: str, wiki_code: str) -> list[str]:
    shards = SHARDS_BY_LANG[wiki_code]
    all_lines: list[str] = []
    articles_fetched = 0
    for shard in shards:
        remaining = ARTICLES_PER_LANG - articles_fetched
        if remaining <= 0:
            break
        url = f"{BASE_URL}/{shard}"
        query = f"""
            SELECT title, text FROM read_parquet('{url}')
            WHERE length(text) >= {MIN_ARTICLE_CHARS}
            LIMIT {remaining}
        """
        try:
            rows = con.execute(query).fetchall()
        except Exception as e:
            print(f"  WARNING: failed to fetch {shard}: {e}", file=sys.stderr)
            continue
        for _title, text in rows:
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
            kept = [p for p in paragraphs if len(p) >= MIN_PARAGRAPH_CHARS][:MAX_PARAGRAPHS_PER_ARTICLE]
            all_lines.extend(kept)
            articles_fetched += 1
    return all_lines


def main() -> None:
    out_dir = Path("data/raw/wiki_sample")
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")

    total_lines = 0
    for lang_tag, wiki_code in LANGUAGES.items():
        print(f"fetching {lang_tag} ({wiki_code})...", file=sys.stderr)
        lines = fetch_language(con, lang_tag, wiki_code)
        if not lines:
            print(f"  WARNING: 0 paragraphs kept for {lang_tag}", file=sys.stderr)
            continue
        out_path = out_dir / f"{lang_tag}.txt"
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  wrote {len(lines)} paragraphs -> {out_path}", file=sys.stderr)
        total_lines += len(lines)

    print(f"done: {total_lines} paragraphs across {len(LANGUAGES)} languages", file=sys.stderr)


if __name__ == "__main__":
    main()
