# Data Pipeline

## Status: implemented (bootstrap-scale), documented honestly

## Corpus used in this repository

`data/raw/wiki_sample/` contains **real Wikipedia excerpts** — up to 5
paragraphs from each of ~40 articles per language edition, for English,
Hindi, Marathi, Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, and
Punjabi, fetched via the `wikimedia/wikipedia` dataset on Hugging Face
(dump `20231101`). `data/raw/codemixed_sample/` (10 Hindi-English /
Marathi-English / Tamil-English code-switched sentences per pair) and
`data/raw/junk_sample/` (10 lines of deliberately noisy boilerplate/spam
text) remain hand-authored on purpose — see below.

**How this changed from the original bootstrap corpus.** The first
version of this repository used a fully hand-authored placeholder corpus
here, because an earlier attempt to fetch a small real multilingual
sample (Unicode.org's UDHR translations) via general-purpose web-fetch
tooling ran into that tooling summarizing/paraphrasing pages through a
model rather than returning verbatim content — unusable for tokenizer
training. That was solved by going around page-scraping entirely:
`scripts/fetch_real_corpus.py` reads the Wikipedia dataset's Parquet
shards directly via DuckDB's `httpfs` extension, using HTTP range
requests to pull only the row groups it needs (a few hundred KB per
language) rather than each language's full 80–140MB shard. See
`data/raw/wiki_sample/SOURCE.md` for the exact method and the CC BY-SA
3.0 / GFDL license this content carries — `data.pipeline`'s
`LICENSE_BY_SOURCE` map tags every `Document` ingested from `wiki_sample/`
with that license, distinct from the `hand-authored-sample` tag on
`codemixed_sample/` and `junk_sample/`.

**`codemixed_sample/` and `junk_sample/` are staying hand-authored.**
Natural code-switching and spam/boilerplate patterns aren't abundant in
encyclopedic Wikipedia prose, and the pipeline needs adversarial input to
exercise its langid/quality-filter code paths (EXP-008's filtering
ablation, in particular, depends on `junk_sample/` existing) — swapping
those two for real data would mean sourcing them from somewhere else
entirely (social media, forums), which is a separate, lower-priority
task from getting real prose into the corpus.

**This is still a small corpus.** ~1,800 real paragraphs across 10
languages is enough to validate every pipeline stage end-to-end and
produce non-degenerate tokenizer/model behavior, but it is nowhere near
production pretraining scale. Swapping in a much larger real corpus
(e.g. AI4Bharat's IndicCorp/Sangraha, subject to their license terms, or
more Wikipedia shards) means pointing `indiclm data prepare --raw-dir` at
a directory of `.txt` files organized the same way, or raising
`ARTICLES_PER_LANG` / iterating more shards in `fetch_real_corpus.py` —
no pipeline code needs to change either way.

## Pipeline stages (all implemented)

```
raw .txt files -> ingest -> normalize (NFC, whitespace) -> language ID
  -> quality scoring -> exact + near dedup -> shard (per-language JSONL)
```

Run: `indiclm data prepare --raw-dir data/raw --output-dir data/processed`

Each stage is independently toggleable (`--no-enable-quality-filter`,
`--no-enable-exact-dedup`, `--no-enable-near-dedup`) — this is what
EXP-008/EXP-009 (data-quality and dedup ablations) use to isolate each
stage's effect.

### Measured result on the real corpus (dataset_version=v1)

From `data/processed/pipeline_stats.json`, after the Wikipedia swap:

- 1,830 total documents ingested, 1,827 accepted, 3 rejected (2
  `low_langid_confidence`, 1 exact `duplicate`)
- Mean quality score: ~1.0 (see "Known limitation" below — real
  Wikipedia prose scores near-perfectly on this rule-based scorer, which
  is expected, not a bug)
- Duplicate rate: 0.11%, near-duplicate rate: 0%
- Per-language accepted counts: eng=253 (includes code-mixed lines whose
  dominant script is Latin), hin=217, mar=138, ben=186, tam=153, tel=187,
  kan=178, mal=172, guj=170, pan=173

### Known limitation: Hindi/Marathi disambiguation

Language ID for Devanagari-script text uses small hand-curated
function-word lists to distinguish Hindi from Marathi (see
`src/indiclm/data/langid.py`). On this corpus it correctly labels most
Marathi paragraphs but is biased toward a lower-confidence Hindi default
on paragraphs that happen not to contain one of the listed function
words — visible in the 217/138 hin/mar split above against `hin.txt`
and `mar.txt` starting from a roughly even 184/177 fetched paragraphs
(`wc -l data/raw/wiki_sample/{hin,mar}.txt`). This is a
heuristic limitation, documented rather than hidden; a real deployment
would use a trained LID model (e.g. IndicLID or fastText lid.176).

### Known limitation: quality thresholds are lenient on this corpus

The rule-based quality scorer's default thresholds (`quality.py`) pass
essentially all real Wikipedia paragraphs (expected — they're
well-formed prose) but also pass most of the intentionally-noisy
`junk_sample/boilerplate.txt` lines, because repeated-word spam like "buy
buy buy now now now" still has a high alphabetic ratio and only mildly
low unique-word ratio. EXP-008 (data filtering ablation)
measured this directly: `data/processed_raw` (no filtering) and
`data/processed_filtered` (default thresholds) both accepted 238
documents — the filter did not reject the junk in this corpus. See
`experiments/manifests/EXP-008/report.md` for the full comparison; this
is reported as a negative result, not hidden.

## Provenance

Every `Document` (see `src/indiclm/data/schema.py`) carries
`document_id`, `source`, `language`, `language_confidence`, `script`,
`is_code_mixed`, `license`, `quality_score`, `dedup_cluster`,
`contamination_status`, `token_count`, `processing_version`, and
`dataset_version` through every stage — nothing is silently dropped;
rejections are recorded with a `rejection_reason` and aggregated in
`PipelineStats`.

## Format choice: JSONL, not Parquet

At this corpus's scale (hundreds of documents, tens of KB per shard),
JSONL's simplicity outweighs Parquet's columnar I/O benefits. This is a
measured tradeoff, not an assumption — Parquet/Arrow is the documented
upgrade path (Section 28 of the original spec) once corpus size makes
row-oriented JSONL a demonstrated throughput bottleneck.
