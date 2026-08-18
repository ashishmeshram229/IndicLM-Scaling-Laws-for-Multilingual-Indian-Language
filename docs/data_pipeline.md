# Data Pipeline

## Status: implemented (bootstrap-scale), documented honestly

## Corpus used in this repository

`data/raw/` contains a **hand-authored bootstrap corpus** — 20 short,
topically simple sentences per language for English, Hindi, Marathi,
Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, and Punjabi
(`wiki_sample/`), 10 Hindi-English / Marathi-English / Tamil-English
code-switched sentences per pair (`codemixed_sample/`), and 10 lines of
deliberately noisy boilerplate/spam text (`junk_sample/`) used to exercise
the quality filter.

**This is not a production Indic corpus.** An attempt was made to source a
small real multilingual sample (Unicode.org's UDHR translations, and a
GitHub mirror of the same dataset) via automated web fetch, but the
available fetch tooling summarizes/paraphrases fetched pages through a
model rather than returning verbatim byte content, which is unusable for
tokenizer/corpus purposes. Given that constraint, the corpus here was
written directly (by the assistant, with the languages that data covers)
to have genuine multi-script Unicode content, real code-switching
patterns, and real boilerplate/spam patterns — enough to validate every
pipeline stage end-to-end — but it is a few hundred documents, not a
crawled corpus, and should not be used to draw linguistic or
model-quality conclusions. Swapping in a real corpus (e.g. AI4Bharat's
IndicCorp/Sangraha, subject to their license terms) means pointing
`indiclm data prepare --raw-dir` at a directory of `.txt` files organized
the same way — no pipeline code needs to change.

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

### Measured result on the bootstrap corpus (dataset_version=v1)

From `data/processed/pipeline_stats.json`:

- 240 total documents ingested, 237 accepted, 3 rejected (2
  `low_langid_confidence`, 1 exact `duplicate`)
- Mean quality score: 0.998 (see "Known limitation" below)
- Duplicate rate: 0.83%, near-duplicate rate: 0%
- Per-language accepted counts: eng=57 (includes code-mixed lines whose
  dominant script is Latin), hin=30, mar=10, ben=20, tam=20, tel=20,
  kan=20, mal=20, guj=20, pan=20

### Known limitation: Hindi/Marathi disambiguation

Language ID for Devanagari-script text uses small hand-curated
function-word lists to distinguish Hindi from Marathi (see
`src/indiclm/data/langid.py`). On this corpus it correctly labels the
first sentence of most Marathi documents (confidence 0.99) but falls back
to a lower-confidence Hindi default (0.6) on sentences that happen not to
contain one of the listed function words — visible in the 30/10 hin/mar
split above rather than the true 20/20. This is a heuristic limitation,
documented rather than hidden; a real deployment would use a trained LID
model (e.g. IndicLID or fastText lid.176).

### Known limitation: quality thresholds are lenient on this corpus

The rule-based quality scorer's default thresholds (`quality.py`) pass
most of the intentionally-noisy `junk_sample/boilerplate.txt` lines
(mean quality score 0.998 across the whole corpus) because repeated-word
spam like "buy buy buy now now now" still has a high alphabetic ratio and
only mildly low unique-word ratio. EXP-008 (data filtering ablation)
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
