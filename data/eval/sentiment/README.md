# Sentiment classification eval set

A small, hand-authored sentiment-classification set (short movie/food/
service reviews, positive vs negative), one file per language, used by
`indiclm evaluate downstream`. This is **not** a public benchmark and
makes no claim to be one — it exists for the same reason
`data/raw/wiki_sample/` does (see `docs/data_pipeline.md`): a small,
license-clean, fully-authored set that lets the pipeline demonstrate a
real methodology end-to-end without depending on an external dataset's
availability or license terms.

Format: one JSON object per line — `{"text": ..., "label": "positive"|"negative"}`.

Coverage: all 10 pipeline languages, 6-8 examples each (3-4 per class).
This is far too small to support any claim about real-world sentiment
accuracy; treat results from it exactly like the rest of this repo's
numbers — see `docs/reproducibility.md`.
