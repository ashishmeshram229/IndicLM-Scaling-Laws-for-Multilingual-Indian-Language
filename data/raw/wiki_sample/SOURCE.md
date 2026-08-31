# Source and license

The `.txt` files in this directory are real excerpts from Wikipedia,
fetched via the [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia)
dataset on Hugging Face (dump `20231101`), one language edition per
file. See `scripts/fetch_real_corpus.py` for the exact fetch logic —
it is fully reproducible (deterministic `LIMIT`-based sampling, no
random seed needed since the query itself is fixed).

For each of the 10 project languages, the script reads the first
Parquet shard of that language's Wikipedia dump via HTTP range
requests (no full-file download), keeps articles with at least 500
characters of body text, and takes up to 5 paragraphs of at least 80
characters from each, until ~40 articles have been sampled.

**License:** Wikipedia article text is dual-licensed under
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and the
[GFDL](https://www.gnu.org/licenses/fdl-1.3.html). Both require
attribution and share-alike redistribution of derivative works. This
directory's content is a direct excerpt (not a derivative transformation)
of Wikipedia article text; the `wikimedia/wikipedia` dataset card
carries the same license as Wikipedia proper. `data.pipeline`'s
`LICENSE_BY_SOURCE` map tags every `Document` ingested from this
directory with this license explicitly, distinct from the
`hand-authored-sample` tag used for `codemixed_sample/` and
`junk_sample/` (see the module-level docstring in
`scripts/fetch_real_corpus.py` for why those two stay synthetic).

This is still a small, bootstrap-scale sample (~1,800 paragraphs
total across 10 languages) relative to real pretraining corpora —
see `docs/data_pipeline.md` and the root `README.md` for what that
does and doesn't change about the trustworthiness of results in this
repository.
