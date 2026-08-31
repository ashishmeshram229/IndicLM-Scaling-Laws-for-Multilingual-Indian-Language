# Evaluation

## What's implemented

`src/indiclm/evaluation/perplexity.py` loads a checkpoint independently
of the training loop and computes overall loss/perplexity plus
per-language loss/perplexity (macro-average = unweighted mean across
languages; weighted-average = weighted by tokens evaluated). Every
`indiclm experiment run` writes `evaluation.json` with this breakdown.

`src/indiclm/data/contamination.py` implements exact and n-gram-overlap
matching between a training and an evaluation corpus, producing a
`ContaminationReport` with a flagged-document rate. This has not yet been
wired into the experiment runner as an automatic pre-evaluation gate
(planned; currently invoked directly — see the module's tests).

`src/indiclm/evaluation/downstream.py` adds one downstream task: zero-shot
sentiment classification, scored via length-normalized log-likelihood of
each candidate label as a continuation (the standard technique for
evaluating a causal LM with no classification head — see the module's
docstring for why label words are fixed English strings across all
languages rather than translated). The eval set
(`data/eval/sentiment/`) is a small, hand-authored, license-clean set —
6-8 examples per language — built for the same reason
`data/raw/wiki_sample/` is: no dependency on an external dataset's
availability or license terms. Run it with:

```bash
indiclm evaluate-downstream --checkpoint <path/to/final.pt> \
  --tokenizer-path <path/to/tokenizer.model> \
  --out-path experiments/manifests/<exp_id>/downstream_evaluation.json
```

`indiclm report dashboard` compares every experiment that has a
`downstream_evaluation.json` in one table.

**Result actually observed, across every mixture/tokenizer/data-quality
variant run so far (EXP-004 both tokenizers, EXP-005/006/007/010/011,
EXP-008 all three variants), on the real Wikipedia-sourced corpus (see
`docs/data_pipeline.md`):** every configuration scores within a few
points of chance (0.5) — from 0.4844 (`EXP-004__bpe`) to 0.5625
(`EXP-008__raw`/`EXP-008__filtered`) — and, unlike on the earlier
hand-authored placeholder corpus, no longer always predicts the same
label regardless of input (see the per-example `predictions` array in
any `downstream_evaluation.json` to verify). That's a real change from
the previous fully-degenerate result, but the spread is still small and
mostly explained by data mechanics rather than mixture/tokenizer policy:
`EXP-008__raw` and `EXP-008__filtered` score identically (0.5625) because
the quality filter accepted essentially all real Wikipedia text on this
corpus (mean quality score ~1.0 — see `docs/data_pipeline.md`) and so
removed nothing, leaving the two runs' training data numerically
identical; only `EXP-008__filtered_dedup` (0.5) differs, from the small
number of near-duplicate paragraphs dedup actually removed. None of this
should be read as a mixture or tokenizer choice actually working. At
this project's model scale
(55K-400K parameters, 8K-24K training tokens per run), there still isn't
enough signal for *any* mixture or tokenizer choice to reliably
differentiate — the finding remains that scale, not mixture policy, is
the binding constraint at this size. Re-running this eval after scaling
up model size and/or training-token budget (see `docs/reproducibility.md`)
is the natural next experiment; a mixture/tokenizer effect that is
invisible below some scale threshold would itself be a meaningful
result.

## Not implemented in this milestone

- Downstream tasks beyond sentiment classification (QA, translation,
  summarization, reasoning) against public Indic benchmarks — no
  benchmark datasets are bundled or downloaded in this environment.
- A dedicated code-mixed *evaluation* benchmark (EXP-011 trains on
  code-mixed data and reports its own held-out perplexity, which is not
  the same as a labeled code-mixed downstream task).
- Model-based quality scoring (the quality module's `HybridQualityScorer`
  interface exists; no model is bundled).

These gaps are listed here rather than silently absent from the CLI.

## Sample result

From EXP-011 (code-mixed training), `experiments/manifests/EXP-011/evaluation.json`:
final validation loss 6.374, overall perplexity 566.7, on a held-out
split of the hand-authored Hindi/Marathi/Tamil-English code-mixed
corpus (`data/raw/codemixed_sample/`). This should be read as "the
pipeline correctly trains and evaluates on code-mixed data," not as a
benchmark result — see `docs/data_pipeline.md` for corpus caveats.
