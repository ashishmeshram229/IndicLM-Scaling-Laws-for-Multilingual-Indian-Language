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
EXP-008 all three variants):** every single configuration scores exactly
at chance (0.5) on this task, always predicting the same label
regardless of the input text (see the per-example `predictions` array in
any `downstream_evaluation.json` to verify — this is a real, reproducible
finding, not a placeholder). At this project's model scale (55K-400K
parameters, 8K-24K training tokens per run), there is no measurable
zero-shot task signal for *any* mixture or tokenizer choice to
differentiate — the finding is that scale, not mixture policy, is the
binding constraint at this size. Re-running this eval after scaling up
model size and/or real training data (see `docs/reproducibility.md`) is
the natural next experiment; a mixture/tokenizer effect that is invisible
below some scale threshold would itself be a meaningful result.

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
