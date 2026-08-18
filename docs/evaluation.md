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

## Not implemented in this milestone

- Downstream task evaluation (QA, classification, translation,
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
