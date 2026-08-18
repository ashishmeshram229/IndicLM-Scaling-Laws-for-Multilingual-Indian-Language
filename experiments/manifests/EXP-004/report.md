# Ablation: EXP-004

## Hypothesis

Does tokenizer efficiency (tokens/char, compression) predict actual model quality (validation perplexity) when model architecture and token budget are held constant? Compares BPE vs Unigram SentencePiece tokenizers trained on the same corpus (see data/tokenizer_v1 and data/tokenizer_unigram benchmark_report.json for the efficiency numbers being correlated against).

## Results

| Variant | Final val loss | Overall perplexity | Macro-avg perplexity |
|---|---|---|---|
| bpe | 6.606203 | 733.2537 | 733.596 |
| unigram | 6.500119 | 660.7111 | 661.3335 |

Each variant's full report is at `experiments/manifests/EXP-004__<variant>/report.md`.

## Limitations

Single seed per variant, tiny hand-authored corpus (see docs/data_pipeline.md).
Differences below roughly one perplexity point at this scale should be read
as noise, not a robust effect — see docs/reproducibility.md.
