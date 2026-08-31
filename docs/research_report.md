### EXP-001: Baseline tiny model (smallest size in scaling sweep)

**Status: run.** Final val loss: 7.057156. Overall perplexity: None. See `experiments/manifests/EXP-001/report.md` for the full writeup.

### EXP-002: Baseline mid-size model (scaling sweep)

**Status: run.** Final val loss: 7.011094. Overall perplexity: None. See `experiments/manifests/EXP-002/report.md` for the full writeup.

### EXP-003: Baseline larger model (scaling sweep)

**Status: run.** Final val loss: 6.941626. Overall perplexity: None. See `experiments/manifests/EXP-003/report.md` for the full writeup.

### EXP-004: Tokenizer comparison (BPE vs Unigram)

**Status: run.** Comparison results: {"bpe": {"final_val_loss": 6.88879, "overall_perplexity": 991.8323, "macro_avg_perplexity": 992.3379}, "unigram": {"final_val_loss": 6.363677, "overall_perplexity": 597.4121, "macro_avg_perplexity": 598.7749}} See `experiments/manifests/EXP-004/report.md` for the full writeup.

### EXP-005: English-heavy mixture

**Status: run.** Final val loss: 6.683803. Overall perplexity: 1040.9505. See `experiments/manifests/EXP-005/report.md` for the full writeup.

### EXP-006: Indic-balanced mixture

**Status: run.** Final val loss: 6.882474. Overall perplexity: 991.6731. See `experiments/manifests/EXP-006/report.md` for the full writeup.

### EXP-007: Temperature sampling mixture

**Status: run.** Final val loss: 6.896834. Overall perplexity: 988.3124. See `experiments/manifests/EXP-007/report.md` for the full writeup.

### EXP-008: Data filtering ablation

**Status: run.** Comparison results: {"raw": {"final_val_loss": 6.813816, "overall_perplexity": 903.4774, "macro_avg_perplexity": 905.4335}, "filtered": {"final_val_loss": 6.813816, "overall_perplexity": 903.4774, "macro_avg_perplexity": 905.4335}, "filtered_dedup": {"final_val_loss": 6.814866, "overall_perplexity": 901.9895, "macro_avg_perplexity": 903.9558}} See `experiments/manifests/EXP-008/report.md` for the full writeup.

### EXP-009: Deduplication ablation

**Status: run.** Comparison results: {"filtered": {"final_val_loss": 6.813816, "overall_perplexity": 903.4774, "macro_avg_perplexity": 905.4335}, "filtered_dedup": {"final_val_loss": 6.814866, "overall_perplexity": 901.9895, "macro_avg_perplexity": 903.9558}} See `experiments/manifests/EXP-009/report.md` for the full writeup.

### EXP-010: Low-resource oversampling

**Status: run.** Final val loss: 6.882474. Overall perplexity: 991.6731. See `experiments/manifests/EXP-010/report.md` for the full writeup.

### EXP-011: Code-mixed training

**Status: run.** Final val loss: 6.429962. Overall perplexity: 609.6055. See `experiments/manifests/EXP-011/report.md` for the full writeup.

### EXP-012: Compute-optimal scaling sweep

**Status: run.** Comparison results: {"fit_status": "ok", "A": 7.698561572697784, "alpha": 0.009187551785706724, "alpha_stderr": 0.8499994838668876, "B": 7991354.651867304, "beta": 1.999108873629356, "beta_stderr": 0.14409393392537337, "L_infinity": 1.9147037184713376e-29, "L_infinity_stderr": 641.4242392758439, "r_squared": 0.846802584926019, "n_observations": 8, "observations": [{"run_id": "n_tiny_d8000", "n_params": 63136, "n_params_non_embedding": 24736, "d_tokens": 8192, "final_val_loss": 7.089459, "mean_tokens_per_sec": 20331.1}, {"run_id": "n_small_d8000", "n_params": 108528, "n_params_non_embedding": 50928, "d_tokens": 8192, "final_val_loss": 7.077849, "mean_tokens_per_sec": 19118.76}, {"run_id": "n_medium_d8000", "n_params": 175424, "n_params_non_embedding": 98624, "d_tokens": 8192, "final_val_loss": 7.059865, "mean_tokens_per_sec": 17386.04}, {"run_id": "n_large_d8000", "n_params": 420000, "n_params_non_embedding": 304800, "d_tokens": 8192, "final_val_loss": 7.019166, "mean_tokens_per_sec": 10863.35}, {"run_id": "n_tiny_d24000", "n_params": 63136, "n_params_non_embedding": 24736, "d_tokens": 23872, "final_val_loss": 7.057156, "mean_tokens_per_sec": 27476.34}, {"run_id": "n_small_d24000", "n_params": 108528, "n_params_non_embedding": 50928, "d_tokens": 23872, "final_val_loss": 7.011094, "mean_tokens_per_sec": 23836.43}, {"run_id": "n_medium_d24000", "n_params": 175424, "n_params_non_embedding": 98624, "d_tokens": 23872, "final_val_loss": 6.941626, "mean_tokens_per_sec": 19844.48}, {"run_id": "n_large_d24000", "n_params": 420000, "n_params_non_embedding": 304800, "d_tokens": 23872, "final_val_loss": 6.812897, "mean_tokens_per_sec": 12186.64}]} See `experiments/manifests/EXP-012/report.md` for the full writeup.
