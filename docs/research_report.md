### EXP-001: Baseline tiny model (smallest size in scaling sweep)

**Status: run.** Final val loss: 6.792248. Overall perplexity: None. See `experiments/manifests/EXP-001/report.md` for the full writeup.

### EXP-002: Baseline mid-size model (scaling sweep)

**Status: run.** Final val loss: 6.739266. Overall perplexity: None. See `experiments/manifests/EXP-002/report.md` for the full writeup.

### EXP-003: Baseline larger model (scaling sweep)

**Status: run.** Final val loss: 6.661185. Overall perplexity: None. See `experiments/manifests/EXP-003/report.md` for the full writeup.

### EXP-004: Tokenizer comparison (BPE vs Unigram)

**Status: run.** Comparison results: {"bpe": {"final_val_loss": 6.606203, "overall_perplexity": 733.2537, "macro_avg_perplexity": 733.596}, "unigram": {"final_val_loss": 6.500119, "overall_perplexity": 660.7111, "macro_avg_perplexity": 661.3335}} See `experiments/manifests/EXP-004/report.md` for the full writeup.

### EXP-005: English-heavy mixture

**Status: run.** Final val loss: 6.548765. Overall perplexity: 789.1673. See `experiments/manifests/EXP-005/report.md` for the full writeup.

### EXP-006: Indic-balanced mixture

**Status: run.** Final val loss: 6.593474. Overall perplexity: 726.0022. See `experiments/manifests/EXP-006/report.md` for the full writeup.

### EXP-007: Temperature sampling mixture

**Status: run.** Final val loss: 6.616812. Overall perplexity: 733.1068. See `experiments/manifests/EXP-007/report.md` for the full writeup.

### EXP-008: Data filtering ablation

**Status: run.** Comparison results: {"raw": {"final_val_loss": 6.609044, "overall_perplexity": 734.025, "macro_avg_perplexity": 734.2832}, "filtered": {"final_val_loss": 6.609044, "overall_perplexity": 734.025, "macro_avg_perplexity": 734.2832}, "filtered_dedup": {"final_val_loss": 6.606203, "overall_perplexity": 733.2537, "macro_avg_perplexity": 733.596}} See `experiments/manifests/EXP-008/report.md` for the full writeup.

### EXP-009: Deduplication ablation

**Status: run.** Comparison results: {"filtered": {"final_val_loss": 6.609044, "overall_perplexity": 734.025, "macro_avg_perplexity": 734.2832}, "filtered_dedup": {"final_val_loss": 6.606203, "overall_perplexity": 733.2537, "macro_avg_perplexity": 733.596}} See `experiments/manifests/EXP-009/report.md` for the full writeup.

### EXP-010: Low-resource oversampling

**Status: run.** Final val loss: 6.593474. Overall perplexity: 726.0022. See `experiments/manifests/EXP-010/report.md` for the full writeup.

### EXP-011: Code-mixed training

**Status: run.** Final val loss: 6.37383. Overall perplexity: 566.7175. See `experiments/manifests/EXP-011/report.md` for the full writeup.

### EXP-012: Compute-optimal scaling sweep

**Status: run.** Comparison results: {"fit_status": "ok", "A": 7.5089395322213885, "alpha": 0.010579537413075461, "alpha_stderr": 0.7528256356846404, "B": 8966067.693519078, "beta": 1.9973105795374708, "beta_stderr": 0.1217759731591103, "L_infinity": 5.0187219578566894e-29, "L_infinity_stderr": 473.6241596804496, "r_squared": 0.8786937482097386, "n_observations": 8, "observations": [{"run_id": "n_tiny_d8000", "n_params": 55072, "n_params_non_embedding": 24736, "d_tokens": 8192, "final_val_loss": 6.836364, "mean_tokens_per_sec": 16710.86}, {"run_id": "n_small_d8000", "n_params": 96432, "n_params_non_embedding": 50928, "d_tokens": 8192, "final_val_loss": 6.826179, "mean_tokens_per_sec": 16645.67}, {"run_id": "n_medium_d8000", "n_params": 159296, "n_params_non_embedding": 98624, "d_tokens": 8192, "final_val_loss": 6.796077, "mean_tokens_per_sec": 14725.54}, {"run_id": "n_large_d8000", "n_params": 395808, "n_params_non_embedding": 304800, "d_tokens": 8192, "final_val_loss": 6.749945, "mean_tokens_per_sec": 8726.12}, {"run_id": "n_tiny_d24000", "n_params": 55072, "n_params_non_embedding": 24736, "d_tokens": 23872, "final_val_loss": 6.792248, "mean_tokens_per_sec": 22420.66}, {"run_id": "n_small_d24000", "n_params": 96432, "n_params_non_embedding": 50928, "d_tokens": 23872, "final_val_loss": 6.739266, "mean_tokens_per_sec": 18456.2}, {"run_id": "n_medium_d24000", "n_params": 159296, "n_params_non_embedding": 98624, "d_tokens": 23872, "final_val_loss": 6.661185, "mean_tokens_per_sec": 16194.1}, {"run_id": "n_large_d24000", "n_params": 395808, "n_params_non_embedding": 304800, "d_tokens": 23872, "final_val_loss": 6.532998, "mean_tokens_per_sec": 9579.54}]} See `experiments/manifests/EXP-012/report.md` for the full writeup.
