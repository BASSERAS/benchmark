# Metrics — A1-A15

All evaluation metrics for the TimeGAN Heston benchmark.

## Files

| File | Purpose |
|------|---------|
| `metrics_np.py` | A1-A12, A15 — imports from reference tsg-benchmark/metrics.py |
| `discriminative_score.py` | A13 — GRU + MLP classifiers (PyTorch) |
| `predictive_score.py` | A14 — GRU + MLP predictors, TSTR protocol (PyTorch) |
| `compute_all.py` | Orchestrator: all metrics x 5 seeds |

## Run

```bash
python compute_all.py
```

Outputs `results/seed_{i}_metrics.json`, `results/metrics_summary.csv`, `results/plots/`.
