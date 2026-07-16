# Time-Series Generation Benchmark

A public benchmark for evaluating **generative models of financial time series**.

Each method is trained on the same target dataset and evaluated with the same 15 metrics
(A1–A15) across 5 random seeds.

---

## Results

> Cross-method comparison — columns = methods, rows = metrics.
> Values are **mean ± std** across 5 seeds.

*More methods coming. Full table will appear here once ≥ 2 methods are benchmarked.*

Detailed per-seed results, plots, and classifier loss curves:
→ [`results/Heston/TimeGAN/`](results/Heston/TimeGAN/)

---

## Datasets

| Dataset | Paths | Seq len | Description |
|---------|-------|---------|-------------|
| [Heston](dataset/Heston/) | 8 192 | 128 | Heston stochastic volatility model, daily prices (~6 months) |

→ [`dataset/Heston/README.md`](dataset/Heston/README.md) — parameters, SDE formula, reproduce instructions.

---

## Methods

| Method | Paper | Authors | Year | Venue | Original code |
|--------|-------|---------|------|-------|---------------|
| [TimeGAN](methods/TimeGAN/) | [Time-series GAN](https://arxiv.org/abs/2010.00782) | Yoon, Jarrett, van der Schaar | 2019 | NeurIPS | [jsyoon0823/TimeGAN](https://github.com/jsyoon0823/TimeGAN) |

---

## Metrics (A1–A15)

| ID | Name | Lower = better | Perfect score |
|----|------|---------------|---------------|
| A1 | Path MMD² | ✓ | 0 |
| A2 | Terminal MMD² | ✓ | 0 |
| A3 | Increment MMD² | ✓ | 0 |
| A4 | Volatility MMD | ✓ | 0 |
| A5 | Terminal SWD | ✓ | 0 |
| A6 | Path SWD | ✓ | 0 |
| A7 | Cov Error (%) | ✓ | 0 |
| A8 | Mean RMSE | ✓ | 0 |
| A9 | Std Error | ✓ | 0 |
| A10 | Kurtosis Error | ✓ | 0 |
| A11 | ACF Abs Error | ✓ | 0 |
| A12 | ACF Sq Error | ✓ | 0 |
| A13 | Disc Score (GRU) | ✓ | 0 |
| A13 | Disc Score (MLP) | ✓ | 0 |
| A14 | Pred Score (GRU) | ✓ | baseline MAE |
| A14 | Pred Score (MLP) | ✓ | baseline MAE |
| A15 | Sigma Corr | ✗ | 1 |
| A15 | Sigma RMSE | ✓ | 0 |

Full formulas and per-seed results: [`results/Heston/TimeGAN/README.md`](results/Heston/TimeGAN/README.md)

---

## Reproducing

```bash
# 1. Generate target dataset
cd dataset/Heston && python generate_heston.py

# 2. Train TimeGAN (5 seeds, 2 A100 GPUs)
cd methods/TimeGAN/code && python train.py --gpu0 0 --gpu1 3

# 3. Compute all 15 metrics
cd metrics && python compute_all.py --method TimeGAN --dataset Heston
```

---

## Adding a new method

1. Create `methods/<NewMethod>/` with subfolders `generated_paths/`, `weights/`, `losses/`, `code/`
2. Implement `code/train.py` — accepts `--seed`, `--n-samples`, `--gpu0`, `--gpu1`
3. Save generated paths as `generated_paths/seed_{i}/generated_paths_NxT.npy`
4. Run `python metrics/compute_all.py --method NewMethod --dataset Heston`
5. Results appear in `results/Heston/NewMethod/`
