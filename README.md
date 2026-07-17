# Time-Series Generation Benchmark

A public benchmark for evaluating **generative models of financial time series**.

Each method is trained on the same target dataset and evaluated with the same 15 metrics
(A1–A15) across 5 random seeds.

---

## Results — Heston (mean ± std, 5 seeds)

Cross-method comparison on 8 192 Heston price paths (seq\_len=128).
↓ = lower is better. ↑ = higher is better. **Bold** = best across methods.

| Metric | SBTS | TimeGAN | Winner |
|--------|:----:|:-------:|:------:|
| A1  Path MMD² ↓      | **0.0110 ± 0.0016** | 0.0180 ± 0.0147 | **SBTS** |
| A2  Terminal MMD² ↓  | **0.0090 ± 0.0035** | 0.0296 ± 0.0235 | **SBTS** |
| A3  Increment MMD² ↓ | **0.0071 ± 0.0005** | 0.0078 ± 0.0037 | **SBTS** |
| A4  Volatility MMD ↓ | **0.3125 ± 0.0176** | 0.3798 ± 0.2351 | **SBTS** |
| A5  Terminal SWD ↓   | 3.465 ± 0.588 | **2.850 ± 1.079** | **TimeGAN** |
| A6  Path SWD ↓       | 2.497 ± 0.288 | **1.501 ± 0.583** | **TimeGAN** |
| A7  Cov Error (%) ↓  | 145.35 ± 4.89 | **17.75 ± 6.71** | **TimeGAN** |
| A8  Mean RMSE ↓      | 1.301 ± 0.278 | **0.739 ± 0.455** | **TimeGAN** |
| A9  Std Error ↓      | 0.249 ± 0.002 | **0.152 ± 0.089** | **TimeGAN** |
| A10 Kurtosis Error ↓ | **0.119 ± 0.006** | 2.955 ± 2.099 | **SBTS** |
| A11 ACF Abs Error ↓  | **0.057 ± 0.001** | 0.134 ± 0.073 | **SBTS** |
| A12 ACF Sq Error ↓   | **0.062 ± 0.001** | 0.092 ± 0.039 | **SBTS** |
| A13 Disc Score GRU ↓ | **0.029 ± 0.028** | 0.050 ± 0.034 | **SBTS** |
| A13 Disc Score MLP ↓ | **0.071 ± 0.008** | 0.151 ± 0.142 | **SBTS** |
| A14 Pred Score GRU ↓ | 0.0091 ± 0.0000 | **0.0087 ± 0.0002** | ≈ tie |
| A14 Pred Score MLP ↓ | 0.0093 ± 0.0006 | **0.0090 ± 0.0005** | ≈ tie |
| A15 Sigma Corr ↑     | 0.0011 ± 0.0035 | **0.0031 ± 0.0101** | ≈ tie |
| A15 Sigma RMSE ↓     | **0.821 ± 0.002** | 0.966 ± 0.124 | **SBTS** |
| PS-MC CRPS H=32 ↓    | **2.761 ± 0.004** | 3.087 ± 0.340 | **SBTS** |
| PS-MC CRPS H=64 ↓    | **3.900 ± 0.008** | 4.372 ± 0.431 | **SBTS** |

**SBTS wins 12/20, TimeGAN wins 6/20, 2 ties.**

Detailed per-seed results and plots:
→ [`results/Heston/SBTS/`](results/Heston/SBTS/) — SBTS metrics, diagnostics, PS-MC
→ [`results/Heston/TimeGAN/`](results/Heston/TimeGAN/) — TimeGAN metrics, diagnostics, PS-MC

---

## Datasets

| Dataset | Paths | Seq len | Description |
|---------|-------|---------|-------------|
| [Heston](dataset/Heston/) | 8 192 | 128 | Heston stochastic volatility model, daily prices (~6 months) |

→ [`dataset/Heston/README.md`](dataset/Heston/README.md) — parameters, SDE formula, reproduce instructions.

---

## Methods

| Method | Full name | Paper | Authors | Year | Venue | Original code |
|--------|-----------|-------|---------|------|-------|---------------|
| [TimeGAN](methods/TimeGAN/) | Time-series GAN | [arXiv:2010.00782](https://arxiv.org/abs/2010.00782) | Yoon, Jarrett, van der Schaar | 2019 | NeurIPS | [jsyoon0823/TimeGAN](https://github.com/jsyoon0823/TimeGAN) |
| [SBTS](methods/SBTS/) | Score-Based Time Series (Schrödinger Bridge) | [arXiv:2503.02943](https://arxiv.org/abs/2503.02943) | Alouadi, Barreau, Carlier, Pham | 2025 | ICAIF | [alexouadi/SBTS](https://github.com/alexouadi/SBTS) |

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
| A15 | Sigma Corr | ✗ (↑) | 1 |
| A15 | Sigma RMSE | ✓ | 0 |

Full formulas and per-seed results:
→ [`results/Heston/SBTS/README.md`](results/Heston/SBTS/README.md)
→ [`results/Heston/TimeGAN/README.md`](results/Heston/TimeGAN/README.md)

---

## Reproducing

```bash
# 1. Generate target dataset
cd dataset/Heston && python generate_heston.py

# 2a. Train TimeGAN (5 seeds, 2 A100 GPUs, ~45 min)
cd methods/TimeGAN/code && python train.py --gpu0 0 --gpu1 3

# 2b. Generate SBTS paths (5 seeds, CPU, 64 workers, ~30 min)
source /home/tbasseras/sbts-venv/bin/activate
cd methods/SBTS/code && SBTS_NWORK=64 python run_all.py

# 3. Compute all metrics (GPU)
CUDA_VISIBLE_DEVICES=0 python metrics/compute_all.py --method TimeGAN --dataset Heston
CUDA_VISIBLE_DEVICES=0 python metrics/compute_all.py --method SBTS    --dataset Heston

# 4. Regenerate diagnostics plot (any method, seed 0)
python metrics/plot_diagnostics.py --method SBTS    --dataset Heston --seed 0
python metrics/plot_diagnostics.py --method TimeGAN --dataset Heston --seed 0
```

---

## Adding a new method

1. Create `methods/<NewMethod>/` with subfolders `generated_paths/`, `weights/`, `losses/`, `code/`
2. Implement generation code — save paths as `generated_paths/seed_{i}/generated_paths_NxT.npy` (price space, S₀≈100)
3. Run `python metrics/compute_all.py --method NewMethod --dataset Heston`
4. Run `python metrics/plot_diagnostics.py --method NewMethod --dataset Heston --seed 0`
5. Results appear in `results/Heston/NewMethod/`
