# Time-Series Generation Benchmark

A public benchmark for evaluating **generative models of financial time series**.

Each method is trained on the same target dataset and evaluated with the same 15 metrics
(A1–A15) across 5 random seeds.

---

## Results — Heston (mean ± std, 5 seeds)

Cross-method comparison on 8 192 Heston price paths (seq\_len=128).
↓ = lower is better. ↑ = higher is better. **Bold** = best across methods.
"Perfect Recovery" = two independent halves of the real dataset evaluated against each other.

| Metric | Perfect Recovery | SBTS | TimeGAN | Winner | RW (GBM) |
|--------|:---------------:|:----:|:-------:|:------:|:--------:|
| A1  Path MMD² ↓ | 0.0018 ± 0.0002 | 0.0110 ± 0.0016 | 0.0180 ± 0.0147 | **RW** | **0.0049 ± 0.0018** |
| A2  Terminal MMD² ↓ | 0.0016 ± 0.0002 | 0.0090 ± 0.0035 | 0.0296 ± 0.0235 | **RW** | **0.0053 ± 0.0022** |
| A3  Increment MMD² ↓ | 0.0008 ± 0.0000 | 0.0071 ± 0.0005 | 0.0078 ± 0.0037 | **RW** | **0.0014 ± 0.0000** |
| A4  Volatility MMD ↓ | 0.0082 ± 0.0006 | 0.3125 ± 0.0176 | 0.3798 ± 0.2351 | **RW** | **0.054 ± 0.006** |
| A5  Terminal SWD ↓ | 0.763 ± 0.117 | 3.465 ± 0.588 | 2.850 ± 1.079 | **RW** | **1.865 ± 0.364** |
| A6  Path SWD ↓ | 0.554 ± 0.062 | 2.497 ± 0.288 | 1.501 ± 0.583 | **RW** | **1.163 ± 0.254** |
| A7  Cov Error (%) ↓ | 4.76 ± 2.50 | 145.35 ± 4.89 | 17.75 ± 6.71 | **RW** | **17.47 ± 3.01** |
| A8  Mean RMSE ↓ | 0.140 ± 0.130 | 1.301 ± 0.278 | 0.739 ± 0.455 | **RW** | **0.090 ± 0.059** |
| A9  Std Error ↓ | 0.0048 ± 0.0031 | 0.249 ± 0.002 | 0.152 ± 0.089 | **RW** | **0.049 ± 0.001** |
| A10 Kurtosis Error ↓ | 0.017 ± 0.016 | **0.119 ± 0.006** | 2.955 ± 2.099 | **SBTS** | 0.482 ± 0.008 |
| A11 ACF Abs Error ↓ | 0.0017 ± 0.0006 | 0.060 ± 0.000 | 0.125 ± 0.067 | **RW** | **0.049 ± 0.000** |
| A12 ACF Sq Error ↓ | 0.0014 ± 0.0006 | 0.062 ± 0.001 | 0.084 ± 0.035 | **RW** | **0.044 ± 0.000** |
| A13 Disc Score GRU ↓ | 0.0042 ± 0.0048 | 0.029 ± 0.028 | 0.050 ± 0.034 | **RW** | **0.010 ± 0.008** |
| A13 Disc Score MLP ↓ | 0.0112 ± 0.0079 | 0.071 ± 0.008 | 0.151 ± 0.142 | **RW** | **0.021 ± 0.018** |
| A14 Pred Score GRU ↓ | 0.0085 ± 0.0001 | 0.0091 ± 0.0000 | 0.0087 ± 0.0002 | **RW** | **0.0083 ± 0.0000** |
| A14 Pred Score MLP ↓ | 0.0087 ± 0.0002 | 0.0093 ± 0.0006 | 0.0090 ± 0.0005 | **RW** | **0.0087 ± 0.0001** |
| A15 Sigma Corr ↑ | 0.614 ± 0.002 ⁽¹⁾ | **0.0046 ± 0.0019** | 0.0021 ± 0.0090 | **SBTS** | −0.003 ± 0.004 |
| A15 Sigma RMSE ↓ | 0.065 ± 0.000 ⁽²⁾ | 0.096 ± 0.000 | 0.118 ± 0.018 | **RW** | **0.085 ± 0.000** |
| PS-MC CRPS H=32 ↓ | — | **2.761 ± 0.004** | 3.087 ± 0.340 | **SBTS** | — |
| PS-MC CRPS H=64 ↓ | — | **3.900 ± 0.008** | 4.372 ± 0.431 | **SBTS** | — |
| A16 Tail Survival ↓ | 0.0009 ± 0.0005 | 0.0367 ± 0.0002 | 0.0216 ± 0.0111 | **RW** | **0.0075 ± 0.0003** |
| Training (8 192×128) | — | — (no training) | **~6.5 min / A100** | **SBTS** | <1 s / any CPU |
| Generation (8 192×128) | — | ~6.3 min / 64 CPUs | **<1 s / A100** | **TimeGAN** | <1 s / any CPU |

> ⁽¹⁾ A15 Sigma Corr floor = 0.614 (not 1.0): 5-step rolling QV is a noisy estimator of vₜ even
> for real Heston paths vs their own true variance (ρ≈0.614). Neither SBTS (0.005) nor TimeGAN (0.002)
> preserves stochastic volatility; both score near zero. ≈ tie.
> ⁽²⁾ A15 Sigma RMSE floor = 0.065 (not 0): rolling-QV measurement noise creates an irreducible
> baseline. Both methods score above the floor (SBTS 0.096, TimeGAN 0.118) — correct ordering. SBTS wins.

> ⚠️ **RW baseline:** A calibrated GBM (i.i.d. Gaussian log-returns, matched μ and σ) wins 17/21
> metrics over SBTS and TimeGAN. The 4 metrics where another method beats RW are **A10** (kurtosis —
> fat tails absent in GBM), **A15 Sigma Corr** (no stochastic volatility), and **PS-MC CRPS H=32/64** (path
> shadowing exploits genuine temporal structure). A11/A12 (log-return ACF) reveal that RW also fails
> to match Heston's ARCH autocorrelation (RW = 0.049 vs. perfect = 0.0017, i.e. 29× worse), but since
> SBTS and TimeGAN fail even more (0.060 and 0.125), RW happens to rank best among the three. This
> reveals which metrics actually test Heston-specific features vs. merely matched marginals.

**RW wins 17/21, SBTS wins 4/21, TimeGAN wins 0/21.** (RW = calibrated GBM baseline — see note below.)

Detailed per-seed results and plots:
→ [`results/Heston/SBTS/`](results/Heston/SBTS/) — SBTS metrics, diagnostics, PS-MC
→ [`results/Heston/TimeGAN/`](results/Heston/TimeGAN/) — TimeGAN metrics, diagnostics, PS-MC
→ [`results/Heston/RW/`](results/Heston/RW/) — RW metrics (calibrated GBM baseline)

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
| [SBTS](methods/SBTS/) | Schrödinger Bridge Time Series | [arXiv:2503.02943](https://arxiv.org/abs/2503.02943) | Alouadi, Barreau, Carlier, Pham | 2025 | ICAIF | [alexouadi/SBTS](https://github.com/alexouadi/SBTS) |
| [RW](methods/RW/) | Calibrated Random Walk (GBM) | — | — | — | baseline | — |

---

## Metrics (A1–A16)

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
| A16 | Tail Survival Error | ✓ | 0 |

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
CUDA_VISIBLE_DEVICES=0 python metrics/compute_all.py --method RW       --dataset Heston

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
