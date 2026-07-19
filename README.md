# Time-Series Generation Benchmark

A public benchmark for evaluating **generative models of financial time series**.

Each method is trained on the same target dataset and evaluated with **34 metrics (A1–A34)**
plus **6 curve-shape diagnostics (B)** — each scored by both **MSE** and **% error** — across 5 random seeds.
Every table carries a **Perfect floor** column: the reproducible best-case score from a row-shuffled copy
of the real data (see [`methods/perfect_recovery/`](methods/perfect_recovery/)).

---

## Results — Heston (mean ± std, 5 seeds)

Cross-method comparison on 8 192 Heston price paths (seq\_len=128).
↓ = lower is better. ↑ = higher is better. **Bold** = best across methods.

### A1–A34 — Metrics by category

| Metric | SBTS | TimeGAN | Perfect | Winner |
|--------|:----:|:-------:|:-------:|:------:|
| **— Fat Tail —** | | | | |
| A1 Kurtosis Error ↓ | **0.1187 ± 0.0060** | 2.955 ± 2.099 | 0 | **SBTS** |
| A2 \|r\| q95 Error ↓ | 0.0063 ± 3.00e-05 | **0.0032 ± 0.0018** | 0 | **TimeGAN** |
| A3 \|r\| q99 Error ↓ | 0.0098 ± 4.80e-05 | **0.0043 ± 0.0028** | 0 | **TimeGAN** |
| A4 Tail QQ Error ↓ | 0.0062 ± 2.60e-05 | **0.0034 ± 0.0015** | 0 | **TimeGAN** |
| A5 Hill Tail Index Error ↓ | **9.499 ± 0.3457** | 36.885 ± 17.053 | 0 | **SBTS** |
| **— Distribution —** | | | | |
| A6 Path MMD² ↓ | **0.0110 ± 0.0026** | 0.0165 ± 0.0127 | 0.0015 | **SBTS** |
| A7 Terminal MMD² ↓ | **0.0100 ± 0.0036** | 0.0267 ± 0.0192 | 0.0016 | **SBTS** |
| A8 Increment MMD² ↓ | **0.0071 ± 2.47e-04** | 0.0077 ± 0.0041 | 7.45e-04 | **SBTS** |
| A9 Volatility MMD ↓ | **0.3038 ± 0.0071** | 0.3789 ± 0.2430 | 0.0071 | **SBTS** |
| A10 Terminal SWD ↓ | 3.539 ± 0.7368 | **2.658 ± 0.8567** | 0.6873 | **TimeGAN** |
| A11 Path SWD ↓ | 2.415 ± 0.4104 | **1.417 ± 0.4914** | 0.4381 | **TimeGAN** |
| A12 RV Law Loss ↓ | 2.148 ± 0.0074 | **1.551 ± 0.3788** | 0 | **TimeGAN** |
| A13 Mean Path RMSE ↓ | 0.7499 ± 0.1823 | **0.5289 ± 0.2624** | 0 | **TimeGAN** |
| A14 KS Log-returns ↓ | **0.0534 ± 3.62e-04** | 0.0848 ± 0.0374 | 0 | **SBTS** |
| A15 Skewness Error ↓ | **0.0227 ± 0.0037** | 0.3404 ± 0.3344 | 0 | **SBTS** |
| A16 QQ RMSE (300-pt) ↓ | 0.0028 ± 1.20e-05 | **0.0025 ± 6.43e-04** | 0 | **TimeGAN** |
| A17 Terminal Price KS ↓ | **0.0921 ± 0.0051** | 0.1121 ± 0.0556 | 0 | **SBTS** |
| **— Adversarial —** | | | | |
| A18 Disc Score GRU ↓ | 0.2755 ± 0.2166 | **0.0077 ± 0.0050** | 0.0071 | **TimeGAN** |
| A18 Disc Score MLP ↓ | **0.0079 ± 0.0049** | 0.1031 ± 0.0395 | 0.0033 | **SBTS** |
| **— Predictive —** | | | | |
| A19 Pred Score GRU ↓ | 0.0586 ± 5.90e-05 | **0.0574 ± 0.0019** | 0.0537 | **TimeGAN** |
| A19 Pred Score MLP ↓ | 0.0583 ± 2.55e-04 | **0.0570 ± 0.0012** | 0.0537 | **TimeGAN** |
| **— Temporal —** | | | | |
| A20 Covariance Error ↓ | 145.35 ± 4.886 | **17.751 ± 6.707** | 0 | **TimeGAN** |
| A21 ACF \|r\| Error (lags) ↓ | **0.0596 ± 4.70e-04** | 0.1252 ± 0.0674 | 0 | **SBTS** |
| A22 ACF r² Error (lags) ↓ | **0.0619 ± 5.08e-04** | 0.0839 ± 0.0348 | 0 | **SBTS** |
| A23 ACF \|r\| Lag-1 Error ↓ | **0.1437 ± 0.0012** | 0.2264 ± 0.1034 | 0 | **SBTS** |
| A24 ACF r² Lag-1 Error ↓ | **0.1665 ± 0.0017** | 0.1719 ± 0.0626 | 0 | **SBTS** |
| **— Vol —** | | | | |
| A25 Mean RMSE ↓ | 1.301 ± 0.2776 | **0.7385 ± 0.4552** | 0 | **TimeGAN** |
| A26 Return Std Error ↓ | 0.2492 ± 0.0018 | **0.1519 ± 0.0888** | 0 | **TimeGAN** |
| A27 Log-Return Std Error ↓ | 0.0030 ± 1.20e-05 | **0.0017 ± 7.78e-04** | 0 | **TimeGAN** |
| A28 Kurtosis Ratio (→ 1) | **1.989 ± 0.0182** | −1.095 ± 3.525 | 1.000 | **SBTS** |
| A29 Sigma Mean Error ↓ | 0.0440 ± 1.84e-04 | **0.0307 ± 0.0089** | 0 | **TimeGAN** |
| A30 Cross-Sect. Vol Path RMSE ↓ | 3.276 ± 0.0637 | **0.3534 ± 0.1253** | 0 | **TimeGAN** |
| A31 Rolling Vol KS (w=5) ↓ | 0.3435 ± 6.43e-04 | **0.2540 ± 0.1093** | 0 | **TimeGAN** |
| A32 Vol-of-Vol Error ↓ | 0.0021 ± 6.00e-06 | **8.97e-04 ± 8.69e-04** | 0 | **TimeGAN** |
| **— Heston Spec —** | | | | |
| A33 Teacher-Sigma Corr ↑ | **0.0046 ± 0.0019** | 0.0021 ± 0.0090 | 0.6143 | **SBTS** |
| A34 Teacher-Sigma RMSE ↓ | **0.0955 ± 9.10e-05** | 0.1183 ± 0.0184 | 0.0654 | **SBTS** |
| PS-MC CRPS H=32 ↓ | **2.761 ± 0.004** | 3.087 ± 0.340 | — | **SBTS** |
| PS-MC CRPS H=64 ↓ | **3.900 ± 0.008** | 4.372 ± 0.431 | — | **SBTS** |

**SBTS wins A: 19/38. TimeGAN wins 19/38.**

### B — Curve-shape metrics (6 diagnostic plots)

Each of the 6 diagnostic plots yields a **curve** L (a list of values), not a scalar. For each plot we build three lists — the curve L, its first finite difference (der), and its second finite difference (sec\_der) — then combine the three sub-scores into **one number per plot** under two error measures:

- **MSE row**: for each list, dᵢ = mean((L_real − L_gen)²). Combined mean = sum of the three seed-means; combined std = sqrt(std\_funct² + std\_der² + std\_sec\_der²) (quadrature).
- **% err row**: the **function-level MAPE on the curve L itself**, dᵢ = mean(|L_gen − L_real| / (|L_real| + 1e-6)) × 100 — one division. The derivative / 2nd-derivative MAPE is **excluded** (near-zero true differences make it explode); combined std = sample std across the 5 seeds. Bold marks the lower % error.

↓ lower is better. Histogram bin edges use [0.5th, 99.5th]-percentile of **real data only**, so the reference curve is fixed. **Perfect floor = 0** for every plot (row-shuffle preserves all marginals exactly). Winner is by MSE.

| Plot | Measure | SBTS | TimeGAN | Perfect | Winner |
|------|---------|:----:|:-------:|:------:|:------:|
| **Log-return histogram** | MSE | **12.138 ± 0.1605** | 144.21 ± 120.61 | 0 | **SBTS** |
| | % err | 38.98% ± 0.132% | **33.42% ± 6.512%** | 0 | |
| **QQ plot** | MSE | 8.90e-06 ± 6.77e-08 | **7.09e-06 ± 3.34e-06** | 0 | **TimeGAN** |
| | % err | **21.27% ± 0.364%** | 34.29% ± 11.19% | 0 | |
| **ACF \|r\| lags 1–20** | MSE | **0.0046 ± 3.70e-05** | 0.0105 ± 0.0085 | 0 | **SBTS** |
| | % err | **143% ± 1.580%** | 164% ± 101% | 0 | |
| **ACF r² lags 1–20** | MSE | **0.0052 ± 5.67e-05** | 0.0058 ± 0.0033 | 0 | **SBTS** |
| | % err | 160% ± 1.615% | **110% ± 60.72%** | 0 | |
| **Rolling vol histogram** | MSE | 1227.30 ± 5.109 | **439.33 ± 216.74** | 0 | **TimeGAN** |
| | % err | 84.04% ± 0.124% | **56.06% ± 20.98%** | 0 | |
| **Tail survival** | MSE | **0.0057 ± 6.60e-05** | 0.0117 ± 0.0092 | 0 | **SBTS** |
| | % err | 26.48% ± 0.114% | **23.60% ± 6.040%** | 0 | |

The function-level % err stays in a sane range (≈ 21–164%): the largest values are the ACF plots, where the true ACF ≈ 0.05 sits near zero so any deviation is a big *relative* error. It no longer explodes to 10⁴-% now that the ill-posed derivative MAPE is excluded. TimeGAN's log-return-histogram MSE std (±120.61) is driven by a genuine seed-2 collapse (504.48 vs 11–170 for the other seeds).

**SBTS wins B: 4/6 plots on MSE. TimeGAN wins 2/6** (QQ plot, rolling-vol histogram). Each value is computed over the same **5 seeds** per method.

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
| [SBTS](methods/SBTS/) | Schrödinger Bridge Time Series | [arXiv:2503.02943](https://arxiv.org/abs/2503.02943) | Alouadi, Barreau, Carlier, Pham | 2025 | ICAIF | [alexouadi/SBTS](https://github.com/alexouadi/SBTS) |

---

## Metrics (A1–A34 + B)

### A1–A34 — Core Metrics by category

| ID | Name | Category | Dir | Formula reference |
|----|------|----------|-----|------------------|
| A1 | Kurtosis Error | Fat Tail | ↓ | \|κ_real − κ_gen\| on log-returns |
| A2 | \|r\| q95 Error | Fat Tail | ↓ | \|q_0.95(\|r_real\|) − q_0.95(\|r_gen\|)\| |
| A3 | \|r\| q99 Error | Fat Tail | ↓ | \|q_0.99(\|r_real\|) − q_0.99(\|r_gen\|)\| |
| A4 | Tail QQ Error | Fat Tail | ↓ | QQ RMSE restricted to top-5% tail quantiles |
| A5 | Hill Tail Index Error | Fat Tail | ↓ | \|Hill_real − Hill_gen\|; Hill (1975), top 5% threshold |
| A6 | Path MMD² | Distribution | ↓ | RBF kernel on full paths; Gretton et al. (2012) |
| A7 | Terminal MMD² | Distribution | ↓ | RBF kernel on terminal prices S_T |
| A8 | Increment MMD² | Distribution | ↓ | RBF kernel on log-return increments |
| A9 | Volatility MMD | Distribution | ↓ | RBF kernel on rolling 5-step realized vol |
| A10 | Terminal SWD | Distribution | ↓ | Sliced Wasserstein on S_T; Rabin et al. (2012) |
| A11 | Path SWD | Distribution | ↓ | Sliced Wasserstein on full paths |
| A12 | RV Law Loss | Distribution | ↓ | W₁(RV_real, RV_gen); RV_i=Σ_t r²_{i,t}/dt; Barndorff-Nielsen & Shephard (2002) |
| A13 | Mean Path RMSE | Distribution | ↓ | RMSE between real/gen mean trajectories |
| A14 | KS Log-returns | Distribution | ↓ | KS statistic on pooled log-returns; Massey (1951) |
| A15 | Skewness Error | Distribution | ↓ | \|skew_real − skew_gen\| on log-returns; Cont (2001) |
| A16 | QQ RMSE (300-pt) | Distribution | ↓ | QQ RMSE over 300 uniform quantile levels |
| A17 | Terminal Price KS | Distribution | ↓ | KS statistic on terminal prices S_T |
| A18 | Disc Score GRU / MLP | Adversarial | ↓ | \|accuracy − 0.5\| on log-returns; Esteban et al. (2017) |
| A19 | Pred Score GRU / MLP | Predictive | ↓ | TSTR MAE on log-returns; Esteban et al. (2017) |
| A20 | Covariance Error | Temporal | ↓ | ‖Σ_real − Σ_gen‖_F / ‖Σ_real‖_F × 100% |
| A21 | ACF \|r\| Error (lags) | Temporal | ↓ | Mean \|ACF_real(k) − ACF_gen(k)\| over lags 1–20 on \|r\| |
| A22 | ACF r² Error (lags) | Temporal | ↓ | Mean \|ACF_real(k) − ACF_gen(k)\| over lags 1–20 on r² |
| A23 | ACF \|r\| Lag-1 Error | Temporal | ↓ | \|ACF_real(1) − ACF_gen(1)\| on \|r\|; Heston ≈ +0.052 |
| A24 | ACF r² Lag-1 Error | Temporal | ↓ | \|ACF_real(1) − ACF_gen(1)\| on r²; Heston ≈ +0.050 |
| A25 | Mean RMSE | Vol | ↓ | RMSE of per-step mean price E[S_t] |
| A26 | Return Std Error | Vol | ↓ | \|std(r_real) − std(r_gen)\| on price increments ΔS_t |
| A27 | Log-Return Std Error | Vol | ↓ | \|std(r_real) − std(r_gen)\| on log-returns |
| A28 | Kurtosis Ratio (→ 1) | Vol | — | κ_real / κ_gen; perfect = 1.0 |
| A29 | Sigma Mean Error | Vol | ↓ | \|mean(σ_real) − mean(σ_gen)\| annualized per-path vol |
| A30 | Cross-Sect. Vol Path RMSE | Vol | ↓ | RMSE of cross-sectional vol trajectory |
| A31 | Rolling Vol KS (w=5) | Vol | ↓ | KS on rolling-5 vol histograms; Mandelbrot (1963) |
| A32 | Vol-of-Vol Error | Vol | ↓ | \|std(rolling-vol_real) − std(rolling-vol_gen)\| |
| A33 | Teacher-Sigma Corr | Heston Spec | ↑ | Pearson ρ of QV-estimated vol vs true teacher v_t |
| A34 | Teacher-Sigma RMSE | Heston Spec | ↓ | RMSE of QV-estimated vol vs true teacher v_t |

### B — Curve-shape metrics (6 diagnostic plots)

For each of 6 diagnostic plots we build three lists — the curve L, its first finite difference (der), and its second finite difference (sec\_der) — and score each list under **two measures**: MSE (absolute squared error) and % err (relative error). For MSE the three sub-scores are summed (std in quadrature); the **% err reports the function-level MAPE of the curve L only** (the derivative MAPE is ill-posed near zero). Histogram bin edges use [0.5th, 99.5th]-percentile of **real data only**, making the reference curve fixed across seeds.

| Plot | Key | What the curve represents |
|------|-----|--------------------------|
| Log-return histogram | `B_log_ret_hist_*` | Density of log-returns r=log(S_{t+1}/S_t) |
| QQ plot | `B_qq_plot_*` | Quantile function at 100 uniform levels |
| ACF \|r\| (lags 1–20) | `B_acf_abs_r_*` | Mean per-path ACF of \|r\| at each lag |
| ACF r² (lags 1–20) | `B_acf_sq_r_*` | Mean per-path ACF of r² at each lag |
| Rolling vol hist. | `B_roll_vol_hist_*` | Density of rolling-5 vol |
| Tail survival | `B_tail_surv_*` | P(\|r\|>x) at thresholds of real \|r\| |

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
source /path/to/sbts-venv/bin/activate
cd methods/SBTS/code && SBTS_NWORK=64 python run_all.py

# 3. Compute all metrics (GPU for A13/A14)
CUDA_VISIBLE_DEVICES=0 python metrics/compute_all.py --method TimeGAN --dataset Heston
CUDA_VISIBLE_DEVICES=0 python metrics/compute_all.py --method SBTS    --dataset Heston

# 4. Compute perfect-recovery floor
CUDA_VISIBLE_DEVICES=0 python metrics/perfect_recovery.py --dataset Heston
```

See [`GUIDELINE.md`](GUIDELINE.md) for the full reproducibility protocol.

---

## Adding a new method

1. Create `methods/<NewMethod>/` with subfolders `generated_paths/`, `weights/`, `losses/`, `code/`
2. Implement generation code — save paths as `generated_paths/seed_{i}/generated_paths_NxT.npy` (price space, S₀≈100)
3. Run `python metrics/compute_all.py --method NewMethod --dataset Heston`
4. Results appear in `results/Heston/NewMethod/` with full A1–A34 + B tables
