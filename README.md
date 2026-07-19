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
| A10 Kurtosis Error ↓ | **0.1187 ± 0.0060** | 2.9545 ± 2.0988 | 0.0000 | **SBTS** |
| A17 \|r\| q95 Error ↓ | 0.0063 ± 0.0000 | **0.0032 ± 0.0018** | 0.0000 | **TimeGAN** |
| A18 \|r\| q99 Error ↓ | 0.0098 ± 0.0000 | **0.0043 ± 0.0028** | 0.0000 | **TimeGAN** |
| A30 Tail QQ Error ↓ | 0.0062 ± 0.0000 | **0.0034 ± 0.0015** | 0.0000 | **TimeGAN** |
| A34 Hill Tail Index Error ↓ | **9.499 ± 0.346** | 36.88 ± 17.05 | 0.0000 | **SBTS** |
| **— Distribution —** | | | | |
| A1  Path MMD² ↓ | **0.0112 ± 0.0011** | 0.0181 ± 0.0147 | 0.0015 | **SBTS** |
| A2  Terminal MMD² ↓ | **0.0102 ± 0.0014** | 0.0308 ± 0.0229 | 0.0016 | **SBTS** |
| A3  Increment MMD² ↓ | **0.0069 ± 0.0005** | 0.0077 ± 0.0039 | 0.0007 | **SBTS** |
| A4  Volatility MMD ↓ | **0.2964 ± 0.0126** | 0.3933 ± 0.2553 | 0.0071 | **SBTS** |
| A5  Terminal SWD ↓ | 3.7097 ± 0.3209 | **3.1284 ± 0.9227** | 0.687 | **TimeGAN** |
| A6  Path SWD ↓ | 2.5335 ± 0.2212 | **1.6343 ± 0.5763** | 0.438 | **TimeGAN** |
| A24 RV Law Loss ↓ | 2.1482 ± 0.0074 | **1.5512 ± 0.3788** | 0.0000 | **TimeGAN** |
| A25 Mean Path RMSE ↓ | 0.7499 ± 0.1823 | **0.5289 ± 0.2624** | 0.0000 | **TimeGAN** |
| A27 KS on Log-returns ↓ | **0.0534 ± 0.0004** | 0.0848 ± 0.0374 | 0.0000 | **SBTS** |
| A28 Skewness Error ↓ | **0.0227 ± 0.0037** | 0.3404 ± 0.3344 | 0.0000 | **SBTS** |
| A29 QQ RMSE (300-pt) ↓ | 0.0028 ± 0.0000 | **0.0025 ± 0.0006** | 0.0000 | **TimeGAN** |
| A33 Terminal Price KS ↓ | **0.0921 ± 0.0051** | 0.1121 ± 0.0556 | 0.0000 | **SBTS** |
| **— Adversarial —** | | | | |
| A13 Disc Score GRU ↓ | 0.2740 ± 0.2208 | **0.0099 ± 0.0084** | 0.008 | **TimeGAN** |
| A13 Disc Score MLP ↓ | **0.0063 ± 0.0038** | 0.0921 ± 0.0463 | 0.010 | **SBTS** |
| **— Predictive —** | | | | |
| A14 Pred Score GRU ↓ | 0.0586 ± 0.0000 | **0.0570 ± 0.0013** | 0.054 | **TimeGAN** |
| A14 Pred Score MLP ↓ | 0.0582 ± 0.0002 | **0.0573 ± 0.0015** | 0.054 | **TimeGAN** |
| **— Temporal —** | | | | |
| A7  Cov Error (%) ↓ | 145.35 ± 4.89 | **17.75 ± 6.71** | 0.00 | **TimeGAN** |
| A11 ACF \|r\| Error ↓ | **0.0596 ± 0.0005** | 0.1252 ± 0.0674 | 0.0000 | **SBTS** |
| A12 ACF r² Error ↓ | **0.0619 ± 0.0005** | 0.0839 ± 0.0348 | 0.0000 | **SBTS** |
| A22 ACF \|r\| Lag-1 Error ↓ | **0.1437 ± 0.0012** | 0.2264 ± 0.1034 | 0.0000 | **SBTS** |
| A23 ACF r² Lag-1 Error ↓ | **0.1665 ± 0.0017** | 0.1719 ± 0.0626 | 0.0000 | **SBTS** |
| **— Vol —** | | | | |
| A8  Mean RMSE ↓ | 1.3013 ± 0.2776 | **0.7385 ± 0.4552** | 0.000 | **TimeGAN** |
| A9  Return Std Error ↓ | 0.2492 ± 0.0018 | **0.1519 ± 0.0888** | 0.0000 | **TimeGAN** |
| A16 Log-Return Std Error ↓ | 0.0030 ± 0.0000 | **0.0017 ± 0.0008** | 0.0000 | **TimeGAN** |
| A19 Kurtosis Ratio (→ 1) | **1.9890 ± 0.0182** | −1.095 ± 3.525 | 1.0000 | **SBTS** |
| A20 Sigma Mean Error ↓ | 0.0440 ± 0.0002 | **0.0307 ± 0.0089** | 0.0000 | **TimeGAN** |
| A26 Cross-Sect. Vol Path RMSE ↓ | 3.2760 ± 0.0637 | **0.3534 ± 0.1253** | 0.0000 | **TimeGAN** |
| A31 Rolling Vol KS ↓ | 0.3435 ± 0.0006 | **0.2540 ± 0.1093** | 0.0000 | **TimeGAN** |
| A32 Vol-of-Vol Error ↓ | 0.0021 ± 0.0000 | **0.0009 ± 0.0009** | 0.0000 | **TimeGAN** |
| **— Heston Spec —** | | | | |
| A15 Sigma Corr ↑ | **0.0046 ± 0.0019** | 0.0021 ± 0.0090 | 0.614 | **SBTS** |
| A15 Sigma RMSE ↓ | **0.0955 ± 0.0001** | 0.1183 ± 0.0184 | 0.065 | **SBTS** |
| A21 Oracle Sigma Corr ↑ | **0.0046 ± 0.0019** | 0.0021 ± 0.0090 | 0.614 | **SBTS** |
| PS-MC CRPS H=32 ↓ | **2.761 ± 0.004** | 3.087 ± 0.340 | — | **SBTS** |
| PS-MC CRPS H=64 ↓ | **3.900 ± 0.008** | 4.372 ± 0.431 | — | **SBTS** |

**SBTS wins A: 20/39. TimeGAN wins 19/39.**

### B — Curve-shape metrics (6 diagnostic plots)

Each of the 6 diagnostic plots yields a **curve** L (a list of values), not a scalar. For each plot we build three lists — the curve L, its first finite difference (der), and its second finite difference (sec\_der) — then combine the three sub-scores into **one number per plot** under two error measures:

- **MSE row**: for each list, dᵢ = mean((L_real − L_gen)²). Combined mean = sum of the three seed-means; combined std = sqrt(std\_funct² + std\_der² + std\_sec\_der²) (quadrature).
- **% err row**: for each list, dᵢ = mean(|L_gen − L_real| / (|L_real| + 1e-6)) × 100. Combined the same way.

↓ lower is better. Histogram bin edges use [0.5th, 99.5th]-percentile of **real data only**, so the reference curve is fixed. **Perfect floor = 0** for every plot (row-shuffle preserves all marginals exactly). Winner is by MSE.

| Plot | Measure | SBTS | TimeGAN | Perfect | Winner |
|------|---------|:----:|:-------:|:------:|:------:|
| **Log-return histogram** | MSE   | **12.14 ± 0.16** | 144.2 ± 120.6 | 0 | **SBTS** |
|                          | % err | **14264% ± 8212%** | 75440% ± 24681% | 0 | |
| **QQ plot**              | MSE   | 8.90e-6 ± 6.8e-8 | **7.09e-6 ± 3.3e-6** | 0 | **TimeGAN** |
|                          | % err | **113.1% ± 5.8%** | 157.9% ± 27.2% | 0 | |
| **ACF \|r\| lags 1–20**  | MSE   | **4.57e-3 ± 3.7e-5** | 1.05e-2 ± 8.5e-3 | 0 | **SBTS** |
|                          | % err | **1269.9% ± 40.0%** | 2342.1% ± 1351.8% | 0 | |
| **ACF r² lags 1–20**     | MSE   | **5.17e-3 ± 5.7e-5** | 5.77e-3 ± 3.3e-3 | 0 | **SBTS** |
|                          | % err | **1603.5% ± 142.8%** | 4578.1% ± 4266.4% | 0 | |
| **Rolling vol histogram**| MSE   | 1227.3 ± 5.1 | **439.3 ± 216.7** | 0 | **TimeGAN** |
|                          | % err | **715.9% ± 20.2%** | 882.8% ± 410.8% | 0 | |
| **Tail survival**        | MSE   | **5.74e-3 ± 6.6e-5** | 1.17e-2 ± 9.2e-3 | 0 | **SBTS** |
|                          | % err | **17151% ± 910%** | 24327% ± 11324% | 0 | |

The **% err** row blows up when the real curve passes through near-zero values (empty histogram bins, tail-survival ≈ 0, near-zero ACF lags), so triple-digit-plus percentages are expected — a property of the curve, not a bug. TimeGAN's log-return-histogram MSE std (±120.6) is driven by a genuine seed-2 collapse (504.5 vs 11–170 for the other seeds).

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
| A10 | Kurtosis Error | Fat Tail | ↓ | \|κ_real − κ_gen\| on log-returns |
| A17 | \|r\| q95 Error | Fat Tail | ↓ | \|q_0.95(\|r_real\|) − q_0.95(\|r_gen\|)\| |
| A18 | \|r\| q99 Error | Fat Tail | ↓ | \|q_0.99(\|r_real\|) − q_0.99(\|r_gen\|)\| |
| A30 | Tail QQ Error | Fat Tail | ↓ | QQ RMSE restricted to top-5% tail quantiles |
| A34 | Hill Tail Index Error | Fat Tail | ↓ | \|Hill_real − Hill_gen\|; Hill (1975), top 5% threshold |
| A1 | Path MMD² | Distribution | ↓ | RBF kernel on full paths; Gretton et al. (2012) |
| A2 | Terminal MMD² | Distribution | ↓ | RBF kernel on terminal prices S_T |
| A3 | Increment MMD² | Distribution | ↓ | RBF kernel on log-return increments |
| A4 | Volatility MMD | Distribution | ↓ | RBF kernel on rolling 5-step realized vol |
| A5 | Terminal SWD | Distribution | ↓ | Sliced Wasserstein on S_T; Rabin et al. (2012) |
| A6 | Path SWD | Distribution | ↓ | Sliced Wasserstein on full paths |
| A24 | RV Law Loss | Distribution | ↓ | W₁(RV_real, RV_gen); RV_i=Σ_t r²_{i,t}/dt; Barndorff-Nielsen & Shephard (2002) |
| A25 | Mean Path RMSE | Distribution | ↓ | RMSE between real/gen mean trajectories |
| A27 | KS on Log-returns | Distribution | ↓ | KS statistic on pooled log-returns; Massey (1951) |
| A28 | Skewness Error | Distribution | ↓ | \|skew_real − skew_gen\| on log-returns; Cont (2001) |
| A29 | QQ RMSE (300-pt) | Distribution | ↓ | QQ RMSE over 300 uniform quantile levels |
| A33 | Terminal Price KS | Distribution | ↓ | KS statistic on terminal prices S_T |
| A13 | Disc Score GRU / MLP | Adversarial | ↓ | \|accuracy − 0.5\| on log-returns; Esteban et al. (2017) |
| A14 | Pred Score GRU / MLP | Predictive | ↓ | TSTR MAE on log-returns; Esteban et al. (2017) |
| A7 | Cov Error | Temporal | ↓ | ‖Σ_real − Σ_gen‖_F / ‖Σ_real‖_F × 100% |
| A11 | ACF \|r\| Error | Temporal | ↓ | Mean \|ACF_real(k) − ACF_gen(k)\| over lags 1–20 on \|r\| |
| A12 | ACF r² Error | Temporal | ↓ | Mean \|ACF_real(k) − ACF_gen(k)\| over lags 1–20 on r² |
| A22 | ACF \|r\| Lag-1 Error | Temporal | ↓ | \|ACF_real(1) − ACF_gen(1)\| on \|r\|; Heston ≈ +0.052 |
| A23 | ACF r² Lag-1 Error | Temporal | ↓ | \|ACF_real(1) − ACF_gen(1)\| on r²; Heston ≈ +0.050 |
| A8 | Mean RMSE | Vol | ↓ | RMSE of per-step mean price E[S_t] |
| A9 | Return Std Error | Vol | ↓ | \|std(r_real) − std(r_gen)\| on price increments ΔS_t |
| A16 | Log-Return Std Error | Vol | ↓ | \|std(r_real) − std(r_gen)\| on log-returns |
| A19 | Kurtosis Ratio (→ 1) | Vol | — | κ_real / κ_gen; perfect = 1.0 |
| A20 | Sigma Mean Error | Vol | ↓ | \|mean(σ_real) − mean(σ_gen)\| annualized per-path vol |
| A26 | Cross-Sect. Vol Path RMSE | Vol | ↓ | RMSE of cross-sectional vol trajectory |
| A31 | Rolling Vol KS | Vol | ↓ | KS on rolling-5 vol histograms; Mandelbrot (1963) |
| A32 | Vol-of-Vol Error | Vol | ↓ | \|std(rolling-vol_real) − std(rolling-vol_gen)\| |
| A15 | Sigma Corr / RMSE | Heston Spec | ↑/↓ | Pearson ρ / RMSE of QV-estimated vol vs true v_t |
| A21 | Oracle Sigma Corr | Heston Spec | ↑ | Same as A15 Corr; grouped with Heston vol metrics |

### B — Curve-shape metrics (6 diagnostic plots)

For each of 6 diagnostic plots we build three lists — the curve L, its first finite difference (der), and its second finite difference (sec\_der) — and score each list under **two measures**: MSE (absolute squared error) and % err (relative error). The three sub-scores are combined into one number per plot per measure (mean = sum, std = quadrature). Histogram bin edges use [0.5th, 99.5th]-percentile of **real data only**, making the reference curve fixed across seeds.

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
