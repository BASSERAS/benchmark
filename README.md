# Time-Series Generation Benchmark

A public benchmark for evaluating **generative models of financial time series**.

Each method is trained on the same target dataset and evaluated with **20 metrics (A1–A20)**
plus **14 stylized metrics (B1–B14)** across 5 random seeds.

---

## Results — Heston (mean ± std, 5 seeds)

Cross-method comparison on 8 192 Heston price paths (seq\_len=128).
↓ = lower is better. ↑ = higher is better. **Bold** = best across methods.
"Perfect Recovery" = two independent halves of the real dataset evaluated against each other.

### A1–A20 — Core Metrics

| Metric | Perfect Recovery | SBTS | TimeGAN | Winner |
|--------|:---------------:|:----:|:-------:|:------:|
| A1  Path MMD² ↓ | 0.0018 ± 0.0002 | **0.0112 ± 0.0011** | 0.0181 ± 0.0147 | **SBTS** |
| A2  Terminal MMD² ↓ | 0.0016 ± 0.0002 | **0.0102 ± 0.0014** | 0.0308 ± 0.0229 | **SBTS** |
| A3  Increment MMD² ↓ | 0.0008 ± 0.0000 | **0.0069 ± 0.0005** | 0.0077 ± 0.0039 | **SBTS** |
| A4  Volatility MMD ↓ | 0.0077 ± 0.0006 | **0.2964 ± 0.0126** | 0.3933 ± 0.2553 | **SBTS** |
| A5  Terminal SWD ↓ | 0.7635 ± 0.1174 | 3.7097 ± 0.3209 | **3.1284 ± 0.9227** | **TimeGAN** |
| A6  Path SWD ↓ | 0.5542 ± 0.0624 | 2.5335 ± 0.2212 | **1.6343 ± 0.5763** | **TimeGAN** |
| A7  Cov Error ↓ | 4.76 ± 2.50 | 145.35 ± 4.89 | **17.75 ± 6.71** | **TimeGAN** |
| A8  Mean RMSE ↓ | 0.1400 ± 0.1303 | 1.3013 ± 0.2776 | **0.7385 ± 0.4552** | **TimeGAN** |
| A9  Std Error ↓ | 0.0048 ± 0.0031 | 0.2492 ± 0.0018 | **0.1519 ± 0.0888** | **TimeGAN** |
| A10 Kurtosis Error ↓ | 0.0172 ± 0.0155 | **0.1187 ± 0.0060** | 2.9545 ± 2.0988 | **SBTS** |
| A11 ACF \|r\| Error ↓ | 0.0017 ± 0.0006 | **0.0596 ± 0.0005** | 0.1252 ± 0.0674 | **SBTS** |
| A12 ACF r² Error ↓ | 0.0014 ± 0.0006 | **0.0619 ± 0.0005** | 0.0839 ± 0.0348 | **SBTS** |
| A13 Disc GRU ↓ | 0.0128 ± 0.0068 | 0.2740 ± 0.2208 | **0.0099 ± 0.0084** | **TimeGAN** |
| A13 Disc MLP ↓ | 0.0080 ± 0.0081 | **0.0063 ± 0.0038** | 0.0921 ± 0.0463 | **SBTS** |
| A14 Pred GRU ↓ | 0.0564 ± 0.0022 | 0.0586 ± 0.0000 | **0.0570 ± 0.0013** | **TimeGAN** |
| A14 Pred MLP ↓ | 0.0565 ± 0.0022 | 0.0582 ± 0.0002 | **0.0573 ± 0.0015** | **TimeGAN** |
| A15 Sigma Corr ↑ | 0.6135 ± 0.0019 | **0.0046 ± 0.0019** | 0.0021 ± 0.0090 | **SBTS** |
| A15 Sigma RMSE ↓ | 0.0653 ± 0.0002 | **0.0955 ± 0.0001** | 0.1183 ± 0.0184 | **SBTS** |
| A16 Tail RMS ↓ | 0.0008 ± 0.0008 | 0.0428 ± 0.0002 | **0.0234 ± 0.0109** | **TimeGAN** |
| A17 Oracle MAE ↓ | 0.0097 ± 0.0000 | 0.0097 ± 0.0000 | 0.0097 ± 0.0000 | Tie |
| A18 Agent MAE ↓ | 0.0097 ± 0.0000 | 0.0106 ± 0.0000 | **0.0101 ± 0.0003** | **TimeGAN** |
| A19 Oracle-Agent Corr ↑ | −0.058 ± 0.430 | −0.342 ± 0.171 | **−0.332 ± 0.306** | **TimeGAN** |
| A20 RV Law Loss ↓ | 0.0673 ± 0.0362 | 2.1482 ± 0.0074 | **1.5512 ± 0.3788** | **TimeGAN** |
| PS-MC CRPS H=32 ↓ | — | **2.761 ± 0.004** | 3.087 ± 0.340 | **SBTS** |
| PS-MC CRPS H=64 ↓ | — | **3.900 ± 0.008** | 4.372 ± 0.431 | **SBTS** |

**SBTS wins A1–A20: 10/23. TimeGAN wins 12/23. Tie 1/23.**

### B1–B14 — Stylized Metrics

Extracted from 8 diagnostic plot panels (return distribution, QQ, ACF, vol clustering, tail survival).
Each quantifies a known stylized fact (Cont 2001, Bollerslev 1986, Hill 1975).

| Metric | Perfect Recovery | SBTS | TimeGAN | Winner |
|--------|:---------------:|:----:|:-------:|:------:|
| B1  Mean Path RMSE ↓ | 0.1511 ± 0.0708 | 0.7499 ± 0.1823 | **0.5289 ± 0.2624** | **TimeGAN** |
| B2  Cross-Sect. Vol RMSE ↓ | 0.1355 ± 0.0735 | 3.2760 ± 0.0637 | **0.3534 ± 0.1253** | **TimeGAN** |
| B3  KS on Log-returns ↓ | 0.0018 ± 0.0009 | **0.0534 ± 0.0004** | 0.0848 ± 0.0374 | **SBTS** |
| B4  Skewness Error ↓ | 0.0060 ± 0.0048 | **0.0227 ± 0.0037** | 0.3404 ± 0.3344 | **SBTS** |
| B5  QQ RMSE (300-pt) ↓ | 0.0001 ± 0.0000 | 0.0028 ± 0.0000 | **0.0025 ± 0.0006** | **TimeGAN** |
| B6  Tail QQ Error ↓ | 0.0001 ± 0.0001 | 0.0062 ± 0.0000 | **0.0034 ± 0.0015** | **TimeGAN** |
| B7  ACF lag-1 \|r\| Err ↓ | 0.0018 ± 0.0016 | **0.1449 ± 0.0012** | 0.2282 ± 0.1042 | **SBTS** |
| B8  ARCH Persistence Err ↓ | 0.0011 ± 0.0005 | **0.0274 ± 0.0004** | 0.0591 ± 0.0359 | **SBTS** |
| B9  ACF lag-1 r² Err ↓ | 0.0017 ± 0.0014 | **0.1678 ± 0.0017** | 0.1732 ± 0.0631 | **SBTS** |
| B10 GARCH Persistence Err ↓ | 0.0010 ± 0.0006 | **0.0227 ± 0.0004** | 0.0328 ± 0.0151 | **SBTS** |
| B11 Rolling Vol KS ↓ | 0.0046 ± 0.0024 | 0.3435 ± 0.0006 | **0.2540 ± 0.1093** | **TimeGAN** |
| B12 Vol-of-Vol Error ↓ | 0.0000 ± 0.0000 | 0.0021 ± 0.0000 | **0.0009 ± 0.0009** | **TimeGAN** |
| B13 Terminal Price KS ↓ | 0.0145 ± 0.0043 | **0.0921 ± 0.0051** | 0.1121 ± 0.0556 | **SBTS** |
| B14 Hill Tail Index Err ↓ | 0.499 ± 0.610 | **9.499 ± 0.346** | 36.88 ± 17.05 | **SBTS** |

**SBTS wins B1–B14: 8/14. TimeGAN wins 6/14.**

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

## Metrics (A1–A20 + B1–B14)

### A1–A20 — Core Metrics

| ID | Name | Dir | Formula reference |
|----|------|-----|------------------|
| A1 | Path MMD² | ↓ | RBF kernel on full paths; Gretton et al. (2012) |
| A2 | Terminal MMD² | ↓ | RBF kernel on terminal prices S_T |
| A3 | Increment MMD² | ↓ | RBF kernel on log-return increments |
| A4 | Volatility MMD | ↓ | RBF kernel on rolling 5-step realized vol |
| A5 | Terminal SWD | ↓ | Sliced Wasserstein on S_T; Rabin et al. (2012) |
| A6 | Path SWD | ↓ | Sliced Wasserstein on full paths |
| A7 | Cov Error | ↓ | ‖Σ_real − Σ_gen‖_F / ‖Σ_real‖_F × 100% |
| A8 | Mean RMSE | ↓ | RMSE of per-step mean price E[S_t] |
| A9 | Std Error | ↓ | |std(r_real) − std(r_gen)| on log-returns |
| A10 | Kurtosis Error | ↓ | |κ_real − κ_gen| on log-returns |
| A11 | ACF \|r\| Error | ↓ | Mean |ACF_real(k) − ACF_gen(k)| over lags 1–20 on |r| |
| A12 | ACF r² Error | ↓ | Mean |ACF_real(k) − ACF_gen(k)| over lags 1–20 on r² |
| A13 | Disc Score (GRU+MLP) | ↓ | |accuracy − 0.5| on log-returns; Esteban et al. (2017) |
| A14 | Pred Score (GRU+MLP) | ↓ | TSTR MAE on log-returns; Esteban et al. (2017) |
| A15 | Sigma Corr / RMSE | ↑/↓ | Pearson ρ of QV-estimated vol vs true v_t (Heston-specific) |
| A16 | Tail Survival Error | ↓ | RMS of P(\|r\|>q) mismatch at q={0.90,0.95,0.99} |
| A17 | Oracle MAE | ↓ | AR(5) OLS on real log-returns, predict real test inputs |
| A18 | Agent MAE | ↓ | AR(5) OLS on synthetic log-returns, predict real test inputs |
| A19 | Oracle-Agent Corr | ↑ | Pearson ρ(oracle_pred, agent_pred) on real test inputs |
| A20 | RV Law Loss | ↓ | W₁(RV_real, RV_gen); RV_i=Σ_t r²_{i,t}/dt; Barndorff-Nielsen & Shephard (2002) |

### B1–B14 — Stylized Metrics

| ID | Name | Panel | Reference |
|----|------|-------|-----------|
| B1 | Mean Path RMSE | Sample paths | Path-space L2 |
| B2 | Cross-Sect. Vol RMSE | Sample paths | Cross-sectional volatility spread |
| B3 | KS on Log-returns | Return histogram | Massey (1951) |
| B4 | Skewness Error | Return histogram | Cont (2001) |
| B5 | QQ RMSE (300-pt) | QQ plot | Quantile-quantile L2 |
| B6 | Tail QQ Error | QQ plot | Tail quantiles {1-5%, 95-99%} |
| B7 | ACF lag-1 \|r\| Error | ACF |r| | ARCH effect; Engle (1982) |
| B8 | ARCH Persistence Error | ACF |r| | Mean ACF |r| over lags 1–20 |
| B9 | ACF lag-1 r² Error | ACF r² | GARCH effect; Bollerslev (1986) |
| B10 | GARCH Persistence Error | ACF r² | Mean ACF r² over lags 1–20 |
| B11 | Rolling Vol KS | Rolling vol hist. | Vol clustering; Mandelbrot (1963) |
| B12 | Vol-of-Vol Error | Rolling vol hist. | Std of rolling 5-step realized vol |
| B13 | Terminal Price KS | Tail survival | Massey (1951) on S_T |
| B14 | Hill Tail Index Error | Tail survival | Hill (1975), top 10% threshold |

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
4. Results appear in `results/Heston/NewMethod/` with full A1–A20 + B1–B14 tables
