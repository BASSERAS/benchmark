# Metrics, LS4 on Heston — **SBTS log-return preprocessing** (5 seeds)

**The experiment.** Take the exact SBTS volatility-scaled log-return transform and feed it to LS4
in place of LS4's native standardized-price input. Everything else — SDE, parameters, RNG streams,
metric code, seeds 0–4 — is held fixed. This folder is the answer to a single question: *does
SBTS-style preprocessing help or hurt LS4?* The reference point throughout is the **original LS4 run**
at [`../../LS4/`](../../LS4/README.md), which uses standardized prices.

**Dataset:** 4 096 Heston price paths, seq_len = 128 (the main benchmark uses 8 192; this experiment
uses 4 096 for train / test / disc, see [`../README.md`](../README.md)).
Parameters: μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250.

**Data split (test set everywhere).** Disjoint 4 096-path Heston draws: the generator is **trained on
seed 0**; every A/B metric and diagnostic compares generated paths against the **test set (seed 1)**;
A18 discriminative uses a **third real set (seed 2)** as the "real" class. No metric is scored against
the generator's own training data.

## Preprocessing (the only thing that changed)

```python
DT, S0 = 1.0/250.0, 100.0
R        = np.log(S[:, 1:] / S[:, :-1])        # (M,127) log-returns
sigma    = float(R.std())                       # pooled ddof=0, TRAIN seed 0, frozen
R_tilde  = R * np.sqrt(DT) / sigma              # std -> sqrt(DT) = 0.063246
X_sbts   = np.hstack([np.zeros((M,1)), R_tilde])# (M,128) dummy-0 column prepended
X_train  = (X_sbts - x_mu) / x_sd               # LS4 unit-variance wrapper (see note)
```

**Reported SBTS sigma (frozen, shared with SBTS):** `sigma = 0.01263163`
(estimated on the 4 096-path train seed 0, ddof=0, pooled over all M×127 raw log-returns).
After scaling, `R̃.std() = √dt = 0.063246`.

> ⚠️ **Unit-variance wrapper (required, does not touch the reported sigma).** SBTS-scaled returns have
> std ≈ 0.063, which sits **below LS4's decoder noise `sigma = 0.1`** and collapses the VAE. We therefore
> apply LS4's own standardization `(X_sbts − x_mu)/x_sd` (x_mu = 0.000703383, x_sd = 0.062998) **on top of**
> the SBTS transform, and invert it symmetrically before the SBTS de-scaling. The **reported SBTS sigma is
> unchanged** — the wrapper is an internal training convenience, inverted exactly at generation time. The
> Cauchy conjugate-pair fix at `code/reference/models/s4.py:795` from the original LS4 run is carried over.

**Model:** LS4 (Zhou et al., ICML 2023, arXiv:2212.12749), released `solar_weekly` preset,
2 146 857 params, z_dim = 5, d_model = 64, n_layers = 4, s4_type = s4, decoder sigma = 0.1.
Trained **100 epochs** (canonical count, matching the original LS4 run), AdamW + ReduceLROnPlateau +
EMA(0.99, start_step 200). See [`code/train_ls4_logret.py`](code/train_ls4_logret.py).

**Convention:** lower is better for all metrics **except A33 Teacher-Sigma Corr ↑**. A28 Kurtosis
Ratio: perfect = 1.0.

---

## Headline: log-return preprocessing **hurts LS4**

Straight to the verdict, no softening. Swapping LS4's standardized-price input for SBTS log-returns
**degrades almost every fidelity metric** and **destabilizes training across seeds**. The original
LS4 was the benchmark's best density/path matcher; under this preprocessing it loses that status.

- **Density / path collapse.** A6 Path MMD² **0.004843** vs original **0.001926** (2.5× worse);
  A7 Terminal MMD² **0.009408** vs **0.001520** (6× worse); A17 Terminal-price KS **0.07261** vs
  **0.01584** (4.6× worse); grid_tvd **6.99 %** vs **2.77 %** (2.5× worse).
- **Curves.** Log-return-histogram curve MSE **2.313** vs **0.4517** (5× worse).
- **Adversarial / predictive slip off the floor.** A18 GRU **0.007748** vs **0.005890**;
  A19 GRU **0.05641** vs **0.05001**. Still low, but no longer floor-level.
- **Moments blow up in the mean.** A1 Kurtosis Error **0.6156** vs **0.3684**; A25 Mean RMSE
  **1.380** vs **0.327**.
- **Seed instability (the real story).** The std columns explode. Seed 4 degenerates
  (A1 = 1.435, A5 Hill = 3.44, log-ret-hist MSE = 7.77); seed 1 also unstable
  (A6 = 0.0158, A7 = 0.0381, grid_tvd = 18.2 %). The price-input LS4 was seed-robust; the
  log-return input is **not**.

**The one genuine improvement — A28.** Kurtosis Ratio moves to **0.7334** (|Δ to 1| = 0.267) from the
original **1.565** (|Δ to 1| = 0.565). Original LS4 was mildly *platykurtic-above* (ratio > 1); the
log-return version lands **closer to the 1.0 target**, though now slightly *below* (thinner extreme
tail). This is the only axis where the preprocessing clearly helps, and it is a moments-ratio effect,
not a distributional win (A6/A7/A17 all worse). **A33 remains ≈ 0** (0.003 vs −0.0004): as with every
single-factor generator here, the per-path latent variance is unrecoverable from prices — the
preprocessing does nothing for it.

---

## Results (mean ± std across 5 seeds) — LS4+logret vs original LS4

Lower is better except **A33 ↑**; A28 target = 1.0. **Δ** column: ✅ log-return better, ❌ worse,
≈ negligible. "Original" = [`../../LS4/`](../../LS4/README.md) (standardized-price input).

| Metric | LS4 + logret (mean ± std) | Original LS4 (mean) | Δ |
|--------|---------------------------|---------------------|---|
| **, Fat Tail, ** | | | |
| A1 Kurtosis Error ↓ | 0.6156 ± 0.4445 | 0.3684 | ❌ |
| A2 \|r\| q95 Error ↓ | 0.001418 ± 6.2e-04 | 3.99e-04 | ❌ |
| A3 \|r\| q99 Error ↓ | 6.94e-04 ± 9.7e-04 | 0.001156 | ✅ |
| A4 Tail QQ Error ↓ | 0.001612 ± 3.4e-04 | 4.05e-04 | ❌ |
| A5 Hill Tail Index Error ↓ | 4.257 ± 3.274 | 1.225 | ❌ |
| **, Distribution, ** | | | |
| A6 Path MMD² ↓ | 0.004843 ± 0.005499 | 0.001926 | ❌ |
| A7 Terminal MMD² ↓ | 0.009408 ± 0.014354 | 0.001520 | ❌ |
| A8 Increment MMD² ↓ | 0.001310 ± 3.6e-04 | 9.63e-04 | ❌ |
| A9 Volatility MMD ↓ | 0.03631 ± 0.01918 | 0.01447 | ❌ |
| A10 Terminal SWD ↓ | 2.197 ± 1.712 | 0.7480 | ❌ |
| A11 Path SWD ↓ | 1.076 ± 0.8519 | 0.5744 | ❌ |
| A12 RV Law Loss ↓ | 0.5482 ± 0.2282 | 0.2415 | ❌ |
| A13 Mean Path RMSE ↓ | 0.7794 ± 0.8059 | 0.1722 | ❌ |
| A14 KS Log-returns ↓ | 0.01940 ± 0.01036 | 0.01258 | ❌ |
| A15 Skewness Error ↓ | 0.1688 ± 0.1020 | 0.02998 | ❌ |
| A16 QQ RMSE (300-pt) ↓ | 8.77e-04 ± 3.0e-04 | 3.41e-04 | ❌ |
| A17 Terminal Price KS ↓ | 0.07261 ± 0.07271 | 0.01584 | ❌ |
| **, Adversarial, ** | | | |
| A18 Disc Score GRU ↓ | 0.007748 ± 0.006227 | 0.005890 | ❌ |
| A18 Disc Score MLP ↓ | 0.007138 ± 0.003524 | 0.006256 | ❌ |
| **, Predictive, ** | | | |
| A19 Pred Score GRU ↓ | 0.05641 ± 5.1e-05 | 0.05001 | ❌ |
| A19 Pred Score MLP ↓ | 0.05643 ± 1.3e-04 | 0.05006 | ❌ |
| **, Temporal, ** | | | |
| A20 Covariance Error ↓ | 56.78 ± 42.52 | 13.63 | ❌ |
| A21 ACF \|r\| Error ↓ | 0.01490 ± 0.004887 | 0.01294 | ❌ |
| A22 ACF r² Error ↓ | 0.01120 ± 0.003591 | 0.006752 | ❌ |
| A23 ACF \|r\| Lag-1 Error ↓ | 0.02867 ± 0.009143 | 0.01743 | ❌ |
| A24 ACF r² Lag-1 Error ↓ | 0.02300 ± 0.01036 | 0.009068 | ❌ |
| **, Vol, ** | | | |
| A25 Mean RMSE ↓ | 1.380 ± 1.360 | 0.3270 | ❌ |
| A26 Return Std Error ↓ | 0.05371 ± 0.03238 | 0.004853 | ❌ |
| A27 Log-Return Std Error ↓ | 6.19e-04 ± 3.7e-04 | 4.63e-05 | ❌ |
| A28 Kurtosis Ratio (→ 1) | 0.7334 ± 0.2007 | 1.565 | ✅ |
| A29 Sigma Mean Error ↓ | 0.008282 ± 0.006064 | 0.001445 | ❌ |
| A30 Cross-Sect. Vol Path RMSE ↓ | 1.320 ± 1.181 | 0.3372 | ❌ |
| A31 Rolling Vol KS (w=5) ↓ | 0.05317 ± 0.03392 | 0.03798 | ❌ |
| A32 Vol-of-Vol Error ↓ | 2.53e-04 ± 1.7e-04 | 3.21e-04 | ✅ |
| **, Heston Spec, ** | | | |
| A33 Teacher-Sigma Corr ↑ | 0.003268 ± 0.008388 | -3.94e-04 | ≈ |
| A34 Teacher-Sigma RMSE ↓ | 0.09953 ± 0.003354 | 0.09513 | ❌ |

**Scoreboard: 4 of 34 rows improve** (A3, A28, A32, and A33≈), the rest regress. The improvements are
tail-moment ratios (A3 q99, A28 kurtosis ratio, A32 vol-of-vol), consistent with the log-return
parametrization tightening the *extreme-return moment scaling* while loosening the *joint density and
path structure* everywhere else.

---

## Stylised Facts Diagnostic (Heston vs LS4+logret, seed 0)

Eight-panel comparison (Murex Fig. 1 style): sample paths, return distribution, QQ plot,
ACF |r|, ACF r², rolling-vol histogram (window 5), tail survival (log-log).

![Heston Diagnostics](plots/heston_diagnostics.png)

---

## Curve-shape metrics (B) — funct-level, mean ± std across 5 seeds

Each diagnostic plot yields a **curve** L. Below is the curve-level (funct) MSE, %err and NRMSE for the
new run, plus the grid_tvd path-cloud metric. The original LS4's reported "MSE" is a mean of
funct/der/sec_der; the new figures are **funct-only** (the curve itself), which is the dominant term —
directional comparison is unaffected. Lower is better.

| Plot | funct MSE | funct %err | funct NRMSE | Original LS4 (MSE, mean-of-3) |
|------|-----------|-----------|-------------|-------------------------------|
| Path cloud (grid_tvd 50×50 %) | **6.989 % ± 5.606 %** | — | — | 2.772 % |
| Log-return histogram | 2.313 ± 2.789 | 34.16 % | 15.66 % | 0.4517 |
| QQ plot | 2.14e-06 ± 1.5e-06 | 15.02 % | 6.86 % | 4.59e-08 |
| ACF \|r\| lags 1-20 | 8.90e-05 ± 2.9e-05 | 43.72 % | 33.5 % | 5.14e-05 |
| ACF r² lags 1-20 | 5.79e-05 ± 2.4e-05 | 32.44 % | 24.53 % | 2.48e-05 |
| Rolling vol histogram | 44.90 ± 42.84 | 16.34 % | 6.71 % | 8.514 |
| Tail survival | 7.74e-04 ± 9.7e-04 | 6.93 % | 2.22 % | 6.90e-05 |

Every curve degrades relative to price-input LS4 — the log-return-histogram and rolling-vol curves by
~5×, the QQ curve by ~50× — confirming the A-table: the marginal density and rolling-vol distribution
no longer sit on top of Heston's.

**Plot → key mapping:** `B_log_ret_hist_*`, `B_qq_plot_*`, `B_acf_abs_r_*`, `B_acf_sq_r_*`,
`B_roll_vol_hist_*`, `B_tail_surv_*` (funct / der / sec_der sub-scores, funct shown).

> ACF %err is a near-zero-denominator artefact (true ACF ≈ 0.05); read MSE for absolute agreement.

---

## Latent projections (per seed)

PCA and t-SNE 2-D projections, real (test seed 1) vs generated, one pair per seed. Seeds 1 and 4 show
the visible real/fake separation that the elevated A6/A17 predict.

| Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|--------|--------|--------|--------|--------|
| ![](plots/seed_0_pca.png) | ![](plots/seed_1_pca.png) | ![](plots/seed_2_pca.png) | ![](plots/seed_3_pca.png) | ![](plots/seed_4_pca.png) |
| ![](plots/seed_0_tsne.png) | ![](plots/seed_1_tsne.png) | ![](plots/seed_2_tsne.png) | ![](plots/seed_3_tsne.png) | ![](plots/seed_4_tsne.png) |

---

## Path Shadowing — strict paper protocol (arXiv:2308.01486)

> This section uses the **exact protocol from the paper**, *not* the simplified `methods/LS4`
> reference eval (65D murex embedding, K=77, prefix-price L2, CRPS/MAE/RMSE only). See
> [`../GUIDELINE.md` §9 + M7](../GUIDELINE.md).

Each real ps-split prefix (65 points → 64 log-returns) is embedded with the **4-block weighted,
bank-standardized** feature vector — recent returns (last 32, w1.0) · cumulative path (downsampled
24, w0.5) · rolling vol (windows 5/10/20 last/mean/std, w2.0) · dependence (ACF of `|r|` & `r²` at
lags 1,2,5,10, w1.0), with `z̃ = √w·(z−μ_bank)/σ_bank`. Retrieve **K = 256** nearest LS4+logret bank
paths; their futures give the predictive ensemble for three return-based quantities (**cumulative
return, one-step return, horizon RV**). Split `s = 64`, horizon `H = 32`, **512** independent query
paths (seed 3). **Banks = 1 000 000 generated paths per seed** (`path_shadowing/bank/generated_bank_seed{0..4}_1000000x128.npy`,
~0.5 GB each), evaluated as **nested prefixes** over the bank-size sweep
{4096, 16384, 65536, 262144, 1 000 000}. Metrics per quantity: predictive-mean RMSE, CRPS (energy),
coverage 50/90, band width 50/90, lower/upper-90 miss — with **2000-resample bootstrap 95% CIs**.

<!-- PS-PDF-TABLE-START -->
All numbers are **mean ± std across the 5 seeds** at the full **1 000 000-path bank** (log-return
scale; lower CRPS/RMSE better). Nominal coverage in parentheses. The RW baseline resamples each
query's own prefix returns.

**Headline — 1M bank, LS4+logret vs random-walk baseline**

| Quantity | RMSE (LS4) | RMSE (RW) | CRPS (LS4) | CRPS (RW) | cov90 (0.90) | width90 (LS4/RW) |
|----------|-----------:|----------:|-----------:|----------:|:------------:|:----------------:|
| **cumulative return** | **0.0697 ± 0.0012** | 0.0836 | **0.03785 ± 0.00075** | 0.04668 | 0.842 ± 0.029 (0.818) | 0.190 / 0.223 |
| one-step return       | **0.01234 ± 0.00006** | 0.01248 | **0.006762 ± 0.000043** | 0.006885 | 0.859 ± 0.020 (0.893) | 0.0349 / 0.0395 |
| horizon RV            | **0.01814 ± 0.00100** | 0.01876 | **0.01032 ± 0.00064** | 0.01168 | 0.855 ± 0.034 (**0.533**) | 0.0512 / 0.0287 |

LS4+logret **beats the random walk on every quantity and metric**. The margin is largest on
cumulative return (CRPS −19%, RMSE −17%) and realized vol (CRPS −12%); one-step return is nearly a
coin-flip (−2%), as expected since a single Heston increment is almost pure noise. RV is where PS-MC
clearly earns its keep: the RW's RV band is badly miscalibrated (coverage **0.53** vs nominal 0.90),
while the shadowed ensemble reaches 0.855.

**Bank-size sweep — CRPS mean ± std (nested prefixes of the one 1M bank)**

| bank size | cum CRPS | one-step CRPS | RV CRPS | unique-cand frac | prefix dist (mean) |
|----------:|---------:|--------------:|--------:|:----------------:|:------------------:|
| 4 096     | 0.03784 ± 0.00038 | 0.006810 ± 0.000034 | 0.01124 ± 0.00072 | 0.993 | 8.56 |
| 16 384    | 0.03791 ± 0.00043 | 0.006785 ± 0.000021 | 0.01092 ± 0.00072 | 0.938 | 7.99 |
| 65 536    | 0.03784 ± 0.00059 | 0.006769 ± 0.000019 | 0.01065 ± 0.00065 | 0.705 | 7.53 |
| 262 144   | 0.03790 ± 0.00064 | 0.006776 ± 0.000027 | 0.01046 ± 0.00065 | 0.342 | 7.14 |
| 1 000 000 | 0.03785 ± 0.00075 | 0.006762 ± 0.000043 | 0.01032 ± 0.00064 | 0.116 | 6.80 |

**The sweep is flat for cumulative and one-step return** — growing the bank 244× (4k→1M) leaves their
CRPS unchanged. Only **realized vol** improves monotonically (0.01124→0.01032, −8%). Meanwhile the
mean prefix distance keeps shrinking (8.56→6.80) and the unique-candidate fraction collapses
(0.99→0.12): a bigger bank does supply geometrically closer shadows, but for return-level forecasts
LS4's generated distribution saturates the useful shadowing content by ~4k paths. The 1M bank is
required by the protocol and pays off only for the vol quantity.

**Diagnostics (1M bank, 5-seed mean)**

| terminal RMSE | prefix dist mean / median / p95 | unique-cand frac | RV mean bias |
|:-------------:|:-------------------------------:|:----------------:|:------------:|
| 0.0697 | 6.80 / 6.62 / 9.53 | 0.116 | −0.0062 |

Coverage sits **mildly below nominal** (~0.84–0.86 at the 90% level) across quantities → the ensemble
is slightly over-confident; RV is under-predicted by ~0.6% (negative bias). Per-seed **2000-resample
bootstrap 95% CIs** on RMSE and CRPS are in `path_shadowing/pdf_results_seed{i}.json`; the ±std above
is the 5-seed dispersion.
<!-- PS-PDF-TABLE-END -->

Driver: [`path_shadowing/path_shadowing_pdf.py`](path_shadowing/path_shadowing_pdf.py)
(bank builder: [`path_shadowing/gen_banks.py`](path_shadowing/gen_banks.py)).
Plots: ![](path_shadowing/plots/pdf_crps_vs_banksize.png)
![](path_shadowing/plots/pdf_coverage_calibration.png)

---

## Files

| File | Description |
|------|-------------|
| `metrics_summary.csv` | Mean ± std across 5 seeds, all A + B + grid_tvd metrics |
| `seed_{i}_metrics.json` | Full per-seed metric dict |
| `code/train_ls4_logret.py` | Training driver (SBTS transform + unit-variance wrapper + inverse) |
| `weights/seed_{i}_model.pt` | Checkpoint (model + ema_model + sbts_sigma + x_mu/x_sd + dt/s0) |
| `plots/heston_diagnostics.png` | 8-panel stylised-facts diagnostic (seed 0) |
| `plots/seed_{i}_pca.png` / `_tsne.png` | 2-D real-vs-fake projections per seed |
| `path_shadowing/path_shadowing_pdf.py` | **strict paper-protocol** evaluator (4-block embedding, bank-size sweep, cum/step/RV, coverage/width, bootstrap CIs) |
| `path_shadowing/gen_banks.py` | 1M-bank builder (per seed) |
| `path_shadowing/bank/` | `generated_bank_seed{0..4}_1000000x128.npy` (~0.5 GB each) |
| `path_shadowing/{pdf_results_seed{i},pdf_summary}.json` · `logs/` · `plots/pdf_*.png` | per-seed + aggregate metrics, run logs, sweep + calibration plots |

→ Experiment overview & pipeline: [`../README.md`](../README.md) ·
Recipe for adding methods: [`../GUIDELINE.md`](../GUIDELINE.md) ·
Original price-input LS4: [`../../LS4/README.md`](../../LS4/README.md)
