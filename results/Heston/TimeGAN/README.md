# Metrics — TimeGAN on Heston (5 Seeds)

**Dataset:** 8 192 Heston price paths, seq\_len = 128.
Parameters: μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250.

**Model:** PyTorch TimeGAN, 20 k steps (5 k embed + 5 k supervised + 10 k adversarial), 2×A100 80 GB.

**Convention:** lower is better for all metrics **except A33 Teacher-Sigma Corr ↑**. A28 Kurtosis Ratio: perfect = 1.0.

---

## Data split — train / test / disc

Every number on this page is an **out-of-sample** score. The benchmark uses three disjoint Heston
draws of 8 192 paths each:

- **Train (seed 0)** — the paths the generator was fitted on. Never scored here.
- **Test (seed 1)** — the held-out real reference. All A1–A17, A20–A34, every B curve, the diagnostic
  plots and PS-MC are computed **generated-vs-test**.
- **Disc (seed 2)** — a third independent real draw, used only as the "real" class for the A18
  discriminative and A19 predictive-TSTR classifiers, so the adversary never sees the test set.

---

## Results (mean ± std across 5 seeds)

### A1–A34 — Metrics by category

Last column = **Perfect floor**: the non-zero finite-sample noise floor a perfect generator reaches.
It is measured by scoring an **independent Heston draw** (fresh seeds, identical parameters) against the
test set — the same real-vs-real comparison every generated batch faces, so the floor is the best score
attainable when the model *is* the true process. Floors are **non-zero** because two independent finite
samples of the same law never coincide exactly; they are identical across methods (same test set, same
independent-draw protocol). See [`../../../methods/perfect_recovery/`](../../../methods/perfect_recovery/).

<!-- ===== PER-METHOD A TABLE ===== -->
| Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|--------|-----------|--------|--------|--------|--------|--------|---------------|
| **— Fat Tail —** | | | | | | | |
| A1 Kurtosis Error ↓ | 2.954 ± 2.098 | 0.01519 | 5.359 | 3.767 | 0.9585 | 4.672 | 0.008092 |
| A2 \|r\| q95 Error ↓ | 0.003196 ± 0.001907 | 0.004248 | 0.005644 | 0.001564 | 4.36e-04 | 0.004088 | 6.57e-05 |
| A3 \|r\| q99 Error ↓ | 0.004342 ± 0.002767 | 0.007384 | 0.006912 | 0.005192 | 0.001759 | 4.64e-04 | 5.98e-05 |
| A4 Tail QQ Error ↓ | 0.003401 ± 0.001522 | 0.004204 | 0.005410 | 0.001582 | 0.001658 | 0.004149 | 6.75e-05 |
| A5 Hill Tail Index Error ↓ | 36.32 ± 17.05 | 40.13 | 18.22 | 51.19 | 14.93 | 57.13 | 0.5266 |
| **— Distribution —** | | | | | | | |
| A6 Path MMD² ↓ | 0.01866 ± 0.01472 | 0.01025 | 0.003450 | 0.03561 | 0.006646 | 0.03735 | 0.001842 |
| A7 Terminal MMD² ↓ | 0.03072 ± 0.02472 | 0.01905 | 0.005805 | 0.06974 | 0.009835 | 0.04915 | 0.001983 |
| A8 Increment MMD² ↓ | 0.008280 ± 0.004303 | 0.005390 | 0.007798 | 0.01429 | 0.002251 | 0.01167 | 8.69e-04 |
| A9 Volatility MMD ↓ | 0.3975 ± 0.2486 | 0.1864 | 0.3624 | 0.7126 | 0.07878 | 0.6473 | 0.008554 |
| A10 Terminal SWD ↓ | 2.917 ± 1.131 | 2.517 | 1.402 | 4.550 | 2.274 | 3.840 | 1.151 |
| A11 Path SWD ↓ | 1.678 ± 0.5770 | 1.689 | 0.7691 | 2.383 | 1.371 | 2.176 | 0.6191 |
| A12 RV Law Loss ↓ | 1.558 ± 0.3879 | 1.515 | 1.776 | 1.816 | 0.8208 | 1.863 | 0.05202 |
| A13 Mean Path RMSE ↓ | 0.5356 ± 0.2514 | 0.4458 | 0.3018 | 0.9468 | 0.6922 | 0.2913 | 0.1205 |
| A14 KS Log-returns ↓ | 0.08474 ± 0.03769 | 0.03948 | 0.06328 | 0.1259 | 0.06167 | 0.1334 | 0.001491 |
| A15 Skewness Error ↓ | 0.3412 ± 0.3279 | 0.006432 | 0.4384 | 0.09798 | 0.2341 | 0.9290 | 0.005274 |
| A16 QQ RMSE (300-pt) ↓ | 0.002506 ± 6.49e-04 | 0.001915 | 0.002621 | 0.002769 | 0.001705 | 0.003521 | 4.19e-05 |
| A17 Terminal Price KS ↓ | 0.1109 ± 0.05875 | 0.1039 | 0.05347 | 0.2140 | 0.05603 | 0.1273 | 0.01099 |
| **— Adversarial —** | | | | | | | |
| A18 Disc Score GRU ↓ | 0.03305 ± 0.05328 | 0.003814 | 0.001984 | 0.01755 | 0.002899 | 0.1390 | 0.006195 |
| A18 Disc Score MLP ↓ | 0.08792 ± 0.04703 | 0.1384 | 0.06210 | 0.008697 | 0.1088 | 0.1216 | 0.005951 |
| **— Predictive —** | | | | | | | |
| A19 Pred Score GRU ↓ | 0.05277 ± 0.001115 | 0.05154 | 0.05456 | 0.05353 | 0.05221 | 0.05199 | 0.05002 |
| A19 Pred Score MLP ↓ | 0.05322 ± 0.001031 | 0.05136 | 0.05452 | 0.05350 | 0.05318 | 0.05352 | 0.05036 |
| **— Temporal —** | | | | | | | |
| A20 Covariance Error ↓ | 21.36 ± 9.068 | 14.84 | 24.77 | 8.799 | 35.38 | 22.99 | 4.923 |
| A21 ACF \|r\| Error (lags) ↓ | 0.1278 ± 0.06738 | 0.06730 | 0.1072 | 0.2036 | 0.05023 | 0.2106 | 0.002234 |
| A22 ACF r² Error (lags) ↓ | 0.08676 ± 0.03470 | 0.04795 | 0.08225 | 0.1202 | 0.05091 | 0.1325 | 0.002206 |
| A23 ACF \|r\| Lag-1 Error ↓ | 0.2301 ± 0.1034 | 0.1574 | 0.2157 | 0.3706 | 0.08767 | 0.3192 | 0.002652 |
| A24 ACF r² Lag-1 Error ↓ | 0.1760 ± 0.06259 | 0.1218 | 0.2041 | 0.2675 | 0.09154 | 0.1949 | 0.002790 |
| **— Vol —** | | | | | | | |
| A25 Mean RMSE ↓ | 0.7781 ± 0.3669 | 0.6340 | 0.5870 | 1.254 | 1.143 | 0.2722 | 0.1392 |
| A26 Return Std Error ↓ | 0.1525 ± 0.08911 | 0.1529 | 0.2389 | 0.03119 | 0.07782 | 0.2618 | 0.002523 |
| A27 Log-Return Std Error ↓ | 0.001703 ± 7.89e-04 | 0.002047 | 0.002359 | 5.93e-04 | 9.43e-04 | 0.002573 | 3.15e-05 |
| A28 Kurtosis Ratio (→ 1) | -1.116 ± 3.593 | 2.017 | 0.1387 | 0.2498 | -8.162 | 0.1752 | 1.006 |
| A29 Sigma Mean Error ↓ | 0.03089 ± 0.009106 | 0.03044 | 0.03768 | 0.02770 | 0.01602 | 0.04260 | 4.96e-04 |
| A30 Cross-Sect. Vol Path RMSE ↓ | 0.4742 ± 0.2079 | 0.4317 | 0.6883 | 0.1711 | 0.7242 | 0.3557 | 0.1432 |
| A31 Rolling Vol KS (w=5) ↓ | 0.2552 ± 0.1101 | 0.1895 | 0.2727 | 0.3634 | 0.07956 | 0.3709 | 0.003814 |
| A32 Vol-of-Vol Error ↓ | 8.96e-04 ± 8.69e-04 | 3.55e-04 | 2.76e-04 | 0.002530 | 2.62e-04 | 0.001060 | 1.54e-05 |
| **— Heston Spec —** | | | | | | | |
| A33 Teacher-Sigma Corr ↑ | 0.002745 ± 0.01354 | -0.003191 | 0.009072 | -0.01798 | 0.002783 | 0.02304 | 0.6163 |
| A34 Teacher-Sigma RMSE ↓ | 0.1186 ± 0.01863 | 0.1022 | 0.1112 | 0.1488 | 0.1000 | 0.1310 | 0.06559 |

**Footnotes.**
- **A18** — discriminative classifier trained on log-returns; score = |accuracy − 0.5|, 0 = indistinguishable (GRU + MLP). The GRU judge reaches the floor on 4 of 5 seeds but seed 4 collapses (0.139), while the MLP judge separates real from generated easily (0.088) — a signature of GAN samples that fool a recurrent critic but not a flatten-then-MLP one.
- **A19** — TSTR predictive MAE; all methods cluster near 0.050 (irreducible log-return floor) (GRU + MLP).
- **A33** — Teacher-Sigma correlation, **higher is better**; perfect floor ≈ 0.616 (unreachable from prices alone — the latent variance process is hidden).
- **A34** — Teacher-Sigma RMSE, perfect floor ≈ 0.066.

**Reading the table.** TimeGAN **wins none of the 36 A-metric rows** and none of the 6 B-plots — the
adversarial training is the least stable generator in this benchmark. The instability shows as very large
seed-to-seed variance (A6 path MMD² 0.0035 → 0.0356 across seeds; a seed-2 mode collapse recurs throughout)
and as the worst tail behaviour in the field: A1 kurtosis error 2.954, A5 Hill index error 36.32, and an
**A28 kurtosis ratio of −1.116** — the only negative (i.e. sign-flipped, degenerate) kurtosis of any method,
dragged there by a seed-3 blow-up (−8.162). Its autocorrelation block (A21–A24) is also the weakest, an
order of magnitude above the diffusion/state-space models. The one place it is not last is A20 covariance
error (21.36) and A30 cross-sectional vol RMSE (0.474), where the GAN's sharp per-path structure happens to
help — but neither is a category win. Overall TimeGAN reproduces the broad shape of a price path but not the
fat tails, volatility clustering, or run-to-run stability that Heston demands.

---

## Stylised Facts Diagnostic (Heston vs TimeGAN, seed 0)

Eight-panel comparison matching the Murex paper (Fig. 1 style): sample paths, return distribution,
QQ plot, ACF of |returns|, ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

![Heston Diagnostics](plots/heston_diagnostics.png)

---

## Curve-shape metrics (B) — mean ± std across 5 seeds

Each of the 6 diagnostic plots above yields a **curve** L (a list of values), not a scalar. For each plot
we build three lists — the curve L, its first finite difference L' (der), and its second finite difference
L'' (sec\_der) — and report **three rows per plot**:

- **MSE** — `mean((L_gen − L_real)²)`, averaged over the three lists (funct/der/sec\_der). This is the
  quantity that decides the cross-method winner.
- **% err** — function-level MAPE `mean(|L_gen − L_real| / (|L_real| + 1e-6)) × 100` on the curve L
  only (funct-only); the derivative / 2nd-difference MAPE is excluded as ill-posed (near-zero denominators).
- **NRMSE** — `sqrt(mean((L_gen − L_real)²)) / (max|L_real| − min|L_real| + 1e-12) × 100` on the curve L
  only (funct-only).

For **all six plots** the % err and NRMSE rows use the **function-level curve only** (funct); the MSE
row stays the mean of the three sub-scores.

↓ lower is better for all three rows. **Perfect floor** = the same independent-draw-vs-test floor as the A
table (non-zero, finite-sample). Winner between methods is decided by the **MSE** row. TimeGAN's enormous
log-return-histogram MSE (45.40) and its ±57.91 std are driven by a genuine seed-2 collapse (153.7 vs
3.6–58 on the other seeds).

<!-- ===== PER-METHOD B TABLE ===== -->
| Plot | Measure | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|------|---------|-----------|--------|--------|--------|--------|--------|---------------|
| **Log-return histogram** | MSE | 45.40 ± 57.91 | 3.579 | 6.973 | 153.7 | 4.778 | 57.94 | 0.1098 |
|  | % err | 33.41% ± 6.533% | 26.37% | 36.99% | 32.64% | 27.13% | 43.94% | 1.799% |
|  | NRMSE | 21.38% ± 14.34% | 8.194% | 12.08% | 44.34% | 10.17% | 32.12% | 0.5328% |
| **QQ plot** | MSE | 2.38e-06 ± 1.14e-06 | 1.41e-06 | 2.44e-06 | 2.71e-06 | 1.05e-06 | 4.29e-06 | 1.09e-09 |
|  | % err | 34.50% ± 11.22% | 20.51% | 23.70% | 47.99% | 34.21% | 46.12% | 0.4629% |
|  | NRMSE | 6.960% ± 1.738% | 5.488% | 7.318% | 7.627% | 4.709% | 9.660% | 0.1206% |
| **ACF \|r\| lags 1–20** | MSE | 0.003597 ± 0.003199 | 0.001026 | 0.001841 | 0.006211 | 3.79e-04 | 0.008526 | 9.61e-06 |
|  | % err | 186.2% ± 107.8% | 107.0% | 128.7% | 264.2% | 72.79% | 358.2% | 8.724% |
|  | NRMSE | 224.6% ± 123.4% | 116.9% | 181.2% | 346.7% | 85.32% | 392.9% | 6.071% |
| **ACF r² lags 1–20** | MSE | 0.001982 ± 0.001602 | 7.02e-04 | 0.001482 | 0.002501 | 3.90e-04 | 0.004835 | 9.17e-06 |
|  | % err | 130.0% ± 65.84% | 95.37% | 88.69% | 113.6% | 91.94% | 260.6% | 11.34% |
|  | NRMSE | 168.2% ± 70.21% | 94.68% | 157.1% | 224.2% | 94.36% | 270.8% | 6.486% |
| **Rolling vol histogram** | MSE | 150.2 ± 75.22 | 96.14 | 215.1 | 207.3 | 27.97 | 204.5 | 1.372 |
|  | % err | 56.76% ± 21.18% | 53.37% | 72.71% | 84.37% | 22.36% | 51.01% | 2.264% |
|  | NRMSE | 22.64% ± 7.203% | 19.01% | 28.61% | 27.91% | 10.07% | 27.61% | 0.8688% |
| **Tail survival** | MSE | 0.003912 ± 0.003064 | 9.04e-04 | 0.002042 | 0.007307 | 0.001354 | 0.007952 | 5.22e-07 |
|  | % err | 23.64% ± 6.097% | 18.46% | 24.62% | 25.99% | 15.89% | 33.26% | 0.3302% |
|  | NRMSE | 10.02% ± 4.365% | 5.257% | 7.902% | 14.94% | 6.434% | 15.59% | 0.1050% |

**Plot → curve mapping** (each curve is the shape whose funct/der/sec\_der are scored above):

| Plot | Key prefix | What the curve represents |
|------|-----------|--------------------------|
| Log-return histogram | `B_log_ret_hist_*` | Density of log-returns r=log(S_{t+1}/S_t) over shared bins |
| QQ plot              | `B_qq_plot_*`      | Quantile function at 100 uniform percentile levels |
| ACF \|r\| (lags 1–20) | `B_acf_abs_r_*`  | Mean per-path ACF of \|r\| at each lag |
| ACF r² (lags 1–20)  | `B_acf_sq_r_*`     | Mean per-path ACF of r² at each lag |
| Rolling vol hist.   | `B_roll_vol_hist_*` | Density of rolling-5 vol over shared bins |
| Tail survival       | `B_tail_surv_*`    | P(\|r\|>x) evaluated at thresholds of real \|r\| |

> Full formulas: [`metrics/README.md`](../../../metrics/README.md).

---

## Comparison with Yoon et al. NeurIPS 2019 (Table 2)

> ⚠️ **Not a direct comparison.** The paper evaluates on Sines (d=5, T=24) and Stocks (d=6)
> with a **2-layer LSTM** classifier. We evaluate on Heston (d=1, T=128) with GRU and MLP.

| Metric | Paper — Sines | Paper — Stocks | Ours — Heston GRU | Ours — Heston MLP |
|--------|:------------:|:-------------:|:-----------------:|:-----------------:|
| Disc Score ↓  | 0.011 ± 0.008 | 0.102 ± 0.021 | 0.033 ± 0.053 | 0.088 ± 0.047 |
| Pred Score ↓  | 0.093 ± 0.019 | 0.038 ± 0.001 | 0.053 ± 0.001 | 0.053 ± 0.001 |

Our GRU discriminative score (0.033) sits between the paper's Sines (0.011) and Stocks (0.102), and the MLP
judge (0.088) lands near the Stocks level — consistent with Heston being a moderately challenging 1-D
financial process for a GAN. Our predictive score (0.053) is below the paper's Sines result because Heston
is 1-D and next-step prediction is inherently simpler than 5-D Sines; it sits at the same irreducible
log-return floor every method reaches.

---

## Paper reproduction on Stocks (our port vs Yoon et al. Table 1)

Before running TimeGAN on Heston we reproduced the **original TimeGAN paper result on the
Stocks dataset** with the same PyTorch port, using the paper's own hyperparameters
(seq_len 24, hidden 24, 3 layers, 50 000 iterations/phase, batch 128). This validates the
generator port independently of Heston. Full write-up:
[`../../../methods/TimeGAN/paper_reimplementation/`](../../../methods/TimeGAN/paper_reimplementation/).

| Dataset | Metric | Ours — 2-layer judge, 1 seed | Ours — 1-layer judge, 5 seeds | Paper (Table 1) | Verdict |
|---------|--------|:----------------------------:|:-----------------------------:|:---------------:|---------|
| Stocks | Discriminative ↓ | 0.219 ± 0.066 | **0.119 ± 0.036** | 0.102 ± 0.031 | **matches** ✓ (within 0.5σ) |
| Stocks | Predictive ↓ | 0.039 ± 0.000 | **0.042 ± 0.002** | 0.038 ± 0.001 | **matches** ✓ |

Once the discriminative judge is matched to the paper's own depth (1-layer GRU, hidden = ⌊d/2⌋)
across 5 training seeds, the score drops from 0.219 → **0.119 ± 0.036**, overlapping the paper's
**0.102 ± 0.031** (gap < 0.5σ). The predictive score matched all along (0.042 vs 0.038). The
original 2× discrepancy was an artefact of the scoring judge's depth, not the generator.

---

## Path Shadowing MC (arXiv:2308.01486)

Model-agnostic PS-MC forecast: embed each real prefix (steps 0–63) as a 65D murex-style feature vector,
retrieve K nearest TimeGAN paths by L2 in z-scored space, forecast with their price-anchored futures.
Full analysis: [`path_shadowing/README.md`](path_shadowing/README.md).

<!-- ===== PER-METHOD PS-MC TABLE ===== -->
| Metric | Value (mean ± std) | RW baseline |
|--------|--------------------|-------------|
| PS-MC CRPS H=32 ↓ | 3.085 ± 0.3332 | 3.738 |
| PS-MC CRPS H=64 ↓ | 4.337 ± 0.4329 | 5.246 |

PS-MC still **beats the naive RW on CRPS** at both horizons (3.085 < 3.738 at H=32; 4.337 < 5.246 at H=64),
but by the narrowest margin of any generator that beats RW — TimeGAN's unstable, seed-dependent pool yields
the loosest nearest-neighbour shadowing set among the models that clear the baseline (only TimeVAE and
COSCI-GAN fail RW outright). It trails the diffusion and state-space methods (LS4 2.704 / 3.763,
CSDI 2.718 / 3.776) by a wide gap.

---

## Files

| File | Description |
|------|-------------|
| `metrics_summary.csv` | Mean ± std across 5 seeds for all metrics |
| `seed_{i}_metrics.json` | Full per-seed metric dict |
| `curve_b_aggregate.json` | B three-subline aggregates (MSE + % err + NRMSE) |
| `seed_{i}_disc_gru_loss.csv` | GRU discriminator BCE loss per training step (every 50 steps) |
| `seed_{i}_disc_mlp_loss.csv` | MLP discriminator BCE loss per training step (every 50 steps) |
| `seed_{i}_pred_gru_loss.csv` | GRU predictor MAE loss per training step (every 100 steps) |
| `seed_{i}_pred_mlp_loss.csv` | MLP predictor MAE loss per training step (every 100 steps) |
| `plots/seed_{i}_pca.png` | PCA 2-D projection, real vs fake |
| `plots/seed_{i}_tsne.png` | t-SNE 2-D projection, real vs fake |
| `plots/disc_classifier_loss.png` | All-seed discriminator training loss (GRU + MLP) |
| `plots/pred_score_loss.png` | All-seed predictor training loss (GRU + MLP) |
| `plots/heston_diagnostics.png` | 8-panel stylised facts diagnostic (seed 0) |
| `path_shadowing/` | Path-shadowing MC forecasts |

→ Cross-method comparison with SBTS: [`results/README.md`](../../README.md)
