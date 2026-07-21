# Metrics — LS4 on Heston (5 Seeds)

**Dataset:** 8 192 Heston price paths, seq\_len = 128.
Parameters: μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250.

**Model:** LS4 — Latent State-Space (Zhou, Poli, Xu, Massaroli, Ermon, **ICML 2023**,
*Deep Latent State Space Models for Time-Series Generation*, arXiv:2212.12749,
`github.com/alexzhou907/ls4`). VAE-style latent generative model: an S4 structured-state-space
**prior** rolls a continuous latent `z` forward in time, an S4 **posterior** infers `z` from data,
and an S4 **decoder** reconstructs observations (`latent_type = split`, 2 146 857 params). Trained on
the ELBO with the released `solar_weekly` preset (`z_dim` = 5, `d_model` = 64, `d_state` = 64,
`n_layers` = 4, `backbone` = autoreg, `sigma` = 0.1), AdamW + ReduceLROnPlateau + EMA(0.99). Heston
prices are globally standardised `(X−μ)/σ` (μ = 101.325, σ = 9.972) for training and decoded back. See
[`../../../methods/LS4/code/README.md`](../../../methods/LS4/code/README.md).

> ⚠️ **Cauchy fix (REQUIRED).** LS4's `model.generate` rolls the S4 latent prior with `latent.step`
> (STEP-mode recurrence). On a CUDA-13 A100 the fast keops/CUDA Cauchy kernels are unavailable, so S4
> falls back to the naive Python Cauchy kernel, which as-shipped sums over the *full* pole set instead of
> over **conjugate pole pairs**. The one-line fix at `code/reference/models/s4.py:795` restores the
> conjugate-pair sum. Without it, generation degenerates (Solar-Weekly marginal frozen at 0.197 vs paper
> 0.046 — see [§ Comparison with the paper](#comparison-with-the-paper-solar-weekly-monash-reproduction)).
> This is the **only** change to reference model code.

**Convention:** lower is better for all metrics **except A33 Teacher-Sigma Corr ↑**. A28 Kurtosis Ratio: perfect = 1.0.

---

## Results (mean ± std across 5 seeds)

### A1–A34 — Metrics by category

Last column = **Perfect floor**: the reproducible best-case a perfect generator reaches with finite
samples, from a row-shuffled copy of the real data (see
[`../../../methods/perfect_recovery/`](../../../methods/perfect_recovery/)). Most floors are 0 because a
permutation preserves every column-wise marginal; the residual non-zero floors are pure finite-sample
noise, and are **identical across methods** (same real data, same permutation).

| ID | Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|----|--------|-----------|--------|--------|--------|--------|--------|---------------|
| | **— Fat Tail —** | | | | | | | |
| A1 | Kurtosis Error | 0.3680 ± 0.0161 | 0.3897 | 0.3716 | 0.3779 | 0.3573 | 0.3435 | 0 |
| A2 | \|r\| q95 Error | 3.30e-4 ± 1.13e-4 | 4.43e-4 | 1.61e-4 | 4.63e-4 | 2.63e-4 | 3.19e-4 | 0 |
| A3 | \|r\| q99 Error | 0.0011 ± 1.66e-4 | 0.0013 | 0.0010 | 0.0014 | 9.11e-4 | 0.0012 | 0 |
| A4 | Tail QQ Error | 3.73e-4 ± 7.30e-5 | 4.33e-4 | 2.51e-4 | 4.58e-4 | 3.86e-4 | 3.40e-4 | 0 |
| A5 | Hill Tail Index Error | 1.5639 ± 0.4825 | 2.4230 | 0.9996 | 1.6869 | 1.3047 | 1.4055 | 0 |
| | **— Distribution —** | | | | | | | |
| A6 | Path MMD² | 0.0017 ± 2.16e-4 | 0.0018 | 0.0020 | 0.0018 | 0.0015 | 0.0015 | 0.0015 |
| A7 | Terminal MMD² | 0.0018 ± 7.22e-4 | 0.0013 | 0.0030 | 0.0019 | 0.0020 | 8.48e-4 | 0.0016 |
| A8 | Increment MMD² | 9.17e-4 ± 2.50e-5 | 9.51e-4 | 8.86e-4 | 9.25e-4 | 8.91e-4 | 9.35e-4 | 7.45e-4 |
| A9 | Volatility MMD | 0.0152 ± 0.0015 | 0.0165 | 0.0148 | 0.0173 | 0.0141 | 0.0133 | 0.0071 |
| A10 | Terminal SWD | 0.8701 ± 0.2595 | 0.7124 | 1.1211 | 1.2127 | 0.7869 | 0.5172 | 0.6873 |
| A11 | Path SWD | 0.4728 ± 0.0725 | 0.5873 | 0.4713 | 0.5123 | 0.3973 | 0.3957 | 0.4381 |
| A12 | RV Law Loss | 0.2324 ± 0.0163 | 0.2421 | 0.2172 | 0.2568 | 0.2122 | 0.2337 | 0 |
| A13 | Mean Path RMSE | 0.1204 ± 0.0983 | 0.3063 | 0.0549 | 0.0522 | 0.1357 | 0.0528 | 0 |
| A14 | KS Log-returns | 0.0123 ± 6.65e-4 | 0.0130 | 0.0128 | 0.0113 | 0.0127 | 0.0117 | 0 |
| A15 | Skewness Error | 0.0318 ± 0.0168 | 0.0323 | 0.0241 | 0.0109 | 0.0621 | 0.0295 | 0 |
| A16 | QQ RMSE (300-pt) | 3.43e-4 ± 1.20e-5 | 3.38e-4 | 3.57e-4 | 3.36e-4 | 3.58e-4 | 3.29e-4 | 0 |
| A17 | Terminal Price KS | 0.0142 ± 0.0030 | 0.0194 | 0.0129 | 0.0123 | 0.0154 | 0.0110 | 0 |
| | **— Adversarial —** | | | | | | | |
| A18 GRU | Discriminative Score GRU | 0.0077 ± 0.0054 | 0.0014 | 0.0069 | 0.0035 | 0.0099 | 0.0169 | 0.0042 |
| A18 MLP | Discriminative Score MLP | 0.0096 ± 0.0074 | 0.0053 | 0.0163 | 0.0047 | 0.0203 | 0.0011 | 0.0067 |
| | **— Predictive —** | | | | | | | |
| A19 GRU | Predictive Score GRU | 0.0537 ± 1.10e-5 | 0.0537 | 0.0537 | 0.0537 | 0.0537 | 0.0537 | 0.0537 |
| A19 MLP | Predictive Score MLP | 0.0540 ± 3.61e-4 | 0.0538 | 0.0545 | 0.0539 | 0.0544 | 0.0536 | 0.0539 |
| | **— Temporal —** | | | | | | | |
| A20 | Covariance Error | 8.5453 ± 5.4272 | 14.4896 | 2.3088 | 15.6136 | 4.9084 | 5.4060 | 0 |
| A21 | ACF \|r\| Error (lags) | 0.0155 ± 0.0018 | 0.0176 | 0.0151 | 0.0176 | 0.0137 | 0.0135 | 0 |
| A22 | ACF r² Error (lags) | 0.0097 ± 0.0017 | 0.0117 | 0.0095 | 0.0117 | 0.0079 | 0.0077 | 0 |
| A23 | ACF \|r\| Lag-1 Error | 0.0211 ± 0.0055 | 0.0298 | 0.0191 | 0.0249 | 0.0168 | 0.0148 | 0 |
| A24 | ACF r² Lag-1 Error | 0.0132 ± 0.0053 | 0.0220 | 0.0114 | 0.0159 | 0.0096 | 0.0069 | 0 |
| | **— Vol —** | | | | | | | |
| A25 | Mean RMSE | 0.2449 ± 0.1755 | 0.5559 | 0.1174 | 0.1373 | 0.3224 | 0.0917 | 0 |
| A26 | Return Std Error | 0.0054 ± 0.0040 | 0.0021 | 0.0117 | 2.74e-4 | 0.0071 | 0.0060 | 0 |
| A27 | Log-Return Std Error | 5.20e-5 ± 3.50e-5 | 8.98e-6 | 1.05e-4 | 3.03e-5 | 7.85e-5 | 3.87e-5 | 0 |
| A28 | Kurtosis Ratio | 1.5347 ± 0.0769 | 1.4795 | 1.5321 | 1.6839 | 1.4945 | 1.4834 | 1.000 |
| A29 | Sigma Mean Error | 0.0018 ± 6.99e-4 | 0.0011 | 0.0028 | 9.52e-4 | 0.0024 | 0.0018 | 0 |
| A30 | Cross-Sect. Vol Path RMSE | 0.1640 ± 0.0759 | 0.2330 | 0.0947 | 0.2750 | 0.1287 | 0.0887 | 0 |
| A31 | Rolling Vol KS (w=5) | 0.0398 ± 0.0014 | 0.0408 | 0.0410 | 0.0407 | 0.0390 | 0.0373 | 0 |
| A32 | Vol-of-Vol Error | 3.20e-4 ± 4.20e-5 | 3.47e-4 | 2.86e-4 | 3.89e-4 | 2.78e-4 | 2.98e-4 | 0 |
| | **— Heston Spec —** | | | | | | | |
| A33 | Teacher-Sigma Corr ↑ | -0.0022 ± 0.0041 | 0.0024 | -0.0066 | 0.0027 | -0.0065 | -0.0028 | 0.6143 |
| A34 | Teacher-Sigma RMSE | 0.0951 ± 6.95e-4 | 0.0945 | 0.0957 | 0.0941 | 0.0958 | 0.0954 | 0.0654 |

**Reading the table — best-in-class density and path fidelity.** LS4 is the **strongest
density- and path-matcher in this benchmark**, sitting at or near the perfect-recovery floor on the
kernel, quantile, adversarial and predictive metrics.

The strengths. On the hard distributional metrics LS4 is essentially **at the perfect floor**:
A6 Path MMD² **0.0017** (floor 0.0015), A8 Increment MMD² **9.17e-4** (floor 7.45e-4), A11 Path SWD
**0.4728** (floor 0.4381), A16 QQ RMSE **3.43e-4**, A14 KS **0.0123**. Adversarially, **A18 = 0.0077
(GRU) / 0.0096 (MLP)** — the discriminative score sits at chance, right at the floor (0.0042 / 0.0067):
a modern sequence classifier **cannot separate** LS4 paths from real Heston paths (the exact opposite of
COSCI-GAN's A18 = 0.50). The predictive score **A19 GRU = 0.0537 equals the perfect-recovery floor to
four decimals** — models trained on LS4 output forecast real Heston as well as models trained on real
data. Scalar moments lead the field too: **A1 Kurtosis Error 0.368** (best of any method),
**A15 Skewness Error 0.032**, **A5 Hill 1.56**.

The two honest caveats. **A28 Kurtosis Ratio 1.53** (perfect 1.0, stable across seeds): the generated
log-returns are mildly *platykurtic* — the marginal centre is excellent (A1/A14/QQ top-tier) but the
extreme tail is slightly under-heavy. And as with **every** single-factor generator here,
**A33 teacher-sigma correlation ≈ −0.002** (perfect 0.614) with A34 RMSE 0.095: LS4 reproduces the
*marginal* and *rolling* volatility distributions (A9 0.015, A31 0.040) superbly but does not reconstruct
the *path-wise* stochastic-vol trajectory — the latent variance is unrecoverable from prices alone.
Net: **the new best-in-class method** on density/path/adversarial/predictive fidelity, with slightly
thin tails and no per-path latent-vol structure.

---

## Stylised Facts Diagnostic (Heston vs LS4, seed 0)

Eight-panel comparison matching the Murex paper (Fig. 1 style): sample paths, return distribution,
QQ plot, ACF of |returns|, ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

![Heston Diagnostics](plots/heston_diagnostics.png)

---

## Curve-shape metrics (B) — mean ± std across 5 seeds

Each of the 6 diagnostic plots above yields a **curve** L (a list of values), not a scalar. For each plot
we build three lists — the curve L, its first finite difference L' (der), and its second finite difference
L'' (sec\_der) — then combine the three sub-scores into **one number per plot** under two error measures:

- **MSE row**: for each list, dᵢ = mean((L_real − L_gen)²). Combined mean = sum of the three seed-means;
  combined std = sqrt(std\_funct² + std\_der² + std\_sec\_der²) (quadrature).
- **% err row**: for each list, dᵢ = mean(|L_gen − L_real| / (|L_real| + 1e-6)) × 100, a proper MAPE — one
  division: the **function-level MAPE on the curve L itself** — the derivative / 2nd-derivative MAPE is
  **excluded** (near-zero true diffs make it explode). Combined mean/std = mean and sample std across the 5 seeds.

↓ lower is better for both rows. **Perfect floor = 0** for every plot (row-shuffle preserves all marginals).
**Best-in-class curves.** LS4 posts the **best log-return-histogram MSE (1.43)** and the **best
rolling-vol-histogram MSE (26.5)** of the entire benchmark, and near-perfect QQ (MSE 1.41e-7). Its
full-density curves confirm the A-table story: the marginal return density and the rolling-volatility
distribution both lie essentially on top of Heston's.

| Plot | Measure | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|------|---------|-----------|--------|--------|--------|--------|--------|---------------|
| **Log-return histogram** | MSE | 1.4285 ± 0.0817 | 1.3367 | 1.4889 | 1.5410 | 1.4579 | 1.3183 | 0 |
| | % err | 5.346% ± 0.168% | 5.277% | 5.153% | 5.444% | 5.623% | 5.233% | 0 |
| **QQ plot** | MSE | 1.41e-7 ± 6.73e-9 | 1.38e-7 | 1.48e-7 | 1.40e-7 | 1.46e-7 | 1.31e-7 | 0 |
| | % err | 5.994% ± 0.595% | 6.647% | 6.444% | 5.317% | 6.326% | 5.237% | 0 |
| **ACF \|r\| lags 1–20** | MSE | 1.92e-4 ± 2.77e-5 | 2.45e-4 | 2.07e-4 | 2.04e-4 | 1.56e-4 | 1.49e-4 | 0 |
| | % err | 42.327% ± 2.740% | 43.688% | 45.710% | 43.866% | 40.219% | 38.153% | 0 |
| **ACF r² lags 1–20** | MSE | 8.90e-5 ± 1.70e-5 | 1.23e-4 | 1.04e-4 | 9.18e-5 | 6.31e-5 | 6.46e-5 | 0 |
| | % err | 32.247% ± 2.974% | 32.730% | 36.014% | 34.653% | 29.886% | 27.952% | 0 |
| **Rolling vol histogram** | MSE | 26.5199 ± 2.6360 | 29.1176 | 26.1623 | 29.9427 | 23.3211 | 24.0557 | 0 |
| | % err | 11.577% ± 1.144% | 12.570% | 10.797% | 13.206% | 10.106% | 11.205% | 0 |
| **Tail survival** | MSE | 2.16e-4 ± 2.50e-5 | 1.92e-4 | 2.62e-4 | 1.99e-4 | 2.23e-4 | 2.04e-4 | 0 |
| | % err | 3.388% ± 0.129% | 3.287% | 3.633% | 3.377% | 3.365% | 3.275% | 0 |

**Plot → curve mapping** (each curve is the shape whose funct/der/sec\_der are scored above):

| Plot | Key prefix | What the curve represents |
|------|-----------|--------------------------|
| Log-return histogram | `B_log_ret_hist_*` | Density of log-returns r=log(S_{t+1}/S_t) over shared bins |
| QQ plot              | `B_qq_plot_*`      | Quantile function at 100 uniform percentile levels |
| ACF \|r\| (lags 1–20) | `B_acf_abs_r_*`  | Mean per-path ACF of \|r\| at each lag |
| ACF r² (lags 1–20)  | `B_acf_sq_r_*`     | Mean per-path ACF of r² at each lag |
| Rolling vol hist.   | `B_roll_vol_hist_*` | Density of rolling-5 vol over shared bins |
| Tail survival       | `B_tail_surv_*`    | P(\|r\|>x) evaluated at thresholds of real \|r\| |

> The ACF %err (42% / 32%) is a near-zero-denominator artefact: the true ACF ≈ 0.05, so a small absolute
> error (MSE ~1.9e-4) becomes a large *relative* error. Read MSE for absolute agreement, %err for shape.
> Full formulas: [`metrics/README.md`](../../../metrics/README.md).

---

## Discriminative & Predictive Classifier Losses (A18 / A19)

BCE loss during GRU/MLP discriminator training (A18) and MAE loss during GRU/MLP predictor training on
*synthetic* data (A19, TSTR), 5 seeds. A discriminator BCE near ln(2) ≈ 0.693 means real and generated
are indistinguishable — **exactly what we see here.** LS4's judge BCE **stays near ln 2**, the direct
evidence that A18 ≈ 0.008 reflects genuinely inseparable paths.

![Discriminative Classifier Loss](plots/disc_classifier_loss.png)

![Predictive Score Loss](plots/pred_score_loss.png)

---

## Comparison with the paper (Solar-Weekly Monash reproduction)

> ⚠️ **LS4's paper metrics are a generative-quality protocol on a different dataset.** LS4's ICML 2023
> Monash evaluation (Table 1) reports three post-hoc scores — **marginal**, **classification** and
> **prediction** — on the Monash time-series benchmarks (GT-GAN / TimeGAN protocol), not on Heston. There
> is therefore no native "Ours — Heston" entry for the paper metric; we validated our LS4 port **on the
> paper's own Solar-Weekly dataset** before running the Heston experiment above.

| Dataset | Metric | Ours (Cauchy-fixed, 5 resamples) | Ours (as-shipped naive) | Paper (Table 1) | Verdict |
|---------|--------|:--------------------------------:|:-----------------------:|:---------------:|---------|
| Solar Weekly | Marginal ↓ | **0.047 ± 0.003** (best 0.044) | 0.197 ± 0.003 | 0.0459 | **matches** ✓ |
| Solar Weekly | Classification ↑ | **0.717 ± 0.097** (best 0.771) | 0.001 ± 0.001 | 0.683 | same regime ✓ |
| Solar Weekly | Prediction ↓ | **0.113 ± 0.036** (best 0.076) | 0.624 ± 0.078 | 0.141 | **matches** ✓ |

With the Cauchy fix, our LS4 port reaches marginal **0.047** (paper 0.0459) and prediction **0.113**
(paper 0.141), squarely in the paper regime, and classification **0.717** meets/exceeds the paper's
0.683. The as-shipped naive-Cauchy column is the **broken** run (marginal frozen at 0.197, classifier
trivially separating the degenerate samples) — the signature of the mis-summed generation-time recurrence
the fix corrects, not a hyperparameter issue. This confirms the port reproduces LS4's generative quality
faithfully; the same fix is carried into the Heston generator above. Numbers read from the on-disk
training logs (`code/reference/outputs_shipped_fix_train.log` ep 32 000;
`outputs_shipped_train.log` ep 25 400). Full write-up:
[`../../../methods/LS4/paper_reimplementation/`](../../../methods/LS4/paper_reimplementation/).

---

## Path Shadowing MC (arXiv:2308.01486)

Model-agnostic PS-MC forecast: embed each real prefix (steps 0–63) as a 65D murex-style feature vector,
retrieve K=77 nearest LS4 paths by L2 in z-scored space, forecast with their price-anchored futures.

| Metric | H=32 Uniform | H=32 Gaussian | H=64 Uniform | H=64 Gaussian | Naive RW |
|--------|:------------:|:-------------:|:------------:|:-------------:|:--------:|
| **CRPS** | 2.7007 ± 0.0034 | 2.7007 ± 0.0034 | 3.7999 ± 0.0055 | 3.7999 ± 0.0055 | 3.73 / 5.30 |
| MAE    | 3.7177 ± 0.0046 | 3.7177 ± 0.0047 | 5.2484 ± 0.0078 | 5.2484 ± 0.0078 | 3.73 / 5.30 |
| RMSE   | 5.0700 ± 0.0040 | 5.0700 ± 0.0039 | 7.1600 ± 0.0056 | 7.1600 ± 0.0055 | 5.07 / 7.18 |

PS-MC over the LS4 pool **beats the naive random walk on CRPS at both horizons** (2.70 < 3.73 at H=32;
3.80 < 5.30 at H=64) — the **largest CRPS margin of any method in the benchmark** — and also edges the RW
on MAE (3.72 < 3.73; 5.25 < 5.30) and ties/beats it on RMSE (5.07 ≈ 5.07; 7.16 < 7.18). Because LS4's
generated pool matches Heston's density and path structure closely (floor-level A18, A6, A11), its
nearest-neighbour futures form a well-calibrated ensemble. Uniform ≈ Gaussian: Heston is time-homogeneous.
Full breakdown: [`path_shadowing/README.md`](path_shadowing/README.md).

---

## Files

| File | Description |
|------|-------------|
| `metrics_summary.csv` | Mean ± std across 5 seeds for all metrics |
| `seed_{i}_metrics.json` | Full per-seed metric dict |
| `curve_b_aggregate.json` | B two-subline aggregates (MSE + % err) |
| `seed_{i}_disc_gru_loss.csv` | GRU discriminator BCE loss per training step |
| `seed_{i}_disc_mlp_loss.csv` | MLP discriminator BCE loss per training step |
| `seed_{i}_pred_gru_loss.csv` | GRU predictor MAE loss per training step |
| `seed_{i}_pred_mlp_loss.csv` | MLP predictor MAE loss per training step |
| `plots/seed_{i}_pca.png` | PCA 2-D projection, real vs fake |
| `plots/seed_{i}_tsne.png` | t-SNE 2-D projection, real vs fake |
| `plots/disc_classifier_loss.png` | All-seed discriminator training loss (GRU + MLP) |
| `plots/pred_score_loss.png` | All-seed predictor training loss (GRU + MLP) |
| `plots/heston_diagnostics.png` | 8-panel stylised facts diagnostic (seed 0) |
| `path_shadowing/` | Path-shadowing MC forecasts (summary.json + per-seed + plots + README) |

→ Cross-method comparison with TimeGAN, SBTS, Fourier Flow, Diffusion-TS, CSDI, TimeVAE, TimeVQVAE & COSCI-GAN: [`results/README.md`](../../README.md)
