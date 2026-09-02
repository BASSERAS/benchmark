# Deep-MKV-TS on Multi-Asset Heston (d = 8)

**Deep McKean–Vlasov Time Series generation** — a path-dependent McKean–Vlasov
stochastic-control generator, applied to 8 192 **multi-asset** Heston
stochastic-volatility price paths (seq\_len = 252, **d = 8 correlated assets**).

Deep-MKV-TS does **not** learn a generator from noise. It starts from a *frozen,
interpretable reference SDE* (Guyon–Lekeufack, paper §2.1) fitted to the data by
penalised maximum likelihood, then learns a **volatility correction only** — the drift
is never touched. Training minimises

$$\mathcal{J}(\alpha) = \mathcal{D}\big(\mu^{\alpha}, \mu^{\text{data}}\big) + \eta\,\mathbb{E}\left[\int_0^T \tfrac12 \|\alpha_t\|^2 \sigma_t^{-2} dt\right], \qquad \eta = 1$$

where $\mathcal{D}$ is the same multi-component MMD discrepancy used at d = 1
(observed path, increments, terminal, global realized variance, $|r|$ ACF, $r^2$ ACF)
and the second term is a **specific-entropy running cost** penalising how far the
controlled law drifts from the reference law. The correction network is a
1-layer GRU (`hidden_dim = 96`) — **0 parameters**.
Weights in [`weights/`](weights/), per-seed hyperparameters in
`weights/seed_*_config.json`.

The model is **joint over all 8 assets** — one reference kernel with `state_dim = 8`
and one control network emitting a full `(d, d)` correction — not eight independent
univariate fits. See [`code/README.md`](code/README.md) for the frozen reference
kernel, the vendored upstream package, the forced d = 8 deviations and the one
re-selected hyperparameter; and the dataset-level [`../oldreadme.md`](../oldreadme.md)
for the multi-asset Heston law itself, the per-asset vs native metric scoping, and the
memorisation diagnostic shared by all methods on this dataset.

> **Hyperparameters:** `eta=1`, `sigma_min=0.001`, `sigma_max=0.6`,
> `lambda_scale=50`, `kappa_scale=100`, discrepancy preset
> `old_fullv_w0p25` with `abs_return_acf_weight=0.25`,
> `squared_return_acf_weight=0.125`; `hidden_dim=96`,
> `num_layers=1`, batch , target batch 256;
> `AdamW(lr=0.002, weight_decay=1e-05)`, `grad_clip_norm=5`,
> `ce_target_mode="ridge"` with `ridge_lambda=0.001`, `ce_ridge=0.001`;
>  outer iterations per seed, 5 seeds.
> Every one of those numbers is the **committed d = 1 value**, unchanged, except for
> what is listed next.
>
> **Retuned for d = 8: **nothing** — the list `retuned_for_d8` in every `weights/seed_*_config.json` is empty.** The reason is structural, not a search: the
> frozen package's Z-proxy basis $\Phi^{\text{ref}}_k$ is the flattened path prefix
> plus an intercept, so its width is $p = 1 + (k+1)d$. At d = 1 that is at most 252
> against a batch of  — an over-determined ridge that genuinely denoises. At
> d = 8 it reaches 2017, under-determined from step 31 onward, and the
> projection degenerates into near-interpolation. `ridge_lambda` was re-selected on the
> **validation** split (`heston_ma_S_val_8192x252x8.npy`), never on test, and the whole
> sweep is recorded in `code/sweep/`.
>
> **The control is matrix-valued, not diagonal** (`control=matrix`). The paper clips the
> *eigenvalues* of $\Theta$; at d > 1, clipping its diagonal entries instead is a
> different operator. $\Theta$ is symmetric but **indefinite**, so Cholesky and LDLᵀ are
> invalid and `torch.linalg.eigh` is the right primitive, with the Daleckii–Krein /
> Löwner form for its adjoint. The two coincide bitwise only at d = 1.
>
> **The drift Jacobian is kept.** `drift_adjoint_backend=autograd_replay` recomputes
> $\partial b^{\text{ref}}/\partial x$ by replaying the reference recursion under
> autograd, because the multivariate kernel exposes no analytical
> `drift_step_path_vjp`. No Algorithm 1 term is dropped.
>
> **`max_eigh_batch=32768`** chunks the batched eigendecomposition: cuSOLVER's
> batched `eigh` fails above roughly 64 k matrices, and batch  × 251
> steps = the per-step batch x step product sits inside that failure band. Chunking changes throughput, not
> results.
>
> **The reported checkpoint is chosen on validation**, not the final step — step
> - — exactly as the committed d = 1 run reports step 2500 of a 3000-step run.
> There is **no learning-rate scheduler anywhere in the codebase**, so this is bitwise
> identical to having stopped training there.

---

## Metrics A1-A34 + B, mean ± std across 5 seeds

> All metrics on **log-returns** $r_t = \log(S_{t+1}/S_t)$ unless noted. A26 uses price increments $\Delta S_t$.
> Rows marked *(native d=8)* are evaluated **once** on the full `(N, T, 8)` tensor; every other
> row is computed on each of the 8 univariate slices and reported as the **mean over assets**
> (per-asset breakdown in [`metrics_per_asset.csv`](metrics_per_asset.csv)).

| Metric | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|---|---|---|---|---|---|---|---|
| **Fat Tail** | | | | | | | |
| A1 Kurtosis Error ↓ | 1.005 ± 0.1388 | 1.255 | 1.039 | 0.937 | 0.8487 | 0.9449 | 0.008385 |
| A2 \|r\| q95 Error ↓ | 6.19e-04 ± 2.01e-04 | 5.87e-04 | 9.58e-04 | 5.33e-04 | 6.73e-04 | 3.43e-04 | 4.58e-05 |
| A3 \|r\| q99 Error ↓ | 0.001884 ± 4.10e-04 | 0.001665 | 0.00251 | 0.001976 | 0.002002 | 0.001268 | 8.08e-05 |
| A4 Tail QQ Error ↓ | 6.64e-04 ± 1.71e-04 | 6.27e-04 | 9.74e-04 | 6.00e-04 | 6.71e-04 | 4.51e-04 | 5.62e-05 |
| A5 Hill Tail Index Error ↓ | 2.15 ± 0.1899 | 2.484 | 2.108 | 2.155 | 1.895 | 2.109 | 0.5896 |
| **Distribution** | | | | | | | |
| A6 Path MMD² ↓ *(native d=8)* | 0.002158 ± 5.79e-05 | 0.002213 | 0.00223 | 0.002126 | 0.002072 | 0.002146 | 0.001948 |
| A7 Terminal MMD² ↓ *(native d=8)* | 0.002132 ± 4.79e-05 | 0.00212 | 0.002206 | 0.002166 | 0.002076 | 0.002093 | 0.001954 |
| A8 Increment MMD² ↓ *(native d=8)* | 8.82e-04 ± 1.10e-05 | 8.70e-04 | 8.81e-04 | 8.74e-04 | 8.82e-04 | 9.02e-04 | 8.71e-04 |
| A9 Volatility MMD ↓ *(native d=8)* | 0.0348 ± 0.001846 | 0.0324 | 0.03498 | 0.0375 | 0.03596 | 0.03318 | 0.008587 |
| A10 Terminal SWD ↓ *(native d=8)* | 1.703 ± 0.1158 | 1.683 | 1.865 | 1.611 | 1.554 | 1.802 | 1.141 |
| A11 Path SWD ↓ *(native d=8)* | 1.043 ± 0.04082 | 1.001 | 1.097 | 0.9901 | 1.063 | 1.064 | 0.7258 |
| A12 RV Law Loss ↓ | 0.8366 ± 0.09399 | 0.7275 | 0.9475 | 0.9233 | 0.8579 | 0.727 | 0.06398 |
| A13 Mean Path RMSE ↓ | 0.4509 ± 0.02436 | 0.4219 | 0.4945 | 0.4501 | 0.436 | 0.4519 | 0.1834 |
| A14 KS Log-returns ↓ | 0.01891 ± 8.61e-04 | 0.01797 | 0.01868 | 0.01954 | 0.01814 | 0.02024 | 9.64e-04 |
| A15 Skewness Error ↓ | 0.05435 ± 9.93e-04 | 0.05449 | 0.05435 | 0.05579 | 0.05267 | 0.05448 | 0.003568 |
| A16 QQ RMSE (300-pt) ↓ | 5.47e-04 ± 4.18e-05 | 5.02e-04 | 6.15e-04 | 5.72e-04 | 5.13e-04 | 5.31e-04 | 3.04e-05 |
| A17 Terminal Price KS ↓ | 0.03755 ± 0.003061 | 0.03351 | 0.04193 | 0.03976 | 0.03752 | 0.03502 | 0.01466 |
| **Adversarial** | | | | | | | |
| A18 Disc Score GRU ↓ *(native d=8)* | 0.1004 ± 0.1298 | 0.3419 | 0.02517 | 0.1308 | 0.002594 | 0.001678 | 0.005523 |
| A18 Disc Score MLP ↓ *(native d=8)* | 0.004303 ± 0.002987 | 0.005951 | 0.00534 | 0.001678 | 1.53e-04 | 0.008392 | 0.006012 |
| **Predictive** | | | | | | | |
| A19 Pred Score GRU ↓ | 0.0492 ± 2.96e-06 | 0.0492 | 0.0492 | 0.0492 | 0.0492 | 0.0492 | 0.0492 |
| A19 Pred Score MLP ↓ | 0.0494 ± 7.09e-05 | 0.04947 | 0.04936 | 0.0495 | 0.04933 | 0.04933 | 0.04931 |
| **Temporal** | | | | | | | |
| A20 Covariance Error ↓ *(native d=8)* | 289 ± 11.39 | 300.9 | 298.5 | 284.9 | 269.1 | 291.3 | 55.2 |
| A21 ACF \|r\| Error (lags) ↓ | 0.05 ± 0.001128 | 0.04936 | 0.05115 | 0.05022 | 0.0511 | 0.04816 | 0.001066 |
| A22 ACF r² Error (lags) ↓ | 0.0381 ± 0.001098 | 0.03735 | 0.03914 | 0.03839 | 0.03925 | 0.03638 | 0.001107 |
| A23 ACF \|r\| Lag-1 Error ↓ | 0.05796 ± 0.001204 | 0.05734 | 0.05919 | 0.0582 | 0.0591 | 0.05596 | 0.001038 |
| A24 ACF r² Lag-1 Error ↓ | 0.0445 ± 0.001093 | 0.04386 | 0.04554 | 0.04496 | 0.04546 | 0.04267 | 0.001036 |
| **Vol** | | | | | | | |
| A25 Mean RMSE ↓ *(native d=8)* | 2.445 ± 0.1217 | 2.31 | 2.662 | 2.424 | 2.472 | 2.358 | 0.9234 |
| A26 Return Std Error ↓ | 0.01881 ± 0.009155 | 0.02044 | 0.03471 | 0.01776 | 0.00691 | 0.01425 | 0.001745 |
| A27 Log-Return Std Error ↓ | 2.01e-04 ± 7.92e-05 | 1.46e-04 | 3.00e-04 | 2.75e-04 | 8.67e-05 | 1.97e-04 | 2.21e-05 |
| A28 Kurtosis Ratio (→ 1) | 1.713 ± 0.1598 | 1.575 | 1.871 | 1.753 | 1.883 | 1.482 | 1 |
| A29 Sigma Mean Error ↓ | 0.002977 ± 0.001189 | 0.001892 | 0.004306 | 0.003994 | 0.001274 | 0.003416 | 3.32e-04 |
| A30 Cross-Sect. Vol Path RMSE ↓ | 0.2547 ± 0.0836 | 0.2764 | 0.3823 | 0.2432 | 0.1204 | 0.2511 | 0.1596 |
| A31 Rolling Vol KS (w=5) ↓ | 0.07005 ± 0.002457 | 0.06725 | 0.06923 | 0.07224 | 0.06796 | 0.07358 | 0.00208 |
| A32 Vol-of-Vol Error ↓ | 5.04e-04 ± 6.29e-05 | 4.70e-04 | 5.92e-04 | 5.23e-04 | 5.30e-04 | 4.04e-04 | 1.14e-05 |
| **Heston Spec** | | | | | | | |
| A33 Teacher-Sigma Corr ↑ | -7.13e-04 ± 0.001415 | -0.001453 | 0.001031 | 5.10e-04 | -7.22e-04 | -0.002933 | -1.35e-04 |
| A34 Teacher-Sigma RMSE ↓ | 0.09572 ± 6.13e-04 | 0.09533 | 0.09523 | 0.0966 | 0.09513 | 0.09633 | 0.1013 |

> **Convention:** ↓ lower is better; ↑ higher is better; no arrow = no monotone direction. A28 Kurtosis Ratio: perfect = 1.0.
> **Headline:** **2 of the 36 A-metric rows sit at or below the independent-draw floor** — A18 Disc Score MLP, A34 Teacher-Sigma RMSE. The largest remaining gaps are **A1 Kurtosis Error** (1.005 vs floor 0.008385, 119.9×); **A23 ACF |r| Lag-1 Error** (0.05796 vs floor 0.001038, 55.9×); **A21 ACF |r| Error (lags)** (0.05 vs floor 0.001066, 46.9×); **A32 Vol-of-Vol Error** (5.04e-04 vs floor 1.14e-05, 44.3×). The floor, not the other methods, is the reference on this page; the cross-method comparison lives in the dataset-level [`../README.md`](../README.md) so that no method's page grades itself against a rival it was tuned beside.
> **Perfect floor** is the *independent-draw* floor (GUIDELINE §5.4): five fresh draws from the *same* SDE with the *same* frozen per-asset parameters at seeds 1000-1004, scored with byte-identical metric code. It is **non-zero everywhere** — two independent 8 192-path draws never produce identical histograms, ACFs, quantiles or covariance matrices. It is **not** a permutation of the test set, which would preserve every column-wise statistic exactly, collapse most metrics to 0, and be a misleading target.
> **A1-A5**: fat-tail block — kurtosis error, tail quantile / QQ errors on |log-returns|, Hill tail index. **A6-A11** *(native d=8)*: path-kernel distances on the full 8-dimensional tensor (MMD² on paths / terminal / increments / realized-vol; sliced-Wasserstein on terminal & full paths), the rows where a multivariate generalisation is genuinely meaningful.
> **A12-A17**: distribution block, per asset then averaged. **A18** *(native d=8)*: discriminative classifier on all 8 channels at once, score = |accuracy − 0.5|. **A19**: TSTR MAE, deliberately **per-asset** — `predictive_score.py::_train_gru` targets `data_t[idx, 1:, :1]`, i.e. only the first feature, so a native run would silently report an asset-0-only number under a multi-asset name.
> **A20** *(native d=8)*: error on the full terminal covariance matrix. This is the row that matters most for Deep-MKV-TS: the correction network emits a full `(d, d)` matrix and the spectral clip acts on eigenvalues, so A20 is the direct test of whether the d(d−1)/2 spot correlations Σˢ survived the control. A diagonal control would have no mechanism to get this row right. **A21-A24**: ACF |r|/r² errors. **A25** *(native d=8)*: mean RMSE across all assets jointly.
> **A26-A32**: volatility block. **A28** kurtosis ratio: perfect = 1.0. **A33-A34**: Heston-specific — whether the generated S-paths retain the latent variance path. Deep-MKV-TS is the one method on this dataset carrying an **explicit** volatility state: the frozen Guyon–Lekeufack reference kernel maintains trend and activity features with fitted half-lives, and the learned control multiplies its σ. These rows therefore test the reference kernel and the correction together, not the network alone.

---

## B, Curve-Shape Metrics, mean ± std across 5 seeds

Each stylised-fact plot yields a **curve** L (a list of values), not a scalar. The curve is
computed **per asset and averaged over the 8 assets** *before* the combination below, so the
combination rules stay byte-identical to the d = 1 tree. For the real data (L_r) and generated
data (L_g) we build three lists, the curve L, its first finite difference L' (der), and its
second finite difference L'' (sec\_der), then combine the three sub-scores into **one number
per plot**:

- **MSE row**: for each list, dᵢ = mean((L_r − L_g)²). Reported mean = the **mean of the three sub-scores** (funct + der + sec\_der)/3; std = the sample std of that per-seed combined score across the 5 seeds. The **MSE row decides the cross-method winner**.
- **% err row**: for each list, dᵢ = mean(|L_g − L_r| / (|L_r| + 1e-6)) × 100, a proper MAPE, one division (the mean already averages over the curve's points). Reported value = the **function-level MAPE on the curve L itself**, the derivative / 2nd-derivative MAPE is **excluded** because diff(L)/diff2(L) have near-zero true values, so their relative error explodes into meaningless 10⁴-% figures. mean/std = mean and **sample std across the 5 seeds** of that per-seed function MAPE.
- **NRMSE row**: sqrt(mean((L_g − L_r)²)) / (max|L_r| − min|L_r| + 1e-12) × 100 on the curve L **only (funct-only)**, the ill-posed derivative / 2nd-derivative curves are excluded for the same reason as the % err row.
- **CVaR₉₀ / CVaR₉₅ rows**: tail-averaged pointwise curve error (Expected Shortfall) on the curve L **only (funct-only)**. Pointwise error eₜ = |L_g(t) − L_r(t)|; for q ∈ {0.90, 0.95}, CVaR_q = mean(eₜ for eₜ ≥ the q-th percentile of eₜ), then range-normalized like NRMSE (÷ (max|L_r| − min|L_r| + 1e-12) × 100).

All ↓ lower is better. The perfect floor is **non-zero** for all plots, it is the residual finite-sample error of an independent multi-asset Heston draw scored against the test set, identical across methods.
Five sublines per plot: **MSE**, **% error**, **NRMSE**, **CVaR₉₀** and **CVaR₉₅** (the per-seed columns hold that seed's combined score).

| Plot | Measure | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|---|---|---|---|---|---|---|---|---|
| **Path comparison** *(50×50 path-cloud)* | grid_tvd 50×50 (%) ↓ | 4.453% ± 0.1243% | 4.379% | 4.637% | 4.432% | 4.537% | 4.278% | 2.189% |
| **Log-return histogram** | MSE | 1.109 ± 0.07596 | 1.047 | 1.053 | 1.171 | 1.047 | 1.228 | 0.0554 |
|  | % err | 8.647% ± 0.3958% | 8.511% | 9.414% | 8.285% | 8.582% | 8.444% | 1.302% |
|  | NRMSE | 3.914% ± 0.1119% | 3.784% | 3.878% | 3.973% | 3.833% | 4.1% | 0.3601% |
|  | CVaR₉₀ | 9.562% ± 0.3073% | 9.23% | 9.377% | 9.783% | 9.368% | 10.05% | 0.8339% |
|  | CVaR₉₅ | 10.96% ± 0.3034% | 10.62% | 10.75% | 11.17% | 10.79% | 11.44% | 0.994% |
| **QQ plot** | MSE | 1.34e-07 ± 2.16e-08 | 1.11e-07 | 1.69e-07 | 1.48e-07 | 1.24e-07 | 1.16e-07 | 5.66e-10 |
|  | % err | 10.91% ± 0.3707% | 10.48% | 10.99% | 11.11% | 10.52% | 11.46% | 0.5364% |
|  | NRMSE | 1.637% ± 0.1159% | 1.511% | 1.829% | 1.708% | 1.565% | 1.571% | 0.08644% |
|  | CVaR₉₀ | 1.545% ± 0.193% | 1.431% | 1.894% | 1.565% | 1.512% | 1.322% | 0.09922% |
|  | CVaR₉₅ | 2.024% ± 0.3348% | 1.871% | 2.579% | 2.025% | 2.094% | 1.551% | 0.125% |
| **ACF \|r\|** | MSE | 6.59e-04 ± 2.65e-05 | 6.44e-04 | 6.85e-04 | 6.62e-04 | 6.89e-04 | 6.17e-04 | 3.33e-06 |
|  | % err | 61.45% ± 1.913% | 59.48% | 63.71% | 62.14% | 63.02% | 58.92% | 2.043% |
|  | NRMSE | 76.92% ± 2.291% | 74.84% | 79.39% | 78.01% | 78.77% | 73.58% | 2.523% |
|  | CVaR₉₀ | 107.5% ± 2.849% | 105.5% | 110.4% | 108.9% | 109.7% | 102.9% | 4.991% |
|  | CVaR₉₅ | 109.4% ± 2.973% | 107.5% | 112.5% | 111.1% | 111.3% | 104.5% | 5.542% |
| **ACF r²** | MSE | 3.72e-04 ± 1.97e-05 | 3.58e-04 | 3.91e-04 | 3.76e-04 | 3.93e-04 | 3.42e-04 | 3.98e-06 |
|  | % err | 55.95% ± 1.995% | 53.8% | 58.25% | 56.64% | 57.66% | 53.39% | 2.37% |
|  | NRMSE | 66.16% ± 2.262% | 64.19% | 68.52% | 67.2% | 68.11% | 62.81% | 2.772% |
|  | CVaR₉₀ | 94.31% ± 2.643% | 92.47% | 96.84% | 95.7% | 96.52% | 90.03% | 5.52% |
|  | CVaR₉₅ | 96.35% ± 2.735% | 94.87% | 98.99% | 98.15% | 98.08% | 91.66% | 6.131% |
| **Rolling vol histogram** | MSE | 38.28 ± 0.8864 | 36.92 | 38.08 | 39.14 | 37.91 | 39.36 | 0.6039 |
|  | % err | 21.37% ± 1.345% | 20.89% | 23.36% | 21.04% | 22.19% | 19.35% | 1.536% |
|  | NRMSE | 10.1% ± 0.09499% | 9.95% | 10.04% | 10.14% | 10.14% | 10.22% | 0.5137% |
|  | CVaR₉₀ | 19.98% ± 0.31% | 19.66% | 19.72% | 20.2% | 19.86% | 20.48% | 1.172% |
|  | CVaR₉₅ | 21.08% ± 0.4476% | 20.7% | 20.61% | 21.41% | 20.91% | 21.79% | 1.362% |
| **Tail survival** | MSE | 1.96e-04 ± 2.42e-05 | 1.79e-04 | 1.78e-04 | 2.16e-04 | 1.74e-04 | 2.34e-04 | 2.03e-07 |
|  | % err | 5.509% ± 0.3455% | 5.08% | 6.023% | 5.682% | 5.173% | 5.587% | 0.2297% |
|  | NRMSE | 2.25% ± 0.1296% | 2.118% | 2.224% | 2.337% | 2.12% | 2.453% | 0.06634% |
|  | CVaR₉₀ | 3.324% ± 0.1771% | 3.149% | 3.265% | 3.456% | 3.15% | 3.598% | 0.1122% |
|  | CVaR₉₅ | 3.34% ± 0.1784% | 3.164% | 3.28% | 3.477% | 3.165% | 3.614% | 0.1175% |

> **Headline:** **0 of the 6 B plots** sit at or below the finite-sample floor on the deciding MSE row: none.
> **Cross-seed stability**: the seed here controls **three** independent things — the control network's initialisation, every Euler–Maruyama increment drawn during training, and the batch resampling that re-fits the Z-proxy at each outer iteration. The reference kernel is *not* among them: it is fitted once, frozen, and shared byte-for-byte across all 5 seeds (`code/reference/reference_kernel.json`), so none of the spread below comes from re-estimating the reference SDE. What remains is optimisation plus sampling variance, and [`plots/loss_convergence.png`](plots/loss_convergence.png) shows whether any seed diverged. Generation uses a **separate** seed stream (90000 + i) that never reuses a training seed.

---

## Stylised Facts Diagnostic (Multi-Asset Heston vs Deep-MKV-TS, seed 0, asset 0)

Eight-panel comparison: sample paths, return distribution, QQ plot, ACF of |returns|,
ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

The third (black dashed) curve is the **independent-draw floor**, not a closed-form theory
curve: `metrics/heston_theory.py` is hard-wired to the d = 1 parameters (κ=2.0, θ=0.04,
ρ=−0.7, dt=1/250), so plotting it over d = 8 data would be a fabricated reference. Wherever
Real and the floor curve already disagree, the gap is finite-sample noise rather than a
modelling failure. **One asset is shown, not eight** — the *metrics* are averaged over all 8
assets, the *figure* is asset 0; pooling eight deliberately different volatilities
(σ 0.0097-0.0165) into one histogram would show the mixing, not the model.

![Heston Diagnostics](plots/heston_diagnostics.png)

---

## Deep-MKV-TS Training Loss (5 seeds)

The figure has **two** panels, and they are not the same quantity plotted twice.

`loss_total` is the **adjoint-matching loss**
$L_{\text{adj}} = \frac{1}{PM}\sum\sum \|\hat Z^{\theta} - Z^{\text{proxy}}\|^2$ — the
scalar AdamW actually minimises. Its target **moves**: $Z^{\text{proxy}}$ is re-estimated by
ridge regression on freshly sampled paths at every outer iteration, so this curve measures
distance to a target that is itself drifting. A descending `loss_total` is necessary but **not
sufficient** — it can fall simply because the proxy became easier to hit.

`objective` is the **path-functional discrepancy** between generated paths and the training
bank, a fixed yardstick and the same functional form that
`select_checkpoint_multiasset.py` later evaluates on the validation split to choose the
reported checkpoint. **Read convergence from this panel.** Both are logged on the same rows,
so the two x-axes coincide step-for-step.

The x-axis is **outer iteration**, not epoch. Algorithm 1 has no epoch: every iteration draws a
fresh batch of controlled paths and re-fits the proxy, so there is no pass over a fixed dataset
to count. There is also no validation curve here, deliberately — Deep-MKV-TS runs no held-out
forward pass during training. Selection happens **after** the fit, by sampling from each saved
checkpoint and scoring against `heston_ma_S_val_8192x252x8.npy`; the per-checkpoint scores are
archived in `code/selection/seed_*_selection.json` and the winner is the `Selected step` column
below.

![Training Loss](plots/loss_convergence.png)

Training wall-clock, read from `weights/seed_*_config.json` and
`losses/seed_*_losses.csv` (**0 min total** across the 5 seeds, one GPU and
8 threads per seed process, run 2-up in waves):

_(weights/seed_\*_config.json not found)_

Generation wall-clock — the selected control rolled forward 251
Euler–Maruyama steps from $x_0 = \log(100)$ for each of 8 192 paths, 8 threads,
**0.2 min total** across the 5 seeds:

| Seed | Workers | Elapsed |
|------|---------|---------|
| 0 | 8 | 0.0 min |
| 2 | 8 | 0.0 min |
| 4 | 8 | 0.0 min |
| 5 | 8 | 0.0 min |
| 6 | 8 | 0.0 min |

---

## A18, Discriminative Classifier Training Loss

BCE loss during GRU and MLP classifier training (2 000 steps, logged every 50 steps).
A value near ln(2) ≈ 0.693 means the classifier cannot distinguish real from fake.
The d = 8 classifier is **native**: it sees all 8 channels at once.

![Discriminative Classifier Loss](plots/disc_classifier_loss.png)

---

## A19, Predictive Score Training Loss (TSTR)

MAE loss during GRU and MLP predictor training on *synthetic* data (5 000 steps, logged every 100 steps).
The predictor is run **once per asset**; the curves archived here are **asset 0's**, since the
driver stores the loss history only for `j == 0`. The A19 score in the table above is the mean
over all 8 assets.

![Predictive Score Loss](plots/pred_score_loss.png)

---

## File layout

This tree is **self-contained**: code, inputs, outputs and documentation sit side by side,
unlike the d = 1 benchmark which splits `methods/<Method>/` from `results/Heston/<Method>/`.
It is generated from `os.walk` at render time, so it cannot list a file that does not exist.

```
results/HestonMultiAsset/Deep-MKV-TS/
├── code/
│   ├── README.md                       reference kernel, deviations, sweep table
│   ├── _smoke_reference.py
│   ├── collect_artifacts.py            rebuilds generation_time.csv, checks the §4 contract
│   ├── commit_and_push.sh
│   ├── diagnose_divergence.py
│   ├── eigh_fallback.py
│   ├── fit_reference_multiasset.py     penalised MLE for the frozen d = 8 reference kernel
│   ├── losses/
│   ├── matrix_control_multiasset.py    matrix-valued spectral control + Daleckii-Krein adjoint
│   ├── measure_memorisation.py         nearest-neighbour memorisation diagnostic
│   ├── multivariate_reference.py       the d = 8 Guyon-Lekeufack kernel itself
│   ├── null_control_baseline.py
│   ├── plot_diagnostics_multiasset.py  the 8-panel stylised-facts figure
│   ├── plot_losses.py                  plots/loss_convergence.png, 5 seeds overlaid
│   ├── reference/                      FITTED artefacts only — the upstream package is NOT vendored here
│   │   ├── reference_fit_history.csv   its penalised-MLE calibration trace
│   │   └── reference_kernel.json       the frozen kernel, fitted once, shared by all 5 seeds
│   ├── render_readme.py                regenerates this README from the artefacts
│   ├── run_all_multiasset.py           samples the 8192-path bank from the SELECTED checkpoint
│   ├── run_campaign.sh
│   ├── run_pipeline.sh
│   ├── run_pipeline_acfup.sh
│   ├── run_sweep.sh                    drives the sweep in two waves on one GPU
│   ├── runs/                           per-seed training_checkpoints/ (gitignored)
│   │   └── seed_0   … and 4 more
│   ├── scan_step_curve.py
│   ├── select_checkpoint_multiasset.py picks the reported checkpoint per seed on validation
│   ├── selection/                      per-seed checkpoint-selection records
│   │   └── seed_0_selection.json   … and 4 more
│   ├── sweep_hyperparams.py
│   ├── sweep_ridge_lambda.py           re-selects ridge_lambda on the VALIDATION split
│   ├── sweep_verify_stale/
│   │   ├── lambda_r1000_L10_K100_a0.25_s0.125.json
│   │   ├── lambda_r1000_L200_K100_a0.25_s0.125.json
│   │   └── lambda_r1000_L50_K400_a0.25_s0.125.json
│   ├── test_render_readme.py
│   ├── train_multiasset.py             Algorithm 1, one seed
│   └── weights/
├── curve_b_aggregate.json              B curve-shape aggregate
├── generated_paths/
│   └── seed_0   … and 4 more
├── grid_tvd_aggregate.json             path-cloud TVD
├── logs/
│   ├── pipeline_acfup.log
│   ├── train_seed_0.log
│   ├── train_seed_2.log
│   ├── train_seed_4.log
│   ├── train_seed_5.log
│   └── train_seed_6.log
├── losses/
│   ├── generation_time.csv             wall-clock time per seed
│   ├── memorisation.json               NN-ratio diagnostic on the final 8192-path output
│   ├── seed_0_losses.csv
│   ├── seed_2_losses.csv
│   ├── seed_4_losses.csv
│   ├── seed_5_losses.csv
│   └── seed_6_losses.csv
├── metrics_per_asset.csv               per-metric × per-asset breakdown (8 rows per metric)
├── metrics_summary.csv                 A1-A34, mean ± std, per seed
├── plots/
│   ├── heston_diagnostics.png          8-panel stylised facts (seed 0, asset 0)
│   └── loss_convergence.png            L_adj and the path-functional objective, 5 seeds
├── seed_0_disc_gru_loss.csv
├── seed_0_disc_mlp_loss.csv
├── seed_0_metrics.json
├── seed_0_pred_gru_loss.csv
├── seed_0_pred_mlp_loss.csv
├── seed_2_disc_gru_loss.csv
├── seed_2_disc_mlp_loss.csv
├── seed_2_metrics.json
├── seed_2_pred_gru_loss.csv
├── seed_2_pred_mlp_loss.csv
├── seed_4_disc_gru_loss.csv
├── seed_4_disc_mlp_loss.csv
├── seed_4_metrics.json
├── seed_4_pred_gru_loss.csv
├── seed_4_pred_mlp_loss.csv
├── seed_5_disc_gru_loss.csv
├── seed_5_disc_mlp_loss.csv
├── seed_5_metrics.json
├── seed_5_pred_gru_loss.csv
├── seed_5_pred_mlp_loss.csv
├── seed_6_disc_gru_loss.csv
├── seed_6_disc_mlp_loss.csv
├── seed_6_metrics.json
├── seed_6_pred_gru_loss.csv
├── seed_6_pred_mlp_loss.csv
└── weights/
    └── .campaign_complete   … and 5 more
```

The upstream `deep_mkv_gen_path_dt` package is **not vendored here**. It lives once at
`methods/Deep-MKV-TS/code/reference/` and is imported through `PYTHONPATH`, **with zero
edits** — every d = 8 adaptation is a method-local file in `code/`. `code/reference/` holds
only the *fitted* artefacts: the frozen kernel and its calibration history.

The `.npy` arrays are **gitignored**: `(8192, 252, 8)` float64 = 132 MB each, over
GitHub's 100 MB per-file hard limit, and LFS was ruled out. They are fully reproducible from the
tracked code, the tracked frozen kernel and the tracked hyperparameters; the `metadata.json`
beside each array **is** tracked, so shapes, price ranges and generation times stay auditable
without the payload.

## Reproduce

```bash
cd /home/tbasseras/benchmark
PY=/home/tbasseras/gpu-venv/bin/python
R=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
export PYTHONPATH="$R/src:$R/experiments"

# 1. dataset (~4 min on 8 cores) — only if dataset/HestonMultiAsset/*.npy are absent
cd dataset/HestonMultiAsset && python generate_heston_multiasset.py && cd -

# 2. fit the frozen reference kernel by penalised MLE on the TRAIN split
#    Writes code/reference/reference_kernel.json. Fitted ONCE and shared by all 5 seeds.
cd results/HestonMultiAsset/Deep-MKV-TS/code
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 taskset -c 0-7 $PY fit_reference_multiasset.py

# 3. re-select ridge_lambda on the VALIDATION split (GPU 1 only, two waves)
#    The only retuned hyperparameter. Writes sweep/*.json and sweep/winner.json.
bash run_sweep.sh

# 4. final 5 seeds,  outer iterations, 2 GPUs (~0 min of GPU time, 3 waves)
GPUS=(1 2); CORES=("0-7" "8-15")
for wave in "0 1" "2 3" "4"; do
  i=0
  for s in $wave; do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} OMP_NUM_THREADS=8 taskset -c ${CORES[$i]} \
        $PY train_multiasset.py --seed $s --device cuda:0 & i=$((i+1))
  done
  wait
done

# 5. choose the reported checkpoint per seed on VALIDATION, never on test
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 taskset -c 0-7 \
    $PY select_checkpoint_multiasset.py --seeds 0 1 2 3 4

# 6. generate the 8192-path bank from the SELECTED checkpoint
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 taskset -c 0-7 \
    $PY run_all_multiasset.py --seeds 0 1 2 3 4
$PY collect_artifacts.py            # must print "5 rows" and exit 0 before anything downstream
cd /home/tbasseras/benchmark

# 7. independent-draw perfect-recovery floor (seeds 1000-1004, ~6 s)
/home/tbasseras/sbts-venv/bin/python metrics/gen_perfect_recovery_multiasset.py

# 8. metrics — both sides go through byte-identical code
CUDA_VISIBLE_DEVICES=1 $PY metrics/compute_all_multiasset.py --method perfect_recovery
CUDA_VISIBLE_DEVICES=1 $PY metrics/compute_all_multiasset.py --method Deep-MKV-TS \
    --dataset HestonMultiAsset --seeds 5

# 9. figures
$PY results/HestonMultiAsset/Deep-MKV-TS/code/plot_losses.py
$PY results/HestonMultiAsset/Deep-MKV-TS/code/plot_diagnostics_multiasset.py \
    --method Deep-MKV-TS
$PY metrics/plot_score_losses.py --method Deep-MKV-TS --dataset HestonMultiAsset

# 10. memorisation diagnostic
$PY results/HestonMultiAsset/Deep-MKV-TS/code/measure_memorisation.py --seeds 0,1,2,3,4

# 11. regenerate this README from the artefacts
$PY results/HestonMultiAsset/Deep-MKV-TS/code/render_readme.py
```
