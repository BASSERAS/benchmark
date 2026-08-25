# SBTS — source and implementation notes (d = 8)

Implementation of **Schrödinger Bridge Time Series** for the multi-asset Heston dataset.

> Alouadi, Barreau, Carlier & Pham, *Schrödinger Bridge Time Series Generation*,
> ICAIF 2025 — [arXiv:2503.02943](https://arxiv.org/abs/2503.02943)

The generated paths, metric tables and figures are on the method page
[`../README.md`](../README.md). The dataset law, the metric scoping rules and the
independent-draw floor are one level up in [`../../README.md`](../../README.md).
Requirements for adding a new method are in
[`../../MULTIASSET_GUIDELINE.md`](../../MULTIASSET_GUIDELINE.md).

---

## Files

| File | Role |
|------|------|
| `sbts_generate_multiasset.py` | core module — radial kernel, per-asset σ, bridge weights |
| `run_all_multiasset.py` | driver: 5 seeds × 8192 paths × 252 steps × 8 assets |
| `plot_diagnostics_multiasset.py` | the 8-panel stylised-facts figure |
| `measure_memorisation.py` | nearest-neighbour memorisation diagnostic |
| `render_readme.py` | regenerates `../README.md` from the metric artefacts |

Generation runs under `/home/tbasseras/sbts-venv/bin/python` (numpy 1.26.4 + numba 0.66.0, no
torch). `render_readme.py` and `measure_memorisation.py` run under
`/home/tbasseras/gpu-venv/bin/python`.

---

## The four questions (MULTIASSET_GUIDELINE §0)

**1. Does the method natively accept `(N, T, d)` input?**
Yes — **one joint model**, not a per-asset ensemble. The kernel is evaluated on full
d-dimensional increment vectors, so the cross-asset dependence carried by the training
increments is preserved rather than reassembled from 8 independent marginals.

**2. Does any hyperparameter fail to cross dimension?**
**Yes — the bandwidth `h`.** The kernel

```
K_h(x) = (h² − ‖x‖²)² · 1{‖x‖₂ < h}
```

is **radial**, so its support is a ball and `h` must scale with the typical distance between
d-dimensional increments. That distance grows like √d: median pairwise distance is
**0.0372 at d = 1** vs **0.2643 at d = 8**. The author's d = 1 value `h = 0.05` collapses
outright at d = 8 — 40.8 % volatility error, excess kurtosis 45.66 against a real 1.12, and a
minimum price of 0.0.

`h` was re-selected **on the validation split** (`heston_ma_S_val_8192x252x8.npy`), never on
test. The sweep and the pre-registered selection criterion are recorded in
`../losses/bandwidth_selection.json` and `../losses/selection_criterion.md`.

**3. What is the memorisation risk?**
**High, and it is realised.** SBTS is non-parametric: it resamples and reweights *training*
increments, so it can produce near-copies by construction. The NN-ratio diagnostic is therefore
mandatory here and is reported on the method page. Zero exact duplicates does **not** clear a
kernel method — it interpolates between training paths, so it yields near-copies rather than
bitwise copies.

**4. Compute budget.**
Generation is CPU-bound and embarrassingly parallel over paths. Measured: **79.5 min** for seed 0
at 16 workers, **22.7–23.2 min** per seed at 64 workers (a one-off exception to the 16-core
limit, explicitly authorised). Metrics add ~11 min per seed on one GPU. Per-seed wall-clock is in
`../losses/generation_time.csv`.

---

## Hyperparameters

| Symbol | Value | Source |
|--------|-------|--------|
| `h` | **0.31** | **re-selected for d = 8** on the validation split |
| `K` | 20 | paper |
| `N_pi` | 50 | paper |
| `dt` | 1/252 = 0.003968… | dataset |
| `M_simu` | 8192 | matches the train split |

---

## Deviations from the committed d = 1 implementation

Three, all deliberate. None changes the d = 1 numbers, which stay reproducible from the
untouched `methods/SBTS/code/sbts_generate.py`.

**1. Per-asset σ, not a pooled scalar.** The eight assets have genuinely different volatilities
(σ ranges 0.0097–0.0165), so log-returns are standardised per asset:

```
sigma = [0.010497, 0.014211, 0.012532, 0.016460,
         0.011853, 0.014539, 0.011151, 0.009748]
```

Scaled std is 0.062995 against a target √dt = 0.062994. A single pooled σ would distort every
marginal toward the cross-sectional average.

**2. A radial kernel, not a product of d univariate kernels.** Following the reference
`sbts_multi_markovian.py`. See question 2 above for why this forces `h` to be re-selected.

**3. The correct `weights_tilde` form.** The upstream *univariate* file
(`sbts_uni_markovian.py`, lines 96–98) computes `exp(diff² · dt / 2)`, while the *multivariate*
file (line 101) computes `exp(diff² / (2·dt))`. Continuity at `k = 0` proves the **multivariate**
form correct: at `dt = 1/252` the univariate form evaluates to `exp(8e-6) ≈ 1` instead of
`exp(0.5) ≈ 1.65`, silently disabling the bridge correction. This implementation uses the
multivariate form.

---

## There are no weights

SBTS performs **no gradient training**. There is no checkpoint to load: the "model" *is* the
training array `dataset/HestonMultiAsset/heston_ma_S_8192x252x8.npy` plus the three
hyperparameters `(h, K, N_pi)`. See [`../weights/README.md`](../weights/README.md).

Consequently the method page carries **no training-loss section**. Neural methods on this
dataset must add one — see MULTIASSET_GUIDELINE §3 and §8.6.

---

## Reproduce

```bash
cd /home/tbasseras/benchmark

# 5 seeds of (8192, 252, 8) paths
/home/tbasseras/sbts-venv/bin/python \
    results/HestonMultiAsset/SBTS/code/run_all_multiasset.py

# score them
CUDA_VISIBLE_DEVICES=3 /home/tbasseras/gpu-venv/bin/python \
    metrics/compute_all_multiasset.py --method SBTS --dataset HestonMultiAsset --seeds 5

# memorisation diagnostic
CUDA_VISIBLE_DEVICES=3 /home/tbasseras/gpu-venv/bin/python \
    results/HestonMultiAsset/SBTS/code/measure_memorisation.py --seeds 0,1,2,3,4

# figures
/home/tbasseras/gpu-venv/bin/python \
    results/HestonMultiAsset/SBTS/code/plot_diagnostics_multiasset.py \
    --method SBTS --seed 0 --asset 0
/home/tbasseras/gpu-venv/bin/python \
    metrics/plot_score_losses.py --method SBTS --dataset HestonMultiAsset

# regenerate ../README.md
/home/tbasseras/gpu-venv/bin/python \
    results/HestonMultiAsset/SBTS/code/render_readme.py
```
