# Multi-Asset Heston (d = 8)

Benchmark dataset and results for **d = 8 correlated** Heston stochastic-volatility
price paths. This page holds everything that belongs to the *dataset* rather than to
any one generator: the law, the metric scoping rules, the independent-draw floor, the
cross-method leaderboard, and the memorisation diagnostic.

Per-method pages mirror the d = 1 layout of `methods/<Method>/README.md` exactly, so
the two can be read side by side:

| Method | A rows at/below floor | B plots at/below floor | NN memorisation ratio |
|--------|----------------------:|-----------------------:|----------------------:|
| [SBTS](SBTS/README.md) | 12 / 103 | 0 / 6 | 0.219 |

---

## The dataset

A **parsimonious multi-asset Heston** model (Szimayer, Dimitroff & Lorenz 2011,
*IJTAF* 14(8) 1299-1333): eight independent 1-D Heston marginals coupled **only**
through the d(d−1)/2 spot correlations Σˢ. The joint correlation of (W, W̃) is block
diagonal, `diag(Σˢ, I_d)`, which makes it positive semi-definite **by construction** —
no repair step, no eigenvalue clipping, no silently altered law.

Per-asset parameters are perturbations of the daily SPX physical-measure estimates in
Ait-Sahalia & Kimmel (2007), *JFE* 83(2), Table 6: κ̄ = 5.07, θ̄ = 0.0457, η̄ = 0.48,
ρ̄ = −0.767. They are frozen by `PARAM_SEED = 1234`, kept inside the Feller condition
with `FELLER_SAFETY = 0.70`, and dumped to the tracked
[`dataset/HestonMultiAsset/parameters.json`](../../dataset/HestonMultiAsset/parameters.json).

| Split | File | Role |
|-------|------|------|
| train | `heston_ma_S_8192x252x8.npy` | what generators fit |
| test | `heston_ma_S_test_8192x252x8.npy` | the "real" side of every A/B metric |
| disc | `heston_ma_S_disc_8192x252x8.npy` | real side for the A18 discriminator |
| val | `heston_ma_S_val_8192x252x8.npy` | hyperparameter selection only, never scored |
| valdisc | `heston_ma_S_valdisc_8192x252x8.npy` | discriminator half of the validation split |

Shape `(8192, 252, 8)` float64, S0 = 100 for every asset, dt = 1/252.

> **The `.npy` arrays are gitignored.** 132 MB each, over GitHub's 100 MB per-file hard
> limit, and Git LFS was ruled out for this repo. They are fully reproducible from the
> tracked generator and the tracked `parameters.json`; the `metadata.json` beside every
> generated array **is** tracked, so shapes, price ranges and timings stay auditable
> without the payload.

Reproducing `sample_parameters()` is checked, not assumed:
`metrics/gen_perfect_recovery_multiasset.py` hashes the freshly sampled per-asset block
and aborts if it does not match the committed digest. `numpy.random.Generator` is **not**
guaranteed bit-stable across numpy releases (unlike the legacy `RandomState`), so without
that guard a numpy upgrade could silently draw the floor from a different law than the
train/test splits.

---

## The independent-draw floor

Every metric table carries a **Perfect floor** column. It is the score a *perfect*
generator would obtain — one whose output is distributionally identical to the truth.
It is realised honestly, per GUIDELINE §5.4, by drawing **5 fresh independent sets** of
8192 paths from the *same* SDE with the *same* frozen parameters at brand-new RNG seeds
(1000-1004), then scoring them with byte-identical metric code.

The floor is **non-zero everywhere**: two independent 8192-path draws never produce
identical histograms, ACFs, quantiles, covariance matrices or moments. A method that
matches the floor has hit the finite-sample noise ceiling and cannot be improved on by
any generator, however good.

> **It is not a permutation baseline.** Row-shuffling the test set would preserve every
> column-wise statistic exactly and collapse most metrics to 0 — a target no honest
> generator can or should reach. An independent re-simulation is the real lower bound.

---

## How the metrics are computed in d = 8

Two scopes, and the scope of every row is recorded in the `scope` column of each
method's `metrics_summary.csv`:

- **Per-asset then averaged** — the metric is computed independently on each of the 8
  univariate slices `S[:, :, j]` and the **mean over assets** is the headline number.
  This covers **A1-A5, A12-A17, A19, A21-A24, A26-A34**, every **B curve-shape** metric
  and `grid_tvd`. The per-asset breakdown is kept in each method's
  `metrics_per_asset.csv` (one row per metric × asset), so an average cannot hide a
  single bad asset.
- **Native d = 8** — the metric is evaluated **once** on the full `(N, T, 8)` tensor and
  is marked *(native d=8)* in the per-method tables. This covers **A6-A11, A18, A20, A25**: the
  rows where a multivariate generalisation is genuinely meaningful — kernel and
  Wasserstein distances between d-dimensional path distributions, a d-channel
  discriminator, and the full terminal covariance matrix.

**A20 is the row that actually tests the multi-asset structure.** It is the error on the
full terminal covariance matrix, so it is the only metric that fails loudly if a
generator reproduces all eight marginals perfectly but destroys the spot correlations Σˢ
that make this dataset multi-asset in the first place.

**A19 is deliberately per-asset, not native.** `metrics/predictive_score.py::_train_gru`
builds its target as `Y_b = data_t[idx, 1:, :1]` — it predicts **only the first feature**.
Run natively on a d = 8 tensor it would silently report an *asset-0-only* score under a
multi-asset name. Running it 8 times on 8 univariate slices and averaging is the only
honest reading, and it is why A19 is absent from the native list above.

---

## Memorisation — are the generated paths actually new?

A generator can score well on every distributional metric by simply reproducing its
training set. The A/B tables cannot detect this: a near-copy of the training data has,
by construction, almost exactly the right histograms, ACFs and moments.

The diagnostic is a nearest-neighbour ratio in **log-return space** (prices are all
anchored at S0 = 100, so raw price distance is dominated by the shared anchor and
understates similarity), flattening each path to a `(T-1)·d = 2008`-vector:

```
nn_ratio = median_i min_j ||gen_i  - train_j||  /  median_i min_j ||real_i - train_j||
```

where `real` is the **held-out test split** — data from the same law the generator never
saw. That denominator is the whole point: it calibrates "how close is close" using the
true process itself, so the number cannot be gamed by the dataset's intrinsic scale or
dimension.

- **ratio ≈ 1.0** — generated paths sit no closer to the training set than genuine
  held-out data does. No memorisation.
- **ratio ≪ 1.0** — generated paths hug the training set; `1/ratio` is how many times
  closer they sit than real data does.

The diagnostic is **calibrated**: run on the independent-draw floor — samples that are
provably new, from the same law — it returns **1.0001**. Any value materially below 1 is
therefore a real property of the generator, not an artefact of d = 8.

| Method | NN ratio | Std across seeds | Times closer than held-out data | Exact duplicates |
|--------|---------:|-----------------:|--------------------------------:|-----------------:|
| [SBTS](SBTS/README.md) | 0.2189 | 0.0003 | 4.6× | 0 |

> **Zero exact duplicates does NOT clear a method.** A kernel generator interpolates
> between training paths, so it produces *near*-copies rather than bitwise copies. The
> ratio is the number that matters.

---

## Per-method implementation notes

### SBTS

The d = 8 entry deviates from the committed d = 1 implementation in three places. All
three are deliberate and none of them changes the d = 1 numbers, which stay reproducible
from the untouched `methods/SBTS/code/sbts_generate.py`.

1. **Per-asset σ, not a pooled scalar.** The eight assets have genuinely different
   volatilities (σ ranges 0.0097-0.0165), so log-returns are scaled per asset. Pooling a
   single σ would distort every marginal toward the cross-sectional average.
2. **A radial kernel, not a product of d univariate kernels.**
   `K_h(x) = (h² − ‖x‖²)²·1{‖x‖₂ < h}`, following the reference
   `sbts_multi_markovian.py`. Its support is a **ball**, so the bandwidth `h` cannot
   cross dimension: the typical distance between d-dimensional increments grows like √d
   (median pairwise distance 0.0372 at d = 1 vs 0.2643 at d = 8). The author's d = 1
   value `h = 0.05` collapses outright at d = 8 — 40.8 % volatility error, excess
   kurtosis 45.66 against a real 1.12, and a minimum price of 0.0.
3. **The correct `weights_tilde` form.** The upstream *univariate* file
   (`sbts_uni_markovian.py`, lines 96-98) computes `exp(diff² · dt / 2)`, while the
   *multivariate* file (line 101) computes `exp(diff² / (2·dt))`. Continuity at `k = 0`
   proves the **multivariate** form correct; at `dt = 1/252` the univariate form
   evaluates to `exp(8e-6) ≈ 1` instead of `exp(0.5) ≈ 1.65`, silently disabling the
   bridge correction. This implementation uses the correct multivariate form.

---

## Adding a method to this dataset

The tree is **self-contained per method** — code, inputs, outputs and documentation sit
side by side, unlike the d = 1 benchmark which splits `methods/<Method>/` from
`results/Heston/<Method>/`.

```
results/HestonMultiAsset/
├── oldreadme.md               ← this file (generated by tools/render_dataset_readme.py)
├── tools/render_dataset_readme.py
├── perfect_recovery/          the independent-draw floor
└── <Method>/
    ├── README.md              per-method page, mirrors methods/<Method>/README.md
    ├── code/                  generator + that method's README renderer
    ├── generated_paths/seed_{0..4}/   .npy gitignored, metadata.json tracked
    ├── losses/                training curves, or hyperparameter records
    ├── plots/                 heston_diagnostics, disc/pred loss figures
    └── weights/               checkpoints, or a README explaining their absence
```

1. Create `results/HestonMultiAsset/<Method>/` with the five slots above.
2. Generate 5 seeds of `(8192, 252, 8)` float64 paths into
   `generated_paths/seed_{i}/generated_paths_8192x252x8.npy`.
3. Score them — the same driver every method goes through, so no method gets bespoke
   metric code:
   ```bash
   /home/tbasseras/gpu-venv/bin/python metrics/compute_all_multiasset.py --method <Method>
   ```
4. Run the memorisation diagnostic and the figures.
5. Re-run this script; the leaderboard and memorisation tables pick the method up
   automatically.

## Reproduce the dataset and the floor

```bash
cd /home/tbasseras/benchmark

# dataset (~4 min on 8 cores)
cd dataset/HestonMultiAsset && python generate_heston_multiasset.py && cd -

# independent-draw floor, seeds 1000-1004 (~6 s), then score it
/home/tbasseras/sbts-venv/bin/python metrics/gen_perfect_recovery_multiasset.py
CUDA_VISIBLE_DEVICES=3 /home/tbasseras/gpu-venv/bin/python \
    metrics/compute_all_multiasset.py --method perfect_recovery

# regenerate this page
/home/tbasseras/gpu-venv/bin/python \
    results/HestonMultiAsset/tools/render_dataset_readme.py
```
