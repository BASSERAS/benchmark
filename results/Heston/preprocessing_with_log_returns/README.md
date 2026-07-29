# Heston — `preprocessing_with_log_returns` experiment

This folder holds generator results for the **SBTS log-return preprocessing** applied to
methods **other than SBTS**. The question it answers: *does feeding a generator
volatility-scaled log-returns (the way SBTS does) instead of standardized prices change its
scores on the Heston benchmark?*

The first method carried through the full pipeline is **LS4** (`LS4/`). Each subsequent method
gets its own sibling folder built the same way — see [`GUIDELINE.md`](GUIDELINE.md) for the
step-by-step recipe.

---

## What differs from the main benchmark

| Axis | Main benchmark (`results/Heston/<method>/`) | This experiment |
|------|---------------------------------------------|-----------------|
| Generator input | Standardized **price** `(S − μ)/σ` | Volatility-scaled **log-returns** `R·√dt/σ` (SBTS scaler) |
| Sample count | 8192 train / 8192 test / 8192 disc | **4096** train / 4096 test / 4096 disc |
| Path-shadowing queries | (per method) | **512** fresh, strictly-independent real paths (seed 3) |
| Everything else | — | **identical**: SDE, params, RNG, metric code, seeds 0–4 |

The datasets live in [`dataset/Heston/preprocessing_with_log_returns/`](../../../dataset/Heston/preprocessing_with_log_returns/README.md).
Only the **preprocessing** and **sample count** change; the SDE draws are byte-identical
truncations of the same Heston stream.

---

## The preprocessing pipeline (exact, reproducible)

This is the SBTS transform, copied verbatim from `methods/SBTS/code/sbts_generate.py`. It maps a
price panel `S ∈ ℝ^{M×128}` to a model input `X ∈ ℝ^{M×128}` and back.

### Forward (price → model input)

```python
import numpy as np
DT = 1.0 / 250.0          # daily step, identical to the SDE
S0 = 100.0

# 1. log-returns  (M, 127)
R = np.log(S[:, 1:] / S[:, :-1])

# 2. SBTS pooled sigma  (single scalar, estimated on TRAIN split only, then frozen)
sigma = float(R.std())    # population std, ddof=0, pooled over all M*127 returns

# 3. volatility-scaled returns  (std becomes exactly sqrt(DT))
R_tilde = R * np.sqrt(DT) / sigma

# 4. prepend a dummy zero column -> (M, 128), same length as the price panel
X = np.hstack([np.zeros((R.shape[0], 1)), R_tilde])
```

The **dummy zero column** exists purely so the model input has the same sequence length (128) as
the price series. It carries no information (it is the same constant for every path) and is
**discarded** on the way back. Train the generator on `X`; sample `X_gen ∈ ℝ^{M×128}`.

### Inverse (model output → price)

```python
# 1. drop the dummy column, undo the vol scaling
R_tilde_gen = X_gen[:, 1:]                 # (M, 127)
R_gen = R_tilde_gen * sigma / np.sqrt(DT)  # SAME frozen sigma as the forward pass

# 2. re-integrate to a price path anchored at S0 = 100
S_gen = np.empty((R_gen.shape[0], 128))
S_gen[:, 0] = S0
S_gen[:, 1:] = S0 * np.exp(np.cumsum(R_gen, axis=1))
```

Every generated path therefore starts at exactly `S0 = 100` (so `S_gen[:,0].std() == 0`),
matching the real data convention. `sigma` is **the same scalar** in the forward and inverse
directions — it is estimated once on the train split and never re-estimated on test/disc/generated.

---

## How `sigma` is estimated from SBTS (the exact estimator)

SBTS uses a **single pooled standard deviation** of the raw log-returns, computed on the
**train split only** (seed 0), and freezes it. The one line that defines it, verbatim from SBTS:

```python
sigma = float(R.std())     # R = log(S[:,1:]/S[:,:-1]),  R.shape == (M, 127)
```

- `.std()` is NumPy's **population** standard deviation (`ddof=0`).
- It is **pooled**: one scalar over all `M × 127` returns, not per-path or per-timestep.
- It is estimated **once, on train (seed 0)**, and reused unchanged for the inverse transform of
  every seed's generated paths. This mirrors SBTS exactly and keeps the scaling a fixed property
  of the dataset rather than something the generator can drift.

**Value estimated on the train split (seed 0, 4096 × 128):**

| Quantity | Value |
|----------|-------|
| `DT` | 0.004000  (= 1/250) |
| `sqrt(DT)` | 0.063246 |
| pooled count | 4096 × 127 = 520 192 returns |
| **`sigma = R.std()`** | **0.01263163** |
| `std(R_tilde)` (check) | 0.063246  (= `sqrt(DT)`, by construction) |

Why scale to `sqrt(DT)`? After `R_tilde = R·√dt/σ`, the scaled returns have unit-per-`√dt`
variance, i.e. `std(R_tilde) == sqrt(DT)` exactly. This puts the series on the natural
Brownian scale the generators expect, independent of the absolute price level.

**Reproduce the value:**

```bash
cd dataset/Heston/preprocessing_with_log_returns
python generate_datasets.py       # prints DT, sqrt(DT), sigma=std(R), and the scaling check
```

---

## Datasets

See [`dataset/Heston/preprocessing_with_log_returns/README.md`](../../../dataset/Heston/preprocessing_with_log_returns/README.md)
for the full split table and file list. Summary:

| Split | Seed | N | Role |
|-------|:----:|:----:|------|
| train | 0 | 4096 | only data a generator sees (after preprocessing) |
| test | 1 | 4096 | held-out real reference for every A/B metric |
| disc | 2 | 4096 | A18/A19 discriminative/predictive classifiers |
| ps | 3 | 512 | fresh path-shadowing queries, independent of the 1M bank |

---

## Methods

| Method | Status | Folder | Verdict |
|--------|--------|--------|---------|
| LS4 | ✅ done (5 seeds + 1M PS bank) | [`LS4/`](LS4/README.md) | log-returns **hurt** LS4 — 30/34 A-metrics regress, seeds destabilize; only A28 kurtosis-ratio improves. On the matched 4096/seed-0 control the preprocessing wins the stylised facts **19–16**. |
| CSDI | ✅ done (5 seeds + 1M PS bank) | [`CSDI/`](CSDI/README.md) | **mirror-opposite of LS4** — log-returns **fix** the vol/tail/adversarial facts (A9 −70%, A31 −48%, A32 −42%, A18-GRU halved) but the `cumsum→exp` reconstruction **drifts the terminal +3.25%**, breaking price-location (A13, A17, A25, grid_tvd). Seed-stable. Matched control: raw wins **19–16** on row-count, log-returns win the facts. |

Each method folder mirrors `results/Heston/<method>/`: `code/`, `losses/`, `weights/`,
`generated_paths/seed_{0..4}/`, `plots/`, `path_shadowing/`, and a per-method `README.md` with
the A1–A34 table, B-curve table, diagnostics figure, and Path-Shadowing-MC CRPS table.

---

## Per-method diagnostics — log-return preprocessing vs raw price (side-by-side)

For each method the **same** 8-panel stylised-facts diagnostic is shown twice at the **matched
4096-path / seed-0 control**: the **log-return preprocessing** run (left) against the **raw-price,
no-preprocessing** baseline (right, `baseline_no_preproc/`). Same real test set on both sides — the
only difference is the input transform, so any divergence between the two panels is the preprocessing's
effect in isolation. Follow each method's `README.md` for the per-metric Δ% head-to-head behind these
pictures.

### LS4

| log-return preprocessing (with) | raw price, no preprocessing |
|:-------------------------------:|:---------------------------:|
| ![LS4 logret diagnostics](LS4/plots/heston_diagnostics.png) | ![LS4 raw diagnostics](LS4/baseline_no_preproc/plots/heston_diagnostics.png) |

Preprocessing recovers the ACF-of-|r| / ACF-of-r² and rolling-vol curves the raw-price VAE flatlines
(A21 −66%, A22 −55%); the raw model keeps a tighter return-std marginal. Net **19–16 for log-returns**.

### CSDI

| log-return preprocessing (with) | raw price, no preprocessing |
|:-------------------------------:|:---------------------------:|
| ![CSDI logret diagnostics](CSDI/plots/heston_diagnostics.png) | ![CSDI raw diagnostics](CSDI/baseline_no_preproc/plots/heston_diagnostics.png) |

Preprocessing tightens the return-histogram / QQ marginal and the vol block (A9 −62%, A31 −42%,
A32 −48%), but the `cumsum→exp` **terminal drift (+3.25%)** hands the raw model every price-location
row (A13, A17, A25, grid_tvd). Net **19–16 for raw** on row-count — the mirror of LS4.
