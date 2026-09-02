#!/usr/bin/env python3
"""
render_dataset_readme.py
------------------------
Generate results/HestonMultiAsset/oldreadme.md -- the DATASET-level page shared
by every method benchmarked on the d = 8 multi-asset Heston data.

Why this file exists separately from the per-method READMEs
-----------------------------------------------------------
Each `results/HestonMultiAsset/<Method>/README.md` mirrors the d = 1 layout of
`methods/<Method>/README.md` exactly, so a reader can put the two side by side.
Anything that is a property of the *dataset* rather than of a *method* -- the
generating law, the metric scoping rules, the independent-draw floor, the
cross-method leaderboard and the memorisation diagnostic -- would be duplicated
(and would drift) if it were copied into every method page. It lives here once.

The leaderboard and memorisation tables are DISCOVERED, not hard-coded: any
directory under results/HestonMultiAsset/ that holds a metrics_summary.csv is
picked up automatically, so adding a method means running its metrics and
re-running this script. Nothing here needs editing when a method is added.

Reads
  results/HestonMultiAsset/<method>/metrics_summary.csv
  results/HestonMultiAsset/<method>/curve_b_aggregate.json
  results/HestonMultiAsset/<method>/losses/memorisation.json   (optional)
  results/HestonMultiAsset/perfect_recovery/*                  (the floor)

Writes
  results/HestonMultiAsset/oldreadme.md

Usage
    /home/tbasseras/gpu-venv/bin/python \
        results/HestonMultiAsset/tools/render_dataset_readme.py
"""

import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MA = os.path.dirname(HERE)                      # results/HestonMultiAsset
REPO = os.path.abspath(os.path.join(MA, "../.."))
FLOOR_NAME = "perfect_recovery"

# Metric direction. Everything not listed is "lower is better"; only these two
# rows are scored differently, and getting them wrong would silently invert the
# leaderboard, so they are named explicitly rather than inferred.
DIRECTION = {
    "A28_kurtosis_ratio": "none",   # perfect = 1.0, scored as |x - 1|
    "A33_sigma_corr": "up",         # higher is better
}

# The rows evaluated once on the full (N, T, 8) tensor rather than per asset.
NATIVE_LABEL = "A6-A11, A18, A20, A25"

SKIP_DIRS = {"tools", FLOOR_NAME, "__pycache__"}


def read_summary(path):
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["metric"]] = row
    return out


def read_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def _better(sm, fm, direction):
    if direction == "up":
        return sm >= fm
    if direction == "none":
        return abs(sm - 1.0) <= abs(fm - 1.0)
    return sm <= fm


def discover_methods():
    """Every subdir holding a metrics_summary.csv, floor and tools excluded."""
    out = []
    for name in sorted(os.listdir(MA)):
        p = os.path.join(MA, name)
        if not os.path.isdir(p) or name in SKIP_DIRS:
            continue
        if os.path.exists(os.path.join(p, "metrics_summary.csv")):
            out.append(name)
    return out


def score_method(method, floor_sum, floor_b):
    """Count A rows and B plots at or below the independent-draw floor."""
    d = os.path.join(MA, method)
    s = read_summary(os.path.join(d, "metrics_summary.csv"))
    b = read_json(os.path.join(d, "curve_b_aggregate.json"))
    mem = read_json(os.path.join(d, "losses", "memorisation.json"))

    a_at = a_tot = 0
    for k, row in s.items():
        f = floor_sum.get(k)
        if not f or row.get("mean") in (None, "") or f.get("mean") in (None, ""):
            continue
        try:
            sm, fm = float(row["mean"]), float(f["mean"])
        except (TypeError, ValueError):
            continue
        a_tot += 1
        if _better(sm, fm, DIRECTION.get(k, "down")):
            a_at += 1

    b_at = b_tot = 0
    for pre, blk in b.items():
        fb = floor_b.get(pre, {})
        if "mse" not in blk or "mse" not in fb:
            continue
        b_tot += 1
        if float(blk["mse"]["mean"]) <= float(fb["mse"]["mean"]):
            b_at += 1

    return {"method": method, "a_at": a_at, "a_tot": a_tot,
            "b_at": b_at, "b_tot": b_tot, "mem": mem}


def method_cell(method):
    """The method's name, linked only if the page it would point at exists.

    ``discover_methods`` keys on ``metrics_summary.csv``, so a directory that
    has been scored but never had its README rendered still earns a row here.
    Linking it unconditionally produced a 404 on github.com for anyone browsing
    the results without cloning. Testing for the file means the link reappears
    by itself the moment the page is written -- nothing to remember later.
    """
    if os.path.exists(os.path.join(MA, method, "README.md")):
        return f"[{method}]({method}/README.md)"
    return f"{method} _(no page yet)_"


def leaderboard(rows):
    if not rows:
        return "_(no method has metrics yet)_"
    out = ["| Method | A rows at/below floor | B plots at/below floor | NN memorisation ratio |",
           "|--------|----------------------:|-----------------------:|----------------------:|"]
    for r in sorted(rows, key=lambda x: (-x["a_at"], x["method"])):
        nn = r["mem"].get("nn_ratio")
        nn_s = f"{float(nn):.3f}" if nn is not None else "-"
        out.append(f"| {method_cell(r['method'])} | "
                   f"{r['a_at']} / {r['a_tot']} | {r['b_at']} / {r['b_tot']} | {nn_s} |")
    return "\n".join(out)


def memorisation_table(rows):
    have = [r for r in rows if r["mem"].get("nn_ratio") is not None]
    if not have:
        return "_(no memorisation diagnostic has been run yet)_"
    out = ["| Method | NN ratio | Std across seeds | Times closer than held-out data | Exact duplicates |",
           "|--------|---------:|-----------------:|--------------------------------:|-----------------:|"]
    for r in sorted(have, key=lambda x: float(x["mem"]["nn_ratio"])):
        m = r["mem"]
        v = float(m["nn_ratio"])
        sd = m.get("nn_ratio_std")
        sd_s = f"{float(sd):.4f}" if sd is not None else "-"
        out.append(f"| {method_cell(r['method'])} | {v:.4f} | {sd_s} | "
                   f"{1.0 / max(v, 1e-9):.1f}× | {m.get('n_exact_duplicates', '-')} |")
    return "\n".join(out)


def main():
    floor_sum = read_summary(os.path.join(MA, FLOOR_NAME, "metrics_summary.csv"))
    floor_b = read_json(os.path.join(MA, FLOOR_NAME, "curve_b_aggregate.json"))
    methods = discover_methods()
    rows = [score_method(m, floor_sum, floor_b) for m in methods]

    if not floor_sum:
        print("  [warn] no floor metrics yet -- leaderboard will be empty")

    md = f"""# Multi-Asset Heston (d = 8)

Benchmark dataset and results for **d = 8 correlated** Heston stochastic-volatility
price paths. This page holds everything that belongs to the *dataset* rather than to
any one generator: the law, the metric scoping rules, the independent-draw floor, the
cross-method leaderboard, and the memorisation diagnostic.

Per-method pages mirror the d = 1 layout of `methods/<Method>/README.md` exactly, so
the two can be read side by side:

{leaderboard(rows)}

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
  is marked *(native d=8)* in the per-method tables. This covers **{NATIVE_LABEL}**: the
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

{memorisation_table(rows)}

> **Zero exact duplicates does NOT clear a method.** A kernel generator interpolates
> between training paths, so it produces *near*-copies rather than bitwise copies. The
> ratio is the number that matters.

> **A high ratio is not a ranking.** This column must be read jointly with the A/B tables,
> never on its own. A generator whose paths miss the data manifold **in every direction**
> scores near 1.0 here by default — being far from the training set is trivial if you are
> also far from the truth. The ratio only carries information for a method that is
> *simultaneously* accurate, so the honest sentence for a method scoring high here while
> winning few A rows is "clean, for an uninteresting reason", not "better than the method
> above it". Concretely: SBTS sits at 0.219 and wins 29 of the 36 A rows, so its low ratio
> is a real interpolation property of the most accurate generator on this dataset; LS4
> (0.838) and CSDI (0.859) each win 0 A rows, so their high ratios say only that they are
> not copying — not that they are closer to the law.

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
   `K_h(x) = (h² − ‖x‖²)²·1{{‖x‖₂ < h}}`, following the reference
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

### LS4

Three deviations from the released d = 1 implementation, none of which changes the d = 1
numbers.

1. **Cauchy kernel patch** (`reference_models/s4.py` line 795). The naive pure-PyTorch
   kernel must sum over conjugate pole *pairs* to match the keops/CUDA path. This is not
   cosmetic: `model.generate` rolls the prior through `latent.step` in STEP mode, where
   the unpatched kernel disagrees with conv mode — so generation and training would run
   on different dynamics. Inherited from the d = 1 port, not introduced at d = 8.
2. **Invertible scaler.** The released `normalize_per_seq` preset has an identity decode
   that cannot map prior samples back to price scale. The d = 1 port replaced it with a
   global standardise; this port uses the **per-channel** version, because the eight
   assets have a 1.66× spread in price σ.
3. **S0 rescaling.** LS4 prior samples do not land on `S0 = 100` — at d = 1 the generated
   `S0` spread over `[99.30, 100.48]`, std 0.055. Each path is rescaled per asset. A
   per-path constant multiplier is exactly a shift of the log-price *level*, so every
   log-return is bit-identical and A1-A25 / A27-A34 are unaffected. A contract fix, not
   a performance fix — it cannot flatter the results.

### CSDI

Four deviations, all narrow. Hyperparameters are the released `config/base.yaml` verbatim:
`retuned_for_d8` is **empty** in all five configs.

1. **One vendored-file edit.** `reference/diff_models.py` had a top-level
   `from linear_attention_transformer import ...` reachable only through the forecasting
   path this benchmark never runs, which made the module unimportable without that
   dependency. Moved inside its function, marked in place, changes no computation.
2. **`generate()` generalised past asset 0.** The d = 1 wrapper ends its sampler with
   `samples[:, 0, 0, :]` — correct and invisible at K = 1, but at K = 8 it would have
   silently kept **asset 0 only** and returned a degenerate tensor. Replaced with
   `samples[:, 0].permute(0, 2, 1)`. A bug in the d = 1 wrapper's shortcut, and exactly
   the class of error a copy-paste port ships without noticing.
3. **Per-channel z-score**, as for LS4. At K = 1 the d = 1 global statistic *is* the
   per-feature statistic, so this is a scoping consequence rather than a retune, and the
   scaler is diagonal — the correlation matrix, which A20 scores, is invariant under it.
4. **S0 rescaling, with the clip counted.** The multiply preserves log-returns exactly;
   the `≤ 0` clip that precedes it does **not**, and a first price clipped to `1e-6`
   rescales an entire path by `1e8`. Both counters are therefore reported per seed rather
   than left implicit. Across all 5 seeds and 82 575 360 entries the clip **never fired**.

The parameter count moves 412 945 → 413 057 between d = 1 and d = 8, exactly **+112** =
`nn.Embedding(target_dim, 16)` growing from 1×16 to 8×16. Nothing else in the architecture
depends on the number of assets: the feature axis is handled by attention, not by width.

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
    ├── generated_paths/seed_{{0..4}}/   .npy gitignored, metadata.json tracked
    ├── losses/                training curves, or hyperparameter records
    ├── plots/                 heston_diagnostics, disc/pred loss figures
    └── weights/               checkpoints, or a README explaining their absence
```

1. Create `results/HestonMultiAsset/<Method>/` with the five slots above.
2. Generate 5 seeds of `(8192, 252, 8)` float64 paths into
   `generated_paths/seed_{{i}}/generated_paths_8192x252x8.npy`.
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
CUDA_VISIBLE_DEVICES=3 /home/tbasseras/gpu-venv/bin/python \\
    metrics/compute_all_multiasset.py --method perfect_recovery

# regenerate this page
/home/tbasseras/gpu-venv/bin/python \\
    results/HestonMultiAsset/tools/render_dataset_readme.py
```
"""
    out = os.path.join(MA, "oldreadme.md")
    with open(out, "w") as fh:
        fh.write(md)
    print(f"wrote {out}  ({len(md.splitlines())} lines)")
    print(f"methods discovered: {methods or '(none)'}")


if __name__ == "__main__":
    main()
