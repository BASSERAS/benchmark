# d = 8 bandwidth selection — criterion, fixed before the sweep finished

SBTS has three hyperparameters: `h` (kernel bandwidth), `K` (Markovian order) and
`N_pi` (Euler substeps). `K = 20` and `N_pi = 50` are the author's values for this
benchmark and are **carried over unchanged**. `h` is the only one that **cannot** be
carried across dimension: the kernel is radial, `K_h(x) = (h² − ‖x‖²)² · 1{‖x‖₂ < h}`,
so its support is a ball whose radius must scale with the typical distance between
d-dimensional increments. That distance grows like √d — the median pairwise distance
in the training set is `0.0372` at d = 1 and `0.2643` at d = 8. The author's
`h = 0.05` is therefore not a transferable constant, and is empirically catastrophic
here (see `bandwidth_selection.json`, first sweep: 41 % volatility error, kurtosis
45.7 against a target of 1.12, minimum price 0.0 — outright path collapse).

## Split discipline

`h` is scored on the **validation split** (`heston_ma_S_val_8192x252x8.npy`, seed 3),
**never** on the test split (seed 1). The test split is the real reference for every
reported metric; tuning a hyperparameter on it would leak. This is stricter than the
first exploratory sweep, which was scored against test and is retained in
`bandwidth_selection.json` only as a coarse orientation, not as the selection basis.

## Criterion

Stated before the h = 0.28 / 0.31 / 0.35 rows were available. Rows h = 0.20 and
h = 0.25 **were** already visible when this was written; that is disclosed rather
than concealed.

**Hard constraints** (a violation disqualifies, whatever the other numbers say):

1. no path collapse — `min(S_gen) > 10` and all prices finite and positive;
2. mean absolute per-asset volatility error `< 5 %`;
3. mean absolute per-asset excess-kurtosis error `< 20 %`.

**Primary objective:** among the survivors, **maximise the nearest-neighbour ratio**

    NNratio = median NN(generated → train) / median NN(val → train)

computed in log-return space. A perfect generator sits at `NNratio ≈ 1`: its samples
are no closer to the training set than genuinely held-out data is. `NNratio ≪ 1` is
memorisation — the generator is reproducing training paths. Larger `h` widens the
kernel support, spreads the weights over more training paths, and therefore *reduces*
memorisation, at the cost of degrading the stylised facts. The two hard constraints
above bound that cost; within them, the least-memorising bandwidth wins.

**Tie-break** (NNratio within 0.02): lowest mean cross-asset correlation error, since
the cross-asset correlation structure is the whole point of the multi-asset dataset.

## Why NNratio is the primary objective and not ESS

The obvious diagnostic — effective sample size `ESS = (Σwᵢ)² / Σwᵢ²` — is degenerate
here. At `K = 20`, **zero** training paths lie inside the kernel support, and this is
true at d = 1 with the author's own `h = 0.05` (median min-distance 0.188 ≫ 0.05)
exactly as much as it is at d = 8 (median min-distance 0.778 ≫ any h ≤ 0.40):

| config | paths in support, K=1 | K=5 | K=20 | median min-distance at K=20 |
|--------|----------------------:|----:|-----:|----------------------------:|
| d=1, h=0.05 (author) | 3558.0 | 41.9 | 0.0 | 0.1881 |
| d=8, h=0.25 | 5119.9 | 1.5 | 0.0 | 0.7777 |
| d=8, h=0.28 | 6004.1 | 9.5 | 0.0 | 0.7777 |
| d=8, h=0.30 | 6469.0 | 25.4 | 0.0 | 0.7777 |
| d=8, h=0.32 | 6845.6 | 60.1 | 0.0 | 0.7777 |
| d=8, h=0.35 | 7270.4 | 173.2 | 0.0 | 0.7777 |
| d=8, h=0.40 | 7715.0 | 639.8 | 0.0 | 0.7777 |

ESS therefore sits at its floor (`ESS = 1`, a single training path carrying all the
weight) in **both** dimensions, so it cannot discriminate between bandwidths. At
`K = 20` SBTS is a nearest-path-following scheme by construction; the generated
trajectory stays inside the kernel support only because the drift glues it to one
training path — which *is* the memorisation mechanism. This is a property of the
method as published, not an artefact of the d = 8 port, and it is **not** silently
patched here.

> Note also that the earlier prediction that SBTS would degenerate to Brownian motion
> at d = 8 (all-zero weights → zero drift) was **wrong**: the zero-weight fraction is
> 0.00 % everywhere. The failure mode is the opposite one, memorisation.

## What this means for reading the results

The `Perfect floor` column in the results README is an independent draw from the true
Heston law and therefore sits at `NNratio = 1` by construction. Any SBTS row whose
distributional metrics look *better* than that floor should be read as evidence of
memorisation, not of superiority. The measured `NNratio` at the selected `h` is
reported alongside the metrics for exactly this reason.
