# There are no weights

SBTS is **non-parametric**. There is no network, no parameter vector, no gradient
descent, no checkpoint. This directory exists so the tree matches every other
method entry in the benchmark and so a reader who goes looking for `weights/`
finds an answer instead of an absence.

## What plays the role of a checkpoint

The complete, sufficient description of the fitted model is:

| Component | Value on this run | Where it lives |
|---|---|---|
| Training array | `(6144, 128, 8)` float64 prices | `dataset/TrueDataset/variants/om_2022-07_N6144/true_S_6144x128x8.npy` |
| Bandwidth `h` | `0.07` (units of √Δt) | `../losses/seed_<S>_bandwidth.json` |
| Markovian order `K` | `1` | same |
| Euler substeps `N_pi` | `50` | same |
| Per-asset σ | 8 floats, estimated from the train split | `../generated_paths/seed_<S>/metadata.json` |

Given those five things, generation is fully determined up to the RNG seed. That
is the whole "model".

## Why the training array is not stored here

It is the dataset, not an artefact of fitting. Copying it into `weights/` would
duplicate 50 MB of gitignored data and, worse, would let the two copies drift:
a re-built dataset would silently no longer match the array the results were
computed on. The build is pinned by directory name instead
(`variants/om_2022-07_N6144`), and `dataset_stats.json` in that directory records
the exact UTC window of every split.

## Consequence for the memorisation question

Because the training array *is* the model, "memorisation" is not a failure mode
SBTS can avoid by construction — it is a **tunable**, controlled entirely by `h`.
Small `h` means the kernel's support ball contains few training increments and the
generated path tracks one of them; large `h` over-smooths. This is why the
selection criterion for this dataset minimises `|log NNratio|` rather than any
goodness-of-fit statistic, and why the best-fitting `(h, K)` in the grid was
rejected. See `../losses/selection_criterion.md`.
