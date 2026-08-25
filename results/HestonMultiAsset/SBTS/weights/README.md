# No weights — SBTS is non-parametric

SBTS (Alouadi, Barreau, Carlier & Pham, ICAIF 2025, [arXiv:2503.02943](https://arxiv.org/abs/2503.02943))
estimates the Schrödinger-bridge drift by a **kernel-weighted conditional expectation over the
training set itself**. There is no neural network, no parameter vector, no gradient step and
therefore no checkpoint to serialise: the "model" *is* `dataset/HestonMultiAsset/heston_ma_S_8192x252x8.npy`
plus the three hyperparameters `(h, K, N_pi)`.

This directory is kept so that every method under `results/HestonMultiAsset/` presents the same
four slots — `code/`, `generated_paths/`, `losses/`, `weights/` — and a reader looking for a
checkpoint finds this explanation instead of an unexplained absence. (The d = 1 benchmark's
`methods/SBTS/` ships an empty `weights/` for the same reason.) The reproducible configuration
lives in:

| File | Contents |
|------|----------|
| `../losses/seed_{0..4}_bandwidth.json` | the `(h, K, N_pi, dt)` used for that seed |
| `../losses/bandwidth_selection.json`   | the d = 8 bandwidth sweep and the selection criterion |
| `../losses/generation_time.csv`        | wall-clock time per seed |
| `../generated_paths/seed_{i}/metadata.json` | per-asset scaling `sigma`, shapes, min/max |

Given the training array and those hyperparameters, `code/run_all_multiasset.py` reproduces
every generated path exactly (the per-worker RNG seeds are derived deterministically from the
run seed).
