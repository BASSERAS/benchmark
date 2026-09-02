# Weights — the reference SDE, frozen

Unlike `SBTS/weights/`, this directory is **not** empty-with-an-explanation. The
reference method genuinely has parameters; there are just very few of them, and
none of them were fitted here.

| File | Contents |
|------|----------|
| `reference_kernel.json` | the fitted reference SDE: 7 covariance features, 7 off-diagonal, 5 drift, plus the two half-life pairs and the clipping bounds |
| `reference_fit_history.csv` | the 300-step calibration/validation NLL trace (31 logged rows) |
| `SHA256SUMS` | checksums of both, so a silent divergence from the source is detectable |

## Where these came from

They are **byte-identical copies** of

```
Deep-MKV-TS/code/reference/reference_kernel.json
Deep-MKV-TS/code/reference/reference_fit_history.csv
```

produced by `Deep-MKV-TS/code/fit_reference_multiasset.py`, `fitted_at`
**2026-08-26T00:54:11**, by a 300-step Gaussian-NLL fit on the training split
only (80/20 internal calibration/validation split, `validation_fraction = 0.2`,
`train_file = heston_ma_S_8192x252x8.npy`). Final calibration NLL **−10.336229**,
validation NLL **−10.384663**.

They are copied rather than symlinked so this folder is self-contained the way
`SBTS/`, `LS4/` and `CSDI/` are — a reader who takes only
`results/HestonMultiAsset/reference/` still has everything needed to regenerate
the bank. `SHA256SUMS` exists because a copy can drift from its source without
anything failing:

```bash
# from results/HestonMultiAsset/reference/
sha256sum -c weights/SHA256SUMS
cmp weights/reference_kernel.json ../Deep-MKV-TS/code/reference/reference_kernel.json
```

If the second command reports a difference, the reference SDE was refitted after
this bank was generated and **the bank must be regenerated** — the numbers in
`../metrics_summary.csv` would otherwise describe a model that no longer exists.

## Why there is no neural checkpoint

There is nothing to serialise. The reference method builds the full Deep-MKV-TS
network — all **56 136** parameters, exactly the count `train_multiasset.py` logs
as `[model] parameters=56136` — and then never takes a gradient step. Algorithm 1
zero-initialises the final layer of both output heads:

```
expected_adjoint_next_head.2.weight        (8, 96)    all zero
expected_adjoint_next_head.2.bias          (8,)       all zero
expected_adjoint_noise_next_head.2.weight  (64, 96)   all zero   <- Zhat, d^2 = 64
expected_adjoint_noise_next_head.2.bias    (64,)      all zero
```

A zero weight **and** a zero bias in the last layer means the head's output is
`W @ h + b = 0` for every hidden state `h`, whatever the GRU beneath it computes.
So `Zhat == 0` exactly, the control

```
Theta = eta * (sigma^ref)^-1 + Zhat / sqrt(dt)
```

reduces to `eta * (sigma^ref)^-1`, and the sampler integrates `sigma^ref` itself.
Serialising 56 136 numbers that are either untrained noise (the GRU, whose output
is multiplied by zero downstream) or literally zero would be storage without
information — the reference SDE is fully specified by `reference_kernel.json`
plus the architecture in `Deep-MKV-TS/code/train_multiasset.py:build_model`.

`code/generate_reference_multiasset.py` **verifies** the four tensors above are
zero before sampling and refuses to write a bank if any is not, so the claim on
this page is enforced rather than asserted. The measured maxima are recorded per
seed in `../generated_paths/seed_{i}/metadata.json` under `zhat_head_max_abs`.

## What is in `reference_kernel.json`

| Key | Value |
|-----|-------|
| `state_dim` | 8 |
| `dt` | 0.003968253968253968 (= 1/252) |
| `num_steps` | 251 |
| `trend_half_lives`, `activity_half_lives` | 2 each |
| `activity_update` | `structural_variance` |
| `trend_weight`, `activity_weight` | 0.798126, 0.237690 |
| `initial_activity` | 0.041312 (pooled mean r²/dt on train) |
| `sigma_min`, `sigma_max` | 0.001, 0.6 — the spectral clip bounds |
| `ridge_covariance`, `ridge_drift` | 1e-4, 1e-3 |
| `gamma_diagonal`, `gamma_offdiagonal` | 7 coefficients each |
| `gamma_drift` | 5 coefficients |
| `covariance_feature_names`, `drift_feature_names` | the 7 and 5 feature labels |

> **Recorded, not resolved.** Whether 7 covariance + 5 drift features is the
> intended `φ^ref` set for d = 8 is an open question against the paper; see the
> "Open question, recorded rather than hidden" section of
> `../../Deep-MKV-TS/code/README.md`. It is noted here because *this* folder's
> entire score is that feature set — if it is wrong, the reference row is wrong,
> and so is the attribution argument built on it.

## Reproducing the bank

```bash
R=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
D=/home/tbasseras/benchmark/results/HestonMultiAsset/Deep-MKV-TS/code
cd results/HestonMultiAsset/reference/code
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
PYTHONPATH="$R/src:$R/experiments:$D" taskset -c 15 \
  /home/tbasseras/gpu-venv/bin/python generate_reference_multiasset.py
```

No GPU, one core, about two minutes per seed. Generation seeds are `90000 + i`,
imported from `Deep-MKV-TS/code/run_all_multiasset.py` rather than redefined, so
each seed draws the same Brownian stream as the corresponding Deep-MKV-TS bank.
