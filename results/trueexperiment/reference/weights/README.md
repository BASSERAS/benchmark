# Weights — the reference SDE, frozen

Unlike `SBTS/weights/`, this directory is **not** empty-with-an-explanation. The
reference method genuinely has parameters; there are just very few of them, and
none of them were fitted here.

| File | Contents |
|------|----------|
| `reference_kernel.json` | the fitted reference SDE: 7 covariance-diagonal features, 7 off-diagonal, 5 drift, plus the two half-life pairs and the clipping bounds |
| `reference_fit_history.csv` | the 300-step calibration/validation NLL trace (31 logged rows) |
| `SHA256SUMS` | checksums of both, so a silent divergence from the source is detectable |

## Where these came from

They are **byte-identical copies** of

```
results/trueexperiment/Deep-MKV-TS/code/reference/reference_kernel.json
results/trueexperiment/Deep-MKV-TS/code/reference/reference_fit_history.csv
```

produced by `Deep-MKV-TS/code/fit_reference_true.py`, `fitted_at`
**2026-08-27T07:24:25**, by a 300-step Gaussian-NLL fit (`lr = 0.01`, `seed = 0`)
on the training split only — an 80/20 internal calibration/validation split,
`validation_fraction = 0.2`, `train_file = true_S_6144x128x8.npy`,
`num_paths = 6144`, `num_steps = 127`, on `cuda:0` under torch `2.13.0+cu130`.
Final calibration NLL **−1.444873**, validation NLL **−1.426201**.

They are copied rather than symlinked so this folder is self-contained the way
`SBTS/`, `LS4/` and `CSDI/` are — a reader who takes only
`results/trueexperiment/reference/` still has everything needed to regenerate the
bank. `SHA256SUMS` exists because a copy can drift from its source without
anything failing:

```bash
# from results/trueexperiment/reference/
sha256sum -c weights/SHA256SUMS
cmp weights/reference_kernel.json ../Deep-MKV-TS/code/reference/reference_kernel.json
```

If the second command reports a difference, the reference SDE was refitted after
this bank was generated and **the bank must be regenerated** — the numbers in
`../metrics_summary.csv` would otherwise describe a model that no longer exists.

## The two NLL numbers do not match the last row of the history — on purpose

The history's last row is

```
step,calibration_nll,validation_nll
300,-1.4439056068,-1.4262007206
```

and the kernel JSON records `validation_nll = -1.4262007206143001` (identical)
but `calibration_nll = -1.4448726930412596` (different in the 3rd decimal). That
is not a transcription error and not a nondeterminism. In
`Deep-MKV-TS/code/multivariate_reference.py`:

- the history row's `calibration_nll` is `loss.detach()`, captured **before**
  `optimiser.step()` runs for that step;
- the history row's `validation_nll` is evaluated **after** that step;
- both JSON figures are fresh `no_grad` evaluations taken **after** the
  best-validation `best_state` has been loaded back into the parameterisation.

Best validation was attained **at step 300**, so `best_state` *is* the final
state and the validation numbers agree exactly. The calibration numbers differ by
precisely one optimiser step. Recorded here because a reader who diffs the CSV
against the JSON will otherwise conclude one of them is wrong.

## Why there is no neural checkpoint

There is nothing to serialise. The reference method builds the full Deep-MKV-TS
network — all **56 136** parameters, exactly the count `train_true.py` logs as
`[model] parameters=56136` — and then never takes a gradient step. Algorithm 1
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
plus the architecture in `Deep-MKV-TS/code/train_true.py:build_model`.

`code/generate_reference_true.py` **verifies** the four tensors above are zero
before sampling and refuses to write a bank if any is not, so the claim on this
page is enforced rather than asserted. The measured maxima are recorded per seed
in `../generated_paths/seed_{i}/metadata.json` under `zhat_head_max_abs`; all
four are `0.0` for all five seeds.

## What is in `reference_kernel.json`

| Key | Value |
|-----|-------|
| `state_dim` | 8 |
| `dt` | 9.512937595129376e-07 (30-second bars, 1 051 200 bars/year) |
| `num_steps` | 127 |
| `trend_half_lives` | 9.383326, 76.296128 |
| `activity_half_lives` | 10.205451, 49.056079 |
| `activity_update` | `structural_variance` |
| `trend_weight`, `activity_weight` | 0.592768, 0.082653 |
| `initial_activity` | 0.710096 (pooled mean r²/dt on train) |
| `sigma_min`, `sigma_max` | 0.001, 5.0 — the spectral clip bounds |
| `ridge_covariance`, `ridge_drift` | 1e-3, 1e-2 |
| `gamma_diagonal`, `gamma_offdiagonal` | 7 coefficients each |
| `gamma_drift` | 5 coefficients |
| `covariance_feature_names` | `1`, `sqrt(V_j V_k)`, `(V_j+V_k)/2`, `mean_k V_k`, `T_j T_k`, `(T_j+T_k)/2`, `mean_k T_k` |
| `drift_feature_names` | `1`, `T_j`, `mean_k T_k`, `V_j`, `mean_k V_k` |

### Why `sigma_max` is 5.0 here and 0.6 on Heston

The clip bounds are in annualised volatility units, and this panel's `dt` is
9.512937595129376e-07 years against Heston's 1/252. `1/sqrt(dt) = 1025.28`, some
**64.6×** Heston's, so the same per-step move corresponds to a far larger
annualised number. A 0.6 ceiling would clip almost the whole spectrum. The fit
records what the ceiling actually binds on:

| Diagnostic | Value |
|------------|-------|
| `spectrum_min` | 0.142092 |
| `spectrum_mean` | 0.672312 |
| `spectrum_p99` | 3.240360 |
| `spectrum_max` | 5.000000 |
| `clip_saturation_high` | 0.002401 |
| `clip_saturation_low` | 0.0 |
| `saturation_num_paths` | 512 |

So the upper clip is active on **0.24 %** of eigenvalues and the lower clip on
none. The ceiling is a guard rail, not the model: the p99 sits at 3.24, well
inside 5.0.

### The Heston freeze guard was deleted, not adapted

`fit_reference_true.py` drops Heston's `validation_nll > -9.0` freeze guard
outright. That threshold was a Heston-scale number; carrying it over would have
rejected every fit on this panel, since the NLL scale moves with `dt`. It is
replaced by a scale-free test — the fit is rejected unless it beats the
i.i.d. Gaussian baseline:

| Quantity | Value |
|----------|-------|
| `baseline_nll` | −0.002319 |
| `validation_nll` | −1.426201 |
| `validation_gain_over_baseline` | 1.423882 |

A gain of 1.42 nats is the whole claim that this reference SDE carries structure
rather than fitting noise.

> **Recorded, not resolved.** Whether 7 covariance + 5 drift features is the
> intended `φ^ref` set for d = 8 is an open question against the paper; see the
> "Open question, recorded rather than hidden" section of
> `../../Deep-MKV-TS/code/README.md`. It is noted here because *this* folder's
> entire score is that feature set — if it is wrong, the reference row is wrong,
> and so is the attribution argument built on it.

## Reproducing the bank

```bash
R=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
D=/home/tbasseras/benchmark/results/trueexperiment/Deep-MKV-TS/code
cd results/trueexperiment/reference/code
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
PYTHONPATH="$R/src:$R/experiments:$D" taskset -c 58 \
  /home/tbasseras/gpu-venv/bin/python generate_reference_true.py --seeds 0
```

No GPU, one core, about **27 seconds** per seed. Generation seeds are `90000 + i`,
imported from `Deep-MKV-TS/code/generate_bank_true.py` rather than redefined, so
each seed draws the same Brownian stream as the corresponding Deep-MKV-TS bank.

The five seeds are run as five concurrent one-seed processes, then joined:

```bash
for s in 0 1 2 3 4; do ... generate_reference_true.py --seeds $s & done; wait
/home/tbasseras/.cc-venv/bin/python merge_generation_time.py
```

See the docstring of `code/merge_generation_time.py` for why the per-seed CSVs
are merged afterwards instead of five processes appending to one file.
