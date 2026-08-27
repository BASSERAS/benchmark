# Deep-MKV-TS on TrueDataset (real crypto, d = 8) — implementation notes

Deep McKean–Vlasov control for time-series generation, run on the locked
TrueDataset build `om_2022-07_N6144`, tag `6144x128x8`.

This file is the §2 pre-flight of `../../truedatasetguideline.md`. Read it before
reading the result tables; **question 3 below constrains what the memorisation
number is allowed to claim, and the constraint is tighter here than for any
sibling method.**

The rendered `../README.md` is produced by `render_readme.py` from artefacts.
This file is hand-written and is the only place the *reasoning* lives.

---

## The five §2 questions

### 1. Does the method natively accept `(N, T, d)` input?

**Yes. One joint model over all eight assets.** `weights/seed_i_config.json`
records `joint_or_per_asset: "joint"`, and that string is backed by the
architecture rather than asserted by it:

| Config key | Value | Why it settles the question |
|---|---|---|
| `state_dim` | 8 | the controlled SDE is a single process in `R^8` |
| `noise_dim` | 8 | one 8-dimensional Brownian driver, shared |
| `adjoint_dim` | 8 | |
| `noise_adjoint_dim` | **64 = 8 × 8** | the diffusion adjoint is a **full matrix**, not 8 scalars |

The control is
`MultivariateReferenceDriftSpecificEntropyMatrixControl`: at every path and every
one of the 127 steps it forms a **full 8 × 8 specific-entropy matrix `Theta`**
and eigendecomposes it. A per-asset ensemble has no such object — its
cross-asset covariance is diagonal by construction. Here the off-diagonal
entries are learned, so **A20 (the cross-asset covariance row) is a question this
architecture can actually answer.**

That matters on this dataset: the 28 realised cross-asset correlations average
**0.609** (range 0.515–0.801), so a per-asset ensemble would be discarding the
majority of the dependence structure before the first gradient step.

### 2. Does any hyperparameter fail to cross datasets?

**Yes — five of them. They moved a lot.** `paper_hyperparams: false`. Every move
below is forced by a measured property of this dataset, and every one is recorded
in `retuned_for_true`.

**The single root cause.** TrueDataset is 30-second bars:
`dt = 9.512937595129376e-07` years, `bars_per_year = 1051200`. So

```
1/sqrt(dt) = 1025.28     here (30-second crypto bars)
1/sqrt(dt) =   15.87     the d = 8 Heston sibling (daily)
                ratio  =  64.6x
```

Every quantity in this method that carries a `1/sqrt(dt)` — the diffusion scale,
the entropy ceiling, the MMD bandwidths — is therefore off by up to two orders of
magnitude if the Heston numbers are carried across. They are not knobs that
"might not transfer"; they provably cannot.

| Hyperparameter | Heston | TrueDataset | The measurement that forced it |
|---|---|---|---|
| `lr` | 1e-3 | **1e-2** | Heston's value **diverged** here — objective `+336.29` |
| `sigma_max` | 0.6 | **5.0** | 5 of 8 assets have realised vol above 0.6; the fitted spectrum has **p99 = 3.871** |
| `ridge_covariance` | — | **1e-3** | reference-kernel sweep, validation NLL |
| `ridge_drift` | — | **1e-2** | reference-kernel sweep, validation NLL |
| `ridge_lambda` | 1000 | **100** | conditional-expectation `ce_r2` goes **negative** at 1000 |
| MMD bandwidths | — | **regauged per block** | the 64.6× above; `losses/bandwidth_gauge.json` |

**`ridge_lambda`, the evidence.** A 5-point grid at 250 steps, scored by the
section-7 objective `|log NNratio|` on the validation split:

| `ridge_lambda` | `\|log NNratio\|` | `ce_projection_r2` | vol_err % | corr_err | |
|---|---|---|---|---|---|
| 1 | 0.3447 | +0.5712 | 47.34 | 0.2888 | |
| 10 | 0.7540 | +0.4846 | 63.96 | 0.4819 | |
| **100** | **0.2450** | **+0.5882** | 46.23 | 0.1646 | **selected** |
| 1000 | 2.6858 | **−0.0470** | 95.02 | 0.5450 | Heston's value |
| 10000 | 2.5248 | **−0.5289** | 94.74 | 0.6675 | |

The optimum is **interior** — 100 beats both neighbours, 10 (0.7540) and 1000
(2.6858) — so §7.2 is satisfied and this is an optimum rather than a boundary.
The committed 3-point grid `(100, 1000, 10000)` had 100 at its lower endpoint,
which is a boundary; adding 1 and 10 is what converted it into a bracketed
minimum, and is why `LAMBDAS` in `run_pipeline_true.sh` has five entries.

Note what `ce_projection_r2 < 0` means at Heston's value: the ridge-fitted
conditional expectation predicts its target *worse than that target's own mean*,
i.e. the regulariser has erased the very signal it exists to estimate, so the
adjoint never reaches the control.

**These are 250-step screen numbers, and every row is outside the section-7
envelope** — including the winner. That is expected at 8 % of the training
budget and is not a failure of the sweep; see the note on stage 3 in
`run_pipeline_true.sh`. The ceilings are enforced on **trained** checkpoints by
`select_checkpoint_true.py`, which is the only place they mean anything. Do not
quote this table as an envelope result.

**`sigma_max`, the honest caveat.** The confirm grid was `{5.0, 8.0}` and 5.0 is
its **lower endpoint**, so on the letter of §7.2 that is a boundary. Two facts
make it a benign one, both recorded in `losses/reference_selection.json`: the
fitted spectrum's `p99 = 3.8713` sits **below** 5.0, and only
`clip_saturation_high = 0.465 %` of the spectrum reaches the ceiling — under the
1 % admissibility cap. Both grid points converge to the same p99, so the ceiling
is very nearly inactive and moving it further up cannot change the model. Moving
it *down* would bind, which is why the grid was not extended downward.

**Selection pressure is therefore real and must be disclosed.** Unlike the CSDI
sibling — which retuned nothing and can call its diagnostics out-of-sample — five
numbers on this page were chosen by looking at a metric computed on this dataset.
All five were scored on **`val`/`valdisc`**, never on `test` or `disc`. See
question 3 for the one place this bites.

### 3. What is the memorisation risk?

**Real, measured, and — uniquely for this method — *not* an independent readout.
This is the most important disclosure in this file.**

The capacity argument that applies to a 413k-parameter diffusion model does not
apply here: this is **56 136 parameters** fitted to 6 144 training paths. The
model is small, and it has no bandwidth knob that can interpolate the training
increments. On capacity alone the risk is low.

**That is not the reason to be careful. This is:**

> `select_checkpoint_true.py` **selects the reported checkpoint by minimising
> `|log NNratio|`.** `measure_memorisation.py` then **reports NNratio** on the
> bank drawn from that checkpoint. The selection objective and the reported
> diagnostic are *the same statistic*.

So the NNratio in the rendered README is a quantity that was **optimised
towards**, not observed. It cannot be read as evidence that memorisation was
absent — it is close to 1 partly *because a checkpoint was chosen for being close
to 1*. Two things keep this from being circular, and both are limits rather than
absolutions:

- **The two are computed on different data.** Selection scores a small probe bank
  against the **validation** split; `measure_memorisation.py` scores the **full
  reported 6 144-path bank**. The second is strictly the harder test, so
  agreement between them is informative and disagreement is a finding.
- **`screen_envelope.py` is the independent check.** It re-screens the full
  reported bank against the section-7 ceilings, which selection only ever saw
  through a probe. A checkpoint that passed selection and fails the screen means
  the probe was optimistic — and that gap is itself a reportable result.

Denominator is **`val`, not `test`** (§9.1), pinned as the module constant
`DENOMINATOR_SPLIT = "val"` so that changing it is a visible edit. The build is
holdout-era and the test era sits *closer* to train than val does (0.932×) purely
because its returns are smaller, so a test denominator would inflate every ratio
by ~7 % and push a memorising generator towards the healthy end. Healthy band
**0.932–1.000**.

`n_exact_duplicates` is expected to be 0 and clears nothing — sampling a
continuous SDE lands on a bit-identical float64 path with probability zero. That
counter can only fire on a plumbing bug. **The ratio is the number that matters,
with the caveat above attached to it.**

### 4. Compute budget

Measured on this run, three seeds concurrent on three A100-SXM4-80GB, 8 pinned
cores each, `OMP_NUM_THREADS=8`:

| Stage | Cost |
|---|---|
| Training, 3 000 steps, N = 6 144, T = 128, d = 8 | **2.11–2.92 s/step → 105–146 min/seed** |
| Checkpoint selection, 6 checkpoints × probe bank | written by `select_checkpoint_true.py` |
| A/B bank, 6 144 paths | written by `generate_bank_true.py` |
| Conditional-CRPS pool, 8 192 paths | written by `generate_bank_true.py` |
| Metrics (`compute_all_multiasset.py`) | ≈ 8.5 min/seed |

The per-step spread across seeds is GPU contention, not variance in the work: the
three lanes share a host. The rendered `../README.md` carries the measured
wall-clock per seed from `losses/generation_time.csv` and
`weights/seed_i_config.json:train_time_sec`; **do not hand-copy the numbers above
into any results table** — they are a planning estimate taken mid-run.

Selection costs far more per point than a training step (it samples a bank and
computes envelope statistics), which is why `CHECKPOINT_STEPS` is a 6-point grid
and not a dense curve.

> **This run used 3 GPUs under an explicit, time-boxed override from the repo
> owner ("until the experiment completes"). The standing limit is 2. Do not
> treat the 3-lane layout in `run_seeds_true.sh` as the default.**

### 5. Which era does the model see?

**Train, val and valdisc. All three are past-era, so the run is valid.**

Verified by reading the loaders, not by trusting a config string:

| Script | Split(s) opened | Constant |
|---|---|---|
| `train_true.py` → `fit_reference_true.load_log_prices` | **train only** | `TRAIN_FILE = f"true_S_{SEQ_TAG}.npy"` |
| `select_checkpoint_true.py` | val, valdisc | `VAL_FILE`, `VALDISC_FILE` |
| `measure_memorisation.py` | val (denominator) | `DENOMINATOR_SPLIT = "val"` |

Guideline §2.5 permits `val`/`valdisc` in the selection loop precisely because
they are past-era. **`disc` and `test` are never opened by any script in this
directory that writes a weight or a bank.** They are read only by the metric
scripts, which is where they belong.

---

## Numerics — the float64 eigh patch, and why it is not a hyperparameter

Seeds diverged with an all-NaN `Theta` and no error. The root cause is **not**
eigenvalue ties at the `sigma_max` ceiling — that hypothesis was tested and
refuted: only 2 of 130 048 matrices have two or more eigenvalues at the ceiling,
and 0 of 910 336 eigenvalue pairs have a float64 gap below 1e-9.

The actual cause is float32 round-off in the eigh **backward**, which carries a
`1/(lambda_i - lambda_j)` factor. With float32 eps = 1.19e-07, **0.891 %** of the
frozen kernel's real `sigma_ref` matrices round two eigenvalues to identical
bits; the gap becomes exactly `0.0` and the gradient is NaN. Measured: **1 044 of
130 048 matrices (0.803 %) produce non-finite gradients.** One NaN then poisons
the entire batch, and `clip_grad_norm_` **rescales** finite gradients rather than
filtering them, so it cannot repair it.

Three call sites now run the decomposition in float64 (`eigh_float64.py`).
`max|grad|` was measured **identical** in both precisions at all three sites
(3.698e+00 / 1.510e+01 / 6.525e+01), which is the proof that this is a **pure
precision fix and not a modelling change** — no setting in `seed_i_config.json`
moves because of it. Per-site call counts land in the config key `eigh_float64`,
the justification in `eigh_float64_rationale`.

Separately, `MAX_EIGH_BATCH = 32768` because batched cuSOLVER `eigh` uses a
16-bit batch counter: **65 535 succeeds, 65 536 fails** with
`CUSOLVER_STATUS_INTERNAL_ERROR`. `eigh_fallback.py` counts any CPU-LAPACK
retries into the config key `eigh_fallback`; zero is expected, and non-zero does
not invalidate a run (same decomposition, other backend) but belongs in the audit
record rather than only in an uncommitted log.

---

## The frozen reference package and the fitted kernel

Two different things share the word "reference" and conflating them is the
mistake this section prevents.

**1. The author package — imported, never copied, never edited.**
`run_seeds_true.sh` puts it on the path:

```
REF=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
PYTHONPATH="$REF/src:$REF/experiments:$HERE"
```

274 `.py` files, tree `sha256_16 = e47b77d6814fa401`. **Zero edits are permitted
in that tree.** Verified no drift at the time of writing.

**2. `code/reference/` here — the *fitted* kernel, not code.** It holds two
artefacts and no Python at all:

| File | sha256_16 | Role |
|---|---|---|
| `reference_kernel.json` | `1826647cc692c74c` | the frozen Guyon–Lekeufack kernel |
| `reference_fit_history.csv` | | its fit trace |

The kernel is fitted **once**, on the train split, by `fit_reference_true.py`,
and every seed loads the identical file — so the reference measure is a constant
of the experiment rather than a per-seed nuisance parameter. Its digest is
recorded into every `seed_i_config.json` as `reference_kernel_sha256_16`, which
is what makes "five seeds share one reference measure" checkable rather than
asserted. Fitted values:

```
sigma_max          5.0            ridge_covariance   0.001
sigma_min          0.001          ridge_drift        0.01
state_dim          8              num_steps          127
dt                 9.512937595129376e-07
validation_nll    -1.4262007206143001   (baseline -0.002318924070943583)
calibration_nll   -1.4448726930412596
initial_activity   0.7100957304544644
trend_weight       0.5927682655119418
activity_weight    0.08265270288579064
activity_update    structural_variance
```

The MMD bandwidth gauge (`losses/bandwidth_gauge.json`,
`sha256_16 = bdb5c56666d42edc`) is likewise fitted once and digested into every
config as `bandwidth_gauge_sha256_16`.

---

## Anticipated result: where Deep-MKV-TS should lose, and why that is not a bug

Read **A14/A16 (the zero-return mass) per asset, never averaged.** TrueDataset is
30-second bars and on the thin assets a large share of bars have **exactly zero**
return — LINK 24.4 %, ADA 22.0 %, BNB 20.9 %. A controlled diffusion with a
non-degenerate `Theta` cannot place an atom at zero; it can only place a narrow
continuous bump there. That loss is a property of the model class, not of this
port. An eight-asset average would smear a structural failure on three assets
into a mild-looking aggregate.

The second thing to read sceptically is **realised volatility against correlation
error**. These fail *independently*, and the pattern is a diagnosis rather than a
score: a model that reproduces the 28 cross-asset correlations while halving
realised vol has learned the **shape** of the joint law and lost its **scale** at
the highest frequency — it is smoothing the bar-to-bar increments.
`screen_envelope.py` prints exactly that diagnosis when it fires. **Report the
two numbers separately; an average over them is meaningless.**

---

## Files

| File | Role |
|---|---|
| `fit_reference_true.py` | Fits the frozen Guyon–Lekeufack kernel **once** on the train split; `load_kernel`, `load_log_prices`. |
| `sweep_reference_true.py` | The `sigma_max` × `ridge_covariance` × `ridge_drift` grid behind `losses/reference_selection.json`. |
| `calibrate_bandwidths_true.py` | Regauges the MMD bandwidths per block for this `dt`. Writes `losses/bandwidth_gauge.json`. |
| `sweep_ridge_lambda_true.py` | The 5-point `ridge_lambda` grid above. Also owns `VAL_FILE`/`VALDISC_FILE`/`load_split`. |
| `train_true.py` | Fits one seed, writes six checkpoints, the loss CSV and `seed_i_config.json`. |
| `select_checkpoint_true.py` | **Section-7 selection.** Scores the six checkpoints on `val`, minimises `\|log NNratio\|` subject to the two ceilings, promotes the winner into the reported weights slot. |
| `selection_true.py` | The admissibility / objective / tie-break rule, isolated so it can be reasoned about on its own. |
| `generate_bank_true.py` | Draws the 6 144-path A/B bank and the 8 192-path CRPS pool from the *same* selected weights. |
| `collect_artifacts.py` | **A gate, not a report.** Recomputes the §4 contract from every `.npy` and exits non-zero on violation. Writes `losses/generation_time.csv`. |
| `measure_memorisation.py` | NNratio diagnostic (§9). Read question 3 before quoting it. |
| `screen_envelope.py` | Full-bank section-7 screen. **Byte-identical to `CSDI/code/screen_envelope.py` below the docstring** — the envelope is a dataset property, so the methods must be screened by identical arithmetic or their verdicts are not comparable. |
| `eigh_float64.py`, `eigh_fallback.py` | The numerics above. |
| `diagnose_theta_nan.py` | The instrument that produced the 0.891 % / 0.803 % measurements. |
| `matrix_control_multiasset.py`, `multivariate_reference.py` | The control and the reference drift. |
| `plot_losses.py` | `plots/loss_convergence.png`. **Row 1 is optimisation, row 2 is selection — they are not train-vs-val.** See its docstring. |
| `plot_diagnostics_true.py` | `plots/truedata_diagnostics.png`. |
| `render_readme.py` | Renders `../README.md` from the artefacts. **Never hand-edit the rendered file.** |
| `run_seeds_true.sh` | Queued 5-seed training across the lanes. |
| `run_pipeline_post.sh` | Everything after training, unattended: selection → banks → gate → metrics → memorisation → CRPS → figures → screen → README. |

---

## Reproducing

```bash
cd /home/tbasseras/benchmark/results/trueexperiment/Deep-MKV-TS/code
V=/home/tbasseras/benchmark/dataset/TrueDataset/variants/om_2022-07_N6144
PY=/home/tbasseras/gpu-venv/bin/python
REF=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
export PYTHONPATH="$REF/src:$REF/experiments:$PWD"

# 1. the reference kernel, fitted ONCE on train. Every seed loads this file.
CUDA_VISIBLE_DEVICES=1 $PY fit_reference_true.py

# 2. the MMD bandwidth gauge for this dt (1/sqrt(dt) = 1025.28, not Heston's 15.87)
CUDA_VISIBLE_DEVICES=1 $PY calibrate_bandwidths_true.py

# 3. train 5 seeds, queued across the lanes, detached
setsid nohup bash run_seeds_true.sh > ../losses/run_seeds.log 2>&1 < /dev/null & disown

# 4. everything after training, unattended
setsid nohup bash run_pipeline_post.sh > ../logs/pipeline_post.log 2>&1 < /dev/null & disown
```

Step 4 blocks until step 3 deletes `code/runs/.seed_queue`, then runs selection,
both banks, the contract gate, the metrics, the memorisation diagnostic, the 12
conditional-CRPS runs, the figures, the envelope screen and the README render.
`render_readme.py` is the last stage and takes no arguments.
