# `reference/code/` — sampling σ<sup>ref</sup> with the control switched off

This folder produces the **starting-line** bank for the TrueDataset panel: 5 × 6 144 paths of
length 128 in d = 8, drawn from the frozen reference SDE of Deep-MKV-TS Algorithm 1 with the
neural control provably inactive. It is the "before" half of the attribution argument — see
[`../README.md`](../README.md) for how it reads against `real_floor/` and `Deep-MKV-TS/`.

> **The bracket on this panel is one-sided, and that is not a defect to be fixed.**
> On `HestonMultiAsset` there is a `perfect_recovery/` row that re-draws from the *true* Heston
> SDE, and the reference sits between that ceiling and the trained method. TrueDataset has no
> such row and **cannot** have one: a real market has no generating law to re-draw from. The
> comparable object here is [`real_floor/`](../../real_floor/), which is real-vs-real — one half
> of the actual data scored against the other half — a different *kind* of quantity from a draw
> off a known SDE. So this folder supplies a starting line and nothing supplies a ceiling.
> Reading the reference row as "how much room was left" imports an interpretation this dataset
> does not license.

## The five questions (GUIDELINE §2)

**1. Does the method natively accept `(N, T, d)` input?**
Yes, and unusually literally. σ<sup>ref</sup> is a *d-dimensional* SDE, not d univariate ones:
`state_dim = 8`, the diffusion is an 8×8 matrix built from **7 diagonal + 7 off-diagonal**
covariance coefficients, and the drift couples assets through cross-sectional features
(`mean_k V_k`, `mean_k T_k`). One model, all 8 channels, one joint law. Nothing here is an
ensemble of 8 univariate fits.

That matters more on this panel than on Heston. §2 records that the 28 realised cross-asset
correlations average **0.609** (range 0.515–0.801) — a per-asset ensemble would discard the
majority of the dependence structure. The off-diagonal block exists precisely to carry those 28
numbers, which is why **A20 Covariance Error is the row that actually tests this folder**.

**2. Does any hyperparameter fail to cross datasets?**
Yes — three of them, and they were changed rather than carried across. Nothing was *swept*
(there is one fit, one result), so the question lands on the constants:

| Constant | Heston | Here | Why it does not cross |
|----------|--------|------|-----------------------|
| `dt` | 1/252 | 9.512937595129376e-07 | 30-second bars, 1 051 200/year. `1/√dt = 1025.28`, **64.6×** Heston's |
| `sigma_max` | 0.6 | 5.0 | the clip is in annualised vol units; 0.6 would clip essentially the whole spectrum at this `dt` |
| freeze guard | `validation_nll > -9.0` | **deleted** | −9.0 is a Heston-scale number and the NLL scale moves with `dt`; it would reject every fit here |

The freeze guard was **replaced, not simply dropped**: the fit is rejected unless it beats the
i.i.d. Gaussian baseline. Measured `baseline_nll = −0.002319`, `validation_nll = −1.426201`,
**gain = 1.423882 nats**. That gain is the entire claim that this σ<sup>ref</sup> carries
structure rather than fitting noise. The 5.0 ceiling is likewise audited rather than asserted:
it binds on **0.24 %** of eigenvalues, the lower clip on none, and the spectral p99 sits at 3.24
— well inside it. Full table in [`../weights/README.md`](../weights/README.md).

The coefficients themselves were **refitted for this panel, not carried over**. See
[`../weights/README.md`](../weights/README.md) for the fit provenance and checksums.

> **Open, recorded rather than hidden.** Whether 7 covariance + 5 drift features is the intended
> φ<sup>ref</sup> set for d = 8 is an unresolved question against the paper — see the "Open
> question, recorded rather than hidden" section of
> [`../../Deep-MKV-TS/code/README.md`](../../Deep-MKV-TS/code/README.md). This folder's entire
> score *is* that feature set. If it is wrong, the reference row is wrong, and so is the
> attribution argument built on top of it.

**3. What is the memorisation risk?**
**Arithmetically zero**, and the diagnostic ships anyway because §9 says it ships *whatever it
says*. σ<sup>ref</sup> has **25 fitted numbers** — 7 covariance-diagonal + 7 off-diagonal + 5
drift (= 19 γ coefficients), plus 2 trend half-lives, 2 activity half-lives and 2 mixing weights
— against 6 144 × 128 × 8 = **6 291 456** training values, i.e. ~**251 658 values per
coefficient**. Nothing else is fitted: the network is built and never trained, and
`generate_reference_true.py` *verifies* `Ẑ == 0` before writing a path. Sampling touches no
training path at all; it rolls the fitted drift forward from `x0 = log(100)` on fresh Gaussian
increments.

So `measure_memorisation.py` runs here **not to test this generator but to calibrate the
estimator**. It reports what `NNratio` reads for a generator that certainly did not memorise, on
this dataset, in this metric, at this sample size: **1.5094 ± 0.0124**. Without that null,
"method X scored 0.9" is unreadable — 0.9 could be mild copying or simply what a clean model
scores here. The estimator is byte-identical to `SBTS/code/measure_memorisation.py` and
`CSDI/code/measure_memorisation.py`, which is the whole point: the columns have to be comparable.

The denominator is **`val`, never `test`** (§9.1). `test` sits behind a 45.5-day embargo in a
later market regime, so NN(test → train) measures regime shift rather than novelty; the measured
gap is 0.9316, so using `test` would inflate every ratio by ~7 % for a reason that has nothing to
do with any generator.

**4. Compute budget.**
Generation: **26.3–26.7 s per seed on ONE CPU core**, measured, recorded per seed in
[`../losses/generation_time.csv`](../losses/generation_time.csv). No GPU, no CUDA runtime — the
workload is a sequential 127-step Euler loop over `(6144, 8, 8)` tensors and is
kernel-launch-bound, not compute-bound. The five seeds run as five concurrent one-core processes,
so wall-clock for the whole bank is ~27 s.

Metrics are the expensive half and **do** need a GPU, because `compute_all_multiasset.py` trains
A18 discriminators and A19 predictors; §2 budgets ~8.5 min per seed. The measured figure for this
folder is in [`../logs/metrics.log`](../logs/metrics.log). Any "no GPU needed" claim covers
**generation only**; stating it unqualified would be wrong.

**There is no conditional-CRPS pool and no table C on this page.** §8's pool is 4 seeds × 8 192
paths, and this folder never produced one. Rendering an empty table C would claim a measurement
that was never made, so `render_readme.py` omits the section outright rather than filling it with
dashes.

**5. Which era does your model see?**
**Train only.** `fit_reference_true.py` reads `true_S_6144x128x8.npy` and splits it **internally
80/20** into calibration and validation. It never opens `true_S_val_*`, `true_S_valdisc_*`,
`true_S_disc_*` or `true_S_test_*`. The build is holdout-era and this row is compliant with room
to spare: the model is selected on 20 % of the *train* array, so even the past-era splits that §2
would permit are untouched. Sampling reads no data at all.

---

## Files

| File | What it does |
|------|--------------|
| `generate_reference_true.py` | builds the model, **verifies the control is zero**, samples 5 seeds, applies the §4 S0 contract |
| `merge_generation_time.py` | joins the five per-seed timing CSVs into `../losses/generation_time.csv` |
| `run_metrics_true.sh` | the exact A1–A32 + curve-B invocation, pinned to GPU 3 and cores 96–111 |
| `measure_memorisation.py` | NNratio diagnostic; estimator byte-identical to SBTS/CSDI, only the prose differs |
| `plot_diagnostics_true.py` | stylised-facts figure (copied from Deep-MKV-TS; `--method` defaults to `reference`) |
| `render_readme.py` | regenerates [`../README.md`](../README.md) from the artefacts — every number read from disk |

There is **no** `train_true.py` and **no** `<method>_model.py`. That is not an omission to be
filled in later: training is what this folder deliberately does not do, and the model is imported
wholesale from Deep-MKV-TS rather than redefined. `run_pipeline.sh` is likewise absent because
generation *is* `generate_reference_true.py`; splitting a 27-second sampler into a driver plus a
module to satisfy a template would be structure without content.

## How the zero-control claim is enforced

Algorithm 1 zero-initialises the **final layer of both output heads** — weight *and* bias:

```
expected_adjoint_next_head.2.weight        (8, 96)    all zero
expected_adjoint_next_head.2.bias          (8,)       all zero
expected_adjoint_noise_next_head.2.weight  (64, 96)   all zero   <- Zhat, d^2 = 64
expected_adjoint_noise_next_head.2.bias    (64,)      all zero
```

A zero weight **and** a zero bias mean the head emits `W @ h + b = 0` for *every* hidden state
`h`, whatever the GRU beneath it computes. So `Ẑ ≡ 0` exactly — not approximately, not "small" —
the control

```
Theta = eta * (sigma^ref)^-1 + Zhat / sqrt(dt)
```

reduces to `eta * (sigma^ref)^-1`, and the sampler integrates σ<sup>ref</sup> itself.

`_assert_zero_control()` reads those four tensors out of the built model and **raises
`SystemExit` before a single path is written** if any is absent or non-zero. The measured maxima
are recorded per seed in `../generated_paths/seed_{i}/metadata.json` under `zhat_head_max_abs` —
all four are `0.0` for all five seeds — alongside `zero_control_verified: true` and
`n_trained_parameters: 0`. `n_parameters: 56136` cross-checks against the
`[model] parameters=56136` line that `Deep-MKV-TS/code/train_true.py` logs, confirming the
architecture is the same one Deep-MKV-TS trains. The *only* difference between the two folders is
the gradient steps.

## Deviations and things done on purpose

**Constants are imported, not copied.** `S0`, `EXPECTED_SHAPE` and `GEN_SEED_BASE` come from
`Deep-MKV-TS/code/generate_bank_true.py` via `PYTHONPATH`; `DATASET`, `DT`, `SEQ_TAG` and
`STATE_DIM` from `fit_reference_true.py`. Duplicating them would let the two folders drift apart
silently. Importing them also buys something better than convenience — reference seed *i* draws
the **same Brownian stream** as Deep-MKV-TS seed *i*, so the comparison between the two columns
is **paired** rather than merely averaged, and a paired difference is resolvable at 5 seeds where
an unpaired one frequently is not. Both campaigns run `{0, 1, 2, 3, 4}`, so all five are paired.

**`build_model` returns `(model, sigma_max)`, not a bare model.** The spectral ceiling is read
out of the frozen `reference_kernel.json` rather than hardcoded, so each `metadata.json` records
which ceiling was actually in force (`sigma_max: 5.0`) instead of leaving a reader to assume it.

**`num_steps` is derived, not assumed.** 127 comes from the train split's own length, not from a
literal. The Heston file could hardcode 251; here the array decides.

**`eigh_float64` is installed as well as `eigh_fallback`.** The batched `torch.linalg.eigh`
inside `project_theta_to_sigma` can fail on cuSOLVER; `eigh_fallback` retries on CPU *only* when
Θ is finite and hard-fails otherwise, so an exploding control can never be laundered into a
"backend problem". `eigh_float64` is the second, dataset-specific shim: **0.891 %** of the real
σ<sup>ref</sup> matrices on this panel have a relative eigenvalue gap at or below float32 eps, and
in float32 the eigh *backward* returns NaN on those. Sampling takes no gradient, so it could not
have fired here — the counters in `metadata.json` confirm `n_calls_failed: 0`,
`n_fallback_cpu: 0`, `n_fallback_cpu_float64: 0`. It is installed anyway because a sampler whose
numerics differ from the trainer's undermines the very comparison it exists to support.

**The S0 rescale is mandatory, not cosmetic.** The model runs in float32, so
`exp(log(100)) = 100.00000762939453`. §4 requires `S[:, 0, :] == 100.0` exactly, so
`generate_reference_true.py` applies `prices *= S0 / prices[:, 0:1, :]` and then pins row 0.
Multiplying an entire path by a constant is exactly a shift in log-price, so **every log-return
is bit-identical** before and after. The largest pre-rescale residual was **6.395e-06**, recorded
in each `metadata.json` as `s0_max_abs_residual_before_rescale` — measured, not assumed. The bank
itself is written `float64`.

**Per-seed timing CSVs are merged, not appended.** The five seeds run as five concurrent
processes; five processes appending to one CSV interleave and corrupt it. Each writes
`generation_time_seed_{i}.csv` and `merge_generation_time.py` joins them, aborting on a wrong row
count, a missing or duplicated seed, or a shape disagreement between seeds.

**Zero edits to the frozen package.** Nothing under `methods/Deep-MKV-TS/code/reference/` is
modified. Everything in this folder is method-local.

**The weights are copied, not symlinked.** `../weights/` holds byte-identical copies of the
Deep-MKV-TS originals so this tree stands alone, with `SHA256SUMS` to detect drift. See
[`../weights/README.md`](../weights/README.md) for the checksums and the `cmp` command that
catches a silent refit.

## Run it

```bash
R=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
D=/home/tbasseras/benchmark/results/trueexperiment/Deep-MKV-TS/code
cd /home/tbasseras/benchmark/results/trueexperiment/reference/code

# generation: five concurrent one-core CPU processes, ~27 s wall
for s in 0 1 2 3 4; do
  OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
  PYTHONPATH="$R/src:$R/experiments:$D" taskset -c $((58 + s)) \
    /home/tbasseras/gpu-venv/bin/python generate_reference_true.py --seeds $s &
done
wait
/home/tbasseras/.cc-venv/bin/python merge_generation_time.py
```

`losses/generation_time_seed_{i}.csv` is written per seed, so a crash at seed 4 still leaves
seeds 0–3 auditable. Metrics, memorisation, figure and the rendered page: the `## Reproduce`
block of [`../README.md`](../README.md).
