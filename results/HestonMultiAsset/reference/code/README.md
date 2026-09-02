# `reference/code/` — sampling σ<sup>ref</sup> with the control switched off

This folder produces the **starting-line** bank for the d = 8 benchmark: 5 × 8 192 paths drawn
from the frozen reference SDE of Deep-MKV-TS Algorithm 1, with the neural control provably
inactive. It is the "before" half of the attribution argument — see
[`../README.md`](../README.md) for the three-way bracket against `perfect_recovery/` and
`Deep-MKV-TS/`.

## The four questions (GUIDELINE §0)

**1. Does the method natively accept `(N, T, d)` input?**
Yes, and unusually literally. σ<sup>ref</sup> is a *d-dimensional* SDE, not d univariate ones:
`state_dim = 8`, the diffusion is an 8×8 matrix built from 7 diagonal + **7 off-diagonal**
covariance coefficients, and the drift couples assets through cross-sectional features
(`mean_k V_k`, `mean_k T_k`). One model, all 8 channels, one joint law. The off-diagonal block
exists precisely to carry the d(d−1)/2 = 28 spot correlations, which is why **A20 Covariance
Error is the row that actually tests this folder** rather than a footnote.

**2. Does any hyperparameter fail to cross dimension?**
None were chosen here, so the question shifts to the coefficients — and the honest answer is
that they were **refitted for d = 8, not carried over from d = 1**. The d = 1 reference has a
scalar σ<sup>ref</sup> with no off-diagonal block at all; the concept does not survive the
crossing, so `Deep-MKV-TS/code/fit_reference_multiasset.py` fits a fresh 19-coefficient set on
`heston_ma_S_8192x252x8.npy`. That fit uses an **internal 80/20 calibration/validation split of
the training array** — it never touches `heston_ma_S_val_*.npy`, and never touches test. No
sweep is recorded because nothing was swept: there is one fit, one result, archived as
[`../weights/reference_fit_history.csv`](../weights/reference_fit_history.csv).

> **Open, recorded rather than hidden.** Whether 7 covariance + 5 drift features is the
> intended φ<sup>ref</sup> set for d = 8 is an unresolved question against the paper — see the
> "Open question, recorded rather than hidden" section of
> [`../../Deep-MKV-TS/code/README.md`](../../Deep-MKV-TS/code/README.md). This folder's entire
> score *is* that feature set. If it is wrong, the reference row is wrong, and so is the
> attribution argument built on top of it.

**3. What is the memorisation risk?**
**Arithmetically zero**, and the diagnostic ships anyway. σ<sup>ref</sup> has **19 fitted
coefficients** (7 covariance + 7 off-diagonal + 5 drift) against 8192 × 252 × 8 ≈ 16.5 M
training values — ~870 000 values per coefficient. Nothing else is fitted: the neural network
is built but never trained, and `generate_reference_multiasset.py` *verifies* `Ẑ == 0` before
sampling. Sampling touches no training path at all; it rolls the fitted drift forward from
`x0 = log(100)` driven by fresh Gaussian noise.

So `measure_memorisation.py` is run here **not to test this generator but to calibrate the
estimator**: it reports what `nn_ratio` reads for a generator that certainly did not memorise,
on this dataset, in this metric, at this sample size. Without that null, "method X scored 0.9"
is unreadable — 0.9 could be mild memorisation or simply what a clean model scores here. The
estimator is byte-identical to `SBTS/code/measure_memorisation.py` and
`LS4/code/measure_memorisation.py`, which is the whole point: the columns have to be comparable.

**4. Compute budget.**
Generation: **~80 s per seed on ONE CPU core**, ~7 min for all 5 — no GPU, no CUDA runtime.
Metrics: **~11 min per seed on GPU**, because `compute_all_multiasset.py` trains A18
discriminators (2 000 steps) and A19 predictors (5 000 steps). The "no GPU needed" claim covers
generation only; stating it unqualified would be wrong.

---

## Files

| File | What it does |
|------|--------------|
| `generate_reference_multiasset.py` | builds the model, **verifies the control is zero**, samples 5 seeds, applies the §4 S0 contract |
| `plot_diagnostics_multiasset.py` | 8-panel stylised-facts figure (copied from Deep-MKV-TS; `--method` defaults to `reference`) |
| `measure_memorisation.py` | NN-ratio diagnostic; estimator byte-identical to SBTS/LS4, only the prose differs |
| `render_readme.py` | regenerates `../README.md` from the artefacts — every number read from disk |

There is **no** `train_multiasset.py` and **no** `<method>_model.py`. That is not an omission to
be filled in later: training is what this folder deliberately does not do, and the model is
imported wholesale from Deep-MKV-TS rather than redefined. `run_all_multiasset.py` is likewise
absent because generation *is* `generate_reference_multiasset.py`; splitting a short sampler
into a driver plus a module to satisfy a template would be structure without content.

## How the zero-control claim is enforced

Algorithm 1 zero-initialises the **final layer of both output heads** — weight *and* bias:

```
expected_adjoint_next_head.2.weight        (8, 96)    all zero
expected_adjoint_next_head.2.bias          (8,)       all zero
expected_adjoint_noise_next_head.2.weight  (64, 96)   all zero   <- Zhat, d^2 = 64
expected_adjoint_noise_next_head.2.bias    (64,)      all zero
```

A zero weight **and** a zero bias mean the head emits `W @ h + b = 0` for *every* hidden state
`h`, whatever the GRU beneath it computes. So `Ẑ ≡ 0` exactly — not approximately, not "small"
— the control

```
Theta = eta * (sigma^ref)^-1 + Zhat / sqrt(dt)
```

reduces to `eta * (sigma^ref)^-1`, and the sampler integrates σ<sup>ref</sup> itself.

`_assert_zero_control()` reads those four tensors out of `model.checkpoint_state()` and
**raises `SystemExit` before a single path is written** if any is absent or non-zero. It was
negative-controlled during development: perturbing one entry by 1e-9 makes it fire. The measured
maxima are recorded per seed in `../generated_paths/seed_{i}/metadata.json` under
`zhat_head_max_abs`, alongside `zero_control_verified: true` and `n_trained_parameters: 0`.
`n_parameters: 56136` cross-checks against the `[model] parameters=56136` line that
`Deep-MKV-TS/code/train_multiasset.py` logs, confirming the architecture is the same one
Deep-MKV-TS trains — the *only* difference between the two folders is the gradient steps.

## Deviations and things done on purpose

**Constants are imported, not copied.** `S0`, `EXPECTED_SHAPE` and `GEN_SEED_BASE` come from
`Deep-MKV-TS/code/run_all_multiasset.py` via `PYTHONPATH`. Duplicating them would let the two
folders drift apart silently; importing them also buys something better than convenience —
reference seed *i* draws the **same Brownian stream** as Deep-MKV-TS seed *i*, so the
comparison between the two columns is **paired** rather than merely averaged.

**The S0 rescale is mandatory, not cosmetic.** The model runs in float32, so
`exp(log(100)) = 100.00000762939453`. GUIDELINE §4 requires `S[:, 0, :] == 100.0` exactly, so
`generate_reference_multiasset.py` applies `prices *= S0 / prices[:, 0:1, :]` and then pins row
0. Multiplying an entire path by a constant is exactly a shift in log-price, so **every
log-return is bit-identical** before and after. The largest pre-rescale residual across all
seeds was **6.395e-06**, recorded in each `metadata.json` as
`s0_max_abs_residual_before_rescale` — measured, not assumed.

**`eigh_fallback` is installed.** The batched `torch.linalg.eigh` inside
`project_theta_to_sigma` can fail on cuSOLVER; the shim retries on CPU *only* when Θ is finite,
and hard-fails otherwise so that an exploding control can never be laundered into a "backend
problem". It is installed here for parity with the training runs even though this folder runs on
CPU and has never triggered it — a sampler that behaved differently from the trainer would
undermine the very comparison it exists to support.

**Zero edits to the frozen package.** Nothing under
`methods/Deep-MKV-TS/code/reference/` is modified. Everything in this folder is method-local.

**The weights are copied, not symlinked.** `../weights/` holds byte-identical copies of the
Deep-MKV-TS originals so this tree stands alone, with `SHA256SUMS` to detect drift. See
[`../weights/README.md`](../weights/README.md) for the checksums and the `cmp` command that
catches a silent refit.

## Run it

```bash
R=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
D=/home/tbasseras/benchmark/results/HestonMultiAsset/Deep-MKV-TS/code
cd /home/tbasseras/benchmark/results/HestonMultiAsset/reference/code

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
PYTHONPATH="$R/src:$R/experiments:$D" taskset -c 15 \
  /home/tbasseras/gpu-venv/bin/python generate_reference_multiasset.py
```

`losses/generation_time.csv` is rewritten after **every** seed, not once at the end, so a crash
at seed 4 still leaves seeds 0-3 auditable. Full pipeline including metrics, figures and the
comparison table: the `## Reproduce` block of [`../README.md`](../README.md).
