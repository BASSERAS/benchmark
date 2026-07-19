# Fourier Flow Code — Sources & Implementation

## Original work

| Field | Details |
|-------|---------|
| **Paper** | *Generative Time-series Modeling with Fourier Flows* |
| **Authors** | Ahmed M. Alaa, Alex J. Chan, Mihaela van der Schaar |
| **Venue** | ICLR 2021 |
| **Original code** | https://github.com/ahmedmalaa/Fourier-flows |
| **Method type** | Explicit-likelihood normalizing flow in the **frequency domain** |

The verbatim reference implementation is kept under [`reference/`](reference/)
for transparency (`.git` stripped). `SequentialFlows.FourierFlow` and its
`fourier/`, `filters/` helpers are imported **unmodified**; only the data
plumbing (`train_heston.py`) is ours.

---

## Our implementation

`train_heston.py` is a thin data-plumbing wrapper around the **released**
`FourierFlow` class. It loads the 8192×128 Heston price paths, min-max scales
them, trains one flow per seed by exact maximum likelihood, samples 8192
synthetic paths, and writes weights / losses / generated paths / metadata in the
benchmark's standard layout. **The flow, DFT, and coupling filters are the
authors' own, untouched.** The one addition is a thin subclass,
`FourierFlowClamp`, that copies the reference `fit()` **verbatim** and adds **two
numerical guards** that leave the flow math intact: (i) clamping the one
structurally-degenerate spectral bin's std (see *Fixes applied* #2), and (ii)
clipping the gradient norm to 1.0 (see *Fixes applied* #3). Guard (i) rescales a
degenerate normalisation constant; guard (ii) rescales gradients, exactly like
the `ExponentialLR` scheduler already in the reference loop. Neither touches the
DFT, coupling, base density, or loss.

### Method overview (paper §3)

A Fourier Flow maps a time series to independent Gaussian latents through the
**Discrete Fourier Transform** followed by learnable **spectral coupling**:

$$x \xrightarrow{\text{DFT}} X(f) \xrightarrow{\text{affine coupling}} z \sim \mathcal{N}(0, I)$$

- The DFT is a fixed linear (Vandermonde) map, so its **log-Jacobian is 0** — it
  contributes nothing to the likelihood but moves the signal to a basis where an
  affine coupling flow fits the spectrum well.
- Each flow layer is a **RealNVP affine coupling** on the (real, imag) spectral
  bins: half the bins condition scale/shift nets that transform the other half.
- Training is **exact max-likelihood**: `loss = (-log_pz - log_jacob).mean()`,
  full-batch, Adam + `ExponentialLR(γ=0.999)`.

### Architecture (released `FourierFlow`)

| Component | Definition |
|-----------|-----------|
| Spectral transform | `DFT` (`fourier/transforms.py`): per-row `fftshift(fft(x))`, cropped to `k = ⌊N/2⌋+1` real+imag bins, log-Jacobian 0 |
| Coupling net | `SpectralFilter` (`filters/spectral.py`): `Linear(in,hidden) → Sigmoid → Linear(hidden,hidden) → Sigmoid → Linear(hidden,out)` for both scale (`sig_net`) and shift (`mu_net`) |
| Flow | `n_flows` couplings with alternating `flip`, base `MultivariateNormal(d+1)` |

> **⚠️ Paper-vs-code discrepancy (documented, kept as-is).** The paper describes
> the coupling nets as **bidirectional RNNs**. The released `FourierFlow` uses
> **MLPs** (above); the BiRNN lives in a separate, *unused* `AttentionFilter` /
> `TimeFlow`. Per the locked task decision we run the **released code as-is** —
> the MLP path is what produced the paper's Table 2, and it reproduces it (see
> `../paper_reimplementation/`). We did not touch the model.

---

## Fixes applied

Four shims in `train_heston.py`; **none change the flow math** (DFT, coupling,
base density, loss are the authors' own). #1 and #4 are compatibility; #2 and #3
are numerical guards (a normalisation-constant rescale and a gradient rescale).

| # | Issue | Fix |
|---|-------|-----|
| 1 | **ODD `fft_size` required.** `FourierFlow.forward` reshapes the cropped DFT output (`2·crop_size = N+2` values for even N) to size `d+1 = N+1`. This is only consistent for **odd** input length. Even N=128 crashes: `shape '[-1,129]' invalid for size 8192×130`. | Prepend a single 0 "anchor" column → length **T+1 = 129 (odd)**, exactly the released Stocks pipeline (`real_data_loading` does `hstack((0, series))`). Strip the anchor back off after sampling. |
| 2 | **Degenerate imaginary-DC bin → literal 0/0 NaN.** For any real signal the imaginary-DC spectral component is identically 0, so its across-batch std is **exactly 0**. `normalize=True` then divides `(x−mean)/std = 0/0` → immediate NaN on the very first forward pass. Stocks (odd length 101) never hit this: odd-length FFT roundoff gave its imag-DC a tiny **nonzero** std that `normalize=True` rescaled to unit variance. | **Keep the paper's `normalize=True` regime** and clamp only the degenerate bin: `FourierFlowClamp.fit` sets that bin's `fft_std` from 0 → 1 (it then passes through as a constant 0, carrying no information) while every other bin normalises as in the paper. One bin is clamped for Heston (`n_clamped_bins` recorded in each `seed_*_config.json`). **Necessary but not sufficient** — it removes the first-pass NaN, but the run still diverged later (see #3). |
| 3 | **Late-training NaN from near-singular spectral covariance.** Even *with* #2, all five un-clipped 1000-epoch runs descended smoothly then went **NaN at epoch ~499** (grad-norm exploding to ~8300). Root cause is a **data property, not a code bug**: every Heston path starts at the *identical* deterministic S₀=100, so `Var(value at t=1)` across samples is ≈0 (**3.6e-15** vs Stocks **6.1e-2**) and the spectral covariance is near-singular (**92/130** bins with std<1e-3 vs Stocks 41/102). A max-likelihood flow matching a near-singular target drives its affine-coupling log-scales → −∞ along the degenerate directions, which eventually overflows. **Proof:** the *unmodified* reference flow (with #2 applied, no clip, 1000 epochs) is stable on Stocks (`first_nan=−1`, max grad-norm 2142) but NaN on Heston (`first_nan=499`, max grad-norm 8304) — same code, only the data differs. Stocks avoids it because sliding-window slicing gives every window a different start value, so no spectral direction is degenerate. | **Clip the gradient norm to 1.0** (`torch.nn.utils.clip_grad_norm_(params, 1.0)` between `loss.backward()` and `optim.step()`). This caps the log-scale blowup **without changing any flow math** — a pure gradient rescale, like the `ExponentialLR` scheduler already in the loop. Verified: **no NaN**, all 5 seeds converge to a positive NLL plateau ~98–121. Configurable via `fit(..., grad_clip=…)`. |
| 4 | **torch 2.13 / numpy 2.4 vs paper's torch 1.3 / numpy 1.18.** | `torch.distributions.Distribution.set_default_validate_args(False)` to match the old-torch default; deprecated `Variable(..., volatile=True)` is tolerated on torch 2.13. No math changed. |

---

## Hyperparameters (from the paper's Stocks config)

Fourier Flow's released driver ships **per-dataset** hyperparameters. The natural
template for Heston is the **Stocks** config (`FF_model_params["stock"]` /
`FF_train_params["stock"]`): both are **univariate price series**, so the same
width/depth/optimiser transfer directly. These are the defaults in
`train_heston.py`.

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `hidden` | 200 | Released Stocks width. Univariate price series ⇒ same coupling-MLP capacity as Stocks. |
| `n_flows` | 3 | Released Stocks depth. |
| `normalize` | **True** (paper regime, both) | Per-bin spectral standardisation, as in the paper. The one degenerate Heston bin is clamped (Fixes #2), not disabled; gradient clipping (Fixes #3) keeps the run stable. |
| `fft_size` | 129 = T+1 | Odd length required (Fixes #1). Stocks used 101 = 100+1. |
| `epochs` | 1000 | Released Stocks epoch count; full-batch exact MLE. |
| `batch_size` | 500 | **Ignored** — `fit()` is full-batch (kept for CLI parity). |
| `learning_rate` | 1e-3 | Released Stocks LR; Adam + `ExponentialLR(γ=0.999)`. |

The Stocks config's reproduction of Table 2 (F-score 0.9920 vs 0.984, MAE 0.0084
vs 0.009) is documented in `../paper_reimplementation/README.md`.

---

## How to change hyperparameters

All hyperparameters are CLI flags on `train_heston.py` (or env vars in
`train_all.sh`). Override any of them:

| Flag | env in `train_all.sh` | Default | Effect |
|------|-----------------------|---------|--------|
| `--hidden` | `HIDDEN` | 200 | Coupling-MLP width. Larger → more capacity, slower. |
| `--n-flows` | `NFLOWS` | 3 | Number of spectral coupling layers. |
| `--normalize` | `NORMALIZE` | 1 | Per-bin spectral standardisation (paper regime). `FourierFlowClamp` guards the degenerate bin, so 1 is safe. |
| `--epochs` | `EPOCHS` | 1000 | Full-batch MLE epochs. |
| `--lr` | `LR` | 1e-3 | Adam learning rate. |
| `--n-gen` | `NGEN` | 8192 | Number of synthetic paths to sample. |

```bash
# Example: deeper, wider flow, fewer epochs
cd methods/FourierFlow/code/reference
PYTHONPATH=$PWD /home/tbasseras/gpu-venv/bin/python ../train_heston.py \
  --seed 0 --hidden 300 --n-flows 5 --epochs 500 --normalize 1
```

---

## How to use a different dataset

`train_heston.py` takes `--dataset PATH`. Provide a `.npy` of shape `(N, T)`,
dtype float, in price/level space:

```bash
PYTHONPATH=$PWD /home/tbasseras/gpu-venv/bin/python ../train_heston.py \
  --seed 0 --dataset /path/to/your_NxT.npy --normalize 1
```

The wrapper handles the rest automatically:
1. min-max scales to [0,1] (stores `scale_min`/`scale_max` in the config for
   exact inversion);
2. prepends the 0 anchor → length **T+1**;
3. **if T+1 is even**, `FourierFlow.forward` will crash — pad your series to an
   **odd** post-anchor length (i.e. even T), or drop the anchor and pass an
   already-odd length. This is a hard constraint of the released code, not a bug
   in the wrapper.
4. Generated paths are de-normalised back to your original scale before saving.

---

## How to produce new seeds

Each seed re-seeds torch + numpy, so model init and sampling differ:

```bash
cd methods/FourierFlow/code/reference
# one extra seed
PYTHONPATH=$PWD /home/tbasseras/gpu-venv/bin/python ../train_heston.py \
  --seed 5 --normalize 1

# all 5 seeds in parallel (5 × 3 cores, CPU-only)
bash ../train_all.sh
```

Each seed writes:
- `weights/seed_{s}_model.pt` — full `state_dict`
- `weights/seed_{s}_config.json` — hyperparameters + scaling constants + `fft_size`, `prepend_anchor`
- `losses/seed_{s}_losses.csv` — per-epoch negative-log-likelihood
- `generated_paths/seed_{s}/generated_paths_8192x128.npy` — (8192, 128) price scale
- `generated_paths/seed_{s}/metadata.json`

---

## Reproduce

```bash
# Environment: gpu-venv (torch 2.13, numpy 2.4). Fourier Flow is CPU-only.
cd methods/FourierFlow/code/reference

# All 5 Heston seeds in parallel (~10 min, 5 × 3 cores)
bash ../train_all.sh

# Compute metrics
cd /home/tbasseras/benchmark
/home/tbasseras/gpu-venv/bin/python metrics/compute_all.py \
  --method FourierFlow --dataset Heston
```

The paper reproduction (Stocks, F-score + MAE vs Table 2) lives separately in
`../paper_reimplementation/` — see its README.

---

## Sanity check

The full 1000-epoch run (5 seeds in parallel, `bash ../train_all.sh`, ~8.6 min
wall-clock) confirms both guards work. Seed 0 header + result:

```
[seed 0] target (8192, 128) price[min=39.894 max=155.579]
[seed 0] normalize=True; 1 zero-variance spectral bin(s) will be clamped (std 0->1).
[seed 0] done train=490.0s gen=0.1s loss 210.69 -> 98.14 gen[min=46.72 max=139.78]
```

Expected sane signals (all five seeds, verified):
- exactly **1 clamped bin** reported (the degenerate imaginary-DC term, Fixes #2);
- **no NaN / no Inf** anywhere in the loss trace — gradient clipping (Fixes #3)
  removes the epoch-~499 blowup that killed every un-clipped run;
- NLL loss **decreases smoothly and plateaus at a positive value** (per-seed
  `loss_last` ∈ **[98, 121]** at 1000 ep) — it does **not** dive into the
  negatives;
- generated price range (≈ 43–149) sits **inside** the real range (39.9–155.6);
- generated `col0` mean ≈ **99.9–100.8** vs the real deterministic S₀ = 100.
  Fourier Flow is a *distributional* flow — it spreads the Heston paths'
  deterministic start into a tight distribution around 100. Documented in the
  results README.

> Absolute NLL differs from the Stocks reproduction (~23) and *should*: the loss
> is a **total** (un-normalised) NLL summed over `d+1` spectral dimensions, and
> Heston has 129 vs Stocks' 101. Per-dimension the two are the same order
> (~0.83 vs ~0.23). Only the **convergence behaviour** (smooth → positive
> plateau, no NaN) is comparable across datasets, and it matches.
