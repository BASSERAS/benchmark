# Diffusion-TS Code — Sources & Implementation

## Original work

| Field | Details |
|-------|---------|
| **Paper** | *Diffusion-TS: Interpretable Diffusion for General Time Series Generation* |
| **Authors** | Xinyu Yuan, Yan Qiao |
| **Venue** | ICLR 2024 |
| **Original code** | https://github.com/Y-debug-sys/Diffusion-TS |
| **Method type** | Non-autoregressive **denoising diffusion** (DDPM) with an interpretable seasonal-trend transformer decoder |

The verbatim reference implementation is kept under [`reference/`](reference/) for
transparency (`.git` stripped). `engine.solver.Trainer` (the exact DDPM training
loop), `Models.interpretable_diffusion.gaussian_diffusion.Diffusion_TS` (the exact
model) and `Models...model_utils.unnormalize_to_zero_to_one` are imported
**unmodified**; only the data plumbing (`train_heston.py`) is ours.

---

## Our implementation

`train_heston.py` is a thin data-plumbing wrapper around the **released**
`Diffusion_TS` model and `Trainer`. It loads the 8192×128 Heston price paths,
min-max scales them into the model's `[-1, 1]` space, trains one model per seed
with the authors' own DDPM loop (Adam betas [0.9, 0.96], grad-clip 1.0, EMA,
`ReduceLROnPlateauWithWarmup`), samples 8192 synthetic paths via
`ema.ema_model.generate_mts`, and writes weights / losses / generated paths /
metadata in the benchmark's standard layout. **The diffusion process, the
transformer encoder/decoder, the seasonal-trend decomposition, the Fourier loss,
and the sampler are the authors' own, untouched.** Because the Heston paths are
already windowed `(N, T)` arrays, the CSV-based `Utils.Data_utils` dataset is
bypassed and a bare `TensorDataset` is fed to the verbatim `Trainer` — the same
pattern used by `methods/FourierFlow`.

### Method overview (paper §3)

Diffusion-TS is a **non-autoregressive** DDPM that generates a whole length-T
series in one reverse-diffusion trajectory (no step-by-step roll-out). Two design
choices make it *interpretable*:

- **Direct $x_0$ prediction.** Instead of predicting the added noise $\epsilon$,
  the transformer predicts the clean signal $\hat{x}_0$ at every diffusion step,
  which makes the training target a reconstruction of the series itself.
- **Seasonal-trend decomposition.** The decoder reconstructs $\hat{x}_0$ as an
  explicit sum of a polynomial **trend** block and Fourier-based **seasonal**
  blocks, so each denoising output is a disentangled trend + seasonality signal.

$$x_0 \xrightarrow{\text{forward } q(x_t\mid x_0)} x_t \xrightarrow{\text{transformer}} \hat{x}_0 = \text{trend} + \text{seasonal}$$

Training minimises a **reweighted reconstruction loss** in both the time domain
and the frequency domain:

$$\mathcal{L} = \mathbb{E}_{t}\big[ w_t\, \lVert x_0 - \hat{x}_0 \rVert_1 + \lambda\, \lVert \mathrm{FFT}(x_0) - \mathrm{FFT}(\hat{x}_0) \rVert \big]$$

with a **cosine** $\beta$ schedule over **500** diffusion steps and an L1 base loss.

### Normalisation chain

The wrapper mirrors the repo's own `auto_norm` pipeline plus a price wrapper:

```
S(price) --minmax--> [0,1] --(*2-1)--> [-1,1]   (model trains here)
sample --unnormalize_to_zero_to_one--> [0,1] --minmax invert--> price
```

The MinMax is a **single global fit on the real Heston prices**
(`lo = 39.8936`, `hi = 155.5790`), stored in each `weights/seed_{i}_config.json`
so sampling inverts exactly back to price scale.

---

## Architecture choice — why `mujoco` (enc = 3, dec = 3)

The released Diffusion-TS ships **per-dataset** configs that differ **only** in
encoder/decoder depth and training length; everything else (`d_model=64`,
`timesteps=500`, cosine schedule, L1 loss, `n_heads=4`, `mlp_hidden_times=4`,
`kernel_size=1`) is identical across them. `train_heston.py` exposes the three
closest paper configs via `--arch`:

| `--arch` | enc | dec | max_epochs | params | Paper origin |
|----------|:---:|:---:|:----------:|:------:|--------------|
| `stocks` | 2 | 2 | 10000 | 368 397 | validated Stocks reproduction config |
| **`mujoco`** | **3** | **3** | **12000** | **544 147** | paper's seq_len≈100 config — closest to Heston's 128 |
| `etth`   | 3 | 2 | 18000 | 426 573 | paper's Table-3 long-term-generation config |

To pick between them we ran an **identical 3000-step smoke test** per arch on the
Heston data and scored each with **Context-FID** (the paper's own headline metric —
lower is better), holding every other hyperparameter fixed. The winner was
selected purely on this number.

### Three smoke results (Context-FID, 3000-step smoke, lower = better)

| `--arch` | enc/dec | **Context-FID** | Marg-mean err | Marg-std err | ACF-1 err |
|----------|:-------:|:---------------:|:-------------:|:------------:|:---------:|
| **mujoco** ✅ | 3 / 3 | **0.7367** | 0.0041 | 0.0235 | 0.0415 |
| etth | 3 / 2 | 2.3192 | 0.0322 | 0.0342 | 0.0899 |
| stocks | 2 / 2 | 36.0525 | 0.0109 | 0.3900 | 0.0498 |

Source: [`../paper_reimplementation/results/smoke_comparison.json`](../paper_reimplementation/results/smoke_comparison.json).

**`mujoco` wins by a wide margin** — its Context-FID (0.74) is 3× better than
`etth` (2.32) and ~49× better than `stocks` (36.05). The symmetric depth-3
encoder/decoder gives the seasonal-trend decoder enough capacity to fit Heston's
volatility-driven curvature that the shallow `stocks` config (enc/dec = 2)
cannot, while `etth`'s asymmetric enc=3/dec=2 under-powers the decoder that
actually reconstructs $\hat{x}_0$. The canonical 5-seed run therefore uses
`--arch mujoco` at its full **12000** steps.

---

## Hyperparameters (canonical `mujoco` run)

Identical to the paper's shared config; only depth + step count are arch-specific.

| Parameter | Value | Notes |
|-----------|-------|-------|
| `n_layer_enc` | 3 | mujoco depth |
| `n_layer_dec` | 3 | mujoco depth |
| `d_model` | 64 | shared across all paper configs |
| `timesteps` | 500 | diffusion steps (cosine β) |
| `sampling_timesteps` | 500 | full DDPM sampling (no DDIM skip) |
| `loss_type` | l1 | reweighted L1 + Fourier FFT loss |
| `beta_schedule` | cosine | |
| `n_heads` | 4 | |
| `mlp_hidden_times` | 4 | |
| `kernel_size` / `padding_size` | 1 / 0 | |
| `max_epochs` | 12000 | mujoco training length |
| `patience` | 3000 | `ReduceLROnPlateauWithWarmup` |
| `batch_size` | 128 | |
| `base_lr` | 1e-5 | warmup_lr 8e-4, warmup 500 |
| `ema_decay` | 0.995 | update every 10 steps |
| `gradient_accumulate_every` | 2 | |

---

## How to change hyperparameters

`--arch` selects a depth/length preset; the remaining knobs are CLI flags on
`train_heston.py`:

| Flag | Default | Effect |
|------|---------|--------|
| `--arch` | `etth` | Depth/step preset: `stocks` / `mujoco` / `etth`. Canonical run uses `mujoco`. |
| `--seed` | 0 | Re-seeds torch + numpy (model init + sampling). |
| `--steps` | 0 | Override `max_epochs` (0 = arch default). |
| `--batch_size` | 128 | Training batch. |
| `--gen_num` | 8192 | Synthetic paths to sample. |
| `--frac` | 1.0 | Fraction of training paths (for quick smoke runs). |
| `--tag` | "" | Run tag (e.g. `smoke`) — prefixes output dirs so smoke runs never collide with canonical ones. |

```bash
# canonical mujoco seed 0
cd methods/DiffusionTS/code
PYTHONPATH=reference /home/tbasseras/gpu-venv/bin/python train_heston.py --arch mujoco --seed 0

# 3000-step smoke of a different arch
PYTHONPATH=reference /home/tbasseras/gpu-venv/bin/python train_heston.py \
  --arch etth --seed 0 --steps 3000 --tag smoke
```

---

## How to use a different dataset

`train_heston.py` takes `--data PATH` — a `.npy` of shape `(N, T)`, dtype float,
in price/level space. The wrapper min-max scales to `[0,1]` (stores
`scale_min`/`scale_max` in the config for exact inversion), maps to `[-1,1]`,
trains, samples, and de-normalises back to the original scale before saving. No
odd-length constraint (unlike Fourier Flow): the transformer handles any `T`.

---

## How to produce new seeds

Each seed re-seeds torch + numpy, so model init and sampling differ:

```bash
cd methods/DiffusionTS/code
# one extra seed
PYTHONPATH=reference /home/tbasseras/gpu-venv/bin/python train_heston.py --arch mujoco --seed 5

# all 5 canonical seeds, 2 GPUs in parallel (mujoco, ~15 min/seed on an A100)
for s in 0 1 2 3 4; do
  g=$((s % 2 + 1)); c=$((g*8))
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 taskset -c $c-$((c+7)) \
    PYTHONPATH=reference /home/tbasseras/gpu-venv/bin/python train_heston.py \
      --arch mujoco --seed $s &
done
wait
```

Each seed writes:
- `weights/seed_{s}_model.pt` — model + EMA `state_dict`, arch, minmax
- `weights/seed_{s}_config.json` — full hyperparameters + scaling constants
- `losses/seed_{s}_losses.csv` — per-logged-step diffusion loss (`step,loss`)
- `generated_paths/seed_{s}/generated_paths_8192x128.npy` — (8192, 128) price scale
- `generated_paths/seed_{s}/metadata.json` — seed, shape, min/max, train time, params

---

## Sanity check

The full 12000-step canonical run (seed 0, `--arch mujoco`, ~878 s wall-clock on
one A100) confirms healthy training. Seed 0 signals:

```
[mujoco seed 0] params=544147 steps=12000 N=8192 T=128
loss  100 -> 3.863   ...   12000 -> 0.0729   (min 0.0681)
generated price[min=41.36 max=142.44]   scale[39.89, 155.58]   no NaN
```

Expected sane signals (all five seeds, verified):
- diffusion loss **decreases smoothly** from ~3.9 to ~0.07 and plateaus — the L1 +
  Fourier reconstruction loss converges, it does not diverge or NaN
  (`first_nan_step = null`, `gen_has_nan = false` in every metadata.json);
- generated price range (≈ 41–142) sits **inside** the real range (39.9–155.6);
- 544 147 parameters, 500-step DDPM sampling, EMA weights used for generation.

---

## Reproduce

```bash
# Environment: gpu-venv (torch, CUDA). Diffusion-TS trains on GPU.
cd methods/DiffusionTS/code

# All 5 Heston seeds (mujoco, 2 GPUs in parallel — see "How to produce new seeds")

# Compute metrics
cd /home/tbasseras/benchmark
/home/tbasseras/gpu-venv/bin/python metrics/compute_all.py \
  --method DiffusionTS --dataset Heston
```

The paper reproduction (Stocks len-24, the paper's own 4 metrics vs Table 1)
lives separately in `../paper_reimplementation/` — see its README.
