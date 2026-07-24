# GT-GAN Code — Sources & Implementation

## Original work

| Field | Details |
|-------|---------|
| **Paper** | *GT-GAN: General Purpose Time Series Synthesis with Generative Adversarial Networks* |
| **Authors** | Jinsung Jeon, Jeonghak Kim, Haryong Song, Seunghyeon Cho, Noseong Park |
| **Venue** | NeurIPS 2022 — PDF `../paper_reimplementation/GT-GAN_NeurIPS2022.pdf` |
| **arXiv** | [2210.02040](https://arxiv.org/abs/2210.02040) |
| **Original code** | https://github.com/Jinsung-Jeon/GT-GAN |
| **Original framework** | **PyTorch** (already PyTorch — no framework port needed) |
| **Method type** | **GAN with a continuous-time autoencoder.** A **Neural-CDE embedder** encodes an irregular/regular series into a latent path, a **Neural-ODE recovery** decodes it back, and a **continuous normalizing-flow (CNF) generator** samples the latent space adversarially against a **Neural-ODE discriminator**. The `gtgan` mode used here is the regular-sampling variant that reproduced the paper's Stocks/Energy Table 1. |

The verbatim reference implementation is kept under [`reference/`](reference/) for transparency
(`.git` stripped). The generative sub-networks (`GTGAN_stocks.{FinalTanh, NeuralCDE,
Multi_Layer_ODENetwork}`, the CNF builder `train_misc.build_model_tabular_nonlinear`, and the
CTFP/flow helpers in `ctfp_tools`) are imported **unchanged** by [`train_heston.py`](train_heston.py)
with the released `gtgan` hyperparameters (`parse_arguments` defaults). The one substantive edit is a
**seq\_len / effective\_shape de-conflation** in the CTFP forward pass (see Fixes below) — byte-identical
to the reference on the paper's Stocks case.

---

## Our implementation

The released code is already **PyTorch**, so — unlike the TF baselines — there is **no architecture
port**: [`train_heston.py`](train_heston.py) builds the four GT-GAN sub-networks from the released
`gtgan` settings and reproduces the upstream two-phase training loop (embed-pretrain → joint
adversarial), then generates 8192 length-128 paths and writes weights / losses / generated paths /
metadata in the benchmark's standard layout. It is a data-plumbing + logging wrapper: it loads the
8192×128 Heston price paths, **global min-max normalizes** them to [0,1], trains one GT-GAN per seed,
**generates** via prior noise → CNF generator → ODE recovery, de-normalizes back to price scale, and
clips to ≥ 1e-6.

### The four sub-networks (gtgan mode, feat\_dim = 1, seq\_len = 128)

| Component | Class | Role | Config |
|-----------|-------|------|--------|
| **Embedder** | `NeuralCDE(FinalTanh)` | Neural-CDE encoder `X → h` on a natural-cubic-spline path | `input_channels=1`, `hidden_channels=24`, `output_channels=24`, `num_layers=3` |
| **Recovery** | `Multi_Layer_ODENetwork` | Neural-ODE decoder `h → X̂` (Euler, `delta_t=0.5`) | `num_layer=2` (`r_layer`), `x_hidden=48`, `last_activation=tanh` |
| **Generator** | `build_model_tabular_nonlinear` | CNF sampling the 24-dim latent from prior noise | `dims="32-64-64-32"`, `solver=sym12async`, reg coeffs below |
| **Discriminator** | `Multi_Layer_ODENetwork` | Neural-ODE critic `X̂ → logit` (Euler, `delta_t=0.5`) | `num_layer=1` (`d_layer`), `x_hidden=48`, `last_activation=identity` |

Total **32 957 parameters** (`weights/seed_0_config.json`) — **the smallest model in the benchmark**.
The CNF generator operates in the **24-dim embedding latent** (`effective_shape = input_size = hidden
= 24`), *not* the raw 1-dim feature space, so it is unchanged by the feature-dim switch.

### Two-phase training loop

```
Phase 1 — embed pretrain   (first_epoch = 10 000 steps)
    loss_e = 10 · sqrt(MSE(recovery(embedder(X)), X))     # reconstruction only
Phase 2 — joint adversarial (max_steps = 3 000 steps)
    inner ×2:  update discriminator (only if loss_d > 0.15) + recovery
    every log_time=2 steps:  supervised generator update (CTFP likelihood, loss_s)
    each step:  adversarial generator update  loss_g = loss_g_u + 100·loss_g_v
```

`loss_g_u` = BCE(D(fake), 1); `loss_g_v` = |Δstd| + |Δmean| moment-matching between fake and real;
`loss_d` = BCE(D(real),1) + BCE(D(fake),0). **There is no separate supervisor network** — `loss_s` is
the CTFP supervised-latent likelihood term, not a sub-model. Losses logged every `log_every=100` steps
as `step, phase, loss_e, loss_d, loss_g_u, loss_g_v, loss_s`.

### Normalisation chain

```
S(price) --(X - min)/(max - min + 1e-7)--> [0,1]   (model trains here)
generated --* (max - min + 1e-7) + min--> price, then clip to >= 1e-6
```

A **global min-max** over the whole array (single scalar `data_min`/`data_max`, the paper's
`normalize` transform applied globally): `data_min = 39.893609`, `data_max = 155.578983` — the
real-data extremes, stored in each `weights/seed_{i}_config.json` so sampling inverts exactly back to
price scale.

---

## Fixes / adaptations vs the reference

The sub-network classes are used **unchanged** except for the CTFP de-conflation; the rest of the
adaptations are in the training *driver*, to fit the benchmark's single-GPU / npy / multi-seed layout:

1. **seq\_len / effective\_shape de-conflation (required at seq\_len ≠ 24).** *Location:*
   [`train_heston.py:120` `run_ctfp`](train_heston.py) (a faithful copy of reference
   `ctfp_tools.run_latent_ctfp_model5_3`). *Reference:* reads `values.shape[1]` as the latent feature
   dimension in two spots (latent-noise dim when `z=True`; the `transform_values` reshape when
   `z=False`). In the paper's Stocks/Energy setups `seq_len == hidden_size == effective_shape == 24`,
   so this silently worked. *Our change:* substitute `args.effective_shape` in exactly those two spots.
   **Both substitutions are no-ops when seq\_len == effective\_shape == 24 — byte-identical to the
   reference on the reproduced paper case.** This is the **only** change to reference-derived model code.
2. **Data I/O.** *Reference:* reads the paper datasets via `time_dataset` / `data_loading`. *Our change:*
   load the Heston `.npy` `(N, T)` directly, global min-max normalize to [0,1] (store `data_min`/
   `data_max`), build the `(N, T, 2)` [price, obs-time] tensor the CDE needs, export the generated `.npy`
   back in price scale.
3. **Time grid.** *Reference:* hardcoded `range(24)` / `final_index=23`. *Our change:* `range(seq_len)`
   / `final_index=seq_len-1` so the CDE/ODE integrate over all 128 steps.
4. **Per-seed seeding.** *Location:* top of `main`. *Our change:* seed torch + numpy + random from
   `--seed` so the 5 canonical seeds give distinct model inits and samples.
5. **Logging.** *Our change:* log the 2-phase `step, phase, loss_e, loss_d, loss_g_u, loss_g_v, loss_s`
   to CSV, save the four sub-network `state_dict`s + config + metadata.

No hyperparameter or objective change: the `gtgan`-mode architecture, the two-phase loop, and all
CTFP/CNF regularization coefficients are the released defaults.

---

## Hyperparameters (released gtgan mode / paper Stocks)

| Parameter | Value | Source |
|-----------|-------|--------|
| `hidden_size` | 24 | released gtgan |
| `num_layers` (CDE) | 3 | released gtgan |
| `x_hidden` | 48 | released gtgan |
| `effective_shape` (CNF latent) | 24 (= `input_size` = hidden) | released gtgan |
| `r_layer` (recovery ODE depth) | 2 | released gtgan |
| `d_layer` (discriminator ODE depth) | 1 | released gtgan |
| `last_activation_r` | `tanh` | released gtgan |
| `last_activation_d` | `identity` | released gtgan |
| `solver` (CNF) | `sym12async` | released gtgan |
| `atol` / `rtol` | 1e-3 / 1e-2 | released gtgan |
| `dims` (CNF hidden) | `32-64-64-32` | released gtgan |
| `reconstruction` | 0.01 | released gtgan |
| `kinetic_energy` | 0.05 | released gtgan |
| `jacobian_norm2` | 0.01 | released gtgan |
| `directional_penalty` | 0.01 | released gtgan |
| `batch_size` | 128 | released gtgan |
| `first_epoch` (embed pretrain steps) | 10 000 | paper Stocks |
| `max_steps` (joint adversarial steps) | 3 000 | paper Stocks |
| `log_time` (supervised-gen cadence) | 2 | released gtgan |
| `feat_dim` | **1** | **Heston-specific** — univariate price series |
| `seq_len` | **128** | **Heston-specific** — the Heston sequence length |

Only two knobs are Heston-specific (`feat_dim`, `seq_len`); everything else is the released
`gtgan`-mode preset kept verbatim.

---

## How to change hyperparameters

The `gtgan` preset is loaded through the released `parse_arguments()` (CTFP/CNF defaults) plus the
GT-GAN CLI flags added in [`train_heston.py`](train_heston.py):

| Flag | Default | Effect |
|------|---------|--------|
| `--seed` | 0 | Re-seeds torch + numpy + random (model init + sampling). |
| `--data` | `dataset/Heston/heston_S_8192x128.npy` | Training `.npy` of shape `(N, T)`, price scale. |
| `--first_epoch` | 10000 | Phase-1 embedding pretrain steps. |
| `--max_steps` | 8500 (**3000 used**) | Phase-2 joint adversarial steps. |
| `--batch_size` | 128 | Training batch. |
| `--gen_num` | 8192 | Paths to generate. |
| `--r_layer` / `--d_layer` | 2 / 1 | Recovery / discriminator ODE depth. |
| `--log_time` | 2 | Supervised-generator update cadence. |
| `--frac` | 1.0 | Fraction of training paths (smoke runs). |
| `--tag` | "" | Run tag (e.g. `smoke`) — prefixes outputs, skips canonical weights. |

- **CNF width / solver / regularization** → the `--dims`, `--solver`, `--atol`, `--rtol`,
  `--reconstruction`, `--kinetic_energy`, `--jacobian_norm2`, `--directional_penalty` flags from the
  released `parse_arguments`.
- **Latent / CDE width** → `hidden_size`, `num_layers`, `x_hidden` are set at the top of `main`.

---

## How to use a different dataset

`train_heston.py --data PATH` takes a `.npy` of shape `(N, T)`, dtype float, in price/level space. The
wrapper global-min-max-normalizes to [0,1] (stores `data_min`/`data_max` for exact inversion), builds
the `(N, T, 2)` [value, obs-time] tensor, trains, samples via CNF → recovery, and de-normalizes back
before saving. For a **multivariate** dataset, set `input_size = C` at the top of `main` (with `T` the
per-channel length); the CDE embedder then reads `C` input channels and the recovery emits `C`.

---

## How to produce new seeds

Each seed re-seeds torch + numpy + random, so model init and sampling differ:

```bash
cd methods/GT-GAN/code
# one extra seed
CUDA_VISIBLE_DEVICES=0 /home/tbasseras/gtgan-venv/bin/python train_heston.py --seed 5

# all 5 canonical seeds, 2 GPUs in parallel (~21–34 h/seed on an A100 — ODE-solver bound)
for s in 0 1 2 3 4; do
  g=$((s % 2 == 0 ? 0 : 3)); c=$(((s % 2)*8))
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 taskset -c $c-$((c+7)) \
    /home/tbasseras/gtgan-venv/bin/python train_heston.py --seed $s &
done
wait
```

Each seed writes:
- `weights/seed_{s}_model.pt` — embedder/recovery/generator/discriminator `state_dict`s + data_min/max/seed
- `weights/seed_{s}_config.json` — full hyperparameters + `data_min`/`data_max` + params (32 957)
- `losses/seed_{s}_losses.csv` — `step, phase, loss_e, loss_d, loss_g_u, loss_g_v, loss_s`
- `generated_paths/seed_{s}/generated_paths_8192x128.npy` — (8192, 128) price scale
- `generated_paths/seed_{s}/metadata.json` — seed, shape, min/max, gen/train time, params, gen_has_nan

---

## Sanity check

The canonical runs (5 seeds) confirm healthy two-phase training and NaN-free generation:

```
=== GT-GAN Heston  seed=0  device=A100-SXM4-80GB ===
[data] S(8192, 128) price[min=39.89,max=155.58]  dataset(8192, 128, 2)  first_epoch=10000 max_steps=3000 batch=128 eff_shape=24
[model] total=32957
Start Embedding Network Training ... Finish Embedding Network Training
Start Joint Training ... Finish Joint Training
[done] seed=0 gen=(8192,128) price=[0.00,151.35] nan=False train=77260.0s gen=4.5s
```

Expected sane signals (all five seeds, verified from metadata):
- 32 957 parameters; `gen_has_nan = false` on every seed;
- generated mean/std (seed 0: 101.42 / 10.39) close to real (101.33 / 9.97); price range clipped to ≥ 1e-6;
- **train time 77 260 / 76 054 / 123 402 / 77 414 / 99 826 s** (≈ 21–34 h/seed — ODE-solver bound, by
  far the slowest method in the benchmark); generation ≈ 4.5 s/seed.

---

## Reproduce (exact run path)

```bash
# Environment: gtgan-venv (torch, CUDA, torchdiffeq). GT-GAN trains on GPU.
cd methods/GT-GAN/code

# All 5 Heston seeds (2 GPUs in parallel — see "How to produce new seeds")

# Compute metrics
cd /home/tbasseras/benchmark
/home/tbasseras/gpu-venv/bin/python metrics/compute_all.py --method GT-GAN --dataset Heston
```

**Exact run path — which file produced which committed number:**

| Committed number | Interpreter + env | Command | Input file(s) scored | Output file |
|------------------|-------------------|---------|----------------------|-------------|
| GT-GAN synthetic paths, per seed `i` | `gtgan-venv`, `CUDA_VISIBLE_DEVICES=0/3 taskset -c … OMP_NUM_THREADS=8` | `train_heston.py --seed i` | real `dataset/Heston/heston_S_8192x128.npy`, global min-max internally | `generated_paths/seed_i/generated_paths_8192x128.npy` + `metadata.json` |
| Heston A1–A34 + B, per seed `i` | `gpu-venv`, `CUDA_VISIBLE_DEVICES=0` | `metrics/compute_all.py --method GT-GAN --dataset Heston` | `methods/GT-GAN/generated_paths/seed_i/generated_paths_8192x128.npy` (8192,128) vs real `dataset/Heston/heston_S_8192x128.npy` | `results/Heston/GT-GAN/seed_i_metrics.json` (A) + `curve_b_aggregate.json` (B) |
| A18/A19 disc/pred loss curves, per seed `i` | `gpu-venv` | `metrics/compute_all.py --method GT-GAN --dataset Heston` | same generated `.npy` vs real | `results/Heston/GT-GAN/seed_i_{disc,pred}_{gru,mlp}_loss.csv` → `plots/{disc_classifier,pred_score}_loss.png` |

Each `results/Heston/GT-GAN/seed_i_metrics.json` is the sole source for that seed's column in every
README A-table; the mean±std rows aggregate the 5 files.

The paper reproduction (Stocks discriminative / predictive — the paper's own Table 1 metrics) lives
separately in [`../paper_reimplementation/`](../paper_reimplementation/).
