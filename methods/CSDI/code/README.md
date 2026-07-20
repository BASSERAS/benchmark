# CSDI Code — Sources & Implementation

## Original work

| Field | Details |
|-------|---------|
| **Paper** | *CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation* |
| **Authors** | Yusuke Tashiro, Jiaming Song, Yang Song, Stefano Ermon |
| **Venue** | NeurIPS 2021 |
| **arXiv** | https://arxiv.org/abs/2107.03502 |
| **Original code** | https://github.com/ermongroup/CSDI |
| **Method type** | Conditional **score-based (DDPM) diffusion** with a 2-D (time × feature) Transformer denoiser |

The verbatim reference implementation is kept under [`reference/`](reference/) for transparency
(`.git` stripped). `main_model.CSDI_base` (the diffusion model), `diff_models.diff_CSDI` (the denoiser),
the quad β-schedule, and the DDPM `calc_loss` / `impute` are imported **unmodified**; only the data
plumbing and a thin unconditional subclass (`train_heston.py`) are ours.

---

## Our implementation

`train_heston.py` is a thin data-plumbing wrapper around the **released** `CSDI_base`. It loads the
8192×128 Heston price paths, **z-score standardises** them (diffusion needs zero-mean/unit-var data, not
min-max), trains one model per seed with the authors' DDPM objective, samples 8192 synthetic paths, and
writes weights / losses / generated paths / metadata in the benchmark's standard layout. **The diffusion
process, denoiser, β-schedule, and loss are the authors' own, untouched.**

### Why unconditional generation = `is_unconditional=1` + `cond_mask ≡ 0`

The paper (Sec 4.1 / Appendix C) states the `is_unconditional=1` variant "can also be used for data
generation". In that mode `CSDI_base.set_input_to_diffmodel` feeds the network **only** the noisy
sequence — `cond_mask` never gates the network input, it only selects which points enter the loss
(`target_mask = observed_mask − cond_mask`). The thin subclass `CSDI_Heston` therefore sets
`observed_mask = 1` and `cond_mask = 0` everywhere:

- **training** → `target_mask = 1` everywhere → every timestep is a denoising target (standard
  unconditional DDPM objective $\mathbb{E}_t \lVert \epsilon - \epsilon_\theta(x_t,t)\rVert^2$);
- **sampling** → `impute` with `cond_mask = 0` collapses to pure ancestral sampling (no conditioning term).

Training and generation thus see the identical input distribution (full noisy sequence); the
architecture, diffusion process and hyperparameters are the paper's `is_unconditional` variant,
unchanged. Same pattern as `methods/DiffusionTS` and `methods/FourierFlow`: the windowed Heston paths
bypass the PhysioNet/PM2.5 `Dataset` classes.

> The **conditional** CSDI (the paper's headline imputation task) is run separately in
> [`../paper_reimplementation/`](../paper_reimplementation/) — both the PhysioNet/PM2.5 Table-2
> reproduction and the Heston imputation-CRPS transfer (`heston_imputation/`).

### Architecture (released `diff_CSDI`)

| Component | Definition |
|-----------|-----------|
| Input projection | `Conv1d(2 → channels=64)` on (noisy value, cond-mask) channel pair |
| Diffusion-step embedding | 128-D sinusoidal → 2× Linear (`DiffusionEmbedding`) |
| Side info | time embedding (128) + feature embedding (16), fed to every residual block |
| Residual blocks | **4 layers** × `channels=64`; each has a **temporal Transformer** (`forward_time`) + a **feature Transformer** (`forward_feature`), `nheads=8` |
| Output projection | `Conv1d(channels → 1)` → predicted noise ε |
| Diffusion | 50-step DDPM, **quad** β-schedule β_start=1e-4 → β_end=0.5 |

Params ≈ 412 945; ~1755 s train / seed on one A100; min training loss ≈ 0.0096, no NaN.

---

## Fixes applied

Only **one** non-behavioural change to the reference; the data plumbing is additive.

| # | Location | Reference (upstream) | Our fix |
|---|----------|----------------------|---------|
| 1 | `reference/diff_models.py` (lines 5–23) | Top-level `from linear_attention_transformer import LinearAttentionTransformer` — a hard import of an **optional** package used only by the `is_linear=True` linear-attention path (unused here; we run `is_linear=False`). | Made the import **lazy** (moved into `get_linear_trans`). The module now loads without installing that dependency; the diffusion math is byte-identical. This is the ONLY reference edit, shared with the paper reproduction. |

The unconditional wrapper (`CSDI_Heston.process_data` / `forward` / `generate`) lives entirely in
`train_heston.py` — it does not modify the reference model; it constructs the `observed_mask = 1`,
`cond_mask = 0` batch and calls the parent's `get_side_info` / `calc_loss` / `impute` verbatim.

---

## Hyperparameters (from the paper's `config/base.yaml`)

Kept **verbatim** from the released `reference/config/base.yaml` — the same config that reproduced Table 2
(see `../paper_reimplementation/`). Hard-coded in `BASE_CONFIG` in `train_heston.py`.

| Parameter | Value | Source in paper / repo |
|-----------|-------|------------------------|
| `epochs` | 200 | `base.yaml` train.epochs |
| `batch_size` | 16 | `base.yaml` train.batch_size |
| `lr` | 1e-3 | `base.yaml` train.lr; Adam, weight-decay 1e-6 |
| `layers` | 4 | `base.yaml` diffusion.layers |
| `channels` | 64 | `base.yaml` diffusion.channels |
| `nheads` | 8 | `base.yaml` diffusion.nheads |
| `diffusion_embedding_dim` | 128 | `base.yaml` diffusion.diffusion_embedding_dim |
| `beta_start / beta_end` | 1e-4 / 0.5 | `base.yaml` diffusion (quad schedule) |
| `num_steps` | 50 | `base.yaml` diffusion.num_steps |
| `schedule` | quad (`is_linear=False`) | `base.yaml` diffusion.schedule |
| `timeemb / featureemb` | 128 / 16 | `base.yaml` model |
| `is_unconditional` | **1** | paper Sec 4.1 generation variant (our setting) |
| LR schedule | `MultiStepLR` ×0.1 @ 0.75·epochs, 0.9·epochs | authors' `utils.train` (milestones 150, 180 at 200 ep) |

---

## How to change hyperparameters

Two override flags are exposed on `train_heston.py`; the rest are edited in `BASE_CONFIG` at the top of
the file (a plain dict — change the value and rerun).

| What | How | Default |
|------|-----|---------|
| Epochs | `--epochs N` CLI flag (0 = base.yaml 200) | 200 |
| Batch size | `--batch_size N` CLI flag (0 = base.yaml 16) | 16 |
| Diffusion depth/width, β-schedule, num_steps, embeddings, lr | edit `BASE_CONFIG["diffusion"]` / `["train"]` / `["model"]` in `train_heston.py` | base.yaml verbatim |
| Number of paths to sample | `--gen_num N` (`--gen_batch` for sampler batch) | 8192 / 512 |

```bash
# Example: 20-epoch smoke run on 5% of the data
cd methods/CSDI/code
/home/tbasseras/gpu-venv/bin/python train_heston.py --seed 0 --frac 0.05 --epochs 20 --tag smoke
```

The `--tag` flag prefixes all outputs and **skips** writing the canonical `weights/seed_*` files, so a
smoke run never clobbers a real seed.

---

## How to use a different dataset

`train_heston.py` takes `--data PATH`. Provide a `.npy` of shape `(N, T)`, dtype float, in price/level
space:

```bash
cd methods/CSDI/code
/home/tbasseras/gpu-venv/bin/python train_heston.py --seed 0 --data /path/to/your_NxT.npy
```

The wrapper handles the rest automatically:
1. z-score standardises with the dataset's own global `mean` / `std` (stored in the config + metadata
   for exact inversion);
2. builds fully-observed `(N, T, 1)` batches (`observed_mask = 1`, `cond_mask = 0`);
3. de-standardises generated samples back to your original scale (`sample × std + mean`) before saving.

CSDI is feature-agnostic (`target_dim = K`); for a genuinely multivariate series pass `(N, T, K)` and
set `target_dim=K` in `CSDI_Heston(...)`. Any sequence length `T` works — no odd-length constraint
(unlike Fourier Flow).

---

## How to produce new seeds

Each seed re-seeds torch + numpy, so model init, diffusion noise, and sampling differ:

```bash
cd methods/CSDI/code
# one extra seed
/home/tbasseras/gpu-venv/bin/python train_heston.py --seed 5

# all 5 seeds in parallel (2 A100 GPUs, one --seed per process)
for s in 0 1 2 3 4; do
  gpu=$([ $((s % 2)) -eq 0 ] && echo 0 || echo 3)
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 taskset -c 0-7 \
    /home/tbasseras/gpu-venv/bin/python train_heston.py --seed $s &
done
wait
```

Each real seed (no `--tag`) writes:
- `weights/seed_{s}_model.pt` — `{model state_dict, seed, zscore=[mean,std]}`
- `weights/seed_{s}_config.json` — hyperparameters + z-score constants
- `losses/seed_{s}_losses.csv` — `step, loss` (DDPM noise-prediction MSE per mini-batch)
- `generated_paths/seed_{s}/generated_paths_8192x128.npy` — (8192, 128) price scale
- `generated_paths/seed_{s}/metadata.json` — params, train/gen time, min_loss, first_nan, z-score

---

## Reproduce (EXACT run path)

```bash
# Environment: gpu-venv (torch + CUDA), one A100 per seed
cd /home/tbasseras/benchmark/methods/CSDI/code

# All 5 Heston seeds (GPU0 for even seeds, GPU3 for odd — 2 GPUs, ~30 min each pair)
for s in 0 1 2 3 4; do
  gpu=$([ $((s % 2)) -eq 0 ] && echo 0 || echo 3)
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 taskset -c 0-7 \
    /home/tbasseras/gpu-venv/bin/python train_heston.py --seed $s &
done
wait

# Loss-convergence plot
/home/tbasseras/gpu-venv/bin/python plot_losses.py

# Compute metrics
cd /home/tbasseras/benchmark
/home/tbasseras/gpu-venv/bin/python metrics/compute_all.py --method CSDI --dataset Heston
```

**Exact run path — which file produced which committed number:**

| Committed number | Interpreter + env | Command | Input file(s) scored | Output file |
|------------------|-------------------|---------|----------------------|-------------|
| Heston A1–A34 + B, per seed `i` | `gpu-venv`, 1 GPU | `metrics/compute_all.py --method CSDI --dataset Heston` | `methods/CSDI/generated_paths/seed_i/generated_paths_8192x128.npy` (8192,128) vs real `dataset/Heston/heston_S_8192x128.npy` | `results/Heston/CSDI/seed_i_metrics.json` (A) + `curve_b_aggregate.json` (B) |
| CSDI synthetic paths, per seed `i` | `gpu-venv`, `CUDA_VISIBLE_DEVICES={0,3}`, `taskset -c 0-7`, `OMP_NUM_THREADS=8` | `train_heston.py --seed i` | real Heston `dataset/Heston/heston_S_8192x128.npy` | `generated_paths/seed_i/generated_paths_8192x128.npy` + `weights/seed_i_config.json` |
| Training-loss curve | `gpu-venv` | `plot_losses.py` | `losses/seed_{0..4}_losses.csv` | `losses/loss_convergence.png` |

Each `results/Heston/CSDI/seed_i_metrics.json` is the sole source for that seed's column in every README
A-table; mean±std rows aggregate the 5 files. The paper reproduction (PhysioNet + PM2.5 CRPS vs Table 2)
and the Heston imputation-CRPS transfer live separately in `../paper_reimplementation/` — see its README.

---

## Sanity check

The full 200-epoch run (one A100 per seed) confirms stable DDPM training. Seed 0 header + result:

```
[CSDI seed 0] params=412945 epochs=200 batch=16 N=8192 T=128
  epoch    0/200  avg_loss=0.98...  lr=1.00e-03
  ...
  epoch  199/200  avg_loss≈0.010    lr=1.00e-05
{ "method": "CSDI", "seed": 0, "shape": [8192, 128], "gen_sec": 10.2,
  "train_time_sec": 1755.7, "params": 412945, "min_loss": 0.0096,
  "first_nan_step": null, "gen_has_nan": false,
  "zscore_mean": 101.325..., "zscore_std": 9.9716... }
```

Expected sane signals (all five seeds, verified):
- **no NaN** anywhere (`first_nan_step: null`, `gen_has_nan: false`);
- DDPM noise-MSE loss decreases smoothly to ≈ 0.01 (it is a *noise-prediction* MSE, so ~0.01 is a
  well-fit model, not a collapse);
- generated price range sits inside the real Heston range (~40–156); generated `col0` mean ≈ 100
  (the flow spreads the deterministic S₀ = 100 into a tight distribution);
- z-score constants (`mean` 101.325, `std` 9.972) are the canonical dataset statistics, identical
  across seeds.
