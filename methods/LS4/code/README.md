# LS4 Code — Sources & Implementation

## Original work

| Field | Details |
|-------|---------|
| **Paper** | *Deep Latent State Space Models for Time-Series Generation* |
| **Authors** | Linqi Zhou, Michael Poli, Winnie Xu, Stefano Massaroli, Stefano Ermon |
| **Venue** | ICML 2023 (PMLR, pp. 42625–42643) — PDF `../paper_reimplementation/LS4_ICML2023.pdf` |
| **arXiv** | [2212.12749](https://arxiv.org/abs/2212.12749) |
| **Original code** | https://github.com/alexzhou907/ls4 |
| **Original framework** | **PyTorch** (already PyTorch — no framework port needed) |
| **Method type** | **VAE-style latent state-space model.** A continuous latent `z` evolves through a **structured state-space (S4)** prior; a matching S4 posterior/decoder reconstructs the observation. Trained end-to-end on the **ELBO** (`total = kld_loss + nll_loss`; `mse_loss` is a reconstruction diagnostic, not part of the objective). Generation samples the S4 prior autoregressively in **STEP mode** and decodes. |

The verbatim reference implementation is kept under [`reference/`](reference/) for transparency
(`.git` stripped). The generative model (`models.ls4.VAE`, its S4 prior/posterior/decoder in
[`reference/models/`](reference/models/)) is imported **unchanged** by
[`train_heston.py`](train_heston.py) with the released **solar_weekly** config
(`reference/configs/monash/vae_solarweekly_released.yaml`). The one substantive edit to the
reference is a **numerical fix to the naive Cauchy kernel** (see Fixes below).

---

## Our implementation

The released code is already **PyTorch**, so — unlike the TF baselines — there is **no architecture
port**: [`train_heston.py`](train_heston.py) constructs `models.ls4.VAE` from the released
solar_weekly preset and reproduces the upstream ELBO training loop
(`loss, log_info = model(data, None, masks, plot=False, sum=False)`), the EMA weight averaging, and
the STEP-mode generation. It is a data-plumbing + logging wrapper: it loads the 8192×128 Heston price
paths, **global-standardizes** them to ≈N(0,1) for training, trains one LS4 per seed with the
released hyperparameters, then **generates** 8192 length-128 paths, de-standardizes back to price
scale, and writes weights / losses / generated paths / metadata in the benchmark's standard layout.

### ⚠️ The Cauchy fix (required, and why)

LS4's S4 layer builds its convolution kernel from a **Cauchy sum over the diagonal state-space
poles**. The repo ships three back-ends: a CUDA extension, a `pykeops` path, and a **naive PyTorch
fallback**. The CUDA/keops back-ends sum over **conjugate pole pairs** (real, folded) whereas the
released naive fallback summed over the raw pole list — numerically different in **STEP mode**.

This matters here because `model.generate` **rolls the S4 prior forward one step at a time**
(`latent.step`, the recurrent SSM form), not via the parallel convolution. On a machine without the
compiled Cauchy/keops extension the naive fallback is the only path, and the step-mode recurrence
then drifts: the Solar-Weekly marginal score **plateaus at ≈0.197** instead of the paper's ≈0.046.

The fix is a one-site change at [`reference/models/s4.py:795`](reference/models/s4.py) so the naive
Cauchy kernel sums over conjugate pole **pairs**, matching the compiled back-ends. With the fix, the
step-mode prior reproduces the convolution-mode prior and the Solar-Weekly marginal reaches
**≈0.046–0.047** (paper 0.0459). See [`../paper_reimplementation/README.md`](../paper_reimplementation/README.md)
for the before/after reproduction log. All committed Heston runs use the **cauchy-fixed** model.

### Normalisation chain

```
S(price) --(X - mu)/sigma--> ~N(0,1)   (model trains here; ELBO on standardized data)
generated --* sigma + mu--> price
```

A **global standardize** over the whole array (single scalar `mu`/`sigma`, not per-channel MinMax):
`mu = 101.32547381502401`, `sigma = 9.971659995159825` — the real-data mean/std, stored in each
`weights/seed_{i}_config.json` so sampling inverts exactly back to price scale. LS4's Gaussian
likelihood (`sigma=0.1` observation noise) is well matched to standardized inputs, which is why the
paper's own preprocessing standardizes the Monash series.

---

## Architecture (released solar_weekly preset, in\_channels = 1, seq\_len = 128)

| Component | Role | Config |
|-----------|------|--------|
| **Prior** (S4 latent SSM) | Autoregressive latent dynamics `p(z_t \| z_{<t})` | `latent_type=split`, `z_dim=5`, `d_state=64`, `d_model=64`, `n_layers=4`, `s4_type=s4` |
| **Posterior** (S4 encoder) | `q(z_t \| x_{≤t})` amortised inference | shares the S4 stack width (`d_model=64`, `n_layers=4`) |
| **Decoder** (S4 → obs) | `p(x_t \| z_t)` Gaussian likelihood, `sigma=0.1` | `in_channels=1`, `d_model=64` |

Total **2 146 857 parameters** (`weights/seed_0_config.json`). Backbone `autoreg`; the latent is a
5-dim continuous vector evolved by the S4 prior; the decoder emits a single price channel per step.

---

## Fixes / adaptations vs the reference

The model classes are used **unchanged** except for the Cauchy fix; the rest of the adaptations are in
the training *driver*, to fit the benchmark's single-GPU / npy / multi-seed layout:

1. **Cauchy kernel fix (numerical, required).** *Location:* [`reference/models/s4.py:795`](reference/models/s4.py).
   *Reference (buggy in STEP mode):* naive Cauchy fallback sums over the raw pole list. *Our fix:* sum
   over conjugate pole **pairs** to match the CUDA/keops back-ends, so STEP-mode `model.generate`
   reproduces convolution-mode training (Solar-Weekly marginal 0.197 → 0.046). This is the **only**
   change to reference model code.
2. **Data I/O.** *Location:* dataset loading. *Reference:* reads a Monash `.tsf` via `train_monash.py`.
   *Our change:* load the Heston `.npy` `(N, T)` directly, global-standardize to ≈N(0,1) (store
   `scaler_mu`/`scaler_sigma`), export the generated `.npy` back in price scale.
3. **Per-seed seeding.** *Location:* top of `main`. *Our change:* seed torch + numpy from `--seed` so
   the 5 canonical seeds give distinct model inits and samples.
4. **STEP-mode generation + EMA.** *Location:* end of training. *Reference:* evaluates periodically
   during `train_monash.py`. *Our change:* after training, EMA-average the weights (`AveragedModel`),
   call `gen_model.setup_rnn()` to switch the S4 layers into recurrent STEP mode, and chunk
   `gen_model.generate(b, seq_len)` to prior-generate 8192 paths.
5. **Logging.** *Our change:* log per-epoch `total_loss, kld_loss, nll_loss, mse_loss, lr` to CSV,
   save a single `state_dict` + config + metadata.

No hyperparameter or objective change: the ELBO (`total = kld + nll`), the AdamW + ReduceLROnPlateau
schedule, and the EMA are all the released solar_weekly settings.

---

## Hyperparameters (from the released solar_weekly config)

| Parameter | Value | Source |
|-----------|-------|--------|
| `z_dim` (latent dim) | 5 | `vae_solarweekly_released.yaml` |
| `d_state` (S4 state) | 64 | released config |
| `d_model` (S4 width) | 64 | released config |
| `n_layers` | 4 | released config |
| `s4_type` | `s4` | released config |
| `latent_type` | `split` | released config |
| `backbone` | `autoreg` | released config |
| `sigma` (obs noise) | 0.1 | released config (Gaussian likelihood) |
| `in_channels` | 1 | **Heston-specific** — univariate price series |
| `optimizer` | AdamW, lr 1e-3, weight_decay 0 | released config |
| `lr schedule` | ReduceLROnPlateau (patience 20, factor 0.5) | released config |
| `EMA` | `lamb=0.99`, `start_step=200` | released config |
| `batch_size` | 128 | released config |
| `seq_len` | **128** | **Heston-specific** — the Heston sequence length |
| `epochs` | **100** | **Heston-specific** — flat-loss convergence (see Sanity check) |

Only three knobs are Heston-specific (`in_channels`, `seq_len`, `epochs`); everything else is the
released solar_weekly preset kept verbatim.

---

## How to change hyperparameters

The model preset is the released YAML `reference/configs/monash/vae_solarweekly_released.yaml`
(loaded by [`train_heston.py`](train_heston.py)); training/IO knobs are CLI flags:

| Flag | Default | Effect |
|------|---------|--------|
| `--seed` | 0 | Re-seeds torch + numpy (model init + sampling). |
| `--data` | `dataset/Heston/heston_S_8192x128.npy` | Training `.npy` of shape `(N, T)`, price scale. |
| `--epochs` | 400 (**100 used**) | Training epochs. |
| `--batch_size` | 128 | Training batch. |
| `--gen_num` | 8192 | Paths to generate. |
| `--frac` | 1.0 | Fraction of training paths (smoke runs). |
| `--tag` | "" | Run tag (e.g. `smoke`) — prefixes outputs, skips canonical weights. |

- **Latent width / S4 depth / state size** → edit `z_dim`, `d_model`, `d_state`, `n_layers` in the
  YAML. **Optimiser / schedule / EMA** → the `optimizer`, `scheduler`, `ema` blocks of the YAML.
- **Observation noise** → `sigma` in the YAML (Gaussian decoder likelihood).

---

## How to use a different dataset

`train_heston.py --data PATH` takes a `.npy` of shape `(N, T)`, dtype float, in price/level space. The
wrapper global-standardizes to ≈N(0,1) (stores `scaler_mu`/`scaler_sigma` for exact inversion),
trains, samples in STEP mode, and de-standardizes back before saving. For a **multivariate** dataset,
set `in_channels = C` in the YAML (with `T` the per-channel length); the S4 decoder then emits `C`
channels per step.

---

## How to produce new seeds

Each seed re-seeds torch + numpy, so model init and sampling differ:

```bash
cd methods/LS4/code
# one extra seed
CUDA_VISIBLE_DEVICES=0 /home/tbasseras/gpu-venv/bin/python train_heston.py --seed 5 --epochs 100

# all 5 canonical seeds, 2 GPUs in parallel (~16 min/seed on an A100)
for s in 0 1 2 3 4; do
  g=$((s % 2 == 0 ? 0 : 3)); c=$(((s % 2)*8))
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 taskset -c $c-$((c+7)) \
    /home/tbasseras/gpu-venv/bin/python train_heston.py --seed $s --epochs 100 &
done
wait
```

Each seed writes:
- `weights/seed_{s}_model.pt` — EMA-averaged `state_dict`
- `weights/seed_{s}_config.json` — full hyperparameters + `scaler_mu`/`scaler_sigma` + params (2 146 857)
- `losses/seed_{s}_losses.csv` — per-epoch `epoch, total_loss, kld_loss, nll_loss, mse_loss, lr`
- `generated_paths/seed_{s}/generated_paths_8192x128.npy` — (8192, 128) price scale
- `generated_paths/seed_{s}/metadata.json` — seed, shape, min/max, gen/train time, params, epochs\_run

---

## Sanity check

The full canonical run (seed 0, ~16 min wall-clock on one A100) confirms healthy ELBO training:

```
[data] S(8192, 128) price  standardized mu=101.325 sigma=9.972
[model] params=2146857  (LS4 VAE, released solar_weekly preset, cauchy-fixed)
epoch  0  total=24.564  kld=0.464  nll=24.099  mse=0.5097  lr=0.001
epoch 50  total≈-0.9    kld≈0.22   nll≈-1.15
epoch 99  total=-1.0667 kld=0.2231 nll=-1.2898 mse=0.001878 lr=0.001
[done] seed=0 epochs=100  gen=(8192,128) nan=False  gen_mean=101.56/std=9.77 vs real 101.33/9.97
```

Expected sane signals (all five seeds, verified):
- **Total ELBO drops from ≈24.6 (epoch 0) to ≈−1.07** and flattens by ~epoch 90 (`min_total_loss ≈ −1.068`);
  KLD settles near **0.22**, NLL near **−1.29**, recon MSE diagnostic to **≈0.0019**;
- ReduceLROnPlateau holds lr at 1e-3 for the whole 100-epoch run (loss still improving, no plateau trigger);
- no NaN (`first_nan_epoch = null`, `gen_has_nan = false` in every metadata.json);
- generated mean/std (seed 0: 101.56 / 9.77) close to real (101.33 / 9.97); price range ≈ 47–145;
- 2 146 857 parameters; 100 epochs; ~973 s/seed train, ~9 s/seed generate.

---

## Reproduce (exact run path)

```bash
# Environment: gpu-venv (torch, CUDA). LS4 trains on GPU.
cd methods/LS4/code

# All 5 Heston seeds (2 GPUs in parallel — see "How to produce new seeds")

# Compute metrics
cd /home/tbasseras/benchmark
/home/tbasseras/gpu-venv/bin/python metrics/compute_all.py --method LS4 --dataset Heston
```

**Exact run path — which file produced which committed number:**

| Committed number | Interpreter + env | Command | Input file(s) scored | Output file |
|------------------|-------------------|---------|----------------------|-------------|
| LS4 synthetic paths, per seed `i` | `gpu-venv`, `CUDA_VISIBLE_DEVICES=0/3 taskset -c … OMP_NUM_THREADS=8` | `train_heston.py --seed i --epochs 100` | real `dataset/Heston/heston_S_8192x128.npy`, global-standardize internally (cauchy-fixed `reference/models/s4.py`) | `generated_paths/seed_i/generated_paths_8192x128.npy` + `metadata.json` |
| Heston A1–A34 + B, per seed `i` | `gpu-venv`, `CUDA_VISIBLE_DEVICES=0` | `metrics/compute_all.py --method LS4 --dataset Heston` | `methods/LS4/generated_paths/seed_i/generated_paths_8192x128.npy` (8192,128) vs real `dataset/Heston/heston_S_8192x128.npy` | `results/Heston/LS4/seed_i_metrics.json` (A) + `curve_b_aggregate.json` (B) |
| A18/A19 disc/pred loss curves, per seed `i` | `gpu-venv` | `metrics/compute_all.py --method LS4 --dataset Heston` | same generated `.npy` vs real | `results/Heston/LS4/seed_i_{disc,pred}_{gru,mlp}_loss.csv` → `plots/{disc_classifier,pred_score}_loss.png` |

Each `results/Heston/LS4/seed_i_metrics.json` is the sole source for that seed's column in every
README A-table; the mean±std rows aggregate the 5 files.

The paper reproduction (Solar Weekly marginal / class / prediction — the paper's own Table 1 metrics,
including the cauchy-fix reconciliation) lives separately in
[`../paper_reimplementation/`](../paper_reimplementation/) — see its README.
