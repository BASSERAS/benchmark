# TimeVAE Code — Sources & Implementation

## Original work

| Field | Details |
|-------|---------|
| **Paper** | *TimeVAE: A Variational Auto-Encoder for Multivariate Time Series Generation* |
| **Authors** | Abhyuday Desai, Cynthia Freeman, Zuhui Wang, Ian Beaver |
| **Reference** | arXiv:2111.08095v3 (2021) |
| **Original code** | https://github.com/abudesai/timeVAE |
| **Original framework** | TensorFlow / Keras (`tensorflow==2.16.1`) |
| **Method type** | **Variational auto-encoder** — convolutional encoder → Gaussian latent → interpretable decoder (level + optional trend/seasonal + residual conv) |

The verbatim reference implementation is kept under [`reference/`](reference/) for transparency
(`.git` stripped). The architecture and the custom VAE loss of
`reference/src/vae/{vae_base.py, timevae.py}` are ported **line-for-line** into
[`timevae_torch.py`](timevae_torch.py); every class docstring cites its TF source anchor.

---

## Our implementation

The released code is **TensorFlow/Keras**, and TensorFlow is GPU-broken on this machine
(CUDA-13 driver → cuDNN `INTERNAL_ERROR`, `GPUs = []`), so it would run **CPU-only**.
[`timevae_torch.py`](timevae_torch.py) reimplements the *exact* architecture and loss of the
reference in **PyTorch** so training runs on the A100s. [`train_heston.py`](train_heston.py) is a thin
data-plumbing wrapper: it loads the 8192×128 Heston price paths, per-`(t, feature)` MinMax-scales them
into `[0, 1]`, trains one TimeVAE-Base per seed with the reference schedule (Adam lr 1e-3,
`EarlyStopping`, `ReduceLROnPlateau`), **prior-generates** 8192 length-128 paths
(`decoder(randn(N, latent_dim))`), inverts the scaler back to price scale, and writes weights / losses /
generated paths / metadata in the benchmark's standard layout.

The default variant is **TimeVAE-Base**: level model + residual convolution, with the interpretable
`trend_poly` / `custom_seas` blocks **disabled** — exactly the `config/hyperparameters.yaml` `timeVAE`
preset (`latent_dim = 8`, `hidden = [50, 100, 200]`, `reconstruction_wt = 3.0`, `batch_size = 16`,
`use_residual_conn = true`, `trend_poly = 0`, `custom_seas = null`). This is the **same** preset that
reproduced the paper's sine-data discriminative/predictive table (see
[`../paper_reimplementation/README.md`](../paper_reimplementation/README.md)).

### VAE loss (exact port of `vae_base.py::train_step`)

$$\mathcal{L} = w_{\text{rec}} \cdot \mathcal{L}_{\text{rec}} + \mathcal{L}_{\text{KL}}, \qquad w_{\text{rec}} = 3.0$$

- **Reconstruction** = overall SSE + SSE of the feature-axis means:
  $\mathcal{L}_{\text{rec}} = \sum (x - \hat{x})^2 + \sum (\bar{x}_{\text{feat}} - \bar{\hat{x}}_{\text{feat}})^2$
- **KL** = $-\tfrac{1}{2}\sum (1 + \log\sigma^2 - \mu^2 - \sigma^2)$, summed over latent then batch.

The monitored quantity for `EarlyStopping` / `ReduceLROnPlateau` is the **epoch-mean total loss**
(Keras `Mean` tracker).

### Normalisation chain

```
S(price) --MinMaxScaler(per-(t,feature))--> ~[0,1]   (model trains here)
prior sample decoder(randn) --inverse_transform--> price
```

The `MinMaxScaler` is a **per-`(t, feature)`** min-max (port of `data_utils.MinMaxScaler`): it fits
`mini = data.min(axis=0)` and `range = data.max(axis=0) − mini`, so each timestep gets its own
`[0, 1]` mapping. The fitted `mini` / `range` arrays are stored in each `weights/seed_{i}_model.pt`
(`scaler_mini`, `scaler_range`) so sampling inverts exactly back to price scale.

---

## Architecture (TimeVAE-Base, feat\_dim = 1, seq\_len = 128)

| Component | Layers | Output shape |
|-----------|--------|--------------|
| **Encoder** | 3× `SameConv1d(k=3, s=2)` → ReLU, filters 50 → 100 → 200; Flatten | `(N, L·200)`, L = ⌈128/2³⌉ = 16 |
| ↳ latent heads | `Linear(flat, 8)` × 2 → `z_mean`, `z_log_var` | `(N, 8)` each |
| **Sampling** | reparameterization `z = μ + σ·ε` | `(N, 8)` |
| **Decoder — level** | `Linear(8, 1)` → ReLU → `Linear(1, 1)`, broadcast over T | `(N, 128, 1)` |
| **Decoder — residual** | `Linear(8, flat)` → reshape `(N, 16, 200)` → 3× `SameConvTranspose1d(k=3, s=2)` → ReLU → Flatten → `Linear(→128·1)` | `(N, 128, 1)` |
| **Output** | level + residual (trend / seasonal off) | `(N, 128, 1)` |

Total **247 340 parameters** (`weights/seed_0_config.json`). The interpretable `TrendLayer` /
`SeasonalLayer` are implemented but inactive (`trend_poly = 0`, `custom_seas = None`).

---

## Fixes / porting adaptations vs the reference

TensorFlow → PyTorch is not a line-copy; three adaptations were needed to keep the architecture
**bit-faithful** to the released Keras model:

1. **`padding='same'` with stride 2.** *Location:* encoder `Conv1D` / decoder `Conv1DTranspose`.
   *Reference:* Keras `padding='same'` computes an asymmetric pad automatically (extra pad on the
   right). *Our fix:* `SameConv1d` / `SameConvTranspose1d` replicate TF's exact pad
   (`pad_left = total//2`, `pad_right = total − total//2`; for `k=3, s=2` the transpose uses
   `padding=1, output_padding=1` → output length exactly `2·L_in`), since PyTorch has no `'same'`
   with stride > 1.
2. **Flatten order.** *Location:* encoder Flatten and residual reshape. *Reference:* Keras flattens
   `(N, L, C)` in `L`-major order. *Our fix:* transpose `(N, C, L) → (N, L, C)` before `reshape`, and
   reshape the residual as `(N, enc_last_L, hidden[-1])` to match the TF `(N, L, C)` layout exactly.
3. **GPU backend.** *Location:* whole training stack. *Reference:* `tensorflow==2.16.1` → CPU-only on
   this CUDA-13 box. *Our fix:* the entire model + custom VAE loss + Adam / `EarlyStopping` /
   `ReduceLROnPlateau` schedule reimplemented in PyTorch so training runs on the A100.

No hyperparameter or loss-formula change: the reconstruction (SSE + feature-mean SSE) and KL terms are
computed identically to `vae_base.py::train_step`.

---

## Hyperparameters (from paper / released `timeVAE` preset)

| Parameter | Value | Source |
|-----------|-------|--------|
| `latent_dim` | 8 | `config/hyperparameters.yaml` `timeVAE` |
| `hidden_layer_sizes` | (50, 100, 200) | `config/hyperparameters.yaml` |
| `reconstruction_wt` | 3.0 | `config/hyperparameters.yaml` |
| `trend_poly` | 0 | TimeVAE-Base (interpretable blocks off) |
| `custom_seas` | None | TimeVAE-Base |
| `use_residual_conn` | True | `config/hyperparameters.yaml` |
| `batch_size` | 16 | `vae_base.fit_on_data` |
| `lr` | 1e-3 | Adam, `vae_base.fit_on_data` |
| `max_epochs` | 1000 | `vae_base.fit_on_data` |
| `EarlyStopping` | monitor total\_loss, min\_delta 1e-2, patience 50 | `vae_base.fit_on_data` |
| `ReduceLROnPlateau` | factor 0.5, patience 30 | `vae_base.fit_on_data` |

---

## How to change hyperparameters

The architecture preset lives in the `PRESET` dict at the top of
[`train_heston.py`](train_heston.py); the training schedule lives in `TRAIN`:

```python
PRESET = dict(latent_dim=8, hidden_layer_sizes=(50, 100, 200), reconstruction_wt=3.0,
              trend_poly=0, custom_seas=None, use_residual_conn=True)
TRAIN  = dict(max_epochs=1000, batch_size=16, lr=1e-3,
              es_patience=50, es_min_delta=1e-2, rlr_patience=30, rlr_factor=0.5)
```

- **Latent size / conv widths / recon weight** → edit `PRESET` (e.g. `latent_dim=16`,
  `hidden_layer_sizes=(64, 128, 256)`).
- **Turn on the interpretable decoder (TimeVAE-full)** → set `trend_poly=2` and/or
  `custom_seas=[(4, 32)]` (num\_seasons, len\_per\_season) — the `TrendLayer` / `SeasonalLayer` are
  already wired into `Decoder.forward`.
- **Training length / batch / LR** → edit `TRAIN`, or pass CLI flags: `--epochs`, `--batch_size`.

CLI flags on `train_heston.py`:

| Flag | Default | Effect |
|------|---------|--------|
| `--seed` | 0 | Re-seeds torch + numpy (model init + prior sampling). |
| `--epochs` | 0 | Override `max_epochs` (0 = preset 1000). |
| `--batch_size` | 16 | Training batch. |
| `--gen_num` | 8192 | Prior samples to generate. |
| `--frac` | 1.0 | Fraction of training paths (smoke runs). |
| `--tag` | "" | Run tag (e.g. `smoke`) — prefixes outputs, skips canonical weights. |

---

## How to use a different dataset

`train_heston.py` takes `--data PATH` — a `.npy` of shape `(N, T)`, dtype float, in price/level space.
The wrapper adds a feature axis `(N, T, 1)`, per-`(t, feature)` MinMax-scales to `[0, 1]` (stores
`scaler_mini` / `scaler_range` for exact inversion), trains, prior-samples, and de-normalises back to
the original scale before saving. Any `T` works (the conv stack halves the length 3× — `T` need not be
a power of 2, the `SameConv1d` handles the ceil-division).

---

## How to produce new seeds

Each seed re-seeds torch + numpy, so encoder/decoder init and prior sampling differ:

```bash
cd methods/TimeVAE/code
# one extra seed
CUDA_VISIBLE_DEVICES=0 /home/tbasseras/gpu-venv/bin/python train_heston.py --seed 5

# all 5 canonical seeds, 2 GPUs in parallel (~13 min/seed on an A100)
for s in 0 1 2 3 4; do
  g=$((s % 2 == 0 ? 0 : 3)); c=$(((s % 2)*8))
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 taskset -c $c-$((c+7)) \
    /home/tbasseras/gpu-venv/bin/python train_heston.py --seed $s &
done
wait
```

Each seed writes:
- `weights/seed_{s}_model.pt` — VAE encoder/decoder `state_dict` + `scaler_mini` / `scaler_range`
- `weights/seed_{s}_config.json` — full hyperparameters + params count
- `losses/seed_{s}_losses.csv` — per-epoch `epoch, total_loss, reconstruction_loss, kl_loss, lr`
- `generated_paths/seed_{s}/generated_paths_8192x128.npy` — (8192, 128) price scale
- `generated_paths/seed_{s}/metadata.json` — seed, shape, min/max, train time, params, epochs\_run

---

## Sanity check

The full canonical run (seed 0, ~788 s wall-clock on one A100) confirms healthy training. Seed 0
signals:

```
[data] S(8192, 128) price[min=39.89, max=155.58]  scaled[min=0.0000, max=1.0000]  epochs=1000
[model] params=247340
[epoch 0] total≈450 ... [early stop] epoch 247, best total=83.19
generated price[min=49.09 max=141.83]  real_mean=101.33 gen_mean=100.96  no NaN
```

Expected sane signals (all five seeds, verified):
- total loss **decreases smoothly** from ~450 to a plateau of **~83**; `EarlyStopping` fires between
  epochs **230 and 340** (seed 0 247, seed 1 230, seed 2 340, seed 3 298, seed 4 278) — no NaN
  (`first_nan_epoch = null`, `gen_has_nan = false` in every metadata.json);
- generated price range (≈ 49–142) sits **inside** the real range (39.9–155.6); generated mean/std
  (100.96 / 8.88) close to real (101.33 / 9.97);
- 247 340 parameters; prior generation via `decoder(randn(N, 8))`.

---

## Reproduce

```bash
# Environment: gpu-venv (torch, CUDA). TimeVAE trains on GPU.
cd methods/TimeVAE/code

# All 5 Heston seeds (2 GPUs in parallel — see "How to produce new seeds")

# Compute metrics
cd /home/tbasseras/benchmark
/home/tbasseras/gpu-venv/bin/python metrics/compute_all.py --method TimeVAE --dataset Heston
```

**Exact run path — which file produced which committed number:**

| Committed number | Interpreter + env | Command | Input file(s) scored | Output file |
|------------------|-------------------|---------|----------------------|-------------|
| Heston A1–A34 + B, per seed `i` | `gpu-venv`, `CUDA_VISIBLE_DEVICES=0` | `metrics/compute_all.py --method TimeVAE --dataset Heston` | `methods/TimeVAE/generated_paths/seed_i/generated_paths_8192x128.npy` (8192,128) vs real Heston `dataset/Heston/heston_S_8192x128.npy` | `results/Heston/TimeVAE/seed_i_metrics.json` (A) + `curve_b_aggregate.json` (B) |
| TimeVAE synthetic paths, per seed `i` | `gpu-venv`, `CUDA_VISIBLE_DEVICES=0/3 taskset -c … OMP_NUM_THREADS=8` | `train_heston.py --seed i` (best-total-loss `EarlyStopping` weights used for prior sampling) | real Heston `dataset/Heston/heston_S_8192x128.npy`, per-`(t,feat)` MinMax internally | `generated_paths/seed_i/generated_paths_8192x128.npy` + `metadata.json` |
| A18/A19 disc/pred loss curves, per seed `i` | `gpu-venv` | `metrics/compute_all.py --method TimeVAE --dataset Heston` | same generated `.npy` vs real | `results/Heston/TimeVAE/seed_i_{disc,pred}_{gru,mlp}_loss.csv` → `plots/{disc_classifier,pred_score}_loss.png` |

Each `results/Heston/TimeVAE/seed_i_metrics.json` is the sole source for that seed's column in every
README A-table; the mean±std rows aggregate the 5 files.

The paper reproduction (sine len-24, the paper's own discriminative/predictive metrics vs Table)
lives separately in [`../paper_reimplementation/`](../paper_reimplementation/) — see its README.
