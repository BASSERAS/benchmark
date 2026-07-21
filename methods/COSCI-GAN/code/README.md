# COSCI-GAN Code — Sources & Implementation

## Original work

| Field | Details |
|-------|---------|
| **Paper** | *Generating multivariate time series with COmmon Source CoordInated GAN (COSCI-GAN)* |
| **Authors** | Ali Seyfi, Jean-Francois Rajotte, Raymond T. Ng |
| **Reference** | NeurIPS 2022 (`../paper_reimplementation/COSCI-GAN_NeurIPS2022.pdf`) |
| **Original code** | https://github.com/aliseyfi75/COSCI-GAN |
| **Original framework** | **PyTorch** (already PyTorch — no framework port needed) |
| **Method type** | **3-player GAN** — C univariate "Channel GANs" (LSTM Generator + LSTM Discriminator each), driven by a **shared noise vector**, coupled by one **Central Discriminator** (CD) that sees the concatenation of all channels and enforces inter-channel correlation |

The verbatim reference implementation is kept under [`reference/`](reference/) for transparency
(`.git` stripped). The three `nn.Module` classes we use —
`LSTMGenerator`, `LSTMDiscriminator`, `Discriminator` (the MLP CD) — are imported **verbatim** from
[`reference/Main_modules.py`](reference/Main_modules.py); the training loop in
[`train_heston.py`](train_heston.py) is transcribed **line-for-line** from
`Main_modules.COSCIGAN` (upstream lines 263–381). Only I/O differs.

---

## Our implementation

The released code is already **PyTorch**, so — unlike the TF-based baselines — there is **no
architecture port**: [`train_heston.py`](train_heston.py) imports the upstream classes directly and
reproduces the upstream three-player update. It is a data-plumbing + logging wrapper: it loads the
8192×128 Heston price paths, scalar-MinMax-scales them into `[0, 1]`, trains one COSCI-GAN per seed
with the paper/demo hyperparameters, then **generates** 8192 length-128 paths
(`G(randn(N, noise_len))`), inverts the scaler back to price scale, and writes weights / losses /
generated paths / metadata in the benchmark's standard layout.

### ⚠️ C = 1 (single-channel Heston) — why the Central Discriminator degenerates

COSCI-GAN is a **multivariate** generator. The Heston target here is a **univariate** price series,
so `n_groups = 1`:

- There is exactly **one** Channel GAN (one LSTM generator + one LSTM discriminator).
- The shared-noise mechanism (all generators share one `z`) is trivial with a single generator.
- The **Central Discriminator**, fed the concatenation of "all" channels, receives the **same**
  128-dim vector as the single channel discriminator — it becomes a **redundant second (MLP) critic**
  on the one channel. There is no cross-channel correlation for it to preserve (that notion is
  undefined at C = 1).

We keep the full three-player machinery **on** (`with_CD = True`), exactly as the paper's code runs
it, and **document** the degeneracy rather than silently dropping the CD. The equilibrium signature
is `loss_CD ≈ ln 2 ≈ 0.693` (CD stuck at chance) — observed on all 5 seeds. In this regime COSCI-GAN
reduces to a single LSTM-GAN regularised by an auxiliary MLP critic; it still produces valid
univariate paths, which is what the benchmark evaluates. The paper's own multivariate metric
(Table 4 cross-channel correlation-MAE) is reproduced separately on EEG in
[`../paper_reimplementation/`](../paper_reimplementation/README.md).

### 3-player loss (exact transcription of `Main_modules.COSCIGAN`)

For each channel *i* (here only *i* = 0), with shared noise *z*, `fake_i = G_i(z)`, γ = 5:

- **Channel discriminator:** $\mathcal{L}_{D_i} = \mathrm{BCE}\big(D_i([\text{real}_i; \text{fake}_i]),\ [\mathbf{1}; \mathbf{0}]\big)$
- **Central discriminator:** $\mathcal{L}_{CD} = \mathrm{BCE}\big(CD([\text{fake}_{\text{cat}}; \text{real}_{\text{cat}}]),\ [\mathbf{0}; \mathbf{1}]\big)$
- **Generator:** $\mathcal{L}_{G_i} = \mathrm{BCE}\big(D_i(\text{fake}_i),\ \mathbf{1}\big) - \gamma \cdot \mathcal{L}_{CD}^{\text{new}}$

where $\mathcal{L}_{CD}^{\text{new}}$ recomputes the CD loss with the generator's *current* forward
pass (only channel *i* keeps grad; others are detached — verbatim upstream behaviour). At C = 1 the
`cat` operations are no-ops and `real_cat` = the single real channel.

### Normalisation chain

```
S(price) --(X - min)/(max - min)--> ~[0,1]   (model trains here)
generated --* (max - min) + min--> price
```

A **scalar** MinMax over the whole array (analog of the upstream "ZeroOne" EEG preprocessing, which
min-max scales each channel to `[0, 1]`; with one channel this is one scalar `min`/`max` pair). The
fitted `scaler_min` / `scaler_max` are stored in each `weights/seed_{i}_model.pt` so sampling inverts
exactly back to price scale.

---

## Architecture (C = 1, ts\_dim = 128, noise\_len = 32)

| Component | Layers | Output shape |
|-----------|--------|--------------|
| **Channel Generator** `LSTMGenerator` | `z (N,32)` → reshape `(N,1,32)` → `LSTM(32→256, 1 layer)` → `Linear(256→128)` | `(N, 128)` |
| **Channel Discriminator** `LSTMDiscriminator` | `x (N,128)` → reshape `(N,1,128)` → `LSTM(128→256, 1 layer)` → `Linear(256→1)` → Sigmoid | `(N, 1)` |
| **Central Discriminator** `Discriminator` (MLP) | `Linear(128→256)` → LeakyReLU(0.1) → Dropout(0.3) → `Linear(256→128)` → … → `Linear(64→1)` → Sigmoid | `(N, 1)` |

Total **799 618 parameters** (`weights/seed_0_config.json`): one LSTM generator + one LSTM
discriminator + the 128→256→128→64→1 MLP central discriminator. Linear layers are weight-init
`normal_(0, 0.02)` / `bias = 0` (upstream `initialize_weights`); the LSTM keeps PyTorch defaults.

---

## Fixes / adaptations vs the reference

The upstream module classes are used **unchanged**; the adaptations are all in the training *driver*,
to fit the benchmark's single-GPU / npy / multi-seed layout:

1. **Dropped `nn.DataParallel`.** *Location:* model construction. *Reference:* upstream wraps every
   generator / discriminator / CD in `nn.DataParallel(...)`. *Our change:* single-GPU training, so we
   instantiate the bare modules — this keeps `state_dict` keys clean (no `module.` prefix) for
   reload, and is numerically identical on one GPU.
2. **Data I/O.** *Location:* dataset loading. *Reference:* reads a CSV/pkl from `../Dataset/` with an
   `ID` column and `df.sample(frac=...)`. *Our change:* load the Heston `.npy` `(N, T)` directly, add
   scalar MinMax `[0, 1]` (the ZeroOne analog), and export the generated `.npy` back in price scale.
3. **Per-seed seeding.** *Location:* top of `main`. *Reference:* hard-codes `torch.manual_seed(0)`.
   *Our change:* seed from `--seed` so the 5 canonical seeds give genuinely distinct model inits and
   samples.
4. **Logging + generation.** *Location:* end of training. *Reference:* saves generator `.pt` every 10
   epochs and (commented-out) computes catch22. *Our change:* log per-epoch `loss_D_0 / loss_G_0 /
   loss_CD` to CSV, save a single `state_dict` + config + metadata, and prior-generate 8192 paths.

No hyperparameter, no loss-formula, no architecture change: the three-player update
(`loss_D_i`, `loss_CD`, `loss_G_i = local − γ·loss_CD_new` with the per-channel detach) is computed
identically to `Main_modules.COSCIGAN`.

---

## Hyperparameters (from paper / official demo notebook `COSCI-GAN_demo.ipynb`)

| Parameter | Value | Source |
|-----------|-------|--------|
| `noise_len` | 32 | demo default (shared-noise dim) |
| `gamma` | 5.0 | demo default (CD coupling weight in `loss_G = local − γ·loss_CD`) |
| `generator_lr` (`glr`) | 1e-3 | demo default |
| `discriminator_lr` (`dlr`) | 1e-3 | demo default |
| `central_disc_lr` (`cdlr`) | 1e-4 | demo default — 10× smaller so the CD cannot overpower the channel GAN |
| `batch_size` | 32 | demo default |
| `betas` | (0.5, 0.9) | Adam betas, `Main_modules.COSCIGAN` |
| `hidden_dim` (LSTM) | 256 | `LSTMGenerator` / `LSTMDiscriminator` default |
| `num_layers` (LSTM) | 1 | `LSTMGenerator` / `LSTMDiscriminator` default |
| `alpha` (LeakyReLU, CD) | 0.1 | `Discriminator` default |
| `criterion` | BCE | demo default |
| `CD_type` | MLP | demo default (`Discriminator` on the concatenation) |
| `n_samples` (seq len) | **128** | **Heston-specific** — the Heston sequence length (vs 100 for EEG windows) |
| `num_epochs` | **120** | **Heston-specific** — chosen so total generator updates are comparable to the paper's EEG run |
| `n_groups` (channels) | **1** | **Heston-specific** — univariate price series (see C = 1 note) |

Only three knobs are Heston-specific (`n_samples`, `num_epochs`, `n_groups`); everything else is the
paper/demo preset kept verbatim.

---

## How to change hyperparameters

The architecture/optimiser preset lives in the `PRESET` dict at the top of
[`train_heston.py`](train_heston.py); every knob is also a CLI flag:

```python
PRESET = dict(noise_len=32, n_samples=128, gamma=5.0,
              glr=1e-3, dlr=1e-3, cdlr=1e-4, batch_size=32,
              betas=(0.5, 0.9), alpha=0.1)
```

CLI flags on `train_heston.py`:

| Flag | Default | Effect |
|------|---------|--------|
| `--seed` | 0 | Re-seeds torch + numpy (model init + sampling). |
| `--data` | `dataset/Heston/heston_S_8192x128.npy` | Training `.npy` of shape `(N, T)`, price scale. |
| `--n_groups` | 1 | Channels (Heston = 1). For a multivariate dataset set > 1 — the CD then does real cross-channel work. |
| `--n_samples` | 128 | Per-channel sequence length (must satisfy `T = n_samples · n_groups`). |
| `--noise_len` | 32 | Shared-noise dimension. |
| `--num_epochs` | 120 | Training epochs. |
| `--batch_size` | 32 | Training batch. |
| `--gamma` | 5.0 | CD coupling weight. |
| `--glr` / `--dlr` / `--cdlr` | 1e-3 / 1e-3 / 1e-4 | Generator / channel-D / central-D Adam LR. |
| `--gen_num` | 8192 | Paths to generate. |
| `--frac` | 1.0 | Fraction of training paths (smoke runs). |
| `--tag` | "" | Run tag (e.g. `smoke`) — prefixes outputs, skips canonical weights. |

- **Wider LSTM / different noise dim** → the LSTM `hidden_dim = 256` lives in the reference classes'
  signatures (`LSTMGenerator`/`LSTMDiscriminator`); pass a smaller net by editing those defaults, or
  change `--noise_len` for the input dimension.
- **CD strength** → `--gamma` (coupling) and `--cdlr` (CD learning rate).

---

## How to use a different dataset

`train_heston.py --data PATH` takes a `.npy` of shape `(N, T)`, dtype float, in price/level space.
The wrapper scalar-MinMax-scales to `[0, 1]` (stores `scaler_min` / `scaler_max` for exact
inversion), trains, samples, and de-normalises back before saving. For a **multivariate** dataset,
set `--n_groups C` (with `T = n_samples · C`, block channel-major layout); the Central Discriminator
then does its intended cross-channel coordination and the C = 1 degeneracy disappears.

---

## How to produce new seeds

Each seed re-seeds torch + numpy, so model init and sampling differ:

```bash
cd methods/COSCI-GAN/code
# one extra seed
CUDA_VISIBLE_DEVICES=3 /home/tbasseras/gpu-venv/bin/python train_heston.py --seed 5

# all 5 canonical seeds, 2 GPUs in parallel (~4–5 min/seed on an A100)
for s in 0 1 2 3 4; do
  g=$((s % 2 == 0 ? 0 : 3)); c=$(((s % 2)*8))
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 taskset -c $c-$((c+7)) \
    /home/tbasseras/gpu-venv/bin/python train_heston.py --seed $s &
done
wait
```

Each seed writes:
- `weights/seed_{s}_model.pt` — `{G, D, CD}` state_dicts + `scaler_min` / `scaler_max`
- `weights/seed_{s}_config.json` — full hyperparameters + params count (799 618)
- `losses/seed_{s}_losses.csv` — per-epoch `epoch, loss_D_0, loss_G_0, loss_CD`
- `generated_paths/seed_{s}/generated_paths_8192x128.npy` — (8192, 128) price scale
- `generated_paths/seed_{s}/metadata.json` — seed, shape, min/max, train time, params, epochs\_run

---

## Sanity check

The full canonical run (seed 0, ~257 s wall-clock on one A100) confirms healthy training:

```
=== COSCI-GAN Heston (C=1)  seed=0 ===
[data] S(8192, 128) price[min=29.19..]  scaled[0.0000,1.0000]  epochs=120
[model] params=799618  (G+D per channel x1 + MLP CD)
epoch   0  loss_CD=0.6931  loss_D0=0.69..  loss_G0=-2.8..
...
[done] seed=0 epochs=120  loss_CD_last≈0.693  gen=(8192, 128) nan=False  train=256.9s
```

Expected sane signals (all five seeds, verified):
- **`loss_CD` pinned at ln 2 ≈ 0.693** for the entire run — the C = 1 equilibrium signature (CD at
  chance); `loss_D_0` hovers ~0.69–0.75 (channel critic near chance), `loss_G_0` ≈ −2.8 (dominated by
  the −γ·loss_CD term);
- no NaN (`first_nan_epoch = null`, `gen_has_nan = false` in every metadata.json);
- generated mean/std (seed 0: 99.96 / 10.69) close to real (101.33 / 9.97); price range (≈ 29–158)
  brackets the real range;
- 799 618 parameters; 120 epochs; ~257 s/seed.

---

## Reproduce

```bash
# Environment: gpu-venv (torch, CUDA). COSCI-GAN trains on GPU.
cd methods/COSCI-GAN/code

# All 5 Heston seeds (2 GPUs in parallel — see "How to produce new seeds")

# Compute metrics
cd /home/tbasseras/benchmark
/home/tbasseras/gpu-venv/bin/python metrics/compute_all.py --method COSCI-GAN --dataset Heston
```

**Exact run path — which file produced which committed number:**

| Committed number | Interpreter + env | Command | Input file(s) scored | Output file |
|------------------|-------------------|---------|----------------------|-------------|
| Heston A1–A34 + B, per seed `i` | `gpu-venv`, `CUDA_VISIBLE_DEVICES=0` | `metrics/compute_all.py --method COSCI-GAN --dataset Heston` | `methods/COSCI-GAN/generated_paths/seed_i/generated_paths_8192x128.npy` (8192,128) vs real `dataset/Heston/heston_S_8192x128.npy` | `results/Heston/COSCI-GAN/seed_i_metrics.json` (A) + `curve_b_aggregate.json` (B) |
| COSCI-GAN synthetic paths, per seed `i` | `gpu-venv`, `CUDA_VISIBLE_DEVICES=0/3 taskset -c … OMP_NUM_THREADS=8` | `train_heston.py --seed i` | real `dataset/Heston/heston_S_8192x128.npy`, scalar MinMax internally | `generated_paths/seed_i/generated_paths_8192x128.npy` + `metadata.json` |
| A18/A19 disc/pred loss curves, per seed `i` | `gpu-venv` | `metrics/compute_all.py --method COSCI-GAN --dataset Heston` | same generated `.npy` vs real | `results/Heston/COSCI-GAN/seed_i_{disc,pred}_{gru,mlp}_loss.csv` → `plots/{disc_classifier,pred_score}_loss.png` |

Each `results/Heston/COSCI-GAN/seed_i_metrics.json` is the sole source for that seed's column in every
README A-table; the mean±std rows aggregate the 5 files.

The paper reproduction (EEG eye-state, the paper's own Table-4 cross-channel correlation-MAE — the
metric that is undefined at C = 1) lives separately in
[`../paper_reimplementation/`](../paper_reimplementation/) — see its README.
