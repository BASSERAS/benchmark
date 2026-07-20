# TimeVQVAE Code — Sources & Implementation

## Original work

| Field | Details |
|-------|---------|
| **Paper** | *Vector Quantized Time Series Generation with a Bidirectional Prior Model* |
| **Authors** | Daesoo Lee, Sara Malacarne, Erlend Aune |
| **Reference** | AISTATS 2023 (`lee23d`), arXiv:2303.04743 |
| **Original code** | https://github.com/ML4ITS/TimeVQVAE (paper-era commit `b9650e9d`) |
| **Original framework** | PyTorch + PyTorch-Lightning 1.9 |
| **Method type** | **Two-stage vector-quantized generator** — Stage 1 VQ tokenizes the STFT time-frequency map (separate LF / HF branches); Stage 2 a MaskGIT bidirectional-transformer prior samples token maps that are decoded back to signals |

The verbatim reference implementation is kept under [`reference/`](reference/) for transparency
(`.git` stripped, provenance in `reference/PROVENANCE.txt`). Unlike the TimeVAE port, **nothing is
re-derived**: the encoders/decoders, vector-quantizers, MaskGIT prior, losses, and the
`unconditional_sample` generator in `reference/{encoder_decoders,vector_quantization,generators}/`
are executed **as released**. [`train_heston.py`](train_heston.py) only supplies data and orchestrates
the two reference training stages.

---

## Our implementation

The released code already runs on PyTorch/GPU, so there is **no re-implementation** — only a thin
data-plumbing wrapper. [`train_heston.py`](train_heston.py):

1. loads the 8192×128 Heston price paths (`--data`);
2. **global z-normalizes** by the train mean/std (`mean = np.mean(S)`, `std = np.std(S)`,
   `Xn = (S − mean)/std`) — this is the paper's `data_scaling=True`;
3. `os.chdir`s into a **per-seed copy of the reference tree** (`--workdir`) so the reference's
   `get_root_dir()` and its `saved_models/` writes stay isolated between parallel seeds;
4. runs the reference **Stage 1** (VQ-VAE tokenizer) then **Stage 2** (MaskGIT prior), persisting
   Stage-1 modules into `<workdir>/saved_models` exactly as the reference expects;
5. **generates** 8192 length-128 paths with the reference `unconditional_sample(maskgit, gen_num,
   device, batch_size=256)`;
6. inverts the z-norm (`gen_price = x_new*std + mean`) back to price scale, and writes weights /
   losses / generated paths / metadata in the benchmark's standard layout.

The wrapper trains inside [`/home/tbasseras/tvqvae-venv`](#reproduce) (torch 1.13.1+cu117,
PyTorch-Lightning 1.9.0) — the paper-era stack the reference requires. Metrics and plotting use the
shared harness in `/home/tbasseras/gpu-venv`.

### Normalisation chain

```
S(price) --global z-norm (train mean/std)--> ~N(0,1)   (Stage1 + Stage2 train here)
unconditional_sample(maskgit) --x_new*std + mean--> price
```

A **single** global `(mean, std)` (not per-timestep) is used, matching the paper's `data_scaling`.
The pair is stored in each `generated_paths/seed_{i}/metadata.json` (`znorm_mean`, `znorm_std`) so the
generated pool inverts exactly back to price scale.

---

## Architecture (paper config, in\_channels = 1, seq\_len = 128)

| Stage | Component | Setting |
|-------|-----------|---------|
| **STFT** | `n_fft` | 8 → LF branch = freq bin 0; HF branch = freq bins 1: |
| **Stage 1** | Encoder / Decoder (per branch) | ResNet, `encoder_dim = 64`, `n_resnet_blocks = 4` |
| | Downsampled width | LF 8 / HF 32 |
| | Codebook | size 32 (LF) + 32 (HF), `codebook_dim = 64`, EMA decay 0.8 |
| | Perceptual loss weight | 0 |
| **Stage 2** | MaskGIT prior | `hidden_dim = 256`, `n_layers = 4`, `heads = 2`, `ff_mult = 1`, RMSNorm, `p_unconditional = 0.2` |
| | HF prior | conditioned on the LF token map |
| | Sampling | iterative decode `T = 10`, choice temperature 4, `guidance_scale = 1.0` |

All values are read back from `weights/seed_{i}_config.json`.

---

## Wrapper adaptations vs the reference (NO model/loss change)

Three adaptations were needed; **none touches the model architecture, the loss formulas, or the
generation algorithm** — those run verbatim from `reference/`:

1. **PyTorch-Lightning 1.9 environment.** *Location:* whole training stack. The reference pins the
   paper-era PL-1.9 API; we run it under a dedicated `tvqvae-venv` (torch 1.13.1+cu117, PL 1.9.0)
   rather than the newer `gpu-venv` used for metrics.
2. **Per-seed workdir isolation.** *Location:* `train_heston.py::main` (`sys.path.insert` +
   `os.chdir(workdir)`). The reference writes Stage-1 checkpoints to a `saved_models/` folder resolved
   from the CWD; each seed therefore runs inside its **own** copy of `reference/` so two parallel seeds
   never overwrite each other's Stage-1 modules.
3. **Heston data plumbing.** *Location:* `train_heston.py` `HestonImporter`. A drop-in stand-in for the
   reference `DatasetImporterUCR`, it feeds the z-normalized Heston tensor through the reference's own
   data pipeline. The VQ tokenizer, MaskGIT prior and their losses are **unchanged**.

---

## Hyperparameters (paper config)

| Parameter | Value | Source |
|-----------|-------|--------|
| `n_fft` | 8 | `config.yaml` |
| `encoder_dim` | 64 | `config.yaml` |
| `n_resnet_blocks` | 4 | `config.yaml` |
| `downsampled_width` | LF 8 / HF 32 | `config.yaml` |
| `codebook_sizes` | 32 / 32 | `config.yaml` |
| `codebook_dim` | 64 | `config.yaml` |
| `ema_decay` | 0.8 | `config.yaml` |
| MaskGIT `hidden_dim` | 256 | `config.yaml` |
| MaskGIT `n_layers` / `heads` | 4 / 2 | `config.yaml` |
| `p_unconditional` | 0.2 | `config.yaml` |
| `guidance_scale` | 1.0 | `config.yaml` |
| `LR` | 1e-3 | `config.yaml` |
| `weight_decay` | 1e-5 | `config.yaml` |
| `batch_size` | 128 (Stage 1) / 256 (Stage 2) | `config.yaml` |
| `stage1_epochs` | 250 | our budget (see below) |
| `stage2_epochs` | 1000 | our budget (see below) |

**Epoch budget.** The paper trained 2000 / 10000 epochs on *tiny* UCR sets; Heston is ~16× larger, so
250 / 1000 epochs match the paper's **gradient-step count**. Everything else is the paper default.

---

## How to change hyperparameters

Model hyperparameters live in the reference `configs/config.yaml` inside the per-seed `--workdir`
(the wrapper reads `<workdir>/configs/config.yaml` unless `--config` is given). Edit that YAML to change
`n_fft`, codebook sizes, encoder dim, or the MaskGIT prior. Training length is CLI-driven:

| Flag | Default | Effect |
|------|---------|--------|
| `--seed` | 0 | Re-seeds torch + numpy (init + sampling). |
| `--data` | Heston `.npy` | Training paths, shape `(N, T)` price scale. |
| `--workdir` | **required** | Per-seed copy of `reference/` to run inside. |
| `--config` | `<workdir>/configs/config.yaml` | Model config YAML. |
| `--stage1_epochs` | 2000 | Stage-1 (VQ tokenizer) epochs — benchmark uses 250. |
| `--stage2_epochs` | 10000 | Stage-2 (MaskGIT) epochs — benchmark uses 1000. |
| `--gen_num` | 8192 | Paths to generate via `unconditional_sample`. |
| `--frac` | 1.0 | Fraction of training paths (smoke runs). |
| `--tag` | "" | Run tag — prefixes outputs, skips canonical weights. |

---

## How to use a different dataset

Pass `--data PATH` — a `.npy` of shape `(N, T)`, dtype float, in price/level space. The wrapper adds a
channel axis `(N, 1, T)`, global z-normalizes by the train mean/std (stored in metadata for exact
inversion), runs both reference stages, samples, and de-normalises back to the original scale before
saving. Any `T` works; the STFT `n_fft` and downsampled widths in `config.yaml` may need adjusting for
very different lengths.

---

## How to produce new seeds

Each seed needs its **own** `--workdir` (isolated `saved_models/`). Training is heavy (~53 min/seed on
one A100), so seeds are launched in **waves of two GPUs** (0 and 3), respecting the 2-GPU limit:

```bash
cd methods/TimeVQVAE/code
VENV=/home/tbasseras/tvqvae-venv/bin/python
DATA=/home/tbasseras/benchmark/dataset/Heston/heston_S_8192x128.npy

# wave 1: seeds 0,1  → wave 2: seeds 2,3  → wave 3: seed 4
for s in 0 1; do
  g=$([ $s -eq 0 ] && echo 0 || echo 3); c=$(( (s%2)*8 ))
  WD=$(mktemp -d)/wd_seed$s; cp -r reference "$WD"
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 taskset -c $c-$((c+7)) \
    $VENV train_heston.py --seed $s --data $DATA --workdir "$WD" \
    --stage1_epochs 250 --stage2_epochs 1000 &
done
wait
```

Each seed writes:
- `weights/seed_{s}_model.pt` — Stage-1 + MaskGIT `state_dict`
- `weights/seed_{s}_config.json` — full model config + epochs
- `losses/seed_{s}_stage1_losses.csv` — per-epoch VQ losses (recon time/timefreq, commit LF/HF, perplexity LF/HF, perceptual)
- `losses/seed_{s}_stage2_losses.csv` — per-epoch `stage, epoch, loss, prior_loss`
- `generated_paths/seed_{s}/generated_paths_8192x128.npy` — (8192, 128) price scale
- `generated_paths/seed_{s}/metadata.json` — seed, shape, znorm mean/std, gen/real mean/std, train time, epochs, last losses, `gen_has_nan`

---

## Sanity check

The full canonical run (seed 0, ~3214 s ≈ 53 min on one A100) confirms healthy two-stage training:

```
[data] S(8192, 128) price  →  znorm mean=101.33 std=9.97
[stage1] 250 epochs  last recon+commit loss ≈ 0.0244
[stage2] 1000 epochs last prior_loss ≈ 0.865
generated price mean=102.02 std=9.93   real mean=101.33 std=9.97   gen_has_nan=false
```

Expected sane signals (all five seeds, verified):
- **Stage-1** total loss falls from ~2.6 (epoch 0) to a low plateau (~0.02); LF/HF codebook perplexity
  stays well above 1 (codebook not collapsed);
- **Stage-2** `prior_loss` (masked-token cross-entropy) falls from ~5.4 toward ~0.86 — no NaN
  (`gen_has_nan = false` in every metadata.json);
- generated mean/std (102.0 / 9.93) close to real (101.33 / 9.97); generated price stays in a sane
  band around 100.

---

## Reproduce

```bash
# Training environment: tvqvae-venv (torch 1.13.1+cu117, PyTorch-Lightning 1.9.0)
cd methods/TimeVQVAE/code

# All 5 Heston seeds (2 GPUs per wave — see "How to produce new seeds")

# Compute metrics (shared harness, gpu-venv)
cd /home/tbasseras/benchmark
/home/tbasseras/gpu-venv/bin/python metrics/compute_all.py --method TimeVQVAE --dataset Heston
```

**Exact run path — which file produced which committed number:**

| Committed number | Interpreter + env | Command | Input file(s) scored | Output file |
|------------------|-------------------|---------|----------------------|-------------|
| Heston A1–A34 + B, per seed `i` | `gpu-venv`, `CUDA_VISIBLE_DEVICES=0` | `metrics/compute_all.py --method TimeVQVAE --dataset Heston` | `methods/TimeVQVAE/generated_paths/seed_i/generated_paths_8192x128.npy` (8192,128) vs real Heston `dataset/Heston/heston_S_8192x128.npy` | `results/Heston/TimeVQVAE/seed_i_metrics.json` (A) + `curve_b_aggregate.json` (B) |
| TimeVQVAE synthetic paths, per seed `i` | `tvqvae-venv`, `CUDA_VISIBLE_DEVICES=0/3 taskset -c … OMP_NUM_THREADS=8` | `train_heston.py --seed i --workdir <copy of reference> --stage1_epochs 250 --stage2_epochs 1000` | real Heston `dataset/Heston/heston_S_8192x128.npy`, global z-norm internally | `generated_paths/seed_i/generated_paths_8192x128.npy` + `metadata.json` |
| Stage-1 / Stage-2 loss curves, per seed `i` | `tvqvae-venv` | `train_heston.py --seed i …` (loss logged per epoch) | training tensors | `losses/seed_i_stage{1,2}_losses.csv` → `losses/loss_convergence.png` |
| A18/A19 disc/pred loss curves, per seed `i` | `gpu-venv` | `metrics/compute_all.py --method TimeVQVAE --dataset Heston` | same generated `.npy` vs real | `results/Heston/TimeVQVAE/seed_i_{disc,pred}_{gru,mlp}_loss.csv` → `plots/{disc_classifier,pred_score}_loss.png` |
| Path-Shadowing MC (CRPS/MAE/RMSE) | `gpu-venv` | `methods/DiffusionTS/path_shadowing/run_eval.py --method TimeVQVAE` | generated pool vs real query paths | `results/Heston/TimeVQVAE/path_shadowing/seed_i_results.json` + `summary.json` |

Each `results/Heston/TimeVQVAE/seed_i_metrics.json` is the sole source for that seed's column in every
README A-table; the mean±std rows aggregate the 5 files.

The paper reproduction (ECG5000, the paper's own FID / IS metrics vs Table) lives separately in
[`../paper_reimplementation/`](../paper_reimplementation/) — see its README (FID 0.739±0.084 vs paper
0.7; IS 2.019±0.012 vs paper 2.0).
