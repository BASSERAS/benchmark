# LS4 — Paper Reimplementation (Solar Weekly)

Reproduction of the **LS4** paper result on the **Solar Weekly** dataset using the
*official* LS4 code, run verbatim from the authors' Monash training script.

- **Paper:** *Deep Latent State Space Models for Time-Series Generation* — Linqi Zhou,
  Michael Poli, Winnie Xu, Stefano Massaroli, Stefano Ermon, **ICML 2023**
  (PMLR 202, pp. 42625–42643), arXiv:2212.12749.
- **Official code:** `github.com/alexzhou907/ls4` (mirrored here under `../code/reference/`).
- **This run:** `../code/reference/train_monash.py` with
  `configs/monash/vae_solarweekly_released.yaml`, single A100 (GPU logical 0).
  Wall-clock to the reported checkpoint (~32 k eval epochs): **≈ 2 h 45 m**.

---

## ⚠️ Reproduction caveat — the Cauchy fix (required to reproduce)

The official code **as-shipped does not reproduce the paper on this machine**, and the
reason is subtle rather than cosmetic. LS4's `model.generate` rolls the S4 latent prior
forward with `latent.step` — the **STEP-mode recurrence** (one timestep at a time), not
the convolutional (scan) path used during training. On a CUDA-13 A100 the fast Cauchy
kernels (`pykeops` / the bundled CUDA extension) are unavailable, so S4 falls back to the
**naive Python Cauchy kernel** in `reference/models/s4.py`. That fallback, as written
upstream, sums the kernel over the *full* pole set instead of over **conjugate pole
pairs**, which is correct for the `keops`/CUDA path but wrong for the naive path used at
generation time.

**Symptom (measured on disk, naive kernel — `../code/reference/outputs_shipped_train.log`):**
the Solar-Weekly **marginal score plateaus at 0.197 ± 0.003** (epoch 25 400) and never
descends — ~4× the paper's 0.046 — while the training MSE looks healthy (0.004). The
generator is fine; the *generation-time recurrence* is silently mis-summed.

**Fix (the only change to reference model code):** `reference/models/s4.py:795`, patched so
the naive Cauchy kernel sums over conjugate pole **pairs**, matching the keops/CUDA result.
With the fix (`../code/reference/outputs_shipped_fix_train.log`) the marginal score reaches
**0.047 ± 0.003 (best 0.044)** — i.e. the paper regime. This same fix is carried into the
Heston generator (documented in `../code/README.md`, fix #1), because Heston generation
uses the identical `latent.step` recurrence.

No other reference code was modified; hyperparameters, data pipeline and metric code are
upstream-verbatim.

---

## 1. Paper metrics (as defined in the paper)

LS4 evaluates Monash generative quality with three post-hoc scores (Zhou et al. 2023, §5.1;
they follow the GT-GAN / TimeGAN evaluation protocol):

| Metric | Definition | Direction |
|--------|-----------|-----------|
| **Marginal score** | Absolute difference between the empirical marginal (per-timestep value) histograms of real vs synthetic, averaged over bins. | lower = better |
| **Classification score** | A post-hoc sequence classifier is trained to separate real from synthetic; the reported number is the classifier's normalised score on held-out data. | higher = better |
| **Prediction score** | TSTR: a sequence model is trained on **synthetic** to predict the next step and evaluated (MAE) on **real**. | lower = better |

The training script logs all three every `eval_iter = 100` epochs as
`(current_best, running_mean +- std)` over the evaluation resamples; we report the
running mean ± std at the reported checkpoint.

---

## 2. Hyperparameters (from the official released config)

Exact settings from `configs/monash/vae_solarweekly_released.yaml` (the authors' released
Solar-Weekly preset), applied verbatim:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `z_dim` | 5 | latent state dimension |
| `in_channels` | 1 | Solar Weekly is univariate |
| `d_model` | 64 | S4 hidden width (prior / posterior / decoder) |
| `d_state` | 64 | S4 state size (SSM order) |
| `n_layers` | 4 | S4 blocks per module |
| `backbone` | `autoreg` | autoregressive backbone |
| `s4_type` | `s4` | vanilla S4 kernel |
| `latent_type` | `split` | prior uses the split-latent parameterisation |
| `sigma` | 0.1 | decoder observation noise |
| `lr` | 1e-3 | AdamW learning rate (weight_decay 0) |
| `batch_size` | 128 | |
| `epochs` | 100 000 (max) | we report the ~32 k-epoch checkpoint |
| `eval_iter` | 100 | metrics logged every 100 epochs |
| `preproc` | `normalize_per_seq` | per-sequence normalisation (upstream default) |

Training wall-clock to the reported checkpoint: **≈ 2 h 45 m** on one A100 (naive Cauchy /
Vandermonde fallback — no CUDA extension, hence slower than the paper's keops run).

---

## 3. Dataset

**Solar Weekly** — the Monash Time Series Forecasting Repository weekly solar-power series,
one of the exact generative benchmarks in the LS4 paper. The upstream loader reads the
`.tsf` directly and windows it internally (`normalize_per_seq`), so no manual re-windowing
is needed.

- Source file: `dataset/solar_weekly_dataset.tsf` (Monash TSF `.tsf` format).
- Univariate (`in_channels = 1`); scaling is per-sequence normalisation done inside the loader.
- The sibling `dataset/nn5_daily_dataset_without_missing_values.tsf` is the NN5-Daily series
  (kept for reference; not part of the reported reproduction).

---

## 4. Results — ours vs paper

| Dataset | Metric | **Ours (official LS4 code, Cauchy-fixed)** | Ours (as-shipped, naive Cauchy) | Paper (Table 1) | Verdict |
|---------|--------|--------------------------------------------|---------------------------------|-----------------|---------|
| Solar Weekly | Marginal ↓ | **0.047 ± 0.003** (best 0.044) | 0.197 ± 0.003 | 0.0459 | **matches** ✓ |
| Solar Weekly | Classification ↑ | **0.717 ± 0.097** (best 0.771) | 0.001 ± 0.001 | 0.683 | same regime ✓ |
| Solar Weekly | Prediction ↓ | **0.113 ± 0.036** (best 0.076) | 0.624 ± 0.078 | 0.141 | **matches** ✓ |

**Reproduced — but only after the Cauchy fix.** With the fix, the marginal score
(0.047 vs paper 0.0459) and prediction score (0.113 vs 0.141) land squarely in the paper
regime, and the classification score (0.717) meets/slightly exceeds the paper's 0.683. The
as-shipped naive-Cauchy column is the *broken* run: a marginal score frozen at 0.197 and a
classification score of ~0 (the classifier trivially separates the degenerate samples) — a
clear signature of the mis-summed generation-time recurrence, **not** a hyperparameter or
data issue (training MSE was healthy in both runs).

Numbers are read directly from the on-disk training logs
(`../code/reference/outputs_shipped_fix_train.log`, epoch 32 000, and
`../code/reference/outputs_shipped_train.log`, epoch 25 400); the metric lines have the
form `... | clf_score:(best, mean +- std) | marginal_score:(...) | predictive_score:(...)`.

---

## 5. How to reproduce (EXACT run path — mandatory)

The metrics are computed **inline by the upstream training script** every 100 epochs (there
is no separate metric binary for the Monash reproduction), so "reproduce" = re-run training
with the released config and read the logged `marginal_score / clf_score / predictive_score`.

```bash
cd ../code/reference
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-7 \
  /home/tbasseras/gpu-venv/bin/python train_monash.py \
  --config configs/monash/vae_solarweekly_released.yaml \
  2>&1 | tee outputs_shipped_fix_train.log
```

**The Cauchy fix must be applied first** (`reference/models/s4.py:795`, conjugate-pair
Cauchy sum) — without it you reproduce the *broken* 0.197 marginal column, not the paper.
To reproduce the broken baseline instead, revert that one line and log to
`outputs_shipped_train.log`.

**Exact run path — which file feeds which number (so any cell is traceable):**

| Table cell | Interpreter + env | Script + config | Data scored | Log the number is read from |
|------------|-------------------|-----------------|-------------|-----------------------------|
| §4 Cauchy-fixed row (0.047 / 0.717 / 0.113) | `gpu-venv`, `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-7` | `train_monash.py` + `configs/monash/vae_solarweekly_released.yaml` (**with** the s4.py:795 fix) | `dataset/solar_weekly_dataset.tsf` (per-seq normalised in-loader) | `../code/reference/outputs_shipped_fix_train.log`, epoch 32 000 |
| §4 as-shipped row (0.197 / 0.001 / 0.624) | same | same config, **without** the fix | same | `../code/reference/outputs_shipped_train.log`, epoch 25 400 |

Every §4 number is a field on one `Epoch: N | ...` line of the named log; the script
computes load → train → evaluate inline (no intermediate hand-editing).

---

## 6. Files

| Path | Content |
|------|---------|
| `dataset/solar_weekly_dataset.tsf` | Solar Weekly series (Monash `.tsf`) — the reproduced dataset |
| `dataset/nn5_daily_dataset_without_missing_values.tsf` | NN5-Daily series (reference only, not reported) |
| `LS4_ICML2023.pdf` | the paper (Zhou et al., ICML 2023) |
| `results/`, `logs/`, `metric/` | empty — the Monash reproduction logs metrics inline in the training log under `../code/reference/outputs_*_train.log`; no separate metric JSON is produced |

> **Note.** Unlike the Heston phase (which ships committed per-seed metric JSONs under
> `results/Heston/LS4/`), the paper reproduction is documented from the upstream training
> logs, because the official Monash pipeline emits its scores inline rather than as a
> standalone result file. The exact log paths and epochs are given in §4 and §5.
