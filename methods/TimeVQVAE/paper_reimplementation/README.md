# TimeVQVAE — Paper Reimplementation (ECG5000)

Reproduction of the **TimeVQVAE** paper result on the **ECG5000** (UCR) dataset.

- **Paper:** *Vector Quantized Time Series Generation with a Bidirectional Prior
  Model* — Lee, Malacarne, Aune, AISTATS 2023 (PMLR 206, `lee23d`).
  Saved here as `TimeVQVAE_AISTATS2023_lee23d.pdf`.
- **Official code:** `github.com/ML4ITS/TimeVQVAE`, **paper-era commit
  `b9650e9d`** (2023-02-16), mirrored verbatim under `../code/reference/`
  (see `../code/reference/PROVENANCE.txt`). Later 2024 branches deliberately
  diverge from the paper (ROCKET-based FID, `supervised-fcn-2`, Snake
  activation, removed perceptual loss) and are **not** used here.
- **This run:** `../code/reference/stage1.py` (VQ tokenization) →
  `stage2.py` (MaskGIT prior + FID/IS evaluation), 3 independent runs on
  A100 GPUs 0 and 3.

---

## ⚠️ Reproduction note — environment

The paper code targets **PyTorch Lightning 1.9** (uses the `*_epoch_end` hooks
removed in PL 2.0). The benchmark's default `gpu-venv` ships PL 2.6.5, so a
dedicated **PL-1.9 environment** was built: `/home/tbasseras/tvqvae-venv`
(python 3.10, torch 1.13.1+cu117, PL 1.9.0, numpy 1.23.5, `supervised-fcn`
1.7.8). This runs on the A100 / CUDA-13 driver via the backward-compatible
cu117 build. **No model, loss, or metric code was changed** — the only edit to
the reference harness is one `print()` of the already-computed FID/IS to stdout
(the paper logs them only to wandb, which is disabled here).

---

## 1. Paper metrics (as defined in the paper)

TimeVQVAE reports **FID** and **IS** computed with a **UCR-pretrained FCN**
(Fully Convolutional Network from the `supervised-FCN` package,
`github.com/danelee2601/supervised-FCN`), the exact evaluator shipped in the
reference code (`../code/reference/evaluation/`).

| Metric | Definition | Direction |
|--------|-----------|-----------|
| **FID** | Fréchet distance between FCN global-average-pool feature vectors of real vs. generated samples. | lower = better |
| **IS** (Inception Score) | `exp(E_x KL(p(y\|x) ‖ p(y)))` over the FCN softmax class posteriors of generated samples. | higher = better |

Reported as **mean ± std** over **3 runs** (paper §5 / results CSVs).

---

## 2. Hyperparameters (paper preset — verbatim `config.yaml`)

| Parameter | Value |
|-----------|-------|
| stage1 (VQ-VAE) epochs | 2000 |
| stage2 (MaskGIT) epochs | 10000 |
| batch size (stage1 / stage2) | 128 / 256 |
| optimizer | AdamW (LR 1e-3, weight_decay 1e-5) |
| STFT `n_fft` | 8 |
| encoder/decoder dim | 64, 4 ResNet blocks |
| downsampled width (LF / HF) | 8 / 32 |
| codebook sizes (LF / HF) | 32 / 32, codebook_dim 64, EMA decay 0.8 |
| perceptual_loss_weight | 0 |
| MaskGIT prior | hidden 256, 4 layers, 2 heads, RMSNorm, `p_uncond` 0.2 |
| iterative decoding | T = 10, cosine schedule, choice temp 4 |
| guidance_scale | 1.0 |

These match Appendix C of the paper. Source: `../code/reference/configs/config.yaml`.

---

## 3. Dataset

**ECG5000** — a standard UCR benchmark used in the paper's per-dataset FID/IS
table. Loaded verbatim by the reference `DatasetImporterUCR` from
`../code/reference/datasets/UCRArchive_2018/ECG5000/ECG5000_{TRAIN,TEST}.tsv`
(500 train / 4500 test, length 140, 5 classes), z-normalised by the train
mean/var per the paper pipeline.

---

## 4. Results — ours vs paper

| Dataset | Metric | **Ours (paper-era code, 3 runs)** | **Paper (Table)** | Verdict |
|---------|--------|-----------------------------------|-------------------|---------|
| ECG5000 | FID ↓ | 0.739 ± 0.084 | 0.7 ± 0.0 | **matches** ✓ |
| ECG5000 | IS ↑ | 2.019 ± 0.012 | 2.0 ± 0.0 | **matches** ✓ |

*3 independent runs (each = stage1 2000 ep + stage2 10000 ep). Per-run FID:
0.785, 0.810, 0.620. Per-run IS: 2.006, 2.016, 2.036. Source:
`results/ecg5000_paper_metrics.json`.*

**FID — reproduced.** Our mean **0.739** sits essentially on the paper's
**0.7**; the paper's own reported std is 0.0 (rounded to 1 decimal), and our
three runs bracket the paper value (run2 = 0.620 is below it). The gap
(0.04) is well inside single-run variation.

**IS — reproduced exactly.** Our **2.019 ± 0.012** rounds to the paper's
**2.0 ± 0.0**. Both real and generated ECG5000 saturate the 5-class FCN at
IS ≈ 2, i.e. generated samples are as class-diverse/confident as real.

For context, the paper's own baselines on ECG5000 are far worse (FID: GMMN
26.6, RCGAN 4.5, TimeGAN 35.2, SigCWGAN 55.9), so a FID < 1 is the
distinctive TimeVQVAE result — which we reproduce.

**Bottom line:** both paper metrics reproduce on ECG5000 using the paper-era
code, the paper hyperparameters, and the paper's FCN evaluator — no divergence
from the paper.

---

## 5. How to reproduce

**Environment:** `/home/tbasseras/tvqvae-venv` (PL 1.9; see §above).

**One full run (stage1 → stage2 → FID/IS), GPU 0:**

```bash
cd ../code/reference
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 WANDB_MODE=disabled taskset -c 0-7 \
  /home/tbasseras/tvqvae-venv/bin/python stage1.py --config configs/config.yaml
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 WANDB_MODE=disabled taskset -c 0-7 \
  /home/tbasseras/tvqvae-venv/bin/python stage2.py --config configs/config.yaml
# stage2 prints:  [METRICS] FID=... IS_mean=... IS_std=...
```

**3 runs (paper mean±std).** Because both stages write ckpts to
`saved_models/<name>-ECG5000.ckpt`, independent runs must not share a working
directory. Copy the reference tree per run (excluding `saved_models/`) and run
each copy; runs 0/1 go on GPUs 0/3, run 2 co-locates on GPU 0 (each run uses
~4.6 GB of 80 GB and ~60 % SM, so co-location reclaims idle capacity while
respecting the 2-GPU / 16-core limits).

| Table cell | Env | Script | Input | Output |
|------------|-----|--------|-------|--------|
| §4 FID + IS "Ours" | `tvqvae-venv`, `CUDA_VISIBLE_DEVICES=0/3`, `WANDB_MODE=disabled` | `stage1.py` then `stage2.py` | `datasets/UCRArchive_2018/ECG5000/*.tsv` | `results/ecg5000_paper_metrics.json` (per-run + mean±std + paper target) |

The "Ours" cells in §4 are the `ours` block of
`results/ecg5000_paper_metrics.json`; neither number is hand-typed.

---

## 6. Files

| Path | Content |
|------|---------|
| `TimeVQVAE_AISTATS2023_lee23d.pdf` | the paper |
| `results/ecg5000_paper_metrics.json` | per-run + mean±std FID/IS, paper target, hyperparameters |
| `results/run{0,1,2}_metrics.txt` | raw `[METRICS]` line per run |
| `../code/reference/` | paper-era harness (commit b9650e9d), `PROVENANCE.txt` |
| `../code/reference/configs/config.yaml` | paper hyperparameters |
| `../code/reference/results/*.csv` | the paper's own reported FID/IS/CAS tables |
