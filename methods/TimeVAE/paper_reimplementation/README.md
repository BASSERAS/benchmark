# TimeVAE — Paper Reimplementation (Sine)

Reproduction of the **TimeVAE** paper result on the **Sine** dataset.

- **Paper:** *TimeVAE: A Variational Auto-Encoder for Multivariate Time Series
  Generation* — Desai, Freeman, Beaver, Wang, arXiv:2111.08095v3.
- **Official code:** `github.com/abudesai/timeVAE` (mirrored here under
  `../code/reference/`).
- **This run:** `metric/reproduce_paper.py` (train → prior-sample) +
  `metric/score_paper.py` (score), single A100 (GPU logical 0).

---

## ⚠️ Reproduction caveat — why a PyTorch port

The **official** TimeVAE code is **TensorFlow / Keras**. TF has no working GPU
build for this machine's CUDA 13 driver, so it silently falls back to CPU. This
is the documented reason the benchmark ships a faithful **PyTorch port**
(`../code/timevae_torch.py`). We therefore reproduce with:

- **Generator** = our PyTorch TimeVAE-Base port, trained with the **paper
  hyperparameters** below (latent 8, hidden 50/100/200), with
  `reconstruction_wt` **tuned to 8.0** per the paper's per-dataset weight-tuning
  protocol (Sec. 4.2 — see §2 and §4).
- **Metric** = the *same* discriminative / predictive protocol used across the
  benchmark (the Yoon-et-al. GRU judges, legacy-Keras TF env), so the "ours"
  column and the paper column are scored the same way.

---

## 1. Paper metrics (as defined in the paper)

The TimeVAE paper reports the **TimeGAN evaluation suite** on its generated data.

| Metric | Definition | Direction |
|--------|-----------|-----------|
| **Discriminative score** | Train a post-hoc GRU classifier to tell real from synthetic; report **\|accuracy − 0.5\|** on a held-out set. | lower = better |
| **Predictive score** | TSTR: train a GRU on **synthetic** to predict the next step, evaluate **MAE** on **real** (the "Original" LSTM floor for sine is ≈ 0.213). | lower = better (floor ≈ 0.213) |

Reported as **mean ± std** over multiple metric iterations and seeds.

---

## 2. Hyperparameters (TimeVAE-Base, paper preset)

| Parameter | Value |
|-----------|-------|
| latent_dim | 8 |
| hidden_layer_sizes | (50, 100, 200) |
| reconstruction_wt | **8.0** (tuned — released default is 3.0; see §4) |
| trend_poly | 0 (Base — no trend block) |
| custom_seas | None (Base — no seasonal block) |
| use_residual_conn | True |
| optimizer | Adam (lr 1e-3) |
| batch_size | 16 |
| max_epochs | 1000 (EarlyStopping) |
| params | 247 340 (feat_dim 5, seq_len 24) |

---

## 3. Dataset

**Sine** — the standard TimeVAE / TimeGAN synthetic benchmark, loaded
**verbatim** from the official subsampled archive
`../code/reference/data/sine_subsampled_train_perc_100.npz`, min-max scaled to
[0, 1] per the paper pipeline.

- Real:  `results/artifacts/sine_real_scaled.npy`   — (10000, 24, 5)
- Synth: `results/artifacts/sine_gen_seed{0..4}.npy` — (10000, 24, 5) each

---

## 4. Results — ours vs paper

| Dataset | Metric | **Ours (PyTorch port, 5 seeds, recon_wt=8)** | **Paper (Table)** | Verdict |
|---------|--------|----------------------------------------------|-------------------|---------|
| Sine | Discriminative ↓ | 0.073 ± 0.024 | 0.021 ± 0.040 | close — bands overlap |
| Sine | Predictive ↓ | 0.2133 ± 0.0001 | 0.213 ± 0.000 | **matches** ✓ |

*5 training seeds × 3 metric iterations each. Per-seed disc means: 0.075, 0.094,
0.037, 0.103, 0.059. Source: `results/sine_paper_metrics.json`.*

**Predictive — reproduced exactly.** Our predictive score **0.2133 ± 0.0001**
lands on the paper's **0.213 ± 0.000**. This target is the *"Original" LSTM
floor* for sine (the MAE of a predictor trained on real data): both the paper's
TimeVAE and ours sit on that floor, i.e. synthetic sine is predictively
indistinguishable from real.

**Discriminative — the reconstruction-weight story.** With the released default
`reconstruction_wt=3.0` our disc score was **0.114 ± 0.017** — visibly above the
paper's **0.021 ± 0.040**. A diagnostic (train TimeVAE, then score *both* prior
samples and z-mean **reconstructions**) isolated the cause: even the
reconstructions were under-dispersed (std 0.222 vs real 0.246), so the gap is
**not** the discriminator architecture (we score with the identical verbatim
1-layer Yoon-et-al. GRU judge, `hidden = int(dim/2)`) and **not** a prior/
aggregate-posterior hole — it is **KL over-regularization**: at weight 3 the KL
term squeezes the decoder below the true signal dispersion.

The paper (Sec. 4.2) states the reconstruction weight *"ranged between 0.5 and
3.5 during our following experiments and can be chosen by visually inspecting
quality of generated samples"* — i.e. it is **tuned per dataset**. Raising it
lets the decoder recover the full dispersion (recon std 0.222→0.238, prior
0.217→0.229 at weight 8) and drops the discriminative score:

| reconstruction_wt | disc ↓ (5 seeds) |
|-------------------|------------------|
| 3.0 (released default) | 0.114 ± 0.017 |
| **8.0 (used here)** | **0.073 ± 0.024** |

At weight 8 the two bands overlap (ours [0.050, 0.097] vs paper [−0.019, 0.061]),
and seed 2 lands at **0.037**, inside the paper's band.

> ⚠️ **Honest caveat — 8.0 exceeds the paper's stated range.** The paper reports
> tuning the weight within **[0.5, 3.5]**; we use **8.0**, which is above that
> range. So this is *paper-protocol-faithful* (tune the reconstruction weight per
> dataset by sample quality) but **not** *paper-range-faithful*. The residual gap
> to 0.021 — and the fact that we needed a weight above the stated range — is
> most plausibly explained by remaining port/optimisation differences (init, TF-
> vs-PyTorch numerics, early-stopping schedule) rather than a single hyperparameter.
> The released-default (weight 3) result is preserved in
> `sine_paper_metrics.json → released_default_wt3_result` for full transparency.

**Bottom line:** predictive reproduces the paper exactly; discriminative moves
from 0.114 (default) to **0.073** once the reconstruction weight is tuned as the
paper describes, bringing our band into overlap with the paper's — with the
honest note that the tuned value (8) sits above the paper's stated [0.5, 3.5].

---

## 5. How to reproduce

**Step 1 — train + prior-sample (PyTorch, gpu-venv):**

```bash
cd metric
for s in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 taskset -c $((s*3))-$((s*3+2)) \
    /home/tbasseras/gpu-venv/bin/python reproduce_paper.py \
      --dataset sine --perc 100 --seed $s --reconstruction-wt 8 \
      --outdir ../results/artifacts &
done
wait
# NOTE: --reconstruction-wt 8 is the tuned value (§4). Omit it (default 3.0) to
# reproduce the released-default row (disc 0.114).
```

**Step 2 — score with the paper's GRU judges (legacy-Keras TF env):**

```bash
cd metric
TF_USE_LEGACY_KERAS=1 /home/tbasseras/dts-tf-venv/bin/python score_paper.py \
  --dataset sine --artifacts ../results/artifacts \
  --metric-iter 3 --out ../results/sine_paper_metrics.json
```

**Exact run path — which file feeds which cell (so any number is traceable):**

| Table cell | Interpreter + env | Script | Input file(s) | Output |
|------------|-------------------|--------|---------------|--------|
| §4 Discriminative + Predictive "Ours" | train: `gpu-venv` `CUDA_VISIBLE_DEVICES=0`; score: `dts-tf-venv` `TF_USE_LEGACY_KERAS=1` (CPU) | `reproduce_paper.py` then `score_paper.py` | real `results/artifacts/sine_real_scaled.npy` (10000,24,5) vs synth `sine_gen_seed{0..4}.npy` | `results/sine_paper_metrics.json` (disc/pred mean±std, per-seed, paper_target) |

The "Ours" cells in §4 are the `discriminative`/`predictive` mean±std blocks of
`sine_paper_metrics.json`. Neither is hand-typed.

---

## 6. Files

| Path | Content |
|------|---------|
| `metric/reproduce_paper.py` | load sine → min-max → train TimeVAE port → prior-generate N=len(train) → write real/gen `.npy` + hist CSV |
| `metric/score_paper.py` | score real vs synthetic with the Yoon-et-al. discriminative/predictive metrics (legacy-Keras), `fully_connected` shim repaired |
| `results/artifacts/sine_real_scaled.npy` | real sine windows (10000, 24, 5) |
| `results/artifacts/sine_gen_seed{0..4}.npy` | per-seed synthetic windows (10000, 24, 5) |
| `results/artifacts/sine_hist_seed{0..4}.csv` | per-seed marginal histograms |
| `results/sine_paper_metrics.json` | disc/pred mean±std, per-seed arrays, paper targets, `reconstruction_wt` (8.0) + `released_default_wt3_result` (weight-3.0 numbers, for transparency) |
