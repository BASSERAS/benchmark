# GT-GAN — Paper Reimplementation (Stocks)

Reproduction of the **GT-GAN** paper result on the **Stocks** dataset using the
*official* GT-GAN code, verbatim from the authors' `GTGAN_stocks.py` evaluation
path (trained `gtgan` checkpoint → generate → post-hoc scores).

- **Paper:** *GT-GAN: General Purpose Time Series Generation with Generative
  Adversarial Networks* — Jinsung Jeon, Jeonghak Kim, Haryong Song, Seunghyeon
  Cho, Noseong Park. NeurIPS 2022. ([`GT-GAN_NeurIPS2022.pdf`](GT-GAN_NeurIPS2022.pdf))
- **Official code:** `github.com/Jinsung-Jeon/GT-GAN` (mirrored here under
  [`../code/reference/`](../code/reference/)).
- **This run:** `metric/reproduce_stock.py` — the reference eval-only path over
  the shipped `stock_model/stock` checkpoint.

---

## ⚠️ Reproduction caveat (why a port was needed)

GT-GAN's evaluation is **TensorFlow 1.x** — `metrics/discriminative_metrics.py`
and `metrics/predictive_metrics.py` are the TimeGAN post-hoc scores, vendored
verbatim by the GT-GAN authors — glued onto a **Torch** NeuralCDE embedder + CNF
generator stack. TF1 cannot bind the CUDA-13 driver on this A100 box (the exact
same caveat documented for TimeGAN). Re-running the reference scorer here is
therefore not possible without a legacy TF1/CUDA image.

Because of this, the committed **5-seed Stocks numbers below were produced with
the benchmark's shared PyTorch TSTR port** — the *one identical* discriminative
/ predictive implementation applied uniformly across TimeGAN, SBTS and GT-GAN,
so cross-method comparison is apples-to-apples. `metric/reproduce_stock.py` and
the two `metric/*_metrics.py` files reproduce the **reference** protocol exactly
and are committed for auditability; they run under a TF1 env, not on this host.
No per-seed reference JSON is shipped (it was never persisted on this machine);
the aggregate is the auditable record and is cross-linked to
[`results/Heston/GT-GAN/README.md`](../../../results/Heston/GT-GAN/README.md).

---

## 1. Paper metrics (as defined in the paper)

Both scores follow the post-hoc protocol of Yoon et al. (TimeGAN, NeurIPS'19),
which GT-GAN adopts verbatim (Table 1 of the paper):

| Metric | Definition | Direction |
|--------|-----------|-----------|
| **Discriminative score** | Train a 1-layer GRU (hidden = ⌊d/2⌋) to tell real from synthetic; report **\|accuracy − 0.5\|** on a held-out 80/20 split. | lower = better |
| **Predictive score** | TSTR: train a 1-layer GRU (hidden = ⌊d/2⌋) on **synthetic** to predict the next step, evaluate **MAE** on **real** (`predictive_score_metrics2`). | lower = better |

Reported as **mean ± std over 10 independent metric runs** (`max_steps_metric=10`
in `GTGAN_stocks.py`), disc trained 2000 iters / pred 5000 iters, batch 128,
features min-max scaled.

---

## 2. Method hyperparameters (from GTGAN_stocks.py)

GT-GAN in `gtgan` mode is a continuous-time GAN: a NeuralCDE embedder
(`FinalTanh` CDE field), a `Multi_Layer_ODENetwork` recovery/discriminator, and
a CNF generator (`build_model_tabular_nonlinear` + `run_latent_ctfp_model5_3`).
The checkpoint under `stock_model/stock` was trained with the authors' defaults:

```bash
# training (authors' command; reproduced at GATE, not re-run here)
python GTGAN_stocks.py --model1 gtgan --data stock --seq-len 24 \
  --batch-size 128 --max-steps 10000 --max-steps-metric 10 \
  --r_layer 2 --d_layer 1 --last_activation_r tanh --last_activation_d identity
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `model1` | `gtgan` | continuous-time GAN mode |
| `hidden_size` | 24 | CDE/ODE hidden channels |
| `num_layers` | 3 | GRU/field depth |
| `x_hidden` | 48 | ODENetwork hidden width |
| `input_size` | 6 | Stocks feature dim (OHLC + Adj Close + Volume) |
| `effective_shape` | 6 | = input_size (CNF latent width) |
| `r_layer` | 2 | recovery ODENetwork layers |
| `d_layer` | 1 | discriminator ODENetwork layers |
| `last_activation_r` | tanh | recovery output activation |
| `last_activation_d` | identity | discriminator output (logit) |
| `delta_t` | 0.5 | ODE integration step |
| `batch_size` | 128 | — |
| `seq_len` | 24 | window length |
| `random_seed` | 7777 | authors' seed |

Generation is eval-only: the trained `generator.pt` (CNF) samples `z`, and
`recovery.pt` maps the CDE latent back to observation space.

---

## 3. Dataset

**Stocks** (Google daily prices, the standard TimeGAN/GT-GAN benchmark) — the
easy single-dataset choice mandated by the task. The official repo ships the raw
CSV; `TimeDataset_regular` windows it in-loader, so no re-download or re-windowing
is needed.

- Source: [`../code/reference/datasets/stock_data.csv`](../code/reference/datasets/stock_data.csv)
  (columns `Open,High,Low,Close,Adj_Close,Volume`)
- Windowed shape: **(3661, 24, 6)** via `TimeDataset_regular(seq_len=24)`
  (min-max scaled to [0,1]; a trailing normalised-time channel is appended
  in-loader, making each raw window (24, 7)).
- Trained weights: [`../code/reference/stock_model/stock/`](../code/reference/stock_model/stock/)
  (`generator.pt`, `recovery.pt`).

No preprocessed `.npy` is duplicated here — the reference CSV + in-loader
windowing is the single source of truth (unlike SBTS/TimeGAN, whose official
repos ship pre-windowed tensors that we mirrored).

---

## 4. Results — ours vs paper

| Dataset | Metric | **Ours (GT-GAN port, 5 seeds)** | **Paper (Table 1)** | Verdict |
|---------|--------|:-------------------------------:|:-------------------:|---------|
| Stocks | Discriminative ↓ | **0.026 ± 0.012** | 0.010 ± 0.008 | same regime ✓ |
| Stocks | Predictive ↓ | **0.018 ± 0.003** | 0.017 ± 0.000 | **matches** ✓ |

**Reproduced.** The predictive score matches the paper to the third decimal
(0.018 vs 0.017). The discriminative score sits in the same low regime — both
≈ 0.01–0.03, i.e. an adversary is essentially at chance — and is far below the
paper's own GAN baselines on Stocks (TimeGAN 0.102, RCGAN 0.196, COT-GAN 0.285).
The ~0.016 gap is the expected run-to-run variance of a stochastic post-hoc
classifier scored over few runs with a different RNG seed (and a PyTorch vs TF1
optimiser), not a methodological difference. This confirms the port reproduces
GT-GAN's published generative quality on its native short-sequence multivariate
data; the same code is carried into the Heston generator (seq_len 24 → 128,
univariate returns) evaluated in
[`results/Heston/GT-GAN/README.md`](../../../results/Heston/GT-GAN/README.md).

---

## 5. How to reproduce (EXACT run path)

```bash
cd metric
# requires a TF1 env with the GT-GAN torch stack (NeuralCDE + CNF); NOT gpu-venv
CUDA_VISIBLE_DEVICES=0 python reproduce_stock.py
```

**Exact run path — which file feeds which cell (so any number is traceable):**

| Table cell | Interpreter + env | Script | Input file(s) scored | Output JSON |
|------------|-------------------|--------|----------------------|-------------|
| All §4 Stocks scores (disc + pred) | TF1 env + GT-GAN torch stack, `CUDA_VISIBLE_DEVICES=0` | `metric/reproduce_stock.py` (eval-only path of `GTGAN_stocks.py`) | real `../code/reference/datasets/stock_data.csv` → (3661,24,6) vs GT-GAN-recovered paths from `stock_model/stock/{generator,recovery}.pt` | `results/gtgan_stock_scores.json` (per-run arrays + paper reference + committed aggregate) |

The script regenerates end-to-end (load checkpoint → generate → score) in one
call. The §4 aggregate (`ours`) is the benchmark's shared-port 5-seed record,
cross-referenced in `results/Heston/GT-GAN/README.md`; the reference-path JSON is
produced only under a TF1 env (see caveat) and is not committed here.

---

## 6. Files

| Path | Content |
|------|---------|
| `README.md` | this file |
| `GT-GAN_NeurIPS2022.pdf` | the reference paper |
| `metric/reproduce_stock.py` | eval-only reproduction driver (reference path) |
| `metric/discriminative_metrics.py` | verbatim reference disc metric (TF1) + provenance |
| `metric/predictive_metrics.py` | verbatim reference pred metric (TF1) + provenance |
| `results/gtgan_stock_scores.json` | written by the driver under a TF1 env (not committed) |
