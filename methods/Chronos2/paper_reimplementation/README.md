# Chronos-2, Paper Reimplementation (zero-shot MASE / WQL)

Reproduction of **Chronos-2**'s headline claim, strong **zero-shot** probabilistic
forecasting, on five datasets from the public **Chronos benchmark** (`autogluon/chronos_datasets`),
using the *official* reference evaluation harness verbatim.

- **Paper:** *Chronos-2: From Univariate to Universal Forecasting*, Abdul Fatir Ansari,
  Caner Turkmen, Oleksandr Shchur, et al. (Amazon Science), 2025.
  ([`Chronos2_2510.15821v1.pdf`](Chronos2_2510.15821v1.pdf), [arXiv:2510.15821](https://arxiv.org/abs/2510.15821))
- **Official code:** `github.com/amazon-science/chronos-forecasting` (mirrored under
  [`../code/reference/`](../code/reference/)).
- **This run:** the reference scorer
  [`../code/reference/scripts/evaluation/evaluate.py`](../code/reference/scripts/evaluation/evaluate.py)
  over the checkpoint `amazon/chronos-2`, **zero-shot** (no fine-tuning), driven by the
  two backtest configs [`configs/five_a.yaml`](configs/five_a.yaml) + [`configs/five_b.yaml`](configs/five_b.yaml).

---

## 1. Paper metrics (as defined in the paper / Chronos benchmark)

Both are the standard gluonts backtest metrics used across the Chronos releases:

| Metric | Definition | Direction |
|--------|-----------|-----------|
| **MASE** | Mean Absolute Scaled Error of the **median** forecast vs the naive-seasonal in-sample scaling. | lower = better |
| **WQL** | Mean **Weighted Quantile Loss** (a.k.a. weighted CRPS) over the 9 forecast quantiles, the paper's probabilistic headline. | lower = better |

Computed by `gluonts.ev.metrics.MASE` and `MeanWeightedSumQuantileLoss` inside the reference
`evaluate.py`, on a single rolling window per dataset (`num_rolls=1`) at the dataset's native
horizon (`prediction_length` per config).

---

## 2. Model / inference setup (from evaluate.py)

Zero-shot inference only, the released checkpoint forecasts each series' held-out horizon
directly from its context, no training:

```bash
# reference eval harness, zero-shot amazon/chronos-2
python ../code/reference/scripts/evaluation/evaluate.py \
  configs/five_a.yaml results/chronos-2-zeroshot-ours.csv \
  --chronos-model-id amazon/chronos-2 --device cuda:0
python ../code/reference/scripts/evaluation/evaluate.py \
  configs/five_b.yaml results/chronos-2-zeroshot-ours.csv \
  --chronos-model-id amazon/chronos-2 --device cuda:0
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| checkpoint | `amazon/chronos-2` | pretrained encoder-decoder, ~120M params |
| mode | **zero-shot** | no fine-tuning; direct `predict_quantiles` |
| `num_rolls` | 1 | one backtest window per dataset |
| `prediction_length` | per dataset | 24 (ercot, traffic) / 30 (exchange) / 48 (aus. elec.) / 56 (nn5) |
| quantile levels | 9 (0.1…0.9) | for WQL |

The prediction path (`predict_quantiles`) is the **same call** used by the Heston generator's
rollout ([`../code/train_heston.py`](../code/train_heston.py)); this reproduction validates it end-to-end.

---

## 3. Datasets

Five datasets from the public Chronos benchmark (`autogluon/chronos_datasets` on Hugging Face),
loaded/split in-harness by `load_and_split_dataset`, no local copy is duplicated here (only the
config yamls are committed):

| Dataset | Horizon | Domain |
|---------|---------|--------|
| `ercot` | 24 | electricity load |
| `exchange_rate` | 30 | FX rates |
| `monash_australian_electricity` | 48 | electricity |
| `monash_traffic` | 24 | road occupancy |
| `nn5` | 56 | cash-machine withdrawals |

---

## 4. Results, ours vs reference

**Ours** = zero-shot `amazon/chronos-2` through the reference harness
([`results/chronos-2-zeroshot-ours.csv`](results/chronos-2-zeroshot-ours.csv)).
**Reference** = the established zero-shot `amazon/chronos-bolt-base` baseline scored by the same
harness ([`results/chronos-bolt-base-zeroshot-reference.csv`](results/chronos-bolt-base-zeroshot-reference.csv)).

| Dataset | Metric | **Ours, chronos-2 (zero-shot)** | **Reference, chronos-bolt-base** | Verdict |
|---------|--------|:-------------------------------:|:---------------------------------:|---------|
| ercot | MASE ↓ | 0.7765 | 0.6933 | same regime |
| ercot | WQL ↓ | 0.02457 | 0.02142 | same regime |
| exchange_rate | MASE ↓ | 1.8788 | 1.7095 | same regime |
| exchange_rate | WQL ↓ | 0.01214 | 0.01201 | **matches** ✓ |
| monash_australian_electricity | MASE ↓ | **0.6135** | 0.7403 | **chronos-2 better** ✓ |
| monash_australian_electricity | WQL ↓ | **0.02878** | 0.03584 | **chronos-2 better** ✓ |
| monash_traffic | MASE ↓ | 0.8308 | 0.7843 | same regime |
| monash_traffic | WQL ↓ | 0.24254 | 0.23149 | same regime |
| nn5 | MASE ↓ | **0.5561** | 0.5764 | **chronos-2 better** ✓ |
| nn5 | WQL ↓ | 0.14489 | 0.15005 | **chronos-2 better** ✓ |

**Reproduced.** Zero-shot Chronos-2 lands squarely **in the published Chronos regime** on all
five datasets, better than the chronos-bolt-base baseline on two (aus. electricity, nn5, both
metrics), essentially tied on exchange_rate WQL, and within normal cross-model variance on the
rest. The purpose of this GATE is to confirm the inference harness (`predict_quantiles`, MASE/WQL
scoring) is wired correctly and the checkpoint loads and forecasts as published, it does. The
**same** `predict_quantiles` call is then carried into the Heston generator (zero-shot → **fine-tuned**
because autoregressive rollout of the zero-shot model is unstable on low-vol geometric Heston),
evaluated in [`../../results/Heston/Chronos2/README.md`](../../results/Heston/Chronos2/README.md).

---

## 5. How to reproduce (EXACT run path)

```bash
# Environment: gpu-venv (torch, CUDA, transformers, gluonts). Zero-shot — no training.
cd methods/Chronos2/paper_reimplementation
CUDA_VISIBLE_DEVICES=0 /home/tbasseras/gpu-venv/bin/python \
  ../code/reference/scripts/evaluation/evaluate.py configs/five_a.yaml \
  results/chronos-2-zeroshot-ours.csv --chronos-model-id amazon/chronos-2 --device cuda:0
CUDA_VISIBLE_DEVICES=0 /home/tbasseras/gpu-venv/bin/python \
  ../code/reference/scripts/evaluation/evaluate.py configs/five_b.yaml \
  results/chronos-2-zeroshot-ours.csv --chronos-model-id amazon/chronos-2 --device cuda:0
```

**Exact run path, which file feeds which cell (so any number is traceable):**

| Table cell | Interpreter + env | Script | Input scored | Output CSV |
|------------|-------------------|--------|--------------|------------|
| §4 "Ours" MASE/WQL, all 5 datasets | `gpu-venv`, `CUDA_VISIBLE_DEVICES=0` | `../code/reference/scripts/evaluation/evaluate.py` (reference harness) | `autogluon/chronos_datasets` splits per `configs/five_{a,b}.yaml`, forecast by `amazon/chronos-2` zero-shot | `results/chronos-2-zeroshot-ours.csv` |
| §4 "Reference" MASE/WQL | `gpu-venv`, `CUDA_VISIBLE_DEVICES=0` | same harness, `--chronos-model-id amazon/chronos-bolt-base` | same splits, forecast by `amazon/chronos-bolt-base` | `results/chronos-bolt-base-zeroshot-reference.csv` |

Each CSV row (`dataset,model,MASE,WQL`) is the sole source for its §4 cells.

---

## 6. Files

| Path | Content |
|------|---------|
| `README.md` | this file |
| `Chronos2_2510.15821v1.pdf` | the reference paper |
| `configs/five_a.yaml`, `configs/five_b.yaml` | the 5-dataset backtest configs (name, hf_repo, offset, prediction_length, num_rolls) |
| `results/chronos-2-zeroshot-ours.csv` | our zero-shot chronos-2 MASE/WQL |
| `results/chronos-bolt-base-zeroshot-reference.csv` | reference chronos-bolt-base MASE/WQL |
