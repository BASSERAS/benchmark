# TimesFM — Paper Reimplementation (zero-shot ETT MAE)

Reproduction of **TimesFM**'s headline claim — strong **zero-shot** forecasting — on the **ETT
long-horizon benchmark**, reproducing the paper's **Table 5** ("MAE for ETT datasets for prediction
horizons 96 and 192").

- **Paper:** *A decoder-only foundation model for time-series forecasting* — Abhimanyu Das, Weihao Kong,
  Rajat Sen, Yichen Zhou (Google Research), **ICML 2024**.
  ([`TimesFM_2310.10688v4.pdf`](TimesFM_2310.10688v4.pdf), [arXiv:2310.10688](https://arxiv.org/abs/2310.10688))
- **Official code:** `github.com/google-research/timesfm` (mirrored under [`../code/reference/`](../code/reference/)).
- **This run:** [`run_ett_zeroshot.py`](run_ett_zeroshot.py) — a pure-numpy re-implementation of the
  official `timesfm/data_loader.py::TimeSeriesdata` windowing/normalisation, driving `timesfm.TimesFm`
  **zero-shot** (no fine-tuning) over **both** released checkpoints.

---

## 1. Paper metric

| Metric | Definition | Direction |
|--------|-----------|-----------|
| **MAE** | Mean Absolute Error of the point forecast (**median** = quantile head q0.5) vs the actuals, on the **last test window** of the original test split, on the **train-statistic z-normalised** series. | lower = better |

This is the exact protocol of the paper §6.1 / Appendix A.5.3: *"following llmtime, we compare all methods
on the last test window … given a context length of 512 … results are reported on standard normalized
dataset (using the statistics of the training portion)."* The point forecast is the median
(`quantile_forecast[:, :, 5]`), matching the official `run_eval.py::get_forecasts`.

---

## 2. Model / inference setup

Zero-shot inference only — the released checkpoint forecasts each series' held-out horizon directly from
its 512-step context, no training. Both released checkpoints are evaluated:

| Parameter | 1.0-200m | 2.0-500m |
|-----------|----------|----------|
| checkpoint | `google/timesfm-1.0-200m-pytorch` | `google/timesfm-2.0-500m-pytorch` |
| num_layers | 20 | 50 |
| model_dims | 1280 | 1280 |
| input / output patch | 32 / 128 | 32 / 128 |
| use_positional_embedding | True | False |
| context length | 512 | 512 |
| mode | **zero-shot** | **zero-shot** |

The `1.0-200m` checkpoint is the model reported in the paper (Table 6: 20 layers, model_dims 1280,
input-patch 32, output-patch 128, 16 heads). `2.0-500m` is a newer release **not** in the paper, evaluated
here as an additional data point per the benchmark request.

---

## 3. Datasets

Four ETT (Electricity Transformer Temperature) datasets from the Informer benchmark [ZZP+21], with the
official Informer train/val/test boundaries (`experiments/long_horizon_benchmarks/run_eval.py::DATA_DICT`),
two horizons each (96, 192) → **8 tasks**:

| Dataset | boundaries (train/val/test) | horizons |
|---------|-----------------------------|----------|
| ETTh1 | 8640 / 11520 / 14400 | 96, 192 |
| ETTh2 | 8640 / 11520 / 14400 | 96, 192 |
| ETTm1 | 34560 / 46080 / 57600 | 96, 192 |
| ETTm2 | 34560 / 46080 / 57600 | 96, 192 |

---

## 4. Results — ours (both checkpoints) vs paper Table 5

**Ours** = zero-shot MAE from
[`results/ett_zeroshot_timesfm-1.0-200m-pytorch.json`](results/ett_zeroshot_timesfm-1.0-200m-pytorch.json)
and [`results/ett_zeroshot_timesfm-2.0-500m-pytorch.json`](results/ett_zeroshot_timesfm-2.0-500m-pytorch.json).
**Paper** = the `TimesFM(ZS)` column of the paper's **Table 5** (arXiv:2310.10688v4), which is the 1.0-200m
model.

| Dataset (horizon) | **Ours — 1.0-200m** | **Ours — 2.0-500m** | **Paper `TimesFM(ZS)`** (1.0-200m) |
|-------------------|:-------------------:|:-------------------:|:----------------------------------:|
| ETTh1 (96)  | 0.4283 | 0.4578 | 0.45 |
| ETTh1 (192) | 0.4654 | 0.4967 | 0.53 |
| ETTh2 (96)  | 0.3206 | 0.3994 | 0.35 |
| ETTh2 (192) | 0.5296 | 0.5161 | 0.62 |
| ETTm1 (96)  | 0.1772 | 0.1786 | 0.19 |
| ETTm1 (192) | 0.3055 | 0.2091 | 0.26 |
| ETTm2 (96)  | 0.2103 | 0.2477 | 0.24 |
| ETTm2 (192) | 0.2900 | 0.2263 | 0.27 |
| **Avg**     | **0.3409** | **0.3415** | **0.36** |

**Reproduced.** Our 1.0-200m 8-task average MAE (**0.3409**) matches the paper's `TimesFM(ZS)` average
(**0.36**) closely, and the per-dataset values track the paper column across the board (e.g. ETTm1-96
0.1772 vs 0.19, ETTh2-96 0.3206 vs 0.35). The newer 2.0-500m checkpoint lands essentially the same on
average (0.3415), redistributing accuracy across horizons (better on ETTm1-192 / ETTm2-192, slightly worse
on ETTh1). This GATE confirms the port loads and runs **both** pretrained checkpoints faithfully — the
**same** `tfm.forecast` call is then carried into the Heston forecaster reference (zero-shot + fine-tuned),
evaluated in [`../../results/Heston/TimesFM/README.md`](../../results/Heston/TimesFM/README.md), using the
**1.0-200m** checkpoint.

---

## 5. How to reproduce (EXACT run path)

```bash
# Environment: timesfm-v1-venv (timesfm[torch]==1.3.0, torch cu130). Zero-shot — no training.
cd methods/TimesFM/paper_reimplementation
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-7 \
  /home/tbasseras/timesfm-v1-venv/bin/python run_ett_zeroshot.py \
    --model google/timesfm-1.0-200m-pytorch --context 512
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-7 \
  /home/tbasseras/timesfm-v1-venv/bin/python run_ett_zeroshot.py \
    --model google/timesfm-2.0-500m-pytorch --context 512
```

**Exact run path — which file feeds which cell (so any number is traceable):**

| Table cell | Interpreter + env | Script | Input scored | Output JSON |
|------------|-------------------|--------|--------------|-------------|
| §4 "Ours — 1.0-200m", all 8 tasks + Avg | `timesfm-v1-venv`, `CUDA_VISIBLE_DEVICES=0` | `run_ett_zeroshot.py --model …1.0-200m…` | ETT-small CSVs, Informer splits, forecast zero-shot | `results/ett_zeroshot_timesfm-1.0-200m-pytorch.json` |
| §4 "Ours — 2.0-500m", all 8 tasks + Avg | `timesfm-v1-venv`, `CUDA_VISIBLE_DEVICES=0` | `run_ett_zeroshot.py --model …2.0-500m…` | same splits, forecast zero-shot | `results/ett_zeroshot_timesfm-2.0-500m-pytorch.json` |
| §4 "Paper `TimesFM(ZS)`" | — | — | transcribed from arXiv:2310.10688v4 Table 5, `TimesFM(ZS)` column | — |

Each JSON key (`<dataset>_h<horizon>`, `avg`) is the sole source for its §4 cell.

---

## 6. Files

| Path | Content |
|------|---------|
| `README.md` | this file |
| `TimesFM_2310.10688v4.pdf` | the reference paper |
| `run_ett_zeroshot.py` | zero-shot ETT MAE reproduction (pure-numpy Informer windowing + `timesfm.TimesFm`) |
| `results/ett_zeroshot_timesfm-1.0-200m-pytorch.json` | our zero-shot MAE, 1.0-200m (per-task + avg) |
| `results/ett_zeroshot_timesfm-2.0-500m-pytorch.json` | our zero-shot MAE, 2.0-500m (per-task + avg) |
