# CSDI — Paper Reimplementation (PhysioNet Healthcare + PM2.5 Air Quality)

Reproduction of the **CSDI** headline imputation results using the *official* code,
verbatim from the authors' repository.

- **Paper:** *CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series
  Imputation* — Yusuke Tashiro, Jiaming Song, Yang Song, Stefano Ermon, NeurIPS 2021.
- **Official code:** `github.com/ermongroup/CSDI` (mirrored here under `../code/reference/`).
- **This run:** the authors' own `exe_physio.py` / `exe_pm25.py` → train → impute → score with
  the paper's own `utils.calc_quantile_CRPS`. Target metric: **CRPS** (Table 2).

---

## ⚠️ Reproduction caveat (near-zero port)

The code is the authors' **verbatim**, with a **single non-behavioural** change:

- **`code/reference/diff_models.py` — lazy import.** The upstream top-level
  `from linear_attention_transformer import LinearAttentionTransformer` was moved *inside*
  `get_linear_trans()`. That package is an **optional** dependency exercised **only** by the
  `is_linear=True` forecasting variant. PhysioNet, PM2.5, and our Heston run all use
  `is_linear=False` (the pure-torch `get_torch_trans` path), so the import is **never reached**.
  Making it lazy lets the module load without installing an unused package; behaviour is
  **byte-identical** on every path we run. Rollback documented in `../code/README.md`.

No other file was modified. Hyperparameters, model, training loop, CRPS scorer: all upstream.

---

## 1. Paper metrics (as defined in the paper, Table 2)

CSDI's headline probabilistic-imputation metric:

| Metric | Definition | Direction |
|--------|-----------|-----------|
| **CRPS** | Continuous Ranked Probability Score, approximated by the authors over 19 quantile levels `0.05, 0.10, …, 0.95` from `nsample=100` imputations, normalised by the sum of ground-truth absolute values (`utils.calc_quantile_CRPS`). | lower = better |

Reported by the paper as **mean (standard error) over 5 trials**. RMSE / MAE (median imputation)
are also emitted by `utils.evaluate` and recorded below as secondary numbers.

**Targets (paper Table 2):**

| Setting | Paper CRPS |
|---------|:----------:|
| Healthcare (PhysioNet) — 10% missing | 0.238 (0.001) |
| Healthcare (PhysioNet) — 50% missing | 0.330 (0.002) |
| Healthcare (PhysioNet) — 90% missing | 0.522 (0.002) |
| Air quality (PM2.5) | 0.108 (0.001) |

---

## 2. Method hyperparameters (from the released `config/base.yaml`)

Taken **verbatim** from the released base config (used by both `exe_physio.py` and
`exe_pm25.py` — neither exe script overrides them):

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `diffusion.num_steps` | 50 | diffusion steps `T` |
| `diffusion.schedule` | `quad` | quadratic β schedule |
| `diffusion.beta_start / beta_end` | 0.0001 / 0.5 | β range |
| `diffusion.layers` | 4 | residual layers |
| `diffusion.channels` | 64 | residual channel width |
| `diffusion.nheads` | 8 | attention heads (2-D: time + feature) |
| `diffusion.diffusion_embedding_dim` | 128 | diffusion-step embedding |
| `diffusion.is_linear` | False | pure-torch Transformer (not linear-attention) |
| `model.timeemb` | 128 | time embedding |
| `model.featureemb` | 16 | feature embedding |
| `model.is_unconditional` | 0 | conditional imputation (paper headline) |
| `model.target_strategy` | `random` (physio) / `mix` (pm25) | self-supervised mask |
| `train.epochs` | 200 | training length |
| `train.batch_size` | 16 | batch |
| `train.lr` | 1.0e-3 | Adam lr (weight_decay 1e-6, MultiStepLR @ 0.75/0.9 ×0.1) |

`nsample = 100` at evaluation (paper's imputation-sample count for CRPS). PM2.5 uses the
`mix` target strategy (histmask), as in `exe_pm25.py`; PhysioNet uses `random`.

---

## 3. Dataset

Both are the paper's own datasets, loaded **verbatim** by the released dataset classes:

- **PhysioNet Challenge 2012 (healthcare)** — `dataset_physio.py`: 4 000 ICU stays,
  **35 clinical variables**, **48 hourly** timesteps. Missing ratio `{0.1, 0.5, 0.9}` is the
  fraction of *observed* values held out as imputation targets (`--testmissingratio`). Cache
  `data/physio_missing{r}_seed1.pk` built on first run from the raw `set-a` files.
- **PM2.5 air quality (Beijing)** — `dataset_pm25.py`: **36 monitoring stations**,
  `eval_length = 36` timesteps, real structured missingness (the authors' `pm25_ground` /
  `pm25_missing` split). Normaliser `data/pm25/pm25_meanstd.pk` (per-station mean/std) built
  from the authors' `create_normalizer_pm25`.

---

## 4. Results — CRPS vs paper Table 2

Single trial per setting (the benchmark reproduces the paper's *regime*, not its 5-trial
average — see the verdict note). Primary column is CRPS; RMSE/MAE are secondary (median).

| Dataset / setting | **Ours CRPS** | Paper CRPS (Table 2) | Ours RMSE | Ours MAE | Verdict |
|---|:---:|:---:|:---:|:---:|:---|
| Healthcare — 10% missing | **0.2344** | 0.238 (0.001) | 0.520 | 0.216 | matches ✓ (−1.6 σ) |
| Healthcare — 50% missing | **0.3364** | 0.330 (0.002) | 0.676 | 0.309 | matches ✓ (+1.4 σ) |
| Healthcare — 90% missing | **0.5256** | 0.522 (0.002) | 0.837 | 0.484 | matches ✓ (+0.8 σ) |
| Air quality (PM2.5) | **0.1064** | 0.108 (0.001) | 18.59 | 9.57 | matches ✓ (−0.7 σ) |

**No NaN in any run** (loss finite throughout all four trainings; verified by `grep -c nan` on
every log = 0).

> **Verdict — reproduced within single-trial noise.** The paper's parenthesised value is the
> **standard error over 5 trials**, so the per-trial standard deviation is `SE · √5` (e.g.
> 0.001 → ≈ 0.0022 for the 10% / PM2.5 rows, 0.002 → ≈ 0.0045 for the 50% / 90% rows). We ran
> **one** trial per setting, so the honest comparison is our single draw against the paper's
> 5-trial mean measured in **per-trial σ**: −1.6 σ, +1.4 σ, +0.8 σ, −0.7 σ. All four land inside
> ±1.6 per-trial standard deviations of the paper mean — i.e. exactly the spread you expect from
> one seed of a stochastic diffusion sampler. The configuration is verbatim (§2), so the residual
> gap is **seed/sampling variance, not a hyperparameter error**.

---

## 5. How to reproduce

All four runs use the authors' exe scripts unchanged (each hardcodes `config/base.yaml`).
Run from `../code/reference/`:

```bash
cd ../code/reference

# Healthcare (PhysioNet) — one run per missing ratio
CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 OMP_NUM_THREADS=8 \
  /home/tbasseras/gpu-venv/bin/python exe_physio.py --testmissingratio 0.1 --nsample 100
CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 OMP_NUM_THREADS=8 \
  /home/tbasseras/gpu-venv/bin/python exe_physio.py --testmissingratio 0.5 --nsample 100
CUDA_VISIBLE_DEVICES=3 taskset -c 8-15 OMP_NUM_THREADS=8 \
  /home/tbasseras/gpu-venv/bin/python exe_physio.py --testmissingratio 0.9 --nsample 100

# Air quality (PM2.5)
CUDA_VISIBLE_DEVICES=3 taskset -c 8-15 OMP_NUM_THREADS=8 \
  /home/tbasseras/gpu-venv/bin/python exe_pm25.py --nsample 100
```

**Exact run path — which file feeds which table cell (so any number is traceable):**

| Table cell | Interpreter + env | Script | Output folder | Scored file |
|------------|-------------------|--------|---------------|-------------|
| Healthcare 10% | `gpu-venv`, GPU 0 | `exe_physio.py --testmissingratio 0.1` | `save/physio_fold0_<ts>/` | `result_nsample100.pk` = `[rmse, mae, CRPS]` |
| Healthcare 50% | `gpu-venv`, GPU 0 | `exe_physio.py --testmissingratio 0.5` | `save/physio_fold0_20260720_024154/` | `result_nsample100.pk` |
| Healthcare 90% | `gpu-venv`, GPU 3 | `exe_physio.py --testmissingratio 0.9` | `save/physio_fold0_20260720_034106/` | `result_nsample100.pk` |
| Air quality | `gpu-venv`, GPU 3 | `exe_pm25.py` | `save/pm25_validationindex0_20260720_034106/` | `result_nsample100.pk` |

Read any cell back with:
```bash
/home/tbasseras/gpu-venv/bin/python -c \
 "import pickle; r=pickle.load(open('save/<folder>/result_nsample100.pk','rb')); print(r)"
```

---

## 6. Files

| Path | Content |
|------|---------|
| `../code/reference/exe_physio.py` | PhysioNet train → impute → CRPS driver (authors', unchanged) |
| `../code/reference/exe_pm25.py` | PM2.5 train → impute → CRPS driver (authors', unchanged) |
| `../code/reference/config/base.yaml` | verbatim hyperparameters (§2) |
| `../code/reference/utils.py` | `train`, `evaluate`, `calc_quantile_CRPS` (authors', unchanged) |
| `../code/reference/save/physio_fold0_20260720_024154/result_nsample100.pk` | Healthcare 50% `[rmse, mae, CRPS]` |
| `../code/reference/save/physio_fold0_20260720_034106/result_nsample100.pk` | Healthcare 90% `[rmse, mae, CRPS]` |
| `../code/reference/save/pm25_validationindex0_20260720_034106/result_nsample100.pk` | PM2.5 `[rmse, mae, CRPS]` |
| `CSDI_NeurIPS2021_2107.03502v2.pdf` | the paper |
