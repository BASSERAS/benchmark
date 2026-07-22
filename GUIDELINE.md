# Benchmark — How to Add a New Method

This document is the single source of truth for integrating any new generative model
into the `BASSERAS/benchmark` repository.  Follow every section in order.
The level of detail required is identical to what was done for TimeGAN.

---

## 0. Before You Start — Questions to Answer

Before writing a single line of code, answer the following questions about the new method.
If you cannot answer any of them, ask Theo before proceeding.

| # | Question | Why it matters |
|---|----------|----------------|
| Q1 | Does the method produce **price paths** (levels) or **log-return paths**? | Determines whether min-max normalisation is applied to prices or returns, and how generated data is stored |
| Q2 | Is the method **univariate** (d=1, one asset) or **multivariate** (d>1)? | Heston is univariate (d=1). If the method has separate uni/multi variants, use the uni one. |
| Q3 | Does the method have **multiple algorithmic variants** (e.g. Markovian vs non-Markovian, full vs approximate)? | Pick the variant that best matches Heston's structure. Document why in `code/README.md`. |
| Q4 | What are the **canonical hyperparameters** from the paper (or the repo's demo notebook)? | We replicate the paper's setting; we do not tune on Heston. |
| Q5 | Does the reference code run on **GPU**? Is PyTorch, JAX, or TF used? | Determines which Python environment to use (`gpu-venv` for PyTorch/JAX) |
| Q6 | How long does **one training run** take on one A100? | Needed to plan the 5-seed parallelisation strategy (2 GPUs, 8 cores each) |
| Q7 | Does the model save **checkpoints / weights** natively, or do we need to add that? | We always save weights; if the reference code does not, add `torch.save` or equivalent |
| Q8 | Does the model expose a clean **generate(n_samples)** function, or does generation require re-running training code? | Determines how `train_seed.py` is structured |

---

## 1. Repository Folder Structure

Every new method follows this **exact** layout.  Replace `<Method>` with the method name
(PascalCase, e.g. `SBTS`, `SigWGAN`, `McKean`).

```
benchmark/
├── GUIDELINE.md                          ← this file
├── dataset/                              ← shared across all methods (do NOT duplicate)
│   ├── heston_S_8192x128.npy             real price paths (8192, 128) float64
│   └── heston_v_8192x128.npy             real variance paths (8192, 128) float64
│
└── methods/
    └── <Method>/
        ├── README.md                     ← main results file (see §8)
        ├── generated_paths/
        │   ├── seed_0/
        │   │   ├── generated_paths_8192x128.npy    shape (8192, 128), ORIGINAL price scale
        │   │   └── metadata.json                   see §4.3
        │   ├── seed_1/ … seed_4/
        ├── weights/
        │   ├── seed_0_model.pt           full PyTorch state_dict (or equivalent)
        │   ├── seed_0_config.json        hyperparameters + dataset info
        │   └── … (×5 seeds)
        ├── losses/
        │   ├── seed_0_losses.csv         columns depend on method, see §4.4
        │   └── loss_convergence.png      convergence plot (5 seeds overlaid)
        └── code/
            ├── <method>_torch.py         PyTorch model implementation (or wrapper)
            ├── train.py                  orchestrator — 5 seeds on 2 GPUs in pairs
            ├── train_seed.py             single-seed worker
            ├── reference/                verbatim copy of upstream repo
            │   └── (original files)
            └── README.md                 paper citation, GitHub link, list of fixes
```

Additionally, results land in a **separate** results tree:

```
benchmark/
└── results/
    └── Heston/
        └── <Method>/
            ├── README.md                 summary table + per-seed breakdown (see §9)
            ├── seed_0_metrics.json       all A1–A34 + B_ values for seed 0
            ├── … seed_4_metrics.json
            ├── metrics_summary.json      mean ± std across 5 seeds
            ├── plots/
            │   ├── heston_diagnostics.png   8-panel stylised facts figure (see §7.1)
            │   ├── disc_classifier_loss.png
            │   └── pred_score_loss.png
            └── path_shadowing/           PS-MC evaluation (see §6)
                ├── README.md
                ├── seed_0_results.json
                ├── … seed_4_results.json
                ├── summary.json
                └── plots/
                    ├── ps_mc_example.png
                    └── crps_per_step.png
```

---

## 2. Dataset

**The dataset is fixed and shared.** Do NOT regenerate it.

```
dataset/heston_S_8192x128.npy   — shape (8192, 128), dtype float64
                                   price paths S_t, starting near 100
dataset/heston_v_8192x128.npy   — shape (8192, 128), dtype float64
                                   variance paths v_t (Heston latent process)
```

Parameters used to generate it:
`μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=-0.7, S₀=100, v₀=0.04, dt=1/250, N=8192, T=128, seed=0`

Generation script: `dataset/generate_heston.py` (do not modify).

---

## 3. Reference Code

### 3.0 Reimplement the paper FIRST (before anything else)

> **This is the first thing to do — before generating the 5 Heston seeds, before metrics, before any
> README.** Reproduce the paper's *own* headline results on the paper's *own* dataset, so we prove our
> re-implementation is faithful before we ever point it at Heston. Only once the paper numbers match
> (within a stated tolerance) do we proceed to §4 (5-seed Heston training).

Deliverable lives in `methods/<Method>/paper_reimplementation/`:

```
methods/<Method>/paper_reimplementation/
├── README.md          # paper-vs-ours table + tolerance verdict; if repro FAILS, explain and STOP
├── dataset/           # the paper's dataset (or its exact generator), not Heston
├── metric/            # the metric code used to score on the paper's dataset
└── (generated data + trained weights on the paper dataset)
```

**Procedure:**
1. Obtain the paper's dataset (download or regenerate with the paper's exact parameters) → `dataset/`.
2. Train / run the method on that dataset with the paper's hyperparameters.
3. Score with the paper's own metric(s) → `metric/`.
4. Fill `README.md` with a **paper vs our results** table (paper's reported number | our reproduced
   number | absolute/relative gap | within tolerance? Y/N).
5. **If reproduction fails** (numbers off beyond a reasonable tolerance and you cannot explain why):
   write the failure analysis in `README.md` and **STOP** — do not proceed to the 5-seed Heston run.

**Exact code cells to execute** (same scripts used for the Heston pipeline, retargeted at the paper
dataset via `--dataset`):
- **Metrics** — `python metrics/compute_all.py --method <Method> --dataset <PaperDataset>`
  (writes `seed_*_metrics.json` + `metrics_summary.csv`; B curves via `python metrics/recompute_curve_b.py`).
- **The 8 stylised-facts curve plots** — `python metrics/plot_diagnostics.py --method <Method>
  --dataset <PaperDataset> --seed 0` → 4×2 panel PNG (sample paths ×2, return distribution, QQ,
  ACF |r|, ACF r², rolling-vol histogram, tail survival). These 8 curves are the primary visual proof
  the reimplementation reproduces the paper's stylised facts.

### 3.1 Copy the upstream repo

```bash
# Inside methods/<Method>/code/
git clone https://github.com/<author>/<repo>.git reference/
# OR if it is a package: pip install, then copy relevant .py files
```

### 3.2 Document fixes in `code/README.md`

The code README must contain:
- Full paper citation (title, authors, venue, year, arXiv ID)
- GitHub link
- A numbered list of **every change made** to the reference code
  (e.g. "Fix 1: hidden_dim = int(1/2) = 0 crash for d=1 data; changed to max(8, d*8)")
- Which variant was chosen and why (answers Q2 and Q3 from §0)

---

## 4. Training — 5 Seeds

### 4.1 Hardware constraints (HARD RULES, always enforced)

- **Max 2 GPUs** (use GPU 0 and GPU 3 on the A100 cluster)
- **Max 8 cores per GPU job** (`taskset -c 0-7` for GPU 0, `taskset -c 24-31` for GPU 3)
- **Always check** `nvidia-smi` and `htop` before launching — machine is shared
- Set `OMP_NUM_THREADS=8` to prevent numpy/torch spawning 256 threads
- Never set `num_workers > 4` per DataLoader

Always check first:
```bash
nvidia-smi        # confirm GPU 0 and GPU 3 are free
htop              # confirm RAM < 50% used
```

### 4.2 Parallelisation strategy (copy this pattern exactly)

```bash
# Batch 1: seeds 0 and 1 in parallel
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-7   \
  python train_seed.py --seed 0 --gpu 0 &
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 taskset -c 24-31 \
  python train_seed.py --seed 1 --gpu 3 &
wait

# Batch 2: seeds 2 and 3
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-7   \
  python train_seed.py --seed 2 --gpu 0 &
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 taskset -c 24-31 \
  python train_seed.py --seed 3 --gpu 3 &
wait

# Batch 3: seed 4 alone
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-7   \
  python train_seed.py --seed 4 --gpu 0
```

`train.py` is the orchestrator that automates the above.
`train_seed.py` is the per-seed worker (one GPU, one seed).

### 4.3 What each seed run must save

**Generated paths** — `generated_paths/seed_{i}/generated_paths_8192x128.npy`
- Shape: `(8192, 128)`, dtype `float64`
- Scale: **original price scale** (denormalise before saving — no [0,1] outputs)
- Clip to minimum 1e-6 if the method can produce zero or negative values

**Metadata** — `generated_paths/seed_{i}/metadata.json`
```json
{
  "method": "<Method>",
  "seed": 0,
  "shape": [8192, 128],
  "min_val": 12.34,
  "max_val": 345.67,
  "gen_time_sec": 42.1,
  "train_time_sec": 1823.4,
  "gpu": "A100-SXM4-80GB",
  "date": "YYYY-MM-DD"
}
```

**Model weights** — `weights/seed_{i}_model.pt`
```python
torch.save(model.state_dict(), f"weights/seed_{i}_model.pt")
```
If the method is not PyTorch, save equivalent: `.npz` for numpy, `.pkl` for sklearn, etc.

**Config** — `weights/seed_{i}_config.json`
```json
{
  "method": "<Method>",
  "seed": 0,
  "hidden_dim": 24,
  "num_layers": 3,
  "batch_size": 128,
  "n_steps": 20000,
  "n_train": 8192,
  "seq_len": 128,
  "dataset": "Heston",
  "paper_hyperparams": true
}
```
Fill in fields relevant to the specific method.
`"paper_hyperparams": true` means you used the paper's recommended settings without tuning.

### 4.4 Loss logging

Every method must produce a per-seed loss CSV at `losses/seed_{i}_losses.csv`.

Minimum required columns:
```
step, phase, loss_total
```

Add method-specific columns as needed (e.g. TimeGAN adds `e_loss, s_loss, g_loss, d_loss`).
Log every **100 steps** (or every epoch if step-based logging is not applicable).

**Loss convergence plot** — `losses/loss_convergence.png`
- All 5 seeds overlaid, one subplot per loss component
- Size: 1600×900 px, 150 dpi
- X-axis: step number (not wall time)
- Y-axis: loss value (not log scale unless values span many orders of magnitude)
- Legend: Seed 0 … Seed 4

---

## 5. The 34 A-Metrics (A1–A34) + 18 B Curve Metrics

**All metrics are pre-implemented** across **three files**:
- `metrics/metrics.py` — the A1–A34 NumPy/SciPy metrics **and** the B stylized-fact curve metrics (`compute_curve_metrics`, `aggregate_curve_metrics`).
- `metrics/discriminative_score.py` — A18 (post-hoc GRU + MLP classifiers, PyTorch).
- `metrics/predictive_score.py` — A19 (GRU + MLP predictors, TSTR, PyTorch).

Do NOT reimplement them.

> **Naming note (read before adding a method):** "B" refers **exclusively** to the 6
> stylized-fact CURVES (log-return histogram, QQ, ACF |r|, ACF r², rolling-vol
> histogram, tail survival). There is **no scalar B metric** — the old scalar shape
> metrics were long ago absorbed into A25–A34. Everything lives in one `metrics.py`;
> there is no `metrics_np.py` or `stylized_metrics.py`.

Run everything with:
```bash
cd metrics
/home/tbasseras/gpu-venv/bin/python compute_all.py --method <Method> --dataset Heston --seeds 5
```

### 5.0 Data split — "test set everywhere"

The 8192×128 Heston dataset is drawn under **three disjoint seeds**, and each seed has exactly **one fixed
role** for the whole benchmark:

| Role | Seed | Used as | Scored? |
|------|------|---------|---------|
| **train** | seed 0 | the paths the generators were trained on | **never scored** |
| **test** | seed 1 | the real reference for **A1–A17, A20–A34, all 6 B curves, all diagnostic plots, and PS-MC query prefixes** | yes — this is *the* real set |
| **disc** | seed 2 | the "real" class for the two **learned** scorers only — **A18** (discriminative) and **A19** (predictive-TSTR) | judge-only |

Rationale: the adversary / forecaster (A18/A19) must never be trained on the same real paths it is later
asked to tell apart from / forecast against, so it gets its own **disc** seed (seed 2) while everything
distributional is measured against the held-out **test** seed (seed 1). The five per-method result seeds are
the **five generated draws** (a method's own seeds 0–4), each scored against this one fixed test set.

**Generators are NOT retrained here.** Because every method's paths are already saved to disk, a metrics run
only (re)fits the small A18/A19 judge/forecaster networks per seed; all A1–A17/A20–A34/B/plot/PS-MC numbers
are pure re-scores of the saved paths against the test set. The perfect-recovery floor (§5.4) is a fresh
**independent** Heston draw (seeds `1000+i`) scored against the same test set.

### 5.1 Input format expected by `compute_all.py`

| Variable | Path | Shape | Scale |
|----------|------|-------|-------|
| Real prices | `dataset/heston_S_8192x128.npy` | (8192, 128) | original (~100) |
| Real variance | `dataset/heston_v_8192x128.npy` | (8192, 128) | original (~0.04) |
| Fake prices | `methods/<Method>/generated_paths/seed_{i}/generated_paths_8192x128.npy` | (8192, 128) | original |

`compute_all.py` infers the fake path from `--method` and `--dataset` automatically.

### 5.2 Full metric list

Numbered in **category display order** (Fat Tail → Distribution → Adversarial → Predictive →
Temporal → Vol → Heston Spec). Formulas are unchanged from the previous numbering — only the
IDs were reordered to match the display order.

| ID | Name | Category | Direction | Perfect |
|----|------|----------|-----------|---------|
| A1 | Kurtosis Error | Fat Tail | ↓ | 0 |
| A2 | \|r\| q95 Error | Fat Tail | ↓ | 0 |
| A3 | \|r\| q99 Error | Fat Tail | ↓ | 0 |
| A4 | Tail QQ Error | Fat Tail | ↓ | 0 |
| A5 | Hill Tail Index Error | Fat Tail | ↓ | 0 |
| A6 | Path MMD² | Distribution | ↓ | 0 |
| A7 | Terminal MMD² | Distribution | ↓ | 0 |
| A8 | Increment MMD² | Distribution | ↓ | 0 |
| A9 | Volatility MMD | Distribution | ↓ | 0 |
| A10 | Terminal SWD | Distribution | ↓ | 0 |
| A11 | Path SWD | Distribution | ↓ | 0 |
| A12 | RV Law Loss (W₁ on annualized realized variance) | Distribution | ↓ | 0 |
| A13 | Mean Path RMSE | Distribution | ↓ | 0 |
| A14 | KS on Log-returns | Distribution | ↓ | 0 |
| A15 | Skewness Error | Distribution | ↓ | 0 |
| A16 | QQ RMSE (300-pt) | Distribution | ↓ | 0 |
| A17 | Terminal Price KS | Distribution | ↓ | 0 |
| A18 GRU | Discriminative Score GRU | Adversarial | ↓ | 0 |
| A18 MLP | Discriminative Score MLP | Adversarial | ↓ | 0 |
| A19 GRU | Predictive Score GRU (TSTR) | Predictive | ↓ | baseline |
| A19 MLP | Predictive Score MLP (TSTR) | Predictive | ↓ | baseline |
| A20 | Covariance Error | Temporal | ↓ | 0 |
| A21 | ACF Error (abs returns) | Temporal | ↓ | 0 |
| A22 | ACF Error (sq returns) | Temporal | ↓ | 0 |
| A23 | ACF \|r\| Lag-1 Error | Temporal | ↓ | 0 |
| A24 | ACF r² Lag-1 Error | Temporal | ↓ | 0 |
| A25 | Mean RMSE | Vol | ↓ | 0 |
| A26 | Return Std Error | Vol | ↓ | 0 |
| A27 | Log-Return Std Error | Vol | ↓ | 0 |
| A28 | Kurtosis Ratio (target/model) | Vol | — | 1.0 |
| A29 | Sigma Mean Error | Vol | ↓ | 0 |
| A30 | Cross-Sect. Vol Path RMSE | Vol | ↓ | 0 |
| A31 | Rolling Vol KS (window=5) | Vol | ↓ | 0 |
| A32 | Vol-of-Vol Error | Vol | ↓ | 0 |
| A33 | Teacher-Sigma Correlation | Heston Spec | ↑ | 1 |
| A34 | Teacher-Sigma RMSE | Heston Spec | ↓ | 0 |

**A1**: κ_real vs κ_gen — excess kurtosis (Fisher, bias-corrected) of pooled log-returns.
**A2–A3**: |Q_0.95(|r_real|) − Q_0.95(|r_gen|)| and same at 0.99. Tail quantile reproduction.
**A4**: QQ error restricted to the extreme percentiles (top/bottom 5%).
**A5**: |Hill tail index_real − Hill tail index_gen|. Hill estimator on top 10% of S_T.
**A12 (RV Law Loss):** W₁(RV_real, RV_gen) where RV_i = Σ_t r²_{i,t} / dt (annualized realized
variance per path). Ref: Barndorff-Nielsen & Shephard (2002).
**A13**: RMSE between real and generated cross-sectional mean trajectories (matched by time).
**A14**: Kolmogorov-Smirnov statistic on pooled log-returns.
**A15**: |skew_real − skew_gen|. Heston true skew ≈ −0.45.
**A16**: QQ RMSE over 300 uniform quantile levels. **A17**: KS on terminal prices S_T.
**A18**: Discriminative classifier. Score = |accuracy − 0.5|; 0 = indistinguishable.
**A19**: TSTR MAE; cluster near 0.056–0.059 (irreducible noise floor for Heston).
**A20**: Frobenius norm of terminal covariance difference (|Var| for d=1).
**A21–A22**: ACF on log-returns |r| / r² at lags L = {1, 2, 5, 10}.
**A23–A24**: Single-lag (lag=1) version of A21/A22. Heston true values ≈ +0.052 / +0.050.
**A25**: |E[X_T real] − E[X_T gen]| terminal mean bias. **A26**: return std error (uses ΔS_t).
**A27**: |σ(log-ret real) − σ(log-ret gen)| — log-return std error (distinct from A26).
**A28**: κ_real / κ_gen — excess kurtosis ratio. Perfect = 1.0. Negative = gen lighter-tailed.
**A29**: |mean_paths(σ_i^real) − mean_paths(σ_i^gen)| — annualized per-path vol averaged.
**A30**: RMSE between real and generated cross-sectional vol trajectories.
**A31**: KS statistic on rolling-5 volatility histograms. **A32**: |vol-of-vol_real − vol-of-vol_gen|.
**A33**: Pearson(σ̂_gen, √v_true) — Heston teacher-sigma correlation (higher is better).
**A34**: RMSE(σ̂_gen, √v_true) — absolute scale accuracy of reproduced vol process.

**B — Curve-shape metrics (6 diagnostic plots):**

Each plot yields a **curve** L (a list of values). We build three lists — the curve L, its first finite
difference L′ (der), and its second finite difference L″ (sec\_der) — and score each list under **three
measures**:
- **MSE**: dᵢ = mean((L\_gen − L\_real)²) per list. Decides the winner.
- **% err**: dᵢ = mean(|L\_gen − L\_real| / (|L\_real| + 1e-6)) × 100 — a function-level MAPE with a **fixed
  1e-6 floor** and exactly **one division** (the mean already divides by the number of curve points).
- **NRMSE**: dᵢ = sqrt(mean((L\_gen − L\_real)²)) / (max|L\_real| − min|L\_real| + 1e-12) × 100 per list.

The three measures are combined into **one number per plot** DIFFERENTLY (combined std = sample std across the
5 seeds in every case). **MSE** is **mean-of-3** — combined_per_seed = mean(funct, der, sec\_der) — because the
absolute squared error never explodes; this is the number that decides the winner. **% err** and **NRMSE** are
**funct-only for every plot** — combined_per_seed = the funct sub-metric alone — because the 1st and 2nd finite
differences of every curve have near-zero true values, so their relative errors blow up into meaningless
10⁴-% figures; only the curve itself (funct) carries a meaningful relative error. The raw JSON still stores 9
keys per plot (funct/der/sec\_der × {mse, pct\_err, nrmse}), but the der/sec\_der %err and NRMSE are dropped in
aggregation; the READMEs display the three **combined** measures as three sublines per plot (MSE / % err /
NRMSE). Aggregation lives in `metrics/metrics.py` (`aggregate_curve_metrics`); recompute with
`metrics/recompute_curve_b.py`.

| Plot | JSON key prefix | Curve description |
|------|----------------|-------------------|
| Log-return histogram | `B_log_ret_hist_*` | Density of log-returns over shared bins |
| QQ plot | `B_qq_plot_*` | Quantile function at 100 uniform percentile levels |
| ACF \|r\| (lags 1–20) | `B_acf_abs_r_*` | Mean per-path ACF of \|r\| at each lag |
| ACF r² (lags 1–20) | `B_acf_sq_r_*` | Mean per-path ACF of r² at each lag |
| Rolling vol hist. | `B_roll_vol_hist_*` | Density of rolling-5 vol over shared bins |
| Tail survival | `B_tail_surv_*` | P(\|r\|>x) at thresholds of real \|r\| |

All B metrics ↓ lower is better. **The perfect floor is non-zero and finite** — it is an independent Heston
draw (fresh seeds `1000+i`, identical parameters) scored against the test set (§5.4), so every curve carries
genuine finite-sample noise; no B measure is ever exactly 0. The **% err** row is read against this non-zero
floor and can still be large (triple-digit %) wherever the real curve passes through near-zero values (empty
histogram bins, tail-survival ≈ 0, near-zero ACF lags) — expected, a property of the curve, not a bug. Winner
between methods is decided by the **MSE** row.

Total A-metrics: 36 numbers (A1–A34, with **A18 Discriminative** and **A19 Predictive** each reported ×2
for their GRU + MLP variants). Total B: 6 plots × 3 sub-metrics (funct / der / sec_der) × 3 measures.
Winner is by MSE; the three combined measures (MSE, % err, NRMSE) are what the READMEs display per plot.

### 5.3 Output files

**`compute_all.py`** produces:

| File | Contents |
|------|---------|
| `results/Heston/<Method>/seed_{i}_metrics.json` | All 55 metric values for seed i (A1-A34 + 18 B_ keys) |
| `results/Heston/<Method>/metrics_summary.json` | Mean ± std across 5 seeds |
| `results/Heston/<Method>/metrics_summary.csv` | Same, CSV format |
| `results/Heston/<Method>/plots/disc_classifier_loss.png` | A18 BCE training loss, GRU + MLP, 5 seeds |
| `results/Heston/<Method>/plots/pred_score_loss.png` | A19 MAE training loss, GRU + MLP, 5 seeds |
| `results/Heston/<Method>/plots/seed_{i}_pca.png` | PCA of 500 real vs fake paths (per seed) |
| `results/Heston/<Method>/plots/seed_{i}_tsne.png` | t-SNE of 200 real vs fake paths (per seed) |

**`metrics/plot_diagnostics.py`** produces (run separately after compute_all):
```bash
python metrics/plot_diagnostics.py --method <Method> --dataset Heston --seed 0
```

| File | Contents |
|------|---------|
| `results/Heston/<Method>/plots/heston_diagnostics.png` | 8-panel stylised facts (50 sample paths, see §7.1) |

### 5.4 Perfect-Recovery Floor (independent-draw, non-zero)

The perfect-recovery floor answers: *what would each metric read if the generator were a perfect Heston
simulator?* It is computed by drawing a **fresh, independent Heston sample** — new seeds `1000+i`
(`IND_SEED_BASE = 1000`, `i = 0…4`), identical model parameters — and scoring that draw against the **test
set** (seed 1) exactly as any generator is scored. The two datasets are statistically identical in
distribution but are **different finite samples**, so every metric reads a genuine **non-zero finite-sample
noise floor**. This is *not* a row-shuffle and *not* a permutation of the real data; nothing collapses to 0.

The floor value shown in each method's `Perfect floor` column is the **mean across the 5 independent-draw
seeds** (`perfect_floor_a()` / `perfect_floor_curve()` in `render_tables.py`). Because it depends only on the
test set and the protocol — never on any generator — it is **identical across every method's tables**.

Run it once per dataset with the **reproducible** script (fixed independent seeds, 5-seed average):

```bash
python metrics/compute_perfect_recovery.py            # add --no-pytorch to skip A18/A19 (no GPU)
# Outputs (source of truth for all floor columns):
#   methods/perfect_recovery/results/metrics_summary.csv    ← A1–A34 floors (mean ± std, 5 independent draws)
#   methods/perfect_recovery/results/curve_b_aggregate.json ← B floors (non-zero, MSE / %err / NRMSE per plot)
#   methods/perfect_recovery/results/seed_{0..4}_metrics.json
```

**Display rule (current standard):**
The floor is shown as the **last column** (`Perfect` / `Perfect floor`) of every metric table across the repo:
`methods/<Method>/README.md`, `results/Heston/<Method>/README.md`, `results/README.md`, and root `README.md`.
`methods/perfect_recovery/README.md` is the one standalone place that documents the floor in full (it *is* the
floor, so it carries no floor column of its own).

Because floors are dataset-derived (never method-derived), the `Perfect` column values must be **identical**
across every method's tables. Never hand-type them — regenerate from
`methods/perfect_recovery/results/*` via `render_tables.py` so they match byte-for-byte. PS-MC rows (which
have no perfect analogue) show `—`.

---

## 6. Path Shadowing MC (PS-MC)

PS-MC is **already implemented** in `methods/TimeGAN/path_shadowing/`.
For a new method, copy the entire folder and point it at the new generated paths.

### 6.1 Copy the runner

```bash
cp -r methods/TimeGAN/path_shadowing/ methods/<Method>/path_shadowing/
```

Then edit **only** these two lines in `run_eval.py`:
```python
METHOD   = "<Method>"          # was "TimeGAN"
FAKE_DIR = "methods/<Method>/generated_paths"
```

Everything else — embedding, KNN, CRPS computation, figure code — is **unchanged**.
The PS-MC method is identical across all generative models; only the path pool differs.

### 6.2 Run it

```bash
cd methods/<Method>/path_shadowing
/home/tbasseras/gpu-venv/bin/python run_eval.py
```

Results save automatically to `results/Heston/<Method>/path_shadowing/`.

### 6.3 What PS-MC does (for documentation)

1. **Embedding**: prefix (steps 0–63) → 65D murex-style feature vector: 63 step-by-step log-returns + 1 terminal cumulative return + 1 realized volatility (= sqrt(mean(r²))), z-scored per dimension using the generated pool distribution
2. **KNN retrieval**: L2 distance in the 65D z-scored space → 77 nearest generated paths (sklearn NearestNeighbors, `metric="euclidean"`)
3. **Price anchoring**: each retrieved fake future is scaled by `S_real(t=63) / S_fake(t=63)`
4. **Weighting**: Uniform (flat 1/77) and Gaussian (per-query η = η̃·‖z(x̃)‖, η̃ = median_dist/median_norm, calibrated from data)
5. **Evaluation**: CRPS, MAE, RMSE at H=32 (steps 64–95) and H=64 (steps 64–127)
6. **Baseline**: naive random walk (repeat last prefix price) — CRPS_h32=3.732, CRPS_h64=5.301

The KNN is a plain nearest-neighbour scan (not a combinatorial search). For each of the 8 192 real query paths, all 8 192 fake path embeddings are scanned and the 77 with smallest L2 distance are returned. Cost: O(8192 × 65) per query.

---

## 7. Figures

### 7.1 Heston Diagnostics — 8-panel figure

Compares real Heston (seed 0) vs generated paths (seed 0) on 8 stylised facts.

**Use the shared reusable script — do NOT inline plot code in method files:**
```bash
python metrics/plot_diagnostics.py --method <Method> --dataset Heston --seed 0
# Output: results/Heston/<Method>/plots/heston_diagnostics.png
```

Layout: **4 rows × 2 columns**, 14×18 inches, 120 dpi.
These constants are **benchmark standards** — do NOT change per method:

| Constant | Value | Meaning |
|----------|-------|---------|
| `N_SHOW` | **50** | paths shown in panels [0,0] and [0,1] (alpha=0.25, lw=0.6) |
| `N_STAT` | 5000 | paths used for statistical panels 2–7 |
| Real colour | `#2196F3` | blue |
| Gen colour | `#FF5722` | orange-red |

| Panel | Title | Content |
|-------|-------|---------|
| Row 0, Col 0 | Real Sample Paths | **50** random real paths + thick mean line, y=price |
| Row 0, Col 1 | Generated Sample Paths | **50** random generated paths + thick mean line, y=price |
| Row 1, Col 0 | Return Distribution | Overlaid histogram (80 bins) of log-returns, real vs gen |
| Row 1, Col 1 | QQ Plot | Gen vs real log-return quantiles (300 points), y=x reference |
| Row 2, Col 0 | ACF of \|r\| | Mean ACF of absolute returns, lags 1–20 |
| Row 2, Col 1 | ACF of r² | Mean ACF of squared returns, lags 1–20 |
| Row 3, Col 0 | Rolling Vol Histogram | Distribution of 5-day rolling σ of log-returns |
| Row 3, Col 1 | Tail Survival (log-log) | P(S_T > threshold) on terminal price S_T |

Always include a legend on each panel.

### 7.2 Discriminative Classifier Loss (A18)

Saved to `results/Heston/<Method>/plots/disc_classifier_loss.png`.
- Two subplots side by side: GRU discriminator / MLP discriminator
- X-axis: training step 0–2000, logged every 50
- Y-axis: BCE loss
- All 5 seeds overlaid
- Horizontal dashed reference line at ln(2) ≈ 0.693 (random-chance level)

> **Fresh-retrain guarantee (verified 2026-07-19):** the A18 classifier is **retrained from scratch for
> every method and every seed**. `compute_discriminative_score` (in `metrics/discriminative_score.py`)
> instantiates a new `GRUDiscriminator` / `MLPDiscriminator` and a new Adam optimiser on each call, and
> `compute_all.py` calls it once inside the per-seed loop (never caches weights across seeds or methods).
> The same holds for A19 (`compute_predictive_score`). This means the discriminative/predictive scores are
> never contaminated by a previously-trained classifier — each score reflects a fresh 2000-step (A18) /
> 5000-step (A19) fit on that seed's real-vs-fake data.

### 7.3 Predictive Score Loss (A19)

Saved to `results/Heston/<Method>/plots/pred_score_loss.png`.
- Two subplots: GRU predictor / MLP predictor
- X-axis: training step 0–5000, logged every 100
- Y-axis: MAE loss on synthetic training data
- All 5 seeds overlaid

### 7.4 Training Loss Convergence

Saved to `methods/<Method>/losses/loss_convergence.png`, produced by `train.py`.
- One subplot per loss component (minimum: one for total loss)
- All 5 seeds overlaid, different colour per seed
- X-axis: step number, Y-axis: loss value
- Size: 1600×900 px, 150 dpi

### 7.5 PS-MC Figures

Produced by `run_eval.py`, saved to `results/Heston/<Method>/path_shadowing/plots/`.

**`ps_mc_example.png`** — Fan-out for 4 example real paths:
- Blue solid = real prefix (steps 0–63)
- Blue dashed = real future (steps 64–127)
- Red fan = all 77 retrieved generated futures (price-anchored)
- Bold red = ensemble mean forecast
- Size: 16×10 inches, 150 dpi

**`crps_per_step.png`** — CRPS as a function of forecast step:
- X-axis: step 64–127, Y-axis: mean CRPS at that step
- 5 seeds: mean ± std shaded band
- Horizontal dashed lines: RW baseline at H=32 (3.732) and H=64 (5.301)
- Size: 10×5 inches, 150 dpi

---

## 8. Method README (`methods/<Method>/README.md`)

This is the primary document seen on GitHub when someone opens the method folder.
Copy `methods/TimeGAN/README.md` as a template and fill in every section.
Do NOT leave any placeholder unreplaced.

### Required sections (in this order)

#### Section 1 — Header

```markdown
# <Method> on Heston

<One-sentence description of the method and its mathematical core.>

See [`code/README.md`](code/README.md) for source, paper citation, and list of fixes.

---
```

#### Section 2 — Metrics A1–A34 + B

Rows MUST be grouped by category in this exact order (a `| | **— <Category> —** | | ... |` separator
row before each group, no exceptions): **Fat Tail → Distribution → Adversarial → Predictive → Temporal
→ Vol → Heston Spec**. Do not sort by ID number. See §15.1 Section 2 for the full verbatim table
skeleton with every category populated.

Rules:
- 4 significant figures for values < 1; fewer for larger magnitudes (match the style produced by
  `render_tables.py`'s `fmt()` — plain numbers in the Perfect floor column, no bold, no "baseline" text)
- Include footnotes for A18, A19, and A20–A34 (any non-obvious ID). See §15.1 Section 2 for the exact verbatim footnote block.

#### Section 3 — B Curve-Shape Metrics

**Three** sublines per plot (MSE row + % err row + NRMSE row). **MSE** combines funct/der/sec\_der by
mean-of-3; **% err** and **NRMSE** are **funct-only** (the funct sub-metric alone). Last column = Perfect
floor (non-zero, from the independent draw). Winner is by MSE.

```markdown
## B — Curve-Shape Metrics — mean ± std across 5 seeds

> Each plot yields a **curve** L. From L, its 1st diff (der) and 2nd diff (sec\_der) we compute three measures,
> combined into one number per plot (combined std = sample std across the 5 seeds):
> - **MSE**: mean((L\_gen − L\_real)²) per sub-metric; combined = **mean-of-3** (funct+der+sec\_der)/3. Decides the winner.
> - **% err**: mean(|L\_gen − L\_real| / (|L\_real| + 1e-6)) × 100 — a function-level MAPE with a **fixed 1e-6
>   floor**, one division; **funct-only** (the funct sub-metric alone).
> - **NRMSE**: sqrt(mean((L\_gen − L\_real)²)) / (max|L\_real| − min|L\_real| + 1e-12) × 100; **funct-only**.
> **% err and NRMSE are funct-only for every plot**: the der/sec\_der of every curve have near-zero true
> values, so their relative errors blow up into meaningless 10⁴-% figures; only the curve itself (funct)
> carries a meaningful relative error. MSE has no such problem and keeps mean-of-3.
> All ↓ lower is better. The Perfect floor is **non-zero** (independent Heston draw vs test set, §5.4).
> Winner is by MSE.

| Plot | Measure | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect |
|------|---------|-----------|--------|--------|--------|--------|--------|:------:|
| **Log-return histogram** | MSE   | ... | ... | ... | ... | ... | ... | ... |
|                          | % err | ... | ... | ... | ... | ... | ... | ... |
|                          | NRMSE | ... | ... | ... | ... | ... | ... | ... |
| **QQ plot**              | MSE   | ... | | | | | | ... |
|                          | % err | ... | | | | | | ... |
|                          | NRMSE | ... | | | | | | ... |
| ... (ACF \|r\|, ACF r², rolling vol hist., tail survival — same 3 rows each) |
```

> **Note on the Perfect Recovery floor.** The floor is the **last column** (`Perfect` / `Perfect floor`) of
> the Section 2 (A1–A34) and Section 3 (B) tables above, and is documented in full at
> `methods/perfect_recovery/README.md`. Regenerate the A and B floor values from
> `methods/perfect_recovery/results/*` via `render_tables.py` (§5.4) so they match byte-for-byte; they are
> identical across every method because they are dataset-derived, not method-derived. PS-MC rows show `—`.

#### Section 4 — Stylised Facts Diagnostic

```markdown
## Stylised Facts Diagnostic (Heston vs <Method>, seed 0)

Eight-panel comparison: sample paths, return distribution, QQ plot, ACF of |returns|,
ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

![Heston Diagnostics](../../results/Heston/<Method>/plots/heston_diagnostics.png)
```

#### Section 5 — Training Loss

```markdown
## <Method> Training Loss (5 seeds)

<Describe training phases (if any), total steps, what each loss component measures.>

![<Method> Training Loss](losses/loss_convergence.png)
```

#### Section 6 — A18

```markdown
## A18 — Discriminative Classifier Training Loss

BCE loss during GRU and MLP classifier training (2 000 steps, logged every 50 steps).
A value near ln(2) ≈ 0.693 means the classifier cannot distinguish real from fake.

![Discriminative Classifier Loss](../../results/Heston/<Method>/plots/disc_classifier_loss.png)
```

#### Section 7 — A19

```markdown
## A19 — Predictive Score Training Loss (TSTR)

MAE loss during GRU and MLP predictor training on *synthetic* data (5 000 steps, logged every 100 steps).

![Predictive Score Loss](../../results/Heston/<Method>/plots/pred_score_loss.png)
```

#### Section 8 — Path Shadowing MC

```markdown
## Path Shadowing MC (arXiv:2308.01486)

Given a real path prefix (steps 0–63), embed via multi-scale log-returns (eq. 13,
α=1.15, β=0.9, dim=22), retrieve K=77 nearest <Method> paths by L2 distance,
use their price-anchored futures as a forecast ensemble.
Two variants: **Uniform** (flat 1/K) and **Gaussian** (η = η̃·‖h(x̃)‖, η̃ calibrated from data).

### Example ensemble fan-out (seed 0)

![PS-MC Example](../../results/Heston/<Method>/path_shadowing/plots/ps_mc_example.png)

### CRPS per forecast step

![CRPS per step](../../results/Heston/<Method>/path_shadowing/plots/crps_per_step.png)

### Results (mean ± std, 5 seeds)

| Metric | H=32 Uniform | H=32 Gaussian | H=64 Uniform | H=64 Gaussian | Naive RW |
|--------|:------------:|:-------------:|:------------:|:-------------:|:--------:|
| **CRPS** | X.XXX ± X.XXX | X.XXX ± X.XXX | X.XXX ± X.XXX | X.XXX ± X.XXX | 3.73 / 5.30 |
| MAE    | ... | ... | ... | ... | 3.73 / 5.30 |
| RMSE   | ... | ... | ... | ... | 5.07 / 7.18 |

Full analysis: [`results/Heston/<Method>/path_shadowing/README.md`](../../results/Heston/<Method>/path_shadowing/README.md)
```

#### Section 9 — File layout

Show the exact folder tree of `methods/<Method>/` (fill in actual file names).

#### Section 10 — Reproduce

```markdown
## Reproduce

\`\`\`bash
# Train all 5 seeds (2 A100 GPUs in parallel)
cd methods/<Method>/code
python train.py --gpu0 0 --gpu1 3

# Compute all metrics
cd metrics
python compute_all.py --method <Method> --dataset Heston

# Run Path Shadowing MC
cd methods/<Method>/path_shadowing
python run_eval.py
\`\`\`
```

---

## 9. Results README (`results/Heston/<Method>/README.md`)

This README lives in the results tree and provides deeper analysis.
It is linked from the method README.

Required sections:

1. **Header**: method name, paper reference, one-line description
2. **Metrics table**: same A1–A34 + B tables as §8 Sections 2–3 (keep in sync), each ending with a Perfect column
3. **Per-seed breakdown**: for the 4 most diagnostic metrics (A6 Path MMD², A18 Disc GRU, A33 Teacher-Sigma Corr, A12 RV Law Loss), list each seed's value and note which seed is weakest and why
4. **Observations**: 3–5 bullet points on what the metrics reveal — strengths, weaknesses, unexpected findings
5. **File index table**

> **When introducing a new method**, the top-level comparison tables (`results/README.md` and root
> `README.md`) are **column-oriented**: one column per method plus a `Winner` column and a `Perfect`
> floor column. Adding a method means **adding one column** to every A1–A34 / B / PS-MC row — never
> re-order or renumber existing rows, and recompute the `Winner` cell across all method columns. The
> Perfect floor column is dataset-derived and stays identical regardless of how many methods are added.

---

## 10. Code README (`methods/<Method>/code/README.md`)

```markdown
# <Method> — Code

## Paper
**Title**: <full title>
**Authors**: <comma-separated authors>
**Venue**: <conference/journal>, <year>
**arXiv**: https://arxiv.org/abs/<id>
**GitHub**: https://github.com/<author>/<repo>

## Method summary
<2–4 sentences on the algorithm: what it optimises, how it generates paths>

## Variant chosen
<If multiple variants exist: which one, and why it fits Heston>

## Fixes applied to the reference implementation
1. Fix 1: <description> — <why needed>
2. Fix 2: ...

## Architecture
| Component | Type | Hidden dim | Output |
|-----------|------|-----------|--------|
| ... | ... | ... | ... |

## Hyperparameters (from paper)
| Parameter | Value | Source |
|-----------|-------|--------|
| ... | ... | Paper §X / repo demo |

## Environment
- Python: 3.12 (or 3.11 if required by the method)
- PyTorch: <version>
- GPU: A100-SXM4-80GB
- Training time per seed: ~X min on one A100
```

---

## 11. Path Shadowing README (`results/Heston/<Method>/path_shadowing/README.md`)

Copy `results/Heston/TimeGAN/path_shadowing/README.md` and adapt:
- Replace `TimeGAN` with `<Method>` everywhere
- Update all numerical results (per-seed CRPS, summary table)
- The **method description sections** (embedding, KNN, anchoring, η calibration)
  are **kept word for word** — the PS-MC algorithm is identical across methods

---

## 12. Git Discipline

### Commit format

```
feat(<method-lowercase>): <what was added>

<Optional 1-2 sentence body with key numbers or decisions.>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

Examples:
```
feat(sbts): train 5 seeds, save generated paths + weights
feat(sbts): compute A1–A34 metrics, heston_diagnostics figure
feat(sbts): run PS-MC evaluation, add path_shadowing README
feat(sbts): write all READMEs — method + results + path_shadowing
```

### What NOT to commit

- Large intermediate files (> 50 MB) — check with `ls -lh` before `git add`
- `.omc/` session state (already in `.gitignore`)
- `__pycache__/` directories
- Any file containing API keys or credentials

### Push cadence — push after each milestone, not during

1. After training all 5 seeds completes
2. After metrics computation completes
3. After PS-MC evaluation completes
4. After all READMEs are written

---

## 13. Checklist — Before Marking a Method as Done

Work through this list top to bottom. Every box must be checked.

```
TRAINING
  [ ] nvidia-smi and htop checked before launch
  [ ] 5 separate seed runs completed (2+2+1 batches)
  [ ] generated_paths/seed_{0..4}/generated_paths_8192x128.npy
       shape (8192, 128), dtype float64, original price scale, values > 0
  [ ] generated_paths/seed_{0..4}/metadata.json  — all fields filled
  [ ] weights/seed_{0..4}_model.pt  — file exists, loadable
  [ ] weights/seed_{0..4}_config.json  — all hyperparameters recorded
  [ ] losses/seed_{0..4}_losses.csv  — correct columns, > 0 rows
  [ ] losses/loss_convergence.png  — 5 seeds overlaid, file > 0 bytes

METRICS
  [ ] results/Heston/<Method>/seed_{0..4}_metrics.json  — 55 values each
  [ ] A25-A34 and B_ keys present in every seed JSON — verify:
       python -c "import json,glob; [print(f, json.load(open(f)).get('A34_hill_tail_error','MISSING'), json.load(open(f)).get('B_log_ret_hist_funct','MISSING')) for f in sorted(glob.glob('results/Heston/<Method>/seed_*_metrics.json'))]"
  [ ] results/Heston/<Method>/metrics_summary.json  — mean ± std present
  [ ] results/Heston/<Method>/plots/heston_diagnostics.png  — 8 panels visible
  [ ] results/Heston/<Method>/plots/disc_classifier_loss.png
  [ ] results/Heston/<Method>/plots/pred_score_loss.png

PATH SHADOWING
  [ ] results/Heston/<Method>/path_shadowing/seed_{0..4}_results.json
  [ ] results/Heston/<Method>/path_shadowing/summary.json
  [ ] results/Heston/<Method>/path_shadowing/plots/ps_mc_example.png
  [ ] results/Heston/<Method>/path_shadowing/plots/crps_per_step.png
  [ ] CRPS_h32 < 3.732 for at least 4/5 seeds (beats naive RW)

DOCUMENTATION
  [ ] methods/<Method>/README.md  — all sections present (§8), no placeholder text; A1–A34 table and
       B table both end with a `Perfect` column (NO standalone "## Perfect Recovery Floor" section)
  [ ] methods/<Method>/code/README.md  — paper, variant, fixes, architecture, hyperparams
  [ ] results/Heston/<Method>/README.md  — metrics table (with Perfect column), per-seed breakdown, observations
  [ ] results/Heston/<Method>/path_shadowing/README.md  — method detail, results, figures

PERFECT RECOVERY FLOOR (reproducible, independent draw vs test set)
  [ ] methods/perfect_recovery/results/{metrics_summary.csv, curve_b_aggregate.json, seed_*_metrics.json}
       — run: python metrics/compute_perfect_recovery.py   (add --no-pytorch to skip A18/A19)
  [ ] Perfect column added as the LAST column of every A1–A34 and B table in: methods/<Method>/README.md,
       results/Heston/<Method>/README.md, results/README.md, root README.md
  [ ] Floor values IDENTICAL across all methods (A + B non-zero, regenerated via render_tables.py from
       methods/perfect_recovery/results/*); PS-MC rows show "—"

GITHUB
  [ ] All files committed with correct message format
  [ ] Pushed to origin/master
  [ ] Open GitHub and verify every PNG image renders (not broken link)
  [ ] Verify every relative path (../../results/...) resolves correctly on GitHub
  [ ] GUIDELINE.md checklist updated if any new standard was established
```

---

## 14. SBTS-Specific Notes

**Paper**: "Robust Time Series Generation via Schrödinger Bridge: A Comprehensive Evaluation"
Alouadi, Barreau, Carlier, Pham — ICAIF 2025 — arXiv:2503.02943

**GitHub**: https://github.com/alexouadi/SBTS

**What it does**: Formulates time series synthesis as an entropic optimal transport
problem between a reference measure on path space and the target joint distribution,
yielding a forward-backward SDE that is solved iteratively (Schrödinger Bridge / IPF).

**Repo structure**:
```
SBTS/
├── models/
│   ├── sbts_uni.py              univariate SB (full, non-Markovian)
│   ├── sbts_uni_markovian.py    univariate SB (Markovian approximation)
│   ├── sbts_multi.py            multivariate SB
│   ├── sbts_multi_markovian.py  multivariate SB Markovian
│   └── hyperparams_selection/
├── metrics/                     their metrics — DO NOT USE (use ours in metrics/)
├── utils/
├── data/
├── experiments_demo.ipynb       only entry point — no standalone train script
└── requirements.txt             torch==2.2.2, numpy==1.23.5, numba==0.56.4, …
```

**Dependencies**: `torch==2.2.2`, `numpy==1.23.5`, `scipy==1.12.0`, `numba==0.56.4`,
`scikit_learn==1.2.2`, `pandas==1.5.3`, `seaborn==0.12.2`.
SBTS states Python 3.11. Check compatibility with gpu-venv before starting.

**All decisions confirmed (2026-07-17):**

| # | Decision | Detail |
|---|----------|--------|
| SB-Q1 | **`sbts_uni_markovian.py`, k=1** | Paper explicitly uses k=1 for Heston (Appendix C). Heston is truly Markovian: given (Sₜ, vₜ) the future is independent of the past. Full non-Markovian variant produces null kernel weights on long series. |
| SB-Q2 | **1D price paths, same format as TimeGAN** | SBTS internally operates on scaled log-returns R̃ = R × √Δt / σ(R). After generation, invert: R_gen = R̃_gen × σ(R) / √Δt, then S_gen[:,t+1] = S_gen[:,t] × exp(R_gen[:,t]) with S_gen[:,0] = S_real[:,0] ≈ 100. Save price paths (8192×128, float64). All metrics and plots in price space. |
| SB-Q3 | **CPU + Numba for generation; GPU for A18/A19 metrics only** | SBTS has no neural network. Paper used CPU+Numba (12 cores, 548 s for 1 000 samples at T=252). Our setup (8 192 paths, T=128): **64 workers → ~6.3 min/seed** (seeds 1-4: 370-384 s); 16 workers → ~23 min/seed. Rule: total time ≈ (8 192 / n_workers) × 2.9 s. GPUs 0+3 only for disc/pred scores. Check Numba compatibility with gpu-venv; if conflict, install numba==0.56.4 in gpu-venv or create sbts-venv. |
| SB-Q4 | **h=0.4, k=1, N^π=200, Δt=1/250** | From Appendix C Table 4: h=0.4 for Heston T=100, Δt=1/252. We start with h=0.4 and run CV bandwidth selection (Section 3 of paper) for our T=128, M=8192. Grid H = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6} — pick h* minimising MSEₕ. |
| SB-Q5 | **No loss CSV — log bandwidth CV curve instead** | SBTS is pure kernel density estimation — no parameters, no gradient, no training loop. Instead save: `losses/bandwidth_cv.csv` (h, MSEₕ per seed) and `losses/generation_time.csv` (seed, n_samples, elapsed_sec). Plot MSEₕ vs h in place of loss convergence plot. |

**Measured timing on this cluster (8 192 paths, T=128, Numba JIT, sbts-venv):**

| Method | Hardware | Workers / GPUs | Time / seed | Notes |
|--------|----------|----------------|-------------|-------|
| SBTS generation | CPU (EPYC 7763) | 64 workers | **~6.3 min** (376 s avg) | ~2.9 s/path; rule: (8192/W) × 2.9 s |
| SBTS generation | CPU (EPYC 7763) | 16 workers | ~23.4 min (1 405 s) | Only for seed 0 (initial test) |
| TimeGAN training | 1 × A100 80 GB | 1 GPU | **5.5–8 min** (323–487 s) | Variance reflects shared-machine load |
| TimeGAN generation | 1 × A100 80 GB | 1 GPU | < 1 s | GRU forward pass, milliseconds |

Paper timing for reference: Alouadi et al. report 548 s for 1 000 samples at T=252 with 12 CPU cores (~6.6 s/path).
Our T=128 (vs their T=252) explains the faster per-path time (~2.9 s vs ~6.6 s).

---

## 15. README Writing Protocol (Precise Format)

This section specifies the EXACT format for every README type in the benchmark.
Follow it exactly when adding a new method. Do NOT deviate from section titles or ordering.

---

### 15.1 Method README (`methods/<Method>/README.md`) — Required Sections

#### Title
```markdown
# <Method> on Heston
```

#### Section 2 — Metrics A1–A34 + B
```markdown
## Metrics A1–A34 + B — mean ± std across 5 seeds

> All metrics use log-returns r_t = log(S_{t+1}/S_t) unless noted. A26 uses price increments ΔS_t.

Rows MUST be grouped by category, in this exact order, with a `| | **— <Category> —** | | ... |`
separator row before each group: **Fat Tail → Distribution → Adversarial → Predictive → Temporal → Vol
→ Heston Spec**. This ordering is mandatory across every A1–A34 table in the repo (methods/, results/,
and root README.md) — the ID numbers already follow this order (A1–A5 Fat Tail, A6–A17 Distribution,
A18 Adversarial, A19 Predictive, A20–A24 Temporal, A25–A32 Vol, A33–A34 Heston Spec).

| ID | Metric | Category | Dir | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |
|----|--------|----------|-----|-----------|--------|--------|--------|--------|--------|---------------|
| | **— Fat Tail —** | | | | | | | | | |
| A1 | Kurtosis Error | Fat Tail | ↓ | X.XXXX ± X.XXXX | ... | 0.008092 |
| A2 | \|r\| q95 Error | Fat Tail | ↓ | ... | 6.57e-05 |
| A3 | \|r\| q99 Error | Fat Tail | ↓ | ... | X.XXXX |
| A4 | Tail QQ Error | Fat Tail | ↓ | ... | X.XXXX |
| A5 | Hill Tail Index Error | Fat Tail | ↓ | ... | 0.5266 |
| | **— Distribution —** | | | | | | | | | |
| A6  | Path MMD²             | Distribution  | ↓ | X.XXXX ± X.XXXX | ... | X.XXXX |
...
| | **— Adversarial —** | | | | | | | | | |
| A18 GRU | Discriminative Score GRU | Adversarial | ↓ | ... | X.XXXX |
| A18 MLP | Discriminative Score MLP | Adversarial | ↓ | ... | X.XXXX |
| | **— Predictive —** | | | | | | | | | |
| A19 GRU | Predictive Score GRU (TSTR) | Predictive | ↓ | ... | X.XXXX |
| A19 MLP | Predictive Score MLP (TSTR) | Predictive | ↓ | ... | X.XXXX |
...
| | **— Heston Spec —** | | | | | | | | | |
| A33 | Teacher-Sigma Correlation | Heston Spec | ↑ | ... | 0.6163 |
| A34 | Teacher-Sigma RMSE | Heston Spec | ↓ | ... | 0.06559 |

> **Convention:** ↓ lower is better; ↑ higher is better; — no monotone direction.
> **A1**: excess kurtosis error of pooled log-returns. **A2–A3**: |r| q95 / q99 tail-quantile errors.
> **A4**: Tail QQ error (extreme top/bottom 5%). **A5**: Hill tail index error (top 10% of S_T).
> **A12**: RV Law Loss = W₁(RV_real, RV_gen), RV_i = Σ_t r²_{i,t} / dt. Ref: Barndorff-Nielsen & Shephard (2002).
> **A18**: Discriminative classifier (GRU + MLP). Score = |accuracy − 0.5|; 0 = indistinguishable.
> **A19**: TSTR MAE (GRU + MLP); cluster near 0.056–0.059 (irreducible noise floor for Heston).
> **A20**: covariance error (Frobenius; |Var| for d=1). **A21–A22**: ACF error on |r| / r² at lags {1,2,5,10}.
> **A23–A24**: single-lag (lag=1) ACF errors on |r| / r². Heston true values ≈ +0.052 / +0.050.
> **A28**: Kurtosis ratio κ_real / κ_gen. Perfect = 1.0 (no monotone direction; closest to 1 wins).
> **A29**: Sigma mean error (annualized per-path vol). **A32**: |vol-of-vol_real − vol-of-vol_gen|.
> **A33**: Pearson(σ̂_gen, √v_true) — teacher-sigma correlation (↑ higher is better).
> **A34**: RMSE(σ̂_gen, √v_true) — absolute scale accuracy of the reproduced vol process.
```

#### Section 3 — B Curve-Shape Metrics
```markdown
## B — Curve-Shape Metrics — mean ± std across 5 seeds

> Each plot yields a **curve** L (not a scalar). From L, its 1st diff (der) and 2nd diff (sec\_der) we
> compute three measures, combined into one number per plot (combined std = sample std across the 5 seeds):
> - **MSE**: mean((L\_gen − L\_real)²) per sub-metric; combined = **mean-of-3** (funct+der+sec\_der)/3. Decides the winner.
> - **% err**: mean(|L\_gen − L\_real| / (|L\_real| + 1e-6)) × 100 — a function-level MAPE with a **fixed 1e-6
>   floor**, one division; **funct-only** (the funct sub-metric alone).
> - **NRMSE**: sqrt(mean((L\_gen − L\_real)²)) / (max|L\_real| − min|L\_real| + 1e-12) × 100; **funct-only**.
> **% err and NRMSE are funct-only for every plot** (the der/sec\_der true values are near-zero and blow the
> ratios up into 10⁴-% figures; only funct carries a meaningful relative error). MSE keeps mean-of-3.
> All ↓ lower is better. The Perfect floor is **non-zero** (independent draw vs test set, §5.4). Winner is by MSE.

| Plot | Measure | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect |
|------|---------|-----------|--------|--------|--------|--------|--------|:------:|
| **Log-return histogram** | MSE   | X.XX ± X.XX | ... | X.XX |
|                          | % err | XXXXX% ± XXXX% | ... | XXX% |
|                          | NRMSE | XXX% ± XX% | ... | XX% |
| **QQ plot**              | MSE   | X.XXe-X ± X.XXe-X | ... | X.XXe-X |
|                          | % err | XXX% ± XX% | ... | XX% |
|                          | NRMSE | XX% ± XX% | ... | XX% |
| ... (ACF \|r\|, ACF r², rolling vol hist., tail survival — same 3 rows each) |
```

The % err row can still be large (triple-digit-plus %) wherever the real curve passes through near-zero
values — expected, a property of the curve, not a bug.

> **Note on the Perfect Recovery floor.** The floor is the **last column** (`Perfect` / `Perfect floor`) of
> the Section 2 (A1–A34) and Section 3 (B) tables, documented in full at
> `methods/perfect_recovery/README.md`. Regenerate the A and B floor values from
> `methods/perfect_recovery/results/*` via `render_tables.py` (§5.4) so they match byte-for-byte — they are
> dataset-derived (an independent Heston draw scored against the test set), so they are **non-zero** and
> identical across every method's README. If the outputs don't exist yet, generate them once with
> `python metrics/compute_perfect_recovery.py`. PS-MC rows show `—`.

#### Sections 4–10 — Stylised Facts → Reproduce

The method README continues with, in this exact order. Full verbatim templates below (retaken from
§8 Sections 4–10 — copy each block, replace `<Method>` everywhere, fill the `...`/`X.XXX` cells).

##### Section 4 — Stylised Facts Diagnostic

```markdown
## Stylised Facts Diagnostic (Heston vs <Method>, seed 0)

Eight-panel comparison: sample paths, return distribution, QQ plot, ACF of |returns|,
ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

![Heston Diagnostics](../../results/Heston/<Method>/plots/heston_diagnostics.png)
```

##### Section 5 — Training Loss

```markdown
## <Method> Training Loss (5 seeds)

<Describe training phases (if any), total steps, what each loss component measures.>

![<Method> Training Loss](losses/loss_convergence.png)
```

For training-free methods (e.g. kernel/bandwidth-based), replace the loss curve with the relevant
diagnostic (e.g. bandwidth-CV plot) and describe what is being selected instead of "trained".

##### Section 6 — A18 Discriminative Classifier Training Loss

```markdown
## A18 — Discriminative Classifier Training Loss

BCE loss during GRU and MLP classifier training (2 000 steps, logged every 50 steps).
A value near ln(2) ≈ 0.693 means the classifier cannot distinguish real from fake.

![Discriminative Classifier Loss](../../results/Heston/<Method>/plots/disc_classifier_loss.png)
```

##### Section 7 — A19 Predictive Score Training Loss (TSTR)

```markdown
## A19 — Predictive Score Training Loss (TSTR)

MAE loss during GRU and MLP predictor training on *synthetic* data (5 000 steps, logged every 100 steps).

![Predictive Score Loss](../../results/Heston/<Method>/plots/pred_score_loss.png)
```

##### Section 8 — Path Shadowing MC

```markdown
## Path Shadowing MC (arXiv:2308.01486)

Given a real path prefix (steps 0–63), embed via multi-scale log-returns (eq. 13,
α=1.15, β=0.9, dim=22), retrieve K=77 nearest <Method> paths by L2 distance,
use their price-anchored futures as a forecast ensemble.
Two variants: **Uniform** (flat 1/K) and **Gaussian** (η = η̃·‖h(x̃)‖, η̃ calibrated from data).

### Example ensemble fan-out (seed 0)

![PS-MC Example](../../results/Heston/<Method>/path_shadowing/plots/ps_mc_example.png)

### CRPS per forecast step

![CRPS per step](../../results/Heston/<Method>/path_shadowing/plots/crps_per_step.png)

### Results (mean ± std, 5 seeds)

| Metric | H=32 Uniform | H=32 Gaussian | H=64 Uniform | H=64 Gaussian | Naive RW |
|--------|:------------:|:-------------:|:------------:|:-------------:|:--------:|
| **CRPS** | X.XXX ± X.XXX | X.XXX ± X.XXX | X.XXX ± X.XXX | X.XXX ± X.XXX | 3.73 / 5.30 |
| MAE    | ... | ... | ... | ... | 3.73 / 5.30 |
| RMSE   | ... | ... | ... | ... | 5.07 / 7.18 |

Full analysis: [`results/Heston/<Method>/path_shadowing/README.md`](../../results/Heston/<Method>/path_shadowing/README.md)
```

##### Section 9 — File layout

Show the exact folder tree of `methods/<Method>/` (fill in actual file names): `code/`, `losses/`,
`generated_paths/seed_{0..4}/`, `path_shadowing/`, and the method README itself.

##### Section 10 — Reproduce

```markdown
## Reproduce

\`\`\`bash
# Train all 5 seeds (2 A100 GPUs in parallel)
cd methods/<Method>/code
python train.py --gpu0 0 --gpu1 3

# Compute all metrics
cd metrics
python compute_all.py --method <Method> --dataset Heston

# Run Path Shadowing MC
cd methods/<Method>/path_shadowing
python run_eval.py
\`\`\`
```

Quick-reference ordering (same as the templates above):

4. **Stylised Facts Diagnostic** — 8-panel PNG `plots/heston_diagnostics.png` (sample paths, return
   distribution, QQ, ACF |r|, ACF r², rolling-vol histogram, tail survival).
5. **Training Loss** — `losses/loss_convergence.png` (or bandwidth-CV plot for training-free methods).
6. **A18 Discriminative Classifier Training Loss** — `plots/disc_classifier_loss.png` (GRU + MLP, BCE).
7. **A19 Predictive Score Training Loss (TSTR)** — `plots/pred_score_loss.png` (GRU + MLP, MAE).
8. **Path Shadowing MC** — ensemble fan-out + CRPS-per-step plots + H=32/64 Uniform/Gaussian table.
9. **File layout** — exact folder tree of `methods/<Method>/`.
10. **Reproduce** — bash commands: train all 5 seeds, `compute_all.py`, PS-MC `run_eval.py`.

---

### 15.2 Results README (`results/Heston/<Method>/README.md`) — Required Sections

**Section order (mandatory).** This order is copied from the current `results/Heston/SBTS/README.md`
(the ground-truth reference). Do NOT put "Comparison with paper" before "Curve-shape metrics".

1. **Header** — method name, paper citation, one-line description, convention note (↓ lower is better
   except A33 Teacher-Sigma Corr ↑; A28 Kurtosis Ratio: perfect = 1.0).

2. **What we generate — price paths from the Heston SDE** (if applicable) — SDE, scaling steps, pipeline.

3. **Results (mean ± std across 5 seeds)**
   - Subsection **A1–A34 Core metrics**: full table with columns
     `ID | Metric | Mean ± Std | Seed 0..4 | Perfect` — the **last column is the Perfect floor**
     (non-zero; regenerated from `methods/perfect_recovery/results/*` via `render_tables.py`, §5.4). Rows
     grouped by category in the mandatory order: **Fat Tail → Distribution → Adversarial → Predictive →
     Temporal → Vol → Heston Spec** (see §15.1).
   - Footnotes for A18, A19, A33, A34 (and any other non-obvious IDs).

4. **Stylised Facts Diagnostic** — the 8-panel diagnostic PNG plus one-line description:
   ```markdown
   ## Stylised Facts Diagnostic

   Eight-panel comparison (seed 0): sample paths, return distribution, QQ plot, ACF of |returns|,
   ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

   ![Heston Diagnostics](plots/heston_diagnostics.png)
   ```

5. **Curve-shape metrics (B)** — the three-subline B table (MSE row + % err row + NRMSE row per plot), each
   with a trailing `Perfect` column (non-zero for every plot). Winner is by MSE. Reuse the header blockquote
   from §15.1 Section 3 (MSE combined mean-of-3; % err / NRMSE funct-only for every plot, fixed 1e-6 floor). The
   % err row can still be large (triple-digit %) wherever the real curve passes through near-zero values.

6. **Comparison with the paper** — CANONICAL 3-COLUMN STRUCTURE.
   ⚠️ **This section uses the paper's OWN metrics ONLY** (whatever the paper reports in its results
   table — e.g. F-score + MAE for Fourier Flow, discriminative + predictive score for TimeGAN). Do NOT
   use the A1–A34 suite here; the A-suite comparison lives in the Results section above. The point of
   this section is a like-for-like check against the published numbers.

   - ⚠️ Warning that direct comparison may not be meaningful (state why: different dataset, different T,
     different d, and — if relevant — a scale note, since the paper's metric is applied to Heston prices
     placed on the paper's own normalisation, e.g. [0,1] MinMax).
   - Subsection A: Hyperparameter verification table (columns: `Setting | Our reimplementation | Paper (source)`).
   - Subsection B: **The unified score table.** One row per **paper metric**, three value columns:

     | Column | Meaning |
     |--------|---------|
     | **Paper (Table N, <PaperDataset>)** | the number the paper published (its figures, its dataset) |
     | **Ours — <PaperDataset> (paper dataset)** | our port / the released code run on the paper's OWN dataset (the `paper_reimplementation/` reproduction) |
     | **Ours — Heston** | the **same paper metric functions** applied to our 5-seed Heston paths |

     ```markdown
     | Metric (paper's own) | Paper (Table N, <PaperDataset>) | Ours — <PaperDataset> (paper dataset) | Ours — Heston |
     |----------------------|:-------------------------------:|:-------------------------------------:|:-------------:|
     | <metric1> ↑/↓ | <paper val> | **<ours paper-dataset val ± std>** | **<ours Heston val ± std>** |
     | <metric2> ↑/↓ | <paper val> | **<ours paper-dataset val ± std>** | **<ours Heston val ± std>** |
     ```

     Worked example (`results/Heston/FourierFlow/README.md`, paper metrics = Sajjadi F-score + TSTR MAE):
     ```markdown
     | Metric (paper's own) | Paper (Table 2, Stocks) | Ours — Stocks (paper dataset) | Ours — Heston |
     |----------------------|:-----------------------:|:-----------------------------:|:-------------:|
     | F-score ↑ | 0.984 | **0.9920 ± 0.0017** | **0.9918 ± 0.0009** |
     | MAE ↓ | 0.009 | **0.0084 ± 0.0007** | **0.0210 ± 0.0132** |
     ```
     Quote the exact paper table number, the dataset, and the metric definition. State how the paper
     metric was applied to Heston (normalisation + any length adaptation, e.g. the TSTR LSTM
     `MAX_STEPS` set to the Heston sliced length). The **Ours — Heston** column MUST come from a
     committed JSON produced by a reusable driver in `paper_reimplementation/metric/` (e.g.
     `heston_paper_metrics.py` → `results/heston_paper_metrics.json`), scoring the already-generated
     5-seed pool (no retraining). Never hand-type a Heston paper-metric value — compute it.
   - Subsection C (if applicable): Scaling / implementation notes; the paper-vs-code caveat.

   **Older two-table form (still valid for TimeGAN/SBTS, whose paper metrics = disc/pred score).**
   Methods added before this canonical form kept a Heston-vs-paper table plus a separate
   `## Paper reproduction on <PaperDataset>` block below it. That is equivalent — the reproduction
   block is just the **Ours — <PaperDataset>** column split out — and does not need retrofitting.
   For a method whose OWN official code is used (e.g. SBTS), the **Ours — <PaperDataset>** column is a
   single `Ours (official <Method> code)` column.

7. **Files** — a file index table (or tree) of `results/Heston/<Method>/`: the README, `plots/`,
   per-seed metric JSONs, `curve_b_aggregate.json`, `path_shadowing/`.

> **Note:** The Metric Definitions section (A1–A34 with LaTeX formulas) lives in `metrics/README.md`
> and `methods/<Method>/code/README.md` — NOT in the results README. The results README contains
> tables only; cross-link to the method README for formula details.

---

### 15.3 Code README (`methods/<Method>/code/README.md`) — Required Sections

**Section order (mandatory):**

1. **Original work** table: Paper, Authors, Venue, arXiv, GitHub, Original framework
2. **Our implementation** — what changed vs the reference
3. **Architecture table** (if neural network): component, layers, output shape
4. **Fixes applied vs the reference** — numbered list, each fix: `Location | Reference (buggy) | Our fix`
5. **Hyperparameters (from paper)** table: `Parameter | Value | Source in paper (§, Table, Appendix)`
6. **How to change hyperparameters** — explicit instructions per parameter; how to pass CLI args or edit config
7. **How to use a different dataset** — required file format (.npy, shape, dtype, scale); which lines to change in the code
8. **How to produce new seeds** — command to run single seed; command for all 5; where outputs land
9. **Reproduce (EXACT run path — mandatory)** — this is not a generic recipe, it is **what we actually
   ran to produce the committed numbers**. It MUST contain, copy-pasteable and verbatim:
   - the **exact interpreter path** (e.g. `/home/tbasseras/gpu-venv/bin/python`, not `python`),
   - any **env vars** actually set (`PYTHONPATH=reference`, `CUDA_VISIBLE_DEVICES=0`, `taskset -c 0-7`,
     `OMP_NUM_THREADS=8`, `TF_USE_LEGACY_KERAS=1`),
   - the **precise script file name(s)** invoked, with the exact CLI flags used (`train_heston.py --arch mujoco --seed 0`),
   - the **exact input file(s) consumed** and **exact output file(s) produced** (path + shape), and
   - **which output/checkpoint produced which reported number** — e.g. "metrics row X came from
     `results/generated_paths/seed_0/…npy` scored by `metrics/compute_all.py`; the EMA milestone-10
     checkpoint `…/checkpoint-10.pt` was used for sampling". A reader must be able to regenerate any single
     cell of any table without guessing which file or flag was involved.
10. **Sanity check results** (optional but recommended) — show small-scale test output

---

### 15.3.1 Paper-Reimplementation README (`methods/<Method>/paper_reimplementation/README.md`)

This README documents the §3.0 paper-reproduction step: running the method's own code (or a faithful
port) on the **paper's own dataset** (e.g. Stocks) and comparing to the paper's Table 1 — the honest
"did we reproduce the paper?" check, done BEFORE the 5 Heston seeds. Take the two existing files as
the ground-truth templates: `methods/TimeGAN/paper_reimplementation/README.md` and
`methods/SBTS/paper_reimplementation/README.md`.

**Section order (mandatory):**

1. **Header** — `# <Method> — Paper Reimplementation (<PaperDataset>)`; then a short bullet block:
   **Paper** (title, authors, venue), **Official code** (GitHub URL + where mirrored, e.g.
   `../code/reference/`), **This run** (which script, which GPU, wall-clock).
2. **⚠️ Reproduction caveat** (only if a port was needed) — explain WHY the official code could not be
   run as-is (e.g. TimeGAN's TF1 has no CUDA-13 build → CPU fallback → many hours), and state exactly
   what was substituted: which generator (`../code/<port>.py`) and which metric implementation, and note
   that the shared metric makes the "our-run" columns directly comparable across methods.
3. **1. Paper metrics (as defined in the paper)** — a table `Metric | Definition | Direction` quoting
   the paper's own metric definitions (discriminative = |acc − 0.5| via a GRU judge, predictive = TSTR
   MAE, etc.), plus how the paper reports them (mean ± std over N test runs, `n_temp`, `itt`, batch).
4. **2. Hyperparameters** — the exact command / settings from the official README or notebook, then a
   `Parameter | Value` table (module, seq_len, hidden_dim, num_layers, iterations, batch_size) and the
   training wall-clock.
5. **3. Dataset** — the paper dataset (source, shape, dtype, scaling), where it lives under `dataset/`.
6. **4. Results — ours vs paper** — the reproduction table. Columns depend on whether a port was used:
   - Port (TimeGAN): `Dataset | Metric | Ours — 2-layer judge, 1 seed | Ours — 1-layer judge, 5 seeds |
     Paper (Table 1) | Verdict`.
   - Official code (SBTS): `Dataset | Metric | Ours (official <Method> code) | Paper (Table 1) | Verdict`.
   Bold the primary "Ours" column; Verdict states `matches ✓` / `same regime ✓` with the σ-distance.
   **This is the SAME table that gets copied BELOW the paper-comparison table in the results README (§15.2 Section 6).**
7. **5. How to reproduce (EXACT run path — mandatory)** — not a generic recipe but **the exact commands
   we ran to produce every cell of the §4 table**. Same requirement as §15.3 section 9: verbatim
   interpreter path, all env vars actually set (incl. `TF_USE_LEGACY_KERAS=1` for the legacy-keras metric
   env, and the fact that torch metrics run in `gpu-venv` while TF metrics run in `dts-tf-venv`), the
   precise metric-script file name(s) invoked, GPU pinning (`CUDA_VISIBLE_DEVICES`, `taskset`,
   `OMP_NUM_THREADS`), seed split across the 2 allowed GPUs, **and which input file each metric scored**
   (e.g. "Correlational scored `OUTPUT/stock/ddpm_fake_stock.npy` — a single saved draw, hence zero
   run-to-run variance; the paper re-samples 5× so its ±std is not reproducible from one file"). State
   which script writes which result JSON, so any reported number is traceable to one file + one command.
8. **6. Files** — file index of `paper_reimplementation/`: `dataset/`, `metric/`, `results/`, README.

---

### 15.4 Root & Comparison READMEs (`README.md`, `results/README.md`) — Required Sections

Both top-level READMEs are **column-oriented comparisons** (one column per method), not per-method
tables. They must stay in sync with each other and with every method README.

**Section order (mandatory):**

1. **Header** — project one-liner, dataset (Heston, 8192×128), list of methods benchmarked.
2. **A1–A34 comparison table** — columns `ID | Metric | Category | Dir | <Method A> | <Method B> | … |
   Winner | Perfect floor`. Rows grouped by category in the mandatory order (Fat Tail → Distribution →
   Adversarial → Predictive → Temporal → Vol → Heston Spec) with a `**— <Category> —**` separator row
   before each group. A18/A19 split into GRU + MLP sublines. **Winner** cell: ↓ smaller wins; ↑ larger
   wins; A28 Kurtosis Ratio closest-to-1.0 wins. Report the overall win count (`<A> wins X/36,
   <B> wins Y/36`, counting the A18/A19 GRU+MLP sublines).
3. **B Curve-Shape comparison table** — side-by-side by **MSE** (winner by MSE), each plot with a
   trailing `Perfect` column (non-zero). Include the three-subline (MSE + % err + NRMSE) description; **MSE**
   is mean-of-3, **% err / NRMSE are funct-only for every plot** (§8 Section 3). Report B win count (`X/6` MSE each).
4. **Path Shadowing MC comparison** — H=32 / H=64 × {Uniform, Gaussian} CRPS/MAE/RMSE per method plus
   Naive RW baseline.
5. **Training / Generation timing** row(s).
6. **Interpretation bullets** — 3–6 bullets referencing metrics by their **new** IDs (e.g. A20 cov
   error, A18 GRU/MLP, A33 Teacher-Sigma Corr).

**When a new method is added — exact edits to the root `README.md` and `results/README.md`:**

1. **In the Results (comparison) tables — add ONE column for the new method and UPDATE the Winner, for
   BOTH table A and table B:**
   - **Table A (A1–A34):** insert a new `<NewMethod>` column (positioned before the `Winner` /
     `Perfect floor` columns) and fill every row — including the A18/A19 GRU+MLP sublines — from the new
     method's `metrics_summary.csv`. Then **recompute every `Winner` cell** across all method columns
     (↓ smaller wins; ↑ larger wins; A28 Kurtosis Ratio closest-to-1.0 wins) and **update the overall
     win-count line** (`<A> wins X/36, <B> wins Y/36, <NewMethod> wins Z/36`).
   - **Table B (Curve-Shape):** insert the new `<NewMethod>` column into every plot's MSE, % err and NRMSE
     sublines from `curve_b_aggregate.json`, then **recompute the by-MSE `Winner`** for each of the 6
     plots and **update the B win-count line** (`X/6` MSE each).
   - Leave the `Perfect` / `Perfect floor` column unchanged (dataset-derived, identical for all methods).
   - Also add the new column to the **Path Shadowing MC** comparison table.

2. **In the Methods list — add ONE line (row) for the new method.** The root `README.md` lists methods
   in a table (one row per method). Adding a method means appending a new **row** there (name, paper,
   one-line description, link to `methods/<NewMethod>/`) — and adding it to the "methods benchmarked"
   enumeration in the header and the "Adding a new method" section if present.

The `results/README.md` and root `README.md` A-tables must use identical numbers and identical Winner
cells; only prose framing may differ. Never re-order or renumber existing rows.

- All A/B/PS-MC values must be **read back from disk** (`metrics_summary.csv`, `curve_b_aggregate.json`,
  `path_shadowing/` results), never hand-transcribed.

---

### 15.5 GUIDELINE.md Update Protocol

When a new standard is established (new metric, new section format, new B metric removed, new method added):
1. Update section 5.2 metric list (add/remove/rename)
2. Update section 13 checklist (any new file to check?)
3. Add a dated note in section 14 (SBTS-Specific Notes) or a new section
4. Update this section (15) with the new format requirement

**History of changes:**
- 2026-07-17: B8 (ARCH Persistence Error, lags 1-20) and B10 (GARCH Persistence Error, lags 1-20) removed from B metrics as redundant with A11 and A12. Renumbered: old B9→B8, old B11→B9, old B12→B10, old B13→B11, old B14→B12. Total B metrics: 14 → 12. Total grand total: 37 → 35 scalars per seed.
- 2026-07-17: Replaced A16–A20 (tail survival, oracle AR, RV law) with A16–A24 (log-ret std, tail quantile errors, kurtosis ratio, sigma mean error, learned/oracle sigma corr, ACF lag-1 errors, RV law loss).
- 2026-07-17: Added §15 (README Writing Protocol) with exact section order and format for method, results, and code READMEs.
- 2026-07-18: B1–B12 scalar metrics removed; replaced by 18 B curve-shape keys (6 plots × 3 sub-metrics: funct/der/sec\_der). Old B1–B10 scalars promoted to A25–A34 (distributional shape / tail). Grand total: 39 → 55 scalars per seed. Updated §5.2, §5.3, §13, §15.1, §15.2. Added perfect recovery baseline: `results/Heston/perfect_recovery/` (5 seeds, row-shuffle, A1–A34 + B_ all computed).
- 2026-07-19: **B display → two sublines.** Each B plot now shows two combined rows — **MSE** and **% err** — instead of a funct/der/sec\_der 3-column table. The 3 sub-metrics are combined per measure (mean = sum, std = quadrature) in `metrics/stylized_metrics.py`; recompute via `metrics/recompute_curve_b.py`. Winner between methods is decided by the MSE row. Updated §5.2, §8 Section 3, §15.1 Section 3, §15.2.
- 2026-07-19: **Perfect Recovery Floor → column, not section.** The standalone `## Perfect Recovery Floor` section was removed from method READMEs; the floor is now the **last column** (`Perfect` / `Perfect floor`) of every A1–A34 and B table across `methods/<Method>/`, `results/Heston/<Method>/`, `results/README.md`, and root `README.md`. Source of truth: `methods/perfect_recovery/results/metrics_summary.csv` (A floors) + `curve_b_aggregate.json` (B floors, all 0), produced reproducibly by `metrics/compute_perfect_recovery.py` (full-shuffle `S_real[rng.permutation(N)]`, 5-seed average). Floors identical across all methods (dataset-derived). PS-MC rows show `—`. Updated §5.4, §8 Section 4, §13, §15.1 Section 4, §15.2.
- 2026-07-19: **Verified A13/A14 fresh-retrain.** Confirmed `compute_discriminative_score` / `compute_predictive_score` instantiate a new classifier + optimiser on every call, invoked once per seed in `compute_all.py`'s seed loop — no weight caching across seeds or methods. Documented in §7.2.
- 2026-07-19: **B % err formula → per-point magnitude.** The B % err now divides by curve length: `mean(|L_gen − L_real| / (|L_real| + 1e-6)) × 100 / len(L_real)` per sub-metric; combined mean = **mean** of the three sub-metrics, combined std = **sample std across the 5 seeds** (MSE unchanged: sum of three, quadrature std). Updated §5.2, §8 Section 3, §15.1 Section 3. Recompute via `metrics/recompute_curve_b.py`.
- 2026-07-19: **B % err formula → proper MAPE (REVERSES the per-point-magnitude note above).** The extra `/ len(L_real)` was a **second** division on top of the mean (which already divides by the number of curve points), collapsing the percentage ~100× into a meaningless sub-1% artefact. Corrected to a proper MAPE: `mean(|L_gen − L_real| / (|L_real| + 1e-6)) × 100` per sub-metric — **ONE** division. Combined mean/std unchanged (mean-of-3 / sample-std across 5 seeds); MSE untouched. Fixed in `metrics/stylized_metrics.py` (`_pct`), recomputed via `metrics/recompute_curve_b.py` for TimeGAN / SBTS / perfect_recovery, and propagated to all 7 READMEs (values read back from `curve_b_aggregate.json`). Updated §8 Section 3, §15.1 Section 3, §15.4. Also added the trailing `Perfect` column to the `results/README.md` B table for parity with root `README.md`.
- 2026-07-19: **Removed old A21 (Learned/Oracle Sigma Correlation).** Dropped from every A-table (6 method/results/root READMEs), from `metrics/metrics_np.py`, and from `metrics/README.md`. Not the same as the new-numbering A21 (ACF Error, abs returns).
- 2026-07-19: **Full A-metric renumber to category order.** A1–A34 IDs reassigned so they read in display order (A1 Kurtosis Error … A34 Teacher-Sigma RMSE): Fat Tail A1–A5, Distribution A6–A17, Adversarial A18 (GRU+MLP), Predictive A19 (GRU+MLP), Temporal A20–A24, Vol A25–A32, Heston Spec A33–A34. **Formulas unchanged — only order/numbering.** Scope: JSON keys, `compute_all.py` identifiers, all README labels, `metrics/README.md` and `metrics_np.py` legends, and §5.2/§8/§15.1 in this file. A28 Kurtosis Ratio (perfect = 1.0, closest wins); A33 Teacher-Sigma Corr (↑).
- 2026-07-19: **§8/§15.1 Section renumbering + column note.** Removed the "Section 4 — (removed) Perfect Recovery Floor" meta-blocks; the floor note is now inline and Sections renumber 5→4 … 11→10 (§8). Re-added the Stylised-Facts→Reproduce section list in §15.1. Added §9 note + §15.4 (Root & Comparison README protocol): tables are column-oriented, adding a method = adding one column + recomputing Winner cells; old §15.4 Update Protocol → §15.5.
- 2026-07-19: **Paper-reimplementation-first (§3.0).** New standard: before generating the 5 Heston seeds, reproduce the paper's own results on the paper's own dataset in `methods/<Method>/paper_reimplementation/` (README paper-vs-ours table, `dataset/`, `metric/`, generated data + weights). If repro fails, explain and STOP. Exact code cells: `metrics/compute_all.py --dataset <PaperDataset>` for metrics and `metrics/plot_diagnostics.py --dataset <PaperDataset> --seed 0` for the 8 stylised-facts curve plots.
- 2026-07-19: **§15 detail expansion (additive).** (1) §15.1 Sections 4–10 expanded from a terse list into the full verbatim markdown templates retaken from §8 (Stylised Facts → Training Loss → A18 → A19 → Path Shadowing MC → File layout → Reproduce). (2) §15.2 Results-README section order corrected to match the ground-truth `results/Heston/SBTS/README.md`: Header → What we generate → Results (A1–A34) → **Stylised Facts Diagnostic** → **Curve-shape metrics (B)** → Comparison with the paper → **Files** (previously "Comparison with paper" wrongly preceded the B table and the Stylised-Facts/Files sections were missing). Added the rule that the **paper-reproduction table goes BELOW the paper-comparison table** in the results README, with a worked TimeGAN example. (3) New **§15.3.1 Paper-Reimplementation README** subsection (8-section order, grounded in the existing TimeGAN/SBTS `paper_reimplementation/README.md`). (4) §15.4 enhanced with the exact new-method edits: **add one column + recompute Winner for BOTH table A and table B**, add one **row** to the methods list. No content removed — all changes additive/reordering only.
- 2026-07-19: **Documented the high-MSE / low-%err B anomaly (TimeGAN seed 2, log-return histogram).** MSE = 504.48 (SUM of funct 267.28 + der 100.55 + sec\_der 136.64) yet % err = 32.86%. Cause: seed 2 collapsed the log-return density into a too-tall central peak (gen peak 152.64 vs real 37.53, ≈4×); MSE is an **absolute squared** measure dominated by the few tallest bins (top-5 bins = 89.7% of the sum), while % err is a scale-free **MAPE** averaging bounded per-bin relative errors (median bin rel err 27.69%). High MSE + low % err ⟺ a large absolute mismatch concentrated at a few high-magnitude (peak-density) points. Not a bug — the two aggregations penalise a density-peak collapse differently.
- 2026-07-19: **Added §16 (Pitfalls — Process Errors to Avoid When Adding a Method).** Captured 5 mechanical/workflow errors made while integrating Fourier Flow (P1–P5) plus a General subsection, so future method additions don't repeat them. None were wrong benchmark numbers — all were file-layout / tool-schema / workflow mistakes.
- 2026-07-19: **§15.2 Section 6 → canonical 3-column paper-comparison table.** The "Comparison with the paper" section now uses **the paper's own metrics only** (not A1–A34) with one unified table: rows = paper metric, columns = **Paper (published, paper dataset)** | **Ours — <PaperDataset> (paper dataset reimpl)** | **Ours — Heston** (same paper metric applied to our 5-seed Heston pool). The Heston column must come from a committed JSON built by a reusable driver in `paper_reimplementation/metric/` (never hand-typed). Worked FF example (F-score 0.984/0.9920/0.9918; MAE 0.009/0.0084/0.0210). Old two-table form (TimeGAN/SBTS) noted as equivalent, no retrofit required. Added `methods/FourierFlow/paper_reimplementation/metric/heston_paper_metrics.py` + `results/heston_paper_metrics.json` (released `computeF1` + LSTM `computeMAE`, [0,1] MinMax, MAE `MAX_STEPS=127`).
- 2026-07-20: **§15.3 + §15.3.1 → "EXACT run path" mandate; §16 → P6–P10 (Diffusion-TS lessons).** After integrating Diffusion-TS (4th method), the user flagged that (a) the reproduction table did not match the paper and (b) several Heston metrics printed `0.0000 ± 0.0000` — looking like bugs. Root-caused all as honest artifacts, not code errors: hyperparameters are verbatim-correct vs `Config/stocks.yaml` (gap = single-draw/milestone-10-checkpoint protocol, not config); Correlational ≈ 6×10⁻⁹ on Heston is structural (cross-feature metric on a single feature — `tril_indices(1,1)`); Discriminative = exact 0 (degenerate GRU judge on smooth univariate); Predictive std ≈ 2×10⁻⁵ is a noise floor; Stocks Correlational `±0.0000` is a deterministic single-saved-draw re-score (paper re-samples 5×). Fixes: rewrote both Diffusion-TS paper tables to show real magnitudes (sci-notation) + `[1]–[4]` footnotes explaining each near-zero, and to state the hyperparameters-verbatim / protocol-gap framing (user chose "document honestly", no rerun). Strengthened §15.3 §9 and §15.3.1 §7 to REQUIRE the exact interpreter path, env vars, precise script + flags, exact input/output files, and which checkpoint/output produced each reported cell. Added §16 pitfalls P6 (near-zero rounded to 0.0000 reads as bug — show magnitude), P7 (cross-feature metric structurally ~0 on univariate — flag N/A, don't score), P8 (zero variance from re-scoring one saved draw vs re-sampling), P9 (reproduction gap ≠ hyperparameter error — verify config first, attribute to protocol), P10 (Bash cwd not guaranteed repo root).
- 2026-07-20: **CORRECTION to the Discriminative/Predictive Heston = 0 mechanism (root-caused to a metric-code bug, not "smooth data") + §16 P11.** The user challenged: paper Discriminative Heston = 0.0000 contradicts the benchmark's own **A18 GRU judge = 0.262 ± 0.158** on the *same* paths. Systematic debugging found the true root cause: `Utils/discriminative_metric.py` line 74 `hidden_dim = int(dim/2)` → for univariate Heston (`dim = 1`) `int(1/2) = 0` → `GRUCell(num_units=0)` → a **zero-capacity** judge → exactly 0.5 accuracy → `|0.5 − 0.5| = 0` on all 5 seeds (0-std is impossible for a working judge). The **identical** bug is in `Utils/predictive_metric.py` line 52, so Predictive 0.0653 is a constant-mean-output artifact, not a TSTR forecast. On Stocks (6 features) `int(6/2) = 3` works — univariate-specific. **This supersedes the earlier "degenerate GRU judge on smooth univariate" wording** in the 2026-07-20 entry above (that mechanism was wrong — it blamed data smoothness, not the `int(dim/2)=0` collapse). Fix (user chose "do both: document as degenerate AND state our comparable scores", no rerun): corrected notes [3]/[4] + the Heston caveat in **both** `methods/DiffusionTS/paper_reimplementation/README.md` and `results/Heston/DiffusionTS/README.md` to give the true `int(dim/2)=0` mechanism and cite **A18 disc GRU 0.262 ± 0.158 / A19 pred GRU 0.0549 ± 0.0002** (floored `hidden = max(8, n_features·8)`) as the real, comparable numbers; added §16 **P11** (post-hoc judge silently collapses to a zero-capacity network on univariate `dim=1` data — spot via exact-0/0-std + contradiction with another judge; document degeneracy + cite floored-dim judge, patch+rerun only if user asks). **Follow-up (same day):** the user then asked to re-check the *other* paper metrics' code — confirmed by reading + running each that **Correlational** is the SAME univariate degeneracy via a different path (`cacf_torch` + `tril_indices(1,1)` → only the standardized lag-0 self-correlation ≡ 1.0 for any real/fake → `|Δ|/10 = 1.19×10⁻⁸`; empirically real 0.99999905 vs fake 0.99999917), whereas **Context-FID is genuine** (TS2Vec embeds to 320-dim regardless of `input_dims=1`). So of the 4 paper metrics on univariate Heston, **3 degenerate** (Correlational, Discriminative, Predictive), only Context-FID is real. Reframed note [2] + the Correlational cell + Heston caveat in both DiffusionTS READMEs to "degenerate — real: **A21 ACF-abs 0.0201 ± 0.0030**" (the benchmark's temporal-autocorrelation metric, which stays defined on 1 feature), and strengthened §16 **P7** with the empirical proof, the P11 cross-reference, and the "cite the univariate-defined analog (A21/A22/A23)" cure.
- 2026-07-21: **Added §16 P12–P13 (TimeVQVAE process lessons — workflow ordering, not numbers).** After integrating TimeVQVAE (7th method) the user flagged two ordering errors and asked they be captured in §16: **P12** — racing past the §3.0 paper-reproduction **GATE** straight into the 5 Heston seeds instead of STOPping to show the explicit ours-vs-paper table for sign-off before spending Heston compute (cure: paper-vs-ours table is a hard barrier between Phase 1 reproduce and Phase 2 Heston; if it doesn't match, explain and STOP); **P13** — launching training with a fixed GPU/core layout **without first checking `nvidia-smi` + `htop`** for the actually-free capacity and the measured per-run footprint, so runs either collide with a colleague or under-parallelize (cure: inspect free GPUs/cores first, measure one run's memory/SM, co-locate small runs to reclaim idle capacity, launch in waves within the 2-GPU / 16-core / ~250 GiB limits, verify with `nvidia-smi` + `pgrep -af`). Both are workflow/ordering pitfalls; neither produced a wrong benchmark number. Also completed the TimeVQVAE structure audit (folder tree vs §1 + vs TimeVAE example: no junk tracked; deviations — no `timevqvae_torch.py`, split `stage1/stage2` loss CSVs, verbatim `reference/stage1+stage2` for the paper repro — all documented in the code + paper-reimpl READMEs) and confirmed every scored method's code README carries the §15.3 §9 "Exact run path" table and a `paper_reimplementation/README.md` with exact reference-code links (incl. TimeVAE's `reconstruction_wt` 3.0→8.0 tuning table with honest out-of-[0.5,3.5]-range caveat).
- 2026-07-21: **Metric modules unified into one `metrics/metrics.py`.** An external reviewer was confused by "B" appearing in two places — the live curve metrics **and** a dead legacy scalar `compute_stylized_metrics` (keys `B1_…`–`B12_…`) that no code imported. Root cause: those scalars had already been renamed into A25–A34, but the orphan function still sat in `stylized_metrics.py`. Fix: merged the B stylized-fact **curve** metrics (`compute_curve_metrics`, `aggregate_curve_metrics`, `CURVE_PLOTS`, `_curve_scores`, `_emit`, `_log_returns`/`_rolling_vol`/`_acf_mean` helpers) into `metrics/metrics.py` (renamed from `metrics_np.py`), **deleted** `metrics/stylized_metrics.py` (and with it the dead scalar B1–B12), and kept the two PyTorch scorers (`discriminative_score.py` A18, `predictive_score.py` A19) as separate files. Paper-reference headers preserved verbatim (Cont 2001; Ding/Granger/Engle 1993; Bollerslev 1986). Importers repointed to `from metrics import …` in `compute_all.py`, `recompute_curve_b.py`, `compute_perfect_recovery.py`. No metric formula, JSON key, or value changed — pure file reorganisation (verified: `py_compile` clean, all imports resolve, smoke test returns 36 curve keys + 6 plots). "B" now unambiguously means the 6 stylized-fact curves only. Updated §5 (three-file list + naming note) and §5.2/§15 filename references; historical changelog entries above keep their original `metrics_np.py`/`stylized_metrics.py` names (accurate for their date).
- 2026-07-22: **Test-set-everywhere split protocol + independent-draw floor + 3-subline B tables + `render_tables.py` as sole renderer.** Three coupled standard changes, all recomputed for the full 9-method suite and rebuilt in every README from disk:
  1. **Split protocol — "test set everywhere" (new §5.0).** The 8192×128 Heston dataset is drawn under three disjoint seeds and each has ONE fixed role: **train = seed 0** (generators were trained on it; it is NEVER scored), **test = seed 1** (the real reference every A1–A17/A20–A34, every B curve, every plot, and PS-MC query-prefix is measured against), **disc = seed 2** (supplies the "real" class for the two learned scorers ONLY — A18 discriminative and A19 predictive-TSTR — so the adversary/forecaster never touches the test set it is judged on). Generators are NOT retrained: since the paths are saved, only A18/A19's small judge/forecaster networks are (re)fit per seed. All five per-method result seeds are the five generated draws scored against this one test set.
  2. **Perfect floor → independent draw, non-zero (§5.4 rewrite).** The floor is no longer a row-shuffle/permutation of the real data (which forced marginals to 0). It is a **fresh independent Heston draw** — new seeds `1000+i` (`IND_SEED_BASE = 1000`), identical parameters — scored against the test set exactly like a generator. Every A and B floor is therefore a genuine **non-zero finite-sample noise floor**; the column value is the **mean across the 5 independent-draw seeds** (`perfect_floor_a()`/`perfect_floor_curve()`), identical across all methods. `methods/perfect_recovery/README.md` documents it in full and is the one place with no floor column of its own.
  3. **B tables → 3 sublines (MSE / % err / NRMSE), each mean-of-3.** Added **NRMSE** = sqrt(mean((L_gen−L_real)²))/(max|L_real|−min|L_real|+1e-12)×100. **% err** switched to a **scale-aware ε-floor MAPE** with ε = 1e-3·(max|L_real|+1e-12) (was the fixed `+1e-6`). All three measures combine funct/der/sec_der by **mean-of-3** (winner still decided by MSE). **Tail survival** is the one exception: its % err and NRMSE use the **funct curve only** (der/sec_der of a step-like survival curve blow the ratios up); its MSE stays mean-of-3. Raw JSON now stores 9 keys per plot (funct/der/sec_der × {mse, pct_err, nrmse}).
  4. **`metrics/render_tables.py` is the single source of truth for every table.** `--method <M>` prints the per-method markdown A/B/PS tables; `--which A|B|PS` prints the family-grouped HTML comparison tables + win-count comments. All numbers use its `fmt()` (4 sig-figs / sci-notation) so every README matches byte-for-byte — values are regenerated, never hand-typed. Current win-counts: **A of 36** — LS4 26, Fourier Flow 4, CSDI 3, Diffusion-TS 2, TimeVAE 1, others 0; **B of 6 (MSE)** — LS4 5, CSDI 1. Updated §5.2, §5.4, §8 Section 2–3, §13 checklist, §15.1 Section 2–3, §15.2, §15.4 (win-count denominator corrected 38 → 36).
- 2026-07-22: **B % err → fixed 1e-6 MAPE + funct-only for EVERY plot (REVERSES sub-item 3 of the entry above).** The user re-derived the % err convention: `dᵢ = mean(|L_gen − L_real| / (|L_real| + 1e-6)) × 100` — a proper function-level MAPE with a **fixed 1e-6 floor** and **one division** (verbatim: "% err row: for each list, dᵢ = mean(|L_g − L_real| / (|L_real| + 1e-6)) × 100, a proper MAPE — one division … tke this convention pls"). Two corrections vs the same-day sub-item 3: (a) the **scale-aware ε = 1e-3·(max|L_real|+1e-12)** floor is **reverted to the fixed `+1e-6`**; (b) **% err AND NRMSE are now funct-only for every plot**, not just tail survival — the der/sec_der of every curve have near-zero true values, so their relative errors explode into meaningless 10⁴-% figures; only the funct sub-metric carries a meaningful relative error. **MSE is unchanged** (mean-of-3 (funct+der+sec_der)/3; still decides the winner). Raw JSON still stores all 9 keys/plot (funct/der/sec_der × {mse, pct_err, nrmse}); aggregation (`aggregate_curve_metrics`) drops the der/sec_der %err+NRMSE. mean/std = mean and sample std across the 5 seeds. Fixed in `metrics/metrics.py` (`_pct` fixed-1e-6, `aggregate_curve_metrics` funct-only), recomputed via `metrics/recompute_curve_b.py` for all 9 methods + perfect_recovery, and every README B table + prose rebuilt from `curve_b_aggregate.json` through `render_tables.py`. Also removed the orphan `results/Heston/SBTS/plots/small_test.png` (a `T=32/N=200` dev-sanity artifact, unreferenced by any README). Updated §5.2, §8 Section 3, §15.1 Section 3, §15.2 Sections 5 & 3.

---

## 16. Pitfalls — Process Errors to Avoid When Adding a Method

> These are **mechanical / workflow** errors observed while adding real methods (TimeGAN → SBTS → Fourier Flow). None produced a wrong benchmark number, but each cost a round-trip. Read this list **before** writing any README or running any Bash when onboarding a new method. The cure for every one of them is the same: **verify against the disk before you assert.**

### P1 — Fabricated file names / CSV headers in a README before checking disk
**What happened:** Wrote a method README referencing `fourier_flow_torch.py` when the actual entry point was `train_heston.py`, and described a loss CSV header `step, train NLL loss` when the real header was `epoch,loss`.
**Why it's wrong:** READMEs are contracts. A reader who `cd`s in and runs the named script hits "file not found"; a reader who parses the CSV by the documented columns gets garbage.
**Cure:** Before naming ANY file or column in prose, run `ls <dir>` and `head -1 <file>` (or Read) and copy the exact strings. Never infer a filename from the method name or from a sibling method's convention.

### P2 — Assuming a subfolder layout that mirrors another method
**What happened:** Assumed `losses/` lived under `methods/FourierFlow/code/` (because that felt parallel to the code); it was actually a **sibling** of `code/` at `methods/FourierFlow/losses/`.
**Why it's wrong:** Each method's generator writes artifacts wherever its own `train_*.sh` points — the tree is NOT guaranteed identical across methods.
**Cure:** `find methods/<Method> -maxdepth 2 -type d` once, up front, and pin the real paths. Do not extrapolate one method's tree onto another.

### P3 — Chaining a hard `ls <missing-path>` inside a compound Bash command
**What happened:** A `cmd1 && cmd2 && ls methods/FourierFlow/code/losses/` compound aborted with exit code 2 because the final `ls` hit a non-existent path — killing the whole command and its earlier useful output.
**Why it's wrong:** In `&&` chains a single failure discards everything; a probe for a path you're unsure about should never gate real work.
**Cure:** Probe uncertain paths in a **separate, non-gating** call, or use `ls <path> 2>/dev/null || true`, or use `find` (which returns 0 with empty output rather than failing). Keep verification probes out of `&&` chains that do real work.

### P4 — Batch payload to a per-item tool (TaskUpdate schema)
**What happened:** Called `TaskUpdate({tasks:[{task_id,status},…]})` → `InputValidationError` (schema wants a single `{taskId, status}` per call, and the tool was **deferred** so its schema wasn't even loaded yet).
**Why it's wrong:** Deferred tools have no schema in context until fetched; guessing the shape (batch vs singular, `task_id` vs `taskId`) wastes a call.
**Cure:** For any deferred tool, `ToolSearch(query="select:<ToolName>")` first to load the real schema, then call it exactly as specified — one item per call unless the schema explicitly accepts an array.

### P5 — GateGuard fact-reset by an interposed Read
**What happened:** Stated the required Fact-Forcing Gate facts, then ran a `Read` to double-check an anchor, then issued the first `Edit` — the intervening Read **reset** the gate and the Edit was denied as "stale".
**Why it's wrong:** The gate requires the 5 facts and the guarded command in the **same turn with no intervening tool call**. Any tool call (even a Read) between the facts and the Edit invalidates them.
**Cure:** Do all your Reads/inspection FIRST. Only once you have the exact `old_string` anchor in hand, state the 5 facts and issue the `Edit`/`Bash` **immediately**, with nothing in between. If you must Read again, you must restate the facts again afterward.

### P6 — Rounding a genuine near-zero artifact to `0.0000 ± 0.0000` (reads as a hardcoded bug)
**What happened:** The Diffusion-TS Heston paper-metric table reported Correlational and Discriminative as
`0.0000 ± 0.0000` and Predictive std as `0.0000`. The user flagged this as "the metric predicts 0 and has
0 std — there's an issue", because a flat `0.0000 ± 0.0000` is indistinguishable from a placeholder / a
broken metric that always returns a constant.
**Why it's wrong:** Rounding hides the magnitude AND the mechanism. The real values were float32 machine
epsilon (Correlational ≈ 6×10⁻⁹), an exact structural 0 (Discriminative, degenerate judge), and a tight
noise floor (Predictive std ≈ 2×10⁻⁵) — all legitimate, but the reader can't tell a real ~0 from a bug ~0.
**Cure:** When a metric is genuinely ~0, **show the real magnitude in scientific notation** (`≈ 6×10⁻⁹`,
per-seed `1.2×10⁻⁸, 6.0×10⁻⁹, …`) and attach a one-line footnote saying *why* it's ~0 and that it is
**not** hardcoded. Never let a real near-zero and a placeholder look identical on the page.

### P7 — A cross-feature metric is structurally ~0 on univariate data — flag it, don't score it
**(Same family as P11 — univariate degeneracy — but a different code path; both confirmed by reading AND running the metric.)**
**What happened:** Correlational (`Utils/cross_correlation.CrossCorrelLoss`) returned ~machine-epsilon on
Heston. Root cause: it measures **cross-FEATURE** correlation; Heston here is `(N, T, 1)` — one feature —
so `torch.tril_indices(1,1) = [[0],[0]]` selects only the feature-with-itself term, and `cacf_torch`
**standardizes** each series first → the lag-0 self-correlation is **identically 1.0** for *any* real and
*any* fake. Empirically confirmed (seed 0): `cross_correl_real = 0.99999905`, `cross_correl_fake = 0.99999917`
→ `|Δ|/10 = 1.19×10⁻⁸` = float32 epsilon. It is mathematically incapable of being nonzero on a single feature.
**Why it's wrong:** Presenting it as a "score Diffusion-TS achieved" implies the method did something well;
in fact the metric is inapplicable. Same trap for any multivariate-only metric on univariate data.
**Cure:** Before reporting a metric on Heston, check whether it is *defined* for a single feature. If it
degenerates by construction (cross-feature corr, cross-asset copula, etc.), mark it **degenerate/N-A
(structural)** with the code reason, AND cite the benchmark's own **univariate-defined analog** as the real
number — here the temporal-autocorrelation metrics **A21 ACF-abs (0.0201 ± 0.0030)** / A22 / A23, which
measure lag-autocorrelation (defined on 1 feature) instead of cross-feature correlation. Don't present the
epsilon as a competitive result. Verify by reading **and running** the metric's source (here
`code/reference/Utils/cross_correlation.py`), not by trusting the number. **Note:** of Diffusion-TS's four
paper metrics on univariate Heston, **three degenerate** (Correlational here, Discriminative + Predictive in
P11); only **Context-FID stays genuine** because TS2Vec embeds to 320-dim regardless of `input_dims=1`.

### P8 — Zero run-to-run variance because a "5-run" metric re-scores ONE saved draw
**What happened:** The Stocks Correlational column showed `± 0.0000` across 5 runs. Cause:
`compute_torch_metrics.py` loads **one** saved sample file (`OUTPUT/stock/ddpm_fake_stock.npy`) and scores
it 5×; a deterministic metric on a fixed array yields 5 byte-identical values (ci95 = 0.0). The paper's
±0.001 comes from **re-sampling** the generator 5 times (5 fresh draws) — a different protocol.
**Why it's wrong:** A `± 0.0000` next to a paper's `± 0.001` looks like our error bars are fake, when in
fact we simply didn't resample. It also means the paper's std is *not reproducible* from our one file.
**Cure:** State explicitly in the reproduce section whether each column comes from **one saved draw**
(→ zero variance, deterministic) or from **N fresh generator draws** (→ real variance). If you want the
paper's CI you must re-sample; otherwise document that the zero std is a single-file artifact, not a claim.

### P9 — Reproduction gap is NOT automatically a hyperparameter error
**What happened:** Diffusion-TS Stocks Context-FID (0.202 vs paper 0.147) and Discriminative (0.091 vs
0.067) landed worse than the paper; the instinct was "check the hyperparameters". They were already
verbatim-correct vs the released `Config/stocks.yaml` (verified line by line). The gap was protocol: one
DDPM draw from one milestone-10 EMA checkpoint vs the paper's best-over-resampling/checkpoint-selection.
Predictive was a bulls-eye (0.0371 vs 0.036), proving the pipeline was faithful.
**Why it's wrong:** Chasing a hyperparameter "fix" when the config is already correct wastes time and can
introduce a real divergence from the paper.
**Cure:** When a reproduction is close-but-above, (1) diff every hyperparameter against the released config
first and record that they match, (2) check whether one metric is already exact (evidence the pipeline is
right), then (3) attribute the residual to draw/checkpoint-selection **protocol**, and say so honestly.
Document the gap; do not silently tune to close it (the user chose "document honestly" here).

### P10 — Bash tool cwd is not guaranteed to be the repo root
**What happened:** `cat results/…json` failed with "No such file or directory" because the Bash tool's
working directory was not `/home/tbasseras/benchmark`. The relative path only resolves from the repo root.
**Why it's wrong:** Silent cwd assumptions make read/verify commands fail spuriously and can, worse, write
to the wrong place.
**Cure:** Prefix repo-relative commands with `cd /home/tbasseras/benchmark && …`, or use absolute paths.
Don't assume the shell starts at the repo root.

### P11 — A post-hoc judge metric silently collapses to a zero-capacity network on univariate data
**What happened:** Diffusion-TS Heston **Discriminative = 0.0000 across all 5 seeds** (0 std) looked like
"real and fake are indistinguishable" but contradicted the benchmark's own A18 GRU judge (0.262 ± 0.158) on
the *same* paths. Root cause: `Utils/discriminative_metric.py` line 74 `hidden_dim = int(dim/2)`. For
univariate Heston (`dim = 1`), `int(1/2) = 0` → `GRUCell(num_units=0)` → a **zero-capacity** judge with no
hidden state → constant output → **exactly 0.5** accuracy on the balanced split → `|0.5 − 0.5| = 0` every
seed. The **identical** bug is in `Utils/predictive_metric.py` line 52 (`hidden_dim = int(dim/2)`), so
Predictive = 0.0653 is a constant-mean-output artifact, not a TSTR forecast. On Stocks (6 features)
`int(6/2) = 3` works — the collapse is **univariate-specific**.
**Why it's wrong:** An exact 0 with exactly 0 std across independent seeds is statistically impossible for a
working stochastic judge — it is a broken network, not a quality result. Reporting it as "Discriminative =
0.0" claims the generator is perfect when the judge never ran.
**How to spot it:** (1) exact 0.0000 with 0.0 std across all seeds; (2) it contradicts another judge on the
same data (A18/A19 here); (3) the metric was designed/tested on multivariate data (`dim ≥ 2`). Any of these
⇒ read the metric source for a `hidden_dim = int(dim/2)`, `int(dim*k)`, or similar dimension-derived size
that can floor to 0 on `dim = 1`.
**Cure:** Do **not** silently trust the degenerate 0. Document it as a **metric-code degeneracy on univariate
data** (cite the exact line), and report the benchmark's own floored-hidden-dim judge (A18 disc / A19 pred,
`hidden = max(8, n_features·8)`) as the real, comparable number. Only patch the reference metric's
`hidden_dim` floor + re-run if the user explicitly asks (here the user chose: document degeneracy AND state
the A18/A19 comparable scores — no rerun). Same trap for any `int(dim/·)`-sized post-hoc network on 1-D data.

### P12 — Racing past the paper-reproduction GATE straight into the 5 Heston seeds
**What happened:** After finishing a method's `paper_reimplementation/` (reproducing the paper's own
metric on the paper's own dataset), the next action was to immediately launch the full Heston pipeline —
train 5 seeds, compute all metrics, write READMEs — **without stopping to show the ours-vs-paper table**.
The paper reproduction is the one checkpoint that proves the port is faithful; skipping the review means a
silent reproduction gap (wrong hyperparameter, wrong evaluator, wrong scaling) propagates into all 5 Heston
seeds and every downstream README before anyone has confirmed the method even matches the paper.
**Why it's wrong:** §3.0 makes paper-reproduction a **GATE, not a milestone**. The whole point is that a
human validates "our number ≈ the paper's number" *before* any Heston compute is spent. Launching all
algorithms first inverts the order: if the repro was actually off, you have thrown away hours of A100 time
and produced READMEs that must all be redone.
**Cure:** The moment `paper_reimplementation/` produces its numbers, **STOP** and present the explicit
ours-vs-paper table (paper value | our value | verdict, per metric) and wait for the user's go-ahead. Do
**not** train a single Heston seed until the reproduction is signed off. If the repro does not match, explain
the discrepancy and STOP — do not "proceed and fix later". Treat the paper-vs-ours table as a hard barrier
between Phase 1 (reproduce) and Phase 2 (Heston).

### P13 — Launching training without first checking every GPU and core for the optimal parallelization
**What happened:** Kicked off the method's training run (or all algorithms) with a fixed
`CUDA_VISIBLE_DEVICES` / core layout **without first inspecting `nvidia-smi` and `htop`** to see what was
actually free. This either (a) collided with another user's job on a "free" GPU, or (b) left capacity on the
table — e.g. running one seed at a time when two GPUs were idle, or pinning 8 cores when the runs were tiny
enough to co-locate two per GPU (as TimeVQVAE's ECG5000 repro did: ~4.6 GB / ~60 % SM each).
**Why it's wrong:** The machine is **shared** and the hard limits (max 2 GPUs, 16 cores, ~250 GiB RAM) are a
ceiling, not a target. Blindly launching wastes wall-clock (under-parallelizing) or steps on a colleague
(over-claiming). The right degree of parallelism depends on the *observed* free capacity and the *measured*
per-run footprint, neither of which you know without looking.
**Cure:** **Before** launching, run `nvidia-smi` (which GPUs are free, how much memory) and `htop` (core/RAM
load); pick GPUs 0/3 (or whatever is idle) and a core range that doesn't fight another job; measure one run's
GPU-memory + SM footprint and, if small, co-locate runs to reclaim idle capacity — always within the 2-GPU /
16-core / ~250 GiB limits. Launch seeds in **waves** sized to the free hardware (e.g. 2 GPUs per wave), and
verify with `nvidia-smi` + `pgrep -af` that every run actually started. Never assume the machine is idle.

### General — applies to every method addition
- **Verify-before-claim:** every filename, path, CSV header, and metric value written into a README must be read back from the actual artifact (`grep`/`head`/Read), never recalled from memory or copied from a sibling method.
- **Stage by name:** `git add <explicit files>` only — never `git add -A` / `git add .` (avoids committing `.omc/`, `__pycache__/`, stray large files).
- **`.omc/` is gitignored** at repo root; nested `.omc/` dirs under a method folder are therefore auto-skipped — confirm with `git status --porcelain` before committing.
- **Largest-file check:** `find . -type f -size +50M` before commit; GitHub hard-rejects >100 MB. The 8192×128 float64 arrays are ~8 MB each — fine — but weights/checkpoints can be large.
- **Commit in the benchmark repo ONLY** (`/home/tbasseras/benchmark`), never the `/home/tbasseras` home repo (it holds `.ssh/` and credentials).
