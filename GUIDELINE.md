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
            ├── seed_0_metrics.json       all A1–A16 values for seed 0
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

## 5. The 16 Metrics (A1–A16)

**All metrics are pre-implemented** in `metrics/metrics_np.py`,
`metrics/discriminative_score.py`, and `metrics/predictive_score.py`.
Do NOT reimplement them.

Run everything with:
```bash
cd metrics
/home/tbasseras/gpu-venv/bin/python compute_all.py --method <Method> --dataset Heston --seeds 5
```

### 5.1 Input format expected by `compute_all.py`

| Variable | Path | Shape | Scale |
|----------|------|-------|-------|
| Real prices | `dataset/heston_S_8192x128.npy` | (8192, 128) | original (~100) |
| Real variance | `dataset/heston_v_8192x128.npy` | (8192, 128) | original (~0.04) |
| Fake prices | `methods/<Method>/generated_paths/seed_{i}/generated_paths_8192x128.npy` | (8192, 128) | original |

`compute_all.py` infers the fake path from `--method` and `--dataset` automatically.

### 5.2 Full metric list

| ID | Name | Category | Direction | Perfect |
|----|------|----------|-----------|---------|
| A1 | Path MMD² | Distribution | ↓ | 0 |
| A2 | Terminal MMD² | Distribution | ↓ | 0 |
| A3 | Increment MMD² | Distribution | ↓ | 0 |
| A4 | Volatility MMD | Distribution | ↓ | 0 |
| A5 | Terminal SWD | Distribution | ↓ | 0 |
| A6 | Path SWD | Distribution | ↓ | 0 |
| A7 | Covariance Error (%) | Statistics | ↓ | 0 |
| A8 | Mean RMSE | Statistics | ↓ | 0 |
| A9 | Return Std Error | Statistics | ↓ | 0 |
| A10 | Kurtosis Error | Statistics | ↓ | 0 |
| A11 | ACF Error (abs returns) | Temporal | ↓ | 0 |
| A12 | ACF Error (sq returns) | Temporal | ↓ | 0 |
| A13 GRU | Discriminative Score GRU | Adversarial | ↓ | 0 |
| A13 MLP | Discriminative Score MLP | Adversarial | ↓ | 0 |
| A14 GRU | Predictive Score GRU (TSTR) | Predictive | ↓ | baseline |
| A14 MLP | Predictive Score MLP (TSTR) | Predictive | ↓ | baseline |
| A15 Corr | Teacher-Sigma Correlation | Heston-specific | ↑ | 1 |
| A15 RMSE | Teacher-Sigma RMSE | Heston-specific | ↓ | 0 |
| A16 | Tail Survival Error | Fat-tail | ↓ | 0 |

Total: 19 numbers (some IDs have two variants).

### 5.3 Output files produced by `compute_all.py`

All land in `results/Heston/<Method>/`:

| File | Contents |
|------|---------|
| `seed_{i}_metrics.json` | All 19 metric values for seed i |
| `metrics_summary.json` | Mean ± std across 5 seeds, plus per_seed array |
| `plots/heston_diagnostics.png` | 8-panel stylised facts figure (seed 0 vs real) |
| `plots/disc_classifier_loss.png` | A13 BCE training loss, GRU + MLP, 5 seeds |
| `plots/pred_score_loss.png` | A14 MAE training loss, GRU + MLP, 5 seeds |

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

1. **Embedding**: prefix (steps 0–63) → 22D multi-scale log-return vector via eq. (13) of arXiv:2308.01486, α=1.15, β=0.9, lags {1,2,3,4,5,6,7,8,9,10,12,14,16,18,21,24,28,32,37,43,50,57}
2. **KNN retrieval**: L2 distance in that 22D space → 77 nearest generated paths (sklearn NearestNeighbors, `metric="euclidean"`)
3. **Price anchoring**: each retrieved fake future is scaled by `S_real(t=63) / S_fake(t=63)`
4. **Weighting**: Uniform (flat 1/77) and Gaussian (per-query η = η̃·‖h(x̃)‖, η̃ = median_dist/median_norm, calibrated from data)
5. **Evaluation**: CRPS, MAE, RMSE at H=32 (steps 64–95) and H=64 (steps 64–127)
6. **Baseline**: naive random walk (repeat last prefix price) — CRPS_h32=3.732, CRPS_h64=5.301

The KNN is a plain nearest-neighbour scan (not a combinatorial search). For each of the 8 192 real query paths, all 8 192 fake path embeddings are scanned and the 77 with smallest L2 distance are returned. Cost: O(8192 × 22) per query.

---

## 7. Figures

### 7.1 Heston Diagnostics — 8-panel figure

Compares real Heston (seed 0) vs generated paths (seed 0) on 8 stylised facts.
Produced automatically by `compute_all.py`, saved to `results/Heston/<Method>/plots/heston_diagnostics.png`.

Layout: 2 rows × 4 columns, figure size 18×8 inches, 150 dpi.

| Panel | Title | Content |
|-------|-------|---------|
| Row 0, Col 0 | Real Sample Paths | 5 random real paths, x=step (0–127), y=price |
| Row 0, Col 1 | Generated Sample Paths | 5 random generated paths, same axes |
| Row 0, Col 2 | Return Distribution | Overlaid KDE of all log-returns, real (blue) vs generated (orange) |
| Row 0, Col 3 | QQ Plot | Quantile-quantile of generated vs real log-returns |
| Row 1, Col 0 | ACF of \|r\| | Mean ACF of absolute returns, lags 1–20, real vs generated |
| Row 1, Col 1 | ACF of r² | Mean ACF of squared returns, lags 1–20, real vs generated |
| Row 1, Col 2 | Rolling Vol Histogram | Distribution of rolling σ (window=5), real vs generated |
| Row 1, Col 3 | Tail Survival (log-log) | P(\|r\| > x) for x ∈ [0.01, 0.2], real vs generated |

Colours: real = blue, generated = orange. Always include a legend on each panel.

### 7.2 Discriminative Classifier Loss (A13)

Saved to `results/Heston/<Method>/plots/disc_classifier_loss.png`.
- Two subplots side by side: GRU discriminator / MLP discriminator
- X-axis: training step 0–2000, logged every 50
- Y-axis: BCE loss
- All 5 seeds overlaid
- Horizontal dashed reference line at ln(2) ≈ 0.693 (random-chance level)

### 7.3 Predictive Score Loss (A14)

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

#### Section 2 — Metrics table

```markdown
## Metrics — mean ± std across 5 seeds

| ID | Metric | Category | Dir | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect |
|----|--------|----------|-----|-----------|--------|--------|--------|--------|--------|---------|
| A1 | Path MMD² | Distribution | ↓ | X.XXXX ± X.XXXX | X.XXXX | ... | 0 |
...
| A16 | Tail Survival Error | Fat-tail | ↓ | ... | **0** |
```

Rules:
- 4 decimal places everywhere
- Bold the Perfect column for A13 (perfect = **0**) and A15 Corr (perfect = **1**)
- Use `baseline` (not bold) for A14 Perfect column

Include the footnotes block verbatim:
```markdown
> **A13 discriminative score**: `|accuracy − 0.5|` on held-out test set (80/20 split).
> 0 = indistinguishable from real. 0.5 = perfectly separated (bad generator).
>
> **A14 predictive score**: TSTR MAE — predictor trained on *synthetic*, evaluated on *real*.
>
> **A15 sigma**: Heston-specific. Compares inferred instantaneous vol from generated paths
> against the true variance paths.
>
> **A16 tail survival error**: RMS of survival probability difference at quantiles {0.90, 0.95, 0.99}.
> Tests fat-tail reproduction. 0 = perfect. Lower is better.
```

#### Section 3 — Stylised Facts Diagnostic

```markdown
## Stylised Facts Diagnostic (Heston vs <Method>, seed 0)

Eight-panel comparison: sample paths, return distribution, QQ plot, ACF of |returns|,
ACF of squared returns, rolling vol histogram (window=5), tail survival (log-log).

![Heston Diagnostics](../../results/Heston/<Method>/plots/heston_diagnostics.png)
```

#### Section 4 — Training Loss

```markdown
## <Method> Training Loss (5 seeds)

<Describe training phases (if any), total steps, what each loss component measures.>

![<Method> Training Loss](losses/loss_convergence.png)
```

#### Section 5 — A13

```markdown
## A13 — Discriminative Classifier Training Loss

BCE loss during GRU and MLP classifier training (2 000 steps, logged every 50 steps).
A value near ln(2) ≈ 0.693 means the classifier cannot distinguish real from fake.

![Discriminative Classifier Loss](../../results/Heston/<Method>/plots/disc_classifier_loss.png)
```

#### Section 6 — A14

```markdown
## A14 — Predictive Score Training Loss (TSTR)

MAE loss during GRU and MLP predictor training on *synthetic* data (5 000 steps, logged every 100 steps).

![Predictive Score Loss](../../results/Heston/<Method>/plots/pred_score_loss.png)
```

#### Section 7 — Path Shadowing MC

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

#### Section 8 — File layout

Show the exact folder tree of `methods/<Method>/` (fill in actual file names).

#### Section 9 — Reproduce

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
2. **Metrics table**: same 19-number table as §8 Section 2 (keep in sync)
3. **Per-seed breakdown**: for the 4 most diagnostic metrics (A1 Path MMD², A13 GRU, A15 Corr, A16), list each seed's value and note which seed is weakest and why
4. **Observations**: 3–5 bullet points on what the metrics reveal — strengths, weaknesses, unexpected findings
5. **File index table**

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
feat(sbts): compute A1–A16 metrics, heston_diagnostics figure
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
  [ ] results/Heston/<Method>/seed_{0..4}_metrics.json  — 19 values each
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
  [ ] methods/<Method>/README.md  — all 9 sections present, no placeholder text
  [ ] methods/<Method>/code/README.md  — paper, variant, fixes, architecture, hyperparams
  [ ] results/Heston/<Method>/README.md  — metrics table, per-seed breakdown, observations
  [ ] results/Heston/<Method>/path_shadowing/README.md  — method detail, results, figures

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

**Open questions — Theo must answer before implementation begins:**

| # | Question | Why it blocks |
|---|----------|--------------|
| SB-Q1 | Use `sbts_uni.py` (non-Markovian) or `sbts_uni_markovian.py` (Markovian)? | Heston price path is non-Markovian (hidden vol), but Markovian may train faster with similar quality |
| SB-Q2 | Does SBTS output price paths or log-return paths? | Determines denormalisation step before saving generated_paths_8192x128.npy |
| SB-Q3 | Is our `gpu-venv` compatible with `torch==2.2.2`? Or do we need a separate `sbts-venv`? | Blocks training setup |
| SB-Q4 | What are the paper's canonical hyperparameters for the SB model on a 128-step sequence? | The notebook uses their own datasets; we need to adapt to seq_len=128 |
| SB-Q5 | Does the SB training have an exploitable loss signal to log every 100 steps? | Required for `losses/seed_{i}_losses.csv` |
