# Deep-MKV-TS — Paper Reimplementation (Heston, Table 1)

- **Paper** — *Deep McKean–Vlasov Time Series* (Deep-MKV-TS), ICAIF submission.
  Local copy: [`Deep-MKV-TS_ICAIF.pdf`](Deep-MKV-TS_ICAIF.pdf).
- **Official code** — the authors' package, mirrored verbatim at
  [`../code/reference/`](../code/reference/). See [`../code/README.md`](../code/README.md)
  for the vendoring details and the single local edit.
- **This run** — [`metric/run_reproduction.sh`](metric/run_reproduction.sh)
  (seeds 0, 1, 3, 4) + [`metric/run_seed2.sh`](metric/run_seed2.sh) (seed 2), two
  A100-SXM4-80GB, 8 CPU cores each, ≈ 24–33 min wall-clock per seed.
- **Verdict** — **5 / 5 metrics within tolerance.** Aggregate ratio (ours ÷ paper, mean over
  the five columns) = **0.994**. Table in §4.

---

## ⚠️ Reproduction caveats

**No port was needed.** Unlike TimeGAN (TF1, no CUDA-13 build) or SBTS, the Deep-MKV-TS
reference package is modern PyTorch and runs as-is on this machine. We invoke the authors'
own training script, not a re-implementation. The only edit to the vendored tree is a
dataset-path resolver (documented as Fix 1 in [`../code/README.md`](../code/README.md)),
which changes no numerics.

Three things a reader must know before reading the table.

### Caveat 1 — the published Reference row used a *different* reference model

Paper §2.1 defines exactly one reference model: **Guyon–Lekeufack (2023)**
(*Volatility is (Mostly) Path-Dependent*, paper reference `[11]`). The recursion in eq. (2),

```
V_{i+1} = exp(-lambda_{V,a} * dt_i) * V_i  +  lambda_{V,a} * (v_i^ref)^2 * dt_i
```

is driven by `(v^ref)²`, which pins the codebase option to
`--reference-kind guyon_lekeufack_structural_likelihood` with
`activity_update="structural_variance"`. That is what we run, on every row, everywhere.

However, the codebase also ships a `local_gaussian` reference, and **that** is what
reproduces the *published* Reference row to four decimals (aggregate ratio 0.998 against the
published Reference numbers, versus 0.609 for Guyon–Lekeufack). In other words the published
Table 1 Reference row appears to have been generated with `local_gaussian`, even though §2.1
describes Guyon–Lekeufack.

**Decision (author's instruction, 2026-08-16):** use Guyon–Lekeufack everywhere, because that
is what the paper *describes*. Consequence: **our Reference row is not comparable to the
published Reference row and is not scored.** It is reported as context only. The
Guyon–Lekeufack reference is simply a stronger starting point — it beats the published
Reference row on all five columns — so the Deep-MKV-TS row built on top of it is a *harder*,
not easier, target.

This choice also happens to be the better-performing one for the scored row:
Guyon–Lekeufack gives Deep-MKV-TS aggregate **0.994**, `local_gaussian` gives **1.205**.

### Caveat 2 — the paper's seed set is {0, 1, 3, 4}, the benchmark's is {0, …, 4}

The paper reports **medians over four seeds: 0, 1, 3, 4**. Seed 2 is not in the paper's set.
The table in §4 therefore uses **exactly the paper's four seeds** so the comparison is
like-for-like. GUIDELINE §4 separately requires the benchmark 5-seed set 0–4, so
[`metric/run_seed2.sh`](metric/run_seed2.sh) adds seed 2 with a byte-identical flag block;
seed 2 feeds the Heston benchmark tables in `results/Heston/Deep-MKV-TS/`, **not** this
paper-comparison table.

### Caveat 3 — K = 2500 is a checkpoint, not an early stop

The paper reports **K = 2500**. We train 3000 steps and score the step-2500 checkpoint. This
is bitwise identical to stopping at 2500 because **there is no learning-rate scheduler
anywhere in the codebase** — the optimiser state at step 2500 is independent of whether the
loop continues afterwards. Pinned in [`metric/PROTOCOL.json`](metric/PROTOCOL.json) as
`locked_reporting_checkpoint: 2500`.

---

## 1. Paper metrics (as defined in the paper)

Table 1 reports five columns, all **lower is better**, all computed between the generated
path law and the held-out real path law.

| Metric | Definition | Direction | Key in `metrics.json` |
|---|---|:---:|---|
| **SWD** | Sliced Wasserstein distance between the full path clouds, normalised | ↓ | `path_swd_normalized` |
| **RV W₁** | 1-Wasserstein distance between realised-volatility distributions, normalised | ↓ | `realized_volatility_w1_normalized` |
| **\|r\| ACF** | RMSE between the mean autocorrelation functions of absolute returns | ↓ | `absolute_return_acf_rmse` |
| **Early-future** | Error in the correlation between early-window RV and future-window RV (the "volatility persists" structure) | ↓ | `prefix_future_rv_correlation_error` |
| **MDD W₁** | 1-Wasserstein distance between maximum-drawdown distributions, normalised | ↓ | `maximum_drawdown_w1_normalized` |

**How the paper reports them:** the *median* over the four training seeds — not the mean, not
mean ± std. We match that exactly. There is no re-sampling variance to average over: each
seed produces one 8192-path bank (`bank_seed = 70000 + seed`), scored once.

**Tolerance used here:** within **25 % of the paper value, or 0.005 absolute, whichever is
wider**. The absolute floor matters because four of the five columns sit below 0.03, where a
25 % band is tighter than the seed-to-seed noise — see the per-seed table in §4, where
Early-future spans 0.011 → 0.041 at the *same* configuration.

---

## 2. Hyperparameters

Every value is passed explicitly on the command line; nothing is inherited from a config
file. The Source column cites where in the paper the value comes from.

| Parameter | Value | Source | CLI flag |
|---|---|---|---|
| Task | Heston | §4, Table 1 | `--task heston` |
| Reference model | Guyon–Lekeufack structural likelihood | §2.1, ref `[11]` | `--reference-kind guyon_lekeufack_structural_likelihood` |
| Reference activity update | structural variance, `(v^ref)²`-driven | eq. (2) | (default for this kind) |
| Reference calibration split | 80 % fit / 20 % select | §2.1 | (default) |
| Reference σ clip | `[1e-3, 0.6]` | §2.1 | (default) |
| Physical drift | frozen at the fitted reference drift | §2 | `--fitted-reference-drift-only` |
| Drift-adjoint weight | 0 | §2 (volatility-only correction) | `--adjoint-weight 0` |
| Noise-adjoint weight | 1 | §2 (volatility-only correction) | `--adjoint-noise-weight 1` |
| Running cost | specific entropy | §3 | (default) |
| Entropy weight η | 1 | §3 | `--eta 1` |
| λ scale (path) | 50 | Table 6 / App. B | `--lambda-scale 50` |
| κ scale (vol) | 100 | Table 6 / App. B | `--kappa-scale 100` |
| Learning rate | 2 × 10⁻³ | Table 6 / App. B | `--lr 0.002` |
| Gradient clip (norm) | 5 | Table 6 / App. B | `--grad-clip-norm 5` |
| Bank size | 8192 | matches dataset size | `--bank-size 8192` |
| Sample batch size | 2048 | App. B | `--sample-batch-size 2048` |
| Joint-volatility weight | 0 | Table 6 | `--joint-volatility-weight 0 --source-joint-volatility-weight 0` |
| \|r\| ACF weight | 0.25 | Table 6 | `--abs-return-acf-weight 0.25` |
| r² ACF weight | 0.125 | Table 6 | `--squared-return-acf-weight 0.125` |
| Steps trained | 3000 | App. B | `--source-steps 3000` |
| **Step reported (K)** | **2500** | App. B (K = 2500) | `--source-checkpoint-steps 500 1000 1500 2000 2500 3000` |
| Solver | online | App. B | `--solver online` |
| Network | 1-layer GRU hidden 96 + two `Linear(96,96)→Linear(96,1)` heads, 47 330 params | read off the checkpoint | (architecture default) |
| Batch size / target batch | 256 / 256 | checkpoint `training` dict | (default) |
| Weight decay | 1 × 10⁻⁵ | checkpoint `training` dict | (default) |
| Conditional-expectation target | ridge, λ = 1e-3, 1 fold | checkpoint `training` dict | (default) |

**Wall-clock:** 32.8 / 24.0 / 32.9 / 23.3 min for seeds 0 / 1 / 3 / 4 (the spread is GPU
contention, not configuration). Peak GPU memory ≈ 6 GB.

Fitted Guyon–Lekeufack parameters are recorded per seed in `../weights/seed_{i}_config.json`.
Seed 0: trend half-lives 12.88 / 74.26 steps, activity half-lives 11.78 / 50.34 steps,
calibration NLL −1.1466, validation NLL −1.1542 — validation better than calibration, so the
frozen reference is not overfitting its 80 % fit split.

---

## 3. Dataset

The paper's headline Table 1 experiment **is** a Heston experiment, so the paper dataset and
the benchmark dataset coincide. Per GUIDELINE §2 the shared dataset is **not duplicated
here** — [`dataset/`](dataset/) holds a pointer README only.

| Split | File (under `benchmark/dataset/Heston/`) | Shape | dtype | Role here |
|---|---|---|---|---|
| train | `heston_S_8192x128.npy` | (8192, 128) | float64 | the target law the model is fit to |
| **disc** | `heston_S_disc_8192x128.npy` | (8192, 128) | float64 | **the split every number in §4 is scored on** |
| valdisc | `heston_S_valdisc_8192x128.npy` | (8192, 128) | float64 | hyperparameter search only, never reported |
| test | `heston_S_test_8192x128.npy` | (8192, 128) | float64 | **never touched by this method** |

Generation parameters (from `dataset/Heston/generate_heston.py`, do not modify):
`μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250, N=8192, T=128`.
The splits are disjoint draws under different seeds.

**Split discipline is machine-enforced** by [`metric/PROTOCOL.json`](metric/PROTOCOL.json):

```json
"evaluation_split":             "dataset/Heston/heston_S_disc_8192x128.npy",
"training_split":               "dataset/Heston/heston_S_8192x128.npy",
"locked_reporting_checkpoint":  2500,
"baseline_seeds":               [0, 1, 3, 4],
"selection_scope":              "validation_only",
"test_split_access_authorized": false,
"test_split_loaded":            false
```

The trainer reads this manifest and refuses to score against a split it does not declare.
Hyperparameter search ran against `valdisc`; the winning configuration was scored on `disc`
**exactly once**, which is the table below. `heston_S_test_8192x128.npy` was never opened by
any run in this directory.

---

## 4. Results — ours vs paper

**Median over the paper's four seeds (0, 1, 3, 4), 8192 paths, scored on
`heston_S_disc_8192x128.npy`. Lower is better.**
Tolerance: within 25 % of the paper value or 0.005 absolute, whichever is wider.

| Row | Metric | **Ours (median, 4 seeds)** | Paper (Table 1) | Δ | Ratio | Verdict |
|---|---|---:|---:|---:|---:|:---:|
| Deep-MKV-TS | SWD | **0.0737** | 0.062 | +0.0117 | 1.19× | matches ✓ |
| Deep-MKV-TS | RV W₁ | **0.0128** | 0.014 | −0.0012 | 0.91× | matches ✓ (better) |
| Deep-MKV-TS | \|r\| ACF | **0.0132** | 0.016 | −0.0028 | 0.82× | matches ✓ (better) |
| Deep-MKV-TS | Early-future | **0.0178** | 0.018 | −0.0002 | 0.99× | matches ✓ (better) |
| Deep-MKV-TS | MDD W₁ | **0.0232** | 0.022 | +0.0012 | 1.06× | matches ✓ |
| | **Aggregate (mean ratio)** | **0.994** | 1.000 | | | **matches ✓** |

**5 / 5 metrics within tolerance. 3 / 5 columns strictly better than published.**

Reference row — **context only, not scored** (Caveat 1: a different reference model than the
one that produced the published Reference row):

| Row | Metric | Ours (GL, median 4 seeds) | Paper (published, `local_gaussian`) | Δ |
|---|---|---:|---:|---:|
| Reference | SWD | 0.0381 | 0.060 | −0.0219 |
| Reference | RV W₁ | 0.0566 | 0.089 | −0.0324 |
| Reference | \|r\| ACF | 0.0484 | 0.068 | −0.0196 |
| Reference | Early-future | 0.1019 | 0.214 | −0.1121 |
| Reference | MDD W₁ | 0.0287 | 0.059 | −0.0303 |

The learned correction still buys a large improvement over its own reference: Deep-MKV-TS
cuts RV W₁ by 4.4×, |r| ACF by 3.7×, Early-future by 5.7× and MDD W₁ by 1.2× relative to the
frozen Guyon–Lekeufack model it starts from. SWD is the one column where the correction
*costs* something (0.0381 → 0.0737): the entropy-penalised volatility correction trades raw
path-cloud proximity for volatility structure, which is exactly the intended trade.

### Per-seed values (why the median, and how much noise there is)

**Deep-MKV-TS**

| Seed | SWD | RV W₁ | \|r\| ACF | Early-future | MDD W₁ |
|---|---:|---:|---:|---:|---:|
| 0 | 0.0528 | 0.0210 | 0.0230 | 0.0180 | 0.0103 |
| 1 | 0.0655 | 0.0127 | 0.0128 | 0.0108 | 0.0218 |
| 3 | 0.0818 | 0.0129 | 0.0064 | 0.0413 | 0.0423 |
| 4 | 0.0839 | 0.0112 | 0.0135 | 0.0177 | 0.0246 |
| **median** | **0.0737** | **0.0128** | **0.0132** | **0.0178** | **0.0232** |

**Reference (Guyon–Lekeufack, frozen)**

| Seed | SWD | RV W₁ | \|r\| ACF | Early-future | MDD W₁ |
|---|---:|---:|---:|---:|---:|
| 0 | 0.0368 | 0.0595 | 0.0512 | 0.1202 | 0.0344 |
| 1 | 0.0328 | 0.0574 | 0.0492 | 0.1123 | 0.0276 |
| 3 | 0.0395 | 0.0549 | 0.0476 | 0.0883 | 0.0222 |
| 4 | 0.0555 | 0.0559 | 0.0475 | 0.0915 | 0.0298 |
| **median** | **0.0381** | **0.0566** | **0.0484** | **0.1019** | **0.0287** |

> **Read the noise before reading the margin.** At a *fixed* configuration, |r| ACF spans
> 0.0064 → 0.0230 (3.6×) and Early-future spans 0.0108 → 0.0413 (3.8×) across seeds, and the
> per-seed aggregate ratio ranges roughly 0.87 → 1.10. A single-seed "improvement" smaller
> than that is seed luck, not a result. This is exactly why
> [`metric/rank_confirm.py`](metric/rank_confirm.py) ranks candidate configurations on the
> median over seeds that were **never used to select** (1, 3, 4) rather than on the seed the
> shortlist was drawn from (0).

---

## 5. How to reproduce (EXACT run path, mandatory)

Everything below is what was actually executed. Substitute your clone root for
`/home/tbasseras/benchmark` and your interpreter for `/home/tbasseras/gpu-venv/bin/python`.

### 5.0 Environment

```bash
python -m venv ~/gpu-venv
~/gpu-venv/bin/pip install torch numpy scipy matplotlib
```

Python ≥ 3.11 (`../code/reference/pyproject.toml`), PyTorch with CUDA. The reference package
is **not** pip-installed — `PYTHONPATH` is sufficient. No TensorFlow, no second venv: unlike
TimeGAN and SBTS, every number in this directory comes from one environment.

### 5.1 Check the machine (GUIDELINE §4.1 — hard rule)

```bash
nvidia-smi     # GPUs 0 and 1 must be free
htop           # RAM below 50 %
```

Caps enforced throughout: **2 GPUs max, 8 cores per job, `OMP_NUM_THREADS=8`.**

### 5.2 One seed, verbatim

```bash
CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
PYTHONPATH=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference/src:/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference/experiments:/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference/experiments/scripts \
taskset -c 16-23 \
/home/tbasseras/gpu-venv/bin/python \
  /home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference/experiments/scripts/run_matched_control_synthetic_validation.py \
  --task heston \
  --run-dir /home/tbasseras/benchmark/methods/Deep-MKV-TS/paper_reimplementation/runs/seed_0 \
  --device cuda:0 \
  --protocol-manifest /home/tbasseras/benchmark/methods/Deep-MKV-TS/paper_reimplementation/metric/PROTOCOL.json \
  --seed 0 \
  --bank-size 8192 --sample-batch-size 2048 \
  --lambda-scale 50 --kappa-scale 100 --eta 1 \
  --lr 0.002 --grad-clip-norm 5 \
  --reference-kind guyon_lekeufack_structural_likelihood \
  --joint-volatility-weight 0 --source-joint-volatility-weight 0 \
  --abs-return-acf-weight 0.25 --squared-return-acf-weight 0.125 \
  --source-only --source-steps 3000 --solver online \
  --source-checkpoint-steps 500 1000 1500 2000 2500 3000 \
  --fitted-reference-drift-only \
  --adjoint-weight 0 --adjoint-noise-weight 1 \
  --path-derivative-backend autograd \
  --drift-adjoint-backend analytical_reference \
  > /home/tbasseras/benchmark/methods/Deep-MKV-TS/paper_reimplementation/runs/seed_0/console.log 2>&1
```

**Inputs consumed:** `dataset/Heston/heston_S_8192x128.npy` (train, (8192,128) float64) and
`dataset/Heston/heston_S_disc_8192x128.npy` (eval, (8192,128) float64).
`BENCHMARK_HESTON_EVAL` is left unset, which defaults to the `disc` file — **do not set it**
for a reporting run.

### 5.3 All seeds, as actually run

```bash
cd /home/tbasseras/benchmark/methods/Deep-MKV-TS/paper_reimplementation/metric

bash run_reproduction.sh    # the paper's four seeds: 0, 1, 3, 4
bash run_seed2.sh           # seed 2, for the GUIDELINE 5-seed set only
```

`run_reproduction.sh` runs **batch 1 = seeds 0, 1** then **batch 2 = seeds 3, 4**, two at a
time, GPU 0 pinned to cores 16–23 and GPU 1 to cores 24–31. Each seed writes
`runs/seed_{i}/COMPLETE.json` last; both scripts skip a seed that already has one, so they
are safe to re-run. `run_seed2.sh` additionally blocks until no other trainer is alive
(`pgrep -f run_matched_control_synthetic_validation.py`), because two trainers writing one
run tree corrupts both and the 2-GPU cap is already saturated by the reproduction.

### 5.4 Score and build the table

```bash
cd /home/tbasseras/benchmark/methods/Deep-MKV-TS/paper_reimplementation/metric
/home/tbasseras/gpu-venv/bin/python aggregate_paper_table.py
```

Writes, deterministically:

| Output | Contents |
|---|---|
| `../results/PAPER_VS_OURS.md` | the §4 table, verbatim |
| `../results/paper_vs_ours.json` | machine-readable: `columns`, `paper`, `ours.{Reference,Deep-MKV-TS}.{median,per_seed}`, `metrics_within_tolerance` |

### 5.5 Traceability — which file produced which cell

| Cell in §4 | Produced by |
|---|---|
| Deep-MKV-TS row, seed *i* | `runs/seed_{i}/volatility_only_online_mp/checkpoint_evaluations/step_2500/metrics.json` |
| Reference row, seed *i* | `runs/seed_{i}/reference/metrics.json` |
| Median row | `aggregate_paper_table.py`, `statistics.median` over seeds {0, 1, 3, 4} |
| Paper column | hand-entered from Table 1 of `Deep-MKV-TS_ICAIF.pdf`, frozen in `aggregate_paper_table.py::TARGET` |
| Verdict column | `aggregate_paper_table.py`, `abs(ours − paper) ≤ max(0.25·paper, 0.005)` |
| Wall-clock in §2 | `runs/seed_{i}/volatility_only_online_mp/run_manifest.json` |
| Fitted GL parameters in §2 | `../weights/seed_{i}_config.json`, written by `../code/export_benchmark_artifacts.py` |
| Paths used downstream | `runs/seed_{i}/volatility_only_online_mp/checkpoint_evaluations/step_2500/validation_bank.npy` → `../generated_paths/seed_{i}/generated_paths_8192x128.npy` |

Each seed produces **one** 8192-path bank scored **once**, so there is no run-to-run variance
within a seed — all the spread in the per-seed tables is genuine seed-to-seed sampling and
initialisation variance (`bank_seed = 70000 + seed`, so the four draws are independent).

### 5.6 Hyperparameter search (validation-only, for the record)

```bash
cd /home/tbasseras/benchmark/methods/Deep-MKV-TS/paper_reimplementation/metric
bash run_hpsearch.sh                                             # seed-0 grid, scored on valdisc
/home/tbasseras/gpu-venv/bin/python rank_hpsearch.py             # shortlist on seed 0
bash run_confirm.sh <tag> ...                                    # fill seeds 1, 3, 4 for the shortlist
/home/tbasseras/gpu-venv/bin/python rank_confirm.py <tag> ...    # rank on fresh seeds 1/3/4
```

Every run in that chain sets `BENCHMARK_HESTON_EVAL=heston_S_valdisc_8192x128.npy`, so no
search decision ever touched `heston_S_disc` or `heston_S_test`.

**14 trials were run on seed 0.** Score = mean over the five paper columns of ours ÷ paper;
below 1.000 means the trial beats the published Deep-MKV-TS row on average. Full table in
[`runs/hpsearch/RANKING.md`](runs/hpsearch/RANKING.md); the head of it:

| Rank | Trial | SWD | RV W₁ | \|r\| ACF | Early-future | MDD W₁ | Score |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `lambda100_kappa200_absacf0.50` | 0.0457 | 0.0139 | 0.0142 | 0.0052 | 0.0153 | **0.722** |
| 2 | `kappa200_absacf0.50` | 0.0433 | 0.0169 | 0.0173 | 0.0120 | 0.0128 | 0.848 |
| 3 | `jointvol1.0` | 0.0456 | 0.0228 | 0.0147 | 0.0126 | 0.0147 | 0.930 |
| … | … | | | | | | |
| 11 | `baseline` *(the paper config, what this reproduction reports)* | 0.0420 | 0.0196 | 0.0217 | 0.0254 | 0.0115 | 1.074 |
| 14 | `kappa200_absacf0.50_jointvol0.5` | 0.0467 | 0.0208 | 0.0237 | 0.0275 | 0.0162 | 1.198 |

**Several variants beat the paper configuration on seed 0**, the best by ~33 %
(0.722 vs 1.074), driven mostly by the early-future column (0.0052 vs 0.0254). But seed 0 is
also the seed that *chose* the shortlist, so that margin is biased upward by the selection
itself and cannot be taken at face value. The confirmation run below is what tests it.

#### Confirmation on fresh seeds {1, 3, 4} — the margin does not survive

`run_confirm.sh` filled seeds 1, 3 and 4 for the top two tags (6 runs, all complete) and
`rank_confirm.py` re-ranked on those unbiased seeds only. Full output in
[`runs/hpsearch_confirm/RANKING_CONFIRM.md`](runs/hpsearch_confirm/RANKING_CONFIRM.md):

| Trial | Score(fresh) | Fresh per-seed (1 / 3 / 4) | Seeds beating paper | mean ± sd | t vs 1.0 |
|---|---:|---|:---:|---|---:|
| `lambda100_kappa200_absacf0.50` | **0.791** | 0.836 / **1.159** / 0.790 | 2 / 3 | 0.928 ± 0.201 | 0.62 |
| `kappa200_absacf0.50` | 0.876 | 0.741 / 0.999 / 0.872 | **3 / 3** | 0.871 ± 0.129 | 1.74 |

`RANKING_CONFIRM.md` declares `lambda100_kappa200_absacf0.50` the winner because 0.791 < 1.000.
**That headline is weaker than it looks, in three specific ways:**

1. **One of the three unbiased seeds is worse than the paper.** Seed 3 scores 1.159. The
   nominal winner beats the published row on 2 of 3 seeds, not 3 of 3.
2. **`Score(fresh)` is not the median of the per-seed scores.** It is `score(median of each
   metric column)`, computed column by column, so different seeds can supply different columns
   and the aggregate ends up more favourable than any typical seed. The median of the per-seed
   scores is **0.836** and the mean is **0.928** — both materially worse than the reported
   0.791.
3. **The margin sits inside the noise.** With n = 3 the winner is 0.072 below 1.0 against an sd
   of 0.201 (t = 0.62). `RANKING_CONFIRM.md`'s own stated bar is that "a margin smaller than
   [the seed] range is seed noise, not an improvement" — the range is 0.722–1.159, i.e. 0.437
   wide, against a margin of 0.209. **It fails its own test.**

**The ranking also promotes the less robust of the two.** `kappa200_absacf0.50` beats the paper
on *every* unbiased seed with two-thirds the spread (t = 1.74 vs 0.62). Ranking on a single
point estimate rewards the config that drew one very good seed over the config that is
consistently better. Neither is statistically separable from the paper at n = 3.

> **One gap worth naming explicitly.** There is **no fresh-seed `valdisc` score for the paper
> configuration**: `run_confirm.sh` only filled seeds for the two shortlisted tags, and
> `run_reproduction.sh` sets no `BENCHMARK_HESTON_EVAL`, so §4's reproduction was scored on a
> different split. The `baseline` 1.074 in the seed-0 table above is the *only* number for the
> paper config on this split. Claims of the form "the tuned config beats our baseline by X %"
> therefore **cannot** be made on the unbiased seeds — only "beats the *published* row", and
> weakly. Closing that gap would mean running `baseline` on seeds 1/3/4 under
> `BENCHMARK_HESTON_EVAL=heston_S_valdisc_8192x128.npy`; it was not done.

> **⚠️ Why the reported numbers still use the paper configuration.** Everything in §4 and
> everything Deep-MKV-TS contributes to the benchmark is the **paper's own hyperparameters**
> (rank 11 above), not the search winner. Three reasons, all deliberate:
>
> 1. **This section is a reproduction.** Its job is to show that the published Table 1 is
>    reproducible, which requires running the published configuration. Substituting a tuned
>    variant would answer a different question.
> 2. **Benchmark fairness.** Every other method in `benchmark/methods/` is run at its authors'
>    published settings. Entering a locally-tuned Deep-MKV-TS against untuned baselines would
>    produce a comparison not worth publishing.
> 3. **The tuned config is not demonstrably better.** As shown above, the confirmation run does
>    not separate either shortlisted variant from the paper configuration at n = 3. Swapping in
>    a config that fails its own noise test would trade a defensible number for an
>    undefensible one.
>
> The search is recorded here for completeness and to be explicit that a better configuration
> **may** exist — the seed-0 evidence was suggestive and the fresh-seed evidence is
> inconclusive, which is not the same as a better configuration having been found. Search
> outputs live in `runs/hpsearch/` and `runs/hpsearch_confirm/` and are **not** part of any
> reported number.

---

## 6. Files

```
methods/Deep-MKV-TS/paper_reimplementation/
├── README.md                    ← this file
├── Deep-MKV-TS_ICAIF.pdf        the reference paper (committed per GUIDELINE §3.0)
├── dataset/
│   └── README.md                pointer to the shared benchmark/dataset/Heston/ (§2: never duplicate)
├── metric/
│   ├── PROTOCOL.json            frozen protocol: splits, reporting checkpoint, selection scope
│   ├── run_reproduction.sh      seeds 0,1,3,4 — the paper's set, 2 GPUs, 2 at a time
│   ├── run_seed2.sh             seed 2 — fills the GUIDELINE 0..4 requirement
│   ├── aggregate_paper_table.py builds §4 → results/PAPER_VS_OURS.md + paper_vs_ours.json
│   ├── run_hpsearch.sh          validation-only grid (valdisc), seed 0
│   ├── rank_hpsearch.py         shortlist on seed 0
│   ├── run_confirm.sh           fill seeds 1,3,4 for shortlisted tags
│   ├── rank_confirm.py          re-rank on FRESH seeds 1/3/4 (unbiased by selection)
│   └── run_pipeline.sh          convenience driver chaining the above
├── results/
│   ├── PAPER_VS_OURS.md         generated table (do not hand-edit)
│   └── paper_vs_ours.json       machine-readable version of the same numbers
└── runs/
    ├── seed_0/ … seed_4/        one full run tree per seed (see ../code/README.md §8)
    ├── hpsearch/                validation-only search runs (not reported)
    └── hpsearch_confirm/        validation-only confirm runs (not reported)
```

Downstream of this directory,
[`../code/export_benchmark_artifacts.py`](../code/export_benchmark_artifacts.py) converts
`runs/seed_{0..4}/` into the flat GUIDELINE §4.3/§4.4 layout (`../generated_paths/`,
`../weights/`, `../losses/`), which is what the Heston benchmark metrics in
`results/Heston/Deep-MKV-TS/` are computed from.
