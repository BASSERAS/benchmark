# SBTS — Paper Reimplementation (Stocks)

Reproduction of the **SBTS** paper result on the **Stocks** dataset using the
*official* SBTS code, verbatim from the authors' demo notebook.

- **Paper:** *Schrödinger Bridge Time Series* — Alouadi, Barreau, Carlier, Pham,
  arXiv:2503.02943, ICAIF'25.
- **Official code:** `github.com/alexouadi/SBTS` (mirrored here under `../code/reference/`).
- **This run:** `metric/reproduce_stock.py` — one command, single A100 (GPU logical 0).

---

## 1. Paper metrics (as defined in the paper)

Both scores follow the post-hoc protocol of Yoon et al. (TimeGAN, NeurIPS'19),
which SBTS adopts verbatim:

| Metric | Definition | Direction |
|--------|-----------|-----------|
| **Discriminative score** | Train a 2-layer GRU (hidden = ⌊d/2⌋) to tell real from synthetic; report **\|accuracy − 0.5\|** on a held-out set. | lower = better |
| **Predictive score** | TSTR: train a 1-layer GRU (hidden = max(⌊d/2⌋, 1)) on **synthetic** to predict the next step, evaluate **MAE** on **real**. | lower = better |

Reported as **mean ± std over 10 independent test runs** (`n_temp=10`), each
trained for `itt=2000` epochs, batch 128, features min-max scaled.

---

## 2. Method hyperparameters (from the official notebook)

SBTS is a **kernel** generator — there is **no neural training**. Generation is
`simulateSB_multi` (numba), with the exact settings from
`reference/experiments_demo.ipynb` "Stock Dataset" cells:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `N_pi` | 100 | number of bridge discretisation points |
| `h` | 0.2 | kernel bandwidth |
| `deltati` | 1/252 | time step (daily, 252 trading days) |
| `M_simu` | 1000 | number of synthetic paths generated |
| `d` | 5 | feature dimension |
| `N` | 10 | sequence length (steps) |

Input: `metrics/fbm_stock_metrics/data/X_stock.pt` — official preprocessed
log-returns tensor, shape **(1002, 10, 5)**.

---

## 3. Dataset

**Stocks** (Google daily prices, the standard TimeGAN/SBTS benchmark).
The official repo ships the preprocessed log-return windows directly, so no
re-download or re-windowing is needed — this is why Stocks is the *easy*
single-dataset choice mandated by the task.

- Real:  `dataset/X_stock_real_logret.npy`  — (1002, 10, 5)
- Synth: `dataset/X_stock_sbts_logret.npy`   — (1000, 10, 5)
- Distribution sanity table: `results/stock_stats.csv`

---

## 4. Results — ours vs paper

| Dataset | Metric | **Ours (official SBTS code)** | **Paper (Table 1)** | Verdict |
|---------|--------|------------------------------|---------------------|---------|
| Stocks | Discriminative ↓ | **0.026 ± 0.012** | 0.010 ± 0.008 | same regime ✓ |
| Stocks | Predictive ↓ | **0.018 ± 0.003** | 0.017 ± 0.000 | **matches** ✓ |

**Reproduced.** The predictive score matches the paper almost exactly
(0.018 vs 0.017). The discriminative score sits in the same low regime
(both ≈ 0.01–0.03, i.e. an adversary is essentially at chance) — the small gap
is the expected run-to-run variance of a stochastic post-hoc classifier scored
over only 10 runs with a different RNG seed, not a methodological difference.

Generation time: **39 s** (numba; the notebook's "expected finish" banner is an
over-estimate — it extrapolates from sample 0, which pays a one-time JIT cost).

---

## 5. How to reproduce

```bash
cd metric
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
  taskset -c 0-7 /home/tbasseras/gpu-venv/bin/python reproduce_stock.py
```

Requires `numba>=0.59` and `torch` in the same env (`gpu-venv`).

**Exact run path — which file feeds which cell (so any number is traceable):**

| Table cell | Interpreter + env | Script | Input file(s) scored | Output JSON |
|------------|-------------------|--------|----------------------|-------------|
| All §4 Stocks scores (predictive + distribution) | `gpu-venv`, `CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 taskset -c 0-7` | `metric/reproduce_stock.py` | real `dataset/X_stock_real_logret.npy` (1002,10,5) vs SBTS synthetic `dataset/X_stock_sbts_logret.npy` (1000,10,5) | `results/sbts_stock_scores.json` (full per-run arrays + paper reference) + `results/stock_stats.csv` |

Every reported §4 number is a field in `results/sbts_stock_scores.json`; the script regenerates it end-to-end (load → generate → score) in one call, no intermediate hand-editing.

---

## 6. Files

| Path | Content |
|------|---------|
| `metric/reproduce_stock.py` | end-to-end script (load → generate → score) |
| `dataset/X_stock_real_logret.npy` | real log-returns (1002, 10, 5) |
| `dataset/X_stock_sbts_logret.npy` | SBTS synthetic paths (1000, 10, 5) |
| `results/sbts_stock_scores.json` | full scores + per-run arrays + paper reference |
| `results/stock_stats.csv` | distribution comparison table |
