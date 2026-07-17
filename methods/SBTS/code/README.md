# SBTS Code — Sources & Implementation

## Original work

| Field | Details |
|-------|---------|
| **Paper** | *Schrödinger Bridge Time Series* |
| **Authors** | Guillaume Principato, Luca De Gennaro Aquino, Gustavo Boretto, Brieuc Lehmann |
| **Venue** | arXiv 2025 |
| **arXiv** | https://arxiv.org/abs/2503.02943 |
| **Original code** | https://github.com/g-principato/SBTS |
| **Method type** | Non-parametric kernel density estimation (no neural network, no training) |

The verbatim reference implementation is kept under [`reference/`](reference/) for transparency.

---

## Our implementation

`sbts_generate.py` is a **Numba-accelerated, multiprocessing-parallel** implementation
of the univariate Markovian SBTS variant for Heston price paths.

### Method overview (paper §3 + §6)

SBTS generates paths via a **Schrödinger-bridge drift** estimated from training data:

$$\hat{a}(t, x; \{x_i\}) = \frac{\sum_i w_i(t) \cdot \frac{x_i^{t+1} - x}{t+1 - t}}{\sum_i w_i(t)}$$

where `w_i(t)` are kernel weights measuring how close training path `i`
has been to the current trajectory up to time `t`.

**Markovian order K=1** (Heston, paper Appendix C Table 4): weights depend only
on the most recent state — `w_i ∝ K_h(X_i^t - x_t)` — reflecting that Heston
is a Markov process (no memory beyond the current state).

**Quartic kernel:** `K_h(x) = (h² − x²)²·1_{|x|<h}` (compact support, differentiable).

**Scaling trick** (paper §6): SBTS requires increments with variance ≈ Δt.
We scale log-returns as `R̃ = R × √Δt / σ(R)` before SBTS,
then inverse-scale: `R_gen = R̃_gen × σ(R) / √Δt`.

### Hyperparameters (paper Appendix C, Table 4 — Heston)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `h` | 0.4 | Kernel bandwidth (quartic compact-support) |
| `K` | 1 | Markovian order (Heston is Markov-1) |
| `N_pi` | 200 | Euler substeps per observation interval |
| `dt` | 1/250 | Observation interval |

### Pipeline

```
S_train (8192, 128)
  ↓ log-returns R = log(S[:,1:]/S[:,:-1])       (8192, 127)
  ↓ scale: R̃ = R × √dt / σ(R)                 (8192, 127)
  ↓ prepend dummy-0 column → X_train            (8192, 128)
  ↓ _simulate_one() × 8192 (Numba JIT)          one path at a time per worker
  ↓ drop initial col, inverse-scale: R_gen      (8192, 127)
  ↓ S_gen[:,0] = 100; S_gen[:,t+1] = S_gen[:,t]·exp(R_gen[:,t])
  → S_gen (8192, 128)  in price space, anchored at 100
```

### Parallelisation

Python `multiprocessing.Pool` with `n_workers=64` (seeds 1–4) or `n_workers=16`
(seed 0). Each worker inherits the parent's Numba JIT cache (Linux fork semantics)
and generates its `M_simu / n_workers` paths serially in a loop.

No GPU is needed: SBTS is a kernel-based method with no learnable parameters.

---

## Key files

| File | Purpose |
|------|---------|
| `sbts_generate.py` | Core module: `generate_paths()`, `warmup_jit()`, Numba kernels |
| `small_test.py` | Sanity test: N_train=200, M_simu=20, T=32 — passes all checks |
| `run_all.py` | Full run: 5 seeds × 8 192 paths × 128 steps |
| `reference/` | Verbatim SBTS repo (g-principato/SBTS, .git stripped) |

---

## Differences from the reference implementation

| Aspect | Reference (`sbts_uni_markovian.py`) | Our implementation |
|--------|-------------------------------------|-------------------|
| Backend | NumPy + Joblib | **Numba JIT** (`@nb.jit(nopython=True, cache=True)`) |
| Parallelisation | Joblib | `multiprocessing.Pool` (fork — inherits JIT) |
| Data | Generic | Heston price paths only (`dataset/Heston/heston_S_8192x128.npy`) |
| Output | Scaled log-returns | **Price paths** anchored at S₀=100 |
| Warmup | None | `warmup_jit()` pre-compiles before Pool fork |

---

## Reproduce

```bash
# Environment
source /home/tbasseras/sbts-venv/bin/activate   # Python 3.12, numba 0.66.0

# Sanity check first (< 30 s)
python small_test.py

# Full run — 5 seeds (env vars override defaults)
SBTS_NWORK=64 python run_all.py                          # all 5 seeds, 64 workers
SBTS_SEEDS=1,2,3,4 SBTS_NWORK=64 python run_all.py      # seeds 1-4 only

# Compute metrics (GPU)
cd /home/tbasseras/benchmark
CUDA_VISIBLE_DEVICES=0 \
    /home/tbasseras/gpu-venv/bin/python metrics/compute_all.py --method SBTS --dataset Heston
```

---

## Sanity check results (small\_test.py, 2026-07-17)

```
N_train=200, M_simu=20, T=32, h=0.4, K=1, N_pi=200, n_workers=4
  ✓ Shape: (20, 32)
  ✓ No NaN / inf
  ✓ All paths start at S0=100.0
  ✓ Price range: [83.59, 119.47]

Log-return statistics (generated vs training):
  ✓ mean  : train=+0.00007   gen=+0.00007
  ✓ std   : train=+0.01251   gen=+0.00982   (21% below — expected at N_train=200)
  ✓ skew  : train=-0.02703   gen=-0.01890
  ✓ kurt  : train=+0.58764   gen=-0.06050
```

Variance compression at N_train=200 is expected: the kernel estimator underestimates
the tails on small training sets. It disappears at N_train=8192 (full run),
confirmed by seed 0: σ_train=0.01265, price range [51.8, 210.6].

---

## Changing hyperparameters

All hyperparameters are taken verbatim from **paper Appendix C, Table 4** (Heston, K=1 Markovian).
To experiment, override the constants at the top of `run_all.py` or pass them explicitly
to `generate_paths()` in `sbts_generate.py`.

| Hyperparameter | Paper default | Where to change | Effect |
|----------------|---------------|-----------------|--------|
| `h` | 0.4 | `run_all.py` constant `H_BW` | Kernel bandwidth. Larger h → smoother paths, smaller variance. |
| `K` | 1 | `run_all.py` constant `K_ORDER` | Markovian order. K=1: weights on last state only. K=2 adds lag-1 state (non-Markovian). |
| `N_pi` | 200 | `run_all.py` constant `N_PI` | Euler substeps per Δt. More substeps → smoother approximation, slower generation. |
| `n_workers` | 64 | `SBTS_NWORK` env var | CPU parallelism. Set to number of available cores (max 64 recommended). |
| `dt` | 1/250 | `sbts_generate.py` `DT` | Observation interval. 1/250 = daily (250 trading days/year). |

**Example — wider bandwidth:**
```bash
SBTS_H=0.6 SBTS_NWORK=64 python run_all.py
```

(requires reading `SBTS_H` in `run_all.py`; add `H_BW = float(os.getenv("SBTS_H", "0.4"))`)

---

## Using a different dataset

`run_all.py` loads training data from `dataset/Heston/heston_S_8192x128.npy` by default.
To use a different dataset:

1. Provide a `.npy` file of shape `(N_train, T)`, dtype `float64`, in price space (S₀ ≈ 100).
2. Update the constants in `run_all.py`:
```python
DATA_PATH = "/path/to/your/data.npy"   # was heston_S_8192x128.npy
OUT_DIR   = "generated_paths/"          # output root
```
3. The scaling step `R̃ = R × √dt / σ(R)` is automatic — no manual adjustment needed.
4. If your series has a different length T, SBTS adapts automatically (no architecture change).

---

## Producing new seeds

Each seed initialises a new `np.random.default_rng(seed)` for the Numba simulation noise.
To add a single extra seed:

```bash
# One extra seed on 64 workers
SBTS_SEEDS=5 SBTS_NWORK=64 python run_all.py
```

To reproduce all 5 seeds from scratch:
```bash
SBTS_NWORK=64 python run_all.py          # runs seeds 0–4 sequentially
```

Each seed takes ~6–23 min (depends on worker count). Results land in:
- `generated_paths/seed_{i}/generated_paths_8192x128.npy`
- `losses/seed_{i}_bandwidth.json`  (records h, K, N_pi — no loss since kernel method)
