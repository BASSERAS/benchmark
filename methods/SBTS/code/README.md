# SBTS Code, Sources & Implementation

## Original work

| Field | Details |
|-------|---------|
| **Paper** | *Robust Time Series Generation via Schrödinger Bridge: A Comprehensive Evaluation* |
| **Authors** | Alexandre Alouadi, Baptiste Barreau, Laurent Carlier, Huyên Pham |
| **Venue** | ICAIF 2025 |
| **arXiv** | https://arxiv.org/abs/2503.02943 |
| **Original code** | https://github.com/alexouadi/SBTS |
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

**Markovian order K=20** (author-specified for this length-128 Heston benchmark,
confirmed by A. Alouadi 2026-07-27): weights depend on the last **K** states,
`w_i ∝ ∏_{j} K_h(X_i^{t-j} - x_{t-j})`, giving the kernel enough memory to
reproduce Heston's volatility autocorrelation over the 128-step horizon.

**Quartic kernel:** `K_h(x) = (h² − x²)²·1_{|x|<h}` (compact support, differentiable).

**Scaling trick** (paper §6): SBTS requires increments with variance ≈ Δt.
We scale log-returns as `R̃ = R × √Δt / σ(R)` before SBTS,
then inverse-scale: `R_gen = R̃_gen × σ(R) / √Δt`.

### Hyperparameters (author-specified for this length-128 Heston benchmark)

Confirmed by A. Alouadi (SBTS author) 2026-07-27. These **supersede** the paper's
length-100 Heston values (h=0.4, K=1, N_pi=200): h=0.4 was far too large for this
setup and over-smoothed the paths into a degenerate near-Gaussian. dt=1/250 is kept
to match the shared Heston dataset used by every method.

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `h` | 0.05 | Kernel bandwidth (quartic compact-support), paper's 0.4 was much too large |
| `K` | 20 | Markovian order, enough memory to reproduce Heston vol autocorrelation |
| `N_pi` | 50 | Euler substeps per observation interval (chosen for speed; low impact) |
| `dt` | 1/250 | Observation interval (shared dataset) |

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

Python `multiprocessing.Pool` with `n_workers=64` for all seeds. Each worker inherits
the parent's Numba JIT cache (Linux fork semantics) and generates its
`M_simu / n_workers` paths serially in a loop.

No GPU is needed: SBTS is a kernel-based method with no learnable parameters.

---

## Key files

| File | Purpose |
|------|---------|
| `sbts_generate.py` | Core module: `generate_paths()`, `warmup_jit()`, Numba kernels |
| `small_test.py` | Sanity test: N_train=200, M_simu=20, T=32, passes all checks |
| `run_all.py` | Full run: 5 seeds × 8 192 paths × 128 steps |
| `reference/` | Verbatim SBTS repo (alexouadi/SBTS, .git stripped) |

---

## Differences from the reference implementation

| Aspect | Reference (`sbts_uni_markovian.py`) | Our implementation |
|--------|-------------------------------------|-------------------|
| Backend | NumPy + Joblib | **Numba JIT** (`@nb.jit(nopython=True, cache=True)`) |
| Parallelisation | Joblib | `multiprocessing.Pool` (fork, inherits JIT) |
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

**Exact run path, which file produced which committed number:**

| Committed number | Interpreter + env | Command | Input file(s) scored | Output file |
|------------------|-------------------|---------|----------------------|-------------|
| Heston A1-A34 + B, per seed `i` | `gpu-venv`, `CUDA_VISIBLE_DEVICES=0` | `metrics/compute_all.py --method SBTS --dataset Heston` | `methods/SBTS/generated_paths/seed_i/generated_paths_8192x128.npy` (8192,128) vs the real Heston pool `dataset/Heston/heston_S_8192x128.npy` | `results/Heston/SBTS/seed_i_metrics.json` (A-metrics) + `curve_b_aggregate.json` (B curves) |
| SBTS synthetic paths, per seed `i` | `sbts-venv`, `SBTS_NWORK=64` | `run_all.py` (seed `i`) | canonical Heston SDE params (no input file, simulated) | `methods/SBTS/generated_paths/seed_i/generated_paths_8192x128.npy` |

Each `results/Heston/SBTS/seed_i_metrics.json` is the sole source for that seed's column in every README A-table; the mean±std rows are aggregated across those 5 files.

---

## Sanity check results (small\_test.py, 2026-07-27)

```
N_train=200, M_simu=20, T=32, h=0.05, K=20, N_pi=50, n_workers=4
  ✓ Shape: (20, 32)
  ✓ No NaN / inf
  ✓ All paths start at S0=100.0
  ✓ Price range: [85.91, 116.51]

Log-return statistics (generated vs training):
  ✓ mean  : train=+0.00015   gen=+0.00040
  ✓ std   : train=+0.01251   gen=+0.01207   (only 4% below — h=0.05 avoids compression)
  ✓ skew  : train=-0.05768   gen=-0.09626
  ✓ kurt  : train=+0.31288   gen=+0.80023
```

With the corrected small bandwidth (h=0.05) the variance compression seen under the
paper's h=0.4 is gone even at N_train=200 (std within 4%, vs 21% under h=0.4). On the
full N_train=8192 run the match tightens further: seed 0 σ_train=0.01265, price range
[41.1, 143.8].

---

## Changing hyperparameters

The Heston hyperparameters are **author-specified** (A. Alouadi, 2026-07-27) for this
length-128 benchmark and supersede the paper's length-100 values. To experiment, override
the constants at the top of `run_all.py` or pass them explicitly to `generate_paths()`.

| Hyperparameter | Benchmark value | Where to change | Effect |
|----------------|-----------------|-----------------|--------|
| `h` | 0.05 | `run_all.py` constant `H` | Kernel bandwidth. Larger h → smoother paths, smaller variance. The paper's 0.4 over-smoothed this setup. |
| `K` | 20 | `run_all.py` constant `K` | Markovian order. Larger K → more memory (reproduces vol autocorrelation), slower per path. |
| `N_pi` | 50 | `run_all.py` constant `N_PI` | Euler substeps per Δt. More substeps → smoother approximation, slower generation (low impact on quality). |
| `n_workers` | 64 | `SBTS_NWORK` env var | CPU parallelism. Set to number of available cores. |
| `dt` | 1/250 | `sbts_generate.py` `DT` | Observation interval. 1/250 = daily; kept fixed to match the shared dataset. |

**Example, override on the command line:**
```bash
SBTS_NWORK=64 SBTS_SEEDS=0,1 python run_all.py
```

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
3. The scaling step `R̃ = R × √dt / σ(R)` is automatic, no manual adjustment needed.
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

Each seed takes ~6-23 min (depends on worker count). Results land in:
- `generated_paths/seed_{i}/generated_paths_8192x128.npy`
- `losses/seed_{i}_bandwidth.json`  (records h, K, N_pi, no loss since kernel method)
