# Heston Dataset — seq_len = 128

## Quick description

Synthetic price/variance paths generated from the **Heston stochastic volatility model**
(Heston 1993) using an Euler–Maruyama full-truncation scheme. Every path has **128 time steps**
(≈ half a year of daily prices, dt = 1/250) and starts at $S_0 = 100$, $v_0 = 0.04$.

The benchmark uses **three disjoint 8 192-path splits** — all drawn from the *same* SDE and
parameters, differing only by RNG seed — plus a very large **100 M-path empirical bank** used to
build the high-resolution reference ("theory") curves for the diagnostic plots.

## Stochastic differential equations

$$
dS_t = \mu S_t dt + \sqrt{v_t} S_t dW_t^S
$$
$$
dv_t = \kappa (\theta - v_t^+) dt + \xi \sqrt{v_t^+} dW_t^v
$$
$$
\mathbb{E}[dW_t^S dW_t^v] = \rho dt
$$

where $v_t^+ = \max(v_t, 0)$ is the full-truncation fix that keeps variance non-negative.

## Parameters

| Symbol | Meaning | Value |
|--------|---------|-------|
| $\mu$ | Drift | 0.05 |
| $\kappa$ | Mean-reversion speed | 2.0 |
| $\theta$ | Long-run variance | 0.04 (≈ 20 % vol) |
| $\xi$ | Vol-of-vol | 0.3 |
| $\rho$ | Spot–vol correlation | −0.7 |
| $S_0$ | Initial price | 100.0 |
| $v_0$ | Initial variance | 0.04 |
| $dt$ | Time step | 1/250 (daily) |
| $T$ | Sequence length | 128 |
| $N$ | Paths per split | 8 192 |

The Feller condition $2\kappa\theta \geq \xi^2$ gives $2 \times 2.0 \times 0.04 = 0.16 \geq 0.09$: satisfied.

## The three benchmark splits

Each split is an independent 8 192-path Heston draw (same SDE, same parameters, distinct seed).
Keeping them disjoint is what makes every reported number a genuine **out-of-sample** measurement.

| Split | Seed | Role |
|-------|:----:|------|
| **train** | 0 | The only data a generator ever sees. Every method is fit on this split. |
| **test** | 1 | Held-out **real reference**. Every A/B metric and the Path-Shadowing CRPS score the generated paths against this split (never against train). The **Perfect floor** is a *fresh* draw scored against this same test set. |
| **disc** | 2 | A third independent real split used **only** by the discriminative / predictive metric classifiers (A18 / A19) as their real-vs-fake training data, so those classifiers never touch the test split they are evaluated on. |

## Files

All six split arrays are **float64, shape (8192, 128)** and are committed to the repo.

| File | Split | Field | Description |
|------|-------|-------|-------------|
| `heston_S_8192x128.npy` | train (seed 0) | $S_t$ | Price paths |
| `heston_v_8192x128.npy` | train (seed 0) | $v_t$ | Variance paths |
| `heston_S_test_8192x128.npy` | test (seed 1) | $S_t$ | Price paths — scoring reference |
| `heston_v_test_8192x128.npy` | test (seed 1) | $v_t$ | Variance paths — used by the Heston-spec metrics (A33/A34) |
| `heston_S_disc_8192x128.npy` | disc (seed 2) | $S_t$ | Price paths — A18/A19 classifier real data |
| `heston_v_disc_8192x128.npy` | disc (seed 2) | $v_t$ | Variance paths |
| `generate_heston.py` | — | — | CPU (numpy, float64) generator for the three 8 192-path splits |
| `generate_heston_large.py` | — | — | GPU (torch, float32) generator for the 100 M-path empirical bank |

### `large_dataset/` — 100 M-path empirical reference (not committed)

This directory is a high-resolution empirical bank used **only** to estimate the reference curves for
the diagnostic panels that have no clean closed form (ACF of |r|, ACF of r², rolling-vol histogram,
tail survival). It is large (~50 GiB) and therefore **git-ignored** — regenerate it locally with the
command below.

| File | Shape | dtype | Description |
|------|-------|-------|-------------|
| `large_S_shard{0..3}.npy` | (25 000 000, 128) | float32 | Price paths — 4 GPU shards, 100 M total |
| `large_v_shard{0..3}.npy` | (25 000 000, 128) | float32 | Variance paths — 4 GPU shards |
| `theory_curves_bundle.npz` | 9 arrays | float64 | Pre-computed reference curves (`qq_grid`, `qq_theory`, `lags`, `acf_abs`, `acf_sq`, `rvol_grid`, `rvol_dens`, `tail_oneminusq`, `tail_surv`) consumed by `metrics/heston_theory.py` and `metrics/plot_diagnostics.py` |
| `gen_shard{k}.log` | — | — | Per-shard throughput logs |

Each of the four GPU workers runs one shard with a distinct base seed and shard index (seeds 100–103
by the launch convention documented in `generate_heston_large.py`), so the 100 M paths are fully
independent both of one another and of the three benchmark splits.

## Reproduce

**The three 8 192-path splits** (train / test / disc):

```bash
cd dataset/Heston
python - <<'PY'
import numpy as np
from generate_heston import generate_heston
for name, seed in [("", 0), ("_test", 1), ("_disc", 2)]:
    S, v = generate_heston(seed=seed)          # (8192, 128) float64
    np.save(f"heston_S{name}_8192x128.npy", S)
    np.save(f"heston_v{name}_8192x128.npy", v)
    print(f"{name or '(train)':8s} seed={seed}  S[:,0].mean={S[:,0].mean():.2f}")
PY
```

(`python generate_heston.py` on its own writes only the train split — seed 0.)

**The 100 M-path empirical bank** (one GPU process per shard, ~45 s/shard on an A100):

```bash
cd dataset/Heston
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k python generate_heston_large.py \
      --n 25000000 --chunk 2000000 --seed $((100+k)) --shard $k \
      --outdir large_dataset > large_dataset/gen_shard$k.log 2>&1 &
done
wait
```

Then build the reference-curve bundle from the bank:

```bash
python ../../metrics/heston_theory.py     # writes large_dataset/theory_curves_bundle.npz
```

Sanity check: every `heston_S*.npy` has `S[:, 0].mean() ≈ 100` and `S[:, 0].std() = 0` (all paths
start at $S_0 = 100$).
