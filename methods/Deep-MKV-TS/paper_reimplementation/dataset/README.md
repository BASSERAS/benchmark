# Dataset pointer — deliberately empty

**There are no `.npy` files in this directory, and there never should be.**

GUIDELINE §2 states the Heston dataset is *fixed and shared* and must **not** be duplicated
per method. Deep-MKV-TS's paper reimplementation targets Table 1, which is a **Heston**
experiment, so the paper's dataset and the benchmark's dataset are the same object. Copying
8 × 8 MB of `.npy` in here would create a second copy that can silently drift from the
canonical one.

## Where the data actually lives

```
benchmark/dataset/Heston/
├── generate_heston.py                 the generator (do not modify)
├── heston_S_8192x128.npy              train  — the split the model is fit to
├── heston_S_test_8192x128.npy         test   — NEVER touched by this method
├── heston_S_disc_8192x128.npy         disc   — every reported number is scored here
├── heston_S_val_8192x128.npy          val
├── heston_S_valdisc_8192x128.npy      valdisc — hyperparameter search only
└── heston_v_*.npy                     matching latent-variance paths
```

All price files: shape `(8192, 128)`, dtype `float64`, price levels starting near `S₀ = 100`.
All variance files: same shape/dtype, values near `v₀ = 0.04`.

Generation parameters:
`μ=0.05, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7, S₀=100, v₀=0.04, dt=1/250, N=8192, T=128`.

## How the code finds it

`code/reference/experiments/scripts/run_matched_control_synthetic_validation.py` walks up
from its own location until it finds a directory containing
`dataset/Heston/heston_S_8192x128.npy`. Nothing is hard-coded, and nothing needs to be
configured for the default layout.

Two optional environment variables override the search:

| Variable | Default | Meaning |
|---|---|---|
| `BENCHMARK_HESTON_DIR` | auto-discovered `…/dataset/Heston/` | directory holding the `.npy` splits |
| `BENCHMARK_HESTON_EVAL` | `heston_S_disc_8192x128.npy` | which split the run is **scored** on |

Training always reads `heston_S_8192x128.npy`; only the evaluation split is switchable.

## Which split for which purpose

| Doing this | Set | Why |
|---|---|---|
| Reproducing the paper table | *nothing* (leave both unset) | defaults to `disc`, the reporting split |
| Hyperparameter search | `BENCHMARK_HESTON_EVAL=heston_S_valdisc_8192x128.npy` | selecting on the reporting split invalidates the report |
| Anything at all | **never** `heston_S_test_8192x128.npy` | `test_split_access_authorized: false` in `../metric/PROTOCOL.json` |

## Using your own data

```bash
export BENCHMARK_HESTON_DIR=/absolute/path/to/your/npy/dir
export BENCHMARK_HESTON_EVAL=my_eval_split.npy
```

The directory must contain `heston_S_8192x128.npy` (training paths) and the file named by
`BENCHMARK_HESTON_EVAL`. Both must be 2-D arrays of **strictly positive price levels** —
not returns, not log-prices — with every path starting at the same `S₀`. Log-returns are
computed internally. The Heston task is hard-wired to `d = 1`.
