# SBTS source — TrueDataset (real crypto, d = 8)

## Files

| File | Role |
|---|---|
| `sbts_generate_true.py` | Core library. Radial Schrödinger-bridge kernel, per-asset σ scaling, Numba-JIT drift, multiprocessing fan-out. **No `main`** — it is imported, never run. |
| `generate_bank_true.py` | Driver for the *one* run that produces a scoring bank, at the calibrated `(h, K)`. Writes into the layout `compute_all_multiasset.py` expects. |
| `calibrate_bandwidth_true.py` | Sweeps `(h, K)` and scores each cell against the real-vs-real envelope. Writes `losses/bandwidth_selection_*.json`. |
| `collect_artifacts.py` | Turns the per-bank `metadata.json` into `losses/generation_time.csv` and `losses/seed_<S>_bandwidth.json`. |
| `plot_diagnostics_true.py` | The 8-panel stylised-facts figure. |
| `render_readme.py` | Regenerates `../README.md` from the artefacts. Every number in that file is read from disk. |

## Deviations from the d = 1 / Heston implementation

**1. `h` and `K` are ours; only `N_pi` is the author's.**
The kernel is radial, `K_h(x) = (h² − ‖x‖²)²·1{‖x‖₂ < h}`, so its support radius must scale
with the typical distance between d-dimensional increments — a property of the data. The
author's `h = 0.05` was measured on d = 1 length-128 Heston and does not transfer. `K` was
re-selected too: at 30-second bars the conditional dependence structure is different from daily
Heston, and `K = 1` won the grid.

**2. `DT` is a pure gauge.**
`sbts_generate_true.py` carries `DT = 1/252` at module level, and it does **not** mean the data
is daily. `build_training_tensor` rescales log-returns per asset so that `std(X) = √DT` exactly
(0.062994 here) before the kernel ever sees them, and the inverse scaling is applied on the way
out. `h` is therefore measured in units of `√DT`. Changing `DT` and rescaling `h` by the same
factor produces bit-identical output. The *physical* Δt of this dataset — 9.512937595129376e-07
years, i.e. 30 seconds against 1 051 200 bars/year — is passed separately to the metric driver
via `--dt`, and is the only place the real time base matters.

**3. Per-asset σ, not a pooled σ.**
The eight assets have annualised vol from 0.48 (BTC) to 1.11 (DOGE) on the training split.
A pooled scale would make the kernel ball anisotropic in the wrong way — effectively a tighter
bandwidth on the quiet assets and a looser one on the noisy ones. `metadata.json` records the
eight σ values actually used.

**4. `generate_bank_true.py` registers the module in `sys.modules` before executing it.**
Not cosmetic. `generate_paths` fans out over `multiprocessing`, and fork+pickle pickles the
worker function *by qualified name*. A module loaded through `importlib` but never registered
makes that name resolve to a different object in the child, and the run dies with
`it's not the same object as sbts_generate_true._worker` — after the fan-out has already been
launched. See the docstring in that file.

**5. `M_simu` defaults to the size of the training split, not a round number.**
Every real-vs-real threshold this bank is judged against was measured between real splits of
exactly 6 144 paths, and the volatility estimator's sampling noise falls like 1/√m (1.67 pp at
512, 0.42 pp at 8 192). Generating a different count would compare a differently-noisy estimate
against those thresholds. The one exception is the conditional-CRPS pool, which is pinned to the
paper's 8 192 — see `../README.md` §C.

## Interpreter

`sbts_generate_true.py` and everything that imports it need **`/home/tbasseras/sbts-venv/bin/python`**
(Numba). `render_readme.py` and `collect_artifacts.py` are stdlib-only. Anything importing
torch — the metric driver, the loss plotter — needs **`/home/tbasseras/gpu-venv/bin/python`**.
