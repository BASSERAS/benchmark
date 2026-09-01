# SBBTS — Code

## Paper
**Title**: Schrödinger Bass Bridge for Time Series
**Authors**: Alexandre Alouadi et al.
**Venue**: preprint, 2026
**arXiv**: https://arxiv.org/abs/2604.07159
**GitHub**: https://github.com/alexouadi/SBBTS
**PDF**: [`../paper_reimplementation/SBBTS_arXiv-2604.07159.pdf`](../paper_reimplementation/SBBTS_arXiv-2604.07159.pdf)

## Method summary
SBBTS learns a one-step-ahead conditional transport as a **Brownian bridge with a learned
drift**. For every consecutive pair `(x_i, x_{i+1})` of a trajectory, a bridge is built between
a *transported* pair `(y_0, y_T)`, and a score network is trained to match the analytic bridge
score `(y_T - y_t) / (T - t)`. The transport map itself is refined by **Diffusion Schrödinger
Bridge Matching** (Algorithm 1): at outer iteration `k` the previous drift defines the new
endpoints via `y = x - (1/β)·s_θ(x)`, and the score is retrained. Unlike diffusion models (fixed
volatility) or martingale transport (no drift), SBBTS calibrates drift and volatility jointly.

Generation is autoregressive over the `N` intervals: the transformer encoder summarises the
past into a context vector `h_n`, `N^π` Euler substeps integrate the bridge SDE forward, and the
inverse transport `x = y + (1/β)·s_θ(y)` maps back to data space.

## Variant chosen
The reference ships **two** training branches, selected by `β`:

| Branch | Condition | Inverse transport | File |
|--------|-----------|-------------------|------|
| **Analytic** (chosen) | `β ≥ 100` | closed form, `y = x − (1/β)·s_θ` | `training/training_sbbts_dsbm.py` |
| Learned | `β < 100` | separate `InverseMLP`, alternating optimisation | `training/training_sbbts_inv.py` |

We use the **analytic branch with `β = 100`**, because that is exactly what the authors'
`run_heston.py` does for the Heston experiment — the setting Alexandre Alouadi pointed us to as
canonical (2026-09-01). The learned branch is ported too (`InverseMLP` is present in
`sbbts_torch.py`) but is not exercised by the benchmark runs.

## Fixes applied to the reference implementation
1. **Merged the package into one module** (`sbbts_torch.py`). The reference uses bare imports —
   `from encoder_only import EncoderOnly` in `models/sbbts_model.py`, `from early_stopping import
   EarlyStopping` in `training/training_sbbts_dsbm.py`. These only resolve if each *sub-package
   directory* is itself on `sys.path`, so `from models.sbbts_model import ScoreNN` raises
   `ModuleNotFoundError`. Merging removes the fragility; no maths changed.
2. **Causal mask is sliced to the current sequence length.** `EncoderOnly.forward` used the full
   pre-allocated `self.mask` (size `N × N`), which raises a shape error whenever the input is
   shorter than `N` — which happens on *every* autoregressive generation step, where the past has
   length `i+1 < N`. Fixed to `mask=self.mask[:L, :L]` with `L = y_emb.size(1)`.
3. **Training returns a loss history.** `training_sbbts_dsbm` additionally returns
   `history = [{k, epoch, train_loss, val_loss}, …]` so GUIDELINE §4.4 loss CSVs can be written.
   The reference printed losses and discarded them.
4. **Final `y_0` computation wrapped in `torch.no_grad()`.** The reference builds the returned
   initial condition inside the autograd graph and then calls `.detach()`; harmless but it holds
   a graph alive. Numerically identical.
5. **`sns.kdeplot(shade=…)` → `fill=…`** in the Figure-2 plot helper (`shade` was removed in
   seaborn ≥ 0.14). Plotting only; see `../paper_reimplementation/metric/heston_mle.py`.
6. **Cosmetic only (no behaviour change).** `generate_dsbm` uses `divmod(M_simu, N_batch)` where
   the reference writes `base = M_simu // N_batch; extra = M_simu % N_batch`; the remainder
   distribution `base + (1 if j < extra else 0)` is byte-for-byte the reference's. Per-batch
   progress printing is gated behind `verbose` and the reference's wall-clock ETA prints were
   dropped.

### Known upstream behaviour we deliberately did NOT change
- **Train/inference mask asymmetry.** Training calls the encoder *with* the causal mask;
  `generate_dsbm_batch` calls it *without* a mask and keeps only the last position. At the final
  position masked and unmasked attention agree, but with `n_layers = 2` the layer-1 outputs at
  *earlier* positions differ, so layer 2's final position consumes slightly different inputs than
  it saw in training. This is upstream behaviour and changing it would make our numbers
  non-comparable to the paper's; documented, not patched.
- **Early-stopping tie handling.** `EarlyStopping` uses `score + delta >= best_score` to increment
  the patience counter, i.e. an improvement must *exceed* `delta` to reset it. Kept verbatim.

## Architecture
`ScoreNN` = `EncoderOnly` (context over the past) + `MLP` (drift head).

| Component | Type | Dims | Output |
|-----------|------|------|--------|
| `EncoderOnly.embedding` | `Linear(d → d_model)` + `PositionalEncoding` (sinusoidal, `max_len = 5000`) | `d → 128` | `(B, L, 128)` |
| `EncoderOnly.past_encoder` | `nn.TransformerEncoder`, causal square-subsequent mask, `n_layers = 2`, `nhead = 32` | `d_model = 128` | context `h` `(B, L, 128)` |
| `MLP.t_encoder` | `Linear → LayerNorm → SiLU → Linear` on `get_timestep_embedding(t, 128)` | `128 → 128` | `t_embed (B, L, 128)` |
| `MLP.y_encoder` | `Linear → LayerNorm → SiLU → Linear` | `d → hidden_dim = 64 → 128` | `y_embed (B, L, 128)` |
| `MLP.cond_fusion` | `Linear → SiLU → Linear` on `concat[t_embed, y_embed, h]` | `3·128 = 384 → 128 → d` | score `(B, L, d)` |
| `InverseMLP` *(unused, β ≥ 100)* | `Linear` stack over `[t, x]` | `d_model` | `(B, L, d)` |

Total: **1,277,569** parameters at `d = 1` (benchmark Heston), **1,277,890** at `d = 2`
(paper reimplementation) — verified from the smoke runs.

## Hyperparameters (from paper)
| Parameter | Value | Source |
|-----------|-------|--------|
| `T` | 1 | Table 2 / `run_heston.py` |
| `T̃` (⇒ `safe_t`) | 0.99 (⇒ `1e-2`) | Table 2 |
| `β` | 100 | `run_heston.py` |
| `K` (outer DSBM iterations) | 5 | Table 2 |
| `n_epochs` | 1000 | Table 2 |
| `batch_size` | 128 | Table 2 |
| `lr` | 1e-3 | Table 2 |
| `patience` / `delta` | 15 / 1e-3 | `run_heston.py` |
| `d_model` | 128 | Table 2 |
| `hidden_dim` | 64 | `run_heston.py` |
| `n_layers` | 2 | `run_heston.py` |
| **`nhead`** | **32** (paper Table 2 says 16) | `run_heston.py` — see below |
| **`N^π`** (Euler substeps) | **60** (paper Table 2 says 50) | `run_heston.py` — see below |
| `M` (train paths) | 5000 (paper) / 8192 (benchmark) | §5.1 / benchmark dataset |
| `N` (intervals) | 252 (paper) / 127 (benchmark) | §5.1 / benchmark `seq_len = 128` |

**Paper-vs-repo divergences.** `nhead` (32 vs 16), `N^π` (60 vs 50) and `M_simu` (4000 vs "5000
trajectories") disagree between Table 2 and `run_heston.py`. We follow the **repo**, because
Alexandre Alouadi explicitly identified `run_heston.py` as the canonical pipeline: *"Tu peux
retrouver toute la pipeline d'utilisation et les hyperparams ici: …/run_heston.py"* (2026-09-01).

## Data transform
Per the author's guidance — *"tu transformes tes prix en log returns, tu génères ça et tu inverses
pour retrouver les prix"* — identical in spirit to SBTS, and byte-for-byte compatible with
`methods/SBTS/code/sbts_generate.py`:

```
R          = diff(log S)                       (M, 127)
X[:, 0]    = 0 ; X[:, 1:] = R                  (M, 128, 1)
scale      = X.std(over path, time) / sqrt(T)
X         /= scale
  → train / generate →
ratio      = exp(cumsum(sample · scale))       (M_simu, 127, 1)
S[:, 0]    = S0 = 100 ; S[:, 1:] = S0 · ratio  (M_simu, 128)
```

`d = 1` on the benchmark: the dataset also ships `heston_v_8192x128.npy`, but every other method
here sees price only, so feeding SBBTS the variance channel would be privileged information. The
paper's own experiment *is* bivariate (`d = 2`) because its metric is a (price, variance) MLE —
that run lives in [`../paper_reimplementation/`](../paper_reimplementation/).

## Files
| File | Role |
|------|------|
| `sbbts_torch.py` | self-contained port: `ScoreNN`, `EncoderOnly`, `MLP`, `InverseMLP`, `EarlyStopping`, `get_loss`, `training_sbbts_dsbm`, `generate_dsbm` |
| `train_seed.py` | one seed, one GPU → generated paths + weights + loss CSV |
| `train.py` | 5-seed orchestrator, GUIDELINE §4.2 batching, hard 2-GPU cap |
| `reference/` | verbatim mirror of github.com/alexouadi/SBBTS (unmodified) |

## Environment
- Python: 3.12.3 (`/home/tbasseras/gpu-venv`)
- PyTorch: 2.13.0+cu130
- NumPy 2.4.6 · SciPy 1.18.0 · Numba 0.66.0 (Numba only for the paper's MLE metric)
- GPU: A100-SXM4-80GB (the paper reports a single A100-SXM4 40 GB)
- Training time per seed: _to be filled after the 5-seed run_

## How to change hyperparameters

Two places, and only two. `train_seed.py` holds the authoritative defaults in the `CFG` dict
(lines 50-56); anything exposed as a CLI flag can be overridden without editing the file.

| Parameter | Default | How to change |
|-----------|---------|---------------|
| `beta` | 100 | `--beta 300` (CLI) |
| `K` (outer DSBM iterations) | 5 | `--K 8` (CLI) |
| `n_epochs` (inner, per iteration) | 1000 | `--n-epochs 500` (CLI) |
| `N_pi` (Euler substeps) | 60 | `--N-pi 120` (CLI) |
| `d_model` | 128 | `--d-model 256` (CLI) |
| `nhead` | 32 | `--nhead 16` (CLI) |
| `n_layers` | 2 | `--n-layers 4` (CLI) |
| `hidden_dim` | 64 | `--hidden-dim 128` (CLI) |
| `M_simu` (paths generated) | 8192 | `--M-simu 1000` (CLI) |
| `T` | 1 | **edit `CFG["T"]`** — no flag |
| `safe_t` | 1e-2 | **edit `CFG["safe_t"]`** — no flag |
| `batch_size` | 128 | **edit `CFG["batch_size"]`** — no flag |
| `lr` | 1e-3 | **edit `CFG["lr"]`** — no flag |
| `patience` / `delta` | 15 / 1e-3 | **edit `CFG["patience"]` / `CFG["delta"]`** — no flag |
| `N_batch` (generation chunks) | 4 | **edit `CFG["N_batch"]`** — no flag |

`train.py` forwards unknown arguments verbatim (`args, extra = ap.parse_known_args()`, line 102),
so any `train_seed.py` flag can be passed through the 5-seed orchestrator:
`train.py --seeds 0,1,2,3,4 --beta 300` reaches every child process.

`train_seed.py` records `paper_hyperparams: true/false` in `weights/seed_{n}_config.json` (line 182),
computed from whether `beta, K, N_pi, d_model, nhead, n_layers` all still equal `CFG`. Any tuned run
is therefore self-labelling — you cannot silently ship a swept config as the paper's.

> The sweep in [`../paper_reimplementation/`](../paper_reimplementation/) tested 27 arms of exactly
> these knobs and found **no reproducible winner**; `β = 100` is the shipped value because it is the
> authors', not because it won. See that README for the seed-noise analysis.

## How to use a different dataset

`train_seed.py` takes `--data /path/to/file.npy`. Requirements:

| Property | Required | Enforced at |
|----------|----------|-------------|
| Format | `.npy`, `np.load`-able | line 117 |
| Shape | `(M, 128)` — paths × time, **price levels, not returns** | `assert S_train.shape[1] == SEQ_LEN`, line 118 |
| dtype | any float (cast to `float32` on GPU) | `to_scaled_log_returns` |
| Scale | **strictly positive** — `np.log(S)` is taken | `to_scaled_log_returns`, line 122 |

Scaling is handled internally: the code takes `diff(log S)`, divides by the empirical per-channel
std, trains, then inverts with `S[:, 1:] = S0 · exp(cumsum(sample · scale))`. **Do not pre-normalise.**

To change the sequence length, edit `SEQ_LEN` (line 47) — the assert will otherwise reject the file.
To change the starting price of generated paths, edit `S0` (line 46, default 100.0). For `d > 1`
(multi-channel), the port already supports it — the paper reimplementation runs `d = 2` — but
`train_seed.py` hard-codes the single-price transform, so use
[`../paper_reimplementation/metric/reproduce_heston.py`](../paper_reimplementation/metric/) instead.

## How to produce new seeds

```bash
cd /home/tbasseras/benchmark/methods/SBBTS/code

# one seed
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  taskset -c 0-1 /home/tbasseras/gpu-venv/bin/python train_seed.py --seed 7 --gpu 0

# a new batch of seeds through the orchestrator (it sets the env vars itself)
/home/tbasseras/gpu-venv/bin/python train.py --seeds 5,6,7,8,9 --gpus 0,1 --jobs-per-gpu 3
```

Outputs land under `methods/SBBTS/` (override with `--out-root`):

| Path | Content |
|------|---------|
| `generated_paths/seed_{n}/generated_paths_8192x128.npy` | generated price paths |
| `weights/seed_{n}_model.pt` | `ScoreNN` state dict |
| `weights/seed_{n}_transport.pt` | `{y_0, scale}` — **required** to regenerate paths |
| `weights/seed_{n}_config.json` | full resolved config + `paper_hyperparams` flag |
| `losses/seed_{n}_losses.csv` | `step,phase,loss_total,val_loss` |
| `losses/train_seed_{n}.log` | stdout/stderr |

`train.py` refuses more than `MAX_GPUS = 2` (line 106-107) — this is a hard `sys.exit`, not a warning.
Use `--dry-run` to print the GPU/core assignment without launching.

## Reproduce (EXACT run path)

This is what actually produced the committed numbers, not a generic recipe.

**Driver.** [`../run_pipeline.sh`](../run_pipeline.sh), launched detached:

```bash
cd /home/tbasseras/benchmark
setsid bash methods/SBBTS/run_pipeline.sh \
  >> methods/SBBTS/losses/pipeline.log 2>&1 < /dev/null & disown
```

**Stage 1 — training + generation.** The driver runs exactly:

```bash
/home/tbasseras/gpu-venv/bin/python \
  /home/tbasseras/benchmark/methods/SBBTS/code/train.py \
  --seeds 0,1,2,3,4 --gpus 0,1 --jobs-per-gpu 3 --beta 100
```

which spawns five `train_seed.py` children. Per child, `train.py::launch` (lines 73-91) sets
`CUDA_VISIBLE_DEVICES=<gpu>`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`,
`cwd=methods/SBBTS/code`, and prefixes `taskset -c <slice>`. The resulting placement (verified with
`--dry-run`) is:

| Seed | GPU | Cores | Command |
|------|-----|-------|---------|
| 0 | 0 | 0-1 | `taskset -c 0-1 … train_seed.py --seed 0 --gpu 0 --beta 100` |
| 1 | 0 | 2-3 | `taskset -c 2-3 … train_seed.py --seed 1 --gpu 0 --beta 100` |
| 2 | 0 | 4-5 | `taskset -c 4-5 … train_seed.py --seed 2 --gpu 0 --beta 100` |
| 3 | 1 | 8-9 | `taskset -c 8-9 … train_seed.py --seed 3 --gpu 1 --beta 100` |
| 4 | 1 | 10-11 | `taskset -c 10-11 … train_seed.py --seed 4 --gpu 1 --beta 100` |

**10 cores, 2 GPUs** — inside the GUIDELINE §4.1 caps (16 cores, 2 GPUs).

**Input consumed:** `dataset/Heston/heston_S_8192x128.npy`, shape `(8192, 128)`, price levels.
The variance channel `heston_v_8192x128.npy` exists but is **deliberately not read** — see *Data transform*.

**Outputs produced, and which number each one backs:**

| Output | Shape | Backs |
|--------|-------|-------|
| `generated_paths/seed_{0..4}/generated_paths_8192x128.npy` | `(8192, 128)` | **every A1-A34 and B-curve cell** — scored by `metrics/compute_all.py --method SBBTS --dataset Heston --seeds 5`, which writes `results/Heston/SBBTS/metrics_A.json` and `metrics_B.json` |
| `weights/seed_{n}_model.pt` + `seed_{n}_transport.pt` | — | regenerate the paths above; **both** are needed (`scale` and `y_0` are not recoverable from the model alone) |
| `losses/seed_{n}_losses.csv` | `step,phase,loss_total,val_loss` | `losses/loss_convergence.png`, via `code/plot_losses.py` |
| `results/Heston/SBBTS/path_shadowing/` | — | the PS-MC table, via `path_shadowing/run_eval.py`, which reads **seed 0's** generated paths |

**Stages 2-4** (metrics, figures, path-shadowing) are the remaining commands in `run_pipeline.sh`;
each writes its own log under `losses/`. Every reported cell is therefore traceable to one `.npy`
plus one command.

> **Note on `run_benchmark.sh`.** The original single-file driver is still present and still valid,
> but its stage-0 gate waits for *every* `reproduce_heston.py` process to exit, which two
> long-patience sweep arms made unbounded. It now `exec`s `run_pipeline.sh`, which uses a counting
> gate plus a lock. Either entry point produces the same stages 1-4.

## Sanity check

`code/plot_losses.py` with no seed CSVs present exits cleanly rather than producing an empty figure:

```
$ /home/tbasseras/gpu-venv/bin/python methods/SBBTS/code/plot_losses.py
no seed_*_losses.csv under /home/tbasseras/benchmark/methods/SBBTS/losses; run training first
```

`train.py` refuses an over-wide GPU request:

```
$ /home/tbasseras/gpu-venv/bin/python train.py --gpus 0,1,2 --dry-run
REFUSING: 3 GPUs requested, hard cap is 2.
```
