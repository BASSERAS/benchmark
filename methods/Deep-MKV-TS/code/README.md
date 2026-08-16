# Deep-MKV-TS — code

This directory holds **the exact code that produced every Deep-MKV-TS number in this
benchmark**, plus the one adapter script that converts the training run tree into the
flat layout the benchmark protocol requires.

If you only want to *run* it, jump to [§9 Reproduce](#9-reproduce-copypaste).
If you want to know *what* it is, start at §1.

---

## 1. Original work

| | |
|---|---|
| **Paper** | *Deep McKean–Vlasov Time Series* (Deep-MKV-TS) — ICAIF submission |
| **Local copy** | [`../paper_reimplementation/Deep-MKV-TS_ICAIF.pdf`](../paper_reimplementation/Deep-MKV-TS_ICAIF.pdf) |
| **Reference model** | Guyon & Lekeufack (2023), *Volatility is (Mostly) Path-Dependent* — paper reference `[11]`, described in §2.1 |
| **Upstream code** | vendored under [`reference/`](reference/); see [`reference/UPSTREAM_README.md`](reference/UPSTREAM_README.md) |
| **Licence** | [`reference/LICENSE`](reference/LICENSE) |
| **Table reproduced** | Table 1 (Heston), medians over seeds 0, 1, 3, 4 |

---

## 2. Our implementation

**There is no re-implementation.** We run the authors' own reference package directly.
The only Python we added is one adapter:

| Path | Role |
|---|---|
| `reference/` | Authors' package, vendored verbatim except for **one** edit (see §4) |
| `reference/src/deep_mkv_gen_path_dt/` | The model: networks, controls, discrepancies, conditional-expectation (`ce/`) estimators |
| `reference/experiments/scripts/run_matched_control_synthetic_validation.py` | The training + evaluation entry point. **This is the script that trains the model.** |
| `export_benchmark_artifacts.py` | **Ours.** Converts `paper_reimplementation/runs/seed_*/` into `generated_paths/`, `weights/`, `losses/` per GUIDELINE §4.3–§4.4 |

### What the method does, in one paragraph

Deep-MKV-TS is a **path-dependent McKean–Vlasov** generative model. It does *not* learn a
generator from scratch. It starts from a **frozen, interpretable reference SDE** — the
Guyon–Lekeufack path-dependent volatility model — and learns a **volatility correction**
on top of it. The drift stays fixed at the reference drift. Training minimises

```
  MMD-based discrepancy between generated and target path laws
+ eta * specific-entropy running cost   (eta = 1)
```

The correction is parameterised through the **adjoint of the noise coefficient**, predicted
by a small GRU. Because the drift is frozen and the correction is entropy-penalised, the
generated law stays close to an interpretable model rather than drifting into an arbitrary
black-box fit.

### The variant we run: `volatility_only_online_mp`

Every run directory contains an arm named `volatility_only_online_mp`. That name decodes as:

- **`volatility_only`** — only the volatility (noise) adjoint head is trained
  (`--adjoint-weight 0 --adjoint-noise-weight 1`), and the drift is pinned to the fitted
  reference (`--fitted-reference-drift-only`). This is the paper's setting.
- **`online`** — the target-path bank is resampled online each step (`--solver online`)
  rather than pre-generated once.
- **`mp`** — maximum-principle formulation (the adjoint/BSDE view), as opposed to a direct
  pathwise-gradient formulation.

The sibling `reference/` sub-directory inside each run holds the **Reference row** of
Table 1: the frozen Guyon–Lekeufack model scored with no learned correction at all.

---

## 3. Architecture (read off the trained checkpoint, not transcribed from the paper)

Dumped from `paper_reimplementation/runs/seed_0/volatility_only_online_mp/training_checkpoints/step_2500.pt`:

| Field | Value |
|---|---|
| `state_dim` | 1 |
| `noise_dim` | 1 |
| `hidden_dim` | 96 |
| `num_layers` | 1 |
| `adjoint_input_mode` | `level` |
| **Total parameters** | **47,330** |

```
gru.weight_ih_l0                       (288, 1)     # 3 * 96 gates, input dim 1
gru.weight_hh_l0                       (288, 96)
gru.bias_ih_l0                         (288,)
gru.bias_hh_l0                         (288,)
expected_adjoint_next_head.0.weight    (96, 96)     # Linear(96, 96)
expected_adjoint_next_head.0.bias      (96,)
expected_adjoint_next_head.2.weight    (1, 96)      # Linear(96, 1)
expected_adjoint_next_head.2.bias      (1,)
expected_adjoint_noise_next_head.0.weight (96, 96)
expected_adjoint_noise_next_head.0.bias   (96,)
expected_adjoint_noise_next_head.2.weight (1, 96)
expected_adjoint_noise_next_head.2.bias   (1,)
```

So: **one GRU layer (hidden 96)** feeding **two MLP heads**, each
`Linear(96, 96) → activation → Linear(96, 1)`. Index `.2` is the second `Linear` because
`.1` is the activation inside the `nn.Sequential`.

Only `expected_adjoint_noise_next_head` receives gradient in our configuration
(`--adjoint-weight 0`). `expected_adjoint_next_head` is allocated and saved but stays at
its initialisation.

> **⚠️ Checkpoint format warning.** A checkpoint file is a **dict**, not a bare
> `state_dict`. Keys:
> `format_version, architecture, training, network_state_dict, adjoint_target_mean,
> adjoint_target_scale, noise_adjoint_target_mean, noise_adjoint_target_scale,
> noise_target_timewise_baseline, target_preconditioner_metrics, adjoint_source`.
>
> `noise_adjoint_target_mean` and `noise_adjoint_target_scale` are the
> **output de-standardisation constants**. The network predicts in standardised space;
> without these two tensors the raw head output is meaningless. If you load
> `network_state_dict` alone and sample, you will get garbage and it will *look* like a
> training failure. Load the whole dict.

```python
import torch
ckpt = torch.load("weights/seed_0_model.pt", map_location="cpu", weights_only=False)
arch = ckpt["architecture"]          # build the net from this
net.load_state_dict(ckpt["network_state_dict"])
mu, sd = ckpt["noise_adjoint_target_mean"], ckpt["noise_adjoint_target_scale"]
# prediction in real units = net(x) * sd + mu
```

---

## 4. Fixes applied to the upstream code

**Exactly one edit.** Everything else in `reference/` is byte-identical to upstream.

### Fix 1 — dataset path resolution (`run_matched_control_synthetic_validation.py:244-271`)

*Why:* upstream hard-coded a path relative to its own repo root. In this benchmark the
Heston dataset lives in the **shared** `dataset/Heston/` tree (GUIDELINE §2 forbids
duplicating it per-method), and the benchmark protocol needs to point the evaluation at
different `.npy` splits (`disc` for reporting, `valdisc` for hyperparameter search)
**without editing code** — because editing code between a search run and a reporting run
is exactly how selection-on-eval leaks happen.

```python
def _benchmark_heston_root() -> Path:
    override = os.environ.get("BENCHMARK_HESTON_DIR")
    if override:
        return Path(override)
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "dataset" / "Heston"
        if (candidate / "heston_S_8192x128.npy").is_file():
            return candidate
    raise FileNotFoundError(...)

# inside _task_paths():
eval_name = os.environ.get("BENCHMARK_HESTON_EVAL", "heston_S_disc_8192x128.npy")
return (root / "heston_S_8192x128.npy", root / eval_name, None)
```

Two environment variables, both optional:

| Variable | Default | Meaning |
|---|---|---|
| `BENCHMARK_HESTON_DIR` | auto-discovered by walking up to `dataset/Heston/` | Where the `.npy` splits live |
| `BENCHMARK_HESTON_EVAL` | `heston_S_disc_8192x128.npy` | Which split the run is *scored* on |

Training always reads `heston_S_8192x128.npy`. Only the evaluation split is switchable.

**No other fix was needed.** No numerical patch, no shape patch, no dependency pin change.

---

## 5. Hyperparameters

Every value below is passed explicitly on the command line so the run is fully specified by
its invocation — nothing is inherited silently from a config file.

| Hyperparameter | Value | Source in the paper | CLI flag |
|---|---|---|---|
| Task | `heston` | §4 Experiments, Table 1 | `--task heston` |
| Reference model | Guyon–Lekeufack structural likelihood | §2.1, ref `[11]` | `--reference-kind guyon_lekeufack_structural_likelihood` |
| Reference activity update | structural variance, driven by `(v^ref)²` | eq. (2) | (default for this reference kind) |
| Reference calibration fraction | 0.8 fit / 0.2 select | §2.1 | (default) |
| Drift | frozen at fitted reference | §2 (drift not learned) | `--fitted-reference-drift-only` |
| Adjoint (drift) weight | 0 | §2, volatility-only correction | `--adjoint-weight 0` |
| Adjoint (noise) weight | 1 | §2, volatility-only correction | `--adjoint-noise-weight 1` |
| Entropy weight η | 1 | §3 objective | `--eta 1` |
| λ scale | 50 | Table 6 / App. B | `--lambda-scale 50` |
| κ scale | 100 | Table 6 / App. B | `--kappa-scale 100` |
| Learning rate | 2e-3 | Table 6 / App. B | `--lr 0.002` |
| Gradient clip (norm) | 5 | Table 6 / App. B | `--grad-clip-norm 5` |
| Bank size | 8192 | matches dataset size | `--bank-size 8192` |
| Sample batch size | 2048 | App. B | `--sample-batch-size 2048` |
| Joint-volatility weight | 0 | Table 6 | `--joint-volatility-weight 0 --source-joint-volatility-weight 0` |
| \|r\| ACF weight | 0.25 | Table 6 | `--abs-return-acf-weight 0.25` |
| r² ACF weight | 0.125 | Table 6 | `--squared-return-acf-weight 0.125` |
| Training steps | 3000 (report step **2500**) | App. B, K = 2500 | `--source-steps 3000 --source-checkpoint-steps 500 1000 1500 2000 2500 3000` |
| Solver | online | App. B | `--solver online` |
| Path-derivative backend | autograd | implementation detail | `--path-derivative-backend autograd` |
| Drift-adjoint backend | analytical reference | implementation detail | `--drift-adjoint-backend analytical_reference` |

Internal defaults visible in the checkpoint's `training` dict (not overridden):
`batch_size 256`, `target_batch_size 256`, `weight_decay 1e-5`, `ce_target_mode ridge`,
`ridge_lambda 1e-3`, `ce_crossfit_folds 1`, `target_preconditioner none`,
`noise_target_estimator score`, `log_every 100`.

### Why train 3000 steps but report step 2500?

The paper reports **K = 2500**. We train 3000 steps and evaluate the step-2500 checkpoint.

This is **bitwise identical** to stopping at 2500, because **there is no learning-rate
scheduler anywhere in the codebase** — the optimiser state at step 2500 does not depend on
whether the loop is going to keep going afterwards. Grep for `lr_scheduler` /
`LambdaLR` / `CosineAnnealing` in `reference/` and you will find nothing. Training the extra
500 steps costs ~4 minutes and gives a free look at whether the metric is still moving.

---

## 6. How to change a hyperparameter

Two places, and only two:

**(a) On the command line.** Everything in the table above. Edit the invocation in
[`../paper_reimplementation/metric/run_reproduction.sh`](../paper_reimplementation/metric/run_reproduction.sh)
or write your own — the flags are self-contained.

```bash
# example: sweep kappa
--kappa-scale 200
```

**(b) In the reference package defaults**, if the knob has no flag:

| What you want to change | File |
|---|---|
| Network width / depth / activation | `reference/src/deep_mkv_gen_path_dt/networks/` |
| Control parameterisation, adjoint heads | `reference/src/deep_mkv_gen_path_dt/controls/` |
| MMD kernels, discrepancy terms | `reference/src/deep_mkv_gen_path_dt/discrepancies/` |
| Conditional-expectation estimator (ridge, cross-fitting) | `reference/src/deep_mkv_gen_path_dt/ce/` |
| Argument parser, defaults, run layout | `reference/experiments/scripts/run_matched_control_synthetic_validation.py` |

> **⚠️ Selection discipline.** If you tune anything, tune it against
> `BENCHMARK_HESTON_EVAL=heston_S_valdisc_8192x128.npy`. Do **not** tune against
> `heston_S_disc_8192x128.npy` — that is the reporting split, and selecting on it
> invalidates the reported number. Score the winner on `disc` **exactly once**, then stop.
> `heston_S_test_8192x128.npy` is off-limits entirely for this method
> (`test_split_access_authorized: false` in `PROTOCOL.json`).

> **⚠️ Noise floor.** Re-running the *same* config with a different seed moves the
> aggregate score by σ ≈ 0.26–0.44, and individual columns (Early-future especially) move
> by up to 9.5×. A one-seed improvement of 0.1 is noise, not a result. Rank on the median
> over seeds that were **never used to select** — see
> [`../paper_reimplementation/metric/rank_confirm.py`](../paper_reimplementation/metric/rank_confirm.py),
> which exists precisely to enforce this.

---

## 7. How to use a different dataset

No code edit required:

```bash
export BENCHMARK_HESTON_DIR=/absolute/path/to/your/npy/dir
export BENCHMARK_HESTON_EVAL=my_eval_split.npy
```

Requirements for the directory:

- must contain `heston_S_8192x128.npy` — the **training** paths;
- must contain whatever file `BENCHMARK_HESTON_EVAL` names — the **evaluation** paths;
- both are `float32` or `float64` arrays of shape `(n_paths, n_steps)` holding **price
  levels** (not returns, not log-prices), strictly positive, starting at the same `S₀`.

The pipeline computes log-returns internally. It does not standardise for you, and it does
not accept a `(n, t, d)` tensor for the Heston task — `d = 1` is baked into `--task heston`.

---

## 8. How to produce new seeds

```bash
cd /home/tbasseras/benchmark/methods/Deep-MKV-TS/paper_reimplementation/metric
bash run_reproduction.sh          # seeds 0, 1, 3, 4 — the paper's set, 2 GPUs, 2 at a time
bash run_seed2.sh                 # seed 2 — fills the GUIDELINE 0..4 requirement
```

`run_reproduction.sh` runs seeds **0, 1, 3, 4** because those are the four the paper
reports medians over. GUIDELINE §4 requires the benchmark's 5-seed set **0..4**, so
`run_seed2.sh` adds seed 2 with a byte-identical flag block. `run_seed2.sh` deliberately
leaves `BENCHMARK_HESTON_EVAL` **unset** so seed 2 is scored on the same `disc` split as
the other four — setting it would silently pool one seed from a different split. It also
blocks until every other trainer has exited (two trainers writing one run tree corrupts
both, and GUIDELINE §4.1 caps us at 2 GPUs).

Both scripts skip a seed whose `COMPLETE.json` already exists, so they are safe to re-run.

### Run tree produced per seed

```
paper_reimplementation/runs/seed_0/
├── COMPLETE.json                      # {"seed": 0, "steps": 3000} — written last, the done marker
├── console.log                        # full stdout/stderr of the trainer
├── reference/                         # the FROZEN Guyon-Lekeufack model = Table 1 "Reference" row
│   ├── metrics.json
│   ├── control_diagnostics.json
│   └── validation_bank.npy
└── volatility_only_online_mp/         # the TRAINED model = Table 1 "Deep-MKV-TS" row
    ├── metrics.json                   # <- final scored metrics
    ├── source_history.jsonl           # <- per-step losses (one JSON object per logged step)
    ├── run_manifest.json              # <- full resolved config, incl. bank_seed
    ├── control_diagnostics.json
    ├── validation_bank.npy
    ├── source_model_checkpoint.pt
    ├── training_checkpoints/
    │   ├── step_0500.pt  step_1000.pt  step_1500.pt
    │   └── step_2000.pt  step_2500.pt  step_3000.pt      # <- 2500 is the reported one
    └── checkpoint_evaluations/
        ├── step_0500/ ... step_2500/ ... step_3000/      # metrics + sampled paths per checkpoint
```

Sampling noise is independent across seeds: `bank_seed = 70000 + seed`, so the five runs are
five genuine draws, not one draw re-scored five times.

### Then export to the benchmark layout

```bash
cd /home/tbasseras/benchmark/methods/Deep-MKV-TS
/home/tbasseras/gpu-venv/bin/python code/export_benchmark_artifacts.py
```

This reads `runs/seed_{0..4}/volatility_only_online_mp/` at step 2500 and writes:

| Output | Content |
|---|---|
| `generated_paths/seed_{i}/generated_paths_8192x128.npy` | `float64`, shape `(8192, 128)`, price levels |
| `weights/seed_{i}_model.pt` | the step-2500 checkpoint dict (see §3) |
| `losses/seed_{i}_losses.csv` | `step,phase,loss_total,discrepancy_objective,complete_objective,running_cost,grad_norm,mmd_*` |
| `losses/seed_{i}_convergence.png` | 4-panel 1600×900 @150 dpi, log-y on total loss and discrepancy |
| `metadata.json` | seeds, steps, wall-clock train time, `bank_seed`, device, config hash |

Two deliberate choices in the exporter, so you are not surprised:

- It **asserts** paths are strictly positive rather than clipping them. A geometric SDE
  cannot emit a non-positive price; a clip would hide a real bug.
- Missing loss keys are written as `""`, not `0.0`. `0.0` is a legitimate value for
  `running_cost` and conflating the two would corrupt the convergence plot.

---

## 9. Reproduce (copy-paste)

### Environment

```bash
python -m venv ~/gpu-venv
~/gpu-venv/bin/pip install torch numpy scipy matplotlib
```

Python ≥ 3.11 (`reference/pyproject.toml`). The package itself only declares `torch`;
`numpy`/`scipy`/`matplotlib` are needed by the benchmark metric and export layers.
You do **not** need to `pip install` the reference package — `PYTHONPATH` is enough.

### One seed, literal command, nothing hidden

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
  --drift-adjoint-backend analytical_reference
```

Replace `/home/tbasseras/benchmark` with your clone root and
`/home/tbasseras/gpu-venv/bin/python` with your interpreter. Nothing else needs touching.

**Cost:** ≈ 24 min per seed on one A100-80GB. Five seeds on 2 GPUs, two at a time ≈ 1 h 15.
GPU memory ≈ 6 GB, so two seeds fit comfortably on one card if you are in a hurry — but
GUIDELINE §4.1 caps the whole job at 2 GPUs and 8 cores each, hence the `taskset`.

### Full pipeline

```bash
cd /home/tbasseras/benchmark/methods/Deep-MKV-TS/paper_reimplementation/metric
bash run_reproduction.sh      # seeds 0,1,3,4  -> runs/seed_{0,1,3,4}/
bash run_seed2.sh             # seed 2         -> runs/seed_2/
python aggregate_paper_table.py   # -> the paper-vs-ours Table 1 comparison
cd ../..
/home/tbasseras/gpu-venv/bin/python code/export_benchmark_artifacts.py
```

### Traceability — which output produced which reported number

| Reported number | Comes from |
|---|---|
| Table 1 **Reference** row | `runs/seed_{i}/reference/metrics.json` |
| Table 1 **Deep-MKV-TS** row | `runs/seed_{i}/volatility_only_online_mp/checkpoint_evaluations/step_2500/metrics.json` |
| Paper-vs-ours comparison table | `metric/aggregate_paper_table.py` over the four paper seeds `{0,1,3,4}` |
| Benchmark A1–A34 / B curves | `metrics/compute_all.py` over `generated_paths/seed_{0..4}/` |
| Loss curves in `losses/` | `runs/seed_{i}/volatility_only_online_mp/source_history.jsonl` |
| Saved weights | `runs/seed_{i}/volatility_only_online_mp/training_checkpoints/step_2500.pt` |
| Config actually used | `runs/seed_{i}/volatility_only_online_mp/run_manifest.json` |

The evaluation split is pinned in
[`../paper_reimplementation/metric/PROTOCOL.json`](../paper_reimplementation/metric/PROTOCOL.json)
(`evaluation_split: dataset/Heston/heston_S_disc_8192x128.npy`,
`locked_reporting_checkpoint: 2500`, `selection_scope: validation_only`,
`test_split_access_authorized: false`). The trainer reads that manifest and refuses to run
against a split it does not declare.

---

## 10. Sanity check (30 seconds, no GPU)

```bash
cd /home/tbasseras/benchmark/methods/Deep-MKV-TS
/home/tbasseras/gpu-venv/bin/python - <<'PY'
import torch, numpy as np
ck = torch.load("paper_reimplementation/runs/seed_0/volatility_only_online_mp/"
                "training_checkpoints/step_2500.pt", map_location="cpu", weights_only=False)
sd = ck["network_state_dict"]
print("arch  :", ck["architecture"])
print("params:", sum(v.numel() for v in sd.values()))
x = np.load("generated_paths/seed_0/generated_paths_8192x128.npy")
print("paths :", x.shape, x.dtype, float(x.min()), float(x.max()))
PY
```

Expected:

```
arch  : {'state_dim': 1, 'noise_dim': 1, 'hidden_dim': 96, 'num_layers': 1,
         'adjoint_dim': None, 'noise_adjoint_dim': None, 'adjoint_input_mode': 'level'}
params: 47330
paths : (8192, 128) float64  <positive min>  <max of order 1e2>
```

If `params` is not `47330`, you are loading a different architecture than the one that
produced the reported numbers. If `x.min()` is ≤ 0, the export assertion should have fired —
file a bug rather than clipping.
