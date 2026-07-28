# GUIDELINE — adding a method to the `preprocessing_with_log_returns` experiment

This is the step-by-step recipe for reproducing, and extending, the **SBTS log-return
preprocessing** experiment on the Heston benchmark. It is the analog of the root
[`GUIDELINE.md`](../../../GUIDELINE.md) (which explains adding a brand-new method to the main
benchmark), but specialized for this experiment: *take an existing generator, feed it
SBTS-style volatility-scaled log-returns instead of standardized prices, map the output back to
price space, and score it with the untouched benchmark metrics.*

**The worked example is LS4** (`LS4/`). Follow it exactly; every path, constant, and gotcha
below was learned building it.

> **Golden scope rule (never break):** only ever create/edit files **inside**
> `dataset/Heston/preprocessing_with_log_returns/` and
> `results/Heston/preprocessing_with_log_returns/`. Every piece of reference code (the method's
> model, `metrics/compute_all.py`, `metrics/plot_diagnostics.py`, the parent Heston generator)
> is **imported, never modified**. This keeps the numbers directly comparable to the main
> benchmark and prevents collateral damage.

---

## 0. The clarifications this experiment was built on (Q&A)

These are the exact questions asked while building LS4 and the answers that define the protocol.
Any new method inherits these answers unless told otherwise.

| # | Question | Decision |
|---|----------|----------|
| Q1 | How many **train** paths? | Regenerate a **fresh 4096** (seed 0). Not the main benchmark's 8192. |
| Q2 | How many **test / disc** paths? | **4096** for every A/B metric. **Exception:** path-shadowing uses a **512**-path fresh query split (seed 3), strictly independent of the 1M generated bank, per the PDF protocol. |
| Q3 | How many paths to **generate** for the A/B metrics? | **4096** (`generated_paths_4096x128.npy`), matched to the test split. |
| Q4 | How big is the **path-shadowing bank**? | **1,000,000** generated paths, from a **single seed**. Document precisely how to extend to 5 seeds (§9.5). |
| Q5 | Which **path-shadowing protocol**? | **Full PDF protocol** (arXiv:2308.01486): K=256, prefix 64 / horizon 32, additive endpoint alignment, CRPS energy score, 512 held-out real queries. |
| Q6 | What is the **metrics gate**? | After **seed 0 only**, compute the A/B + B-curve metrics and compare to the original method's seed 0. If there is a **big deviation**, flag it. If the metrics are **bad, STOP** — show old-vs-new diagnostics figures, do a hyperparameter check, and ask before proceeding. Only after approval train seeds 1–4 and build the 1M bank. |
| Q7 | Folder naming? | Underscores: `preprocessing_with_log_returns`. |
| Q8 | Seed plan for the 5 runs? | Pair on 2 GPUs: (0,1) then (2,3) then 4, `taskset`-pinned, `CUDA_VISIBLE_DEVICES` set per process (§6.2). |

### 0.1 The LS4 gate lesson (READ THIS before you pick a method)

The **first** LS4 gate **failed catastrophically** and the fix is now baked into the recipe.

- SBTS's transform scales returns to `std = sqrt(dt) = 0.063`.
- LS4's released decoder has a **fixed Gaussian observation noise `sigma = 0.1`** — *larger than
  the entire 0.063 signal*. Signal-to-noise < 1, so the VAE treated the returns as pure noise,
  collapsed to the mean, and generated near-flat paths (~20× under-dispersion; terminal std 0.45
  vs real 9.9). Every dispersion metric was 10×–1000× worse.
- **Fix:** after the exact SBTS transform, apply the method's **own input standardization**
  (global mean/std → unit variance), then invert symmetrically. This restores the SNR the model
  was tuned for. The SBTS `sigma` is untouched and still reported. See §3.3.

**Takeaway for any new method:** models with a **fixed output-noise scale** (VAEs, some flows)
need **unit-variance input**. Models that are scale-robust (diffusion like SBTS, GANs with
learned output scale) may tolerate the raw `sqrt(dt)` scale. The gate (§7) is what tells you
which camp your method is in — *never skip it.*

### 0.2 Mistakes already made — DO NOT repeat them

These were made building LS4 and cost real time. Each has a one-line guard.

| # | Mistake | What happened | Guard (do this) |
|---|---------|---------------|-----------------|
| M1 | **Epoch count left to the default.** | The train script's `--epochs` **defaults to 400**. Relaunching seed 0 as `python train_..._logret.py --seed 0` (no `--epochs`) silently ran **400** epochs while the canonical baseline is **100**. That (a) confounds the gate — you change *two* variables (preprocessing **and** 4× training) instead of one — and (b) wasted ~20 min before it was caught. | **Always pass `--epochs` explicitly, matching the original method.** Verify the canonical count from the original loss CSV: `wc -l methods/<METHOD>/losses/seed_0_losses.csv` (rows − 1 header = epochs; LS4 = **100**). The *only* intended differences vs the main benchmark are the §3 preprocessing and 4096 paths — **nothing else**, epochs included. |
| M2 | **Training piped through `\| tail -50`.** | `python train.py ... 2>&1 \| tail -50` buffers **all** output until the process exits, and the loss CSV / weights are only written at the end. Result: **zero live progress**, no way to estimate ETA or notice a stall for the whole run. | **Never pipe training through `tail`/`head`.** Launch with `run_in_background` (or redirect to a log file) so the per-epoch `[ep N] ...` lines (printed with `flush=True`) stream live. Sample two epoch lines to get s/epoch → ETA. |
| M3 | **Assuming the GPU was busy at 44%.** | A single LS4 process uses only ~44% of an A100 (2.1M params — too small to saturate it). Treating that as "GPU in use" left ~half the card idle. | **Pack independent seeds onto one GPU** (pin to separate CPU cores: seed 0 → `taskset -c 0-7`, seed 1 → `-c 8-15`). Two LS4 seeds together reach ~95% util at ~6.6 GB — full GPU, one card, no one else's allocation touched. |
| M4 | **Comparability drift in general.** | The root purpose is "change *only* the preprocessing." Any silent divergence (epochs, batch size, preset, sample count beyond the intended 4096) invalidates the gate comparison. | Diff your `seed_N_config.json` against the original method's hyperparameters **before** training seed 0. If a field differs and it isn't the §3 scaler or the 4096 count, fix it. |
| M5 | **Env vars placed *after* `taskset`.** | Launching with `taskset -c 0-3 OMP_NUM_THREADS=4 python train.py` fails instantly: `taskset` treats `OMP_NUM_THREADS=4` as the **command to exec** → `taskset: failed to execute OMP_NUM_THREADS=4: No such file or directory`, exit code **127**. The three seeds "launched" but every one died in <1 s; only caught because `nvidia-smi` showed GPU 0 back at **0 %**. | **Env assignments must come *before* `taskset`** (they are shell prefixes, `taskset` is the command): `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 taskset -c 0-3 $PY train.py …`. After launching, **always** verify with `nvidia-smi -i 0` and `pgrep -af train`: expect one PID per seed and GPU memory within seconds. |
| M6 | **Backgrounded launcher let the children die on exit.** | Starting the training loop from a wrapper shell that itself exits (e.g. a `run_in_background` one-liner that spawns `python … &` then returns) sends **SIGHUP** to the children when the parent shell reaps — the jobs vanish and GPU 0 drops to 0 % moments later. | **Detach each job from the launching shell.** Use `setsid` + redirect stdin from `/dev/null` + `disown`: `… taskset -c 0-3 $PY train.py --seed $s --epochs 100 > logs/seed_$s.log 2>&1 < /dev/null &` inside a `for` loop, then `disown -a`. Confirm survival after the launcher returns: `pgrep -af train_…` must still list the PIDs and `nvidia-smi` must show memory. |
| **M7** | **Path-shadowing done with the *simplified* reference eval instead of the paper protocol.** | `methods/<METHOD>/path_shadowing/` (`path_shadowing.py` + `run_eval.py`) is a **reduced** eval: a 65D **murex** embedding, **K=77**, raw prefix-price L2, a single **8192**-path bank, and **CRPS/MAE/RMSE only** at H=32/64. It is **NOT** the arXiv:2308.01486 protocol. Reusing it (as an earlier draft of §0.3/§9.2 wrongly said to) silently answers a *different, easier* question and makes the numbers non-comparable to the paper. | **This experiment uses the STRICT paper protocol — never the reference subset.** Use `<METHOD>/path_shadowing/path_shadowing_pdf.py`: the **4-block weighted, bank-standardized** embedding (recent-returns w1.0 · cumulative-path w0.5 · rolling-vol w2.0 · dependence-ACF w1.0; `z̃ = √w·(z−μ_bank)/σ_bank`), the **bank-size sweep** {4096, 16384, 65536, 262144, 1 000 000} as **nested prefixes of the one 1M bank**, the **three forecast quantities** (cumulative return, one-step return, horizon RV), and the **full metric set** (predictive-mean RMSE, CRPS, coverage 50/90, band width 50/90, lower/upper-90 miss) with **2000-resample bootstrap 95% CIs**. See §9. The old `path_shadowing_mc.py` (CRPS-only, K=256 murex) is **superseded** — kept only as the 1M-bank builder via `gen_banks.py`. |

---

### 0.3 Where to get the code (exact paths, from `benchmark/` root)

Copy/adapt from these — **read them, never edit them** (§4). All paths are relative to the repo root
(`/home/tbasseras/benchmark`).

| You need… | Exact path | Notes |
|-----------|-----------|-------|
| The untouched model class | `methods/<METHOD>/code/reference/` (LS4: `from models.ls4 import VAE`) | `sys.path.insert(0, REFERENCE)` then import — do **not** copy the model into the preprocessing folder. |
| The released config/preset | `methods/LS4/code/reference/configs/monash/vae_solarweekly_released.yaml` | LS4 decoder noise `sigma: 0.1` lives here (line 23) — the root cause of the §0.1 collapse. |
| The original training entry-point (to mirror hyperparameters) | `methods/<METHOD>/code/train_heston.py` | Diff your config against this (M4). |
| **Canonical epoch count** | `wc -l methods/<METHOD>/losses/seed_0_losses.csv` → rows − 1 header | **LS4 = 100.** Pass `--epochs 100` explicitly (M1). |
| The metric implementations | `methods/<METHOD>/` metric code imported by `compute_metrics_logret.py` as `C` | Redirect its `DATA_DIR`/`GENERATED_DIR` to the 4096 preprocessing files; don't fork the metric math. |
| The 1M-bank builder (reuse verbatim) | `LS4/path_shadowing/path_shadowing_mc.py` → `build_bank`, `load_gen_model`; driven by `gen_banks.py` | Prior-sample the trained model + §3 inverse → persist `bank/generated_bank_seed{i}_1000000x128.npy`. **This is the *only* thing to reuse from the old PS code.** |
| **The path-shadowing evaluator** | `<METHOD>/path_shadowing/path_shadowing_pdf.py` (**strict arXiv:2308.01486 protocol** — see §9) | **Do NOT import `methods/<METHOD>/path_shadowing/path_shadowing.py`** — that is the *simplified* reference eval (65D murex, K=77, prefix-price L2, CRPS/MAE/RMSE only). It answers a different, easier question. **(M7)** The paper protocol is self-contained in `path_shadowing_pdf.py` (4-block weighted bank-standardized embedding + bank-size sweep + cum/step/RV quantities + coverage/width + bootstrap CIs). |
| The 4096 datasets + 512 ps queries | `dataset/Heston/preprocessing_with_log_returns/heston_S_{4096x128,test_4096x128,disc_4096x128,ps_512x128}.npy` | ps split = seed 3, strictly independent (§9.3). |
| The checkpoint schema (to rebuild the generator for the bank) | `LS4/weights/seed_{i}_model.pt` → dict `{model, ema_model, sbts_sigma, x_mu, x_sd, seed}` | Rebuild EMA: wrap `VAE` in `torch.optim.swa_utils.AveragedModel`, `load_state_dict(ckpt["ema_model"])`, use `.module`, then `.eval(); .setup_rnn()`. Invert with the **frozen** `sigma,x_mu,x_sd` from the checkpoint (never re-estimate). |

**Correct multi-seed launch template** (fixes M1, M2, M3, M5, M6 at once):

```bash
PY=/home/tbasseras/gpu-venv/bin/python           # NOT ~/.cc-venv — LS4 needs gpu-venv (torch+cu130)
mkdir -p code/logs
i=0
for seed in 2 3 4; do                            # only GPU 0 is free; pack seeds on it
  lo=$((i*4)); hi=$((i*4+3))                      # 4 cores/seed, ≤16-core cap
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    setsid taskset -c ${lo}-${hi} \
    $PY code/train_ls4_logret.py --seed $seed --epochs 100 \
    > code/logs/seed_${seed}.log 2>&1 < /dev/null &
  i=$((i+1))
done
disown -a
sleep 12 && pgrep -af train_ls4_logret && nvidia-smi -i 0   # MUST show 3 PIDs + GPU memory
```

---

## 1. Folder structure (mirror LS4 exactly)

```
results/Heston/preprocessing_with_log_returns/
├── README.md                     # experiment overview + preprocessing + sigma (already exists)
├── GUIDELINE.md                  # this file
└── <METHOD>/                     # e.g. LS4/
    ├── README.md                 # per-method: A/B tables (5 seeds), figures, STRICT PS tables (§9)
    ├── code/
    │   ├── train_<method>_logret.py   # imports reference model, applies §3 preprocessing
    │   ├── compute_metrics_logret.py  # imports metrics/compute_all.py, redirects paths
    │   └── gate_compare.py            # seed-0 old-vs-new table + diagnostics figure
    ├── losses/  seed_{0..4}_losses.csv
    ├── weights/ seed_{N}_model.pt + seed_{N}_config.json
    ├── generated_paths/seed_{N}/generated_paths_4096x128.npy + metadata.json
    ├── plots/   heston_diagnostics.png, pca_tsne_seed0.png, disc/pred loss plots
    ├── path_shadowing/
    │   ├── gen_banks.py               # 1M-bank builder (reuses path_shadowing_mc.build_bank)
    │   ├── path_shadowing_pdf.py      # §9 STRICT paper protocol evaluator (the deliverable)
    │   ├── path_shadowing_mc.py       # SUPERSEDED CRPS-only driver — kept only for build_bank
    │   ├── bank/  generated_bank_seed{0..4}_1000000x128.npy
    │   ├── logs/  gen_seed{i}.log, pdf_run.log     # generation + eval logged (M2/M7)
    │   ├── plots/ pdf_crps_vs_banksize.png, pdf_coverage_calibration.png
    │   └── pdf_results_seed{i}.json + pdf_summary.json
    ├── seed_{N}_metrics.json      # one per seed (from compute_metrics_logret.py)
    ├── metrics_summary.csv        # all seeds
    └── gate_seed0_compare.md      # the §7 gate artifact
```

---

## 2. Dataset (shared by all methods — build once)

Already built by
[`dataset/Heston/preprocessing_with_log_returns/generate_datasets.py`](../../../dataset/Heston/preprocessing_with_log_returns/generate_datasets.py).
It **imports the untouched parent** `generate_heston.generate_heston`, so the SDE, parameters,
Euler-Maruyama full-truncation scheme, and RNG stream are **byte-identical** to the main
benchmark; only the sample counts differ.

```bash
cd dataset/Heston/preprocessing_with_log_returns
python generate_datasets.py     # writes 8 .npy files; prints the SBTS sigma
```

| Split | Seed | N | File (S and v) | Role |
|-------|:----:|:----:|----------------|------|
| train | 0 | 4096 | `heston_{S,v}_4096x128.npy` | only data a generator sees |
| test | 1 | 4096 | `heston_{S,v}_test_4096x128.npy` | held-out real reference (all A/B) |
| disc | 2 | 4096 | `heston_{S,v}_disc_4096x128.npy` | A18/A19 classifiers |
| ps | 3 | 512 | `heston_{S,v}_ps_512x128.npy` | path-shadowing queries |

Seeds 0/1/2 match the main benchmark's train/test/disc convention. The ps split uses a **new**
seed (3) so it never overlaps train/test/disc or the 1M generated bank.

---

## 3. The SBTS preprocessing — EXACT and reproducible

This is the heart of the experiment. It is copied verbatim from
`methods/SBTS/code/sbts_generate.py`. Constants: `DT = 1/250 = 0.004`, `S0 = 100`.

### 3.1 Forward: price panel `S (M,128)` → model input

```python
import numpy as np
DT, S0 = 1.0/250.0, 100.0

R        = np.log(S[:, 1:] / S[:, :-1])       # (M,127) log-returns
sigma    = float(R.std())                     # SBTS pooled sigma (see §3.4) — TRAIN split only
R_tilde  = R * np.sqrt(DT) / sigma            # (M,127) vol-scaled; std == sqrt(DT) == 0.063246
X_sbts   = np.hstack([np.zeros((M,1)), R_tilde])   # (M,128) prepend a DUMMY-0 column
```

**The dummy-0 column** makes the model input the same length (128) as the price series. It is a
constant for every path, carries no information, and is **discarded** on the way back. Its only
job is length-matching.

### 3.2 Inverse: model output `X_gen (M,128)` → price panel

```python
R_tilde_gen = X_gen[:, 1:]                     # drop the dummy column
R_gen       = R_tilde_gen * sigma / np.sqrt(DT)   # SAME frozen sigma as forward
S_gen       = np.empty((M, 128))
S_gen[:, 0]  = S0                               # anchor every path at 100
S_gen[:, 1:] = S0 * np.exp(np.cumsum(R_gen, axis=1))
```

Every generated path starts at exactly 100 (`S_gen[:,0].std() == 0`), matching real data.

### 3.3 The unit-variance wrapper (apply if your model has a fixed output-noise scale)

If the gate (§7) shows collapse/under-dispersion (the LS4 case, §0.1), wrap the SBTS transform in
the model's own input standardization:

```python
# forward, after building X_sbts:
x_mu = float(X_sbts.mean()); x_sd = float(X_sbts.std())   # x_sd == sqrt(dt) up to the dummy col
X_train = (X_sbts - x_mu) / x_sd                          # std == 1  -> feed this to the model
# ... train, generate X_model_gen ...
# inverse, before §3.2:
X_gen = X_model_gen * x_sd + x_mu                          # destandardize back to R~ space
```

Persist `x_mu, x_sd` in the weights checkpoint, `seed_N_config.json`, and `metadata.json`.
The round-trip must be exact — verify before training:

```python
# feeding real R~ straight through forward+inverse must return the real prices to ~1e-13
assert np.abs(S_gen_roundtrip - S).max() < 1e-9
```

This is mathematically the SBTS transform without the cosmetic `sqrt(dt)` down-scale; `sigma` is
still the SBTS pooled sigma and is still reported.

---

### 3.4 How `sigma` is estimated from SBTS (the exact estimator)

```python
sigma = float(R.std())     # R = log(S[:,1:]/S[:,:-1]), shape (M,127)
```

- `.std()` = NumPy **population** std (`ddof=0`).
- **Pooled**: one scalar over all `M × 127` returns (not per-path, not per-timestep).
- Estimated **once on the train split (seed 0)** and **frozen** — reused unchanged for the
  inverse of every seed. This mirrors SBTS exactly.

**Value on train (seed 0, 4096×128):** `sigma = 0.01263163`, over `4096×127 = 520 192` returns.
Check: `std(R_tilde) == sqrt(DT) == 0.063246` by construction. Reproduce via
`generate_datasets.py` (it prints all four numbers). Also documented in the experiment
[`README.md`](README.md) and the dataset
[`README.md`](../../../dataset/Heston/preprocessing_with_log_returns/README.md).

---

## 4. Reference code — import, never edit

- **Model:** import the method's released model from `methods/<METHOD>/code/reference` (or its
  pip package). For LS4: `configs/monash/vae_solarweekly_released.yaml`, run with
  `~/gpu-venv/bin/python` (torch 2.13+cu130). Match the **original method's hyperparameters
  exactly** (same preset, epochs, batch size). The *only* intended differences vs the main
  benchmark are (a) the §3 preprocessing and (b) 4096 paths.
- **Metrics:** `compute_metrics_logret.py` does `import compute_all as C`, sets
  `sys.argv=[sys.argv[0]]` **before** the import (so `compute_all`'s import-time argparse sees no
  flags), then overrides `C.DATASET_DIR`, `C.GENERATED_DIR`, `C.RESULTS_DIR`, `C.PLOTS_DIR`,
  `C.N_SEEDS`, and the three loaders (`load_data`/`load_disc`/`load_generated`) to the 4096 files.
  Every A1–A34 + B-curve + grid_tvd computation is the untouched `compute_all` implementation.

---

## 5. Training script contract (`train_<method>_logret.py`)

Args: `--seed`, `--data` (default train npy), `--epochs`, `--batch_size`, `--gen_num 4096`,
`--gen_batch`, `--frac 1.0`, `--tag` (non-canonical smoke runs skip weight/config saves),
`--out`. Must:

1. Load train `S (4096,128)`; apply §3 forward (+ §3.3 wrapper if needed).
2. Train with the method's exact released hyperparameters; log per-epoch losses to
   `losses/seed_N_losses.csv` (`epoch,total_loss,...,lr`).
3. Generate `gen_num` paths; apply §3 inverse; assert finite + anchored at 100.
4. Save: `weights/seed_N_model.pt` (state_dict + `sbts_sigma`, `x_mu`, `x_sd`, `dt`, `s0`),
   `weights/seed_N_config.json`, `generated_paths/seed_N/generated_paths_4096x128.npy` +
   `metadata.json` (GUIDELINE 4.3 schema + `sbts_sigma`, `x_mu`, `x_sd`, `preproc`).

---

## 6. Hardware & parallelization

### 6.1 Hard limits (shared machine, enforced 24/7)
Max **2 GPUs**, **16 physical cores**, ~250 GiB RAM. **Always** `nvidia-smi` + `htop` first; if
someone else is on a GPU or RAM > 50%, **ask before launching**. LS4's model is tiny
(~3.3 GB / 80 GB, ~45% util) so it cannot saturate one A100 alone — pack seeds, don't idle.

### 6.2 The 5-seed pattern (after the gate passes)

Pin CPU cores + one GPU per process. **Env vars go BEFORE `taskset`** (M5), pass **`--epochs`
explicitly** (M1), redirect each seed to its own **log file** and **detach with `setsid … <
/dev/null &` + `disown`** so the jobs survive the launching shell (M6).

**Case A — two GPUs free (ideal):** two seeds at a time, one per GPU.

```bash
cd results/Heston/preprocessing_with_log_returns/<METHOD>/code
PY=/home/tbasseras/gpu-venv/bin/python        # torch+cu130, NOT ~/.cc-venv
mkdir -p logs
for i in 0 1; do
  CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    setsid taskset -c $((i*8))-$((i*8+7)) \
    $PY train_<method>_logret.py --seed $i --epochs 100 \
    > logs/seed_$i.log 2>&1 < /dev/null &
done
disown -a
# then repeat the block for seeds 2,3, then seed 4 alone.
```

**Case B — only GPU 0 free (what actually happened for LS4):** pack the seeds on GPU 0, ≤4 cores
each (≤16-core cap). Compute-bound seeds time-slice, so wall-time ≈ sequential, but the card stays
~100% utilised and no one else's allocation is touched.

```bash
PY=/home/tbasseras/gpu-venv/bin/python ; mkdir -p logs ; i=0
for seed in 2 3 4; do
  lo=$((i*4)); hi=$((i*4+3))
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    setsid taskset -c ${lo}-${hi} \
    $PY train_<method>_logret.py --seed $seed --epochs 100 \
    > logs/seed_${seed}.log 2>&1 < /dev/null &
  i=$((i+1))
done
disown -a
```

**Always verify launch** (both cases): `sleep 12 && pgrep -af train_<method> && nvidia-smi`.
Expect one PID per seed and GPU memory within seconds. If a GPU stays at 0 % / 14 MiB, that seed
died — read `logs/seed_$s.log` (usually an M5 exit-127 or a `CUDA_VISIBLE_DEVICES` typo).

### 6.3 Follow the training live (logs + ETA) — never fly blind (M2)

The train scripts print `[ep N] total=… mse=…` every `--log_every` (default 25) epochs with
`flush=True`, plus a final `[done] … train=…s gen=…s` line. To follow and estimate ETA:

```bash
# live stream of all seeds at once
tail -n +1 -f logs/seed_*.log

# last epoch reached per seed, on demand
for s in 0 1 2 3 4; do printf "seed %s: " $s; grep -E '^\[ep' logs/seed_$s.log | tail -1; done

# s/epoch → ETA: watch two [ep] lines, divide Δwall by Δepoch, ×(epochs_left)
```

- **Never pipe training through `| tail`/`| head`** — that buffers the whole run and you see nothing
  until it exits (M2). Redirect to a file (`> logs/seed_$s.log 2>&1`) and `tail -f` the file instead.
- A watcher that only greps the success marker is blind to crashes — grep for progress **and**
  failure signatures together, e.g. `grep -E '\[ep |\[done\]|Traceback|Error|Killed|OOM'`.
- Rough LS4 timing (A100, 100 epochs, 4096 paths): ~5–6 s/epoch solo, ~8 s/epoch when 3 seeds share
  GPU 0 → ~10–13 min/seed; a 1M PS bank adds ~5–11 min/seed (see §9).

---

## 7. The GATE (mandatory, seed 0 only)

**Do not train 5 seeds or build the 1M bank until this passes.**

1. Train **seed 0** only.
2. Run `compute_metrics_logret.py --seeds 1` → `seed_0_metrics.json`.
3. Run `gate_compare.py`:
   - builds `plots/heston_diagnostics.png` (new gen vs 4096 real test), importing
     `metrics/plot_diagnostics.plot_diagnostics` unchanged;
   - tabulates every A-metric and the B-curve funct/%err new-vs-original seed 0 with signed
     `delta%` → `gate_seed0_compare.md`.
4. **Decision:**
   - Metrics comparable (small deviations) → note them, proceed to §6.2 + §9.
   - **Big deviation** → flag it explicitly to the user.
   - **Bad** (10×–1000× worse dispersion metrics: `A26_std_error`, `A6/A7 mmd2`, `A30_vol_path_rmse`,
     `grid_tvd`, `B_log_ret_hist`) → **STOP**. Show original vs new diagnostics figures side by
     side, do the **hyperparameter check** (§0.1: is the model input std below the model's
     output-noise floor? is the preset matched? epochs enough?), propose a fix, and **ask**.

The LS4 first attempt is the reference failure: collapse → root-caused to the sigma/noise-floor
mismatch → fixed with §3.3 → re-gated.

---

## 8. Metrics & figures

- **A1–A34 + B-curve + grid_tvd:** `compute_metrics_logret.py --seeds 5` writes
  `seed_N_metrics.json` per seed + `metrics_summary.csv`. Mostly lower-better; `A28_kurtosis_ratio`
  target 1.0; `A33_sigma_corr` higher-better.
- **Diagnostics figure:** 8-panel `heston_diagnostics.png` (real-vs-gen paths, log-return hist, QQ,
  ACF |r| and r², rolling-vol hist, terminal-price tail survival). Theory reference curves are
  dataset-level (sample-count independent).
- **PCA/t-SNE:** `pca_tsne_seed0.png` (from `compute_all.save_plots`).

---

## 9. Path Shadowing — STRICT paper protocol (arXiv:2308.01486)

> ⚠️ **READ M7 FIRST.** In this experiment path-shadowing is the **exact protocol from the paper**
> — *not* the simplified `methods/<METHOD>/path_shadowing/` reference eval (65D murex embedding,
> K=77, raw prefix-price L2, single 8192 bank, CRPS/MAE/RMSE only). Those answer an **easier,
> different** question and their numbers are **not comparable** to the paper. The canonical driver
> is **`<METHOD>/path_shadowing/path_shadowing_pdf.py`** and it is **self-contained** — it does not
> import the reference PS code. Path-shadowing Monte-Carlo (Morel, Mallat, Bouchaud): retrieve
> nearest paths from a **large generated bank**, form the K-member predictive ensemble, and score
> the forecast distribution with the full metric set below.

### 9.1 Fixed protocol constants (from the PDF)
| Symbol | Value | Meaning |
|--------|-------|---------|
| bank sizes | **{4096, 16384, 65536, 262144, 1 000 000}** | **nested prefixes of the same 1M bank** — the bank-size sweep (§5.2) |
| K | **256** | nearest neighbours retrieved per query = the predictive ensemble (equal weight) |
| split `s` | **64** | prefix = points 0..64 (**65 points / 64 increments**), known history |
| horizon `H` | **32** | forecast points 65..96, anchored at point 64 |
| queries | **512** | held-out real paths from the **ps** split (seed 3), independent of bank |
| embedding | **4-block weighted, bank-standardized** | see §9.2 — recent · cumulative · rolling-vol · dependence |
| alignment | **additive-in-log = multiplicative-in-price** | automatic for return quantities (they cancel the anchor) |
| forecast quantities | **cumulative return · one-step return · horizon RV** | scored independently |
| metrics | **predictive-mean RMSE · CRPS (energy) · coverage 50/90 · band width 50/90 · lower/upper-90 miss** | per quantity |
| uncertainty | **2000 path-level bootstrap** | 95% percentile CIs on RMSE + CRPS |

### 9.2 The 4-block weighted embedding (the heart of the protocol — §3 of the PDF)
Each prefix (65 log-prices → 64 log-returns `r`) maps to a **D≈73** feature vector, four blocks:

| Block | Features | Weight `w` |
|-------|----------|:----------:|
| recent returns | last **32** log-returns | **1.0** |
| cumulative path | cum log-return vs start, downsampled to **24** points | **0.5** |
| rolling vol | rolling-RMS over windows **5/10/20** → **last / mean / std** (9 feats) | **2.0** |
| dependence | **ACF** of `|r|` and `r²` at lags **1,2,5,10** (8 feats) | **1.0** |

**Bank-standardize, then weight:** `z̃ = √w · (z − μ_bank) / σ_bank`, where `μ_bank, σ_bank` are
computed **on the current bank-size prefix** (re-estimated for every point of the sweep, not on the
query). Retrieval = Euclidean KNN (K=256) in this `z̃` space (`sklearn` brute, `n_jobs≤16`).

### 9.3 Pipeline
1. **Build the 1M bank per seed** with `gen_banks.py` (reuses `path_shadowing_mc.build_bank`): prior-
   sample → §3 inverse → `bank/generated_bank_seed{i}_1000000x128.npy` (float32, ~0.5 GB). **Log it**
   (`> logs/gen_seed{i}.log`) so paths/s + ETA stream live (M2).
2. Load the 512 ps queries; build their 4-block features once.
3. For each **bank size** (nested prefix `B[:bs]`): build bank features, standardize both bank and
   query by **that** bank's `μ,σ`, retrieve K=256.
4. From the K neighbours' futures compute the **three quantities** (cum, one-step, RV); the query's
   own future gives the realized target.
5. Score every quantity: predictive-mean RMSE, CRPS, coverage 50/90, band width 50/90, lower/upper-90
   miss. Add **diagnostics** (terminal RMSE, prefix-distance mean/median/p95, unique-candidate
   fraction, RV mean bias) and **2000-resample bootstrap 95% CIs** on RMSE + CRPS.
6. Also score a **random-walk baseline** (resample each query's own prefix returns) for context.
7. Write `pdf_results_seed{i}.json` per seed, aggregate `pdf_summary.json` (mean ± std across seeds
   per bank size), and `plots/pdf_crps_vs_banksize.png` + `plots/pdf_coverage_calibration.png`.

### 9.4 Why the ps split is separate
The 512 queries (seed 3) must be **strictly independent** of the train split **and** of the 1M bank,
or every metric is optimistically biased (the bank could contain near-copies of a query). Seed 3
guarantees no overlap.

### 9.5 Build the 5 banks, then run the sweep — **always log** (so we can follow generation live)
```bash
cd results/Heston/preprocessing_with_log_returns/<METHOD>/path_shadowing
PY=/home/tbasseras/gpu-venv/bin/python
mkdir -p logs bank

# (a) generate the five 1M banks — pack 2 seeds/GPU, LOG each (M2/M3/M5/M6)
for s in 1 2; do base=$(((s-1)*8)); CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
  taskset -c ${base}-$((base+7)) $PY gen_banks.py --seed $s > logs/gen_seed${s}.log 2>&1 < /dev/null & done; wait
for s in 3 4; do base=$(((s-3)*8)); CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
  taskset -c ${base}-$((base+7)) $PY gen_banks.py --seed $s > logs/gen_seed${s}.log 2>&1 < /dev/null & done; wait
# seed 0 bank already built by the smoke/eval; follow live: tail -f logs/gen_seed*.log

# (b) run the STRICT protocol across all seeds — LOG it too (pdf_run.log)
OMP_NUM_THREADS=16 taskset -c 0-15 $PY path_shadowing_pdf.py --seeds 0,1,2,3,4 \
  > logs/pdf_run.log 2>&1 < /dev/null &
tail -f logs/pdf_run.log     # per (seed,bank_size) line: CRPS, coverage, uniq-frac, elapsed
```
Each seed writes `pdf_results_seed{i}.json`; the run writes `pdf_summary.json`. A 1M bank is ~0.5 GB
float32 each (~2.5 GB for five); delete banks after scoring if disk is tight — the **JSON metrics are
the deliverable**, not the banks. **Everything (generation *and* evaluation) must go through a log
file** so progress, ETA, and stalls are visible — never fly blind (M2).

---

## 10. READMEs

- **Experiment README** ([`README.md`](README.md)): overview, the §3 preprocessing (forward +
  inverse + dummy col), the §3.4 sigma estimation with the value table, the datasets table, and
  the methods table. Already written.
- **Per-method README** (`<METHOD>/README.md`): mirror `results/Heston/<method>/README.md` — the
  A1–A34 table across 5 seeds (+ a "Perfect" floor row), `heston_diagnostics.png`, the B-curve
  table (MSE/%err/NRMSE/CVaR90/CVaR95), disc/pred loss plots, and the **STRICT path-shadowing
  tables** (§9): per-quantity (cum/step/RV) metrics across the **bank-size sweep** with CRPS +
  coverage/width + bootstrap CIs vs the RW baseline, plus the two `pdf_*.png` plots — **not** the
  old CRPS-only H=32/64 table (M7). Note the preprocessing variant + any §3.3 wrapper used.
- Follow GitHub math rules: no `\;`/`\,` in `$$`, no trailing comma on intermediate lines, no bare
  `*` in superscripts.

---

## 11. Checklist for a new method

- [ ] Confirmed reference model imports and runs with its released preset (no edits to reference).
- [ ] `train_<method>_logret.py` applies §3 forward + inverse; round-trip asserted to ~1e-13.
- [ ] Decided on the §3.3 unit-variance wrapper (default: include it; the gate confirms).
- [ ] `sigma = 0.01263163` reported; matches SBTS estimator.
- [ ] **`--epochs` passed explicitly and matches the original** (`wc -l methods/<METHOD>/losses/seed_0_losses.csv`; LS4 = 100). Never rely on the 400 default. **(M1)**
- [ ] **`seed_N_config.json` diffed against the original hyperparameters** — only the §3 scaler and 4096 count differ. **(M4)**
- [ ] Training launched **without a `tail` pipe** (background/log file) so `[ep N]` lines stream live for ETA. **(M2)**
- [ ] Seeds **packed on one GPU** on separate cores to reach ~full util (not one process at ~44%). **(M3)**
- [ ] **Env vars (`CUDA_VISIBLE_DEVICES`/`OMP_NUM_THREADS`) placed *before* `taskset`**, not after — else exit 127. **(M5)**
- [ ] Jobs **detached with `setsid … < /dev/null &` + `disown -a`** so they survive the launcher shell exit; verified with `pgrep`+`nvidia-smi`. **(M6)**
- [ ] Correct venv: **`/home/tbasseras/gpu-venv/bin/python`** (torch+cu130), not `~/.cc-venv`. **(§0.3)**
- [ ] Trained **seed 0**; ran metrics; ran `gate_compare.py`.
- [ ] **GATE passed** (or fix approved by user). Old-vs-new figures inspected.
- [ ] Trained seeds 1–4 (§6.2, 2 GPUs max); full `metrics_summary.csv`.
- [ ] Built the five 1M path-shadowing banks (§9.5, logged). **STRICT paper protocol** run via
      `path_shadowing_pdf.py` — 4-block embedding + bank-size sweep + cum/step/RV + coverage/width +
      bootstrap CIs; `pdf_summary.json` + plots produced. **Did NOT reuse the reference murex eval (M7).**
- [ ] **Generation *and* evaluation logged** (`logs/gen_seed*.log`, `logs/pdf_run.log`) — never flew blind (M2).
- [ ] Per-method README written (mirror the main benchmark's).
- [ ] Everything lives under the two `preprocessing_with_log_returns/` folders; no reference file
      touched.
