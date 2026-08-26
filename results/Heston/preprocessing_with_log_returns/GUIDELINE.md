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
| M1 | **Epoch count left to the default.** | The train script's `--epochs` **defaults to 400**. Relaunching seed 0 as `python train_..._logret.py --seed 0` (no `--epochs`) silently ran **400** epochs while the canonical baseline is **100**. That (a) confounds the gate — you change *two* variables (preprocessing **and** 4× training) instead of one — and (b) wasted ~20 min before it was caught. | **Fix the trap at its source, not just at the call site.** Two guards, both required: (1) **set the script's `--epochs` default to the canonical count** (LS4 = 100) so a bare launch cannot run 400 — a passed flag protects one launch, a fixed default protects every future one; (2) still **pass `--epochs 100` explicitly** as a second belt. Verify the canonical count: `wc -l methods/<METHOD>/losses/seed_0_losses.csv` (rows − 1 header; LS4 = **100**). The *only* intended differences vs the main benchmark are the §3 preprocessing and 4096 paths — **nothing else**, epochs included. **See M8: this exact trap re-fired when the baseline script was cloned.** |
| M2 | **Training piped through `\| tail -50`.** | `python train.py ... 2>&1 \| tail -50` buffers **all** output until the process exits, and the loss CSV / weights are only written at the end. Result: **zero live progress**, no way to estimate ETA or notice a stall for the whole run. | **Never pipe training through `tail`/`head`.** Launch with `run_in_background` (or redirect to a log file) so the per-epoch `[ep N] ...` lines (printed with `flush=True`) stream live. Sample two epoch lines to get s/epoch → ETA. |
| M3 | **Assuming the GPU was busy at 44%.** | A single LS4 process uses only ~44% of an A100 (2.1M params — too small to saturate it). Treating that as "GPU in use" left ~half the card idle. | **Pack independent seeds onto one GPU** (pin to separate CPU cores: seed 0 → `taskset -c 0-7`, seed 1 → `-c 8-15`). Two LS4 seeds together reach ~95% util at ~6.6 GB — full GPU, one card, no one else's allocation touched. |
| M4 | **Comparability drift in general.** | The root purpose is "change *only* the preprocessing." Any silent divergence (epochs, batch size, preset, sample count beyond the intended 4096) invalidates the gate comparison. | Diff your `seed_N_config.json` against the original method's hyperparameters **before** training seed 0. If a field differs and it isn't the §3 scaler or the 4096 count, fix it. |
| M5 | **Env vars placed *after* `taskset`.** | Launching with `taskset -c 0-3 OMP_NUM_THREADS=4 python train.py` fails instantly: `taskset` treats `OMP_NUM_THREADS=4` as the **command to exec** → `taskset: failed to execute OMP_NUM_THREADS=4: No such file or directory`, exit code **127**. The three seeds "launched" but every one died in <1 s; only caught because `nvidia-smi` showed GPU 0 back at **0 %**. | **Env assignments must come *before* `taskset`** (they are shell prefixes, `taskset` is the command): `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 taskset -c 0-3 $PY train.py …`. After launching, **always** verify with `nvidia-smi -i 0` and `pgrep -af train`: expect one PID per seed and GPU memory within seconds. |
| M6 | **Backgrounded launcher let the children die on exit.** | Starting the training loop from a wrapper shell that itself exits (e.g. a `run_in_background` one-liner that spawns `python … &` then returns) sends **SIGHUP** to the children when the parent shell reaps — the jobs vanish and GPU 0 drops to 0 % moments later. | **Detach each job from the launching shell.** Use `setsid` + redirect stdin from `/dev/null` + `disown`: `… taskset -c 0-3 $PY train.py --seed $s --epochs 100 > logs/seed_$s.log 2>&1 < /dev/null &` inside a `for` loop, then `disown -a`. Confirm survival after the launcher returns: `pgrep -af train_…` must still list the PIDs and `nvidia-smi` must show memory. |
| **M7** | **Path-shadowing done with the *simplified* reference eval instead of the paper protocol.** | `methods/<METHOD>/path_shadowing/` (`path_shadowing.py` + `run_eval.py`) is a **reduced** eval: a 65D **murex** embedding, **K=77**, raw prefix-price L2, a single **8192**-path bank, and **CRPS/MAE/RMSE only** at H=32/64. It is **NOT** the arXiv:2308.01486 protocol. Reusing it (as an earlier draft of §0.3/§9.2 wrongly said to) silently answers a *different, easier* question and makes the numbers non-comparable to the paper. | **This experiment uses the STRICT paper protocol — never the reference subset.** Use `<METHOD>/path_shadowing/path_shadowing_pdf.py`: the **4-block weighted, dimension-normalized, frozen-reference-standardized** embedding (recent-returns w1.0 · cumulative-path w0.5 · rolling-vol w2.0 · dependence-ACF w1.0; `z̃ = √(w/d)·(z−μ_ref)/σ_ref` with `μ_ref,σ_ref` frozen on the **real test set**), a **single shared 1M bank** whose **nested prefixes** give the **bank-size sweep** {4096, 16384, 65536, 262144, 1 000 000}, the **three forecast quantities** (cumulative return, one-step return, horizon RV), and the **full metric set** (predictive-mean RMSE, CRPS, coverage 50/90, band width 50/90, lower/upper-90 miss) with **2000-resample paired bootstrap 95% CIs over the queries**. See §9 (and §9.6 for every ambiguity choice). The old `path_shadowing_mc.py` (CRPS-only, K=256 murex) is **superseded** — kept only as the 1M-bank builder via `gen_banks.py`. |
| **M8** | **M1 re-fired when the train script was cloned for the no-preproc baseline.** | `baseline_no_preproc/code/train_ls4_raw.py` was copied from `train_ls4_logret.py`, which **still carried `--epochs` default 400**. Launched as `train_ls4_raw.py --seed 0` (no flag) → ran **400** epochs again, exactly the M1 trap, on the *control* run that most needs to match the with-preproc runs' **100**. Caught by Theo ("why 400 epochs… u remade the mistake"). Killed, script default fixed to 100, cleaned, relaunched at 100. | **When cloning a train script, immediately edit its `--epochs` default to the canonical count** (LS4 → `default=100`) before the first launch — a cloned default carries the old bug forward. A guard that lives only in the *call site* (passing `--epochs`) does not survive a copy-paste; a guard baked into the *default* does. This is why M1's guard now demands both. Verify with `grep -n 'epochs' code/train_*.py` after cloning. |
| **M9** | **README metric table transcribed incompletely from the JSON.** | The §8 one-step-return PS table in `LS4/README.md` was missing **coverage 50** and **width 50** rows — the values existed in `pdf_summary.json` but were never copied into the table, so a reader auditing "all 8 metrics present" would find only 6 for that quantity. | **Transcribe metric tables programmatically, never by hand.** For every quantity (cum/step/rv), assert the rendered table has all 8 rows (rmse, crps, coverage50/90, width50/90, lower/upper_miss90) by cross-checking against `pdf_summary.json` keys before committing. A quick check: count metric rows per quantity block == 8. |
| **M10** | **Treated GitHub's 100 MiB push limit as a removable local setting.** | Repeated attempts to push the 489 MiB `bank/generated_bank_seed0_1000000x128.npy` by "removing the gitignore rule" / `--force`. The 100 MiB cap is a **GitHub server-side hard rejection**, not a gitignore/config/size setting; `--force` rewrites history, not file size. Only **Git LFS** pushes files >100 MiB. | **The 1M bank is NOT pushed — it is gitignored, disk-only, and regenerated on demand.** It lives at `<METHOD>/path_shadowing/bank/generated_bank_seed0_1000000x128.npy` and is deterministically rebuilt by `gen_banks.py --seed 0`, which is precisely why shipping it is unnecessary. **When adding a new method: do NOT add a `…/bank/*.npy` line to root `.gitattributes`.** Never push the bank as a plain blob, never `--force`, never strip a gitignore rule (that was the original mistake). *(History: LS4/CSDI/SBTS banks were committed via Git LFS under an earlier decision, commit `cfb22f5`, and their `.gitattributes` lines remain. **Theo reversed that policy on 2026-07-30** — "do not push the 1M samples, save them under `banks`" — because free GitHub LFS gives 1 GiB storage + 1 GiB/month bandwidth, so ~2 fresh clones/month exhausts the tier and LFS fetches start 403-ing. Leave the existing three entries alone; add no new ones.)* |
| **M11** | **`BENCH_ROOT` copied verbatim into a folder one level deeper.** | `TimeDiT/baseline_no_preproc/path_shadowing/path_shadowing_pdf.py` was cloned from `LS4/path_shadowing/path_shadowing_pdf.py`, keeping its **four** `os.path.dirname()` hops. But the no-preproc control sits one directory deeper than a normal method driver, so four hops land on `results/`, not the repo root — every data path became `results/dataset/Heston/…`. The 1M bank finished at 02:20 UTC and the chained eval died **one second later** with `FileNotFoundError`, then the chain script exited. Nobody was watching: **~4.5 h of idle GPUs**. | **When cloning a script into a different directory depth, re-derive the root walk — never copy it.** Count the hops against the new location (`baseline_no_preproc` → `TimeDiT` → `preprocessing_with_log_returns` → `Heston` → `results` → `benchmark` = **five**). Then **assert at import**, before any compute: `for p in (PS_QUERY, REF_STATS): assert os.path.exists(p), p`. A 16 h GPU job must never be gated by a path that is only resolved after it finishes. Equivalently: **have the chain script verify the eval's inputs exist *before* launching the generation**, not after. |
| **M12** | **Assumed every method's checkpoints are small enough to push.** | LS4 weights are 16.8 MiB, so the template ships `weights/*.pt` in Git. TimeDiT is **DiT-S, 32,512,897 fp32 params = 124.1 MiB per checkpoint** — over GitHub's **100 MiB per-file hard cap** — and 7 of them total **869 MiB**, which would also exhaust the entire free LFS tier (1 GiB) in a single push and start 403-ing the existing LS4/CSDI/SBTS bank objects. Caught only by sizing the staged payload (909 MiB) *before* committing. The checkpoints carry **no optimizer state** — pure fp32 weights, so there is nothing to strip. | **Size the payload before you stage it**, not when the push is rejected: `git add -An <folder>` → resolve each path → `du -ch`, and flag any single file > 100 MiB. **Theo's rule (2026-07-31): big checkpoints stay ON DISK, in place, gitignored — "they just do not appear in the github".** Add `<METHOD>/weights/*.pt` (and the `baseline_no_preproc` twin) to `.gitignore`, keep the small `seed_N_config.json` tracked, and say so in README §8. Same logic as M10: deterministic + regenerable ⇒ shipping is unnecessary. **Never** reach for LFS to solve it. |

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
| The 1M-bank builder (reuse verbatim) | `LS4/path_shadowing/path_shadowing_mc.py` → `build_bank`, `load_gen_model`; driven by `gen_banks.py --seed 0` | Prior-sample the trained seed-0 model + §3 inverse → persist the **one** `bank/generated_bank_seed0_1000000x128.npy`. **This is the *only* thing to reuse from the old PS code.** |
| **The path-shadowing evaluator** | `<METHOD>/path_shadowing/path_shadowing_pdf.py` (**strict arXiv:2308.01486 protocol** — see §9) | **Do NOT import `methods/<METHOD>/path_shadowing/path_shadowing.py`** — that is the *simplified* reference eval (65D murex, K=77, prefix-price L2, CRPS/MAE/RMSE only). It answers a different, easier question. **(M7)** The paper protocol is self-contained in `path_shadowing_pdf.py` (4-block weighted, dim-normalized, frozen-reference-standardized embedding + single-bank nested-prefix sweep + cum/step/RV quantities + coverage/width + bootstrap CIs). |
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
    │   ├── bank/  generated_bank_seed0_1000000x128.npy   # ONE shared 1M bank — GITIGNORED, disk-only, regenerable
    │   ├── logs/  gen_seed0.log, pdf_run.log       # generation + eval logged (M2/M7)
    │   ├── plots/ pdf_crps_vs_banksize.png, pdf_coverage_calibration.png
    │   └── pdf_summary.json                        # all metrics + 95% bootstrap CIs (single run)
    ├── baseline_no_preproc/      # §7.1 head-to-head control — ONLY the scaler differs
    │   ├── code/train_<method>_raw.py   # clone of the logret script; global standardize; --epochs default 100 (M8)
    │   ├── logs/train_seed0.log
    │   ├── weights/  losses/  generated_paths/seed_0/
    │   └── seed_0_metrics.json          # A/B only, seed 0 — NO path-shadowing
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

### 7.1 The no-preprocessing baseline (the head-to-head control)

The gate (§7) checks the *preprocessed* variant against the *original* main-benchmark LS4. But the
experiment's headline question — **"does the log-return preprocessing actually help this method?"** —
needs a **matched control trained inside this folder**, changing **only the scaler**. That control is
the **no-preprocessing baseline**, and it is what the §9-below "Results" verdict compares against.

**What it is.** The identical pipeline as `train_<method>_logret.py` with **one** substitution: the
§3 log-return transform is replaced by a plain **global standardize** `(S−μ)/σ` on the raw price panel
(μ, σ scalar over all entries), and the inverse is the direct `X·σ+μ` back to price — **no**
log-return, **no** cumsum, **no** exp. Everything else is byte-for-byte identical: same reference
`VAE` + released preset, same 4096 train paths, **same 100 epochs**, same seed 0, same optimizer/EMA,
same prior-sample → 4096 generated paths, same metric runner.

**Why it is the right control (not the §7 gate).** The §7 gate answers "did preprocessing break the
model vs the main benchmark?" (a *comparability* check across folders, confounded by the main
benchmark's own training run). The baseline answers "**preprocessing vs no-preprocessing, all else
equal, in this folder**" — the only clean A/B for the preprocessing's *effect*. Isolating a single
variable is the whole point (M4); the baseline is that isolation made concrete.

**Location & script.** `<METHOD>/baseline_no_preproc/` mirrors the per-method layout:
```
<METHOD>/baseline_no_preproc/
├── code/train_<method>_raw.py        # clone of train_<method>_logret.py; ONLY the scaler differs
├── logs/train_seed0.log
├── weights/seed_0*  losses/seed_0*  generated_paths/seed_0/
└── seed_0_metrics.json               # A/B only — NOT path-shadowing
```
The script sets `variant="… + NO preprocessing (global standardize)"`, `scaler="global_standardize"`,
meta `preproc="none_global_standardize"`.

**How to run it (M1/M8 apply with full force).** The baseline is the run that *most* needs to match
the with-preproc epoch count, and its script is a **clone** — so its `--epochs` default **must be
edited to 100 at the source** (M8), then launched explicitly:
```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-7 \
  $PY code/train_<method>_raw.py --seed 0 --epochs 100 > logs/train_seed0.log 2>&1 < /dev/null &
```
Verify the log header reads `epochs=100` before walking away.

**Scope.** **A/B metrics only, seed 0** — the baseline is a control for the *preprocessing effect*, not
a full 5-seed / path-shadowing deliverable. Do **not** build a 1M bank for it. Its numbers feed exactly
one place: the "Results — does the variant help?" table (§10, below the per-method template block),
where the **with-preproc seed 0** and **no-preproc seed 0** A/B metrics sit side by side with a signed
Δ and a one-line verdict declaring the winner.

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
| bank design | **one shared 1M bank** (seed-0 generator) | the sweep is nested prefixes of this **single** bank; uncertainty comes from the query bootstrap, **not** from a model-seed spread |
| embedding | **4-block weighted, dim-normalized, frozen-reference-standardized** | see §9.2 — recent · cumulative · rolling-vol · dependence |
| alignment | **additive-in-log = multiplicative-in-price** | automatic for return quantities (they cancel the anchor) |
| forecast quantities | **cumulative return · one-step return · horizon RV** | scored independently |
| metrics | **predictive-mean RMSE · CRPS (energy) · coverage 50/90 · band width 50/90 · lower/upper-90 miss** | per quantity |
| uncertainty | **2000-resample paired bootstrap over the 512 queries** | 95% percentile CI on **every** metric; one fixed resample-index matrix (`boot_seed=20230814`) shared across the whole sweep so differences are comparable |

### 9.2 The 4-block weighted embedding (the heart of the protocol — §3 of the PDF)
Each prefix (65 log-prices → 64 log-returns `r`) maps to a **D≈73** feature vector, four blocks:

| Block | Features | Weight `w` |
|-------|----------|:----------:|
| recent returns | last **32** log-returns | **1.0** |
| cumulative path | cum log-return vs start, downsampled to **24** points | **0.5** |
| rolling vol | rolling-RMS over windows **5/10/20** → **last / mean / std** (9 feats) | **2.0** |
| dependence | **ACF** of `|r|` and `r²` at lags **1,2,5,10** (8 feats) | **1.0** |

**Dimension-normalize, standardize against a frozen reference, then take the KNN.** Per-feature
weight is `w_block / d`: a block of dimension `d` contributes total mass `w_block`, so a wide block
(rolling-vol, 9 feats) does **not** dominate a narrow one purely by feature count. The standardization
`μ_ref, σ_ref` are computed **once on the real Heston test set** (`heston_S_test_4096x128` prefix
features — model-independent) and held **fixed across the entire bank-size sweep** (and, being real
data, comparable across methods). Embedding:

```
z̃ = √(w_block / d) · (z − μ_ref) / σ_ref
```

Retrieval = **exact** Euclidean KNN (K=256) in this `z̃` space (`sklearn` brute, `n_jobs≤16`,
lowest-index tie-break). Every one of these choices resolves a paper ambiguity — see the decision
log in **§9.6**.

### 9.3 Pipeline
1. **Build ONE 1M bank** with `gen_banks.py --seed 0` (reuses `path_shadowing_mc.build_bank`): prior-
   sample → §3 inverse → `bank/generated_bank_seed0_1000000x128.npy` (float32, ~0.5 GB). **Log it**
   (`> logs/gen_seed0.log`) so paths/s + ETA stream live (M2). This single bank backs the whole sweep.
2. Compute the **frozen** `μ_ref, σ_ref` once from the real test set (`heston_S_test_4096x128`); load
   the 512 ps queries and build their 4-block features once.
3. Build **one fixed paired-bootstrap index matrix** `(2000, 512)` from `boot_seed=20230814`.
4. For each **bank size** (nested prefix `B[:bs]` of the one bank): build bank features, standardize
   **both** bank and query by the **frozen** `μ_ref, σ_ref` (never re-estimated), retrieve K=256.
5. From the K neighbours' futures compute the **three quantities** (cum, one-step, RV); the query's
   own future gives the realized target.
6. Score every quantity: predictive-mean RMSE, CRPS, coverage 50/90, band width 50/90, lower/upper-90
   miss — **each with a 95% CI** from the shared bootstrap index. Add **diagnostics** (terminal RMSE,
   prefix-distance mean/median/p95, unique-candidate fraction, RV mean bias).
7. Also score a **random-walk baseline** (resample each query's own prefix returns) for context.
8. Write a single `pdf_summary.json` (`by_bank_size` → per-quantity metrics with CIs, plus
   `rw_baseline` and `protocol`) and `plots/pdf_crps_vs_banksize.png` + `plots/pdf_coverage_calibration.png`.

### 9.4 Why the ps split is separate
The 512 queries (seed 3) must be **strictly independent** of the train split **and** of the 1M bank,
or every metric is optimistically biased (the bank could contain near-copies of a query). Seed 3
guarantees no overlap.

### 9.5 Build the one bank, then run the sweep — **always log** (so we can follow generation live)
```bash
cd results/Heston/preprocessing_with_log_returns/<METHOD>/path_shadowing
PY=/home/tbasseras/gpu-venv/bin/python
mkdir -p logs bank

# (a) generate the ONE 1M bank (seed-0 generator) — LOG it (M2/M3/M5/M6)
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-7 \
  $PY gen_banks.py --seed 0 > logs/gen_seed0.log 2>&1 < /dev/null &
tail -f logs/gen_seed0.log   # paths/s + ETA stream live

# (b) run the STRICT protocol (no --seeds; single shared bank) — LOG it too
OMP_NUM_THREADS=16 taskset -c 0-15 $PY path_shadowing_pdf.py \
  > logs/pdf_run.log 2>&1 < /dev/null &
tail -f logs/pdf_run.log     # per bank_size line: CRPS, coverage, uniq-frac, elapsed
```
The run writes a single `pdf_summary.json` (all metrics + 95% bootstrap CIs) + the two plots. The 1M
bank is ~0.5 GB float32; it is **gitignored and never pushed** — it stays on disk and is
**deterministically regenerable** from `gen_banks.py --seed 0`, which is what makes shipping it
unnecessary. The **JSON metrics + plots are the deliverable** (M10).
**Everything (generation *and* evaluation) must go through a log file** so progress, ETA, and stalls are
visible — never fly blind (M2).

### 9.6 Decision log — every paper ambiguity, and what we chose (READ before touching the driver)

arXiv:2308.01486 under-specifies ~20 knobs. Each was resolved by the **most logical / least
model-dependent** choice and **frozen in `path_shadowing_pdf.py`** so all methods are comparable.
Do **not** silently change any of these — a different choice = a different, non-comparable benchmark.

**A — Standardization of the embedding (the critical group — it defines the retrieval geometry)**

| # | Ambiguity | Choice | Why |
|---|-----------|--------|-----|
| A1/A15 | Where do `μ,σ` come from? | **Frozen on the real Heston test set** (`heston_S_test_4096x128` prefix features), computed once. | Must be **model-independent**: if `μ,σ` came from each method's own bank, every method would warp its own metric space and scores would be incomparable. Real test data is the neutral shared reference. |
| A2/A16 | Re-estimated per bank size? | **No — one fixed `μ_ref,σ_ref` across the entire sweep.** | If re-estimated per prefix, distances at 4k vs 1M live in different spaces and the sweep curve is meaningless. Freezing makes the sweep a clean "more candidates, same ruler" experiment. |
| A3/A17 | Per-feature or one global scalar? | **Per-feature** (each of the 73 dims gets its own `μ,σ`). | Blocks live on wildly different scales (log-returns ~1e-2, ACF ~O(1)); a single scalar lets the largest-scale block dominate the L2 distance. |

**B — Feature construction**

| # | Ambiguity | Choice | Why |
|---|-----------|--------|-----|
| B4 | Cumulative-path downsample | `np.linspace(0, 64, 24).round().astype(int)` — 24 evenly-spaced, endpoint-inclusive indices of the cum-log-return. | Deterministic, covers the whole prefix incl. both ends, fixed length 24. |
| B5 | "Recent returns" order | **Chronological last 32** log-returns (`r[:, -32:]`), not reversed. | Preserves temporal order so the L2 compares like-aligned lags. |
| B6 | Rolling vol | **Causal trailing RMS**, windows 5/10/20, **valid windows only** (no padding) → last / mean / std (9 feats). | Trailing = uses only known prefix; valid-only avoids edge artifacts; last/mean/std summarize level+trend+dispersion of local vol. |
| B7 | ACF estimator | **Biased** (denominator = Σr², i.e. N not N−lag), demeaned on the **prefix only**, on series `|r|` and `r²`, lags 1,2,5,10. | Biased ACF is the standard stylized-fact estimator, bounded in [−1,1], stable at short lengths. Prefix-only demeaning avoids leaking the future. |
| B8 | Weight × dimensionality confound | **Per-block dimension normalization:** per-feature weight `w_block/d`, multiplier `√(w_block/d)`. | Without it a block's influence ∝ `w_block × d`, so recent-returns (32 feats, w1.0) would swamp rolling-vol (9 feats, w2.0) despite vol's 2× intended weight. Normalizing makes each block contribute exactly its `w_block` mass. |

**C — Distance & neighbour selection**

| # | Ambiguity | Choice | Why |
|---|-----------|--------|-----|
| C9/C18 | Distance metric | **Exact Euclidean L2** in `z̃` space (`sklearn` brute, not ANN). | 1M×73 is tractable exactly (~21 s), so there is zero approximation error to argue about. |
| C10/C19 | Tie-break on equal distance | **Lowest index** (`sklearn` default). | Ties are measure-zero on continuous features; the rule is only for determinism. |
| C11/C20 | Ensemble weighting | **K=256, equal weight**, futures taken in **return space**. | The paper's predictive ensemble is uniform over the K nearest; no distance kernel. Return space makes the log-additive shift cancel (endpoint alignment is automatic). |

**D — Shared fixtures & scoring**

| # | Ambiguity | Choice | Why |
|---|-----------|--------|-----|
| D12 | Query set | `heston_S_ps_512x128` (**ps** split, seed 3), disjoint from train **and** bank. | Independence is mandatory — otherwise the bank can contain near-copies of a query and every metric is optimistically biased (§9.4). |
| D13 | Bootstrap | **Paired percentile bootstrap**, 2000 resamples over the 512 queries, **one fixed index matrix** (`boot_seed=20230814`) reused across every bank size/quantity/metric. CI = [2.5, 97.5] percentiles. | Pairing (shared resample indices) makes sweep/quantity **differences** share their resampling noise → directly comparable. Percentile (not BCa/normal) is the simplest defensible interval. |
| D14 | Split/horizon geometry | `s=64` (65 pts / 64 incr), `H=32` (points 65..96 anchored at 64), `SEQ_LEN=128`. | Fixed by the benchmark panel; anchor at the split point. |

**E — Forecast-quantity dimensionality (fixed 2026-07-28; was a real bug before)**

| # | Ambiguity | Choice | Why |
|---|-----------|--------|-----|
| E15 | Are cum / one-step returns scalars or trajectories? | **H-dimensional trajectories over offsets u=1…H.** `cum_u = logS[s+u]−logS[s]`, `step_u = logS[s+u]−logS[s+u−1]`; RMSE/CRPS/coverage/width are computed at every u and **averaged over u=1…H** before the query bootstrap. RV stays the scalar `√Σr²`. | §2/§3.1/§3.3 define cum & step as h-indexed running trajectories aggregated over all future times — **not** a single terminal point. The earlier build scored cum only at h=H and step only at h=1, which (a) answered an easier single-horizon question and (b) made the "terminal RMSE" diagnostic **identical** to cum RMSE by construction (the M-tagged identity bug). Trajectory scoring fixes both; terminal (h=H) MAE is now a genuinely distinct diagnostic — and note it *is* an MAE (`mean_q|e_q|`) despite the `terminal_rmse` JSON key, see §10.4 item 8. |
| E16 | How does the query bootstrap interact with the horizon average? | Per-query metric = **mean over u** of the per-(query,u) value; the 2000-resample bootstrap then resamples **queries** (not (query,u) pairs). | Keeps the bootstrap unit = the 512 independent query paths (D13); horizons within a query are not independent so they are averaged, not resampled. |
| E16b | How does RMSE aggregate across queries? | **Two aggregations of the same per-query squared error are emitted, and every row reports the one its label claims.** With `se_q = (1/H)Σ_u se_{q,u}` (E16), `path_shadowing_pdf.py::METRIC_MAP` writes **both** keys for every quantity: `"rmse"` = `mean_q(√se_q)` (root *inside*, `_agg` kind `rmse`) and `"rmse_textbook"` = `√(mean_q se_q)` (root *last*, `_agg` kind `rmse_last`); `_boot_ci` branches on the same kind so each key gets its own CI. Routing to the tables is **one** table — `build_cross_tables.PS_METRIC_KEY`, always read through `build_cross_tables._ps_key(qn, metric)`: **cum/step → `rmse_textbook`**, labelled `RMSE ↓`; **rv → `rmse`**, labelled `MAE ↓` (`PS_METRIC_LABEL`). `_ps_value`, the Winner ranking, the oracle/RW columns and `report/build_report.py::ps_table` all go through `_ps_key`, so the README table, the winner column and the LaTeX report cannot drift apart. **Changed in two steps.** *2026-07-30:* the single convention was flipped from `√(mean_q se_q)` to `mean_q(√se_q)` to match the reproducibility report (Tables 1–5); by Jensen `mean(√·) ≤ √(mean ·)` so every cell dropped (CSDI cum 0.0493 → 0.0414) — old and new numbers must never be mixed in one table. *2026-07-31:* the second key was added and cum/step routed to it, because for those H=32 quantities `se_q` is **already** the mean over u, so `√(mean_q se_q)` = `√(mean over every (q,u) pair)` = the genuine textbook RMSE, whereas `mean_q(√se_q)` is an *average of per-path horizon-RMS values* — a real statistic, but not an RMSE. rv keeps the root-inside key because at H=1 it collapses **exactly to MAE**, which is what its label now says. | Both conventions are defensible; naming one thing while computing another is not. Emitting both keys instead of replacing one means the reproducibility report's published column (Tables 1–5) survives byte-for-byte in every artefact while the tables show the textbook form — the JSON is a strict superset, no consumer broke. The re-run was controlled leaf-by-leaf against `/tmp/ps_before/`: generators **901 leaves identical, 99 added (all `rmse_textbook`), 0 removed, 10 changed — all `elapsed_sec`**; forecasters 93/95 identical, 9 added, 1 changed (`ft_time_sec`). That proves the aggregation was the sole change. It also let all **15** previously hand-typed historical RMSE cells be deleted from `report/build_report.py` (`HISTORICAL_OLD_RMSE`, `FORECASTER_PRE_FIX`): `rmse_textbook` reproduces the 9 generator cells to every printed digit and the 6 forecaster cells bit-for-bit at float64 — so **10.2 now holds with zero exceptions**. **The switch is not rank-preserving: a measured 7 of 10 cum/step winner cells move** (`report/build_report.py::rmse_winner_moves`, computed never quoted). That is not an artefact — it is the finding: @1M cum the top three are CSDI 0.049253 / LS4 0.048944 / TimeDiT (raw) 0.048875, a spread of 3.8×10⁻⁴ against a CI width of ≈6.3×10⁻³ — **17× wider**, every interval containing every other point estimate. The cum/step RMSE ordering among those three is noise and must not be read as a ranking; CRPS, coverage and miss-rate are the discriminating rows. (SBTS, 13 % worse, *is* separated.) `forecaster/pdf_bridge.py` used to pin Chronos-2 / TimesFM to the old `√(mean_q se_q)` via a local `_metrics_frozen_rmse()`. **Removed 2026-07-31:** the bridge now calls `P.metrics_with_ci` directly, so forecasters, generators, the Heston oracle and the RW floor share one byte-identical code path. That exception was not benign — `_ps_winner_idx` ranks *all* `PS_MODELS` including the forecasters, so the Winner column was ranking forecaster cells inflated by a measured 18.0 % (cum) / 5.6 % (step) / 5.0 % (rv) against un-inflated generator ones, while the README footnote simultaneously told readers those cells were not comparable. (The rv inflation is 5 %, **not** the generators' 28 % — the ratio tracks per-path error dispersion, which differs between retrieval and direct forecasting, so it must never substitute for recomputing.) |

**F — Heston oracle (protocol ceiling, §6)**

| # | Ambiguity | Choice | Why |
|---|-----------|--------|-----|
| F17 | Is there an upper-bound reference besides the RW floor? | **Yes — a Heston oracle.** A fresh **1M-path bank drawn from the true Heston law** (identical SDE params as `dataset/Heston/generate_heston.py`; independent seed **777**), run through the *same* retrieval+forecast+frozen-standardization pipeline over the full sweep. Reported as a third column beside LS4 and RW in the README, and stored under `heston_oracle` in `pdf_summary.json`. | §6 calls the true-DGP retrieval the protocol **ceiling**. Without it, RW (floor) alone cannot separate a method's *retrieval limit* from its *generator law-mismatch*: LS4's gap **to the oracle** = law-mismatch; the oracle's own residual = irreducible finite-bank retrieval error. This is what exposed LS4's RV upper-tail under-coverage (0.799 vs the oracle's 0.924) as a law-mismatch, not a bank-size artefact. |
| F18 | §5.1 eligibility gates (path SW ≤1.10×, stylized-fact error caps)? | **N/A — not applied.** Fixed 100-epoch training, no checkpoint selection, no per-generator gating. Recorded here as the audit trail. | The gates in §5.1 filter *candidate checkpoints*; this experiment has exactly one checkpoint per seed, so there is nothing to gate. Documented so a reader does not assume a silent filter. |

**Ranked reproducibility list (the knobs most able to move the numbers, high → low):**

1. **KNN feature-space freezing (biggest).** Frozen real-test `μ_ref,σ_ref` + dimension-normalized
   weights, fixed across the sweep and across methods (A1/A2/A3/B8). This alone defines who is a
   "neighbour"; get it wrong and nothing downstream is comparable.
2. **Quantile estimator for the bands.** `np.percentile` **linear / type-7** interpolation for the
   [5,25,75,95] edges → coverage/width. (No `method=` kwarg is passed, so this is the NumPy default.)
3. **Sliced-Wasserstein projection params/seed → N/A.** This protocol does **not** use
   sliced-Wasserstein; the distributional score is **scalar-quantity CRPS via the energy-score /
   Gini identity** (`term1 − ½·E|Y−Y'|`), which is **exact** for the empirical ensemble — no random
   projections, no projection seed. Documented here so a future method does not "helpfully" add one.
4. **All RNG seeds.** Bank generator `torch.manual_seed(1000 + seed)` (seed 0 → 1000); bootstrap
   `boot_seed=20230814`; RW baseline `default_rng(0)`. Query randomness is fixed by the dataset
   (seed 3). No other RNG touches the metrics.
5. **Bootstrap CI method.** Percentile, 2000 resamples, paired (see D13).
6. **Log-price vs price.** **Everything is computed in LOG space** — returns, the three quantities,
   and the embedding. Banks/queries are stored as **price** and `np.log`'d on load. The forecast
   quantities are returns, so the price-anchor shift cancels (additive-in-log = multiplicative-in-price).

**One design decision above the paper: number of banks.** We use **ONE shared 1M bank** (seed-0
generator), not one-per-model-seed. The paper describes a single bank; the bank-size sweep is nested
prefixes of it; and **all** uncertainty is the query bootstrap (D13). There is deliberately **no
model-seed spread** in the PS numbers — the A/B metrics (§8) already carry the 5-seed generator
dispersion, and mixing a second, bank-sampling source of variance into the PS CIs would muddy them.

### 9.7 Protocol-conformance audit (vs the arXiv:2308.01486 metrics doc; external review 2026-07-28)

An external reviewer checked our reimplementation against the formal metrics doc
(`path_shadowing_metrics_protocol.pdf`). Verdict: **metrics complete, internally consistent, no bug.**
What remained was (A) reconciling our comparison arms with the doc's §4 declared laws, (B) verifying
the 50%-coverage quantile question, and (C) logging the §5.1 selection gates. All three are resolved
below — keep this record so a future method answers them the same way.

**A · Comparison arms vs §4 declared laws (STRUCTURAL — a labeling/scope decision, not a defect).**
The doc §4 lists the compared laws as *"specific-entropy source; MP-corrected generator; independent
Heston oracle."* Our columns are **`<METHOD>` / Heston oracle / RW floor**. Reconciliation:

- **`<METHOD>` (LS4) is a single fixed-checkpoint standalone generator**, evaluated as *the candidate
  law*. It is **not** part of a source → MP-corrected pipeline, so there is **no source/MP-corrected
  pair** and the whole §5.1 checkpoint-selection / §5.2 CRN-pairing machinery (built around a
  source→MP improvement story) **does not apply**. State this in every README so a strict reader is
  not hunting for a missing MP arm.
- **Heston oracle** = the §4 / §6 protocol ceiling — **present and matched.**
- **RW floor** = an **extension beyond §4** (not one of the doc's three laws). Keep it (it is what
  separates a *retrieval* limit from a *generator-law* mismatch) but **label it as a non-protocol
  addition**, not a doc-declared arm.
- **Answer in protocol terms:** LS4 = candidate law (neither "the source with an MP partner" nor "a
  selected checkpoint"); oracle = ceiling; RW = added floor.

**B · Coverage quantile estimator (VERIFY — resolved: NOT a bug, reviewer's mechanism refuted).**
Reviewer hypothesis: coverage 50 sits ~0.47 everywhere (incl. the oracle) while coverage 90 is
well-calibrated (~0.89) — the fingerprint of a quantile-interpolation quirk (nearest-rank vs linear
shifting the 25/75 band inward), not a model defect. Verification:

1. **Code is correct.** `path_shadowing_pdf.py:208` computes `np.percentile(Y,[5,25,75,95],axis=1)`
   (**type-7 linear**, NumPy default, explicitly documented at `:229`); `:209` assigns
   `[q25,q75]`→50% and `[q05,q95]`→90% — the interval mapping is right, no coding error.
2. **Monte-Carlo settles the mechanism.** A *perfectly calibrated* fresh draw scored against a K=256
   ensemble (40 000 trials; normal / Student-t₃ / lognormal) gives, **under our exact type-7
   estimator**: `cov50 = 0.495–0.501` (≈ nominal 0.50) and `cov90 = 0.892–0.895`. Type-6 (Weibull)
   gives `cov50 = 0.496–0.499`, `cov90 = 0.900–0.901`. **Type-7 does NOT under-cover at the 50%
   level** — so the reviewer's "25/75 pulled inward → 0.47" mechanism is **empirically wrong for
   K=256.**
3. **Therefore the observed shortfall is signal, not artifact.** The **oracle is calibrated** at 50%
   (its cov50 ≈ 0.47–0.49 sits within the bootstrap CI of the 0.495 perfect-draw target); LS4's
   marginally lower cov50 (0.467–0.469) is a **genuine mild central over-confidence** of the generator,
   not a bug. The **one** real estimator effect is at the **90%** level: type-7 yields ~0.892 for a
   perfect law (vs type-6's ~0.901), so the oracle's observed **cov90 ≈ 0.89 is exactly the perfect-law
   value under type-7** — an independent confirmation that the retrieval+scoring pipeline is correct.
4. **No number is subtly wrong; no re-run needed.** Optional purity refinement: pass `method='weibull'`
   (type-6) to `np.percentile` for fresh-draw coverage calibration — it lifts **90%**-coverage ~0.8 pp
   toward nominal and leaves the 50% story unchanged. It is **not required** (doc §3.3 says only "25th
   and 75th percentiles", interpolation unspecified) and would be a cross-method change, so if adopted
   it must be applied to LS4, oracle **and** RW together and re-run for all — otherwise leave type-7 and
   footnote it.

**C · §5.1 selection gates (UNREPORTED → recorded as N/A; this is F18).** We do **no** checkpoint
selection: fixed 100-epoch training, exactly **one** checkpoint per seed, no source→candidate step. The
five §5.1 eligibility gates — (i) RV CRPS strictly improves over source, (ii) `|cov90−0.90|` strictly
improves, (iii) cum-return CRPS worsens ≤ ½ bootstrap SE, (iv) declared stylized-fact errors worsen ≤
½ bootstrap SE, (v) path sliced-Wasserstein ≤ 1.10× source — have **no candidate set to filter** and
are therefore **not applied**. Record this line verbatim in the README's deviations blockquote so a
reader does not assume a silent filter ran.

**D · Presentation nits (fix in each README, not the numbers).**
- **Sweep-table `uniq-frac` / `prefix-dist` are the `<METHOD>` bank's values** (the oracle's differ at
  the 4th decimal — the diagnostics table shows `0.118 / 0.119`). Either **label the sweep columns
  "(LS4)"** or split them into `LS4 / ORC` like the CRPS columns.
- **Miss-rate CIs are uneven** (only RV upper-miss carries `[CI]`). Make it **all-or-none** across the
  miss-rate rows.

**E · §1.1 standardization (declared deviation, restated so the audit is complete).** Doc §1.1
standardizes each feature block with the **candidate bank's** `μ/σ`; we use **frozen real-test-set**
`μ/σ` (A1–A3 / the deviations blockquote). Deliberate: it decouples the embedding from the generator so
`<METHOD>` / oracle / RW live on **one comparable scale**. This is a **motivated, documented deviation**,
not a conformance defect.

---

## 10. READMEs

- **Experiment README** ([`README.md`](README.md)): overview, the §3 preprocessing (forward +
  inverse + dummy col), the §3.4 sigma estimation with the value table, the datasets table, and
  the methods table. Already written.

The **per-method README** (`<METHOD>/README.md`) is the deliverable and must reproduce LS4's README
depth **exactly** — same sections, same table schemas, same conventions, same footnotes — with only
`<METHOD>` / `<variant>` / the numeric cells changed. Everything below is the **literal spec**: copy
each header verbatim, fill the cells from `metrics_summary.csv` / `seed_N_metrics.json` /
`pdf_summary.json`, never re-order or drop a row. Section order is Theo's rule (top block ≡ canonical
report so variant and original are directly comparable); the preprocessing + head-to-head verdict go
**below** this block, never above it.

### 10.0 Per-method README — section-by-section table spec

**§0 · Title + intro (no table).** `# <METHOD> on Heston — <variant>`. One paragraph: what the
reference model is (author, venue, arXiv id), that it is trained on **4 096** paths (seq_len 128)
with the `<variant>` transform in place of the model's native input, everything else held fixed
against the original run at `../../<METHOD>/`. Add the **data-split blockquote** verbatim (it is
dataset-derived, identical across methods): generator trained on **seed 0**; every A/B metric scores
generated vs the **test set (seed 1)**; A18 uses a **third real set (seed 2)** as the "real" class;
no metric scored against training data.

**§1 · `## Metrics A1-A34 + B, mean ± std across 5 seeds`.** Lead with the units blockquote
(`> All metrics on log-returns $r_t = \log(S_{t+1}/S_t)$ unless noted. A26 uses price increments
$\Delta S_t$.`). Then an **8-column** table:

| col | header | source |
|-----|--------|--------|
| 1 | `Metric` | fixed label incl. arrow (`↓`/`↑`/`→ 1`) |
| 2 | `Mean ± Std` | 5-seed mean ± sample std |
| 3–7 | `Seed 0` … `Seed 4` | per-seed value |
| 8 | `Perfect floor` | **dataset-derived, identical across methods — reuse LS4's column verbatim** |

Rows are grouped by **7 category separator rows** `| **<Category>** | | | | | | | |` in this
**exact order and range** (37 metric rows total — A18 and A19 each split into a **GRU** and an **MLP**
row):

| separator | metrics | rows |
|-----------|---------|------|
| `**Fat Tail**` | A1–A5 | 5 |
| `**Distribution**` | A6–A17 | 12 |
| `**Adversarial**` | A18 | 2 (GRU, MLP) |
| `**Predictive**` | A19 | 2 (GRU, MLP) |
| `**Temporal**` | A20–A24 | 5 |
| `**Vol**` | A25–A32 | 8 |
| `**Heston Spec**` | A33–A34 | 2 (A33 `↑`, A34 `↓`) |

> **Changed 2026-08-26.** These rows used to be specified — and rendered — as
> `| **, Fat Tail, ** |`, a literal typo in the format string that leaked the tuple separators
> into the markdown, so every page displayed a stray leading and trailing comma inside the bold.
> The same typo produced `↑ higher is better;, no monotone direction` in the convention
> blockquote. Both were corrected across **all 16 d = 1 pages, the d = 8 pages and the
> TrueDataset page in a single commit**, because the d = 8 tree's requirement is that it display
> exactly as d = 1 does — fixing one tree alone breaks that parity. Only the two format strings
> changed; **no numeric cell was touched**. If you are reading an older page that still shows the
> comma form, it predates that commit — re-render it, do not copy it.

Close with the **convention blockquote** (`↓` lower better · `↑` higher better · A28 Kurtosis Ratio
perfect = 1.0) followed by the **7-line per-metric footnote block** defining A1…A34 (copy LS4's
wording verbatim — the definitions are method-independent: A1 kurtosis error, A2/A3 |r| q95/q99, A4
tail-QQ, A5 Hill; A6–A11 MMD²/SWD family with non-zero floor note, A12 RV W₁, A13 mean-path RMSE, A14
KS, A15 skew (true ≈ −0.45), A16 QQ-RMSE 300-pt, A17 terminal-price KS; A18 disc |acc−0.5| GRU+MLP,
A19 TSTR MAE GRU+MLP; A20 covariance %, A21–A22 ACF |r|/r² lags 1-20, A23–A24 ACF lag-1 (true ≈
+0.052/+0.050); A25 mean RMSE, A26 return-std (Δ Sₜ), A27 log-ret std, A28 kurtosis ratio perfect 1.0,
A29 sigma-mean annualized, A30 cross-sect vol-path RMSE, A31 rolling-5 vol KS, A32 vol-of-vol; A33
teacher-σ corr perfect ≈ 0.614, A34 teacher-σ RMSE perfect ≈ 0.065).

**§2 · `## B, Curve-Shape Metrics, mean ± std across 5 seeds`.** First a **methodology preamble**:
each stylised-fact plot yields a *curve* L, not a scalar; from L build the curve, its first finite
difference L′ (der) and second L″ (sec_der), then collapse to one number per plot via five measures —
define each one:

- **MSE** — dᵢ = mean((L_r − L_g)²) per list; reported = **mean of the three** (funct+der+sec_der)/3;
  std = sample std of that combined per-seed score. **The MSE row decides the cross-method winner.**
- **% err** — mean(|L_g − L_r|/(|L_r|+1e-6))×100, MAPE on **L only (funct)**; der/sec_der excluded
  (near-zero truths explode relative error).
- **NRMSE** — sqrt(mean((L_g−L_r)²))/(max|L_r|−min|L_r|+1e-12)×100, funct-only range-normalized RMSE.
- **CVaR₉₀ / CVaR₉₅** — Expected-Shortfall of pointwise error eₜ=|L_g(t)−L_r(t)|: mean of eₜ above the
  q-th percentile (q∈{0.90,0.95}), range-normalized like NRMSE, funct-only.

State: all `↓`; the perfect floor is **non-zero** for all six plots (residual finite-sample error of an
independent Heston draw vs test set, identical across methods). Then a **9-column** table:

`| Plot | Measure | Mean ± Std | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Perfect floor |`

- **Top row = path-cloud:** `| **Path comparison** *(50×50 path-cloud)* | grid_tvd 50×50 (%) ↓ | … |`
  — a single row (grid_tvd, the 50×50 total-variation of the path cloud, in %).
- **Then 6 plots × 5 sub-rows each** (first sub-row carries the bolded plot name in col 1, the next
  four leave col 1 blank): sub-rows in order **MSE · % err · NRMSE · CVaR₉₀ · CVaR₉₅**. Plot order:
  **Log-return histogram · QQ plot · ACF |r| lags 1-20 · ACF r² lags 1-20 · Rolling vol histogram ·
  Tail survival**. (1 + 6×5 = **31 rows**.)

Close with the reading blockquote: every curve sits above its floor; **seed-degeneracy dominates the
std**; ACF %err is inflated by a near-zero denominator so **read the MSE column** for absolute
agreement.

**§3–§6 · Figure sections (headers only — no tables, keep them, one image each).**
`## Stylised Facts Diagnostic (Heston vs <METHOD>+<variant>, seed 0)` → `plots/heston_diagnostics.png`
(8-panel) · `## <METHOD>+<variant> Training Loss (5 seeds)` → `losses/loss_convergence.png` **with the
loss-column prose** (name every logged column — for a VAE-ELBO model: `total = kld + nll`, goes
negative at convergence, plus kld/nll/mse/lr and the EMA setting; adapt the prose to the method's
actual objective) · `## A18, Discriminative Classifier Training Loss` → `plots/disc_classifier_loss.png`
(BCE, ln2 ≈ 0.693 = indistinguishable) · `## A19, Predictive Score Training Loss (TSTR)` →
`plots/pred_score_loss.png`.

**§7 · `## Path Shadowing — strict paper protocol (arXiv:2308.01486)`.** This is the largest table
block. Structure, in order:

1. **Warning blockquote** — this is the **exact paper protocol, NOT** the simplified `methods/<METHOD>`
   reference eval (65D murex embedding, K=77, prefix-price L2, CRPS/MAE/RMSE only; that is **M7**).
   State the fixed config: **K = 256** neighbours, 512 held-out prefixes, prefix = 64 increments,
   horizon 32, bank sizes {4 096 … 1 000 000}, 2 000 bootstrap replicates.
2. **Embedding paragraph** — the 4-block weighted frozen-reference feature: recent returns (last 32,
   w1.0) · cumulative path (downsampled 24, w0.5) · rolling vol (win 5/10/20 last/mean/std, w2.0) ·
   dependence (ACF |r| & r² lags 1,2,5,10, w1.0); each block **dimension-normalized** (`w_block/d`) and
   standardized against **frozen μ/σ** from `heston_S_test_4096x128`: `z̃ = √(w/d)·(z−μ_ref)/σ_ref`.
   State cum/step are **H-dim trajectories over u=1…H** (§9.6 E15), RV is the scalar `√Σr²`; s=64,
   H=32, 512 query paths (seed 3); the 8 metrics; 2000-resample paired bootstrap `boot_seed=20230814`.
3. **Three-references paragraph** — same pipeline over three nested-prefix banks
   {4096,16384,65536,262144,1000000}: **`<METHOD>+<variant>`** (seed-0 1M bank under test), **Heston
   oracle** (ceiling, fresh 1M true-Heston bank, **seed 777** — gap-to-oracle = law-mismatch, oracle's
   own residual = irreducible finite-bank retrieval limit), **Random-walk** (floor, resamples each
   query's own prefix returns).
4. **Deviations blockquote** — (a) per-block dimension normalization `√(w_block/d)` vs §1.1's `√w_b`;
   (b) frozen-reference standardization vs §1.1's per-bank μ/σ; **§5.1 gates N/A** (fixed-epoch, no
   checkpoint selection). Both decouple the metric from the generator.
5. **`<!-- PS-PDF-TABLE-START -->` marker**, then a one-line note (numbers at the **1 000 000** bank,
   log-return scale, lower better except coverage whose target is the nominal 0.50/0.90; brackets =
   95% bootstrap CI over 512 queries; cum/step horizon-averaged), then **three 4-column quantity
   tables**, header `| metric | <METHOD>+<variant> | Heston oracle | RW floor |` (col alignment
   `|--------|:----------:|:-------------:|:--------:|`), each with the **same 8 metric rows** — bold
   the winning cell per row:

   | # | quantity table | 8 rows |
   |---|----------------|--------|
   | 1 | **Cumulative return (trajectory, u = 1…H)** | RMSE · CRPS · coverage 50 · coverage 90 · width 50 · width 90 · lower-miss 90 · upper-miss 90 |
   | 2 | **One-step return (trajectory, u = 1…H)** | same 8 (LS4 collapses width & miss onto shared `width 50 / 90` and `lower/upper-miss 90` rows when values are near-degenerate — follow LS4's exact row layout) |
   | 3 | **Horizon realized vol (scalar) — the diagnostic quantity** | same 8 |

   RMSE/CRPS/coverage rows carry `[CI_low, CI_high]` brackets; width rows are point values.
6. **Reading paragraph** — where the method sits vs the oracle ceiling and RW floor per quantity;
   name the one quantity with a genuine gap (for LS4: **RV upper-tail under-coverage = law-mismatch**).
7. **Bank-size sweep table** — `| bank size | cum LS4 / ORC | step LS4 / ORC | RV LS4 / ORC | uniq-frac
   | prefix dist (mean) |` (align `|----------:|:...:|`), **5 rows** = the five bank sizes, each cell
   `method / oracle` CRPS; last two cols = unique-candidate fraction and mean prefix distance. Follow
   with the sweep-reading paragraph (cum/step flat across the 244× increase = saturates ~4k; only RV
   improves; the method↔oracle RV gap constant = law-mismatch not finite-bank).
8. **Diagnostics table (1M bank)** — `| terminal (h=H) MAE | prefix dist mean/median/p95 | unique-cand
   frac | RV mean bias |`, one data row, cells `method / oracle`. Note terminal (h=H) MAE is a genuine
   single-horizon diagnostic **distinct** from horizon-averaged cum RMSE (the M-tagged identity bug is
   fixed). **Label it MAE, not RMSE** — the driver computes `float(np.abs(pred_term_mean −
   y_term).mean())`, a mean *absolute* error, even though the JSON key is still `terminal_rmse`
   (`path_shadowing_pdf.py` ~L349, and `forecaster/pdf_bridge.py` L92). It is the h = H analogue of the
   rv MAE row, **not** the textbook RMSE of E16b. Renaming the key would require re-running every bank,
   so the key stays and the *label* carries the truth. Point to `path_shadowing/pdf_summary.json` for
   all CIs.
9. **`<!-- PS-PDF-TABLE-END -->` marker**, then the driver/bank-builder links and the two `pdf_*.png`
   plot embeds.

   > **Never** the old murex K=77 CRPS-only H=32/64 reference eval (M7). This slot's *position* mirrors
   > the template's "Path Shadowing MC" but its *content* is strictly the paper protocol.

**§8 · `## File layout`** — the folder tree (fenced block): `README.md`, `metrics_summary.csv`,
`seed_{0..4}_metrics.json`, `generated_paths/`, `weights/`, `losses/`, disc/pred loss csvs, `plots/`,
`code/` (`train_<method>_<variant>.py` + `compute_metrics_<variant>.py`), and `path_shadowing/`
(`path_shadowing_pdf.py`, `gen_banks.py`, `bank/` **noting it is gitignored / disk-only and regenerable, not pushed (M10)**, `pdf_summary.json`, `logs/`, `plots/`).

**§9 · `## Reproduce`** — one fenced bash block: `PY=/home/tbasseras/gpu-venv/bin/python`, the 5-seed
2-GPU training loop (`for gpu in 0 1 … taskset -c … OMP_NUM_THREADS=8 … --seed $gpu & done; wait`, per
§8 hardware rules), the metrics runner, then the **strict PS** two-liner (build the ONE 1M bank via
`gen_banks.py --seed 0`, then `path_shadowing_pdf.py` — **no `--seeds`**).

### 10.0.1 Below the template block — preprocessing + head-to-head (two sections, this order)

**§A · `## Preprocessing (the <variant> transform)`.** State the *only* changed thing. A fenced
Python block: the forward transform (log-returns → scale by `√DT/sigma` → dummy-0 column → unit-var
wrapper), the **reported frozen sigma value** (`sigma = 0.01263163`, ddof=0, pooled train seed 0; after
scaling `R̃.std() = √dt`). Add the **unit-variance-wrapper blockquote** (§0.1: scaled returns std ≈
0.063 < decoder noise 0.1 collapses the VAE, so apply the model's own `(X−x_mu)/x_sd`, inverted
symmetrically; reported sigma unchanged) and a **Model/Dataset** line (preset, param count, z_dim,
d_model, layers, decoder sigma, 100 epochs, optimizer; 4 096 paths, Heston params).

**§B · `## Results — does <variant> help <METHOD>?`.** The head-to-head verdict built from §7.1's
no-preproc baseline. Required, in order:

1. **One-line thesis** stating the winner up front, with the net row count (e.g. "Net **19 rows to
   16**; wins concentrated in the dependence/tail block, losses in marginal-moment calibration").
2. **Sub-header `### A-metrics (seed 0, 4096 both sides; ↓ lower-better unless noted)`** — a **5-column**
   table `| A-metric | <variant> (with) | raw (no-preproc) | Δ% | Winner |` (align: metric left, two
   value cols right, Δ% right, Winner center). One row per compared A-metric (LS4 lists ~26: A1,A5–A14,
   A17–A22,A25–A28,A30–A34). **Δ% relative to no-preproc; negative = with-preproc better on lower-better
   metrics** (footnote the orientation). `A28`/`A33` (non-lower-better) show `—` in Δ% with the
   ratio/|Δ| inline. **Bold the winning cell**; winner col = `**<variant>**` or `raw`.
3. **Sub-header `### B curve-shape (seed 0; funct MSE, %err, + grid_tvd; ↓ lower-better)`** — same
   5-column shape; rows = the six curve MSE **and** %err pairs (log-ret hist, QQ, ACF|r|, ACFr²) plus
   the `grid_tvd (path cloud)` row.
4. **Side-by-side stylised-facts diagnostic (required, directly below the B-curve table).** A **2-column
   image table** giving the visual A/B of the same eight panels, `<variant>` (with) vs raw (no-preproc):

   ```
   | log-return preprocessing (with) | raw price, no preprocessing |
   |:-------------------------------:|:---------------------------:|
   | ![with](plots/heston_diagnostics.png) | ![raw](baseline_no_preproc/plots/heston_diagnostics.png) |
   ```

   Both panels use the **same real test set at seed 0, 4096 paths** — only the input transform differs,
   so the ACF-of-|r| / ACF-of-r² and rolling-vol panels are the visual counterpart to the ACF/vol Δ%
   rows. The raw baseline's `heston_diagnostics.png` is **not produced by the baseline metrics runner**;
   generate it once (CPU, no retrain) from the already-saved baseline paths:

   ```python
   # from benchmark/ root, ~5 s, CPU
   import sys, numpy as np; sys.path.insert(0, "metrics")
   from plot_diagnostics import plot_diagnostics
   R = "results/Heston/preprocessing_with_log_returns/<METHOD>/baseline_no_preproc"
   S_real = np.load("dataset/Heston/preprocessing_with_log_returns/heston_S_test_4096x128.npy")
   S_gen  = np.load(f"{R}/generated_paths/seed_0/generated_paths_4096x128.npy")
   plot_diagnostics(S_real, S_gen, method="<METHOD> raw (no-preproc)", seed=0,
                    out_path=f"{R}/plots/heston_diagnostics.png")
   ```

   Use `metrics/plot_diagnostics.plot_diagnostics(S_real, S_gen, method, seed, out_path)` — the **shared**
   8-panel generator (identical across methods, price space, S0 ≈ 100). Never hand-roll a second plotter.
5. **Reading list** — 3 bullets: (a) preprocessing wins the **stylised facts** (name the ACF/tail Δ%);
   (b) the raw baseline wins **marginal dispersion** (A26/A27 std, A18 discriminator); (c) **why this
   differs from any "5-seed mean vs 8192-path original" verdict** — the matched 4096/seed-0 control
   isolates the one variable (M4).
6. **The 4096-path caveat** — both sides at 4 096 (not the main benchmark's 8 192): internally valid
   (same budget) but not the main-benchmark *absolute* level.
7. **Bottom-line paragraph** — which scaler wins and **why** (tie to §0.1: does the model have a fixed
   output-noise floor that unit-variance input rescues?), plus the caveats to fix. Latent projections,
   if kept, are a `### Latent projections (per seed)` subsection here (PCA/t-SNE image grid).
8. **Footer links** — `../README.md` (overview) · `../GUIDELINE.md` (recipe) · `../../<METHOD>/README.md`
   (original price-input run).

Follow GitHub math rules: no `\;`/`\,` in `$$`, no trailing comma on intermediate `$$` lines, no bare
`*` in superscripts.

### 10.1 The end-of-run report (what to hand back in chat for a new method)

When a method is finished, deliver a report with **this** structure so every method is reported at the
same depth (mirror the level of detail LS4 received):
1. **Headline verdict** (1–2 sentences): does the preprocessing help this method, and the single most
   telling metric supporting it.
2. **Gate outcome** (§7): pass / flagged / failed-then-fixed, with the root cause if it collapsed
   (§0.1 pattern: input std vs output-noise floor).
3. **A/B metrics** — 5-seed mean±std headline table, plus the **with- vs no-preproc seed-0** head-to-head
   (§7.1) with the winner called per block.
4. **Path shadowing** (§9, strict protocol): the cum/step/RV story across the bank-size sweep vs the
   **Heston-oracle ceiling** and **RW floor** — is the generator on the oracle, and where does it
   under-cover? Full 8-metric set, bootstrap CIs. **Never** the murex reference eval (M7).
5. **Artifacts & repro** — the exact paths written (weights, generated paths, `pdf_summary.json`, plots)
   and the one-command repro. **The 1M bank is NOT pushed.** It stays on disk under
   `<METHOD>/path_shadowing/bank/` and is **gitignored**; it is regenerable via `gen_banks.py --seed 0`
   (M10), which is what makes shipping it unnecessary. *(Historical note: the LS4/CSDI/SBTS banks were
   committed via Git LFS before this rule and still have `.gitattributes` entries; that decision was
   reversed — do not add a new bank to `.gitattributes`.)*
6. **Mistakes ledger delta** — if any new trap fired, it must already be a new `M<n>` row in §0.2
   *before* the report is considered complete (this session added M8–M10).
7. **Commit** — conventional message + `Co-Authored-By`, only the two `preprocessing_with_log_returns/`
   folders touched (no `methods/`, no `results/Heston/LS4/`).


### 10.2 Experiment README — cross-method display (what a NEW method must add)

The experiment README ([`README.md`](README.md)) holds four cross-method tables — **A1-A34**, **B**, the
**PS** five-bank sweep and the **PS sweep summary** — all emitted by
[`build_cross_tables.py`](build_cross_tables.py) (`--which A|B|PS|PSSUM`, `PS` with `--all-bank-sizes`).
**Never hand-type a cell**: register the method in the script and paste its stdout. (M-class trap: a
hand-typed floor silently drifts out of sync with the JSON it claims to quote.)

> The **sweep-summary** table (`--which PSSUM`, marked in the README with a
> `<!-- generated: ... do not hand-edit -->` comment) was hand-typed until 2026-07-31 and went stale
> **twice** — once when TimeDiT (raw) entered `PS_MODELS`, once when cum/step moved to the textbook RMSE
> (E16b). It is now produced by `render_PS_sweep_summary()`, which derives its win-counts from the *same*
> `render_PS_strict()` call that renders the big tables, so a summary contradicting the table it summarises
> is no longer expressible. Ties break alphabetically so a re-run never reshuffles the row text.

**Registering a new method.** Add one tuple to the relevant list at the top of `build_cross_tables.py`:

| list | entry | bank-size axis? |
|------|-------|-----------------|
| `AB_MODELS` | `("<METHOD>", "<METHOD>/metrics_summary.csv")` | n/a (A + B tables) |
| `PS_MODELS` | `("<METHOD>", "<METHOD>/path_shadowing/pdf_summary.json", "gen")` | **yes** — retrieval over the bank |
| `PS_MODELS` | `("<NAME>", "forecaster/<name>_pdf.json", "fc")` | **no** — forecasts directly, retrieves nothing |

Insert the tuple in **model-family position** (Diffusion -> VAE -> Schrodinger Bridge -> forecasters), not
at the end of the list — column order in the README follows family.

**The PS display is a five-table sweep, not one table.** Section 9.1 evaluates every generator at five
nested prefixes of the same 1M bank, so the README shows the *same* table once per bank size:

```bash
python build_cross_tables.py --which PS --all-bank-sizes    # 5 tables + per-size win-counts
python build_cross_tables.py --which PS                     # 1M table alone
```

Invariants the renderer enforces — preserve them if you touch it:

1. **Sizes are read from the data, never hardcoded.** `ps_bank_sizes()` reads `by_bank_size` off the first
   `PS_MODELS` entry, so a Section 9.1 protocol change propagates by itself.
2. **The oracle moves with the bank.** Each table reads `heston_oracle.by_bank_size[<size>]`, so the ceiling
   is size-matched 1:1 against the generator column (4 096 vs 4 096, ... , 1M vs 1M). Showing the 1M oracle
   against a 4 096 generator bank is a protocol violation, not a display choice.
3. **Forecasters and the RW floor have no bank-size axis**, so their values repeat unchanged in all five
   tables. Their headers must carry the `<sup>bank-independent</sup>` tag and the intro paragraph must say
   so — otherwise the repetition reads as a copy-paste bug.
4. **Winner excludes the oracle and the RW floor**; the two `width` rows are diagnostic and select no
   winner (width without its coverage is meaningless). 18 ranked rows = 6 ranked metrics x 3 quantities.
   **`PS_EXTRA_ROWS` + the `"min-uncounted"` rule.** A quantity may carry extra rows spliced in by
   `ps_rows(qn)` directly after its RMSE row. The only current entry is
   `"rv": [("rmse_textbook", "RMSE ↓ (true, root-last)†", "min-uncounted")]` — the rv **true RMSE**,
   added 2026-07-31. At H = 1 the two keys are two standard statistics of the *same* 512-vector of
   errors (MAE = `mean_q|e_q|`, RMSE = `√(mean_q e_q²)`), so both are worth showing, but ranking both
   would **double-count rv** against cum and step and silently reweight the sweep summary. Hence the
   rule: `_ps_winner_idx` still returns the argmin, the cell is still bolded and the winner named
   (rendered `<i>Name</i>†`, italic + dagger, never `<b>`), but the row is **not** added to `wins`.
   The denominator stays **18**; verified byte-identical win-counts across all five banks before and
   after the row landed. Adding a *counted* row means editing `PS_METRICS` and updating every "of 18"
   claim in the README, the sweep summary and the report — do not do it casually.
5. **Layout.** The four sub-1M tables go in collapsed `<details>` blocks whose `<summary>` carries the
   win-count; the **1M table is the expanded headline**, and all prose below it quotes 1M numbers.
6. A **win-count summary table** (`bank | winners | the sweep-sensitive metric`) follows the five tables.

**Read the sweep before writing the verdict.** Bank size can flip the ranking outright: SBTS wins **14/18**
rows at 4 096 and **0/18** at 1M, its cum-CRPS degrading monotonically (0.0254 -> 0.0318) straight past the
RW floor (0.0295) while its coverage90 collapses 0.88 -> 0.66. A small bank returns loose prefix matches, so
the retrieved ensemble inherits its spread from the *marginal* law — which is exactly the property a
conditional-variance collapse hides behind. **Only the 1M table answers the intended question**; quoting a
small-bank win as the headline reports an artefact.

**A method with no PS bank yet** (TimeDiT's case at time of writing) stays **out** of `PS_MODELS` and gets an
explicit blockquote in the README's PS intro stating why, linking its per-method README. Never a row of
`-` placeholders, and never a silent omission.

#### 10.2.1 The two injectors — nothing is pasted by hand any more

Pasting stdout is still a manual step, and manual steps rot. Two scripts close that gap; run **both**
after any change to `build_cross_tables.py`:

| script | target | span it replaces |
|--------|--------|------------------|
| [`inject_readme_ps.py`](inject_readme_ps.py) | experiment `README.md` | the four collapsed `<details>` sweep tables + the expanded 1M headline + the `<!-- PS strict win-counts` comment |
| [`render_method_ps.py`](render_method_ps.py) | `{CSDI,LS4,SBTS,TimeDiT}/README.md` | the three per-method *method vs oracle vs RW* tables at the 1M bank |

Both **detect their bounds, never hardcode line numbers** — an earlier throwaway revision of the first
one pinned literal indices, which is exactly what mangles a file the moment a row is added above the
block. `inject_readme_ps.py` anchors on the first `<details>` following the "nested prefixes of the same
1M bank" paragraph and ends at the win-count comment. `render_method_ps.py` starts at the
`**Cumulative return` heading and ends **the moment the third markdown table closes** — *not* at the
next prose heading. That distinction is load-bearing: TimeDiT carries a bank-size sweep and a
diagnostics table between its rv table and its "Reading it" paragraph, and a heading-based end marker
silently swallowed both on the first run. Prose is never touched by either script.

`render_method_ps.py` shares `ps_rows` / `_ps_key` / `_ps_winner_idx` with the cross-method renderer, so
the per-method tables cannot disagree with the root README again — they *did* disagree between
2026-07-30 and 2026-07-31, showing root-inside cum/step while the root table showed textbook. Row labels
are compact (`ROW_LABEL`) because the per-method header already states the direction; only the dagger
survives, and the per-method preamble must explain what it means.

**Both are idempotent** — a second run must report the identical replaced span and line count. Check it;
a drifting span means the bound detection has broken.


---

## 11. Checklist for a new method

- [ ] Confirmed reference model imports and runs with its released preset (no edits to reference).
- [ ] `train_<method>_logret.py` applies §3 forward + inverse; round-trip asserted to ~1e-13.
- [ ] Decided on the §3.3 unit-variance wrapper (default: include it; the gate confirms).
- [ ] `sigma = 0.01263163` reported; matches SBTS estimator.
- [ ] **`--epochs` default edited to the canonical count in the script *and* passed explicitly** (`wc -l methods/<METHOD>/losses/seed_0_losses.csv`; LS4 = 100). Never rely on the 400 default; fix it at source so a cloned script can't re-fire it. **(M1, M8)**
- [ ] **`seed_N_config.json` diffed against the original hyperparameters** — only the §3 scaler and 4096 count differ. **(M4)**
- [ ] Training launched **without a `tail` pipe** (background/log file) so `[ep N]` lines stream live for ETA. **(M2)**
- [ ] Seeds **packed on one GPU** on separate cores to reach ~full util (not one process at ~44%). **(M3)**
- [ ] **Env vars (`CUDA_VISIBLE_DEVICES`/`OMP_NUM_THREADS`) placed *before* `taskset`**, not after — else exit 127. **(M5)**
- [ ] Jobs **detached with `setsid … < /dev/null &` + `disown -a`** so they survive the launcher shell exit; verified with `pgrep`+`nvidia-smi`. **(M6)**
- [ ] Correct venv: **`/home/tbasseras/gpu-venv/bin/python`** (torch+cu130), not `~/.cc-venv`. **(§0.3)**
- [ ] Trained **seed 0**; ran metrics; ran `gate_compare.py`.
- [ ] **GATE passed** (or fix approved by user). Old-vs-new figures inspected.
- [ ] Trained seeds 1–4 (§6.2, 2 GPUs max); full `metrics_summary.csv`.
- [ ] Built the **one** shared 1M path-shadowing bank (`gen_banks.py --seed 0`, §9.5, logged).
      **STRICT paper protocol** run via `path_shadowing_pdf.py` — 4-block dim-normalized
      frozen-reference embedding + single-bank nested-prefix sweep + cum/step/RV + coverage/width +
      bootstrap CIs; `pdf_summary.json` + plots produced. Every §9.6 decision honoured. **Did NOT
      reuse the reference murex eval (M7).**
- [ ] **Generation *and* evaluation logged** (`logs/gen_seed0.log`, `logs/pdf_run.log`) — never flew blind (M2).
- [ ] **No-preproc baseline (§7.1) trained** — cloned `train_<method>_raw.py` (global standardize, ONLY the scaler differs), **default epochs fixed to 100 (M8)**, seed 0, A/B metrics computed (no bank).
- [ ] **Head-to-head "Results" section** written below the README template block: A + B 3-column with-vs-no-preproc Δ tables, 4096 caveat, verdict declaring the winner (§10).
- [ ] **Side-by-side stylised-facts diagnostic** placed directly below the B-curve table — raw baseline `heston_diagnostics.png` generated via `metrics/plot_diagnostics.plot_diagnostics` (CPU, no retrain), 2-column with-vs-raw image table (§10.0.1 §B item 4).
- [ ] Per-method README written (mirror the main benchmark's).
- [ ] **All metric tables cross-checked against `pdf_summary.json` / `seed_N_metrics.json`** — every quantity has its full metric set, none dropped (M9).
- [ ] 1M bank left **on disk and gitignored — NOT pushed** (`<METHOD>/path_shadowing/bank/*.npy`); it is regenerable via `gen_banks.py --seed 0`, which is why shipping it is unnecessary. Do **not** add a new `.gitattributes` LFS line (the LS4/CSDI/SBTS entries are historical; that decision was reversed). No plain-blob push, no `--force` (M10).
- [ ] **Method registered in [`build_cross_tables.py`](build_cross_tables.py)** — `AB_MODELS` + `PS_MODELS` (in model-family column position, `"gen"` vs `"fc"` kind); the experiment README's A/B/PS tables regenerated from its stdout, **PS emitted with `--all-bank-sizes`** (5 nested-prefix tables, 1M expanded as the headline) **and the sweep summary re-emitted with `--which PSSUM`** (it carries the win-counts, so a new method changes it). No hand-typed cells (§10.2). **Then run both injectors** — `python inject_readme_ps.py` (experiment README) and `python render_method_ps.py` (all four per-method READMEs) — and re-run each once more to confirm the replaced span is identical (§10.2.1).
- [ ] **End-of-run report** delivered in the §10.1 structure; new traps recorded as `M<n>` in §0.2 first.
- [ ] Everything lives under the two `preprocessing_with_log_returns/` folders; no reference file
      touched.
