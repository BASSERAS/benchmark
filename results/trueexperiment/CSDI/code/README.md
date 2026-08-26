# CSDI on TrueDataset (real crypto, d = 8) — implementation notes

CSDI, Tashiro et al., *Conditional Score-based Diffusion Models for Probabilistic
Time Series Imputation*, NeurIPS 2021 ([arXiv:2107.03502](https://arxiv.org/abs/2107.03502)),
run in its **unconditional generation** mode on the locked TrueDataset build
`om_2022-07_N6144`, tag `6144x128x8`.

This file is the §2 pre-flight of `../../truedatasetguideline.md`. Read it before
reading the result tables; several of the answers below constrain what those
tables are allowed to claim.

---

## The five §2 questions

### 1. Does the method natively accept `(N, T, d)` input?

**Yes. This is one joint model over all eight assets, not eight univariate models.**

`target_dim` is a constructor argument that flows into the feature embedding and
into `diff_models.diff_CSDI.forward_feature`, a Transformer that attends **across
assets** at every one of the 4 residual blocks. Setting `target_dim = 8`
therefore gives a genuinely multivariate model, and **A20 (the cross-asset
covariance row) is a question this architecture can actually answer** — unlike a
per-asset ensemble, which would answer it by construction with whatever
correlation the shared noise happened to induce.

That distinction is load-bearing on this dataset: the 28 realised cross-asset
correlations average **0.609** (range 0.515–0.801), so an ensemble of eight
univariate models would be discarding the majority of the dependence structure.

Concretely, the only thing this port adds to the authors' model is
`CSDI_TrueData` in `csdi_true.py` — a subclass that supplies `process_data` and
pins `cond_mask ≡ 0`. `get_side_info`, `calc_loss`, `calc_loss_valid` and
`impute` are the parent's code, untouched.

### 2. Does any hyperparameter fail to cross datasets?

**Unknown, and deliberately not investigated.** Every hyperparameter is the
authors' released `reference/config/base.yaml` verbatim — the config that
reproduced the paper's Table 2, and the same one the `d = 1` run in
`methods/CSDI` and the `d = 8` Heston run both used. `weights/seed_i_config.json`
records `paper_hyperparams: true` and an empty `retuned_for_truedata: []`.

This is a **choice with a cost**, and the cost is stated rather than hidden:

- **The epoch budget is not equalised across datasets.** `epochs = 200` at
  `batch_size = 16` gives 6144/16 = 384 steps/epoch here, i.e. **76 800 gradient
  steps**. The `d = 8` Heston sibling ran N = 8192 → 512 steps/epoch → 102 400
  steps. So this run gets **25 % less optimisation for the same nominal
  hyperparameter.** Decided 2026-08-26 with the repo owner: keeping the number
  the authors published is worth more than equalising a budget across two
  datasets that differ in a dozen other ways, and it keeps
  `paper_hyperparams: true` honest.
- **The evidence for whether 200 was enough is the validation curve**, not an
  assertion. `plot_losses.py` prints a warning if the last 20 % of steps account
  for more than 10 % of the total validation descent, which is the signature of a
  budget that ran out. Read `plots/loss_convergence.png` before trusting the
  A-table.
- Because nothing was tuned here, **there is no selection pressure to disclose**:
  no hyperparameter on this page was chosen by looking at any metric on this
  dataset. That is the one advantage of refusing to re-tune, and it is why the
  memorisation number (§3 below) is an out-of-sample readout rather than a
  constraint that was optimised against.

The one thing that *is* dataset-specific is the input scaling, and it is not a
hyperparameter — see **Input space** below.

### 3. What is the memorisation risk?

**Real, and measured. `measure_memorisation.py` is mandatory here.**

It is tempting to treat the NNratio diagnostic as an SBTS formality — SBTS has a
bandwidth `h` that is literally the copying knob, CSDI has no such knob. That
reasoning is wrong. **413 057 parameters are fitted to 6 144 training paths for
200 epochs**, so every training path is visited 200 times. A DDPM with that much
capacity relative to its sample count can place mass on near-copies of individual
training examples, and it will do so *while winning every distributional metric
in the A-table*, because reproducing the training set is how you win those.

Reported in `losses/memorisation.json`. Denominator is **`val`**, not `test`
(§9.1): the build is holdout-era and the test era sits **closer** to train than
val does (0.932×) purely because its returns are smaller, so a test denominator
inflates every ratio and pushes a memorising generator towards the healthy end.
Healthy band **0.932–1.000**.

Note that `n_exact_duplicates` is expected to be 0 and clears nothing — ancestral
sampling from a continuous density lands on a bit-identical float64 path with
probability zero. That counter can only fire on a plumbing bug. **The ratio is
the number that matters.**

### 4. Compute budget

Measured on one A100-SXM4-80GB, 8 pinned cores, `OMP_NUM_THREADS=8`:

| Stage | Cost |
|---|---|
| Training, 200 epochs, N = 6 144, T = 128, d = 8 | **≈ 11 s/epoch → ≈ 37 min/seed** |
| A/B bank, 6 144 paths, 50 diffusion steps | ≈ 1.5 min/seed (written by `train_true.py`) |
| Conditional-CRPS pool, 8 192 paths | ≈ 2 min/seed (`generate_bank_true.py`) |
| Metrics (`compute_all_multiasset.py`) | ≈ 8.5 min/seed |

Four seeds on two GPUs = two rounds ≈ **75 min wall-clock** for all training.
Sampling is cheap because `num_steps = 50`, not the 1000 a vanilla DDPM would use.

Do **not** run more than 2 GPUs — the machine is shared.

### 5. Which era does the model see?

**Train and val only. Both are past-era, so the run is valid.**

`weights/seed_i_config.json` records `splits_read: ["train", "val"]`.
`train_true.py` loads `true_S_6144x128x8.npy` (train) and
`true_S_val_6144x128x8.npy` (val, used for the loss curve only — there is no
early stopping, the run goes the full 200 epochs). **`disc` and `test` are never
opened by any script in this directory that writes a weight or a bank.** They are
read only by `measure_memorisation.py` (test, for the context denominator) and by
the metric scripts, which is where they belong.

The z-score statistics are fitted on the **training split only** and stored in the
checkpoint. `generate_bank_true.py` reloads them from the checkpoint rather than
recomputing from `--data-dir`, so pointing that script at a different build
produces an error, not a plausible-looking bank.

---

## Input space — the one substantive porting decision

```
S (price) --(-mean_a)/std_a--> standardized   [model trains and samples here]
sample    --*std_a + mean_a--> price
```

Per **channel**, following CSDI's own PhysioNet convention and the `d = 8` Heston
sibling. Decided 2026-08-26 with the repo owner over the log-return alternative.
Two facts a reader needs at the point of use:

- `mean_a ≈ 100.006` for every asset and `std_a` ranges **0.373 (BTC) to 0.853
  (DOGE)**. That is a **30× smaller dynamic range** than the Heston sibling saw
  (std 11.14–18.55), because a 64-minute crypto window moves far less than a
  one-year Heston path. The z-score absorbs it — that is what it is for — but the
  absolute reconstruction error *in price space* is scaled by `std_a`, so a given
  error in standardized space costs 30× less here than there. Do not compare raw
  loss values between the two runs.
- Every real path is anchored at `S[:, 0, :] == 100` exactly, so after
  standardisation **every path starts at the same value**, `(100 − mean_a)/std_a
  ≈ −0.016`. That first marginal is a point mass, and the model must spend
  capacity learning it.

A per-channel affine map is a shift-and-scale of each asset independently, so it
leaves every cross-asset **correlation** invariant. **A20 is unaffected by this
choice.**

### `rescale_to_s0` and the failure mode it exposes

Guideline §4 requires `S[:, 0, :] == 100.0` exactly. `rescale_to_s0` imposes it
with a per-(path, asset) constant multiplier, which is exactly a shift of the
log-price level — so **every log-return is bit-identical before and after**, and
the whole A-table except the level rows is unaffected by construction.
Overwriting the first row alone would instead dump the entire correction into the
first increment.

The clipping branch is different and is **a real distortion**: a non-positive
sampled price is clipped to `1e-6`, which does not preserve log-returns. It is
therefore counted and reported, not hidden:

- `n_nonpositive_total_before_rescale` — interior clips.
- `n_nonpositive_s0_before_rescale` — **clips at t = 0, which are far worse.**
  The multiplier is `100/S[:, 0, :]`, so a first price clipped to `1e-6` rescales
  that entire path by `1e8`.

**This is not hypothetical.** The 2-epoch smoke run produced exactly one such
path out of 64, and the resulting bank had `S_max = 1.27e10` and
`generated_mean = 1.57e7` while still passing the §4 contract — every entry
finite and positive. A single path like that destroys every aggregate in the
A-table. `collect_artifacts.py` prints a warning whenever the counter is
non-zero; **if it is non-zero on a real 200-epoch run, the bank is not usable and
the number must appear in the README.**

---

## Anticipated result: where CSDI should lose, and why that is not a bug

The A-table rows to read sceptically are **A14/A16 (the zero-return mass)**, per
asset rather than averaged. TrueDataset is 30-second bars, and on the thin assets
a large share of bars have **exactly zero** return — LINK 24.4 %, ADA 22.0 %,
BNB 20.9 %. A continuous-density diffusion cannot place an atom at zero; it can
only place a narrow continuous bump there. So CSDI is expected to lose on those
rows specifically, and the loss is a property of the model class, not of this
port. **Report it per asset.** An eight-asset average would smear a structural
failure on three assets into a mild-looking aggregate.

---

## Files

| File | Role |
|---|---|
| `csdi_true.py` | Library half: `CSDI_TrueData` subclass, `load_split`, `zscore_stats`, `rescale_to_s0`, `BASE_CONFIG`. Imported by both entry points so the normalisation chain exists once. |
| `train_true.py` | Fits one seed, writes the checkpoint, the loss CSV and the **6 144-path A/B bank**. |
| `generate_bank_true.py` | Reloads a checkpoint and draws the **8 192-path conditional-CRPS pool** (§8) from the *same weights*, so the A-table stays a sanity gate on Table C. |
| `collect_artifacts.py` | **A gate, not a report.** Recomputes the §4 contract from every `.npy` and exits non-zero on violation. Writes `losses/generation_time.csv`. |
| `measure_memorisation.py` | NNratio diagnostic (§9). Writes `losses/memorisation.json`. |
| `plot_losses.py` | `plots/loss_convergence.png`. Two panels; see the docstring for why their levels are not comparable. |
| `plot_diagnostics_true.py` | `plots/truedata_diagnostics.png`. Verbatim copy of the SBTS file apart from one renamed path constant — keep it that way. |
| `render_readme.py` | Renders `../README.md` from the JSON/CSV artefacts. Never hand-edit the rendered file. |
| `reference/` | The authors' released code, byte-identical. See below. |

### Reference copy

`reference/` is a third tracked copy of the authors' released code, mirroring the
`HestonMultiAsset/CSDI` arrangement, so this tree runs standalone. It is a
selective copy of `methods/CSDI/code/reference/` — all `.py`, `LICENSE`,
`README.md`, `requirements.txt` and `config/*.yaml`; the authors' `data/` and
`save/` directories are excluded because they are inputs and outputs of *their*
experiments, not of this one.

Verified byte-identical at copy time (2026-08-26), md5:

```
8f98c47682904c1981e811d707d73f63  config/base.yaml
063c7ba60ec676e9fc93677276cb76b9  config/base_forecasting.yaml
4694619c6329cbc861e568c0a08ea397  config/base_smoke.yaml
b6eb4a28e1993684846aba75ebb931ec  dataset_forecasting.py
c9708f394d09cdf74b9e8f02d49bca70  dataset_physio.py
f9ae03c80329d54c9a6e82cf63d02e59  dataset_pm25.py
83f22ca56b7abce4da1ead1d3e832777  diff_models.py
25417b358948203b1a94cf0c75d178f8  download.py
519680722044ab8e1a57b7a247c608ac  exe_forecasting.py
2c20f75f441f2c878cd6c3c0965b28a6  exe_physio.py
b13b253e0bc6323d975b7b45a45aba1f  exe_pm25.py
78f626a45d68ea8748a726aebd85849e  LICENSE
6355f329355ec9648f741486166b249a  main_model.py
609059f6fe9634c37111cbd3628cd266  README.md
c96d67c448b2771e5b61ccf4345e59d0  requirements.txt
693bf4f5d64cb6b5ffb983af89a66d5a  utils.py
```

Re-verify with:

```bash
cd /home/tbasseras/benchmark
for f in $(cd results/trueexperiment/CSDI/code/reference && find . -name '*.py' -o -name '*.yaml' -o -name 'LICENSE' -o -name '*.txt' -o -name '*.md' | sort); do
  diff -q "results/trueexperiment/CSDI/code/reference/$f" "methods/CSDI/code/reference/$f" || echo "DRIFT $f"
done
```

### Why unconditional generation is `is_unconditional=1` **plus** `cond_mask ≡ 0`

The paper (Sec 4.1 / Appendix C) states the `is_unconditional=1` variant "can
also be used for data generation". In that mode `CSDI_base.set_input_to_diffmodel`
feeds the network **only** the noisy sequence — `cond_mask` never gates the
network input, it only selects which points enter the loss via
`target_mask = observed_mask − cond_mask`. So with `observed_mask ≡ 1` and
`cond_mask ≡ 0`:

- **training** → `target_mask ≡ 1` everywhere → every timestep is a denoising
  target, i.e. the plain DDPM objective `E_t ‖ε − ε_θ(x_t, t)‖²`;
- **sampling** → `impute` with `cond_mask ≡ 0` collapses to pure ancestral
  sampling, no conditioning term.

Training and generation therefore see the identical input distribution. The
architecture, diffusion process and hyperparameters are the paper's
`is_unconditional` variant, unchanged.

---

## Reproducing

```bash
cd /home/tbasseras/benchmark/results/trueexperiment/CSDI/code
V=/home/tbasseras/benchmark/dataset/TrueDataset/variants/om_2022-07_N6144
R=/home/tbasseras/benchmark/results/trueexperiment/CSDI
PY=/home/tbasseras/gpu-venv/bin/python

# 0. smoke-test first -- --tag skips canonical weights, so a probe can never
#    overwrite a real seed (guideline 13.2)
CUDA_VISIBLE_DEVICES=1 $PY train_true.py --seed 0 --epochs 2 --frac 0.05 \
    --gen-num 64 --tag probe --data-dir $V --seq-tag 6144x128x8

# 1. train, two seeds at a time, max 2 GPUs, detached
setsid nohup env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 taskset -c 0-7 \
    $PY train_true.py --seed 0 --data-dir $V --seq-tag 6144x128x8 --val-n 256 \
    > $R/logs/train_seed0.log 2>&1 < /dev/null & disown
# ... repeat for seeds 1, 2, 3 on the other free GPU

# 2. the 8192-path CRPS pool, from the SAME checkpoints
for S in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=1 $PY generate_bank_true.py --seed $S --m-simu 8192 \
      --data-dir $V --seq-tag 6144x128x8 --results-dir $R --out-root $R/crps_banks
done

# 3. the gate. Non-zero exit here means stop, not "investigate later".
$PY collect_artifacts.py --results-dir $R

# 4. diagnostics and figures
$PY measure_memorisation.py
$PY plot_losses.py
$PY plot_diagnostics_true.py --data-dir $V --seq-tag 6144x128x8 --seed 0 --asset 0
```
