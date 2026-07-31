# Guideline — Blinded Synthetic Path-Law Benchmark (Experiments A & B)

Single source of truth for **Experiment A** (Delayed Drawdown Memory) and
**Experiment B** (Balanced Heston Parameter Mixture).

This file covers three things:

1. **What the two data-generating processes are** and exactly how their datasets were built (§2, §3).
2. **The methodology we followed** — the mandatory reproduction gate first, then the step-by-step
   run of each experiment (§5). *Read this before running anything.*
3. **What a new method must produce** — file layout, JSON/CSV schemas, README specification,
   and a copy-paste checklist (§6, §7, §8).

The root [`GUIDELINE.md`](../../GUIDELINE.md) governs the *original* 8192×128 Heston benchmark.
This file governs the blinded protocol only. Where the two disagree, this file wins **inside
`dataset/Heston/new_experiments/` and `results/new_experiments/`**, and the root file wins everywhere else.

---

## 0. Provenance and Authority

The protocol is defined by `synthetic_benchmark_protocol_drawdown_heston_mixture.pdf`
(31 July 2026). §6 of the PDF names the canonical upstream repository
`/home/samer/scenarios/deep-mkv-gen-path-dt` and pins six authoritative files by SHA-256.

Those six files are **vendored verbatim** into this repository at:

```
dataset/Heston/new_experiments/protocol/experiments/
├── path_dt_experiments/
│   ├── __init__.py                                  (empty — see §10)
│   └── heston_mixture.py                            Experiment B DGP
└── scripts/
    ├── generate_drawdown_memory_dataset.py          Experiment A generator
    ├── generate_heston_parameter_mixture_dataset.py Experiment B generator
    ├── fit_heston_mixture_oracle.py                 Experiment B oracle + gate
    ├── evaluate_drawdown_memory.py                  Experiment A evaluator
    └── evaluate_heston_parameter_mixture.py         Experiment B evaluator
```

All six were SHA-256 verified against the PDF (6/6 match) at vendoring time.

> ### ⛔ Never edit anything under `protocol/`
> PDF §7 checklist item 7 requires: *"The supplied evaluator scripts were run unchanged."*
> If a script does not do what you need, **wrap it, do not patch it**. The wrapper pattern
> used throughout this benchmark is monkey-patching module constants and loader functions
> after import — see `compute_metrics_experiment.py` in §6.3.
>
> A single edit to `protocol/` invalidates every number in both experiment READMEs.

---

## 1. What the Two Experiments Test

| | **Experiment A** | **Experiment B** |
|---|---|---|
| Name | Delayed Drawdown Memory | Balanced Heston Parameter Mixture |
| DGP | Latching-memory local-vol SDE | 8-component Heston mixture |
| Question | Does the model retain information from **early history** that is no longer visible in recent observables? | Does the model reproduce a **balanced mixture over 8 parameter regimes**, or collapse to the mixture mean? |
| Headline metric | `early_history_incremental_r2` | `regime_proportion_tvd` (paired with `std_ratio`) |
| Failure mode it catches | Markovian / short-window models | Mode collapse to average dynamics |
| Side information (evaluator-only) | `*_sigma.npy` (true volatility) | `*_labels.npy` (regime index 0-7) + trained oracle |
| Degenerate baselines shipped | — | `mean_heston.npy`, `whole_path_bootstrap.npy` |

Both use **8192 paths × 128 steps**, price levels, `float64`, `S0 = 100`.

The two experiments are **independent**. A model can pass A and fail B, or vice versa.
A model that passes B but fails A is Markovian-but-diverse; one that passes A but fails B
has memory but collapses modes.

---

## 2. Experiment A — Delayed Drawdown Memory

### 2.1 The data-generating process

Per path, per step `t`:

```
drawdown_t  = 1 - S_t / max(S_0..S_t)
hit_t       = 1 if drawdown_t >= drawdown_threshold else 0
memory_t    = max(decay * memory_{t-1}, hit_t)          # LATCHING, then decaying
              decay = exp(-ln 2 / memory_half_life)
sigma_t     = sigma_low + (sigma_high - sigma_low) * memory_{t - response_delay}
S_{t+1}     = S_t * exp(-0.5 * sigma_t^2 * dt + sigma_t * sqrt(dt) * Z_t)
```

The three properties that make this hard:

1. **Latching** — `memory` jumps to 1 on any drawdown ≥ 4 % and then decays with a 60-step
   half-life. It is not a function of the *current* drawdown.
2. **Delay** — `sigma_t` reads `memory` from **32 steps ago**. By the time volatility responds,
   the triggering drawdown may be invisible in the recent window.
3. **Half-life > window** — 60-step half-life vs 20-step `recent_window`. A drawdown at step 10
   still drives volatility at step 90, long after it has left every short-window statistic.

Consequently `future_rv` depends on `early_hit` (a drawdown in steps `0..history_cutoff`)
**even after conditioning on** `recent_rv`, `current_drawdown` and `current_log_return`.
That residual dependence is what the headline metric measures.

### 2.2 Frozen configuration

Verbatim from `experiment_A/manifest.json → configuration`. **Do not change any value.**

| Key | Value | Role |
|---|---|---|
| `name` | `delayed_drawdown_memory_volatility` | |
| `num_paths` | `8192` | per split |
| `sequence_length` | `128` | |
| `dt` | `0.004` (= 1/250) | |
| `s0` | `100.0` | |
| `sigma_low` | `0.12` | vol floor |
| `sigma_high` | `0.45` | vol ceiling |
| `drawdown_threshold` | `0.04` | latch trigger |
| `memory_half_life` | `60.0` | decay: `exp(-ln2/60)` |
| `response_delay` | `32` | steps between memory and vol |
| `history_cutoff` | `40` | `early_hit` window = steps `0..40` |
| `recent_window` | `20` | `recent_rv` window |
| `future_horizon` | `32` | `future_rv` window |
| `split_index` | `64` | past/future boundary |
| `seeds` | `{train: 0, test: 1, disc: 2}` | |

### 2.3 Regenerating the datasets (idempotent)

```bash
cd /home/tbasseras/benchmark/dataset/Heston/new_experiments
python protocol/experiments/scripts/generate_drawdown_memory_dataset.py \
    --output-dir experiment_A
```

All non-`--output-dir` flags default to the frozen values above; pass none of them.
Produces `train.npy`, `test.npy`, `disc.npy`, the three `*_sigma.npy` companions, and
`manifest.json`. Deterministic given the seeds — rerunning overwrites with identical bytes
(modulo the ULP caveat in §10).

### 2.4 Evaluator

```bash
python protocol/experiments/scripts/evaluate_drawdown_memory.py \
    --train-data      experiment_A/train.npy \
    --test-data       experiment_A/test.npy \
    --generated-data  <.../generated_paths/seed_N/generated_paths_8192x128.npy> \
    --dataset-manifest experiment_A/manifest.json \
    --output          <.../pdf_metrics/seed_N_drawdown_memory.json>
```

Output JSON top-level keys: `configuration`, `target_memory`, `generated_memory`, `errors`,
`novelty`, `sources`.

**`target_memory` / `generated_memory`** (the same 9 keys each, computed on `test.npy` and on
the generated bank respectively). Measured values on `test.npy` are given as the reference target:

| Key | Meaning | Target |
|---|---|---|
| `early_hit_rate` | fraction of paths with a ≥4 % drawdown in steps `0..40` | 0.629639 |
| `future_rv_no_hit_mean` | mean future realised vol among non-hit paths | 0.202984 |
| `future_rv_hit_mean` | mean future realised vol among hit paths | 0.426855 |
| `future_rv_hit_gap` | `hit_mean − no_hit_mean` — the raw memory signal | 0.223872 |
| `early_hit_future_rv_correlation` | correlation of `early_hit` with `future_rv` | 0.808360 |
| `baseline_r2` | R² of `future_rv ~ 1 + recent_rv + current_drawdown + current_log_return` | 0.385467 |
| `augmented_r2` | R² of the same **+ `early_hit`** | 0.673990 |
| `early_history_incremental_r2` | `augmented_r2 − baseline_r2` ← **the headline number** | **0.288523** |
| `early_hit_standardized_coefficient` | the `early_hit` coefficient in the standardized augmented fit | 1.498698 |

Note the evaluator's own `configuration` block renames three manifest fields:
`history_cutoff`→`history_cutoff`, `future_horizon`→`horizon`, `split_index`→`split`,
`drawdown_threshold`→`threshold`, plus `annualization: 250.0`.

**`errors`** — 9 absolute-difference keys, all *lower is better*, all target 0:

```
return_std_error                        excess_kurtosis_error
abs_return_acf_rmse_lags_1_50           squared_return_acf_rmse_lags_1_50
terminal_log_price_ks                   future_rv_wasserstein
early_hit_rate_error                    future_rv_hit_gap_error
early_history_incremental_r2_error
```

**`novelty`** — exact brute-force nearest-training-path search (faiss `IndexFlatL2`) in the
127-dim standardized log-return space:
`median_standardized_nearest_train_path_rmse`, `mean_...`, `distinct_nearest_training_paths`.
A memorising model shows near-zero RMSE and very few distinct neighbours.

### 2.5 How to read the headline metric

`early_history_incremental_r2` is a **two-sided** target: match the real value, do not maximise it.

- Real (`test.npy`) sets the target. A model at ~0 has **no memory** — it is Markovian in the
  observables and has failed the experiment's purpose.
- A model well *above* the target has invented a spurious dependence.
- `early_history_incremental_r2_error` (in `errors`) is the |difference| and is what goes in the table.

Always report it **next to** `future_rv_hit_gap_error`. A model can match the incremental R²
by accident while getting the magnitude of the volatility response wrong.

---

## 3. Experiment B — Balanced Heston Parameter Mixture

### 3.1 The data-generating process

8 equally-weighted regimes, the full Cartesian product of:

| Parameter | Low | High | Label bit |
|---|---|---|---|
| `theta` (long-run variance) | `0.01` | `0.09` | `(l >> 2) & 1` |
| `xi` (vol-of-vol) | `0.35` | `1.10` | `(l >> 1) & 1` |
| `rho` (correlation) | `-0.99` | `+0.99` | `l & 1` |

Shared constants: `kappa = 3`, `mu = 0`, `dt = 1/250`, `s0 = 100`, `sequence_length = 128`,
`v0 = theta` per regime.

Variance is integrated with **full-truncation Euler** (`v` clipped at 0 in the drift and in
the diffusion `sqrt`, the state itself allowed to go slightly negative before truncation) —
this matters if you reimplement, and you should not reimplement.

**Balanced labels.** `balanced_labels = arange(N) % 8`, then shuffled with `seed + 10_000`.
At `N = 8192` this yields **exactly 1024 paths per regime**, every split. Confirmed in
`manifest.json → splits.*.regime_counts` = `[1024]*8` for `train`/`test`/`disc`,
`[4096]*8` for `oracle_train` (32768 paths).

### 3.2 Splits

| File | Seed | Shape | Visible to generator? |
|---|---|---|---|
| `train.npy` | 0 | (8192, 128) | ✅ **yes — the only fittable file** |
| `disc.npy` | 2 | (8192, 128) | ✅ yes (validation / model selection, unlabelled) |
| `test.npy` | 1 | (8192, 128) | ❌ evaluator only, one frozen final eval |
| `train_labels.npy` / `test_labels.npy` / `disc_labels.npy` | — | (8192,) int | ❌ evaluator only |
| `oracle_train.npy` + `_labels` | 100 | (32768, 128) | ❌ evaluator only |
| `oracle_validation.npy` + `_labels` | 101 | (8192, 128) | ❌ evaluator only |
| `mean_heston.npy` | — | (8192, 128) | reference baseline |
| `whole_path_bootstrap.npy` | — | (8192, 128) | reference baseline |

The two baselines are **degenerate references, not competitors**:

- `mean_heston.npy` — a single Heston with the mixture-mean parameters. Its observable
  statistics look plausible, but its `regime_proportion_tvd` is terrible. It is the
  *canonical mode-collapse failure*, and any model near it has collapsed.
- `whole_path_bootstrap.npy` — resampled whole training paths. Perfect on every marginal
  statistic and perfect on TVD, and **zero novelty**. It is the *canonical memorisation
  failure*. It exists so that nobody mistakes a good TVD for a good model without also
  checking `novelty`.

### 3.3 The oracle and its gate

Experiment B cannot be scored without a regime classifier, and the classifier must use only
what a generated bank actually contains: **price paths, no latent variance, no labels.**

```bash
python protocol/experiments/scripts/fit_heston_mixture_oracle.py \
    --data-dir   experiment_B \
    --output-dir experiment_B/oracle \
    --workers    16          # ← override the default 24, see §9
```

`ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, max_features=0.7,
class_weight="balanced", random_state=42)` over **77 observable price-path features**,
trained on `oracle_train` (32768) and gated on `oracle_validation` (8192).

**Gate: 8-regime validation accuracy ≥ 0.90.** Achieved:

| Quantity | Value |
|---|---|
| `eight_regime_accuracy` | **0.909423828125** ✅ |
| `parameter_state_accuracy.theta` | 0.98291015625 |
| `parameter_state_accuracy.rho` | 0.95654296875 |
| `parameter_state_accuracy.xi` | 0.95556640625 |
| `num_features` | 77 |

Artefacts: `experiment_B/oracle/oracle.joblib` (~554 MB, **gitignored** — see §10) and
`gate_report.json` (committed). If the gate fails, **stop**: no Experiment B number is valid.

### 3.4 Evaluator

```bash
python protocol/experiments/scripts/evaluate_heston_parameter_mixture.py \
    --train-data         experiment_B/train.npy \
    --test-data          experiment_B/test.npy \
    --generated-data     <.../generated_paths/seed_N/generated_paths_8192x128.npy> \
    --oracle             experiment_B/oracle/oracle.joblib \
    --oracle-gate-report experiment_B/oracle/gate_report.json \
    --output             <.../pdf_metrics/seed_N_heston_mixture.json>
```

Output top-level keys: `mixture_fidelity`, `observable_fidelity`, `novelty`, `oracle_gate`,
`sources`.

**`mixture_fidelity`**

| Key | Meaning |
|---|---|
| `regime_proportion_tvd` | **headline.** ½·Σ\|p_gen − p_target\| over the 8 oracle-predicted proportions. 0 = perfectly balanced |
| `target_/generated_regime_proportions` | the two 8-vectors |
| `target_/generated_mean_posterior_entropy` | oracle uncertainty. Generated ≫ target ⇒ off-manifold paths the oracle cannot place |
| `target_/generated_mean_max_probability` | companion to the above |
| `target_/generated_low_confidence_fraction` | fraction with max posterior < 0.6 |
| `parameters` | per-parameter (`theta`/`xi`/`rho`) `wasserstein`, `support_normalized_wasserstein`, `mean_error`, `q05_error`, `q95_error`, **`std_ratio`** |

> ### TVD is measured against the *oracle's* proportions on `test.npy`, not against uniform 1/8
> The oracle is 90.9 % accurate, so its predicted proportions on `test.npy` are **not** exactly
> `[0.125]*8` — measured they are
> `[0.1232, 0.1229, 0.1196, 0.1204, 0.1331, 0.1302, 0.1243, 0.1263]`.
> `regime_proportion_tvd` compares generated-vs-target **through the same imperfect classifier**,
> so the oracle's own bias cancels. Never compute TVD against `1/8` by hand — you would be
> charging the model for the oracle's error.

**`observable_fidelity`** — `return_std_error`, `excess_kurtosis_error`, ACF RMSEs, leverage
curve, `realized_volatility_wasserstein`, `terminal_log_price_ks`.

**`novelty`** — same three keys as Experiment A.

### 3.5 The TVD × std_ratio diagnostic pair — read them together

Neither number alone identifies mode collapse:

| `regime_proportion_tvd` | `std_ratio` | Verdict |
|---|---|---|
| low | ≈ 1 | ✅ genuinely covers all 8 regimes |
| low | ≪ 1 | ⚠️ **collapsed** — proportions match by luck, per-path parameter spread is gone |
| high | ≈ 1 | diverse but mis-weighted |
| high | ≪ 1 | ❌ full collapse toward `mean_heston` |

`std_ratio` is `std(generated parameter samples) / std(target parameter samples)`, per
parameter. A model that emits 8192 near-identical average paths can still get its oracle
proportions roughly right by sitting on a decision boundary. `std_ratio ≪ 1` exposes it.

And a low TVD with near-zero `novelty` RMSE is `whole_path_bootstrap`, not a model.
**Always report TVD, std_ratio and novelty in the same table.**

---

## 4. The Information Firewall

This is the core of the protocol. Violating it silently invalidates everything.

| File | Fit on it | Select on it | Score on it |
|---|---|---|---|
| `train.npy` | ✅ | ✅ | ❌ |
| `disc.npy` | ❌ | ✅ (validation, early stopping, HP choice) | ❌ |
| `test.npy` | ❌ | ❌ | ✅ **once, frozen** |
| `*_sigma.npy`, `*_labels.npy` | ❌ | ❌ | evaluator only |
| `oracle_*`, `oracle.joblib`, `gate_report.json` | ❌ | ❌ | evaluator only |
| `perfect_floor/*` | ❌ | ❌ | evaluator only |

### Practical enforcement

Every training script **must** carry this guard (copy verbatim):

```python
data_path = os.path.abspath(a.data)
if os.path.basename(data_path) != "train.npy":
    raise SystemExit(f"firewall: generators may only read train.npy, got {data_path}")
```

Additional rules:

- Training scripts take `--data` as a **required** argument. No hard-coded default that could
  silently point at the wrong split.
- Scaler statistics (`mu`, `sigma`, min/max, …) are computed on `train.npy` **only** and stored
  in `metadata.json` / `seed_N_config.json` so the choice is auditable.
- The evaluator and the trainer live in the same `code/` directory but are **separate files**.
  Never import the evaluator from the trainer.
- `test.npy` is opened exactly once per model, at Stage 3, by the metric scripts.

If you tune anything against `test.npy` — even once, even by looking at a plot — the run is
void and must be redone from a fresh model.

### The perfect floor

Every metric has a non-zero floor because both sides are finite samples. The floor answers
*"how well can anything score?"*

```bash
python make_perfect_floor.py --experiment A   # or B
```

Five independent true-DGP draws, seeds **1000–1004** (repo convention from
`metrics/compute_perfect_recovery.py`; disjoint from every protocol seed 0/1/2/100/101).
Written to `dataset/.../experiment_X/perfect_floor/`, scored against `test.npy`
**through the exact same code path as a model's bank** (`--source floor`).

A model is "at the floor" on a metric when its 95 % CI overlaps the floor's. Read every
number against the floor column, never against 0.

### Measured floor values (computed, 5 draws, seeds 1000–1004)

These are the numbers to beat-or-match. Anything materially above the floor's CI upper bound
is a real deficiency, not sampling noise.

**Experiment A** — headline rows:

| Metric | Floor mean ± std | 95 % CI half-width |
|---|---|---|
| `errors.early_history_incremental_r2_error` | 0.006412 ± 0.00544 | 0.00675 |
| `errors.future_rv_hit_gap_error` | 0.001023 ± 0.000516 | 0.00064 |
| `errors.early_hit_rate_error` | 0.003101 ± 0.00224 | 0.00279 |
| `errors.return_std_error` | 2.796e-05 ± 1.44e-05 | 1.79e-05 |
| `errors.excess_kurtosis_error` | 0.023094 ± 0.0118 | 0.0147 |
| `errors.terminal_log_price_ks` | 0.012012 ± 0.00319 | 0.00396 |
| `errors.future_rv_wasserstein` | 0.001233 ± 0.00021 | 0.00026 |
| `errors.abs_return_acf_rmse_lags_1_50` | 0.001561 ± 0.000141 | 0.000175 |
| `errors.squared_return_acf_rmse_lags_1_50` | 0.001616 ± 0.000193 | 0.000239 |
| `novelty.median_standardized_nearest_train_path_rmse` | 0.989367 ± 0.00106 | 0.00132 |
| `novelty.distinct_nearest_training_paths` | 1600.8 ± 11.5 | 14.2 |

Floor `generated_memory.early_history_incremental_r2` = 0.29328 ± 0.00726, against the
`test.npy` target of 0.288523.

**Experiment B** — headline rows:

| Metric | Floor mean ± std | 95 % CI half-width |
|---|---|---|
| `mixture_fidelity.regime_proportion_tvd` | 0.008472 ± 0.00129 | 0.00161 |
| `parameters.theta.std_ratio` | 0.999393 ± 0.00205 | 0.00254 |
| `parameters.xi.std_ratio` | 1.001830 ± 0.000666 | 0.000827 |
| `parameters.rho.std_ratio` | 0.998133 ± 0.00167 | 0.00207 |
| `novelty.median_standardized_nearest_train_path_rmse` | 0.722292 (draw 0) | — |
| `novelty.distinct_nearest_training_paths` | 2941.8 ± 24.5 | 30.4 |

Note the **novelty floor differs between the two experiments** (A ≈ 0.989, B ≈ 0.722 median
nearest-train RMSE). It is a property of the DGP's path diversity, not of any model. Compare a
model's novelty only to its own experiment's floor.

### Generating the table

```bash
cd results/new_experiments/tools
python aggregate_pdf_metrics.py \
    --model-dir ../experiment_A/LS4 --floor-dir ../experiment_A/perfect_floor \
    --pattern '*_drawdown_memory.json' --label LS4 \
    --exclude-prefix configuration                        # Experiment B: '*_heston_mixture.json'
```

`aggregate_pdf_metrics.py` flattens every numeric leaf of the evaluator JSONs to a dotted path
and aggregates across seeds, so it needs no per-experiment key list. It emits the Markdown table
for README § 1 directly, and warns in an HTML comment if any key has ≠ 5 seeds. `--exclude-prefix`
drops provenance blocks that are constant by construction — `configuration` for Experiment A,
`configuration oracle_gate` for Experiment B, whose gate belongs in the dataset manifest, not in
a per-seed model table.

**Self-check:** every `target_*` row must show `± 0`. A non-zero std there means the seeds were
scored against *different* reference data — stop and find out why.

### The other five tools

```bash
# README § 2 — the A1–A34 and B curve-shape tables, from metrics_summary.csv
python make_metrics_tables.py --model-dir ../experiment_A/LS4 \
    --floor-dir ../experiment_A/perfect_floor --label LS4 --table A     # then --table B

# PDF §1.4 mandatory per-seed manifest (NOT a README section — a submission artefact)
python write_generation_manifest.py --model-dir ../experiment_A/LS4 \
    --experiment A --source-revision "$(git rev-parse --short HEAD)" --repair none

# README § 1 figure — memory structure (A) / mixture structure (B)
python plot_experiment_figures.py --experiment A \
    --model-dir ../experiment_A/LS4 --seed 0 --out ../experiment_A/LS4/plots

# README § 3 figure — the eight-panel stylised-facts diagnostic
python plot_stylised_facts.py --experiment A \
    --model-dir ../experiment_A/LS4 --seed 0 --label LS4

# README § 4 figure — loss convergence, all seeds overlaid
python plot_losses.py --model-dir ../experiment_A/LS4 --title "LS4 (Experiment A)"
```

`make_metrics_tables.py` is deliberately **separate** from `aggregate_pdf_metrics.py`. The two
suites answer different questions and README § 1 and § 2 must never be blended: § 1 asks *did the
model reproduce the DGP structure the protocol targets*, § 2 asks *how does it score on the
repo's standard battery*. Keeping one tool per section makes it impossible to mix them by
accident. A33/A34 are dropped in both experiments (they need the latent σ path, which is not part
of the generator's output contract), so their cells must read `n/a` — never `0`, which would
silently flatter the method.

`write_generation_manifest.py` is the odd one out: it feeds no README section. It exists because
PDF §1.4 makes `generation_manifest.json` **mandatory** and fixes its contents, and the
`metadata.json` our training script writes covers only about half the required fields — it has no
source revision, no separate generation seed, no preprocessing block, no hyperparameters, and no
numerical-repair field. The tool merges `metadata.json` with `weights/seed_<q>_config.json`, then
**re-measures the §1.4 output contract from the bank itself** (finite, strictly positive, dtype,
shape, `S₀ = 100`) rather than asserting it. A manifest that claims conformance while the array
violates it is worse than no manifest. Run it once per method per experiment, after all five
seeds exist. See §11.5 for the `--repair` flag.

`plot_stylised_facts.py` wraps the benchmark-standard `metrics/plot_diagnostics.py` so every
method gets the identical figure, with **one deliberate difference: the black Heston theory curve
is suppressed.** That curve is the closed-form *single-regime* Heston reference. Experiment A is a
latching drawdown-memory process and Experiment B is an 8-regime mixture — neither is
single-regime Heston, so the curve would be a wrong reference drawn with authority. The
suppression is done by monkeypatching `compute_theory_bundle` to raise, never by editing the
shared canonical module. Expect the line `[warn] theory bundle unavailable ...` in the output;
that warning means it worked.

`plot_experiment_figures.py` draws the one picture that the scalar table cannot: for Experiment A,
`future_rv` split by `early_hit` for real and generated side by side; for Experiment B, the eight
oracle-predicted regime proportions plus the per-parameter `std_ratio` spread.

⚠️ It **re-derives** Experiment A's features rather than importing them — the protocol scripts are
vendored verbatim and expose no importable feature API. The re-derivation must mirror
`evaluate_drawdown_memory.drawdown_features` exactly, and two details are easy to get wrong:

| Quantity | Correct (evaluator) | Plausible but WRONG |
|---|---|---|
| drawdown | `cummax(log S/S₀) − log S/S₀` (log space) | `1 − S/cummax(S)` (price space) |
| realised vol | `sqrt(ann · mean(r²))`, `ann = 1/dt` | `std(r) · sqrt(250)` |

Both wrong forms produce plausible-looking figures with silently shifted numbers. **This is why the
script prints its own hit-gap: it must equal `future_rv_hit_gap` in the evaluator JSON to the
printed precision.** If it does not, the figure is lying — fix the re-derivation, not the figure.

`plot_losses.py` builds one panel per column of `losses/seed_*_losses.csv`, taking the panel list
from the CSV header so any method's loss schema plots. It clips each panel's y-range to the
post-epoch-5 data, because epoch-0 values sit one to two orders of magnitude above convergence and
otherwise compress every later epoch into a flat line.

---

## 5. Methodology — Exactly How This Was Run

> This section documents the **procedure we actually followed**, in order. Follow it for
> every new method. Stage 0 is not optional and is not a formality.

### Stage 0 — The reproduction gate (mandatory, FIRST)

**Before touching Experiment A or B at all**, prove the method's code in this repository
reproduces its already-committed result on the **original 8192×128 Heston bank**.

Rationale: if you go straight to Experiment A and the numbers look bad, you cannot tell
whether the *model* failed or *your wiring* failed — wrong config file, wrong scaler, wrong
checkpoint, wrong generation path, a silently missing EMA. The gate removes that ambiguity
permanently. It costs one training run (~16 min for LS4) and buys certainty for the rest
of the project.

**What we ran** (LS4, one seed, unchanged canonical script):

```bash
cd /home/tbasseras/benchmark/methods/LS4/code
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c 0-7 \
  python -u train_heston.py --seed 0 --epochs 100 --tag repro
```

> ⚠️ Use the method's own non-destructive flag. In `train_heston.py`,
> `tagp = (a.tag + "_") if a.tag else ""` and weights/config are written **only when `--tag`
> is empty** — so `--tag repro` cannot overwrite the canonical artefacts. **Verify the
> equivalent property before running any repro; do not assume it.** If the script has no such
> flag, copy the artefacts aside first.

**The four escalating comparison levels.** Climb until you reach the highest level the method
supports, and record which level you reached:

| Level | Check | Verdict |
|---|---|---|
| 1 — Visual | Diagnostic figures vs the figures in the method's committed README | necessary, never sufficient |
| 2 — Curves | `seed_0_losses.csv` old vs new, max abs diff per column | catches optimiser/schedule drift |
| 3 — Scalars | every `metadata.json` scalar: `min_total_loss`, `generated_mean/std`, `real_mean/std`, `min_val`, `max_val`, `params`, `epochs_run`, `gen_has_nan`, `first_nan_epoch` | catches scaler and generation bugs |
| 4 — Bitwise | `np.array_equal(old_bank, new_bank)` | the gold standard |

**What we obtained for LS4 — level 4, the strongest possible:**

```
np.array_equal(canonical, repro) = True      max abs diff = 0.0
losses: max abs difference over all 100 epochs × 5 columns = 0.0
metadata: every scalar EXACT
  min_total_loss   -1.068142032250762      params        2146857
  generated_mean  101.56282806396484       epochs_run    100
  generated_std     9.770788192749023      gen_has_nan   False
  real_mean       101.32547381502401       first_nan_epoch None
  real_std          9.971659995159825
  min_val          47.33000564575195        max_val      144.9393310546875
```

> ### The figures must match in their **defects**, not just their fits
> LS4's committed diagnostics have known, specific shortfalls on Heston: the ACF of |r| and of
> r² sit **below** the real curve, and the tail-survival curve **undershoots**. The repro
> reproduced those same defects.
>
> This is the point of the gate. Anyone can produce a plot where the orange curve roughly
> follows the blue one. Only the correct code, correctly wired, reproduces the *exact same
> failures*. If your repro looks **better** than the committed README, you have changed
> something — find out what before proceeding.

**Gate outcome must be recorded** (which level, the numbers) and **explicitly validated by
Theo** before Stage 1. Only after validation do the Experiment A/B directories get created.

Keep the `repro_*` artefacts until the experiments are committed; they are the evidence.

### Stage 1 — Datasets and evaluator-side assets

One-time per experiment, shared by every method. Skip if already present.

1. Vendor `protocol/` and verify SHA-256 against PDF §6 (6/6).
2. Generate datasets (§2.3, §3.1) → `train/test/disc` + side info + `manifest.json`.
3. **Experiment B only:** fit the oracle and check the gate ≥ 0.90 (§3.3). Stop on failure.
4. Generate the perfect floor, 5 draws, seeds 1000–1004 (§4).
5. Sanity-check the manifest: shapes `(8192, 128)`, `regime_counts == [1024]*8` for B,
   price ranges finite and plausible.

### Stage 2 — Train, 5 seeds, 2 GPUs

Seeds `q ∈ {0,1,2,3,4}`, split across exactly **two** lanes (§9 hard limits).
LS4 takes ~16 min/seed, so 3 waves ≈ 50 min wall clock.

```bash
cd /home/tbasseras/benchmark/results/new_experiments/experiment_A/LS4/code
mkdir -p logs
DATA=/home/tbasseras/benchmark/dataset/Heston/new_experiments/experiment_A/train.npy
PY=/home/tbasseras/gpu-venv/bin/python

# Lane 1 — GPU 2, cores 0-7,  seeds 0 2 4
setsid bash -c "cd $PWD; for s in 0 2 4; do \
  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c 0-7 \
  $PY -u train_ls4_experiment.py --seed \$s --experiment A --data $DATA \
  > logs/expA_seed\$s.log 2>&1; done" < /dev/null > /dev/null 2>&1 & disown

# Lane 2 — GPU 3, cores 8-15, seeds 1 3
setsid bash -c "cd $PWD; for s in 1 3; do \
  CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c 8-15 \
  $PY -u train_ls4_experiment.py --seed \$s --experiment A --data $DATA \
  > logs/expA_seed\$s.log 2>&1; done" < /dev/null > /dev/null 2>&1 & disown
```

> ### Two shell traps that will cost you a GPU-hour
>
> **Trap 1 — `run_in_background` reaps long jobs.** Multi-hour chains **must** use
> `setsid ... & disown`. A previous incident lost 50 minutes of idle GPU to a reaped job.
> Corollary: `$!` after `setsid` is the *wrapper's* PID, not the trainer's — find the real
> one with `pgrep -af train_ls4_experiment.py`.
>
> **Trap 2 — `&` backgrounds the whole `&&` chain.** Writing
> `cd X && ... && setsid A & disown; setsid B > logs/b.log` puts the `cd` inside the
> subshell, so `B`'s **relative** redirect resolves against the *original* cwd and dies with
> `logs/b.log: No such file or directory`. Fix: put an explicit `cd /abs/path;` **inside**
> each `setsid bash -c '...'`, as above. This actually happened to lane 2.

**Verify within 60 s of launch** — a silently-dead lane is the expensive failure:

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
pgrep -af train_ls4_experiment.py            # expect 2 python PIDs
tail -6 logs/expA_seed0.log logs/expA_seed1.log
```

Both logs must show the **correct** data path and a plausible `[data]` line:

```
=== LS4 experiment_A  seed=0  CUDA_VISIBLE_DEVICES=2  device=NVIDIA A100-SXM4-80GB ===
[data] /home/.../new_experiments/experiment_A/train.npy
[data] S(8192, 128) price[min=39.39,max=272.99]  mu=100.1362 sigma=11.9280 ... epochs=100
[model] params=2146857  z_dim=5 n_layers=4 s4_type=s4
```

If the `[data]` line points anywhere but `experiment_X/train.npy`, **kill both lanes** — the
firewall has been breached and the run is void.

### Stage 3 — Metrics

Three separate scoring passes, all reading `test.npy` for the first and only time.

```bash
cd .../experiment_A/LS4/code

# 3a — benchmark A/B suite on the model bank
CUDA_VISIBLE_DEVICES=2 python compute_metrics_experiment.py \
    --experiment A --source model --seeds 5

# 3b — the same code path on the perfect floor
CUDA_VISIBLE_DEVICES=2 python compute_metrics_experiment.py \
    --experiment A --source floor --seeds 5

# 3c — the PDF's own evaluator, per seed, unchanged (§2.4 / §3.4)
for s in 0 1 2 3 4; do
  python .../protocol/experiments/scripts/evaluate_drawdown_memory.py \
      --train-data ... --test-data ... --dataset-manifest ... \
      --generated-data ../generated_paths/seed_$s/generated_paths_8192x128.npy \
      --output ../pdf_metrics/seed_${s}_drawdown_memory.json
done
```

3a/3b write `seed_N_metrics.json` + `metrics_summary.csv`. 3c writes the PDF JSONs, which are
then aggregated (mean ± std across the 5 seeds) into the README's **first** section.

**A33/A34 are `null` in both experiments** — both require the Heston teacher variance `v`,
which neither DGP defines. `load_data()` returns `v = None` and `compute_all`'s existing
try/except records them as `null`. **No edit to `metrics/compute_all.py`.** Report them as
`n/a` with that one-line reason; do not delete the rows.

### Stage 4 — Figures

Three figures are produced here; the metric battery of Stage 3a emits ten more (`seed_<q>_pca.png`,
`seed_<q>_tsne.png`) on its own. **The authoritative inventory, with exact filenames, generators
and which section displays each one, is §7.7 — check the names against it, and against `ls`,
before writing them into a README.**

```bash
python ../../tools/plot_losses.py            --model-dir . --out losses
python ../../tools/plot_stylised_facts.py    --model-dir . --experiment <X> --out plots
python ../../tools/plot_experiment_figures.py --experiment <X> --model-dir . --seed 0 --out plots --label <Method>
```

- `losses/loss_convergence.png` — 5 seeds overlaid.
- `plots/heston_diagnostics.png` — from `plot_stylised_facts.py`, which wraps the
  benchmark-standard `metrics/plot_diagnostics.py`. That wrapped script's Heston-theory third
  curve is **wrong for both of these DGPs** (neither is single-regime Heston); it is drawn inside
  a try/except and falls back to `TB = None`. The wrapper suppresses it — **say so in the README
  caption**, or a reader will think the curve was forgotten.
- Experiment-specific, from `plot_experiment_figures.py`:
  A → `plots/memory_structure_seed0.png`, `future_rv` distributions split by `early_hit`,
  real vs generated.
  B → `plots/mixture_structure_seed0.png`, the two 8-bar regime-proportion histograms
  (target vs generated) plus the three parameter marginals.

Cross-check at least one number printed inside each figure against the evaluator JSON before
publishing it (§7.7). For B seed 0 the figure prints TVD = 0.245483 and
`pdf_metrics/seed_0_heston_mixture.json → mixture_fidelity.regime_proportion_tvd` is 0.245483.
A figure never reconciled with the tables is decoration.

### Stage 5 — README, then commit

Write `README.md` per §7, in the mandated section order, then:

```bash
git add results/new_experiments/experiment_A/LS4 dataset/Heston/new_experiments
git commit -m "Experiment A: LS4 5 seeds, metrics, floor, README"
git push
```

Banks are `.npy` and follow the repo-wide gitignore policy on large artefacts (§10).

### Stage 0 for Experiment B specifically

The reproduction gate is a statement about **the method's code**, not about the dataset.
It is passed **once per method**. Having cleared it for LS4 in Experiment A, Experiment B
starts directly at Stage 1 — but LS4 still trains **5 fresh seeds on Experiment B's own
`train.npy`**. Nothing is reused from A except the validated code.

---

## 6. File Structure

### 6.1 Dataset side (shared, one-time)

```
dataset/Heston/new_experiments/
├── make_perfect_floor.py                floor generator, both experiments
├── logs/                                fit_oracle.log, floor_A.log, floor_B.log
├── protocol/experiments/                ⛔ VENDORED VERBATIM — NEVER EDIT
│   ├── path_dt_experiments/{__init__,heston_mixture}.py
│   └── scripts/{generate,evaluate,fit}_*.py
├── experiment_A/
│   ├── train.npy  test.npy  disc.npy               (8192,128) float64
│   ├── train_sigma.npy  test_sigma.npy  disc_sigma.npy   ❌ evaluator only
│   ├── manifest.json
│   └── perfect_floor/
│       ├── floor_seed{1000..1004}.npy  (+ _sigma.npy)
│       └── perfect_floor_manifest.json
└── experiment_B/
    ├── train.npy  test.npy  disc.npy
    ├── train_labels.npy  test_labels.npy  disc_labels.npy    ❌ evaluator only
    ├── oracle_train.npy (32768,128) + _labels                ❌ evaluator only
    ├── oracle_validation.npy (8192,128) + _labels            ❌ evaluator only
    ├── mean_heston.npy  whole_path_bootstrap.npy             degenerate baselines
    ├── manifest.json
    ├── oracle/{oracle.joblib (gitignored), gate_report.json}
    └── perfect_floor/floor_seed{1000..1004}.npy (+ _labels) + manifest
```

### 6.2 Results side (per method, exact layout)

Mirrors `results/Heston/preprocessing_with_log_returns/<Method>/`
**minus `baseline_no_preproc/` and `path_shadowing/`** — neither applies here.

```
results/new_experiments/
├── guideline_new_experiment.md                  ← this file
├── README_TEMPLATE.md                           literal skeleton to copy for a new method (§7.1)
├── tools/                                       method-neutral, both experiments (§4)
│   ├── aggregate_pdf_metrics.py                 README § 1 table: 5-seed mean ± std + 95% CI
│   ├── make_metrics_tables.py                   README § 2 tables: A1–A34 and B curve-shape
│   ├── plot_experiment_figures.py               README § 1 figure: memory (A) / mixture (B)
│   ├── plot_stylised_facts.py                   README § 3 figure: heston_diagnostics.png
│   ├── plot_losses.py                           README § 4 figure: loss_convergence.png
│   ├── write_generation_manifest.py             PDF §1.4 artefact, no README section
│   └── check_method_layout.py                   §6.6 — verifies a method dir matches this tree
└── experiment_<A|B>/
    ├── perfect_floor/                           method-neutral sibling
    │   ├── pdf_metrics/seed_{0..4}_*.json       floor scored by the PDF evaluator vs test.npy
    │   ├── pdf_metrics_validation/seed_{0..4}_*.json   same, vs disc.npy (§11.4)
    │   ├── seed_{0..4}_metrics.json             floor scored by the A/B suite
    │   ├── metrics_summary.csv
    │   └── plots/
    └── <Method>/
        ├── README.md                            ← §7
        ├── code/
        │   ├── train_<method>_experiment.py     firewall guard mandatory
        │   ├── compute_metrics_experiment.py    the §6.3 adapter
        │   └── logs/exp<A|B>_seed{0..4}.log
        ├── generated_paths/seed_{0..4}/
        │   ├── generated_paths_8192x128.npy     (8192,128) float64, ORIGINAL price scale
        │   ├── metadata.json                    raw run record written by the trainer
        │   └── generation_manifest.json         PDF §1.4 MANDATORY — see §11.3 item 6
        ├── weights/
        │   ├── seed_N_model.pt
        │   └── seed_N_config.json
        ├── losses/
        │   ├── seed_N_losses.csv
        │   └── loss_convergence.png
        ├── pdf_metrics/                         scored vs test.npy   → "metrics_test"
        │   └── seed_N_<drawdown_memory|heston_mixture>.json
        ├── pdf_metrics_validation/              scored vs disc.npy  → "metrics_validation"
        │   └── seed_N_<drawdown_memory|heston_mixture>.json
        ├── metrics_summary.csv
        ├── seed_{0..4}_metrics.json
        └── plots/
```

### 6.3 The metrics adapter (copy this pattern)

`metrics/compute_all.py` is canonical and must not be edited. Wrap it: override its path
constants and its three loaders after import, then call `C.main()`.

```python
sys.argv = [sys.argv[0]]        # hide our flags from compute_all's import-time argparse
sys.path.insert(0, METRICS_DIR)
import compute_all as C

C.DATASET_DIR   = DATA_DIR
C.N_SEEDS       = args.seeds
C.GENERATED_DIR = ...           # model bank, or perfect_floor/ when --source floor
C.RESULTS_DIR   = ...
C.PLOTS_DIR     = os.path.join(C.RESULTS_DIR, "plots")

def load_data():                # v = None -> A33/A34 land as null via existing try/except
    return np.load(os.path.join(DATA_DIR, "test.npy")), None
def load_disc():                # A18/A19 judge "real" class (protocol seed 2)
    return np.load(os.path.join(DATA_DIR, "disc.npy"))
def load_generated(seed): ...

C.load_data = load_data; C.load_disc = load_disc; C.load_generated = load_generated
C.main()
```

The **same** adapter scores the floor via `--source floor`, which is what makes the floor
column comparable — the PDF requires the floor to go through the identical code path.

### 6.4 Required schemas

**`generated_paths/seed_N/metadata.json`** (synthetic values shown; date format `%Y-%m-%d`):

```json
{
  "method": "LS4", "experiment": "A", "seed": 0,
  "shape": [8192, 128],
  "min_val": 47.33, "max_val": 144.93,
  "generated_mean": 101.56, "generated_std": 9.77,
  "real_mean": 101.32, "real_std": 9.97,
  "gen_sec": 12.3, "train_time_sec": 973.4,
  "gpu": "A100-SXM4-80GB", "date": "2026-07-31",
  "params": 2146857, "epochs_run": 100, "epochs_max": 100,
  "min_total_loss": -1.068142, "first_nan_epoch": null, "gen_has_nan": false
}
```

`first_nan_epoch` and `gen_has_nan` are **mandatory** — they are how a silently-diverged seed
is caught before it poisons a mean.

**`weights/seed_N_config.json`** — every hyperparameter, plus `scaler`, `scaler_mu`,
`scaler_sigma`, `data` (absolute path, so the firewall is auditable), `params`.

**`losses/seed_N_losses.csv`** — one row per epoch, first column `epoch`, then method-specific
loss columns, last column `lr`.

**`metrics_summary.csv`** — header exactly `metric,mean,std,seed_0,seed_1,seed_2,seed_3,seed_4`.

### 6.5 The generator contract

Everything downstream of training — the six tools in §4, the adapter in §6.3, the README
spec in §7 — is method-neutral. It works for LS4 and it will work for CSDI, TimeGAN,
DiffusionTS or anything else **provided the trainer satisfies this contract.** Port the
method by satisfying the contract; do not adapt the tools to the method.

**Required CLI.** The trainer must be `code/train_<method>_experiment.py` and accept exactly:

| Flag | Required | Meaning |
|---|---|---|
| `--data` | **yes** | Absolute path to `dataset/Heston/new_experiments/experiment_<X>/`. Never defaulted — a default is how a run silently trains on the wrong experiment. |
| `--experiment` | **yes** | `A` or `B`. Recorded into every artefact; never inferred from `--data`, so the two can be cross-checked. |
| `--seed` | **yes** | `0..4`. Seeds every RNG the method touches (`random`, `numpy`, `torch`, CUDA). |
| `--out` | **yes** | `results/new_experiments/experiment_<X>/<Method>/`. |
| `--epochs` | no | Defaults to the method's frozen budget. If you change it, it is a deviation and goes in §10. |

**Required firewall guard — copy verbatim, do not paraphrase (§4).** The trainer opens
`train.npy` and, for model selection only, `disc.npy`. Any other array under the dataset
directory is a protocol violation, and the guard is what makes that a crash instead of a
silent contamination.

**Required outputs, all five seeds.** These are the *inputs* to the tooling; if any is
missing or misnamed, the tools fail or — worse — silently produce a short table:

1. `generated_paths/seed_N/generated_paths_8192x128.npy` — `(8192, 128)`, float64,
   **original price scale, not log-returns, not standardised.** Whatever preprocessing the
   method applies internally must be inverted before writing. This is the single most common
   porting bug: a bank that looks fine to `np.load` but sits at the wrong scale scores
   catastrophically on every level-sensitive metric and plausibly on none of the others.
2. `generated_paths/seed_N/metadata.json` — the §6.4 schema, **including `first_nan_epoch`
   and `gen_has_nan`.**
3. `weights/seed_N_model.pt` and `weights/seed_N_config.json` — the config carries `scaler_mu`,
   `scaler_sigma`, `data` (absolute) and `experiment`, which is what makes the §8 item-15
   independence proof possible.
4. `losses/seed_N_losses.csv` — one row per epoch, first column `epoch`, last column `lr`.
5. `code/logs/exp<X>_seed<N>.log` — the log must echo the resolved `--data` path, so item 10
   of §8 can confirm the lane trained on what it claimed.

**Scaler rule.** Fit the scaler on `train.npy` **only**, never on the concatenation of train
and test, and record the fitted statistics. Two experiments trained on their own data will
have *different* fitted statistics, and that difference is the one piece of retraining
evidence that does not require trusting the config file (§8 item 15).

**What the contract deliberately does not constrain.** Architecture, optimiser, loss,
preprocessing, epoch budget, whether the method uses EMA weights at generation time. All of
that is the method's business. If generation uses weights other than the last checkpoint —
LS4 uses EMA, `ema_lamb = 0.99` — say so in README §1.4, because it changes how the final-epoch
loss should be read.

### 6.6 Verifying the layout mechanically

"Same structure of files" is a claim, and claims get checked. `tools/check_method_layout.py`
walks a method directory against §6.2 and exits non-zero on the first violation:

```bash
python results/new_experiments/tools/check_method_layout.py \
    --root results/new_experiments/experiment_A/LS4 --experiment A
```

It checks presence and *shape*, not just names: every one of the five seeds present in each
of the five per-seed families; `generated_paths_8192x128.npy` actually `(8192, 128)`, finite,
strictly positive, `S₀ == 100.0`; `metadata.json` carrying `gen_has_nan: false` and
`first_nan_epoch: null`; `metrics_summary.csv` with the exact §6.4 header; the evaluator
JSONs parsing on both the test and validation sides. Run it **before** writing the README —
it is cheaper to find a missing seed here than in a regenerated table.

Run it against `LS4` first. If it does not pass on the reference method, the checker is wrong,
not your new method.

---

## 7. README Specification

This section is the contract. A new method's README is *correct* iff it satisfies every rule
below; anything not specified here is the author's choice. Read §7.0 first — it is the rule that
generates all the others, and it is the rule that was violated in the first pass.

### 7.0 The generating rule — tables are emitted, never typed

> **Every numeric cell in every README table is produced by a tool in
> `results/new_experiments/tools/`, redirected to a file, and spliced in whole. No cell is
> ever typed, retyped, reformatted, annotated, or "improved" by hand.**

If a number needs explanation, the explanation goes in **prose above or below the table**, never
inside the cell. The moment you replace a machine-emitted cell with a word, a symbol, or a dash,
three things happen at once and you will notice none of them:

1. The table stops being reproducible — the `*Reproduce:*` command printed under it now emits
   something different from what is displayed, and nobody re-runs it to find out.
2. Real signal gets erased. This is not hypothetical: see the `—` incident in §7.10 E1.
3. The two experiments drift apart, because you will hand-edit one and not the other.

**Enforcement.** Every table carries, immediately beneath it, a `*Reproduce:*` line with the
literal command. Before committing, re-run each one and diff it against the file:

```bash
cd results/new_experiments/experiment_A/LS4
python ../../tools/aggregate_pdf_metrics.py --model-dir . --floor-dir ../perfect_floor \
    --pattern '*_drawdown_memory.json' --label LS4 \
    --exclude-prefix configuration sources > /tmp/regen.md
# splice /tmp/regen.md into the README and confirm `git diff` is empty
```

A non-empty diff means either the numbers changed (re-run the pipeline) or someone hand-edited
the table (revert the hand edit). Both are things you want to know before pushing.

### 7.1 Section order — non-negotiable

**Start from the template, do not retype it.** `results/new_experiments/README_TEMPLATE.md`
is this skeleton with every rule below embedded as an HTML comment next to the place it
applies. Copy it, fill the `<PLACEHOLDER>`s, delete the comments:

```bash
cp ../../README_TEMPLATE.md README.md
```

The skeleton it contains is:

Five sections, in this order. The PDF metrics come **first and standalone** because they are what
the protocol actually asks; the repo's own battery is context, not verdict.

```markdown
# <Method> — Experiment <X> (<one-line name of the experiment>)

<3-6 line abstract: what the experiment tests, what the headline result is, and the single
number a reader should remember. State the conclusion here; do not make them hunt for it.>

| Section | Contents | Question it answers |
|---|---|---|
| **1** | PDF metrics — protocol evaluator | Does it satisfy the protocol? |
| **2** | Benchmark standard battery (A1–A34 + B curve-shape) | How does it score on the repo's usual metrics? |

Section 1 is the one that decides the experiment. Section 2 is context.

**Perfect floor** = the true DGP re-simulated with a fresh seed (5 independent simulations).
It is the best score any generator can achieve; it is *not* zero, because the evaluator
compares two finite 8192-path samples.

---

## 1. PDF metrics — protocol evaluator
### 1.1 PRIMARY panel — the four metrics the protocol designates
### 1.2 <Headline interpretation table — experiment-specific name>
### 1.3 Validation vs test
### 1.4 §5 reporting block
## 2. Metrics A1–A34 + B, mean ± std across 5 seeds
### 2.1 A1–A34
### 2.2 B curve-shape metrics
## 3. Stylised Facts Diagnostic
## 4. Losses
## 5. File layout
```

### 7.2 § 1 — PDF metrics *(first, and standalone)*

Five blocks, in this order. Blocks 1–4 are tables; block 5 is the §5 reporting block.

**Header prose, before any table.** State: which evaluator script produced this, that it was
**run unchanged** (PDF §7 item 7), and the aggregation rule (mean ± sample std, ddof = 1, over 5
seeds; 95 % CI half-width with t₀.₉₇₅,₄ = 2.776).

---

**Block 1 — the full flattened table, verbatim from `aggregate_pdf_metrics.py`.**

Columns: `Metric | <Method> (mean ± std) | 95% CI half-width | Perfect floor (mean ± std)`.
Every numeric leaf the evaluator emits appears, sorted, including the `target_*` rows.

- Experiment A: the `target_memory` / `generated_memory` pair, all 9 `errors`, all 3 `novelty`.
- Experiment B: all of `mixture_fidelity` (incl. per-parameter `std_ratio`),
  `observable_fidelity`, `novelty`, and `oracle_gate` accuracy.

> **The `target_*` rows carry `± 0` and a CI of `0`, and you print those zeros.**
> They are computed from `test.npy` alone and never touch the generated bank, so they are
> bit-identical across seeds. **Do not** write `(target)` in the mean cell or `—` in the CI cell.
> Explain the zero in prose instead:
>
> *"A zero here means 'this quantity was never resampled', not 'we measured a variance and it
> came out small'."*
>
> Verify the claim, don't assume it — the check is four lines, and in Experiment B it **fails**
> for one row (`target_low_confidence_fraction`, std 8.63 × 10⁻⁵):
>
> ```python
> rows = [flatten(json.load(open(f))) for f in sorted(glob.glob("pdf_metrics/seed_*.json"))]
> for k in [k for k in rows[0] if "target" in k]:
>     vals = [r[k] for r in rows]
>     print("IDENTICAL" if len(set(map(repr, vals))) == 1 else "** VARIES", k, vals[0])
> ```

Do not fold the target into a column of block 1. The evaluator emits `target_*` and `generated_*`
as sibling blocks; flattening them into one row per metric needs a hand-kept key mapping that
silently rots the first time the evaluator gains a field.

---

**Block 2 — the PRIMARY panel.** The metrics the PDF itself designates. No others, no
substitutes. Fixed by the protocol, not a matter of taste:

| Experiment | PDF § | The four primary metrics |
|---|---|---|
| **A** | §2.3 | `early_hit_rate_error`, `future_rv_hit_gap_error`, `early_history_incremental_r2_error`, `future_rv_wasserstein` |
| **B** | §3.4 | regime TVD, **W̄_param**, realized-volatility Wasserstein, leverage-curve RMSE |

Columns: `Primary metric (↓ lower is better) | <Method> (mean ± std) | 95% CI half-width |
Perfect floor | × floor`. The `× floor` column is the one readers use; compute it as
`model_mean / floor_mean` and round to two significant figures.

Everything else in the evaluator output is **secondary diagnostics** and must be labelled so.

For Experiment B, `W̄_param` is **not emitted by the evaluator** and must be derived per seed as
the unweighted mean of the three `parameters.{theta,xi,rho}.support_normalized_wasserstein`
(§11.2). Label it *derived* and give the formula in the README. **Aggregate per seed first, then
across seeds** — averaging the three already-aggregated means gives the right centre and a wrong
std. Do **not** patch the evaluator to emit it; checklist item 7 forbids touching `protocol/`.

PDF §3.4 forbids collapsing B's four into an aggregate score after seeing results. It is a
panel. Report four numbers.

---

**Block 3 — the headline interpretation table.** The raw quantities that explain *why* the
primary metrics landed where they did. This is the table a reader actually reads, so give it a
descriptive `###` heading naming the finding, not a generic one.

- **Experiment A:** `Quantity | Target | <Method> | Perfect floor | Verdict`, over the
  `generated_memory` / `target_memory` pairs (`early_history_incremental_r2`,
  `future_rv_hit_mean`, `future_rv_no_hit_mean`, …).
- **Experiment B:** the eight-regime table, **sorted by Feller ratio 2κθ/ξ²**, not by label
  index. Columns `Label | theta | xi | rho | Feller | Target | <Method> (mean ± std) |
  Perfect floor`. Sorting by label hides the result; sorting by Feller makes the failure
  monotone and self-evident. Find the ordering variable before you write the table.

Interpret it in prose beneath (§2.5 for A, §3.5 for B).

> ⚠️ These are **raw diagnostics, not errors.** PDF §5 singles out "standard-deviation ratio,
> posterior confidence, group means, and novelty" as the exceptions to "lower is better". Mark
> them, or a reader will try to minimise `std_ratio` (target **1.0**) or
> `distinct_nearest_training_paths` (a raw count). And `early_history_incremental_r2` is a
> **two-sided** target: matching 0.2885 is the goal, overshooting is as wrong as undershooting.
> Only the `*_error` form belongs in the primary panel.

---

**Block 4 — validation vs test** (§11.4), from `pdf_metrics_validation/`. Four to six `errors.*`
rows, two columns. This is the artefact that evidences §7 checklist item 2 — that `disc.npy` was
used for validation and `test.npy` stayed blind. Generate it with the same tool via `--subdir`:

```bash
python ../../tools/aggregate_pdf_metrics.py --model-dir . --subdir pdf_metrics_validation ...
```

If validation is materially *better* than test, you have a leak. Say so rather than shipping it.

---

**Block 5 — the §5 reporting block.** PDF §5 requires each comparison table to state seven
things; six are easy and the seventh is always the one forgotten. Present as a two-column
`Requirement | Value` table. **Required rows, all of them:**

| Row | What it must contain |
|---|---|
| Train / validation / test files | Exact paths **and MD5s** |
| Generated bank size | `8192 × 128 float64, per seed` |
| Model seeds | `0, 1, 2, 3, 4`, and whether training seed = generation seed |
| **Independent retraining** | That there are **5 separate trainings, one per seed** — with the evidence, not the assertion: 5 distinct checkpoint MD5s, per-seed `config.json` recording `experiment` and `data`, 5 loss curves with distinct final losses. State the cross-experiment check too (all 10 A+B checkpoints mutually distinct). |
| **Trained on this experiment's own data** | The experiment-specific `train.npy` path + MD5, plus an *independent* corroboration that does not rely on the config file — the fitted scaler statistics differ between A and B (A: μ = 100.13621639651069, σ = 11.927961375843996; B: μ = 100.09714689788022, σ = 11.516370009298313), so the two runs provably did not share a training set. |
| Official code + revision | Repo path, preset name, benchmark revision, and the wrapper's own commit |
| Hyperparameters | **Defaults or validation-selected** — PDF §5's most-forgotten requirement |
| Trainable parameters / training time / generation time / hardware | One row each or one combined |
| Failed / unstable runs | Number **and reason**. `0` still needs the evidence: epochs completed, NaN check. Surface near-misses here too (see §7.10 E6). |
| Declared post-hoc transformation (§1.3) | Any repair, with its effect on the metrics measured, not asserted |
| Known non-conformance | `None.` is an acceptable answer **only** if each §1.4 bullet was re-measured from the array on disk |

All of it is machine-readable in `generated_paths/seed_<q>/generation_manifest.json`; the README
must still say it in prose. See §11.6.

### 7.3 § 2 — Metrics A1–A34 + B, mean ± std across 5 seeds

Two tables, from `metrics_summary.csv` via `make_metrics_tables.py --table A` and `--table B`,
with the perfect-floor column beside each. Per-seed columns are **required** — PDF §5 says report
every seed, not just the aggregate, and explicitly forbids reporting only the best seed.
A33/A34 → `n/a` with the one-line reason. Never delete the rows.

⚠️ Metric labels contain literal pipes (`A2 |r| q95`, `ACF of |log-returns|`). They must be
escaped `\|` or the whole Markdown table renders misaligned on GitHub. `make_metrics_tables.py`
does this; if you hand-edit a row, do it too — which is another reason not to hand-edit a row.

**This section must not restate § 1's numbers, and § 1 must not borrow § 2's.** They are separate
suites answering separate questions, and **the interesting result is usually the disagreement
between them.** Open § 2 with a blockquote that states whether the two suites agree, and if they
disagree, which one to believe and why. Both observed cases are worth copying:

- **Experiment A — they disagree.** A18 discriminative (GRU) 0.00638 and A19 predictive (GRU)
  0.04641 sit at or below the perfect floor, while § 1 shows the model recovers only 59 % of the
  memory signal. Both are correct: A1–A32 measure *marginal and short-lag* structure, whereas the
  protocol evaluator measures a *conditional, 32-step-delayed* dependence no marginal statistic is
  sensitive to. A model can saturate the standard battery and still miss the structure that matters.
- **Experiment B — they agree, except where it counts.** A14 KS 55× floor, A21 ACF |r| 73×, A31
  rolling-vol KS 79× — but A18 (GRU) is only 1.5× floor and A19 (GRU) 1.0×. A GRU discriminator
  cannot detect that two of eight modes are missing, because every individual path it inspects
  *is* plausible; only the **population** is wrong.

The lesson generalises: **`A18`/`A19` are blind to mode collapse.** Never let them carry a verdict.

### 7.4 § 3 — Stylised Facts Diagnostic

`plots/heston_diagnostics.png` (from `plot_stylised_facts.py`), real vs `<Method>`, seed 0, plus a
short honest reading of where the model deviates. The black Heston-theory reference curve is
suppressed by that tool because neither DGP is single-regime Heston — **say so explicitly**, so a
reader does not think it was forgotten.

### 7.5 § 4 — Losses

`losses/loss_convergence.png` (all 5 seeds overlaid) + a per-seed table of final loss, **minimum
loss and the epoch it occurred**, and wall-clock time. Final-loss-only hides late-training
excursions (§7.10 E6).

### 7.6 § 5 — File layout

The tree from §6.2, as a fenced block, plus a `Tool | Produces` table mapping every script in
`tools/` to the section it feeds. **No path-shadowing section** — it is not part of this protocol.

Every file that exists on disk must appear in the tree, including the ones the README does not
display inline (the 10 PCA/t-SNE figures). A file present but undocumented reads as an oversight.

### 7.7 Figure specification

**The full figure inventory — 13 per method per experiment.** Exactly three are displayed inline;
the other ten are produced automatically by the metric battery and are listed in the file tree
only.

| # | Path | Generator | Displayed in | Shows |
|---|---|---|---|---|
| 1 | `plots/memory_structure_seed0.png` (**A**) | `plot_experiment_figures.py --experiment A` | § 1 | Early-drawdown-hit vs future-RV relationship; the delayed-memory mechanism the experiment is about |
| 1 | `plots/mixture_structure_seed0.png` (**B**) | `plot_experiment_figures.py --experiment B` | § 1 | 8-regime proportion bars (target vs generated) + the three parameter marginals; **this is where mode collapse is visible as missing bars** |
| 2 | `plots/heston_diagnostics.png` | `plot_stylised_facts.py` | § 3 | 8-panel stylised-facts battery, real vs method, seed 0 |
| 3 | `losses/loss_convergence.png` | `plot_losses.py` | § 4 | All 5 training curves overlaid |
| 4–8 | `plots/seed_<q>_pca.png` | metric battery (`compute_all.py`) | file tree only | 2-D PCA of the path cloud, real vs generated |
| 9–13 | `plots/seed_<q>_tsne.png` | metric battery (`compute_all.py`) | file tree only | t-SNE embedding, real vs generated |

**Rules for displayed figures.**

- Reference with a relative path from the README: `![mixture structure](plots/mixture_structure_seed0.png)`.
  Never an absolute path, never a URL — the README must render from inside the repo.
- The alt text is a short lowercase description (`mixture structure`, `stylised facts`,
  `loss convergence`), not the filename.
- **Every figure carries a caption in prose that states a number appearing in the figure**, and
  that number must be cross-checked against the evaluator JSON. For B, the figure prints
  TVD = 0.245483 for seed 0 and the evaluator JSON reports 0.245483 — quote it and say they match.
  A figure whose numbers were never reconciled with the tables is decoration.
- Figures are **seed 0** by convention. If you display another seed, say which and why.
- Verify every link resolves before committing:
  ```bash
  grep -oE '!\[[^]]*\]\(([^)]*)\)' README.md | sed -E 's/.*\((.*)\)/\1/' \
    | while read f; do [ -f "$f" ] || echo "BROKEN: $f"; done
  ```

**Rule for figures that do not exist.** If a planned figure could not be produced, **do not put a
dash, a placeholder, or an empty image link.** Delete the row from the inventory and state in
prose why the figure is absent. An empty cell reads as an oversight; a sentence reads as a decision.

### 7.8 Aggregation rules (apply everywhere)

- 5 seeds `q ∈ {0,1,2,3,4}`, **mean ± sample std** (`ddof=1`).
- 95 % CI half-width = `2.776 · std / sqrt(5)` (`t_{0.975,4} = 2.776`).
- **Report every seed, plus mean, plus std, plus CI.** PDF §5: "Do not report only the best seed."
- **Failed seeds are reported, never silently replaced** (PDF §1.5). State which seed failed and
  why (NaN loss, OOM, divergence) and aggregate over the survivors with the reduced `n` written down.
- A model is "at the floor" on a metric iff its 95 % CI overlaps the floor's. Say so explicitly.
- Never compare a raw metric to 0. Compare to the floor.
- Lower is better **except** for the §5 raw diagnostics: std-ratio, posterior confidence, group
  means, novelty. Mark those rows.

### 7.9 Cell-formatting rules

| Situation | Write | Never write |
|---|---|---|
| Metric with zero variance across seeds | `0.67399 ± 0` and CI `0` | `0.67399 (target)`, `—` |
| Metric the method does not produce | `n/a` **+ a one-line reason in prose** | blank, `—`, deleted row |
| Metric that failed to compute | `n/a (<error>)` and a note in §1.4 | silently omitted row |
| A number you are quoting in prose | the value re-read from the JSON | a value recalled from earlier in the session (§7.10 E5) |
| A label containing a pipe | `A2 \|r\| q95` (escaped) | the bare unescaped form |

The single em-dash `—` is reserved for cells in *hand-written* interpretation tables where the
column genuinely does not apply to that row (e.g. a "Target" column for a metric that has no
target). It must never appear in a machine-emitted table.

### 7.10 Errors met building this — do not repeat them

Each entry is a mistake actually made during the LS4 A+B build, with the check that catches it.

**E1 — Hand-annotating a machine-emitted table, which erased real signal.**
The first version of this guideline instructed: *"Write `(target)` in their mean cell and `—` in
their CI cell rather than printing a meaningless `± 0 | 0`."* That was followed for Experiment A.
It was wrong twice over: the README diverged from its own `*Reproduce:*` command, and — decisively —
the same annotation applied to Experiment B would have erased the fact that
`target_low_confidence_fraction` is **not** constant (std 8.63 × 10⁻⁵ across seeds). The "obviously
constant" quantity was not constant, and the hand-annotation was precisely the thing that would
have hidden it. **Check:** re-run every `*Reproduce:*` command and require an empty diff.

**E2 — `taskset` swallowing environment assignments.**
```bash
CUDA_VISIBLE_DEVICES=1 taskset -c 0-7 OMP_NUM_THREADS=8 python train.py   # exit 127
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 taskset -c 0-7 python train.py   # correct
```
Bash treats everything after `taskset` as the command, so `OMP_NUM_THREADS=8` was executed as a
binary. **All env assignments precede `taskset`.** **Check:** the job dies instantly with
`No such file or directory` — read the exit code, don't assume the queue is slow.

**E3 — `grep -i error` against JSON output, giving 15 false failures.**
The evaluator echoes its full JSON to stdout, and that JSON contains `excess_kurtosis_error`,
`return_std_error`, `leverage_curve_rmse`. Every successful job was reported as FAILED.
**Check:** validate structurally, never lexically —
```bash
python -c "import json,glob; [json.load(open(f)) for f in glob.glob('pdf_metrics/*.json')]" \
  && echo OK
```

**E4 — Documenting a filename without listing it.** The README claimed
`losses/seed_<q>_loss.csv`; the files are `seed_<q>_losses.csv`. **Check:** `ls` every path the
README names, don't rely on memory of the writer script.

**E5 — Quoting numbers from memory instead of re-reading them.** Per-seed TVDs were written into
prose as `0.2455/0.3125/0.2456/0.2461/0.2178`; the measured values are
`0.245483/0.243164/0.258911/0.335815/0.250977`. The narrative claim built on them ("seed 2 is
mid-pack") survived, but the identity of the worst seed did not — it is seed 3, not seed 1.
**Check:** every number in prose is re-read from the JSON in the same session it is written.

**E6 — Reporting only the final loss, which nearly hid an unstable seed.**
`Failed / unstable runs: 0` was true, but Experiment B seed 2's last-epoch loss is −0.5683 against
its own epoch-92 minimum of −1.1118. It turned out not to matter — generation uses the **EMA**
weights (`ema_lamb = 0.99`; `train_ls4_experiment.py:154` selects `ema_model.module`), so a
one-epoch excursion cannot reach the bank — but that is a *conclusion*, and it required checking
which weights generation uses. **Check:** tabulate final **and** minimum loss with its epoch, then
confirm which weights the generator actually loads.

**E7 — Overclaiming in a draft.** A draft read "A18/A19 are essentially floor-level"; measurement
gave A18 GRU = 1.5× floor, only A19 = 1.0×. **Check:** every comparative adjective
("essentially", "at the floor", "matches") must be replaced by a ratio before committing.

**E8 — Assuming a shared repo is yours alone.** A concurrent process commits to this repository;
HEAD moved twice mid-session while no lock was held. **Check:** always commit with an explicit
pathspec, and confirm the index is clean first —
```bash
git add -- dataset/Heston/new_experiments results/new_experiments
git diff --cached --name-only \
  | grep -v -E '^(dataset/Heston/new_experiments|results/new_experiments)/' \
  && echo "FOREIGN FILES STAGED" || echo clean
git commit -m "..." -- dataset/Heston/new_experiments results/new_experiments
```

**E9 — Repairing an artefact that exists to be a control.** When the S₀ defect was found, the
instinct was to apply the fix everywhere. The perfect floor must **never** be repaired: it is
drawn from the true DGP, so if it ever violates the output contract that is a *finding* about the
generator, and silently normalising it destroys the only independent reference in the experiment.
**Verify the floor; repair only the model.** See §11.5.

**E10 — Explaining an anomaly instead of bounding it.** `target_low_confidence_fraction` jitters
by ±1 path across seeds when it should be a pure function of `test.npy` and a frozen oracle. A
threading hypothesis was formed and then **falsified** (two runs at different `OMP_NUM_THREADS`
both reproduced 1408 exactly). The correct output was to bound the effect (1.2 × 10⁻⁴ on one raw
diagnostic, zero effect on any primary metric) and write **"The cause was not isolated and is not
claimed."** An unexplained anomaly, bounded and disclosed, is a result. A plausible story with no
evidence is a liability.

---

## 8. Adding a New Method — Checklist

- [ ] 1. Answer the root `GUIDELINE.md` §0 questions Q1–Q8 for the method.
- [ ] 2. **Stage 0 gate**: reproduce the method's committed 8192×128 Heston result. Record the
      level reached (1–4) and the exact numbers.
- [ ] 3. Confirm the figures reproduce the method's **known defects**, not just its fits.
- [ ] 4. Get Theo's explicit validation of the gate. Do not proceed without it.
- [ ] 5. `mkdir -p results/new_experiments/experiment_<X>/<Method>/{code,generated_paths,losses,weights,plots,pdf_metrics}`
- [ ] 6. Write `code/train_<method>_experiment.py` — **firewall guard verbatim**, `--data` required,
      `--seed`, `--experiment {A,B}`, scaler stats from `train.npy` only.
- [ ] 7. Copy `compute_metrics_experiment.py` and adjust only the path constants.
- [ ] 8. `nvidia-smi` + `htop` **before** launching. If a GPU is busy or RAM > 50 %, ask.
- [ ] 9. Launch 5 seeds in **exactly 2 lanes**, `setsid ... & disown`, with `cd` inside each.
- [ ] 10. Verify within 60 s: 2 PIDs, 2 GPUs with memory, correct `[data]` path in both logs.
- [ ] 11. Stage 3a — A/B suite on the model bank.
- [ ] 12. Stage 3b — the same on the perfect floor (skip if the floor already exists for this experiment).
- [ ] 13. Stage 3c — the PDF evaluator, per seed, **unchanged**.
- [ ] 14. Stage 4 — loss convergence + diagnostics + the experiment-specific figure.
- [ ] 15. **Prove the 5 trainings were independent and used this experiment's data** (§7.2 block 5):
      ```bash
      md5sum weights/seed_*_model.pt | awk '{print $1}' | sort -u | wc -l   # must be 5
      for s in 0 1 2 3 4; do python -c "import json;d=json.load(open('weights/seed_${s}_config.json'));print(d['seed'],d['experiment'],d['data'])"; done
      for s in 0 1 2 3 4; do tail -1 losses/seed_${s}_losses.csv; done      # 100 epochs, distinct losses
      ```
      Then check the fitted scaler stats differ from the *other* experiment's — that is the one
      piece of evidence that does not depend on the config file being honest.
- [ ] 15b. **Machine-check the layout before writing any prose** (§6.6). It must print `PASS`:
      ```bash
      python ../../tools/check_method_layout.py --root . --experiment <X>
      ```
      209 checks on the reference method. It re-measures shapes and the §1.4 output contract
      from the arrays themselves, so it catches the classic porting bug — a bank that loads
      fine but sits at the wrong price scale.
- [ ] 16. Write `README.md` by **copying `results/new_experiments/README_TEMPLATE.md`** and
      filling the placeholders — 5 sections in order, PDF metrics first (§7.1).
- [ ] 17. **Re-run every `*Reproduce:*` command in the README and require an empty diff** (§7.0).
      This is the check that catches hand-edited cells; it is the single most valuable step here.
- [ ] 18. Verify every figure link resolves and every file on disk appears in the §5 tree (§7.7).
- [ ] 19. Re-read every number quoted in *prose* from its JSON — not from memory (§7.10 E5).
- [ ] 20. Check every `metadata.json` for `gen_has_nan: false` and `first_nan_epoch: null`,
      and re-measure the §1.4 output contract from the arrays on disk (finite, > 0, shape,
      dtype, `S₀ == 100.0`) rather than asserting it.
- [ ] 21. Commit with an **explicit pathspec** and confirm no foreign files are staged (§7.10 E8),
      then push.

---

## 9. Hardware Rules

**Hard limits, enforced 24/7 including weekends. The machine is shared.**

| Resource | Limit |
|---|---|
| GPUs | **2** (of 4 A100-SXM4-80GB) |
| Physical cores | **16** |
| RAM | **~250 GiB** (of 503 GiB) |
| DataLoader `num_workers` | ≤ 4 per DataLoader per GPU |

```bash
nvidia-smi    # who is using what
htop          # cores, RAM, load
```

If another user holds a GPU or RAM is > 50 %, **ask before launching**.

The pinning trio, on every command:

```bash
CUDA_VISIBLE_DEVICES=<g>   # else the code grabs all 4
taskset -c <lo>-<hi>       # else processes fight over cores
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8   # else numpy/torch spawn 256 threads
```

Standard split: lane 1 = GPU 2 / cores 0-7, lane 2 = GPU 3 / cores 8-15.

⚠️ **`fit_heston_mixture_oracle.py` defaults to `--workers 24`, which exceeds the 16-core cap.
Always pass `--workers 16`.** Same class of check applies to any vendored script's defaults.

Never run a single-threaded job on idle hardware — parallelise **within** the limits.

### Contention arriving mid-run

Checking `nvidia-smi` before launch does not protect you afterwards. During the Experiment A
run another user's jobs landed on GPUs 0–2; our GPU-2 lane dropped to roughly **half** the
throughput of the uncontended GPU-3 lane (seed 0 at epoch 50 while seed 1 was at epoch 75).

**Do not kill their job, and do not migrate yours mid-flight** — restarting throws away
completed epochs. Diagnose first, and distinguish *slow* from *dead*:

```bash
ps -o pid,etime,time,pcpu,args -p $(pgrep -d, -f train_<method>_experiment.py)
```

Accumulating `TIME` at ~100 % `%CPU` means slow-but-alive. A flat `TIME` means dead — only
then intervene.

### Never let two lanes target the same seed

The tempting fix for an unbalanced pair is to hand the idle lane a seed still queued on the
busy lane. **Do not.** Both would write
`generated_paths/seed_N/generated_paths_8192x128.npy` concurrently and corrupt it, and racing
to kill the loser is unreliable.

Give the freed lane work with a **disjoint output path** instead — the other experiment's
seeds, or the next method. That is what was done here: when lane 2 drained, it took
Experiment B rather than Experiment A's remaining seed 4.

---

## 10. Known Deviations and Caveats

| # | Deviation | Status |
|---|---|---|
| 1 | **A33/A34 dropped** in both experiments. Both need the Heston teacher variance `v`, undefined for both DGPs. Implemented by `v = None` + `compute_all`'s existing try/except → `null`. **No canonical code edited.** | Theo's decision, deliberate |
| 2 | `protocol/experiments/path_dt_experiments/__init__.py` is **empty** upstream and vendored empty. Not a truncation. | expected |
| 3 | `manifest.json` float fields can differ in the **last ULP** across numpy versions. Compare with `np.allclose`, not `==`. Shapes, seeds and `regime_counts` are exact and must match. **Confirmed empirically** — see §11.1: both manifests miss the PDF's SHA-256 while all six `.npy` files match bit-for-bit. | expected, measured |
| 4 | `oracle.joblib` (~554 MB) **cannot be byte-reproduced** across scikit-learn/joblib versions even with `random_state=42`. The **gate accuracy** is the reproducible artefact; `gate_report.json` is committed, the blob is gitignored. Refit locally if missing. **Confirmed empirically** — see §11.1: `gate_report.json` matches the PDF's SHA-256 byte-for-byte, including all 64 confusion-matrix entries, while the blob does not. | expected, measured |
| 8 | **`S₀ ≠ 100` in every raw LS4 bank.** PDF §1.4 requires banks to "begin at S₀ = 100, up to ordinary floating-point tolerance"; §7 checklist item 5 repeats it. LS4 generates in standardized *price* space (`x → (x−μ)/σ`) with no anchor at `t = 0`, so raw `S₀` deviated by up to **3.5 × 10⁻²** — ten orders of magnitude beyond float tolerance, a genuine non-conformance, not a rounding artefact. **Measured impact on every PDF metric: nil** (§11.5), but the contract is a contract. Repaired by the declared renormalization `S ← 100·S/S[:,:1]`, which preserves every log-return to 1e-12; §2 battery re-run because it is level-sensitive. | **resolved — declared repair, §11.5** |
| 5 | Perfect-floor seeds are **1000–1004** (`IND_SEED_BASE = 1000` from `metrics/compute_perfect_recovery.py`), disjoint from protocol seeds 0/1/2/100/101. | convention |
| 6 | `plot_diagnostics.py` draws a **Heston-theory** curve that is invalid for both new DGPs. It is inside a try/except (`TB = None` on failure). Suppress it or caption it as non-applicable. | known |
| 7 | **No Git LFS** (decision reversed 2026-07-30). Add no new LFS entries. The 8192×128 `float64` banks are ~8 MB each and **are committed normally**; only artefacts over GitHub's 100 MiB per-file cap get an explicit `.gitignore` entry — currently just `oracle.joblib`. | policy |

---

## 11. Full PDF Conformance Cross-Check

Run in full on **2026-07-31** against `synthetic_benchmark_protocol_drawdown_heston_mixture.pdf`.
Every claim below is **measured**, not asserted. Re-run this whole section before any
submission; it is cheap and it is the only thing standing between "we followed the protocol"
and "we believe we followed the protocol".

### 11.1 §6 integrity — SHA-256 against the PDF's published digests

```bash
cd dataset/Heston/new_experiments
sha256sum experiment_A/{train,disc,test}.npy experiment_A/manifest.json \
          experiment_B/{train,disc,test}.npy experiment_B/manifest.json \
          experiment_B/oracle/oracle.joblib experiment_B/oracle/gate_report.json
```

| Canonical file | PDF §6 digest matches? | Reading |
|---|---|---|
| Drawdown `train.npy` | ✅ **bit-for-bit** | |
| Drawdown `disc.npy` | ✅ **bit-for-bit** | |
| Drawdown `test.npy` | ✅ **bit-for-bit** | |
| Drawdown `manifest.json` | ❌ | diagnostics-float ULP only — see below |
| Heston-mixture `train.npy` | ✅ **bit-for-bit** | |
| Heston-mixture `disc.npy` | ✅ **bit-for-bit** | |
| Heston-mixture `test.npy` | ✅ **bit-for-bit** | |
| Heston-mixture `manifest.json` | ❌ | diagnostics-float ULP only |
| `oracle.joblib` | ❌ | pickle version strings — model is identical |
| `gate_report.json` | ✅ **bit-for-bit** | |

**All six data arrays reproduce exactly.** That is the finding that matters: our DGP
implementation is the PDF's, to the last byte, for both experiments and all three splits.

The three misses are all explained and none is substantive:

* **Manifests.** Both are written with `json.dumps(..., indent=2, sort_keys=True) + "\n"`,
  so serialization is deterministic; the `configuration` / `regimes` / `splits` blocks are
  integers and exact decimals. The only free variables are the derived diagnostic floats
  (`log_return_mean`, `log_return_std`, `sigma_mean`, `price_min` …). Those come from numpy
  pairwise-summation reductions whose blocking changes with numpy version and SIMD width.
  Verified locally: `numpy.mean` reproduces every stored value exactly, and `math.fsum`
  agrees to 1.6 × 10⁻¹⁶ relative. So the difference lives in the 16th significant figure of
  metadata that nothing consumes. **Never "fix" a manifest by hand-editing it to match a
  digest** — regenerate it or accept the ULP.
* **`oracle.joblib`.** A pickled `ExtraTreesClassifier` embeds scikit-learn and joblib
  version strings and per-object layout. Not byte-reproducible by construction. The proof
  that the *model* is identical is that `gate_report.json` matches the PDF's digest exactly
  — that file contains the full 8 × 8 confusion matrix over 8192 validation paths (all 64
  integers), `eight_regime_accuracy = 0.909423828125`, and the three per-parameter
  accuracies. A different forest cannot land on 64 identical cell counts.

Environment that produced the above: numpy 2.4.6, scikit-learn 1.9.0, joblib 1.5.3,
Python 3.12.3.

### 11.2 Metric-by-metric conformance

**Experiment A — `evaluate_drawdown_memory.py`.** The PDF's §2.3 four **primary** metrics
are all emitted, under `errors.`:

| PDF §2.3 primary metric | Emitted key | Present |
|---|---|---|
| Early hit-rate error | `errors.early_hit_rate_error` | ✅ |
| Future-RV hit-gap error | `errors.future_rv_hit_gap_error` | ✅ |
| Early-history incremental R² error | `errors.early_history_incremental_r2_error` | ✅ |
| Future-RV Wasserstein | `errors.future_rv_wasserstein` | ✅ |

Secondary, also emitted and reported: `return_std_error`, `excess_kurtosis_error`,
`abs_return_acf_rmse_lags_1_50`, `squared_return_acf_rmse_lags_1_50`,
`terminal_log_price_ks`. Raw diagnostics (not errors, no "better" direction) live under
`target_memory.*` and `generated_memory.*`; novelty under `novelty.*`.

Frozen configuration re-verified against PDF §2.1/§2.2 field by field — `split_index 64`,
`history_cutoff 40` (**inclusive**), `recent_window 20`, `future_horizon 32`,
`response_delay 32`, `memory_half_life 60.0`, `drawdown_threshold 0.04`,
`sigma_low 0.12`, `sigma_high 0.45`, `dt 0.004`, `s0 100.0`, `sequence_length 128`,
`num_paths 8192`, seeds `{train: 0, test: 1, disc: 2}`. **All match.**

**Experiment B — `evaluate_heston_parameter_mixture.py`.** PDF §3.4's four **primary**
panel entries:

| PDF §3.4 primary metric | Emitted key | Present |
|---|---|---|
| Regime TVD | `mixture_fidelity.regime_proportion_tvd` | ✅ |
| **W̄_param** (unweighted mean of the three normalized W₁) | — | ❌ **derived, see below** |
| Realized-volatility Wasserstein | `observable_fidelity.realized_volatility_wasserstein` | ✅ |
| Leverage-curve RMSE | `observable_fidelity.leverage_curve_rmse_lags_0_20` | ✅ |

> ⚠️ **`W̄_param` is not emitted by the evaluator.** The script emits only the three
> per-parameter `mixture_fidelity.parameters.{theta,xi,rho}.support_normalized_wasserstein`.
> PDF §3.3 requires reporting **all three *and* their unweighted average**. Since checklist
> item 7 forbids editing the evaluator, the README must **derive** it:
>
> ```
> W̄_param = (W^norm_θ + W^norm_ξ + W^norm_ρ) / 3
> ```
>
> computed per seed, then aggregated mean ± std like everything else. Label it explicitly as
> *derived from the evaluator's three components*, so nobody hunts for it in the JSON.

Support widths are computed inside the evaluator as `levels.max() − levels.min()`, giving
**θ 0.08, ξ 0.75, ρ 1.98** — exactly the PDF §3.3 values. Oracle config re-verified against
PDF §3.2: `trees 500`, `min_samples_leaf 2`, `max_features 0.7`, `random_state 42`,
`num_features 77`, `train_paths 32768`, `validation_paths 8192`,
`minimum_accuracy 0.90`, achieved **0.909423828125 → gate PASSES**. Scoring is valid.

### 11.3 §7 Return Checklist — item by item

| # | Checklist item | Status |
|---|---|---|
| 1 | Only `train.npy` used for learned preprocessing and model params | ✅ — scaler μ/σ fit on `train.npy`; firewall enforced (§4) |
| 2 | `disc.npy` used only for validation; `test.npy` not inspected pre-freeze | ✅ — and now **evidenced** by `pdf_metrics_validation/` (§11.4) |
| 3 | No latent vol, regime label, oracle output, or competing-model result used | ✅ — generator reads one array, `train.npy` |
| 4 | Five banks of shape 8192 × 128, or a documented deterministic exception | ✅ — seeds 0–4, no exceptions |
| 5 | Every bank finite, positive, **starts at 100**, in price units | ✅ all four — `S₀ == 100.0` exactly after the declared repair of §11.5; re-measured from the array on disk by `write_generation_manifest.py`, not asserted |
| 6 | Manifests contain code revision, seeds, preprocessing, hyperparameters, compute, failure info | ✅ — **added 2026-07-31** via `write_generation_manifest.py`; the old `metadata.json` alone did *not* satisfy this |
| 7 | Supplied evaluator scripts run unchanged | ✅ — `git status --porcelain dataset/Heston/new_experiments/protocol` is empty; single vendoring commit `ba7c748` |
| 8 | Per-seed metrics and aggregate uncertainty retained; no seed dropped on its score | ✅ — every seed is a column in the README tables; 0 failed runs |

**Item 7 is worth a standing habit.** Before every submission run:

```bash
git status --porcelain dataset/Heston/new_experiments/protocol   # must print nothing
git log --oneline -- dataset/Heston/new_experiments/protocol     # must show only the vendoring commit
```

### 11.4 Validation scoring — `metrics_validation.json` (was missing)

PDF §1.4's recommended layout lists **both** `metrics_test.json` and
`metrics_validation.json`. Only the test-side scoring existed. The validation side is the
artefact that *demonstrates* checklist item 2 rather than merely claiming it: run the same
unchanged evaluator with `disc.npy` substituted for `--test-data`.

```bash
cd dataset/Heston/new_experiments
E=protocol/experiments/scripts/evaluate_drawdown_memory.py
R=../../../results/new_experiments/experiment_A
for d in LS4 perfect_floor; do
  mkdir -p $R/$d/pdf_metrics_validation
  for s in 0 1 2 3 4; do
    OMP_NUM_THREADS=8 taskset -c 0-7 python $E \
      --train-data experiment_A/train.npy --test-data experiment_A/disc.npy \
      --generated-data <bank for $d seed $s> \
      --dataset-manifest experiment_A/manifest.json \
      --output $R/$d/pdf_metrics_validation/seed_${s}_drawdown_memory.json &
  done
done
wait
```

Note the flag is `--dataset-manifest`, **not** `--manifest`; the latter exits 2.

Result for LS4 on Experiment A — validation vs test, mean ± std over 5 seeds:

| `errors.*` | test | validation |
|---|---|---|
| `early_history_incremental_r2_error` | 0.118178 ± 0.0154 | 0.128587 ± 0.0154 |
| `future_rv_hit_gap_error` | 0.0503776 ± 0.00241 | 0.05117 ± 0.00241 |
| `early_hit_rate_error` | 0.0081543 ± 0.00691 | 0.00996094 ± 0.00849 |
| `future_rv_wasserstein` | 0.0119597 ± 0.00142 | 0.0125091 ± 0.00135 |
| `terminal_log_price_ks` | 0.0247803 ± 0.00912 | 0.0226807 ± 0.00829 |

Every metric agrees within one standard deviation, and the validation side is **marginally
worse** on the memory metrics. That is the opposite of the signature of `disc.npy`
overfitting, and it is the cleanest available evidence that the test split stayed blind.
**Keep this table in the submission.**

### 11.5 The S₀ non-conformance — found, quantified, **repaired**

> **Status: resolved.** The repair below was applied to all ten LS4 banks (5 seeds × 2
> experiments) and is declared in every `generation_manifest.json`. This subsection is kept
> in full because the *reasoning* is the reusable part: the same failure will recur for any
> method that generates in price space without a `t = 0` anchor, and the next person needs
> the invariance argument and the measurement, not just the one-line fix.

PDF §1.4 allows "ordinary floating-point tolerance". A float64 round-trip is ~10⁻¹²
relative. LS4's raw banks were off by up to **3.5 × 10⁻²**. The tolerance clause does not
cover this; it was a real violation of checklist item 5.

**Root cause.** `train_ls4_experiment.py` does `Xg = gen_s[:, :, 0] * sigma + mu` — the model
generates the whole path in standardized *price* space and nothing pins `t = 0`. A
log-return preprocessing would have made `S₀ = 100` exact by reconstruction; this pipeline
deliberately uses none.

**Measured impact: nil.** Both evaluators and the oracle feature builder start from

```python
log_paths = np.log(prices / prices[:, :1])     # evaluate_drawdown_memory.py:54
                                               # heston_mixture.py:135 (extract_price_path_features)
```

and Experiment B's terminal statistic is `np.log(x[:, -1] / x[:, 0])`. Every PDF metric is
therefore **invariant to S₀ by construction**. Verified rather than assumed: re-running the
Experiment A evaluator on a renormalized seed-0 bank changed **27 of 36 numeric leaves not
at all**, and the other 9 only in the **15th–16th significant figure**, e.g.

```
early_history_incremental_r2_error  0.13966753072671312 -> 0.139667530726713
future_rv_wasserstein               0.013922297890306213 -> 0.013922297890306216
```

**The remedy, applied.** A one-line declared repair — PDF §1.3 permits declared post-hoc
transformations and §1.4 has a field for exactly this:

```python
S = 100.0 * S / S[:, :1]
```

Applied in place to all ten LS4 banks, each guarded by four assertions that must hold
*after* the write, so a silent corruption cannot pass:

```python
assert np.isfinite(r).all() and (r > 0).all()          # §1.4 finite + strictly positive
assert np.all(r[:, 0] == 100.0)                        # §1.4 / §7 item 5, exactly
assert np.allclose(np.diff(np.log(a), axis=1),         # the path law is untouched:
                   np.diff(np.log(r), axis=1),         # every log-return preserved
                   rtol=0, atol=1e-12)
```

Measured deviations before → after:

| | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| Experiment A | 8.13e-03 | 1.03e-02 | 1.42e-02 | 1.38e-02 | 7.63e-03 |
| Experiment B | 1.48e-02 | 3.25e-02 | 2.23e-02 | 3.48e-02 | 7.07e-03 |
| after (both) | **0** | **0** | **0** | **0** | **0** |

Recorded via `write_generation_manifest.py --repair s0_renormalization`, which writes the
formula, the justification, and the residual deviation into
`generation_manifest.json → numerical_repair`, and independently re-measures
`output_contract` from the array on disk — so the manifest cannot claim a conformance the
bank does not have.

**The cost, paid.** PDF metrics were unaffected (predicted above, then confirmed: A's four
primary metrics are bit-identical pre- and post-repair). But the README §2 battery contains
**level-sensitive** metrics — A13 mean-path RMSE, A25 mean RMSE, and the price-space
MMD/SWD family A6/A7/A10/A11 — so the repair obliged a full re-run of
`compute_metrics_experiment.py` for both experiments and a re-render of README §2.

**Do not touch the perfect floor.** Floor banks are true-DGP draws and already satisfy
`S₀ == 100` exactly. Renormalizing them would be a no-op at best and, if the floor were ever
regenerated with a different construction, would silently mask a real defect. Verify, never
repair, the floor.

The pre-repair banks remain recoverable from commit `ba7c748` should the deviation itself
ever need to be re-examined.

### 11.6 §5 reporting requirements — what every comparison table must state

The PDF is prescriptive here and it is easy to satisfy six of seven and forget the seventh.
The README must state **all** of:

| §5 requirement | Where it lives |
|---|---|
| Exact training, validation, and test files | README header + `generation_manifest.json → data_files` |
| Generated bank size and all model seeds | README header (`8192 × 128`, seeds 0–4) |
| **Whether official code was used and its revision** | `generation_manifest.json → model.{official_implementation, source_revision, source_path}` |
| **Whether hyperparameters were defaults or validation-selected** | `generation_manifest.json → hyperparameter_origin` (`official-default` for LS4) |
| Trainable parameters, training time, generation time, hardware | `generation_manifest.json → model.trainable_parameters`, `compute`, `hardware` |
| Number and reason for failed runs | `generation_manifest.json → failure_information` (0 failed) |

The two bolded rows are the ones that were missing before 2026-07-31.

Also from §5, and easy to violate by accident:

* Report **every** seed, the mean, the sample std (`ddof=1`), and the 95 % CI half-width
  `t₀.₉₇₅,₄ · s/√5` with **t = 2.776**. Never only the best seed.
* For aligned-seed model-vs-model comparisons, additionally report the **five seedwise
  differences** and their paired interval.
* "All displayed fidelity metrics are errors or distances, so lower is better, **except**
  explicitly identified raw diagnostics such as standard-deviation ratio, posterior
  confidence, group means, and novelty." Those must be visually marked as raw, or a reader
  will minimise `std_ratio` (target 1.0) and `distinct_nearest_training_paths` (raw count).

### 11.7 Things the PDF forbids that are easy to do anyway

* **Do not build an aggregate score for Experiment B after seeing results** (§3.4). The
  four primary metrics are a panel, not a sum.
* **Do not combine novelty with fidelity** (§4). Novelty has no monotone "better"
  direction — it is a memorisation check, reported beside the fidelity metrics, never
  folded into them.
* **Do not treat `early_history_incremental_r2` as "higher is better"** (§2.3). It is a
  two-sided target: matching 0.2885 is the goal; overshooting is as wrong as undershooting.
  Only the `*_error` form is a "lower is better" quantity.
* **Do not silently replace a failed seed** (§1.5). Report it with its reason.
* **Do not hand-edit anything under `protocol/`** (§7 item 7). If the evaluator lacks a
  quantity the PDF wants — as with `W̄_param` — derive it in the README, not in the script.

---

## 12. Metric Provenance — the exact file and the exact key for every number

This section exists so that nobody ever has to *guess* where a number came from. Every value
that appears in a README is listed here with the file that holds it and the dotted key inside
that file. All paths are relative to the repo root `/home/tbasseras/benchmark`. Everything
below was read off disk, not recalled.

### 12.1 The four roots

| Root | Absolute path | Writable by a generator? |
|---|---|---|
| Dataset | `dataset/Heston/new_experiments/experiment_<X>/` | **read `train.npy`, `disc.npy` only** |
| Vendored protocol | `dataset/Heston/new_experiments/protocol/experiments/` | ⛔ never edited, never written |
| Deliverables | `results/new_experiments/experiment_<X>/<Method>/` | yes — this is the method's output |
| Shared tooling | `results/new_experiments/tools/` | method-neutral, edit only to add a feature |

### 12.2 Dataset files — exact names, shapes, and who may read them

`dataset/Heston/new_experiments/experiment_A/`

| File | Shape / dtype | Who may read it |
|---|---|---|
| `train.npy` | `(8192, 128)` float64 | **generator** + evaluator |
| `disc.npy` | `(8192, 128)` float64 | **generator** (model selection only) + evaluator |
| `test.npy` | `(8192, 128)` float64 | ⛔ evaluator only |
| `train_sigma.npy`, `disc_sigma.npy`, `test_sigma.npy` | **`(8192, 127)`** float32 | ⛔ evaluator only — latent vol, the answer key |

**The `_sigma` arrays are 127 wide, not 128** — one latent volatility per *increment*, not per
observation. Any code that assumes they align column-for-column with the price paths is wrong.
| `manifest.json` | — | evaluator (passed as `--dataset-manifest`) |
| `perfect_floor/floor_seed{1000..1004}.npy` (+ `_sigma.npy`) | `(8192, 128)` | ⛔ evaluator only |

`dataset/Heston/new_experiments/experiment_B/`

| File | Shape | Who may read it |
|---|---|---|
| `train.npy`, `disc.npy` | `(8192, 128)` | **generator** + evaluator |
| `test.npy` | `(8192, 128)` | ⛔ evaluator only |
| `train_labels.npy`, `disc_labels.npy`, `test_labels.npy` | `(8192,)` int | ⛔ evaluator only — the regime answer key |
| `oracle_train.npy` | `(32768, 128)` | ⛔ oracle fitting only |
| `oracle_validation.npy` | `(8192, 128)` | ⛔ oracle fitting only |
| `oracle_train_labels.npy`, `oracle_validation_labels.npy` | int | ⛔ oracle fitting only |
| `oracle/oracle.joblib` | 529 MiB, **gitignored** | ⛔ evaluator only — refit locally, see §3.3 |
| `oracle/gate_report.json` | — | ⛔ evaluator only (`--oracle-gate-report`) |
| `mean_heston.npy`, `whole_path_bootstrap.npy` | `(8192, 128)` | degenerate baselines, not part of a method run |
| `perfect_floor/floor_seed{1000..1004}.npy` (+ `_labels.npy`) | `(8192, 128)` | ⛔ evaluator only |

**Note that `experiment_A` has `*_sigma.npy` and `experiment_B` has `*_labels.npy`.** They are
not interchangeable, and a firewall guard copied from one experiment to the other must have its
forbidden-file list updated or it will happily let the answer key through.

### 12.3 The evaluators — exact scripts and exact CLI

Both live in `dataset/Heston/new_experiments/protocol/experiments/scripts/` and are run
**unchanged** (PDF §7 item 7).

**Experiment A — `evaluate_drawdown_memory.py`.** All five arguments are `required=True`:

```bash
python dataset/Heston/new_experiments/protocol/experiments/scripts/evaluate_drawdown_memory.py \
  --train-data        dataset/Heston/new_experiments/experiment_A/train.npy \
  --test-data         dataset/Heston/new_experiments/experiment_A/test.npy \
  --generated-data    results/new_experiments/experiment_A/<Method>/generated_paths/seed_0/generated_paths_8192x128.npy \
  --dataset-manifest  dataset/Heston/new_experiments/experiment_A/manifest.json \
  --output            results/new_experiments/experiment_A/<Method>/pdf_metrics/seed_0_drawdown_memory.json
```

**Experiment B — `evaluate_heston_parameter_mixture.py`.** Six required arguments; note there is
**no `--dataset-manifest`**, and two oracle arguments instead:

```bash
python dataset/Heston/new_experiments/protocol/experiments/scripts/evaluate_heston_parameter_mixture.py \
  --train-data          dataset/Heston/new_experiments/experiment_B/train.npy \
  --test-data           dataset/Heston/new_experiments/experiment_B/test.npy \
  --generated-data      results/new_experiments/experiment_B/<Method>/generated_paths/seed_0/generated_paths_8192x128.npy \
  --oracle              dataset/Heston/new_experiments/experiment_B/oracle/oracle.joblib \
  --oracle-gate-report  dataset/Heston/new_experiments/experiment_B/oracle/gate_report.json \
  --output              results/new_experiments/experiment_B/<Method>/pdf_metrics/seed_0_heston_mixture.json
```

⚠️ **The script name and the output name do not match for B.** The script is
`evaluate_heston_parameter_mixture.py`; the file it writes is `seed_N_heston_mixture.json`,
because that is the stem `aggregate_pdf_metrics.py --pattern '*_heston_mixture.json'` looks
for. Deriving either name from the other is exactly error class E4.

**Validation side.** Run each evaluator a second time with `--test-data` pointing at
`disc.npy` and `--output` into `pdf_metrics_validation/`. That second run is what backs
README §1.3, and it is the evidence that `test.npy` stayed blind. It is not optional (§11.4).

### 12.4 Experiment A — all 39 evaluator keys, by block

File: `results/new_experiments/experiment_A/<Method>/pdf_metrics/seed_<q>_drawdown_memory.json`

**`errors.*` — 9 keys. The four PRIMARY metrics live here.**

| Dotted key | PRIMARY? | Meaning |
|---|---|---|
| `errors.early_hit_rate_error` | ★ **yes** | \|generated − target\| early-hit rate |
| `errors.future_rv_hit_gap_error` | ★ **yes** | error on the hit/no-hit future-RV gap |
| `errors.early_history_incremental_r2_error` | ★ **yes** | error on the incremental R² of early history |
| `errors.future_rv_wasserstein` | ★ **yes** | W1 between future-RV distributions |
| `errors.abs_return_acf_rmse_lags_1_50` | no | stylised fact |
| `errors.squared_return_acf_rmse_lags_1_50` | no | stylised fact |
| `errors.excess_kurtosis_error` | no | stylised fact |
| `errors.return_std_error` | no | stylised fact |
| `errors.terminal_log_price_ks` | no | stylised fact |

**`generated_memory.*` — 9 keys**, the raw memory diagnostics measured on the model bank:
`augmented_r2`, `baseline_r2`, `early_history_incremental_r2`, `early_hit_future_rv_correlation`,
`early_hit_rate`, `early_hit_standardized_coefficient`, `future_rv_hit_gap`,
`future_rv_hit_mean`, `future_rv_no_hit_mean`.

**`target_memory.*` — the same 9 keys** measured on `test.npy`. **These are constant across
seeds** — they are a property of the frozen test set, recomputed identically every time — so
the aggregator prints `± 0` and a CI of `0`. **Print the zeros.** Explain them in prose. See
§7.0: hand-annotating this block as "(target) / —" is what erased a real deviation in
Experiment B.

**`novelty.*` — 3 keys**: `distinct_nearest_training_paths`,
`mean_standardized_nearest_train_path_rmse`, `median_standardized_nearest_train_path_rmse`.
This is the memorisation check. PDF §5 exempts it from the mean ± std treatment.

**`configuration.*` — 6 keys** (`annualization`, `history_cutoff`, `horizon`, `recent_window`,
`split`, `threshold`) and **`sources.*` — 3 keys** (`generated`, `test`, `train`, absolute
paths). Both blocks are **excluded from the README table** via
`--exclude-prefix configuration sources`. `sources.*` is nonetheless the audit trail that proves
which arrays the evaluator actually opened — read it, do not print it.

### 12.5 Experiment B — all 56 evaluator keys, by block

File: `results/new_experiments/experiment_B/<Method>/pdf_metrics/seed_<q>_heston_mixture.json`

**`mixture_fidelity.*` — the regime block.**

| Dotted key | PRIMARY? | Note |
|---|---|---|
| `mixture_fidelity.regime_proportion_tvd` | ★ **yes** | the headline mode-collapse number |
| `mixture_fidelity.generated_regime_proportions` | no | **list of 8** — not a scalar, do not put it in a mean ± std table |
| `mixture_fidelity.target_regime_proportions` | no | list of 8, constant across seeds |
| `mixture_fidelity.generated_mean_max_probability` | no | posterior confidence |
| `mixture_fidelity.generated_mean_posterior_entropy` | no | posterior confidence |
| `mixture_fidelity.generated_low_confidence_fraction` | no | fraction with `max_prob < 0.6` |
| `mixture_fidelity.target_mean_max_probability` | no | constant across seeds |
| `mixture_fidelity.target_mean_posterior_entropy` | no | constant across seeds |
| `mixture_fidelity.target_low_confidence_fraction` | no | **NOT constant** — std 8.63e-05. The one target row that moves. |

**`mixture_fidelity.parameters.<p>.*` for `p ∈ {theta, xi, rho}` — 6 keys each, 18 total**:
`mean_error`, `q05_error`, `q95_error`, `std_ratio`, `support_normalized_wasserstein`,
`wasserstein`. The `std_ratio` triple is the diagnostic that corroborates a TVD failure
independently — read §3.5, they are meant to be read together.

**`observable_fidelity.*` — 7 keys.** Two of the four PRIMARY metrics live here:

| Dotted key | PRIMARY? |
|---|---|
| `observable_fidelity.realized_volatility_wasserstein` | ★ **yes** |
| `observable_fidelity.leverage_curve_rmse_lags_0_20` | ★ **yes** |
| `observable_fidelity.abs_return_acf_rmse_lags_1_50` | no |
| `observable_fidelity.squared_return_acf_rmse_lags_1_50` | no |
| `observable_fidelity.excess_kurtosis_error` | no |
| `observable_fidelity.return_std_error` | no |
| `observable_fidelity.terminal_log_price_ks` | no |

**`novelty.*` — 3 keys**, same as A.

**`oracle_gate.*` — 14 keys**, and **all of them are constant across seeds and across methods**:
they describe the frozen oracle, not your model. `eight_regime_accuracy` (0.909423828125),
`num_features` (77), `train_paths` (32768), `validation_paths` (8192), `confusion_matrix`,
`validation_true_counts`, `validation_predicted_counts`, `parameter_state_accuracy.{theta,xi,rho}`,
and `configuration.{trees, max_features, min_samples_leaf, minimum_accuracy, random_state}`.
Excluded from the README table via `--exclude-prefix oracle_gate sources configuration`.
Quote `eight_regime_accuracy` **in prose** in §1 — a reader must know the oracle clears its own
0.9 gate before believing any TVD computed with it.

**`sources.*` — 4 keys** for B (`generated`, `oracle`, `test`, `train`); A has 3 (no `oracle`).

### 12.6 `W̄_param` — the one metric you must derive

The evaluator does **not** emit it. PDF §3.4 defines it as the mean over the three parameters
of the support-normalised Wasserstein distance. Compute it per seed, then aggregate:

```python
W_q = mean over p in {theta, xi, rho} of
      json["mixture_fidelity"]["parameters"][p]["support_normalized_wasserstein"]
```

then report `mean ± std` of the five `W_q`, with the same t(0.975, 4) = 2.776 CI. Derive it in
the README. **Do not add it to the evaluator** — that would violate §7 item 7.

It is the only derived quantity in either experiment. If you find yourself deriving a second
one, you have probably misread a key name.

### 12.7 The A1–A34 + B battery — a different suite, different files

The battery is **not** the PDF evaluator and answers a different question. It is produced by
`code/compute_metrics_experiment.py` (the §6.3 adapter around the canonical
`metrics/compute_all.py`) and lands in:

| File | Contents |
|---|---|
| `results/new_experiments/experiment_<X>/<Method>/seed_<q>_metrics.json` | one seed, all battery metrics |
| `results/new_experiments/experiment_<X>/<Method>/metrics_summary.csv` | the 5-seed aggregate, header `metric,mean,std,seed_0,seed_1,seed_2,seed_3,seed_4` |

`metrics_summary.csv` holds **103 metric rows** (measured with
`wc -l`, not counted by eye): `A1_kurtosis_error` … `A32_vol_of_vol_error`,
then `A33_sigma_corr` and `A34_sigma_rmse` (**null in both experiments** — dropped by Theo's
ruling; they land as null because the adapter's `load_data()` returns `v = None`), then the
`B_*` curve-shape family and `grid_tvd`.

Exact `A*` row names, in file order:

```
A1_kurtosis_error  A2_abs_r_q95_error  A3_abs_r_q99_error  A4_tail_qq_error
A5_hill_tail_index_error  A6_path_mmd2  A7_terminal_mmd2  A8_increment_mmd2
A9_volatility_mmd  A10_terminal_swd  A11_path_swd  A12_rv_law_loss
A13_mean_path_rmse  A14_ks_logreturns  A15_skewness_error  A16_qq_rmse
A17_terminal_ks  A18_disc_score_gru  A18_disc_score_mlp  A19_pred_score_gru
A19_pred_score_mlp  A20_cov_error  A21_acf_abs  A22_acf_sq
A23_acf_lag1_abs_error  A24_acf_lag1_sq_error  A25_mean_rmse  A26_std_error
A27_logreturn_std_error  A28_kurtosis_ratio  A29_sigma_mean_error
A30_vol_path_rmse  A31_rolling_vol_ks  A32_vol_of_vol_error
A33_sigma_corr  A34_sigma_rmse            ← null in both experiments
```

**A18/A19 have two variants each** (`_gru` and `_mlp`) — four rows, not two. The `B_*` family is
six curve types (`log_ret_hist`, `qq_plot`, `acf_abs_r`, `acf_sq_r`, `roll_vol_hist`,
`tail_surv`) × the suffixes `_funct`, `_der`, `_sec_der`, `_funct_pct`, `_der_pct`,
`_sec_der_pct`, `_funct_nrmse`, `_der_nrmse`, `_sec_der_nrmse`, `_funct_cvar90`,
`_funct_cvar95`.

**Read A18/A19 against §7.3.** A GRU discriminator near the floor while the distributional
metrics sit at 15–79× floor does not mean the model is fine; it means every individual path is
plausible and only the population is wrong. **A18/A19 are blind to mode collapse.**

### 12.8 The perfect floor — the denominator of every ratio

`results/new_experiments/experiment_<X>/perfect_floor/` holds the *same* file families as a
method directory, produced by the *same* code paths, from 5 true-DGP draws (seeds 1000–1004):

```
perfect_floor/
├── pdf_metrics/seed_{0..4}_<stem>.json              PDF evaluator, vs test.npy
├── pdf_metrics_validation/seed_{0..4}_<stem>.json   PDF evaluator, vs disc.npy
├── seed_{0..4}_metrics.json                         A1–A34 + B battery
├── metrics_summary.csv
└── plots/
```

It is method-neutral: compute it **once per experiment** and every method divides by it. If it
already exists, do not recompute it — and above all, **never "repair" the floor; verify it**
(§11.5). A floor that gets fixed can mask a real defect. The LS4 floor banks were checked and
already satisfied `S₀ == 100` exactly, being true-DGP draws.

### 12.9 Which command emits which README block

| README block | Emitted by | Reads |
|---|---|---|
| §1 table + §1.1 primary panel | `tools/aggregate_pdf_metrics.py` | `pdf_metrics/*.json` + `../perfect_floor/pdf_metrics/*.json` |
| §1.3 validation column | same tool with `--subdir pdf_metrics_validation` | `pdf_metrics_validation/*.json` |
| §2.1 A1–A34 | `tools/make_metrics_tables.py --table A` | `metrics_summary.csv` (model + floor) |
| §2.2 B curve-shape | `tools/make_metrics_tables.py --table B` | `metrics_summary.csv` (model + floor) |
| §1 figure | `tools/plot_experiment_figures.py --experiment <X>` | model bank + dataset |
| §3 figure | `tools/plot_stylised_facts.py --experiment <X>` | model bank + `test.npy` |
| §4 figure | `tools/plot_losses.py --model-dir .` | `losses/seed_*_losses.csv` |
| PDF §1.4 manifest (**not** a README section) | `tools/write_generation_manifest.py` | `weights/`, `generated_paths/` |

Exact flags for every tool:

```
aggregate_pdf_metrics.py     --model-dir --floor-dir --pattern --label --exclude-prefix --subdir
make_metrics_tables.py       --model-dir --floor-dir --label --table
plot_experiment_figures.py   --experiment --model-dir --seed --out --label
plot_stylised_facts.py       --experiment --model-dir --seed --label
plot_losses.py               --model-dir --title
write_generation_manifest.py --model-dir --experiment --source-revision --repair --hyperparameter-origin
check_method_layout.py       --root --experiment
```

**The `--exclude-prefix` lists differ between experiments** and are not interchangeable:

* A: `--exclude-prefix configuration sources`
* B: `--exclude-prefix oracle_gate sources configuration`

---

## 13. Worked example — the exact command sequence for a new method

Written with `<Method>` and `<X>` as the only placeholders. Run from the repo root unless a
step says otherwise. Python is `/home/tbasseras/gpu-venv/bin/python` — **not** `~/.cc-venv`.

```bash
REPO=/home/tbasseras/benchmark
PY=/home/tbasseras/gpu-venv/bin/python
M=<Method>; X=<A|B>
DATA=$REPO/dataset/Heston/new_experiments/experiment_$X
OUT=$REPO/results/new_experiments/experiment_$X/$M
PROTO=$REPO/dataset/Heston/new_experiments/protocol/experiments/scripts
```

**Step 0 — the reproduction gate (§5 Stage 0). Do not skip, and do not proceed without
Theo's explicit validation.** Reproduce the method's committed 8192×128 Heston result, record
the level reached (1–4) and the exact numbers, and confirm the figures reproduce the method's
**known defects**, not only its fits.

**Step 1 — skeleton.**
```bash
mkdir -p $OUT/{code/logs,generated_paths,losses,weights,plots,pdf_metrics,pdf_metrics_validation}
cp $REPO/results/new_experiments/experiment_A/LS4/code/compute_metrics_experiment.py $OUT/code/
# then edit ONLY the path constants in that copy
```

**Step 2 — check the machine is free (hard limits: 2 GPUs, 16 cores).**
```bash
nvidia-smi; htop     # if a GPU is busy or RAM > 50 %, ASK before launching
```

**Step 3 — train 5 seeds in exactly 2 lanes.** Env assignments must precede `taskset`, or bash
treats the assignment as the command and you get exit 127:
```bash
cd $OUT/code
setsid env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 taskset -c 0-7 \
  bash -c 'for s in 0 2 4; do '"$PY"' train_'"$M"'_experiment.py --data '"$DATA"' --experiment '"$X"' --seed $s --out '"$OUT"' > logs/exp'"$X"'_seed$s.log 2>&1; done' & disown
setsid env CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 taskset -c 8-15 \
  bash -c 'for s in 1 3; do '"$PY"' train_'"$M"'_experiment.py --data '"$DATA"' --experiment '"$X"' --seed $s --out '"$OUT"' > logs/exp'"$X"'_seed$s.log 2>&1; done' & disown
```
Within 60 s confirm: 2 PIDs, 2 GPUs holding memory, and the correct `[data]` path echoed in
**both** logs. Never let two lanes target the same seed.

**Step 4 — the A1–A34 + B battery, model then floor.**
```bash
cd $OUT/code
$PY compute_metrics_experiment.py --seeds 5
$PY compute_metrics_experiment.py --seeds 5 --source floor     # skip if the floor exists
```

**Step 5 — the PDF evaluator, per seed, unchanged, on BOTH sides.** For A:
```bash
for q in 0 1 2 3 4; do
  G=$OUT/generated_paths/seed_$q/generated_paths_8192x128.npy
  $PY $PROTO/evaluate_drawdown_memory.py --train-data $DATA/train.npy \
     --test-data $DATA/test.npy --generated-data $G \
     --dataset-manifest $DATA/manifest.json \
     --output $OUT/pdf_metrics/seed_${q}_drawdown_memory.json
  $PY $PROTO/evaluate_drawdown_memory.py --train-data $DATA/train.npy \
     --test-data $DATA/disc.npy --generated-data $G \
     --dataset-manifest $DATA/manifest.json \
     --output $OUT/pdf_metrics_validation/seed_${q}_drawdown_memory.json
done
```
For B, swap in `evaluate_heston_parameter_mixture.py`, drop `--dataset-manifest`, add
`--oracle $DATA/oracle/oracle.joblib --oracle-gate-report $DATA/oracle/gate_report.json`, and
write `seed_${q}_heston_mixture.json`.

⚠️ **Never check these runs with `grep -qiE "error|traceback"`.** The evaluator echoes JSON
containing `excess_kurtosis_error`, `return_std_error`, … and every job will look FAILED.
Verify by `json.load`-ing all 20 outputs instead.

**Step 6 — figures and the PDF §1.4 manifest.**
```bash
cd $OUT
$PY ../../tools/plot_losses.py --model-dir . --title "$M — Experiment $X"
$PY ../../tools/plot_stylised_facts.py --experiment $X --model-dir . --seed 0 --label $M
$PY ../../tools/plot_experiment_figures.py --experiment $X --model-dir . --seed 0 --label $M
$PY ../../tools/write_generation_manifest.py --model-dir . --experiment $X
```

**Step 7 — machine-check the layout. Must print PASS before any prose is written.**
```bash
$PY ../../tools/check_method_layout.py --root . --experiment $X
```

**Step 8 — prove the 5 trainings were independent and used this experiment's data.**
```bash
md5sum weights/seed_*_model.pt | awk '{print $1}' | sort -u | wc -l    # must be 5
for s in 0 1 2 3 4; do $PY -c "import json;d=json.load(open('weights/seed_${s}_config.json'));print(d['seed'],d['experiment'],d['data'])"; done
for s in 0 1 2 3 4; do tail -1 losses/seed_${s}_losses.csv; done        # 100 epochs, distinct
```
Then compare the fitted `scaler_mu` / `scaler_sigma` against the **other** experiment's. They
must differ. That is the one piece of evidence that does not require trusting the config file.

**Step 9 — the README.**
```bash
cp ../../README_TEMPLATE.md README.md
$PY ../../tools/aggregate_pdf_metrics.py --model-dir . --floor-dir ../perfect_floor \
   --pattern '*_drawdown_memory.json' --label $M --exclude-prefix configuration sources
$PY ../../tools/make_metrics_tables.py --model-dir . --floor-dir ../perfect_floor --label $M --table A
$PY ../../tools/make_metrics_tables.py --model-dir . --floor-dir ../perfect_floor --label $M --table B
```
Paste each table **verbatim**. Then re-run every `*Reproduce:*` line and require an empty diff
(§8 item 17). Re-read every number quoted in prose from its JSON, not from memory (E5).

**Step 10 — commit with an explicit pathspec**, because a concurrent process also commits to
this repo:
```bash
git add -- results/new_experiments/experiment_$X/$M
git diff --cached --name-only | grep -v "^results/new_experiments/experiment_$X/$M/"   # must be empty
```

### 13.1 Porting notes specific to CSDI

CSDI already exists in this repo at `methods/CSDI/` (`code/train_heston.py`,
`code/plot_losses.py`, plus `generated_paths/`, `losses/`, `weights/`,
`paper_reimplementation/`, `path_shadowing/`). Two consequences:

* **Stage 0 is cheap** — the committed 8192×128 Heston result is already there to reproduce
  against. Do it anyway, and record the level reached.
* `methods/CSDI/code/train_heston.py` is the starting point for
  `code/train_csdi_experiment.py`, but it must be **adapted to the §6.5 contract**, not copied:
  `--data` required and never defaulted, `--experiment` required, the firewall guard verbatim,
  and the five required per-seed outputs under the §6.2 names.
* CSDI is a **diffusion imputation** model working on log-returns. The §6.5 warning applies
  directly: the bank written to `generated_paths_8192x128.npy` must be inverted back to the
  **original price scale with S₀ = 100**, not left in log-return or standardised space. That
  bank loads fine and scores catastrophically. `check_method_layout.py` catches it —
  it asserts `S₀ == 100.0` and strict positivity on the array itself.
* `path_shadowing/` and `baseline_no_preproc/` **do not apply here** and must not be created
  under `results/new_experiments/`.
