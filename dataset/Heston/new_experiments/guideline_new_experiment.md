# Guideline — Blinded Synthetic Path-Law Benchmark (Experiments A & B)

Single source of truth for **Experiment A** (Delayed Drawdown Memory) and
**Experiment B** (Balanced Heston Parameter Mixture).

This file covers three things:

1. **What the two data-generating processes are** and exactly how their datasets were built (§2, §3).
2. **The methodology we followed** — the mandatory reproduction gate first, then the step-by-step
   run of each experiment (§5). *Read this before running anything.*
3. **What a new method must produce** — file layout, JSON/CSV schemas, README specification,
   and a copy-paste checklist (§6, §7, §8).

The root [`GUIDELINE.md`](../../../GUIDELINE.md) governs the *original* 8192×128 Heston benchmark.
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

### The other four tools

```bash
# README § 2 — the A1–A34 and B curve-shape tables, from metrics_summary.csv
python make_metrics_tables.py --model-dir ../experiment_A/LS4 \
    --floor-dir ../experiment_A/perfect_floor --label LS4 --table A     # then --table B

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

- `losses/loss_convergence.png` — 5 seeds overlaid.
- `plots/diagnostics_seed0.png` — `metrics/plot_diagnostics.py::plot_diagnostics(S_real, S_gen, method, seed, out_path)`.
  Its Heston-theory third curve is **wrong for these DGPs**; it is drawn inside a try/except and
  falls back to `TB = None`. If it appears, suppress it or state in the caption that it does
  not apply.
- Experiment-specific: A → `future_rv` distributions split by `early_hit`, real vs generated.
  B → the two 8-bar regime-proportion histograms, target vs generated.

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
├── guideline_new_experiment.md          ← this file
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
├── tools/                                       method-neutral, both experiments (§4)
│   ├── aggregate_pdf_metrics.py                 README § 1 table: 5-seed mean ± std + 95% CI
│   ├── make_metrics_tables.py                   README § 2 tables: A1–A34 and B curve-shape
│   ├── plot_experiment_figures.py               README § 1 figure: memory (A) / mixture (B)
│   ├── plot_stylised_facts.py                   README § 3 figure: heston_diagnostics.png
│   └── plot_losses.py                           README § 4 figure: loss_convergence.png
└── experiment_<A|B>/
    ├── perfect_floor/                           method-neutral sibling
    │   ├── pdf_metrics/seed_{0..4}_*.json       floor scored by the PDF evaluator
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
        │   └── metadata.json
        ├── weights/
        │   ├── seed_N_model.pt
        │   └── seed_N_config.json
        ├── losses/
        │   ├── seed_N_losses.csv
        │   └── loss_convergence.png
        ├── pdf_metrics/
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

---

## 7. README Specification

Every `results/new_experiments/experiment_<X>/<Method>/README.md` has these **five sections,
in this order**. The order is not negotiable — the PDF metrics come first because they are
what the protocol actually asks.

### § 1 — PDF metrics *(first, and standalone)*

One table with **every** metric defined in the PDF for this experiment, aggregated over the
5 seeds from `pdf_metrics/seed_N_*.json`.

- Experiment A: the `target_memory` / `generated_memory` pair, all 9 `errors`, all 3 `novelty`.
- Experiment B: all of `mixture_fidelity` (incl. per-parameter `std_ratio`),
  `observable_fidelity`, `novelty`, and the `oracle_gate` accuracy.

**Two tables, in this order.**

1. The full flattened table emitted verbatim by `aggregate_pdf_metrics.py`:
   `Metric | <Method> (mean ± std) | 95% CI half-width | Perfect floor (mean ± std)`.
   Every numeric leaf appears, including the `target_*` rows — those are the target, measured
   on `test.npy`, so they carry `± 0` by construction. Write `(target)` in their mean cell and
   `—` in their CI cell rather than printing a meaningless `± 0 | 0`.
2. A short **headline** table, hand-written, for the 4–6 metrics that decide the experiment:
   `Quantity | Target | <Method> | Perfect floor | Verdict`. This is the one a reader actually
   reads. Interpret it in prose beneath (§2.5 for A, §3.5 for B).

Do not fold the target into a column of table 1. The evaluator emits `target_*` and
`generated_*` as sibling blocks; flattening them into one row per metric requires a hand-kept
key mapping that silently rots the first time the evaluator gains a field.

### § 2 — Metrics A1–A34 + B, mean ± std across 5 seeds

From `metrics_summary.csv` via `make_metrics_tables.py --table A` and `--table B`, with the
perfect-floor column beside it. A33/A34 → `n/a`, with the one-line reason. Never delete the rows.

⚠️ Metric labels contain literal pipes (`A2 |r| q95`, `ACF of |log-returns|`). They must be
escaped `\|` or the whole Markdown table renders misaligned on GitHub. `make_metrics_tables.py`
does this; if you hand-edit a row, do it too.

**This section must not restate § 1's numbers, and § 1 must not borrow § 2's.** They are separate
suites answering separate questions, and the interesting result is usually the *disagreement*
between them — for LS4 on Experiment A, the standard battery's A18/A19 sit at the perfect floor
while the protocol evaluator scores the model at 59 %. Blending the sections destroys that signal.

### § 3 — Stylised Facts Diagnostic (real vs `<Method>`, seed 0)

`plots/heston_diagnostics.png` (from `plot_stylised_facts.py`) + a short honest reading of where
the model deviates. The black Heston-theory curve is suppressed by that tool because neither DGP
is single-regime Heston — say so explicitly, so a reader does not think it was forgotten.

### § 4 — Losses

`losses/loss_convergence.png` + per-seed final/min loss and wall-clock time.

### § 5 — File layout

The tree from §6.2. **No path-shadowing section** — it is not part of this protocol.

### Aggregation rules (apply everywhere)

- 5 seeds `q ∈ {0,1,2,3,4}`, **mean ± sample std** (`ddof=1`).
- 95 % CI = `mean ± 2.776 · std / sqrt(5)` (`t_{0.975,4} = 2.776`).
- **Failed seeds are reported, never silently replaced.** State which seed failed and why
  (NaN loss, OOM, divergence) and aggregate over the survivors with the reduced `n` written down.
- A model is "at the floor" on a metric iff its 95 % CI overlaps the floor's. Say so.
- Never compare a raw metric to 0. Compare to the floor.

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
- [ ] 15. Write `README.md` with the 5 sections in order, PDF metrics first.
- [ ] 16. Check every `metadata.json` for `gen_has_nan: false` and `first_nan_epoch: null`,
      then commit and push.

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
| 3 | `manifest.json` float fields can differ in the **last ULP** across numpy versions. Compare with `np.allclose`, not `==`. Shapes, seeds and `regime_counts` are exact and must match. | expected |
| 4 | `oracle.joblib` (~554 MB) **cannot be byte-reproduced** across scikit-learn/joblib versions even with `random_state=42`. The **gate accuracy** is the reproducible artefact; `gate_report.json` is committed, the blob is gitignored. Refit locally if missing. | expected |
| 5 | Perfect-floor seeds are **1000–1004** (`IND_SEED_BASE = 1000` from `metrics/compute_perfect_recovery.py`), disjoint from protocol seeds 0/1/2/100/101. | convention |
| 6 | `plot_diagnostics.py` draws a **Heston-theory** curve that is invalid for both new DGPs. It is inside a try/except (`TB = None` on failure). Suppress it or caption it as non-applicable. | known |
| 7 | **No Git LFS** (decision reversed 2026-07-30). Add no new LFS entries. The 8192×128 `float64` banks are ~8 MB each and **are committed normally**; only artefacts over GitHub's 100 MiB per-file cap get an explicit `.gitignore` entry — currently just `oracle.joblib`. | policy |
