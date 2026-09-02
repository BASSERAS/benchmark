# Deep-MKV-TS on Multi-Asset Heston — implementation notes

Companion to the generated method page [`../README.md`](../README.md). That page is
rendered from artefacts and reports *what happened*; this one is hand-written and
records *what was decided, and why*. Nothing here is auto-generated, so nothing here
may contradict a number on that page — where the two could drift, this file points at
the artefact instead of copying the value.

`MULTIASSET_GUIDELINE.md` §0 requires the four questions below to be answered in
writing before anything is run. They are answered first, then the parts specific to
this method: the frozen reference kernel, where the upstream package actually lives,
and the three deviations d = 8 forced on the d = 1 code.

---

## §0.1 — Does the method natively accept `(N, T, d)` input?

**Yes, and it is trained jointly.** One reference kernel with `state_dim = 8`, one
control network emitting a full `(d, d)` correction. This is **not** eight independent
univariate fits presented as a multi-asset model.

That distinction is the whole point of this dataset. The d = 8 Heston law carries
`d(d−1)/2 = 28` spot correlations, and a per-asset ensemble cannot represent them at
all: eight univariate generators produce eight marginally-correct, mutually
independent price series. `A20_cov_error` is the metric that detects exactly this, and
it is the reason the guideline insists it appear in the headline whatever it says.

Concretely, the joint structure lives in two places:

- the reference SDE's covariance is a full `(8, 8)` matrix built from 7 features
  (`covariance_feature_names` in `reference/reference_kernel.json`), not 8 scalars;
- the learned correction `Θ` is a symmetric `(8, 8)` matrix, and the spectral clip acts
  on its **eigenvalues** (see *Deviation 1* below).

---

## §0.2 — Does any hyperparameter fail to cross dimension?

**Yes: `ridge_lambda`, the Z-proxy ridge penalty. It is the one hyperparameter re-tuned
for d = 8, and it was re-tuned on the validation split.**

### Why it breaks

Algorithm 1 estimates the adjoint target `Z^proxy` by ridge-regressing the adjoint
moments onto `Φ_k^ref`, the flattened path prefix plus an intercept. That basis has
width

```
p(k) = 1 + (k + 1) · d
```

so it grows with the *dimension*, while the number of regression rows is the path batch
`B = 256` and does not.

| | `d = 1` | `d = 8` |
|---|---|---|
| widest `Φ^ref` (`k = T−1`) | 252 | **2017** |
| rows available (`BATCH_SIZE`) | 256 | 256 |
| underdetermined from step | never | **31** |

At d = 1 the regression is genuinely overdetermined for the whole path and behaves like
a regression: the committed d = 1 run reports `ce_projection_r2 ≈ 0.28`. At d = 8 with
the inherited `ridge_lambda = 1e-3` it is underdetermined from step 31 onward and
reports `ce_projection_r2 = 0.9335` — that is **not** a better fit, it is near
interpolation. The proxy stops denoising the adjoint moments and starts reproducing
their sampling noise, and the adjoint-matching loss is then chasing noise.

There is a second, purely mechanical symptom that confirms the diagnosis: with a
near-singular normal matrix, `cholesky_ex` on `DᵀD + λI` fails at every step and the
code falls back to `lstsq`, which accounted for **~44 % of wall-clock**. A hyperparameter
that changes the runtime by 44 % is not a cosmetic choice.

### How it was re-selected

On `heston_ma_S_val_8192x252x8.npy` — the **validation** split, per §0.2. The
`heston_ma_S_valdisc` bank is reported as a secondary cross-check and never decides.
(An earlier attempt scored on `valdisc` and was killed 7 minutes in and relaunched; the
old value was reproduced bit-for-bit at `0.1207937` before discarding it, so the
restart is auditable rather than merely asserted.)

Driver: [`sweep_ridge_lambda.py`](sweep_ridge_lambda.py), 250-step runs at full problem
size, one λ per process. It writes `sweep/lambda_<λ>.json` per run and
`sweep/winner.json` from `--report`.

**Pilot diagnostic (3 steps each, full size, single GPU)** — this is the measurement
that motivated the sweep, not the sweep itself, and it is reported as such:

| `ridge_lambda` | `ce_projection_r2` | `noise_ce_projection_r2` | s/step | projected h/seed |
|---|---|---|---|---|
| 1e-3 *(inherited from d = 1)* | 0.9339 | 0.9337 | 10.598 | 8.8 |
| 1e-1 | 0.9321 | 0.9317 | 6.251 | 5.2 |
| 1e+0 | 0.9169 | 0.9152 | 6.263 | 5.2 |
| 1e+1 | 0.8152 | 0.8036 | 6.223 | 5.2 |
| 1e+2 | 0.5306 | 0.4930 | 6.219 | 5.2 |
| 1e+3 | 0.2479 | 0.1893 | 6.225 | 5.2 |

Two things to read off it, and one not to:

- every λ ≥ 0.1 restores `cholesky_ex`, removing the `lstsq` fallback — hence the flat
  6.2 s/step column and the 41 % saving against 1e-3;
- λ ≈ 1e+3 is where the d = 8 projection r² returns to the d = 1 statistical regime
  (0.2479 vs 0.28). That is the principled tie-break if the discrepancy ranking turns
  out to be within noise;
- **do not read a winner off this table.** r² is a diagnostic of the proxy, not of the
  generator. The selection criterion is the path-functional discrepancy on the
  validation bank, over 250 steps, and it is recorded in `sweep/winner.json`.

**Selection result (250 steps, seed 0, full problem size).** Selection is on
`val_discrepancy` against `heston_ma_S_val_8192x252x8.npy` and nothing else; the
remaining columns are diagnostics, reported so the winner can be argued with:

| `ridge_lambda` | `val_discrepancy` ↓ **selects** | `valdisc` (2nd draw) | `ce_projection_r2` | `noise_ce_r2` | `final_loss` | s/step | h/seed |
|---|---|---|---|---|---|---|---|
| 1e-3 *(paper, d = 1)* | 0.036413 | 0.034553 | 0.9281 | 0.9140 | 19.36 | 12.345 | 10.29 |
| 1e+0 | **DIVERGED** | — | — | — | — | — | — |
| 1e+1 | 0.040114 | 0.038269 | 0.7831 | 0.7514 | 11.03 | 9.329 | 7.77 |
| 1e+2 | 0.028556 | 0.027214 | 0.4562 | 0.4345 | 3.94 | 6.850 | 5.71 |
| **1e+3** | **0.023836** | **0.023031** | **−0.3758** | 0.1681 | 1.19 | 6.850 | 5.71 |

**Winner: `ridge_lambda = 1000`**, recorded in `sweep/winner.json` together with
`candidates_compared` so a partial sweep cannot masquerade as a complete one.
`RIDGE_LAMBDA` in [`train_multiasset.py`](train_multiasset.py) is set to it, every
`weights/seed_*_config.json` records it under `ridge_lambda`, and `retuned_for_d8`
lists it — derived by comparing against `D1_RIDGE_LAMBDA`, never hand-asserted.
The secondary draw ranks the candidates identically, so the ordering is not an
artefact of one bank.

**The winner's r² is negative, and that was checked rather than waved through.**
`ce_projection_r2 = −0.3758` is worse than predicting the target's mean, and both it
and `final_loss` fall monotonically in λ (0.928 → 0.783 → 0.456 → −0.376; 19.4 → 11.0
→ 3.9 → 1.2). That is exactly the signature of the ridge coefficient collapsing,
`β → 0`. If it had collapsed, then `Z^proxy → 0`, the adjoint loss would be minimised
by `Ẑ → 0`, and `Θ = η(σ^ref)^{-1} + Ẑ/√Δt` would reduce to `σ = σ^ref` — the winner
would be the reference SDE with a training loop bolted on, and the sweep would have
ranked a null model first.

[`null_control_baseline.py`](null_control_baseline.py) measures that null model
directly. Algorithm 1 zero-initialises the final GRU layer, so a model that is built
but never fitted emits `Ẑ ≡ 0` exactly and samples the pure reference SDE. Scored on
the same bank with the same `score_seed`:

| | `val_discrepancy` | `valdisc` |
|---|---|---|
| zero-control floor (`Ẑ ≡ 0`, σ = σ^ref) | 0.300095 | 0.294279 |
| λ = 1000 winner | **0.023836** | 0.023031 |

**12.6× better than doing nothing**, so the hypothesis is falsified: λ = 1000 learns.
The negative r² says the *linear* proxy explains `A_{k+1}ε_{k+1}` badly at d = 8 —
which is what heavy shrinkage on a p = 2017 column basis against B = 256 rows should
do — not that the learning signal vanished. Recorded in `sweep/null_control.json`.

Two honesty notes on this table. First, the 3-step pilot above put λ = 1e+3 at
r² = 0.2479 and offered "returns to the d = 1 statistical regime (0.2479 vs 0.28)" as a
principled tie-break; at 250 steps the same λ sits at −0.3758. **The pilot's r² did not
survive contact with a longer run, and that tie-break argument is withdrawn** — the
winner rests on the discrepancy column alone. Second, `sec_per_step` was measured under
varying core contention and plays no part in selection.

### λ = 1 diverged, and the error message said something else

`ridge_lambda = 1` **did not produce a worse score — it produced no score at all.** Twice,
on seed 0, it died inside 250 steps with:

```
torch._C._LinAlgError: linalg.eigh: (Batch element 0): The algorithm failed to converge
because the input matrix is ill-conditioned or has too many repeated eigenvalues (error code: 1)
```

That message is wrong about the cause. Instrumenting the input to the spectral clip
([`eigh_fallback.py`](eigh_fallback.py)) showed `non_finite_entries = 16384` out of
`256 × 8 × 8 = 16384` — **every entry of every batch element was NaN**. cuSOLVER emits a
conditioning-flavoured message when handed NaN. The control had already diverged several
operations upstream; the eigensolver was the messenger.

The shim's rule is what caught it: if `Θ` is non-finite it raises rather than retrying on
another backend, because a NaN is an exploding control, not a backend problem, and a CPU
retry would either fail identically or return garbage inside a run that then looks
successful. The record is `sweep/lambda_1e0.json`, carrying `"diverged": true` — the
candidate is present in the sweep table as `DIVERGED`, not silently missing.

**Why this matters beyond dropping one candidate.** The divergence is *not monotone in λ*:
λ = 1 died while both neighbours, 1e-3 and 10, completed cleanly, and λ = 1's projection
quality was unremarkable (r² = 0.917, between its two neighbours). So this is not "small λ
is unstable" or "large λ is unstable" — it is a trajectory instability that one setting
happened to hit. It follows that **surviving a 250-step screen does not bound behaviour at
step 3000**, and the campaign was launched knowing that. See MULTIASSET_GUIDELINE.md §12.8.

The mitigation is detection and crash survival, not a change to Algorithm 1:

- `train_multiasset.py` flushes `losses/seed_{i}_losses.csv` on **every** logged step, so a
  crash leaves the full loss history rather than nothing.
- It writes a rolling `latest.pt` every `LOG_EVERY = 100` steps, via write-to-temp-then-
  rename so a checkpoint is never observed half-written. At most 100 steps of work is ever
  unrecoverable, against the 500 between selection checkpoints.
- `CHECKPOINT_STEPS` — the **selection grid** `select_checkpoint_multiasset.py` chooses
  from — is deliberately unchanged. Adding to it would change which checkpoints compete and
  therefore change the method. `latest.pt` is never a selection candidate.
- A non-finite loss or objective prints `[diverged] step=N` immediately, so a failure
  reports the step index instead of leaving a wall-clock crash time and a misleading
  eigensolver traceback.

None of this clips, jitters or rescales anything. On every finite step the arithmetic is
identical to the unguarded run.

### Caveat that goes in the record, not in a footnote

250 steps is **8 % of the 3000-step run**. The ranking therefore reflects early-training
behaviour, and a λ that wins at step 250 is not guaranteed to win at step 3000. A longer
screen was not affordable inside the compute budget below. This is a known weakness of
the selection, stated because §12.3 requires a screen's limits to be described rather
than quietly assumed away. The λ = 1 divergence above makes the point concrete rather
than theoretical.

Two further honesties about the sweep table:

- **`eigh_fallback` is absent from the λ = 1e-3 and λ = 10 records**, because those two runs
  launched before the counters were wired in. Absent is not the same as zero — but both
  ran *without* the shim installed, so any eigensolver failure would have killed them as it
  killed λ = 1. They completed, therefore they had none.
- **`sec_per_step` and `eta_3000_hours` are contended measurements**, taken with three
  candidates sharing 15 cores on one GPU, and λ = 1000 shared cores with a relaunched λ = 1
  for its last 12 minutes. They rank throughput only loosely and played no part in
  selection, which is decided solely on `val_discrepancy`.

### The caveat came true twice: 2 of 6 seeds diverged (33%)

The paragraph above predicts that "surviving a 250-step screen does not bound behaviour at
step 3000". It did not stay theoretical, and it did not happen once. With the **selected**
λ = 1000 — the winner, not a rejected candidate — **two of the six seeds launched died with a
non-finite control**:

| Seed | Died at | Elapsed before death | Checkpoints surviving |
|------|---------|----------------------|-----------------------|
| 3 | step 2 | 0.2 min | none |
| 1 | between step 1400 and 1500 | **255.8 min** | `step_0500.pt`, `step_1000.pt` |

**That is a 33% failure rate on the reported configuration**, and it is the single most
important operational fact about this method at d = 8. It is stated here, in a table, rather
than left to be inferred from a gap in the seed numbering.

**Seed 3 — immediate death.** It failed at step 2 with the identical signature the λ = 1
screen produced:

```
step=    1 loss=4.70082 objective=16.3114 grad=0.609 elapsed=0.2min
[eigh_fallback] cuSOLVER eigh failed: linalg.eigh: (Batch element 0): The algorithm failed
to converge because the input matrix is ill-conditioned or has too many repeated eigenvalues
[eigh_fallback] Theta: shape=(256, 8, 8), dtype=torch.float32,
                non_finite_entries=16384, bad_batch_elements=[0, 1, ..., 7]... of 256
RuntimeError: eigh failed AND Theta is not finite -- this is an exploding control, not a
cuSOLVER convergence problem. Do not retry on another backend.
```

`non_finite_entries = 16384` out of `256 × 8 × 8 = 16384`: every entry of every batch element
was NaN after **one** completed step. Full log: [`logs/train_seed_3.log`](logs/train_seed_3.log).

**Seed 1 — death after four hours of healthy training.** This is the more damaging of the two,
because seed 1 was not sick from the start. It ran 14 logged checkpoints and 255.8 minutes
before failing, and the run-up is visible in
[`../losses/seed_1_losses.csv`](../losses/seed_1_losses.csv):

```
step= 1200 loss=0.997 objective=2.400 grad=0.033   <- healthy
step= 1300 loss=1.705 objective=3.543 grad=0.190   <- objective +48%, grad 5.8x
step= 1400 loss=1.522 objective=2.644 grad=0.066   <- partial recovery, looks survivable
[eigh_fallback] Theta: non_finite_entries=16384, bad_batch_elements=[0..7] of 256
RuntimeError: eigh failed AND Theta is not finite -- this is an exploding control
```

Note the shape of it: a spike at step 1300, an apparent **recovery** at 1400, then total loss of
finiteness before 1500. The recovery is the trap — an early-stopping rule watching `objective`
would have seen the excursion end and concluded the run was fine, one checkpoint before it died.
Full log: [`logs/train_seed_1.log`](logs/train_seed_1.log).

**The instability is seed-dependent, not λ-dependent — and it is not an initialisation
problem.** Seed 4 was launched *simultaneously* with seed 3, same λ, same steps, same device,
same code — and survived, which already ruled λ out. Seed 1 rules out the remaining easy
explanation. A bad initialisation would show up immediately, as seed 3's did; seed 1 trained
stably for over four hours first. So this is a genuine **trajectory** instability: the iterate
can wander into a region where Θ blows up at *any* point in the run, and neither a λ screen nor
a short smoke test of any length bounds it. That is a stronger and worse claim than the one this
section made when only seed 3 had failed.

> **Survivorship bias, stated plainly.** Every score this method reports is conditioned on the
> event "this seed did not diverge". The mean ± std below are computed over survivors, so they
> describe the behaviour of Deep-MKV-TS *given that it trained to completion* — not its
> unconditional behaviour, which includes a 1-in-3 chance of producing nothing at all. No
> reweighting can fix this from 6 samples; the honest response is to report the failure rate
> next to the scores, which is what the table above does. A reader comparing this column to
> SBTS or LS4 should hold both numbers in view at once, because neither of those methods lost a
> seed.

**A restart would have replayed the crash exactly — this was asserted, then tested, and it is
false.** The original claim here was that training is fully deterministic in the seed
(`init_seed = int(seed)` for the network, `build_training(seed=args.seed)` for the data stream),
so re-running `--seed 3` had to reproduce the identical NaN, and therefore retrying was not an
option. The determinism half is correct. The conclusion drawn from it is not. See
["Why the clip did not save them"](#why-the-clip-did-not-save-them-a-post-hoc-diagnosis) below:
seed 3 was re-run three times and **survived every time**, with step 1 bit-identical to the
archived crash. The substitution decision stands — it was taken with the information available
at the time — but the stated justification for it was wrong and is corrected rather than
quietly deleted.

### Why the clip did not save them: a post-hoc diagnosis

The obvious objection to the two deaths is that `sigma_min = 1e-3` and `sigma_max = 0.6` exist
precisely to stop this. They do not, and the reason is structural rather than a matter of
tuning the two numbers.

**1. The clip runs *after* the `eigh` it is supposed to protect.** In
`project_theta_to_sigma` (`reference/src/deep_mkv_gen_path_dt/controls/specific_entropy_matrix.py`)
the order is:

```python
eigenvalues, eigenvectors = torch.linalg.eigh(_symmetrise(theta))   # <- the crash is HERE
lower, upper = eta / sigma_max, eta / sigma_min
clipped = eigenvalues.clamp(min=lower, max=upper)                   # <- the clip is HERE
```

The clip bounds the **output** `σ` *given* a finite `Θ`. It is one line too late to bound `Θ`
itself, and it never executes on the crashing call. `sigma_min`/`sigma_max` are a
well-posedness guarantee for the control map, not a stability guarantee for the trajectory —
they were never capable of preventing this failure mode, whatever values they take.

**2. It is not a gradual explosion.** `control_rms` is flat at ≈ 0.069 across all 14 logged
points of seed 1, including step 1400, and 0.068 for seed 3 at step 1. The `eigh_fallback`
message "this is an exploding control" describes the corpse; the control was not growing
beforehand.

**3. `Θ` cannot be infinite mathematically.** `spectral_sigma` is applied inside the reference
kernel (`multivariate_reference.py`), pinning `σ_ref`'s spectrum into `[1e-3, 0.6]`, so
`η/λ ≤ 1000` and the measured `σ_ref⁻¹` spectrum is `[1.67, 11.8]`. The other additive term is
`Ẑ/√Δt` with `1/√Δt = 15.87`; instrumented `|Ẑ|max ≈ 0.054` gives ≈ 0.86. Both terms of
`Θ = η σ_ref⁻¹ + Ẑ/√Δt` are bounded by construction with finite weights. An `inf` there is not
reachable from the mathematics of Algorithm 1.

**4. The weights were clean going into the fatal forward.** Seed 3's rolling `latest.pt` at
`step = 1` — the state one forward pass before the death, with no intervening optimizer step —
has **0 of 12 tensors non-finite** and `max|w| = 0.1021`. Nothing had corrupted the network.

**5. The degeneracy hypothesis was measured and rejected.** The candidate explanation was that
clipping is many-to-one, so it manufactures repeated eigenvalues, and `eigh`'s backward carries
`1/(μ_i − μ_j)` — which the frozen `project_theta_to_sigma` does *not* guard against, unlike
`multivariate_reference.py`'s own `_ClipSqrtEigh`. Live instrumentation says the clip is barely
active: **0–6 of 2048** `σ_ref` eigenvalues sit on a boundary, and **none** at `sigma_min`.

**6. The decisive experiment: the crash does not reproduce.** Seed 3 was re-run on GPU 3
(61 GB free) three times, writing to a scratch `--run-root` so `runs/seed_3/` was never touched:

| Run | Command | Step 1 | Outcome |
|-----|---------|--------|---------|
| instrumented, `--steps 2` | monkeypatched `theta_from_moments` | `4.70082` — bit-identical to archive | **step 2 survived**, `loss=12.7375` |
| clean, `--steps 2` | unmodified | bit-identical | **survived** (instrumentation exonerated) |
| clean, `--steps 3000` | byte-identical to the original launch | bit-identical | **still training at 5 min**, killed by `timeout` |

Same seed, same code, same data, identical step-1 loss to the last digit — and no crash. The
original death ran on **GPU 1 shared four ways**; the survivals ran on an almost-empty GPU 3.
There is no learning-rate schedule anywhere in `model.py` (`grep` for
`scheduler|LambdaLR|Cosine|OneCycle|warmup` returns nothing), so `--steps 2` and `--steps 3000`
follow the same optimizer trajectory — and the `--steps 3000` control was run regardless.

**Conclusion.** The two deaths are a **numerical event in the cuSOLVER eigendecomposition path
under GPU contention**, not a mathematical instability of Algorithm 1 and not a failure of the
spectral clip — which is architecturally incapable of preventing them either way. Two caveats
keep this from being a clean exoneration. Seed 1's death at step ~1450 was **not** re-tested
(255.8 min per attempt), so it is diagnosed only by analogy to seed 3's identical signature.
And "not reproducible on an idle GPU" is not "will not happen again": the failure rate observed
under real shared-machine conditions was 2 in 6, and that is the number a reader should plan
against. What changes is the *attribution* — this is a robustness gap in the eigensolver call
path, fixable with a degeneracy-safe backward or a finite-guard-and-retry, not evidence that
the method's control diverges.

**Decision: seed 3 was replaced by seed 5, then seed 1 by seed 6.** The campaign therefore
reports **N = 5 seeds, `{0, 2, 4, 5, 6}`** — six launched, two dead, five reported. This is a
real methodological cost and is recorded rather than smoothed over:

- The alternative — reporting N = 4 — would have made this the only method on the dataset with
  a different seed count, which silently changes every ± std against SBTS, LS4 and CSDI.
- Substitution keeps the seed count comparable but means the seed set is **not** the
  pre-registered `0-4`. Nothing was re-selected on the basis of either failure; λ = 1000 was
  already chosen, on seed 0, before the campaign launched, and no hyperparameter was touched
  after a seed died. Changing λ or the learning rate in response to a divergence would have
  contaminated the selection already performed on the surviving seeds — so the replacements ran
  with the byte-identical command line, verified against `/proc/<pid>/cmdline` of a live seed
  before launch.
- **Pairing consequence:** [`../../reference/`](../../reference) sampled generation seeds
  `90000 + i` for `i ∈ {0,1,2,3,4}`. This method now reports `{0, 2, 4, 5, 6}`, so the paired
  comparison — same Brownian stream, difference free of simulation noise — holds for the
  **intersection, `{0, 2, 4}`, three seeds**. Seeds 5 and 6 have no reference counterpart;
  seeds 1 and 3 have no Deep-MKV-TS counterpart. The reference page says so rather than
  implying five paired columns.
- A method that needs *two* substituted seeds to fill a five-column table is a method with a
  stability problem. That belongs in the record next to the score, not in a footnote under it.

**What the substitution required in code — and what it deliberately did not.** The seed set is
non-contiguous, so `run_all_multiasset.py`, `select_checkpoint_multiasset.py`,
`collect_artifacts.py` and `plot_losses.py` all carry `[0, 2, 4, 5, 6]` explicitly instead of
`range(5)`. `plot_losses.py` additionally indexes its colour palette by **position**, not by
seed id, because seed 6 would otherwise run off the end of a five-entry list. The shared scorer
needed one **additive** flag:

```bash
metrics/compute_all_multiasset.py --method Deep-MKV-TS --seed-list 0,2,4,5,6
```

`--seed-list` defaults to `None`, in which case the scorer falls back to `range(--seeds)` — the
exact code path every other method was scored on. That default is the point: SBTS, LS4, CSDI,
`reference/` and `perfect_recovery/` re-run **bit-identically** after the flag was added, so
none of their published numbers moved to accommodate this method's failure. The columns
`metrics_summary.csv` emits read `seed_0, seed_2, seed_4, seed_5, seed_6` because the header is
built from each result's own `seed` field, not from a counter — the two gaps are visible in the
CSV itself rather than renumbered away.

The one thing **not** done: seed 5 was not relabelled "seed 3", nor seed 6 "seed 1". That would
have produced a table indistinguishable from a clean `{0..4}` campaign, which is precisely the
information a reader needs and would not have had.

**Crash survival was already in place and did its job, twice.** `losses/seed_3_losses.csv` holds
the one logged step and `losses/seed_1_losses.csv` the fourteen logged before its death at
~step 1450; both failures printed a step index rather than a bare wall-clock time and a
misleading eigensolver message, which is exactly what `eigh_fallback`'s finite-check exists
for. Nothing was clipped, jittered or rescaled to make either seed survive — that would have
changed the method to hide a property of the method.

### The seed lock — two trainers on one seed

Relaunching seeds by hand into a partly-running campaign risks starting a **second** trainer on
a seed that is already training. Both would write the same `weights/seed_{i}/` checkpoints and
the same `losses/seed_{i}_losses.csv`; the result is not a crash but something worse — a bank
that looks complete and is interleaved garbage.

`_acquire_seed_lock(run_dir, seed)` runs once at startup, before a single tensor is allocated:

- writes the owner PID to `run_dir/.lock`, and prints `[lock] seed {i} owned by PID {pid}`;
- if the file already exists, tests the recorded PID with `os.kill(pid, 0)` and **refuses to
  start** if it is alive, naming the owning PID in the error;
- if the PID is dead, reclaims the lock and says so — a crashed run must not leave a landmine
  that blocks every future attempt, which is the usual failure mode of naive lock files;
- releases via `atexit`.

It is a startup guard, not a runtime one: it writes nothing during training and cannot affect
the arithmetic. The line `[lock] seed 3 owned by PID 3605874` in `logs/train_seed_3.log` is the
guard confirming sole ownership moments before that seed diverged.

---

Everything else — `eta = 1.0`, `hidden_dim = 96`, `num_layers = 1`, `lr = 2e-3`,
`sigma_min/max`, `lambda_scale`, `kappa_scale`, the discrepancy preset and its ACF
weights — is carried over from d = 1 unchanged. There is **no learning-rate scheduler
anywhere in the codebase**, at either dimension.

---

## §0.3 — What is the memorisation risk?

The NN-ratio diagnostic ships regardless: [`measure_memorisation.py`](measure_memorisation.py)
writes `../losses/memorisation.json`. The estimator is byte-identical to
`SBTS/code/measure_memorisation.py` and `LS4/code/measure_memorisation.py`, because
otherwise the three columns would not be comparable.

The honest statement of the risk is specific to Algorithm 1, and it is neither "high"
nor "zero":

- **At generation time the method never touches a training path.** `model.sample` rolls
  the fitted drift and control forward from `x0 = log(100)` with fresh Gaussian
  increments. Contrast SBTS, which is an explicit kernel average *over training paths*,
  so memorisation is its default failure mode.
- **But it is not a pure likelihood model either.** During *training* the training bank
  enters twice: through the path-functional discrepancy, and through the ridge Z-proxy,
  whose design matrix `Φ^ref` is a flattened prefix of the sampled path. Memorisation
  would have to arrive through the fitted control network's weights.
- That is a **capacity argument** — the parameter count (recorded as `n_parameters` in
  `weights/seed_0_config.json`) against 8192 × 252 × 8 ≈ 16.5 M training values — and
  "implausible" is an argument, not a measurement. Hence the script.

**The exact-duplicate count is near-uninformative here and must not be read as
novelty.** A sample is a 251-step Euler–Maruyama rollout driven by fresh Gaussian noise,
so bitwise equality with a training path has probability zero *by construction*. The
**ratio** is the number that matters.

---

## §0.4 — Compute budget

Estimated before launch, as required. The λ-sweep row is **measured**; everything
below it is projected from the winner's measured `sec_per_step`.

| Stage | Unit cost | Total |
|---|---|---|
| λ sweep, 250 steps × 5 values (+1 diverged retry) | 2 waves, 3-up then 2-up on one GPU | ~3.5 h *(actual)* |
| Zero-control floor, `null_control_baseline.py` | sample + score only, no fit | ~3 min *(actual)* |
| Training, 3000 steps × 5 seeds | **5.71 h/seed** at λ = 1000 (6.850 s/step) | ~11.5 h (3 GPUs, 2 waves of 3 + 2) |
| Checkpoint selection, 6 checkpoints × 5 seeds | minutes | < 1 h |
| Generation, 8192 × 252 × 8 × 5 seeds | minutes/seed | < 1 h |
| Metrics, `compute_all_multiasset.py` | ~11 min/seed | ~55 min |
| Memorisation + 4 figures + render | minutes | < 1 h |

**≈ 15 h remaining wall-clock from campaign launch**, dominated by training.

Two corrections to earlier figures in this file, kept rather than overwritten so the
drift is visible. The 3-step pilot projected **5.2 h/seed**; the 250-step sweep measured
**5.71 h/seed** for the winner — the pilot was optimistic by ~10 %, and by ~50 % for the
inherited λ = 1e-3, which measured 12.345 s/step (10.29 h/seed) against a projected 8.8.
**A 3-step timing does not amortise warm-up.** Second, `sec_per_step` was measured with
two or three processes sharing a single GPU; the campaign spreads one process per GPU
across three cards, so the real rate should be at or below this, but two of those cards
are shared with other work on this machine and the figure is an estimate, not a promise.

Hardware: **GPUs 1, 2 and 3 — GPU 0 is off limits by standing instruction**, and
[`run_campaign.sh`](run_campaign.sh) refuses to launch if `0` appears in `GPUS` rather
than trusting the default. One process per seed, `CUDA_VISIBLE_DEVICES=<gpu>` with
`--device cuda:0`, pinned `taskset -c 0-4` / `-c 5-9` / `-c 10-14` at
`OMP_NUM_THREADS=5` — 3 × 5 = 15 cores, inside the 16-core cap. Seeds run **3-up in
waves, never 2-per-GPU**: wave 1 = seeds 0,1,2; wave 2 = seeds 3,4.

---

## The frozen reference SDE

Deep-MKV-TS does not learn a generator from noise. It starts from a *frozen,
interpretable reference SDE* and learns a **volatility correction only** — the drift is
never touched by the control.

The reference is **Guyon–Lekeufack** (paper §2.1), fitted by penalised maximum
likelihood with an 80/20 split, by [`fit_reference_multiasset.py`](fit_reference_multiasset.py).
Its coefficients are frozen in [`reference/reference_kernel.json`](reference/reference_kernel.json)
and are **never refitted per seed** — all 5 seeds share one reference, so cross-seed
spread measures the control, not the reference.

Read from that file (do not re-type these; they are quoted here for orientation only):

| Field | Value |
|---|---|
| `state_dim` | 8 |
| `num_steps` / `dt` | 251 / 1/252 |
| `trend_half_lives`, `activity_half_lives` | 2 each |
| `activity_update` | `structural_variance` |
| `covariance_feature_names` | 7 |
| `drift_feature_names` | 5 |
| `calibration_nll` / `validation_nll` | −10.3362 / −10.3847 |
| `fit.validation_fraction` | 0.2 |
| `fitted_at` | 2026-08-26T00:54:11 |

Validation NLL below calibration NLL is the expected sign here and not a bug: the split
is over paths, and the held-out fifth happens to be marginally easier. The gap is small
enough to be sampling noise at 8192 paths.

**Open question, recorded rather than hidden.** The 7 covariance features and 5 drift
features are the set used by the upstream package's own multivariate smoke test
(`smoke_matrix_control_multivariate.py`). That is strong evidence they are the intended
`φ^ref` at d > 1, but it is evidence, not confirmation from the paper. The kernel is
already fitted and frozen, so revisiting the set would invalidate the artefact and every
number downstream of it. Flagged here so the next reader does not mistake silence for
certainty.

---

## Where the upstream package actually lives

**It is not vendored into this directory.** `train_multiasset.py` line 124 reads:

```python
REFERENCE = Path("/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference")
```

so the `deep_mkv_gen_path_dt` package is imported from the d = 1 method tree, by
absolute path, with **zero edits**. `results/.../Deep-MKV-TS/code/reference/` contains
only two fitted artefacts — `reference_kernel.json` and `reference_fit_history.csv` —
not the library.

This is deliberate: the d = 1 reproduction matched the paper's Table 1 on 5/5 metrics
using that package unmodified, and forking it for d = 8 would mean the two dimensions no
longer share an implementation. Every d = 8 change is therefore **method-local**, in this
directory.

The cost is that the absolute path is **not portable**. A checkout at a different root
must edit that one line. It is named here rather than left to be discovered by an import
error.

---

## The three deviations d = 8 forced

Each of these is a place where the d = 1 code had a choice that did not exist at d = 1,
so the d = 1 behaviour is not evidence about which is right.

### 1. Matrix-valued control, not diagonal — `control = "matrix"`

The paper clips the **eigenvalues** of `Θ`:

```
σ ← U diag( 1 / clip_[1/σmax, 1/σmin](θ_i) ) Uᵀ
```

A diagonal control clips the diagonal **entries** instead. The two coincide bitwise at
d = 1 and diverge everywhere else, so the d = 1 run cannot distinguish them — and the
diagonal variant cannot represent a correction to the off-diagonal correlation
structure, which is the one thing d = 8 exists to test.

`Θ` is symmetric but **indefinite**, so Cholesky and LDLᵀ are invalid; `torch.linalg.eigh`
is the correct primitive. Its adjoint is the Daleckii–Kreĭn / Löwner form
`dF = U [Λ ∘ (Uᵀ dA U)] Uᵀ` with `Λ_ij = (f(μ_i) − f(μ_j)) / (μ_i − μ_j)`, degenerating to
`f'(μ_i)` on the diagonal. Implemented in
[`matrix_control_multiasset.py`](matrix_control_multiasset.py).

### 2. Batched-eigh chunking — `MAX_EIGH_BATCH = 32768`

cuSOLVER's batched symmetric eigensolver has a hard batch limit near 64 k: 64 256
matrices work, 65 536 fail with `CUSOLVER_STATUS_INTERNAL_ERROR`. MAGMA is deprecated in
torch 2.13 and silently redirects to cuSOLVER, so switching backend is not an escape.

`BATCH_SIZE = 256 × 251 Euler steps = 64 256` sits **inside the failure band**. The rollout
therefore chunks the eigendecomposition at 32 768 matrices.

**This changes throughput, not results.** `eigh` is applied independently per matrix;
chunking only alters how many are dispatched per kernel launch.

### 3. The drift Jacobian `∂f/∂x` is kept — `differentiable_sigma = True`

Dropping it would be cheaper and is a common shortcut. It is **not** taken here: the
adjoint of Algorithm 1 carries the derivative of the drift with respect to the state,
and discarding it silently changes the optimisation problem into a different one that
still trains and still produces plausible curves. `drift_adjoint_backend =
"autograd_replay"` computes it by replaying the forward pass on detached paths.

---

## What each file does

| File | Role |
|---|---|
| `fit_reference_multiasset.py` | Fits the frozen GL reference → `reference/reference_kernel.json`. **Run once.** |
| `multivariate_reference.py` | The d > 1 reference kernel: features, covariance assembly, drift. |
| `matrix_control_multiasset.py` | Symmetric-eigenvalue spectral clip + its Daleckii–Kreĭn adjoint. |
| `sweep_ridge_lambda.py` | §0.2 λ screen on the **validation** bank; `--report` writes `sweep/winner.json`. |
| `train_multiasset.py` | Algorithm 1. Writes `weights/seed_*_{model.pt,config.json}`, `losses/seed_*_losses.csv`. |
| `select_checkpoint_multiasset.py` | Scores the 6 checkpoints on validation, names `selected_step`. |
| `run_all_multiasset.py` | Rolls the selected checkpoint → `generated_paths/seed_*/`, applies the §4 `S0` rescale. |
| `collect_artifacts.py` | Rebuilds `losses/generation_time.csv` from per-seed metadata; **enforces the §4 contract, exit code is a gate**. |
| `measure_memorisation.py` | §9 NN-ratio → `losses/memorisation.json`. |
| `plot_losses.py` | `plots/loss_convergence.png`. Panels keyed on **column**, not `phase` — see its docstring. |
| `plot_diagnostics_multiasset.py` | `plots/heston_diagnostics.png` (asset 0). |
| `render_readme.py` | Generates `../README.md`. **Never hand-edit that file.** |
| `test_render_readme.py` | §8.9 structural test of the renderer against synthetic fixtures. |
| `eigh_fallback.py` | Diagnostic wrapper on the spectral clip: reports `Θ`'s state when `eigh` fails, **raises** if `Θ` is non-finite, retries on CPU LAPACK only if it is finite. Frozen package unmodified. |
| `null_control_baseline.py` | Scores a model that is **built but never fitted** (`Ẑ ≡ 0`, so σ = σ^ref) → `sweep/null_control.json`. The floor any candidate must beat to have learned anything; used to test whether the winner's negative `ce_projection_r2` meant a collapsed proxy. It did not. |
| `run_sweep.sh` | Launcher for the λ sweep waves. |
| `run_campaign.sh` | Launcher for the 5-seed campaign. **Refuses to start** unless `sweep/winner.json` covers all 5 candidates *and* matches `train_multiasset.RIDGE_LAMBDA`; refuses GPU 0 outright; writes `weights/.campaign_complete` only if every seed succeeded. |
| `_smoke_reference.py` | Scratch check of the reference kernel. Not part of the pipeline; retained only because deletion was blocked. |

`logs/`, `sweep/` and `runs/` hold run output and are not inputs to anything.

---

## Order of operations

The chain is not optional and each step consumes the previous step's artefact:

```
fit_reference_multiasset.py        (once)
  → sweep_ridge_lambda.py          (once, sets RIDGE_LAMBDA)
    → train_multiasset.py          (×5 seeds)
      → select_checkpoint_multiasset.py
        → run_all_multiasset.py
          → collect_artifacts.py   ← gate: must emit 5 rows, wire its exit code
            → metrics/compute_all_multiasset.py
              → measure_memorisation.py + the 4 plot scripts
                → render_readme.py
```

`collect_artifacts.py` is a gate for a specific reason: `compute_all_multiasset.py`
costs ~55 minutes and will produce entirely normal-looking numbers on a malformed
array. Catch the violation before paying for it.

The exact runnable commands, with the correct interpreter and `CUDA_VISIBLE_DEVICES` per
line, are in the **Reproduce** section of [`../README.md`](../README.md).
