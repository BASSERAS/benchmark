# Deep-MKV-TS with an SBTS reference — source and implementation notes (d = 8)

Deep-MKV-TS trained against a **reference pair taken from SBTS** rather than from a fitted
model: the reference drift `b^ref` is the SBTS Markovian kernel average over the training
bank, and the reference diffusion `sigma^ref` is the corrected constant matrix that the SBTS
scaling trick implies.

> Deep-MKV-TS: the frozen reference package
> `methods/Deep-MKV-TS/code/reference`, benchmark commit
> `0e09c6373fca9343a3fc4eb066ab2643255c99dc` (2026-08-16).
>
> SBTS: Alouadi, Barreau, Carlier & Pham, *Schrödinger Bridge Time Series Generation*,
> ICAIF 2025 — [arXiv:2503.02943](https://arxiv.org/abs/2503.02943)

Generated paths, metric tables and figures are on the method page
[`../README.md`](../README.md). Dataset law, metric scoping and the perfect-recovery floor are
in [`../../oldreadme.md`](../../oldreadme.md). Requirements for adding a method are in
[`../../MULTIASSET_GUIDELINE.md`](../../MULTIASSET_GUIDELINE.md).

---

## Files

| File | Role |
|------|------|
| `sbts_reference.py` | the reference pair: SBTS kernel `b^ref`, its analytic Jacobian, constant `sigma^ref` |
| `matrix_control_multiasset.py` | matrix-valued specific-entropy control at d = 8 |
| `eigh_fallback.py` | batched symmetric eigendecomposition with a CPU fallback |
| `train_multiasset.py` | training, 5 seeds |
| `sweep_hyperparams.py` | one sweep arm; writes `sweep/<stage>__<tag>.json` |
| `tabulate_hfix.py` | tabulates the corrected-adjoint bandwidth re-sweep and names its winner |
| `select_checkpoint_multiasset.py` | validation checkpoint selection |
| `run_all_multiasset.py` | generation, 5 seeds × 8192 × 252 × 8 |
| `collect_artifacts.py` | §4 generated-paths contract gate |
| `measure_memorisation.py` | NN-ratio diagnostic |
| `plot_diagnostics_multiasset.py`, `plot_losses.py` | figures |
| `render_readme.py` | regenerates `../README.md` from the artefacts |
| `run_resweep_h.sh`, `run_auto_campaign.sh`, `run_post_campaign.sh`, `run_pipeline.sh` | unattended drivers |
| `probe_dbref_dx.py`, `probe_jacobian_tail.py`, `probe_dc.py`, `validate_sbts_reference.py` | correctness and magnitude probes |
| `probe_occupancy_low.py` | kernel occupancy over the whole `h` grid; writes `sweep/occupancy_k20.json`, the single source of the `alive%` / `n_eff` columns |

Everything runs under `/home/tbasseras/gpu-venv/bin/python` with
`PYTHONPATH=methods/Deep-MKV-TS/code/reference/{src,experiments}`.

---

## The four questions (MULTIASSET_GUIDELINE §0)

**1. Does the method natively accept `(N, T, d)` input?**
Yes — **one joint model**. The GRU consumes the full 8-dimensional state, the control output is
a full 8 × 8 matrix, and `b^ref` is evaluated on d-dimensional increment vectors. Nothing here
is a per-asset ensemble: the cross-asset dependence is carried end to end, in the reference and
in the learned correction.

**2. Does any hyperparameter fail to cross dimension?**
**Two do, and for different reasons.**

*The SBTS bandwidth `h`.* Same radial-kernel argument as SBTS itself — support is a ball, so `h`
must scale with the typical distance between d-dimensional increments. But the value cannot
simply be inherited from the SBTS method page, because here `h` also controls a **gradient**,
not only a forward average. At `K = 20` the joint support over all 20 lags is far thinner than
the single-lag support that sets SBTS's own choice. Measured median effective bank paths
`n_eff = 1/Σ p²` at step 120, bank M = 4096, 256 disjoint held-out queries:

| K \ h | 0.36 | 0.50 | 0.70 | 1.00 | 1.50 | 2.00 |
|---|---|---|---|---|---|---|
| 1 | 3054.2 | 3740.8 | 3994.0 | 4070.7 | 4091.0 | 4094.4 |
| 5 | 778.8 | 2312.4 | 3449.4 | 3919.3 | 4059.4 | 4084.2 |
| **20** | **16.7** | 378.3 | 1796.1 | 3211.6 | 3876.2 | 4021.4 |

At `(h = 0.36, K = 20)` the "conditional expectation" is an average over ~17 of 4096 paths, and
below `h ≈ 0.2` every weight underflows and `b^ref` is identically zero — a model trained there
is fitting a reference drift of exactly 0, which is not a bad bandwidth but a meaningless run.
`h` was therefore **re-swept against the corrected adjoint** on the validation split, three
seeds per bandwidth, 250 steps, 27 arms:

| `h` | alive % | median `n_eff` | seed 0 | seed 1 | seed 2 | mean | spread |
|---|---|---|---|---|---|---|---|
| 0.28 | 83.2% | 1.9 | 0.133921 | 0.142768 | 0.143091 | 0.139927 | 6.6% |
| 0.31 | 93.0% | 3.9 | 0.082450 | 0.082559 | 0.080786 | 0.081932 | 2.2% |
| 0.33 | 95.7% | 6.7 | 0.092517 | 0.093699 | 0.091379 | 0.092532 | 2.5% |
| **0.36** | 98.8% | 16.7 | 0.074028 | 0.069655 | 0.067299 | **0.070327** | 9.6% |
| 0.50 | 100.0% | 378.3 | 0.131761 | 0.131809 | 0.131068 | 0.131546 | 0.6% |
| 0.70 | 100.0% | 1796.1 | 0.132039 | 0.132107 | 0.131358 | 0.131835 | 0.6% |
| 1.00 | 100.0% | 3211.6 | 0.132281 | 0.132357 | 0.131613 | 0.132084 | 0.6% |
| 1.50 | 100.0% | 3876.2 | 0.132488 | 0.132569 | 0.131830 | 0.132296 | 0.6% |
| 2.00 | 100.0% | 4021.4 | 0.132577 | 0.132659 | 0.131924 | 0.132387 | 0.6% |

**Read the shape, not the winner.** The curve is U-shaped and `h = 0.36` sits at the bottom of
it, with measured points on *both* sides — that is what makes it an optimum rather than an
artefact of where the grid stopped.

*Above 0.36* the score is flat to within 0.6% across a fourfold change in bandwidth. That plateau
is saturation, not robustness: `n_eff` climbs to 4021 of 4096 bank paths, so `b^ref` has collapsed
to the unconditional bank mean and the reference no longer depends on the path at all. Every arm
up there is scoring the same path-independent reference.

*Below 0.36* the score gets worse, and `h = 0.28` is the **worst arm on the entire grid** — worse
even than the saturated plateau. The occupancy columns say why: 17% of query rows have every
kernel weight underflow and so train against `b^ref = 0`, and the rows that survive average under
2 bank paths, which makes the "conditional expectation" a lookup of a single training path.

The first version of this sweep tested only `{0.36 … 2.00}`, where 0.36 won by 87% while sitting
on the grid boundary. That 87% was measured entirely against the saturated plateau and was not
evidence of an optimum; extending the grid downward moved the runner-up to `h = 0.31` and cut the
honest margin to **16.5%**, which still clears the 6.91% seed noise floor. Promoting a boundary
minimum as if it were an optimum is exactly the failure `MULTIASSET_GUIDELINE` §12.3 describes, so
`tabulate_hfix.py --promote` now **refuses** a winner that is the smallest bandwidth tested, or
whose kernel has collapsed (`alive% < 98` or `n_eff < 10`), and exits non-zero so the campaign
supervisor aborts instead of launching on it.

**Independent corroboration.** The original `--stage h` sweep in `SWEEP.md` also minimised at
`h = 0.36`, on a finer grid that brackets it tightly (0.34 → 0.0812, **0.36 → 0.0624**, 0.38 →
0.0870) and degrades below 0.32 exactly as the occupancy argument predicts (0.25 → 0.4722). That
sweep ran 100 steps, one seed, and the **truncated** adjoint, so its absolute numbers are not
comparable to the table above and are not used to select anything. But it is a different gradient
and a different budget arriving at the same interior minimum, which is evidence the location is a
property of the kernel rather than of the optimiser.

Regenerate with `python probe_occupancy_low.py && python tabulate_hfix.py`; the winner is written
to `sweep/incumbent.json`.

*The learning rate.* Correcting the adjoint (below) multiplied the gradient magnitude by ~4.7–5.0×
at the median and up to ~59–82× in the tail, so the learning rate selected against the truncated
gradient was not a valid choice for the corrected one and was re-selected too.

Both were selected on `heston_ma_S_val_8192x252x8.npy`. **Neither was selected on test.**

**3. What is the memorisation risk?**
**Structural, and inherited.** `b^ref` is a kernel average over *training* increments, so the
reference itself can reproduce training behaviour — and §2 shows that at small `h` and `K = 20`
it degenerates toward a nearest-neighbour lookup of a single training path. The learned control
only perturbs the diffusion around that reference. The NN-ratio diagnostic (§9) is therefore
mandatory here, and a near-1.0 ratio must be read as a failure, not a pass.

**4. Compute budget.**
Measured, not estimated: **8.11 min / 100 steps**, so 3000 steps ≈ **4.06 h per seed**. Five
seeds run concurrently on 3 GPUs (two seeds share GPU 0 and GPU 1 — a single arm uses 19 GiB of
an 80 GiB card and leaves the SMs latency-bound, so a second round on idle cards would be
strictly slower). Generation plus the full A1–A34 suite adds roughly another hour. The
bandwidth re-sweep that precedes the campaign is 18 arms × 250 steps in two rounds of 9,
~20 min per round.

---

## Hyperparameters

| Symbol | Value | Source |
|--------|-------|--------|
| `lr` | 2.5e-05 | swept on validation |
| `ridge_lambda` | 10.0 | swept on validation |
| `h` | 0.36 | **re-swept against the corrected adjoint**, 9 bandwidths × 3 seeds; interior minimum, see §2 |
| `K` (`markov_order`) | 20 | fixed, matches SBTS |
| `N_pi` | 1 | swept on validation |
| `jacobian_lags` | −1 (all K lags) | **correctness, not tuned** — see below |
| `weight_grad_mode` | `analytic` | **correctness, not tuned** |
| `hidden_dim` / layers | 96 / 1 | reference package default |
| `batch_size` | 256 | reference package default |
| `grad_clip_norm` | 5.0 | reference package default |
| `eta` | 1.0 | specific-entropy weight |
| `[sigma_min, sigma_max]` | `[1e-3, 0.6]` | spectral clip in `project_theta_to_sigma` |
| `dt` | 1/252 | dataset |

`sigma^ref = diag(sigma_assets) / sqrt(dt)`, where `sigma_assets` is the per-asset standard
deviation of training log-returns. It is **computed at load time, never hard-coded**; the
resulting eight numbers are recorded per seed in `../weights/seed_*_config.json` under
`sigma_ref`.

---

## The one substantive fix: the adjoint carried a single lag

This is the reason the campaign was re-run from scratch, so it is worth stating precisely.

`b^ref` at step `i` depends on the last `K = 20` scaled returns through the kernel weights. Its
path-derivative therefore has `K` blocks. The first implementation built only the newest one —
equivalent to `jacobian_lags = 1` — which is not an approximation with a small error but a
gradient with **~100 % relative error** at `K = 20`.

`probe_dbref_dx.py` proves this by finite differences, perturbing one coordinate of one state at
a time and comparing against autograd on the composite `d b^ref / d x_j` that training actually
uses. On a 27-state prefix, of which only the last `K + 1 = 21` states can matter:

| setting | relative error (max / median) | states receiving a gradient |
|---|---|---|
| `jacobian_lags = -1` (now default) | **7.9e-10** / 7.6e-10 | 21 / 27 analytic, 21 / 27 numeric — support matches exactly |
| `jacobian_lags = 1` (the old bug) | **9.97e-01** / 9.78e-01 | 2 / 27 analytic vs 21 / 27 numeric — 19 states silently zero |

The test covers the whole chain, not just the kernel: `d rt / d x = sqrt(dt)/sigma` converts a
return-derivative to a state-derivative, and the backward pass must scatter each lag block onto
the two states that bracket it (`+g` right, `−g` left). Either half can be wrong while the other
is right, and both failures produce a finite, plausible-looking gradient — which is why the
check is finite differences against `b^ref` itself rather than an inspection of the Jacobian.

`jacobian_lags = 1` is kept as a deliberate ablation switch, not as a default.

**Consequence.** Every hyperparameter chosen before this fix was chosen against a gradient with
100 % relative error, so those choices were not evidence about the code that now runs. `h` and
`lr` were re-selected; the old sweep arms are kept under distinct tags so nothing is overwritten,
but their absolute discrepancies are **not** comparable to the new ones (the re-sweep runs 250
steps, the original h stage ran 100).

`probe_jacobian_tail.py` measures the magnitude consequence: the corrected adjoint is ~4.7–5.0×
the truncated one at the median but up to 59–82× at the maximum. That is a heavy tail, not a
pole — each individual lag is integrable because the weight `p_m` vanishes like `(h² − ‖u‖²)²`
faster than `d log K_h/du` diverges — and it shrinks as `h` grows, which is the other reason the
bandwidth had to be re-selected rather than inherited.

---

## Other deviations, and why

**1. Log-weights everywhere.** The kernel is a product of 20 factors each below 1; in float32 the
raw product underflows to zero well before the true weight is negligible. Weights are computed in
log space and normalised with a shift, and rows where every weight underflows are flagged
`alive = False` rather than silently normalised into a uniform average.

**2. Bank and queries are disjoint in every probe.** A first version of `probe_jacobian_tail.py`
drew query prefixes from the same array it used as the bank. Every query was then its own nearest
neighbour at distance exactly 0, survived any bandwidth, and dominated the weights — reporting
`n_eff = 1` at every `h` and inviting the conclusion "the kernel has collapsed", which was an
artefact of the probe. Queries now come from rows beyond the bank. This also matches training,
where the prefixes are model rollouts and can never be bank members.

**3. The reference package carries one pre-existing extension, and this method adds nothing to it.**
An earlier statement here said `methods/Deep-MKV-TS/code/reference` is "not modified". That was
wrong, and the correction matters more than the original claim did. `git status` on the vendored
tree shows:

```
 M src/deep_mkv_gen_path_dt/controls/__init__.py          (+8 lines, re-exports only)
?? src/deep_mkv_gen_path_dt/controls/specific_entropy_matrix.py   (untracked, new)
```

`specific_entropy_matrix.py` is the matrix-valued specific-entropy control, and the eight added
lines in `__init__.py` do nothing but re-export its three public names. Both are dated
**2026-08-24**, i.e. *after* the vendoring commit `0e09c637` (2026-08-16), and both are shared by
`Deep-MKV-TS`, `Deep-MKV-TS-acfup` and `Deep-MKV-TS-base2500` — they predate this method and are
not its doing. What this method contributes is `matrix_control_multiasset.py`, which lives here and
subclasses the extension through the documented control interface; the frozen tree is untouched by
anything in this directory.

**This is a reproducibility hole in the repository, not a stylistic note.** The extension is
untracked, so a fresh clone has no `specific_entropy_matrix.py`, and *every* Deep-MKV-TS variant —
including the ones already published — fails at import. Committing someone else's vendored tree is
not a decision to take unattended, so it is recorded here and left for review rather than fixed.

**4. The pipeline does not push.** `run_pipeline.sh` renders both READMEs and stops. Publishing
is a decision; a script running unattended for hours should not make it.

---

## Reproduce

```bash
cd /home/tbasseras/benchmark/results/HestonMultiAsset/Deep-MKV-TS-SBTSref/code
PY=/home/tbasseras/gpu-venv/bin/python

# 0. correctness probes (CPU, float64, a few minutes)
$PY probe_dbref_dx.py          # finite-difference proof of d b^ref / d x_j
$PY probe_jacobian_tail.py     # occupancy grid and adjoint magnitude
$PY validate_sbts_reference.py

# 1. bandwidth re-sweep, 27 arms = 9 bandwidths x 3 seeds, on validation
bash run_resweep_h.sh          # h in {0.36, 0.50, 0.70, 1.00, 1.50, 2.00}
bash run_resweep_h_low.sh      # h in {0.28, 0.31, 0.33} -- see below
$PY probe_occupancy_low.py     # occupancy; REQUIRED before promoting
$PY tabulate_hfix.py --promote

# 2. 5-seed campaign at the promoted bandwidth (~4 h)
#    run_auto_campaign.sh does 1 and 2 unattended; run_post_campaign.sh chains
#    the post-training pipeline behind it
setsid nohup bash run_auto_campaign.sh  > /dev/null 2>&1 < /dev/null & disown
setsid nohup bash run_post_campaign.sh  > /dev/null 2>&1 < /dev/null & disown

# 3. or, if the campaign is already done, the post-training chain on its own
setsid bash run_pipeline.sh > /tmp/pipeline.log 2>&1 & disown
```
