# Open questions to the protocol author — sent message and remediation runbook

**Status:** drafted 2026-07-31, awaiting reply.
**Blocking:** nothing. CSDI may proceed without any of these answers — see §0.
**Governing constraint on every remediation below:** nothing under
`dataset/Heston/new_experiments/protocol/` may be edited. PDF §7 item 7 requires
the supplied evaluator scripts to run unchanged. Wrap, never patch.

This file exists so that when answers arrive, whoever holds them can act without
reconstructing the reasoning. §1 is the message as sent. §2 is what to do with
each possible reply, with exact paths and costs.

---

## 0. Why none of this blocks CSDI

CSDI passes through **the same two evaluator scripts, on the same machine, with the
same faiss / scipy / numpy** as LS4. A defect in that path is shared by both sides
of the comparison.

| Item | Worst-case reply | Reaches the LS4-vs-CSDI comparison? |
|---|---|---|
| Q1 conformance fixture | absolute values for A and B shift | same shift both methods |
| Q2 pooled-ACF index set | 4 **secondary** keys move ~15% | same 4 keys both methods; **no primary-panel metric moves** |
| Q3 undeclared faiss | nothing — installed and working locally | no |
| Q4 `W̄_param` order | one derived cell's std / CI; mean is invariant | negligible |

**Honest caveat, do not let this be forgotten.** The Q2 cancellation is *not* a
constant offset. Reading (a)'s bias shrinks with lag, so it reshapes the metric
rather than shifting it. If two methods land close on
`abs_return_acf_rmse_lags_1_50`, their relative order on that key could in
principle differ between readings. What bounds the exposure: those keys are
secondary, and no primary-panel metric is affected — measured, not assumed.

**The property that makes deferral safe:** generation and evaluation are separate,
re-runnable steps. `experiment_{A,B}/LS4/generated_paths/seed_{0..4}/` is retained,
so any late answer costs an **evaluation re-run, not a retraining**. Preserve this
property for CSDI.

---

## 1. Message as sent

> **Subject:** Path-law benchmark protocol — one conformance fixture request, and three spec gaps
>
> Hello,
>
> I've implemented the drawdown-memory and Heston-parameter-mixture protocols end to
> end and produced the five-bank submissions for both. Four things surfaced that I
> can't resolve locally. Only the first needs anything from you; the other three are
> reports, with the readings I adopted stated so you can see whether I chose as you
> intended.
>
> Line references are to my vendored copy of the scripts, which I have not modified.
>
> ---
>
> **1. §6 pins no artefact that exercises the two evaluators. Request: one conformance fixture.**
>
> §6 lists six authoritative implementations, then gives ten SHA-256 checksums — all
> ten are data files. Four of the six scripts are cleared transitively: the two
> generators and `path_dt_experiments/heston_mixture.py` regenerate the six pinned
> `.npy` digests exactly, and `fit_heston_mixture_oracle.py` reproduces
> `gate_report.json` byte-for-byte including all 64 confusion-matrix entries.
>
> `evaluate_drawdown_memory.py` and `evaluate_heston_parameter_mixture.py` produce no
> pinned output. Nothing verifies them, and every number I report passes through them.
> §7 item 7 asks me to confirm the evaluators ran unchanged; as things stand that
> confirmation is my account of my own conduct, not something you or a third party can
> check.
>
> A file digest would only partially fix this. Identical script bytes on a different
> stack can still yield different metrics — the evaluators' output depends on numpy's
> reduction order, `scipy.stats.wasserstein_distance`, sklearn's unpickling of
> `oracle.joblib`, and faiss (see item 3), none of which is pinned anywhere in the tree.
>
> What would close it completely:
>
> > Run both evaluators with `disc.npy` supplied as the generated bank, and publish the
> > two resulting metrics JSONs (or their digests).
>
> `disc.npy` is already pinned in §6, so the input is fully determined and any
> implementer can reproduce the run with no new artefact distributed. The output is
> non-degenerate, and it exercises the whole path — pooled ACF, Wasserstein, faiss
> novelty, oracle `predict_proba`. That turns §7 item 7 into a check rather than an
> assertion, and it subsumes the script digests.
>
> If a fixture isn't practical, the SHA-256 of the two evaluators would still help, and
> having §6's two lists correspond would be worth the six.
>
> ---
>
> **2. §2.4: the two means in the pooled ACF cannot range over the same index set.**
>
> §2.4 gives
>
> ```
> ACF_Q(l) = mean_{i,n}[(Q_{i,n} - Qbar)(Q_{i,n+l} - Qbar)] / mean_{i,n}[(Q_{i,n} - Qbar)^2]
> ```
>
> with `i,n` subscripting both means. They can't share a range: the numerator needs
> both `n` and `n+l` valid and runs over `N*(T-l)` terms, the denominator as written
> over all `N*T`.
>
> Two readings:
>
> - **(a) literal** — numerator normalised by `N*(T-l)`, denominator by `N*T`. Biased,
>   shrinking toward zero as `l` grows. This is what the reference implementation does,
>   so I adopted it per §7 item 7.
> - **(b) matched** — both means over the common index set; the textbook estimator.
>
> Measured on the drawdown-memory test bank:
>
> ```
> reading (a), as implemented   abs_return_acf_rmse_lags_1_50 = 0.018115
> reading (b), matched sets                                   = 0.015340
> ```
>
> A 15% gap, wider than my own five-seed spread on that metric (0.01221–0.01811).
> Reading (a) reproduces my committed seed-0 value to five decimals.
>
> Scope: `pooled_acf` is duplicated byte-identically in both evaluators
> (`evaluate_drawdown_memory.py:30`, `evaluate_heston_parameter_mixture.py:42`), so this
> reaches **four reported keys across both benchmarks** — `abs_return_acf_rmse_lags_1_50`
> and `squared_return_acf_rmse_lags_1_50` in each. `leverage_curve` re-centres on the
> truncated slices and is unaffected; no primary-panel metric moves.
>
> I'm not asking which reading governs — §7 item 7 settles that in favour of the
> implementation. The problem is that nothing in the outputs would reveal that two
> conforming implementations had read the formula differently: the discrepancy sits
> inside the plausible seed-to-seed range. One line stating the denominator's index set
> in the next revision would remove that failure mode entirely.
>
> ---
>
> **3. Both evaluators import `faiss`, and no file in the tree declares it.**
>
> `evaluate_drawdown_memory.py:121` and `evaluate_heston_parameter_mixture.py:71` both do
> a lazy `import faiss`. There is no `requirements.txt`, `environment.yml`,
> `pyproject.toml` or `setup.py` anywhere in the distributed tree, so an implementer
> following §6 hits an `ImportError` at first evaluation with nothing in the spec
> explaining it — and `faiss-cpu` vs `faiss-gpu` is a choice the protocol never makes for
> them.
>
> A short pinning file covering faiss, numpy, scipy and scikit-learn would also close the
> environment-drift half of item 1.
>
> ---
>
> **4. §3.4: `W̄_param` is a primary-panel member that the evaluator does not emit.**
>
> §3.4 states the primary Heston-mixture panel is regime TVD, `W̄_param`, realised-
> volatility Wasserstein, and leverage-curve RMSE. But
> `evaluate_heston_parameter_mixture.py` emits only the three individual
> `support_normalized_wasserstein` values — no key corresponds to `W̄_param`. Every
> implementer must reconstruct it downstream, and the aggregation order is unspecified.
>
> I averaged the three normalised distances within each seed, then across seeds, on the
> grounds that `W̄_param` is defined as a per-bank quantity and so must exist per bank
> before anything aggregates across banks. Under a balanced 5×3 design the mean is
> invariant to the order, but the sample standard deviation and the 95% interval are not
> — so two groups reporting the same panel would publish different uncertainty on it.
>
> Emitting `W_param` as a key would settle it at the source. Failing that, one sentence
> fixing the order would do.
>
> ---
>
> Happy to send the regeneration logs behind the digest reproductions in item 1, or the
> measurements behind item 2, in whatever form is useful.
>
> Best,
> Theo

---

## 2. Remediation runbook — what to do when each answer arrives

Python for every command below: `/home/tbasseras/gpu-venv/bin/python`.
Repo root: `/home/tbasseras/benchmark`.

### Q1 — conformance fixture or script digests

**If digests only.** Compute ours and compare; no artefact changes either way.

```bash
cd /home/tbasseras/benchmark/dataset/Heston/new_experiments/protocol/experiments/scripts
sha256sum evaluate_drawdown_memory.py evaluate_heston_parameter_mixture.py
```

- **Match** → close the §11.0 carve-out and the §10 row-10 gap in
  `guideline_new_experiment.md`. No re-run.
- **Mismatch** → our vendored copy diverged from canonical. Stop. Do **not** edit
  `protocol/`; re-vendor from the author's copy, then run the full Q1-fixture branch
  below, then re-evaluate both experiments.

**If a fixture arrives (the requested `disc.npy`-as-generated run).** Reproduce it
before touching anything else:

```bash
# drawdown — substitute the author's exact CLI if it differs from ours
/home/tbasseras/gpu-venv/bin/python \
  dataset/Heston/new_experiments/protocol/experiments/scripts/evaluate_drawdown_memory.py \
  --generated <path-to>/drawdown/disc.npy ...
```

- **Our output matches theirs** → §7 item 7 becomes a check rather than an assertion.
  Record the fixture and our reproduction in `guideline_new_experiment.md` §11.0,
  close the carve-out. No experiment re-run.
- **Our output differs** → environment drift is real. Bisect on faiss / scipy /
  scikit-learn / numpy versions until the fixture reproduces, pin the winning versions
  (see Q3), then re-evaluate **both** experiments **and** CSDI from stored banks —
  retraining is not required. See §2.5.

### Q2 — pooled-ACF index set

**If reading (a) is confirmed normative** (expected — §7 item 7 already implies it):
documentation only. Update `guideline_new_experiment.md` §10.1 item A2 to say the
ambiguity was resolved by the author, cite the reply. No re-run, no value changes.

**If reading (b) is declared normative:** this is the only reply that changes printed
numbers. `protocol/` still may not be edited, so the recomputation must happen in a
wrapper alongside the untouched evaluator, and both readings should be reported.

Affected keys — **four**, all secondary, none primary-panel:

| Benchmark | Keys |
|---|---|
| drawdown (Experiment A) | `abs_return_acf_rmse_lags_1_50`, `squared_return_acf_rmse_lags_1_50` |
| Heston mixture (Experiment B) | `abs_return_acf_rmse_lags_1_50`, `squared_return_acf_rmse_lags_1_50` |

Unaffected: `leverage_curve_rmse_lags_0_20` (re-centres on the truncated slices) and
every primary-panel metric in both experiments.

Reference values already measured, drawdown test bank:
`(a) 0.018115` → `(b) 0.015340`, −15.3%, against a five-seed spread of 0.01221–0.01811.

Steps: add a matched-index ACF to a new tool under `results/new_experiments/tools/`
(do **not** touch `protocol/`), recompute the four keys from
`generated_paths/seed_{0..4}/`, regenerate `metrics_summary.csv` and `pdf_metrics/`,
then run the checkers in §2.5.

### Q3 — undeclared faiss

**Do this now; it needs no reply.** Our results depend on an undeclared binary
dependency. Declare faiss, numpy, scipy and scikit-learn with exact installed
versions in a requirements file **of ours** — never inside `protocol/`.

```bash
/home/tbasseras/gpu-venv/bin/python -m pip freeze | grep -iE '^(faiss|numpy|scipy|scikit-learn)'
```

Record the versions in `guideline_new_experiment.md` alongside the environment
section. If the author later ships a pinning file, reconcile and re-run §2.5 only if
a version actually changes.

### Q4 — `W̄_param` aggregation order

**If per-seed-first is confirmed:** documentation only, no value changes.

**If `W_param` becomes an emitted key, or a different order is mandated:** the mean is
invariant under the balanced 5×3 design, so only the std and the 95% interval move.
One derived cell per experiment.

Components, per seed, in `seed_*_metrics.json`:

```
mixture_fidelity.parameters.theta.support_normalized_wasserstein
mixture_fidelity.parameters.xi.support_normalized_wasserstein
mixture_fidelity.parameters.rho.support_normalized_wasserstein
```

Current implementation: `WBAR_COMPONENTS` in
`results/new_experiments/tools/check_readme_values.py`, function `wbar_cell` —
per-seed mean **then** aggregate, formatted `f"{a.mean():.6g} ± {a.std(ddof=1):.3g}"`.
Change the order there and in whatever generates the README cell, then run §2.5.

### 2.5 — Verification after *any* remediation that changes a number

Run all four. All must pass before commit.

```bash
cd /home/tbasseras/benchmark
P=/home/tbasseras/gpu-venv/bin/python

for E in A B; do
  $P results/new_experiments/tools/check_method_layout.py \
     --root results/new_experiments/experiment_$E/LS4 --experiment $E
  $P results/new_experiments/tools/check_readme_values.py \
     --root results/new_experiments/experiment_$E/LS4 \
     --floor <floor-path-for-$E> --experiment $E
done

$P results/new_experiments/tools/check_oracle_gate.py \
   --gate-report <...> --oracle <...> --minimum-accuracy 0.9

git status -- dataset/Heston/new_experiments/protocol   # MUST be clean
```

Expected: layout 209/209 both; README values PASS both; oracle gate exit 0
(`eight_regime_accuracy = 0.909423828125`, margin +0.009424); protocol tree clean.

**Also required if CSDI exists by then:** re-run the same evaluation branch for CSDI
from its stored `generated_paths/`, so both methods are always evaluated under one
reading. Never publish a comparison where the two methods went through different
evaluator behaviour.

---

## 3. Cost table — what each reply actually costs us

| Reply | Retraining? | Re-evaluation? | Rough cost |
|---|---|---|---|
| Q1 digests match | no | no | minutes (doc edit) |
| Q1 fixture reproduces | no | no | ~1 h (run + record) |
| Q1 fixture differs | **no** | yes, both experiments + CSDI | version bisect + full re-eval |
| Q2 reading (a) | no | no | minutes (doc edit) |
| Q2 reading (b) | **no** | 4 secondary keys, both experiments + CSDI | new tool + re-eval + checkers |
| Q3 | no | only if a version changes | minutes |
| Q4 | no | one derived cell | minutes |

**No branch requires retraining.** That is the whole reason this was safe to defer,
and it holds only as long as `generated_paths/` is retained per seed for every
method — including CSDI.
