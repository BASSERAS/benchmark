# SBBTS — Paper Reimplementation (Heston)

- **Paper**: *Schrödinger Bass Bridge for Time Series*, Alexandre Alouadi et al., preprint 2026 — [arXiv:2604.07159](https://arxiv.org/abs/2604.07159) · local copy [`SBBTS_arXiv-2604.07159.pdf`](SBBTS_arXiv-2604.07159.pdf)
- **Official code**: https://github.com/alexouadi/SBBTS — mirrored verbatim at [`../code/reference/`](../code/reference/); the canonical entry point is the authors' `run_heston.py`
- **This run**: [`metric/reproduce_heston.py`](metric/reproduce_heston.py) on one A100-SXM4-80GB, ~64 min per arm (60.6 min train + 2.4 min generate + 1.3 min MLE, measured on the shipped `β = 100` arm)

---

## ⚠️ Reproduction caveat

**The paper publishes no numbers for the Heston experiment.** Section 5.1 is Figure 2 — three KDE
curves (Data, SBTS, SBBTS) over five per-path Heston parameters — plus prose. There is no Table 1
to match, no mean ± std, no metric value anywhere. So "did we reproduce it?" cannot be answered by
comparing digits, and any table that pretended otherwise would be fabricated.

What we did instead: **turn the paper's prose claims into falsifiable statistics** (§1), then test
them. That requires the SBTS curve too, because every claim in §5.1 is about the *gap* between SBTS
and SBBTS, not about an absolute level.

Two substitutions were needed, both documented:

| What | Substituted | Why |
|------|-------------|-----|
| SBBTS generator | [`../code/sbbts_torch.py`](../code/sbbts_torch.py) — merged single-module port | The reference uses bare intra-package imports (`from encoder_only import EncoderOnly`) that only resolve if each sub-directory is separately on `sys.path`; `from models.sbbts_model import ScoreNN` raises `ModuleNotFoundError` as shipped. Maths unchanged — see [`../code/README.md`](../code/README.md) for the full fix list. |
| SBTS generator | `simulate_kernel_vectorized_mark`, imported **unmodified** from `methods/SBTS/code/reference/` | The authors' SBBTS repo has no SBTS path. The recipe (`K=1, h=0.05, N_pi=100`) is taken from the SBTS repo's own notebook, *Parameters Estimation: Illustration on Heston Data*, cells 8-11. |

Both columns are scored by the **same** MLE ([`metric/heston_mle.py`](metric/heston_mle.py)) against
the **same** cached data-side estimates (`dataset/params_data_5000x5.npy`), which is what makes them
comparable to each other.

---

## 1. Paper metrics (as defined in the paper)

The paper's Heston metric is a **per-path bivariate Heston MLE**: fit `(κ, θ, ξ, ρ, r)` to each
trajectory independently by L-BFGS-B, then compare the *distribution* of the 5000 estimates from
generated paths against the distribution from real paths. Figure 2 plots this as a KDE per parameter.

| Metric | Definition | Direction |
|--------|------------|-----------|
| KDE overlap (Figure 2) | visual agreement of the per-path MLE densities, generated vs data | qualitative only |
| `std_ratio` *(ours)* | `std(θ̂_gen) / std(θ̂_data)`, after a joint 5-95 percentile filter | **→ 1.0** |
| `W1` *(ours)* | 1-Wasserstein between the two estimate distributions | ↓ |
| `corr` *(ours)* | `std` across paths of `Corr(Δlog S, Δlog v)`, generated / data — **MLE-free** | **→ 1.0** |

`std_ratio` is the honest formalisation of the paper's own language. §5.1 says SBTS parameters are
*"projected onto an effective average, yielding a concentrated distribution around the centre of the
parameter range"* — "concentrated" is a statement about **spread**, and spread is `std`. So the
paper's qualitative claim becomes:

> **Falsifiable form of the paper's claim.** `std_ratio ≪ 1` for `ξ` and `ρ` under SBTS,
> and `std_ratio ≈ 1` for `κ`, `θ`, `r` under SBTS and for all five under SBBTS.

`corr` exists because `std_ratio` inherits every pathology of the MLE (L-BFGS-B hitting bounds,
flat likelihood directions). `corr` measures the same leverage effect the paper cares about —
the price/variance coupling that `ρ` and `ξ` control — with no estimator in the loop. It is the
ranking key used in the sweep.

**Not reproducible from the paper:** any ± std. The paper reports a single figure, not repeated runs.

### 1b. The paper's one quantitative table — Table 4

§5.1 is the paper's Heston experiment and it publishes no numbers, but the paper is not entirely
number-free about generation quality. **Table 4** (Appendix C.2.1) reports tail-risk and annualised
statistics for **Real vs SBBTS vs SBTS**, averaged across S&P 500 instruments:

| | Real | SBBTS | SBTS |
|---|---:|---:|---:|
| VaR₉₉ (%) | 3.60 | 3.57 | 3.49 |
| VaR₉₅ (%) | 2.11 | 2.19 | 2.17 |
| ES₉₉ (%) | 4.65 | 4.44 | 4.36 |
| ES₉₅ (%) | 3.15 | 3.18 | 3.07 |
| Ann. Ret (%) | 19.02 | 16.31 | 14.68 |
| Ann. Std (%) | 24.22 | 24.38 | 23.06 |

This matters for two reasons. First, it is the anchor for GUIDELINE §15.2 Section 6 in the
**results** README, because every entry is a functional of the price series alone — unlike the §5.1
MLE, which needs the variance leg and so cannot be applied to the benchmark's univariate `d = 1`
paths at all. Second, it already carries an SBTS column, so our benchmark table can mirror the
paper's own three-way layout. Driver: [`metric/heston_paper_metrics.py`](metric/heston_paper_metrics.py)
→ [`results/heston_paper_metrics.json`](results/heston_paper_metrics.json).

⚠️ The paper gives **no formulas** for these six statistics. The standard definitions (VaR as the
negated 1−q return quantile, ES as the mean below it, annualisation by ×`P` and ×√`P`) are pinned
down by the table's own internal consistency: Ann. Std 24.22% ⇒ daily σ = 1.526% ⇒ Gaussian VaR₉₉ =
2.326 σ = 3.55% against the reported 3.60%, with ES₉₉ 4.65% above the Gaussian 4.07% (fat left
tail). On our side the same estimator recovers the Heston generator's known truth: Ann. Std
**20.043%** against √θ = 20.00%, Ann. Ret **2.853%** against the log-drift μ − θ/2 = 3.00%.

---

## 2. Hyperparameters

From the authors' `run_heston.py`, which Alexandre Alouadi identified as canonical
(*"Tu peux retrouver toute la pipeline d'utilisation et les hyperparams ici: …/run_heston.py"*, 2026-09-01):

| Parameter | Value | Source |
|-----------|-------|--------|
| `T` | 1 | Table 2 / `run_heston.py` |
| `safe_t` (⇐ `T̃ = 0.99`) | 1e-2 | Table 2 |
| `β` | 100 | `run_heston.py` |
| `K` (outer DSBM iterations) | 5 | Table 2 |
| `n_epochs` (inner) | 1000 | Table 2 |
| `batch_size` | 128 | Table 2 |
| `lr` | 1e-3 | Table 2 |
| `patience` / `delta` | 15 / 1e-3 | `run_heston.py` |
| `d_model` / `hidden_dim` / `n_layers` | 128 / 64 / 2 | Table 2 + `run_heston.py` |
| `nhead` | **32** (Table 2 says 16) | `run_heston.py` |
| `N^π` | **60** (Table 2 says 50) | `run_heston.py` |
| `M` / `N` / `d` | 5000 / 252 / 2 | §5.1 |
| `M_simu` | 1000 | ours — see below |

Parameter count at `d = 2`: **1,277,890** (recorded in every scores JSON as `config.n_params`).

**Three divergences, all resolved in favour of the repo**, because the author named the repo as
canonical: `nhead` (32 vs 16), `N^π` (60 vs 50), `M_simu` (4000 vs the paper's "5000 trajectories").
We generate `M_simu = 1000` rather than 4000 purely for sweep throughput; the MLE is the bottleneck
and 1000 paths already gives a stable `std`.

**SBTS comparator:** `K = 1`, **`h = 0.05`**, `N^π = 100`, `dt = 1/252`, **no rescaling** (SBTS
consumes raw log returns; SBBTS rescales by the per-channel std). From the SBTS notebook cell 9.
`h = 0.05` is the notebook's value and the SBTS author's standing instruction for this benchmark
(it also supersedes the paper's Table 4 `h = 0.4`, which over-smooths at this length). An earlier
version of this file used `h = 0.2`; see the correction note in §4.

---

## 3. Dataset

Generated by [`dataset/generate_paper_heston.py`](dataset/generate_paper_heston.py) from §5.1 + Table 3.
Each of the 5000 paths gets its **own** parameter draw, uniform over the paper's ranges — that is the
whole point of the experiment: the per-path MLE has to recover a *distribution* of parameters, not one.

| Property | Value |
|----------|-------|
| File | `dataset/X_heston_paper_5000x253x2.npy` (20.2 MB) |
| Shape / dtype | `(5000, 253, 2)` float64 — channels `[price, variance]` |
| `S0`, `v0` | 1.0, 1.0 |
| `dt`, `N` | 1/252 = 0.0039683, 252 |
| Seed | 0 |
| Sanity | `n_nonfinite = 0`, `n_nonpositive_price = 0`; price ∈ [0.0040, 45.75], variance ∈ [0.0039, 3.620] |

Per-path parameter ranges (Table 3), columns of `params_true_5000x5.npy` in order `r, κ, θ, ρ, ξ`:

| `r` | `κ` | `θ` | `ρ` | `ξ` |
|-----|-----|-----|-----|-----|
| [0.01, 0.1] | [0.5, 4.0] | [0.5, 1.5] | [-0.9, 0.9] | [0.1, 0.9] |

`params_true_*.npy` holds the **generating** parameters; `params_data_*.npy` holds the **MLE estimates
on the real paths** — the latter is the correct comparison baseline, because the MLE's own bias and
spread must appear on both sides of the ratio. Every arm reuses this cache, so the data column is
identical across all 28 runs.

---

## 4. Results, ours vs paper

**The paper's side of this table is prose, not numbers.** Quotes are from §5.1 verbatim.

> **`h` correction (2026-09-01, from the method author via T. Basseras).** The SBTS comparator
> was first run at `h = 0.2`. That is wrong: the SBTS Heston notebook uses **`h = 0.05`**, and the
> author's standing instruction is to keep `h` this low for any further SBTS experiment. The
> canonical SBTS column below is therefore the **`h = 0.05`** arm; `h = 0.2` survives only as one
> row of the bandwidth sweep. This changed no conclusion — see the sweep table — and it moved the
> one number that had been awkward (`θ`) in the *favourable* direction, so the earlier `h = 0.2`
> reporting was if anything harsher on SBTS than the correct setting.

| Parameter | Paper's claim about **SBTS** | **SBTS, our run** `std_ratio` (h=0.05) | Paper's claim about **SBBTS** | **SBBTS, our run** `std_ratio` (β=100, mean ± sd, n=4) | Verdict |
|-----------|------------------------------|-------------------------------|-------------------------------|--------------------------------------------------------|---------|
| `ξ` (vol of vol) | *"failed to reproduce"*, *"concentrated distribution around the centre"* | **0.401** | *"captures the full real range"* | **0.763 ± 0.077** | **reproduced ✓** — SBTS collapses, SBBTS ~1.9× wider |
| `ρ` (correlation) | *"failed to reproduce"*, same | **0.154** | *"captures the full real range"* | **0.876 ± 0.069** | **reproduced ✓** — SBTS collapses to 15% of the real spread, SBBTS ~5.7× wider |
| `κ` | *"remain identifiable and well recovered"* | **1.156** | well recovered | **1.104 ± 0.059** | matches ✓ |
| `θ` | *"remain identifiable and well recovered"* | **1.317** | well recovered | **0.914 ± 0.086** | SBBTS ✓; **SBTS over-disperses 1.32×**, which the paper does not mention |
| `r` | *"remain identifiable and well recovered"* | **1.103** | well recovered | **0.970 ± 0.143** | matches ✓ |

**Verdict: the paper's central qualitative claim reproduces.** The `ξ`/`ρ` collapse under SBTS is
large and unambiguous (0.401 and 0.154 against a target of 1.0), the `κ`/`r` recovery is good for
both methods, and SBBTS is dramatically wider on exactly the two parameters the paper singles out.
The MLE-free check agrees: leverage-spread ratio **0.160 for SBTS vs 0.794 ± 0.015 for SBBTS**.

> **Correction (2026-09-01, after the 5-seed benchmark run).** The `std_ratio` cells above and the
> `corr` figures in this section were first published from a `t00s3` scores JSON and a
> `corr_spread_cache.json` that both predated their final re-runs. `sweep_paper.py:corr_ratio` caches
> by trial tag on the documented assumption that "generated arrays are immutable once written", which
> is false — a re-run reuses the tag and overwrites the array while the cache keeps serving the old
> value. Four of the eight `β` tags had drifted (`t00s2, t00s3, t11s1, t11s2`). Every number in §4 has
> been recomputed from the artifacts now on disk by `rebuild_corr_cache.py`; the deltas are small
> here (`ξ` 0.783→0.763, `ρ` 0.865→0.876, `corr` 0.816→0.794) and change no conclusion in this
> section, **but they do overturn the `β` conclusion below** — see the correction there.

**One discrepancy we did not expect, and its resolution.** The paper lumps `θ` with the "well
recovered" group, but our SBTS run over-disperses it by 1.32× — a *wider* distribution than the data,
the opposite of the "projection onto an effective average" story. `h` directly controls kernel
spread, so we swept it (this sweep is also what made the `h = 0.2` error above harmless — the
correct setting was already measured):

| `h` | `corr` | `κ` | `θ` | `ξ` | `ρ` | `r` |
|-----|--------|-----|-----|-----|-----|-----|
| **0.05** *(notebook / author-specified, canonical)* | **0.1603** | 1.156 | **1.317** | 0.401 | 0.154 | 1.103 |
| 0.1 | 0.1544 | 1.083 | **1.718** | 0.379 | 0.149 | 1.067 |
| 0.2 *(our initial error)* | 0.1430 | 1.107 | **1.520** | 0.392 | 0.131 | 1.078 |
| 0.4 *(paper Table 4 value)* | 0.1407 | 1.130 | **1.784** | 0.417 | 0.133 | 1.079 |

**The sweep answers one question cleanly and refuses to answer the other:**

1. **`ξ`, `ρ`, `κ`, `r` and `corr` are flat in `h`** — `ξ` ∈ [0.379, 0.417], `ρ` ∈ [0.131, 0.154],
   `corr` ∈ [0.141, 0.160], across an **8× bandwidth range**. The collapse the paper reports is
   therefore **a property of SBTS, not an artifact of our bandwidth choice.** The paper's central
   claim survives its most obvious confound. This is the important result.
2. **`θ` is *not* explained by `h`.** It is 1.317 → 1.718 → 1.520 → 1.784 as `h` goes 0.05 → 0.4:
   **non-monotone**, swinging 0.40 with no trend. What is robust is that it is **always ≥ 1.3** —
   SBTS over-disperses `θ` at every bandwidth tested. What is *not* established is any mechanism.
   With one run per `h`, the non-monotonicity is indistinguishable from Monte-Carlo noise in the
   generation, and separating the two would need replicates per `h`, which we have not spent.

> **Correction.** An earlier version of this section, written before the `h = 0.1` arm finished,
> read the three available points (1.317, 1.520, 1.784) as a "strictly monotone dose-response" and
> concluded the `θ` gap was a bandwidth artifact. The fourth point broke it. That was the same
> three-points-and-a-story error this README criticises in the `β` sweep below, and it is recorded
> here rather than quietly overwritten.

So the honest status of the `θ` discrepancy is: **real, robust, and unexplained.** The paper groups
`θ` with the "well recovered" parameters; at every bandwidth we tried, SBTS disperses it 1.3-1.8×
wider than the data. We report `h = 0.05` because that is the notebook's value and the author's
standing instruction, **not** because it is the kindest of the four — and we do not tune `h` to make
the discrepancy smaller, which would be choosing a bandwidth to fit prose. It happens that `h = 0.05`
is also the minimum-`θ` arm; that is a coincidence of the sweep, and the claim above is stated over
*all* bandwidths precisely so it does not rest on which one we ship.

The contrast with the `β` sweep below is about *what the data supports*, not about effect size. For
`ξ`/`ρ` the flatness across 8× is itself the finding, and it is solid. For `θ` and for `β` alike, a
handful of unreplicated points is not enough to claim a mechanism, and neither gets one here.

### The hyperparameter sweep: `β` does move the leverage spread, and we shipped `β = 100` anyway

27 arms over `β, safe_t, K, N^π, patience, delta, batch_size, lr, d_model, n_layers`, with 4 seed
replicates on the two most promising. The apparent leader was `β = 300`. On the corrected numbers it
survives on the ranking metric and only on the ranking metric:

| stat | β=100 (n=4) | β=300 (n=4) | diff | Welch t | p |
|------|-------------|-------------|------|---------|---|
| `corr` *(leverage spread, MLE-free — the pre-registered ranking metric)* | 0.794 ± 0.015 | 0.873 ± 0.038 | **+0.079** | **−3.90** | **0.018** |
| `ξ` | 0.763 ± 0.077 | 0.851 ± 0.074 | +0.088 | −1.66 | 0.148 |
| `ρ` | 0.876 ± 0.069 | 0.907 ± 0.075 | +0.031 | −0.61 | 0.565 |
| `κ` | 1.104 ± 0.059 | 1.008 ± 0.042 | −0.096 *(toward 1.0)* | 2.66 | 0.041 |
| `θ` | 0.914 ± 0.086 | 1.067 ± 0.071 | +0.153 *(overshoots 1.0)* | −2.74 | 0.035 |
| `r` | 0.970 ± 0.143 | 1.037 ± 0.073 | +0.067 | −0.83 | 0.447 |

There *is* a dose response on `corr`, and it is near-monotone with an interior optimum:

| β | 150 | 100 | 200 | 300 | 500 | 1000 |
|---|-----|-----|-----|-----|-----|------|
| `corr` | 0.767 | 0.785 | 0.870 | **0.905** | **0.913** | 0.884 |
| `ξ` | 0.745 | 0.767 | 0.907 | 0.853 | 0.765 | 0.826 |
| `ρ` | 0.827 | 0.792 | 0.885 | 0.973 | 0.973 | 0.915 |

Sweeping `β` over 150→1000 (6.7×) moves `corr` by **0.147**, while re-running one fixed config
(`β = 100`) on four seeds moves it by **0.034** — the curve is ~4.3× the seed spread, not inside it.
By comparison `safe_t` over 1e-4→1e-2 (100×) spans 0.108 with no ordering (0.739, 0.816, 0.832,
0.847, 0.804, 0.785 — the optimum is in the middle and the endpoints are the two worst), so `safe_t`
remains a non-lever.

> **Correction (2026-09-01). This section previously concluded "`β` is a plateau, not a lever."
> That was wrong, and it was wrong because of the stale-cache bug documented in §4.** The old cache
> served `t00s2 = 0.9027` (from an array that had since been overwritten) instead of the true 0.7800.
> That single phantom outlier inflated the β=100 mean to 0.816 and its sd to 0.058, which is what
> made the arms look indistinguishable (t = 1.22, p = 0.27). With the cache rebuilt from the arrays
> actually on disk, β=100 is 0.794 ± 0.015 and the difference is t = −3.90, p = 0.018.
>
> **We nevertheless ship `β = 100`, and here is the honest accounting of that choice:**
> 1. `β = 100` is the authors' `run_heston.py` default. GUIDELINE §3 asks for the authors' setting
>    unless there is a strong, established reason to deviate.
> 2. `β` does **not** significantly move `ξ` (p = 0.148) or `ρ` (p = 0.565) — the two parameters the
>    paper's own claim is about. It moves the MLE-free proxy for them, which is not the same evidence.
> 3. The parameter-level effects point in **opposite directions**: β=300 improves `κ` toward 1.0 but
>    pushes `θ` from 0.914 past 1.0 to 1.067. It is a trade, not a strict improvement.
> 4. n = 4 per arm, and six statistics were tested. Under Bonferroni (α = 0.0083) nothing survives,
>    including `corr`. `corr` was pre-registered as *the* ranking metric before this wave was run,
>    which is why it is reported at face value above — but a p = 0.018 on n = 4 is a lead, not a result.
> 5. The full 5-seed benchmark run in `results/Heston/SBBTS/` was executed at `β = 100`, before the
>    cache bug was found. Re-running it at `β = 300` costs ≈25 min and has **not** been done.
>
> **Open item, flagged rather than buried:** if the leverage effect is what matters downstream,
> `β ≈ 300–500` is the better setting and the benchmark under-reports SBBTS. This should go to the
> method authors with the rest of the results.

---

## 5. How to reproduce (EXACT run path)

Interpreter is `/home/tbasseras/gpu-venv/bin/python` everywhere; no other env is involved.

**Dataset** (deterministic, seed 0, ~8.4 s):
```bash
cd /home/tbasseras/benchmark/methods/SBBTS/paper_reimplementation/dataset
/home/tbasseras/gpu-venv/bin/python generate_paper_heston.py
#  -> X_heston_paper_5000x253x2.npy  (5000, 253, 2)
#  -> params_true_5000x5.npy         (5000, 5)  columns r, kappa, theta, rho, xi
```

**SBBTS arm** — this exact command produced the `β = 100` column of §4 (`tag t00`, seed 0):
```bash
cd /home/tbasseras/benchmark/methods/SBBTS/paper_reimplementation/metric
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 taskset -c 0 \
  /home/tbasseras/gpu-venv/bin/python reproduce_heston.py \
    --seed 0 --out ../results/sweep --tag t00 --M-simu 1000 --mle-jobs 1
#  reads  ../dataset/X_heston_paper_5000x253x2.npy
#         ../dataset/params_data_5000x5.npy            (MLE cache, data side)
#  writes ../results/sweep/sbbts_heston_scores_t00.json   <- every SBBTS cell in §4
#         ../results/sweep/params_sbbts_t00.npy
#         ../results/sweep/figure2_params_kde_t00.png
```
The other three replicates are the same command with `--seed 1|2|3 --tag t00s1|t00s2|t00s3`; the
`β = 300` arm adds `--set beta=300` (`--tag t11`, `t11s1..3`). Every `--set K=V` override is recorded
in the JSON's `config` block, so no arm can be mistaken for the default.

**SBTS comparator** — this exact command produced the SBTS column of §4:
```bash
cd /home/tbasseras/benchmark/methods/SBBTS/paper_reimplementation/metric
/home/tbasseras/gpu-venv/bin/python sbts_baseline.py \
    --M-simu 1000 --workers 3 --mle-jobs 1 --h 0.05 --tag sbtsh0p05
#  writes ../results/sweep/sbts_heston_scores_sbtsh0p05.json  <- every SBTS cell in §4
#         ../results/sweep/X_sbts_1000x252x2_sbtsh0p05.npy
#         ../results/sweep/figure2_params_kde_sbtsh0p05.png
```
CPU-only (Numba kernel, forked workers), 31.6 min generation + 1.4 min MLE, 1000/1000 paths
MLE-eligible (0 dropped). **No neural training and no seed-to-seed variance beyond the Brownian
increments** — the empirical training set *is* the model, which is why the SBTS column carries no ±.

⚠️ The `--tag sbts` artifacts in `results/sweep/` are the **superseded `h = 0.2` run** (see the §4
correction). They are kept because they are one row of the bandwidth sweep, but they are *not* the
SBTS column of §4. The command that produced them was the same one with `--workers 6` and no `--h`.

**Bandwidth sensitivity** (resolves the `h` question in §4; the `0.05` arm is also the canonical one):
```bash
for h in 0.05 0.1 0.4; do
  /home/tbasseras/gpu-venv/bin/python sbts_baseline.py \
    --M-simu 1000 --workers 3 --mle-jobs 1 --h $h --tag sbtsh0p${h#0.} &
done; wait
```

**Board / statistics.** `sweep_paper.py` computes `corr` from the generated `.npy` directly (it is
not in the JSONs) and caches it in `results/sweep/corr_spread_cache.json`:
```bash
cd /home/tbasseras/benchmark/methods/SBBTS/paper_reimplementation
/home/tbasseras/gpu-venv/bin/python sweep_paper.py board --seeds 0,1,2,3
```

> **Snapshot.** `results/sweep/snapshot_pre_wave4/` holds the 28 scores JSONs exactly as they stood
> when §4 was written, because later sweep waves re-ran some tags and would otherwise overwrite the
> files these numbers came from. Any §4 cell can be re-derived from that directory alone.

---

## 6. Files

| Path | Role |
|------|------|
| `dataset/generate_paper_heston.py` | per-path-parameter Heston simulator, §5.1 + Table 3 |
| `dataset/X_heston_paper_5000x253x2.npy` | the paper's dataset, `(5000, 253, 2)` |
| `dataset/params_true_5000x5.npy` | generating parameters |
| `dataset/params_data_5000x5.npy` | MLE estimates on real paths — shared baseline for every arm |
| `dataset/metadata.json` | provenance + sanity counters |
| `metric/heston_mle.py` | per-path bivariate MLE, `summarize`, `plot_figure2` |
| `metric/reproduce_heston.py` | SBBTS arm driver (`--set K=V` overrides, `--tag`) |
| `metric/sbts_baseline.py` | SBTS comparator, kernel imported unmodified from `methods/SBTS` |
| `metric/check_author_metrics.py` | cross-check against the authors' own metric helpers |
| `metric/heston_paper_metrics.py` | **Table 4** driver (VaR/ES/annualised) for the benchmark's `d=1` Heston paths — feeds §15.2 Section 6 of the results README |
| `results/heston_paper_metrics.json` | its output: Real + per-seed SBBTS + per-seed SBTS, both aggregations |
| `sweep_paper.py` | `board` sub-command, `corr` computation, `TRIALS` definitions |
| `wave2_noise.sh` … `wave5.sh` | the sweep waves as actually launched |
| `results/sweep/` | 28 scores JSONs, generated `.npy`, Figure-2 KDEs, `corr_spread_cache.json` |
| `results/sweep/snapshot_pre_wave4/` | frozen copy of the JSONs backing §4 |
| `SBBTS_arXiv-2604.07159.pdf` | the paper |
