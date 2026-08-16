# Experiments

This directory contains reproducible experiment utilities for the
`deep-mkv-gen-path-dt` model package.

The boundary is deliberate:

- `src/deep_mkv_gen_path_dt/` contains reusable model logic.
- `experiments/path_dt_experiments/` contains experiment and evaluation helpers.
- `experiments/scripts/` should contain thin launchers.
- Run outputs, checkpoints, figures, and generated paths should be written under
  `runs/` or `outputs/`, which are ignored by git.

The utilities here cover the common evaluation layer:

- stylized-fact metrics on generated versus target paths;
- discriminative, predictive, and CRPS probabilistic forecasting scores;
- path shadowing, where a large unconditional generated path bank is filtered by
  weighted standardized return/path/volatility/ACF prefix features and the
  selected continuations are rebased to the observed prefix endpoint and
  compared against genuinely future held-out values. Large banks can be sampled
  in chunks with `--sample-batch-size`, and top-k prefix search can use exact
  batched search or FAISS with `--shadow-search`. The scored continuation can be
  restricted with `--shadow-future-steps`;
- context-bank forecast shadowing, where the top-k nearest forecast contexts are
  selected and all model continuations attached to those contexts are pooled;
- plotting helpers for Heston-style diagnostics, covariance heatmaps, and
  shadowing fan charts;
- synthetic target generators for the migrated Heston, routed two-moons, and
  multivariate diagonal-diffusion experiments.

The intended workflow for a script is:

1. create or load target paths;
2. build a `DiscreteMPModel` from the package;
3. train the model;
4. sample generated paths;
5. evaluate with `path_dt_experiments.metrics`;
6. optionally run `path_dt_experiments.scores` and `path_dt_experiments.shadowing`;
7. save metrics and figures under an ignored run directory.

## Migrated Launchers

The migrated end-to-end launchers are:

```bash
python experiments/scripts/run_heston.py
python experiments/scripts/run_two_moons.py
python experiments/scripts/run_multivariate_diagonal.py
```

Each launcher trains the discrete MP model, samples generated paths, and writes
`metrics.json` plus `history.jsonl`. By default it first fits the local Gaussian
reference kernel used by the specific-entropy control. The Heston launcher also
supports a reference-free entropy-barrier control described below. Outputs are
written under `runs/*_dt_*` unless `--run-dir` is provided.

The entropy-barrier running cost controls both drift and volatility without a
fitted prefix-dependent reference law:

```text
0.5 * alpha**2 + 0.5 * kappa * sigma**2 - tau * log(sigma)
```

Use it in the stable direct-target pipeline with:

```bash
python experiments/scripts/run_heston.py \
  --control-map entropy_barrier \
  --entropy-kappa 25 \
  --ce-target-mode direct \
  --target-preconditioner timewise \
  --noise-target-control-variate timewise \
  --grad-clip-norm 0
```

A common-random-number specific-entropy intervention is available through
`run_heston.py --specific-entropy-r-ablation`. In the matched seed-1234 Heston
run, forcing `R=0` increased absolute-return ACF error from `0.0577` to
`0.1941`, squared-return ACF error from `0.0137` to `0.1061`, and excess-
kurtosis error from `0.2006` to `2.7175`. The reference-only `P=R=0` result was
nearly identical to `R=0`. On full-model prefixes, `R` changed sigma by `20.9%`
on average in relative absolute value while leaving mean sigma essentially
unchanged (`0.17293` versus `0.17303`). The fitted reference anchors global
scale, whereas the maximum-principle `R` correction supplies most of the
volatility clustering and heavy tails in this controlled run.

The matching 4,096-bank path-shadowing ablation used a 65-point prefix, a
32-step future, and validation-selected `top_k=256`. Full `P,R` reduced prefix
distance from `11.0560` to `8.7273`, realized-volatility CRPS from `0.02763` to
`0.02434`, and raised realized-volatility 90% coverage from `0.4570` to
`0.6973` relative to `R=0`. An equal-size Heston control obtained `0.02363`
realized-volatility CRPS and `0.8770` coverage. Cumulative-return CRPS was
nearly identical for Heston, full specific entropy, and the ablations, so the
main shadowing contribution of `R` is conditional volatility dispersion rather
than conditional mean prediction.

The next controlled specific-entropy experiment targets that remaining
dispersion error directly with one fixed target-standardized joint MMD. Its
six-dimensional feature contains log prefix RMS volatility at 16/32/64 fine
steps, a recent return, current drawdown, and log future realized volatility
over 32 fine steps. On the observed-only discrepancy grid these become
windows 4/8/16, split 16, and future horizon 8. The specific-entropy running
cost, fitted reference, base discrepancy, network loss, optimizer, and seed
are unchanged. Post-update checkpoints are selected every 200 steps using
realized-volatility CRPS on 512 independent validation paths and a fixed
4,096-path bank; the 512-path test bank remains unread until selection.

The seed-1234 selector chooses step 1,000. Against the original equal-size
specific-entropy bank, test realized-volatility CRPS improves from `0.02434`
to `0.02322`, coverage from `0.6973` to `0.7480`, and prefix distance from
`8.7273` to `7.5417`; cumulative-return CRPS is unchanged within uncertainty
(`0.03702` to `0.03692`). The paired candidate-minus-baseline RV-CRPS interval
is `[-0.00147, -0.00081]`, and the coverage-change interval is
`[+0.0293, +0.0742]`. The RV band widens from `0.0805` to `0.0949`, still below
the equal-size Heston width `0.1146`, so this is a partial rather than complete
repair.

The shadowing gate passes, but the stricter overall gate does not. Absolute-
and squared-return ACF errors and tail-survival error improve, while excess-
kurtosis error worsens from `0.2006` to `1.2073` because the generated
kurtosis overshoots the Heston target (`4.735` versus `3.528`). The terminal
step 2,000 reaches `0.8516` RV coverage but overshoots kurtosis further. Thus
the joint-law signal is effective and statistically validated, but its current
unit weight/checkpoint rule is not yet the final paper configuration. Artifacts
are in `runs/heston_dt_specific_entropy_joint_shadow_seed1234_20260721`.

A validation-frontier audit resolves the checkpoint-selection confound without
changing that unit-weight objective or its deterministic training trajectory.
Every 200-step checkpoint is eligible only when RV CRPS improves, RV coverage
moves toward 90%, cumulative-return CRPS and four stylised-fact errors remain
inside fixed half-bootstrap-standard-error budgets, and path SWD remains within
1.10 times the source value. Steps 1,200 and 1,400 pass all validation guards;
step 1,200 has the lower eligible RV CRPS and is selected. The test law is then
newly simulated from the stored Heston parameters, rather than reusing the
source held-out artifact that informed earlier iterations.

On this untouched law, selected-minus-source RV CRPS is `-0.000259` with paired
95% interval `[-0.000417, -0.000103]`; coverage changes from `0.6816` to
`0.6914`, but its interval includes zero. Excess-kurtosis, tail-survival,
absolute-return ACF, and squared-return ACF errors all improve, and path SWD
falls from `0.02168` to `0.01251`. Cumulative-return CRPS increases by
`0.000291` (about `0.87%`): it is within the predeclared tolerance, although
the paired interval excludes zero. Thus the full tolerance gate passes, but
the result should be reported as a statistically supported conditional-
volatility improvement with a small return-score tradeoff, not as a Pareto
improvement. Reproduce it with
`run_specific_entropy_shadow_frontier.py`; artifacts are in
`runs/heston_dt_specific_entropy_shadow_frontier_seed1234_20260721`.

The formal dated-marginal extension has also been tested directly with the
specific-entropy model. Starting from the later validation-selected joint
specific-entropy checkpoint, it adds local realized-volatility laws at
fine-grid windows 4/8/16/32 and every fourth date: 31 running costs, one
terminal marginal cost, and 117 compatible window/date pairs. The entire
dated family has total weight one, equal to the pre-existing joint-discrepancy
block; the specific-entropy running cost, calibrated `sigma_ref`, complete
pre-existing discrepancy, neural adjoint loss, nested solver, trust fraction,
and absence of clipping are unchanged.

This incremental seed-1234 experiment retains step zero. Across ten
200-inner-step checkpoints, no candidate simultaneously improves validation
RV CRPS and coverage while satisfying the return, stylised-fact, and SWD
guards. Step 1,000 gives the best RV CRPS (`0.022610` versus baseline
`0.022663`) but reduces coverage (`0.7793` versus `0.7871`) and worsens path
SWD and both volatility-ACF errors. Since no checkpoint is selected, the
fresh-test result is unchanged: RV CRPS `0.022267`, RV 90% coverage `0.7578`,
and return CRPS `0.033431`. Thus intermediate one-time volatility marginals do
not, by themselves, repair the missing prefix--future conditional dispersion.
The run is in
`runs/heston_dt_specific_entropy_dated_marginal_w1_nested_trust100_10x200_seed1234_20260729/`.

The previously rejected step 1,600 has since been frozen as a
coverage-focused candidate and evaluated on five newly simulated Heston laws.
This removes the concern that its original `0.875` validation coverage was a
single-law accident. Across 2,560 new prefixes, its RV coverage is `0.8605`,
compared with `0.7602` for the selected joint specific-entropy checkpoint and
`0.8797` for an equal-size Heston oracle. The paired step-1,600-minus-selected
change is `+0.1004`, with 95% interval `[+0.0875,+0.1129]`.

RV CRPS also changes favorably from `0.022217` to `0.021998`, although the
paired interval `[-0.000480,+0.000059]` includes zero. Cumulative-return CRPS
changes by only `+0.000024`, with interval `[-0.000066,+0.000119]`. The frozen
literal rule nevertheless fails because it requires a non-positive point
change in both CRPS values; only one of five individual seeds passes all three
conditions.

This checkpoint is a real conditional-calibration improvement, but it is not
a full-law improvement. Its mean excess kurtosis is `4.863` against a Heston
reference mean of `3.644`; mean tail-survival error is `0.01027` versus
`0.00337` for the selected model, path SWD is `0.02410` versus `0.01976`, and
squared-return ACF error is `0.01167` versus `0.00818`. It can therefore
support a coverage-focused result or a Pareto-frontier discussion, but not a
claim that the same checkpoint dominates the selected model on unconditional
stylized facts. Artifacts are in
`runs/heston_dt_step1600_locked_fresh5_seed917001_20260729/`.

The matched nested-bank sweep shows that the low RV coverage is not primarily
a 4,096-scenario retrieval effect. At bank sizes 4K, 16K, 65K, 262K, and 1M,
frontier-selected coverage is respectively `0.691`, `0.686`, `0.664`, `0.670`,
and `0.674`; the equal-size Heston oracle obtains `0.879`, `0.863`, `0.873`,
`0.885`, and `0.887`. At 1M the selected-minus-oracle gap is `-0.2129` with
paired 95% interval `[-0.2500, -0.1777]`. The selected RV band is `27.2%`
narrower, and its RV CRPS is `0.00145` worse with interval
`[0.00087, 0.00206]`. The checkpoint still improves source RV CRPS by
`0.000232` and coverage by `0.0254`, but it does not recover the oracle's
conditional dispersion. This establishes structural prefix--future
volatility-law error and prevents interpreting a larger scenario bank as the
remaining repair. Artifacts are in
`runs/heston_dt_specific_entropy_frontier_bank_sweep_seed1234_20260721`.

The matched shock-dominance diagnostic further rules out a scalar-RV artifact.
At horizons 8, 16, and 32, specific-entropy RV coverage is `0.684`, `0.664`,
and `0.674`, compared with Heston's `0.889`, `0.895`, and `0.887`. At horizon
32, removing the largest squared return lowers model coverage to `0.654`, and
bipower-volatility coverage is `0.646`; Heston remains near `0.89` for both.
Maximum-absolute-return coverage is closer but still deficient (`0.770` versus
`0.893`). The largest return contributes 19.2% of selected-model squared
variation on average, below 23.9% in both realized continuations and Heston
shadows. Of 167 model RV misses, 101 are below the lower bound and 66 exceed
the upper bound. The failure therefore combines insufficient temporal
concentration of variation with an overly compressed conditional RV law; it
is neither a single-shock effect nor a purely high-volatility upper-tail miss.
Artifacts are in
`runs/heston_dt_specific_entropy_rv_shock_diagnostic_1m_seed1234_20260721`.

The core now implements the missing complete path-dependent running-cost
source. It evaluates `dt` times the specific-entropy integrand on each
generated prefix, detaches the realized controls, differentiates through the
fitted prefix-dependent reference volatility, and adds that full-grid
derivative to the discrepancy source. The `dt` factor is the one in the
paper's actual stage cost; the displayed control Hamiltonian is normalized by
`dt`. The neural loss remains the same adjoint-moment MSE. Exact-value, finite-
difference, and adjoint-timing tests cover the new term.

Retraining the same seed-1234 unit-joint-discrepancy experiment selects step
1,000. On the same 512 held-out prefixes and common-seed 4,096 model banks,
the original source, reduced joint model, and complete-MP joint model obtain RV
coverage `0.6973`, `0.7480`, and `0.7285`, respectively; Heston obtains
`0.8789`. Their RV CRPS values are `0.02434`, `0.02322`, `0.02324`, and
`0.02368`.

Complete MP improves the original source, but not the otherwise identical
reduced joint model. Complete-minus-reduced RV CRPS is `+0.000022` with paired
95% interval `[-0.000283, 0.000360]`; coverage is `-0.01953` with interval
`[-0.04102, 0.00195]`. At the selected checkpoint, running-source RMS is only
`0.02198` versus discrepancy-source RMS `25.699` (ratio `0.000855`) with cosine
`0.0105`. The running cost still constrains the controls through the
Hamiltonian argmin; it is its additional prefix derivative that is tiny.
Clipping at norm 5 is active on `99.55%` of updates, with full-run gradient-
norm mean `33.65` and q95 `72.73`. Artifacts are in
`runs/heston_dt_specific_entropy_complete_mp_dt_joint_shadow_seed1234_20260721`.

The zero-noise-adjoint volatility is `sqrt(tau / kappa)`. If
`--entropy-tau` is omitted, the Heston launcher sets this baseline to one
global diffusion scale estimated from the target increments; this calibrates a
single unit of scale but does not fit a state- or prefix-dependent reference.
Pass `--entropy-tau` explicitly when the scale must be entirely exogenous. The
saved configuration records the resulting baseline and its source.

To target volatility dynamics rather than only marginal volatility, add the
model-free conditional block:

```bash
python experiments/scripts/run_heston.py \
  --control-map entropy_barrier \
  --entropy-kappa 25 \
  --preset old_fullv_w0p25 \
  --conditional-volatility-discrepancy \
  --conditional-volatility-windows 4 8 16 32
```

At the path midpoint, this block forms recent RMS and mean-absolute volatility
over the requested windows and the matching future quantities. Each empirical
law is standardized separately, and an MMD compares all prefix/future
interactions. This makes the block invariant to marginal volatility scale;
the existing volatility-law terms continue to control marginal scale and
dispersion. The window arguments are expressed in original grid steps and
converted to the observed grid, so with `--observed-every 4` the example uses
observed windows 1, 2, 4, and 8. No latent Heston variance or fitted volatility
state is used.

By default, the Heston launcher also writes
`discrepancy_diffusion_scale_sensitivity` to `metrics.json`. For each
discrepancy block, it differentiates the block along timewise rescalings of
centered generated increments while holding mean increments fixed. Negative
gradients ask local gradient descent to increase diffusion; positive gradients
ask it to decrease. Use `--sensitivity-paths` to set the diagnostic subsample,
or `--no-discrepancy-sensitivity` to disable it.

The conditional block is experimental and remains off by default. In the
three-seed 500-update Heston ablation it increased the mean h32 RV IQR ratio
from 0.729 to 0.785 and slightly reduced absolute-return ACF error, but worsened
mean path SWD from 0.0734 to 0.0955 and moved prefix/future volatility
correlation from 0.728 to 0.750, farther from the Heston value 0.523. See the
technical note for the complete table and seed variation.

For a more directed persistence signal, use the stabilized correlation block
instead:

```bash
python experiments/scripts/run_heston.py \
  --control-map entropy_barrier \
  --entropy-kappa 25 \
  --preset old_fullv_w0p25 \
  --conditional-volatility-correlation-discrepancy \
  --conditional-volatility-correlation-weight 0.025 \
  --conditional-volatility-correlation-windows 16 32
```

For each window, the block centers prefix and future RMS realized volatility
within each empirical law and computes their Pearson correlation. Each
generated standard deviation is floored at 0.25 times the corresponding
detached target standard deviation. Above the floor this is the ordinary,
scale-invariant correlation; at collapsed generated dispersion the floor keeps
the statistic and gradient finite. The default targets only the matched 16-
and 32-step pairs, where the persistence mismatch is largest. Add
`--conditional-volatility-correlation-full-matrix` to compare every selected
prefix/future pair. The MMD and direct-correlation flags are mutually exclusive.

The preceding target-normalized covariance ablation used weight 0.1. Against
an independent Heston reference, it raised the mean h32 RV IQR ratio from 0.727
to 0.905 and reduced kurtosis error from 1.37 to 0.37, but moved correlation
only from 0.728 to 0.722 versus a target of 0.546 while increasing mean h32
volatility from 1.051 to 1.210 times the reference. The stabilized Pearson
replacement prevents that objective from being satisfied by changing marginal
volatility amplitude. Its weight defaults to 0.025; the MMD block keeps its
separate weight-1 default.

In the subsequent 50-update sweep, weights 0.025, 0.05, and 0.1 were all
finite and the scale floor was inactive by the final update. Weight 0.025 was
selected as the least distortive pilot. Over three 500-update seeds it reduced
mean h32 volatility inflation to 1.089 times the reference, improved the h32
IQR ratio from 0.727 to 0.793, and reduced absolute-return ACF error from 0.0390
to 0.0326. It did not solve persistence: h32 correlation moved only from 0.728
to 0.722 versus the 0.546 reference, while mean path SWD worsened from 0.0734
to 0.0988. The block therefore remains an explicit negative/diagnostic
ablation rather than the recommended default training configuration.

A subsequent block-specific nested diagnostic used 64 prefixes, 32 independent
future branches, and steps 32, 48, 64, and 80. After removing nested-mean noise
bias with split-half cross moments, the correlation block supplied only
0.9%, 2.7%, 4.2%, and 3.8% of the full control-effective volatility signal at
those dates. Its (R) direction was aligned with or nearly orthogonal to the
full-minus-block remainder, rather than cancelled by it. Conditional (R)
reliability became measurable around the midpoint (0.089 at step 64 and 0.154
at step 80), while the block-specific (P) reliability remained at or below
0.028. The block is therefore volatility-oriented but too small at weight
0.025 relative to the other discrepancy sources.

A controlled 500-update seed-1234 follow-up increased only that weight from
0.025 to 0.1 and repeated the same diagnostic. The larger weight improved path
SWD from 0.0892 to 0.0475 and excess-kurtosis error from 3.279 to 0.092 for
this seed, but it did not improve the intended persistence statistic: h32
prefix/future correlation changed from 0.7207 to 0.7224. Instead, h32 realized-
volatility mean, standard-deviation, and IQR ratios rose from 0.925, 1.019, and
0.611 to 1.540, 1.592, and 1.531. At step 64 the correlation block's effective
volatility share increased from 4.2% to 9.4%, but its mean cross-fitted (R)
cosine with the full-minus-block remainder changed from +0.221 to -0.366; the
larger block is therefore opposed by the rest of the objective. The scale floor
remained inactive and the 95th-percentile gradient norm remained stable
(2.52), so neither floor clipping nor training instability explains the
failure. This one-seed result rejects simple further weight increases: stronger
coupling creates discrepancy competition and volatility-level distortion
without correcting persistence.

The subsequent frozen-response audit stepped back from retraining. A stable
500-update seed-1234 barrier baseline was frozen, and the weight-0.025
correlation block was introduced only as a diagnostic candidate. Directly
descending that block in path space reduced the h32 correlation error by
0.00201 under a 1% return-RMS perturbation, while changing the h32 RV mean and
standard-deviation ratios by only +0.00008 and +0.00019. Thus the statistic's
path gradient points in the intended direction and does not intrinsically use
volatility amplitude to improve correlation. In contrast, the full composite
path direction increased correlation error by 0.00239. The instantaneous,
state-conditioned, global, rolling-RV, absolute-return, and squared-return
volatility blocks each moved correlation in the wrong direction; they are the
source of objective competition.

The learned target-forcing response preserves the candidate's sign but
entangles it with scale. At a common-noise parameter perturbation of `1e-5`,
the candidate reduced correlation error by 0.00148 but raised h32 RV mean,
standard-deviation, and IQR ratios by 0.00690, 0.00202, and 0.00759 and raised
path SWD by 0.00247. The full instantaneous adjoint-MSE direction instead
increased correlation error by 0.00228 while reducing the three RV scale
ratios by 0.02843, 0.01738, and 0.02864. Candidate central derivatives at
fractions `1e-5` and `3e-5` agreed to within 0.4% for correlation and RV mean,
confirming that these signs are local rather than a large-step artifact.

The scale loss occurs between path gradients and neural controls. The
candidate norm is 0.335% of the full path-gradient norm but only 0.0268% of the
full parameter-update norm. Its (P) and (R) parameter directions have similar
norms and cosine approximately zero, so this is not (R) compensating for (P).
After normalizing every block row, the path-direction matrix has condition
number 12.0 and stable rank 2.54, whereas the neural-direction matrix has
condition number about 1.22 million and stable rank 1.79. Distinct discrepancy
signals therefore collapse into a few nearly dependent network/control
directions. The next intervention should reduce the redundant volatility
blocks and introduce separately anchored, time-local marginal-scale controls;
another global weight increase is not supported.

That reduced intervention is available as `--preset reduced_time_local`. It
contains only the observed-path MMD, independently standardized local
realized-volatility marginals at selected dates, and the optional direct
prefix/future correlation block:

```bash
python experiments/scripts/run_heston.py \
  --control-map entropy_barrier \
  --entropy-kappa 25 \
  --preset reduced_time_local \
  --conditional-volatility-correlation-discrepancy \
  --conditional-volatility-correlation-weight 0.025 \
  --time-marginal-volatility-window 4 \
  --time-marginal-volatility-endpoints 8 16 24 32
```

The time-marginal window and endpoints are indices on the discrepancy grid;
with observed-only training they therefore count observed returns. If no
endpoints are supplied, the builder uses the four grid quartiles. Each local
RMS-volatility marginal is normalized by its own detached target mean and
standard deviation, and the generated marginal uses those same target
statistics. Level, dispersion, and shape errors remain visible, while temporal
dependence is left to the observed-path and correlation blocks. The
construction uses only observed path increments and remains model-free and
path-dependent.

The frozen pre-training audit justified one controlled seed: relative to the
old composite, row-normalized neural conditioning improved from about
`1.22e6` to `168`, and the correlation direction grew from `0.0268%` to
`0.129%` of the full parameter direction. The 500-update seed-1234 gate did
not justify a multi-seed run. Against the same held-out Heston bank, the reduced
model corrected h32 RV mean from `1.519` to `1.002` times the reference, RV
standard deviation from `1.299` to `0.914`, and path SWD from `0.146` to
`0.070`. However, h32 RV IQR fell from `1.193` to `0.788`, absolute-return ACF
error rose from `0.0542` to `0.0987`, and prefix/future correlation moved from
`0.697` to `0.731` instead of toward the Heston value `0.523`. Training was
finite without clipping (gradient-norm q95 `1.45`, no non-finite gradients),
so this is a statistical-control failure rather than numerical explosion.

The post-training response audit localizes the remaining conflict. The
correlation block alone improves correlation under a local neural perturbation,
but it is anti-aligned with the full learned direction (cosine `-0.575`); the
full perturbation still worsens correlation. The final minibatch discrepancy
metrics are evaluated before the last optimizer update, whereas the saved
checkpoint, fresh probe, generated bank, and frozen-response audit use the
post-update model. The latter artifacts must therefore be used for the
single-seed gate. The next experiment should isolate the persistence update
from the scale, control-variate, and prediction-shrinkage directions rather
than restore the removed global volatility proxies or launch more seeds.

That isolation experiment is implemented by
`run_residual_r_persistence.py` and has now been run. It freezes the reduced
generator and its ordinary-adjoint output, adds a zero-initialized causal GRU
only to the noise-adjoint `(R)` channel, and trains it with the unchanged
adjoint MSE and entropy-barrier control. One previously hidden convention was
corrected before the gate: the observed-only 32-step block actually measured
eight aggregated returns and had full-bank target correlation `0.4313`, while
the reported fine-grid h32 statistic has target `0.5315`. The residual uses the
exact fine-grid statistic and fixes that target once from all 8,192 training
paths.

The validation result is decisive. The best unconstrained checkpoint, at step
15, reduced h32 correlation error by 90.2%, from `0.1682` to `0.0164`, and
improved path SWD from `0.05297` to `0.04771`. Thus the conditional learner and
`R` control can transmit a strong persistence correction. It achieved that
correction by collapsing volatility-state dispersion: h32 RV standard-
deviation ratio fell from `0.960` to `0.660`, IQR ratio from `0.796` to `0.615`,
and mean ratio from `1.030` to `0.956`. The h5 and h10 correlations fell from
`0.474` and `0.568` to `0.213` and `0.272`, versus validation Heston values
`0.587` and `0.611`. All three scale-preservation checks therefore failed.

Checkpoint selection used only the independent validation bank and retained
step 0; rejected residual checkpoints were not evaluated on held-out paths.
The held-out report consequently records the unchanged base generator and an
overall failed gate. No multi-seed run is justified. This result rejects the
hypothesis that lack of signal transmission is the remaining problem. A single
correlation objective is underconstrained: even though Pearson correlation is
invariant to a global scale factor, a path-dependent volatility control can
change it by compressing volatility-state dispersion. Any next formulation
must impose the time-local volatility marginals as constraints, or project the
persistence update onto their constraint tangent space. Another correlation
weight, volatility proxy, or unconstrained residual phase would repeat the
same failure mode.

The residual runner now also supports explicit intermediate volatility-
marginal anchors. For every requested horizon it fits the complete 8,192-path
training-bank empirical marginal of prefix and future RMS volatility, then
uses a standardized one-dimensional Wasserstein discrepancy against that
fixed empirical quantile function. The target is deterministic, uses no
latent-volatility model, and is independent of the dummy target minibatch used
by the fixed-correlation residual:

```bash
python experiments/scripts/run_residual_r_persistence.py \
  --source-run-dir runs/heston_dt_entropy_barrier_k25_reduced_time_local_corr_w0p025_seed1234_20260718 \
  --run-dir runs/heston_dt_residual_r_h32_marginal_anchor_w0p025_paramdiag_seed1234_20260718 \
  --device cuda:3 \
  --eval-every 5 \
  --correlation-weight 0.025 \
  --marginal-anchor-weight 0.025 \
  --marginal-anchor-horizons 5 10 20 32
```

Equal `0.025` weights make the baseline noise-adjoint target norms comparable:
`767` for the marginal block and `812` for correlation. Their noise-adjoint
target cosine is `-0.027`, so the two mathematical signals are distinct. The
residual GRU does not preserve that distinction: at its zero initialization,
the two block-specific parameter-gradient directions have cosine `0.974`, and
the marginal gradient norm is `0.0706` versus `0.0295` for correlation. The
network therefore maps two nearly orthogonal adjoint targets into essentially
the same trainable direction.

The 100-step validation gate retained step 0, and even the best correlation
checkpoint was step 0. At the terminal rejected checkpoint, h32 correlation
moved from `0.7079` to `0.7367`, away from the Heston value `0.5397`; h32 mean
and standard-deviation ratios moved from `1.030` and `0.960` to `1.047` and
`1.111`. The IQR ratio improved from `0.796` to `0.911`, but that partial
marginal gain does not compensate for the other failures. Path SWD moved from
`0.05297` to `0.05581`. Training remained finite, with loss `0.543 -> 0.388`,
gradient-norm maximum `0.121`, and no non-finite gradients.

This rejects a weighted scalar marginal-penalty fix for the residual gate. The
marginal laws must be imposed as actual constraints, or the persistence
direction must be projected in a representation where their tangent
directions remain distinguishable. `block_direction_diagnostics.json` records
path-adjoint, noise-adjoint, and residual-parameter norms and cosines at the
baseline, terminal, selected, and best-correlation checkpoints.

The constraint-tangent alternative is tested directly by
`run_constrained_feasibility_audit.py`. At the selected zero-residual
checkpoint it differentiates the complete generated rollout with respect to
the residual parameters, uses h32 correlation absolute error as the objective,
and treats h32 RV mean, standard-deviation, and IQR absolute ratio errors as
three separate non-worsening constraints. It then solves the exact
minimum-distance projection onto the three linearized half-spaces and checks
the result with common-noise central finite differences:

```bash
python experiments/scripts/run_constrained_feasibility_audit.py \
  --residual-run-dir runs/heston_dt_residual_r_h32_marginal_anchor_w0p025_paramdiag_seed1234_20260718 \
  --run-dir runs/heston_dt_residual_r_h32_constrained_feasibility_seed1234_20260720 \
  --device cuda:3 \
  --audit-dtype float64 \
  --num-paths 2048 \
  --parameter-step-fractions 1e-6 3e-6 1e-5 \
  --seed 1234
```

This audit exposed and corrected an important differentiation bug in the
residual wrapper. The base-network parameters were frozen correctly, but its
forward pass was also inside `no_grad`, which removed the base response to the
evolving generated state. The wrapper now keeps the base weights frozen while
retaining its input-state Jacobian. Audit noise is drawn once in float32 and
cast to the requested precision, so switching to float64 for small finite
differences does not change the Monte Carlo realization.

Across seeds 1234--1238, each with 2,048 generated paths, only
`0.000419--0.002406` of the unconstrained correlation direction survives the
marginal constraints (mean `0.001482`, or `0.148%`; the materiality gate is
`10%`). Mean and standard-deviation constraints are active in every seed, with
IQR additionally active in one. All projected directions agree with the
common-noise finite differences, but their correlation improvement is only
numerical-scale because the surviving direction is so small. Thus the local
gate fails in all five seeds. This rejects a multi-marginal optimizer as the
next change *within the current shared residual parameterization*: the next
intervention must first give persistence and marginal control independent
tangent directions. The per-seed artifacts are under
`runs/heston_dt_residual_r_h32_constrained_feasibility_seed*_20260720/`.

The audit also identifies why this is a local parameterization result rather
than a capacity verdict on the full GRU. Because the last residual layer is
initialized exactly to zero, only its 96 weights and one bias have nonzero
first-order gradients: `97/37,921` parameters (`0.256%`). The GRU and preceding
nonlinear head are invisible to an infinitesimal update at step 0. A precise
next architecture test is therefore a centered residual
`f_theta(path) - f_anchor(path)`, where the anchor weights are frozen but its
path-input derivative is retained. It is zero at initialization, its input
Jacobians cancel, and its complete parameter Jacobian remains active.

That centered parameterization is implemented and available through
`--residual-parameterization centered_full_tangent` in both the residual
training runner and the feasibility audit. It changes neither the maximum-
principle targets nor the entropy-barrier control, running cost, or normalized
adjoint-MSE loss. The repeated feasibility gate activates all `37,921`
trainable parameters, but still retains only `0.000779--0.004022` of
correlation descent across generated-path seeds 1234--1238 (mean `0.002506`,
or `0.251%`). This is only a 1.6--2.0 times increase over the zero-final
readout, and remains far below 10%. With the generated-path bank fixed at seed
1234, centered initialization seeds 1234--1238 give the narrow range
`0.003211--0.003396`, so the negative gate is not a random-initialization
artifact. Every common-noise finite-difference check passes.

The full centered residual was therefore not trained substantively. A one-step
smoke run verifies that it uses the existing adjoint-MSE pipeline and that its
checkpoint can be reloaded by the audit. The result now localizes the obstacle
beyond zero-head initialization: the current shared scalar `R` response still
maps persistence and marginal-scale corrections into nearly the same rollout
tangent direction.

The remaining control-versus-representation question is resolved by
`run_causal_r_basis_feasibility.py`. It bypasses the conditional neural
representation and injects a direct causal basis into `R`. At every time, the
basis contains standardized rolling log realized volatility over windows
4, 8, 16, and 32, their squares, all pairwise interactions, and an intercept.
Its 15 features are calibrated only on an unperturbed generated bank; no
Heston state, latent volatility, or reference paths enter the features. The
1,920 coefficients are time-local, and the unchanged entropy-barrier map still
converts `R` into volatility:

```bash
python experiments/scripts/run_causal_r_basis_feasibility.py \
  --residual-run-dir runs/heston_dt_residual_r_h32_marginal_anchor_w0p025_paramdiag_seed1234_20260718 \
  --run-dir runs/heston_dt_causal_r_basis_feasibility_seed1234_20260720 \
  --device cuda:3 \
  --audit-dtype float64 \
  --num-paths 2048 \
  --num-calibration-paths 4096 \
  --bootstrap-replicates 200 \
  --seed 1234
```

This gate passes decisively. Across path seeds 1234--1238, the exact
mean/std/IQR non-worsening projection retains `74.35%--85.05%` of direct
correlation descent (mean `81.75%`). Allowing each marginal error to move by at
most half of its bootstrap standard error per targeted 10% correlation-error
reduction retains `93.85%--99.97%` (mean `97.52%`); a one-standard-error budget
retains 100% in all five seeds. Every local common-noise finite difference
passes.

The finite-step result is also practically material. For seed 1234, an exact-
tangent step of `0.012` reduces correlation error by `0.01810`, or `9.6%`,
while mean error changes by only `+0.00034` and standard-deviation/IQR errors
improve by `0.00112/0.00066`. The half-standard-error direction reduces
correlation error by `0.02017`, or `10.7%`, and all three finite marginal
changes stay inside their bootstrap allowances.

The feasible direction is genuinely path-local: across seeds, essentially no
coefficient energy lies before the measured prefix window or after the future
window. The split between the prefix and future windows ranges from
`27.7%/72.3%` to `59.4%/40.6%`; seed 1234 is `50.2%/49.8%`. Thus the scalar
maximum-principle `R -> sigma` channel is not the obstruction. The next model
should expose these causal rolling-volatility features as a time-local skip
basis in the residual representation, then enforce the intermediate
marginals through the multi-marginal maximum-principle formulation. The
neural adjoint-MSE loss itself need not change.

That first integration is implemented by
`run_causal_r_basis_training.py`. The fitted GRU base is frozen and verified
bitwise unchanged; only the 1,920 calibrated basis coefficients are optimized.
The experiment uses the existing fixed correlation and intermediate-
volatility-marginal MP targets with equal `0.025` weights and the unchanged
normalized adjoint MSE. It uses a fixed `2e-4` learning rate, without clipping
or a schedule. Validation selection requires a 10% h32 correlation-error
reduction while limiting each mean/std/IQR error increase to half of its
baseline bootstrap standard error.

The five seed-1234--1238 runs are a mixed positive result. The previous GRU
parameter-gradient collapse disappears: the marginal/correlation cosine is
now `-0.353`--`-0.243`, rather than `+0.974`. The best-correlation checkpoints
reduce held-out correlation error by `8.0%--29.4%` (mean `17.2%`) and four of
five satisfy the held-out half-SE marginal constraints. Seeds 1237 and 1238
pass the complete validation-plus-held-out gate; seeds 1234 and 1236 remain at
step 0 after validation selection, and seed 1235 narrowly misses held-out
confirmation. The unchanged MP training signal therefore can use the causal
basis, but the current two-block objective and single validation-bank selector
are not yet robust enough for the final empirical claim.

The formal multi-marginal extension is implemented in
`deep_mkv_gen_path_dt.discrepancies.multimarginal`. A
`TimeIndexedWeightedDiscrepancy` is evaluated only on its dated path prefix;
`MultiMarginalPathFunctionalDiscrepancy` combines these running law costs with
terminal law costs. The existing reverse cumulative source sum then produces
the discrete adjoint jumps from all future dated marginals. An analytic test
checks the resulting jump locations and magnitudes without changing the
conditional-adjoint regression loss.

For the Heston basis experiment, `--marginal-formulation multi_marginal`
installs realized-volatility marginal laws at all observed dates 4, 8, ...,
128, using every compatible window among 4/8/16/32. There are 31 running dates,
one terminal marginal date, and 117 window/date statistics. The original total
marginal weight `0.025` is divided uniformly over the 32 dates; the correlation
block remains at `0.025`. Thus this is neither a weight sweep nor a change to
the entropy running cost.

This resolves most of the seed instability. Seeds 1234--1237 pass validation
selection and held-out confirmation, with selected held-out correlation-error
reductions of `64.1%`, `24.6%`, `63.8%`, and `58.1%`. Seed 1238 reduces the
error by `16.0%` but misses the half-SE standard-deviation allowance by only
`0.00040`, so the reported result remains four passes out of five. The mean
held-out reduction is `45.3%`, compared with `9.7%` for checkpoints selected
under the earlier global sparse anchor.

The subsequent held-out path-shadowing gate is negative. With identical
4,096-path banks, 512 validation/test prefixes, a 32-step future, and
validation-selected `top_k=256`, the multi-marginal checkpoints improve mean
prefix distance in every seed (`10.365` to `9.022`). However, they worsen
cumulative-return CRPS in all five seeds (`0.03774` to `0.03803` on average)
and realized-volatility CRPS in all five (`0.02847` to `0.03061`). Mean
realized-volatility 90% coverage falls from `0.625` to `0.603`; an equal-size
Heston bank reaches `0.877`. Every paired realized-volatility CRPS interval
excludes zero in favor of the uncorrected entropy-barrier baseline. Thus the
dense dated costs successfully target the selected h32 correlation statistic
but do not improve the full neighbor-conditioned future law. This is a failed
path-shadowing gate, not a final generator improvement.

The Heston launcher can form out-of-fold conditional-adjoint targets and report
training-stability diagnostics without changing the discrepancy, running cost,
or adjoint MSE:

```bash
python experiments/scripts/run_heston.py \
  --ce-crossfit-folds 2 \
  --batch-size 1024 \
  --ridge-lambda 1e-3 \
  --grad-clip-norm 50 \
  --log-every 25
```

With two folds, each ridge conditional-expectation fit predicts paths excluded
from that fit. `--grad-clip-norm 0` disables clipping. Logged metrics include
out-of-fold CE RMS and R-squared, head-specific gradient norms, relative
adjoint errors, parameter-update ratios, volatility-control quantiles and bound
activity, plus rolling and full-run gradient/clipping summaries.

The shadow-frontier launcher also exposes the nested forward--backward solver:

```bash
python experiments/scripts/run_specific_entropy_shadow_frontier.py \
  --solver nested \
  --nested-outer-steps 10 \
  --nested-inner-steps 200 \
  --nested-outer-relaxation-mode adjoint_blocks \
  --nested-outer-block-coordinate-passes 2 \
  --grad-clip-norm none \
  --selection-objective coverage_error
```

One outer sweep freezes the generated bank and complete adjoint targets for all
inner updates. The resulting parameter displacement is then backtracked against
the unchanged complete empirical objective before the law is refreshed. Use
`--no-nested-outer-objective-line-search` only for an explicit undamped
diagnostic.

On the seed-1234 complete-MP Heston run, the first 200-step frozen solve reduces
adjoint MSE by 28.8% and lifts the candidate `P/R` RMS ratios to 0.387/0.551.
The undamped candidate nevertheless raises the complete objective from 7.56 to
718.87: its discrepancy contribution is 519.88 and its specific-entropy running
cost is 198.99, so neither term alone explains the rejection. The line search
accepts a 1/32 displacement, lowers that batch objective to 7.09 (7.081
discrepancy plus 0.009 running cost), and reduces fixed-point path RMS movement
from 11.49 in the undamped diagnostic to 0.043. Across ten sweeps, accepted
relaxation falls through 1/1024 to zero and no checkpoint passes the validation
gate; RV coverage is
0.518--0.570 versus 0.707 for the source. A 1,000-step first backward solve fits
the frozen targets further but makes the undamped objective worse (2,023.47),
so the failure is not attributable to clipping or to refreshing targets after
every single optimizer step. Artifacts are under
`runs/heston_dt_specific_entropy_complete_mp_dt_nested*linesearch*seed1234_20260721`.

The follow-up directional audit tests whether this failure comes from an
adjoint sign, date, or time-step scaling error:

```bash
python experiments/scripts/run_nested_directional_diagnostic.py \
  --device cuda:3 \
  --run-dir runs/heston_dt_specific_entropy_complete_mp_nested_directional_seed1234_20260722
```

It replays the identical first frozen batch. Direct autograd through the
open-loop Euler dynamics reproduces the implemented pathwise adjoint exactly:
the drift-chain relative RMSE is zero and the noise-chain relative RMSE is
`5.2e-8`, with cosine one. The implemented oracle control has zero free/KKT
Hamiltonian residual. Reversing `R` changes the closed-loop objective
directional derivative from `-88.65` to `+56.88`; reversing both gives
`+88.72`. Replacing `R/sqrt(dt)` by either `R` or `R/dt` violates the original
Hamiltonian KKT conditions. Thus the source timing, signs, and `dt` factors are
consistent.

The remaining failure is block-imbalanced outer stiffness. `R` alone supplies
`-72.77`, or about 82% of the initial descent, and reaches objective `6.288`
from `7.563`; `P` alone supplies only `-15.89` and reaches `7.476`. Yet at the
full joint proposal, `198.978` of the `198.991` running cost is the quadratic
drift term, while the volatility term is only `0.0127`. The correct joint curve
has its sampled minimum at relaxation `1/64`, then rises to `718.87` at one.
Apparent improvements from multiplying `P` by `dt` are therefore numerical
suppression of an inaccurate/stiff fitted `P` block, not evidence that the
maximum-principle formula is missing `dt`. The next solver experiment should
use separately controlled outer updates for `P` and `R`, while leaving the
objective and Hamiltonian equations unchanged. Artifacts and the objective
curve plot are in the run directory above.

That solver experiment is implemented as `adjoint_blocks`. It interpolates the
same fully fitted candidate in three parameter groups: the `P` output head, the
`R` output head, and the shared GRU. A deterministic coordinate search chooses
each rate from the existing geometric backtracking grid using the same outer
paths, noises, target batch, discrepancy, and running cost. A zero rate is
allowed, so a harmful block can be held while another advances. This remains a
single ordinary network after the update and changes neither the complete
path-dependent source nor the neural regression loss.

On the matched seed-1234 run, the first update chooses rates `1/128`, `1/32`,
and `1` for `P`, `R`, and the shared block. It lowers the complete objective
from `7.563` to `4.115`, versus `7.091` under joint relaxation, and raises RV
coverage from the joint run's `0.559` to `0.633`. Across ten outer sweeps,
coverage reaches `0.738`, compared with `0.545` at the end of the joint run.
This is a genuine stability improvement, but not a model improvement over the
source: source coverage is `0.707`, source RV CRPS is `0.023760`, and every
blockwise checkpoint has worse RV CRPS (`0.024259`--`0.024778`). The return-tail
fit also degrades: excess-kurtosis error grows from source `0.196` to roughly
`3.7`--`4.2` after the early transient. Consequently no checkpoint passes the
predeclared gate and the fresh-test selector correctly retains step zero.

The key numerical finding is that the accepted parameters retain
`0.9985`--`0.9999` of the pre-sweep fixed-target regression loss. Independent
relaxation prevents the explosive Picard move, but does so by shrinking almost
all of the solved backward correction away. The remaining limitation is
therefore not clipping, simultaneous P/R relaxation, or a sign/scaling error:
descent of the in-sample complete objective is not aligned with improved
out-of-sample path shadowing for the present conditional targets. Artifacts are
in
`runs/heston_dt_specific_entropy_complete_mp_dt_nested10x200_blockrelax_unclipped_frontier_seed1234_20260722`.

The matched held-out audit is:

```bash
python experiments/scripts/run_nested_holdout_diagnostic.py \
  --initialization source \
  --locked-scale 0.125 \
  --device cuda:3 \
  --run-dir runs/heston_dt_specific_entropy_complete_mp_nested_holdout_warmstart_locked_fresh_seed1234_20260722
```

It first exposed an important initialization confounder in the frontier above:
that candidate was trained from a fresh random network, while step zero in the
selector was the separately saved specific-entropy checkpoint. On the fresh
network the training-selected objective update generalizes to the independent
bank (`-3.448` fitting and `-2.905` held out) and improves the candidate's RV
CRPS from `0.027667` to `0.025018`, but it still cannot match the saved source's
`0.024296`.

The scientifically relevant run therefore warm-starts the candidate from the
source checkpoint. The fitting-bank coordinate search selects outer rates
`P=1/16`, `R=1/32`, and `shared=1/32`; the independent bank selects the same
`P/R` rates and `shared=1/64`. The fitting-selected update lowers the complete
objective by `1.228` on its own bank and by `0.815` independently, ruling out a
simple conditional-target overfit. It also improves RV CRPS from `0.024296` to
`0.023686`, coverage from `0.703` to `0.779`, ACF errors, tail-survival error,
and path SWD. Its excess-kurtosis error nevertheless rises from `0.057` to
`0.967`, so the full blockwise objective step fails the predeclared gate only
on kurtosis.

A fixed one-eighth trust fraction of that same MP update gives rates
`P=1/128`, `R=1/256`, and `shared=1/256`. It still lowers the complete objective
on both banks (`-0.228` fitting and `-0.193` held out), improves RV CRPS to
`0.024219`, coverage to `0.717`, and excess-kurtosis error to `0.053`, and is
the first correction in this investigation to pass every predeclared
path-shadowing and stylized-fact check. Fractions `1/4` through `1` improve RV
CRPS more but fail only the kurtosis guard. Thus the MP direction contains
useful out-of-sample information; the remaining numerical issue is trust-region
scale from the already good source law, not target overfitting or a defective
maximum-principle sign. Full objective curves, common-noise generated banks,
bootstrap gate decisions, and checkpoints are in the run directory above.

After locking the one-eighth scale, a newly simulated Heston test law with new
reference and scenario-bank seeds confirms the validation result. RV CRPS
improves from `0.022904` to `0.022868`; the paired 95% interval for the change
is `[-6.84e-5, -3.37e-6]`. Coverage improves from `0.6816` to `0.6895`, with
paired interval `[0.0020, 0.0156]`. Excess-kurtosis error improves from `0.382`
to `0.272`; both ACF errors, tail-survival error, and path SWD also improve.
The locked candidate therefore passes the complete fresh-test gate rather than
only the validation sweep.

The locked solver-level protocol was then run at seeds `1234`, `2234`, and
`3234` with a source-checkpoint warm start and
`--nested-outer-block-trust-fraction 0.125`. At the fixed one-outer-step
horizon, all three fresh-test gates pass. RV CRPS improves in every seed by an
average `-4.42e-5`, and every paired 95% interval excludes zero. Coverage rises
by an average `0.0091`; all paired intervals are nonnegative. Kurtosis error and
path SWD also improve in all three seeds.

Allowing ten outer solves and selecting checkpoints on validation strengthens
the mean fresh RV improvement to `-3.69e-4` and coverage improvement to
`0.0384`; all three RV-CRPS intervals still exclude zero. Selected steps are
`1400`, `1800`, and `2000`. However, only two of three complete fresh gates
pass: seed `3234` improves RV CRPS, coverage, ACF, tail, and SWD but worsens
fresh kurtosis. The defensible default is therefore the fixed one-step
correction. The multi-step trajectory is evidence that the MP signal can
accumulate, but it needs a more robust tail-aware stopping rule before becoming
the primary specification. Aggregate artifacts are in
`runs/heston_dt_specific_entropy_complete_mp_nested_warmtrust_multiseed_summary_20260722`.

Removing the extra trust shrinkage while retaining the objective-checked
blockwise line search improves mean three-seed fresh RV coverage from `0.721`
at trust `1/8` to `0.764` at full trust. No run uses gradient clipping and no
non-finite gradient is observed. A matched bank sweep on seed 1234 nevertheless
reduces full-trust coverage from `0.758` at 4K scenarios to `0.707` at 1M,
versus `0.887` for the Heston oracle. The remaining gap is therefore not a
finite-retrieval effect.

`run_rv_shadow_regime_diagnostic.py` localizes that one-million-path gap using
tertiles of the observed prefix RMS volatility. Full-trust coverage is
`0.556`, `0.700`, and `0.866` in the low, middle, and high regimes, versus
oracle values `0.895`, `0.865`, and `0.901`. Low-regime bands have width
`0.0565` versus the oracle's `0.0881`, with excessive misses on both sides.
Thus globally correct future-RV dispersion is being allocated incorrectly
across prefix regimes.

The optional `--regime-weight` block adds fixed empirical soft-prefix-regime
times future-log-RV moment and tail claims to the unchanged complete MP
objective. `--regime-conditional-moment-weight` additionally exposes the
normalized conditional mean and standard deviation directly. At unit claim
weight, the claim-only seed-1234 checkpoint raises 1M coverage to `0.744`, but
its RV CRPS (`0.02213`) is worse than the unaugmented full-trust value
(`0.02187`). The direct-moment variant obtains coverage `0.717` and CRPS
`0.02188`, essentially the original proper-score result. Both pass their 4K
fresh full-law gate, but neither recovers the oracle conditional law. These are
retained as a negative/partial result rather than promoted as the default.

`run_hamiltonian_regime_audit.py` diagnoses this failure on the maximum-
principle side. It reconstructs the complete-path conditional P/R targets and
separates the discrepancy source from the path derivative of the specific-
entropy running cost. Current controls, their pointwise Hamiltonian targets,
eta sensitivity, and declared sigma-bound activity are then compared in the
same low/middle/high prefix-volatility regimes. Source and selected policies
use common noise; saved line-search rates are reported only as global context.

The subsequent frozen-backward learnability audit keeps the selected forward
paths and complete P/R teachers fixed for 5,000 optimizer steps. It never
rolls the fitted candidate back into the dynamics. Fit-bank relative RMSE
plateaus at `0.506` for P and `0.713` for R; applying the same network and
transferred teacher to an independent path bank gives `0.532` and `0.725`, so
ordinary fit-bank overfitting is not the explanation. Independently rebuilt
teachers are not stable: their global agreement is only correlation `0.452`
for P and `0.182` for R, with relative RMSE `0.978` and `1.833`. In the future
control window, the fitted P output is constant within every prefix regime and
the R fit has correlations only `0.239--0.256`. The current bottleneck is
therefore twofold: the recurrent representation learns mainly timewise target
structure, while the empirical ridge conditional target itself varies strongly
across Monte Carlo banks. Another discrepancy or outer trust adjustment is not
supported before stabilizing the conditional-expectation layer.

The fixed-bank ridge sweep covers lambda `0.001` through `10,000`. Values at or
below `1` are indistinguishable from the unstable baseline. Increasing lambda
raises global R teacher correlation from `0.182` to `0.623` at `1,000`, but
retains only `37.5%` of baseline conditional R variation and reduces raw-target
R² from `0.050` to `0.012`. At `10,000`, global R correlation is `0.769`, but
only `24.8%` of R variation and raw-target R² `0.003` remain. The critical low-
volatility regime is not repaired: even at `10,000`, R teacher correlation is
only `0.266`, relative RMSE is `1.917`, and conditional R variation retention
is `13.3%`. Strong ridge therefore creates apparent global stability mainly by
shrinking away the signal, while low-regime sigma agreement remains poor. No
ridge value in the sweep is a defensible replacement for the baseline.

The frozen direct-target audit removes that ridge teacher entirely while
holding the selected forward law, complete pathwise adjoints, architecture,
optimizer rate, running cost, discrepancy, and Hamiltonian fixed:

```bash
python experiments/scripts/run_frozen_direct_backward_learnability_audit.py \
  --run-dir runs/heston_dt_specific_entropy_complete_mp_nested_trust100_frozen_direct_backward_learnability_seed1234_20260722 \
  --device cuda:3
```

The unscaled arm is the exact direct squared-error regression. The second arm
divides each time/output target by a fixed fit-bank RMS; this changes numerical
weighting but not the timewise population minimizer. Both use the selected
causal P output as an unbiased control variate in `(P_pathwise - P_selected) *
epsilon`, train without clipping or a schedule for 5,000 steps, and evaluate on
an untouched independent bank against baselines estimated only from the fit
bank.

Direct scaling exposes a real P signal: at the fit-loss-selected step its
held-out fixed-baseline R-squared is `0.144`, versus `0.011` for the unscaled
arm, and it reaches `0.254` early in the diagnostic trajectory. It does not
solve R. The selected scaled fit has held-out R R-squared `-0.012`; no trained
checkpoint exceeds the selected generator's initial `0.039`. The proposed P
control variate also changes the R target second moment by factors `1.003` on
the fit bank and `1.004` held out, so it slightly increases rather than reduces
variance. Thus fixed scaling repairs an important P/R numerical imbalance, but
the remaining obstacle is construction of a lower-variance unbiased R target,
not ridge tuning, clipping, or another outer trust adjustment. The fitted
networks in this audit are diagnostic backward fits and are never fed into the
forward dynamics.

The follow-up replaces that ineffective selected-P baseline by the conditional
variance-minimizer
`c_t(prefix)=E[P_t epsilon_t^2|prefix]/E[epsilon_t^2|prefix]`. It is estimated
with five-fold cross-fitted 128-nearest-neighbor local weighted least squares
on 17 fixed causal features: current/cumulative/last returns, rolling
mean/RMS/log-RMS at windows 4/8/16/32, drawdown, and run-up. A fitting query
cannot use any P or noise sample from its own fold; independent queries use
only the fitting bank. Therefore `c_t(prefix) epsilon_t` has zero conditional
mean and the MP R quantity is unchanged.

```bash
python experiments/scripts/run_frozen_direct_backward_learnability_audit.py \
  --r-control-variate-mode crossfit_knn \
  --control-variate-neighbors 128 \
  --run-dir runs/heston_dt_specific_entropy_complete_mp_nested_trust100_frozen_direct_crossfit_r_seed1234_20260722 \
  --device cuda:3
```

The variance reduction transfers: controlled/raw R second-moment ratios are
`0.615` cross-fitted and `0.816` on the independent bank. This still does not
improve neural R learning. The selected generator starts at held-out R-squared
`0.0436`; no unscaled or scaled training checkpoint exceeds it. At the
fit-loss-selected scaled step, held-out R-squared is only `0.0073`, while P is
`0.0798`. Low/middle/high initial R-squared values are `0.0330/0.0877/0.0383`;
only low volatility briefly reaches `0.0411`, while the other regimes retain
their best value at step zero. Lower Monte Carlo variance is therefore useful
but not sufficient. The next isolation target is the shared recurrent
representation and joint P/R optimization, not another control-variate or
ridge sweep.

`run_frozen_r_representation_isolation.py` tests that hypothesis without a
new forward rollout. Both arms reuse the identical cross-fitted direct R
targets and fixed timewise scale. The first updates only the 9,409 R-head
parameters while freezing the selected GRU and P head. The second gives R an
independent diagnostic copy of the selected GRU plus R head (37,921 trainable
parameters), with no P loss or P gradient. Both use fixed-rate AdamW for 5,000
unclipped steps and select only by fitting-bank R loss:

```bash
python experiments/scripts/run_frozen_r_representation_isolation.py \
  --run-dir runs/heston_dt_specific_entropy_complete_mp_nested_trust100_frozen_r_representation_isolation_seed1234_20260722 \
  --device cuda:3
```

The result rejects shared P/R gradients as the primary failure. Head-only
training lowers normalized fit loss from `0.9993` to `0.9676`, preserves P
exactly, but lowers held-out R-squared from `0.0436` to `0.0288`. The separate
R encoder lowers fit loss further to `0.8733` while held-out R-squared falls to
`-0.0323`. Every arm and regime achieves its best global independent score at
the selected initialization, apart from an immaterial low-regime fluctuation.
Extra representation capacity therefore fits frozen-bank residual noise rather
than a transferable conditional signal. The next justified diagnostic is
nested branch averaging at fixed prefixes, which directly tests whether a
higher-precision conditional R target is learnable before any generator
training is changed.

`run_nested_representation_diagnostic.py` performs that test while also
checking whether the causal input representation is the bottleneck. The
selected complete-MP generator, specific-entropy running cost, discrepancy,
pathwise adjoint construction, and log-price dynamics remain frozen.
Independent future branches are averaged at fixed prefixes, and matched causal
GRUs are trained against the resulting P/R means using raw log prices, a
standardized log-price scaling control, standardized log returns, or a hybrid
input. Only the GRU input representation changes.

The initial 32-branch experiment found a large apparent standardized-return
advantage at the split-time control: held-out R-squared was `0.390`, versus
`-0.062` for log prices, and the low-volatility value was `0.405`, versus
`0.020`. Its aggregate paired interval nevertheless included zero, and the
nested variance decomposition estimated that roughly 87 branches were needed
for unit R signal-to-noise at that time. The result was therefore treated as a
pilot rather than used to modify the generator.

The confirmation uses 128 branches and the actual future-control times
`64/72/80/88/95` on independent 256-prefix fitting and held-out banks:

```bash
python experiments/scripts/run_nested_representation_diagnostic.py \
  --run-dir runs/heston_dt_specific_entropy_nested_representation_future128_seed1234_20260728 \
  --device cuda:0 \
  --fit-prefixes 256 \
  --heldout-prefixes 256 \
  --branches 128 \
  --step-indices 64 72 80 88 95 \
  --primary-step 64 \
  --representations log_price standardized_log_return
```

Standardized returns materially outperform log prices in relative terms. The
paired global R-squared difference is `+0.0867` with 95% interval
`[+0.0485,+0.1255]`, and the step-64 difference is `+0.0792` with interval
`[+0.0367,+0.1236]`. This does not clear the substantive gate. Fit-selected
global R-squared remains negative for both representations (`-0.131` log
price, `-0.044` return), and the critical step-64 low-volatility difference is
`-0.0070` with interval `[-0.0636,+0.0462]`. The return representation also
worsens global P R-squared from `-0.017` to `-0.055`.

Thus differencing and scaling make the frozen R regression less bad, but do
not recover a transferable low-regime conditional signal. The predeclared
promotion rule rejects every representation, so no full nested generator run
is launched. This prevents a noisy 32-branch result from being mistaken for a
law improvement. Artifacts are in the run directory above.

The final frozen objective-estimator audit separates future-noise variance
from target-law minibatch variance and the complete-MP components:

```bash
python experiments/scripts/run_objective_estimator_variance_audit.py \
  --run-dir runs/heston_dt_specific_entropy_objective_estimator_audit_seed1234_20260728 \
  --device cuda:0
```

It crosses 128 common prefixes and 32 future branches with 16 disjoint
256-path target batches at steps 64, 80, and 95. The three isolated targets
(specific-entropy running cost, base discrepancy, and joint prefix/future
discrepancy) reconstruct the direct complete target to float32 precision.
For complete R, future branches account for `84.3%/81.7%/81.1%` of conditional
variance, target batches for only `1.6%/2.2%/2.6%`, and interactions for the
remaining `14.1%/16.1%/16.3%`. Averaging eight target batches per half gives
teacher correlations `0.992/0.992/0.992`, whereas averaging 16 future branches
per half gives only `0.082/0.279/0.288`.

The running-cost conditional R RMS is only `0.26%--0.36%` of complete R RMS,
and its low-regime marginal sigma effect is at most `0.0016` in magnitude.
The partition-averaged complete teacher instead changes low-regime sigma by
`-0.0444/-0.0142/-0.0325`; base and joint discrepancies determine that
direction. Therefore neither a larger target-law minibatch nor direct
full-pool target averaging is supported as the next fix. The unresolved
estimation problem is the future-noise covariance in R, consistent with the
independent 128-branch representation confirmation.

The final variance-reduction gate tests antithetic future shocks without
training:

```bash
python experiments/scripts/run_antithetic_r_audit.py \
  --run-dir runs/heston_dt_specific_entropy_antithetic_r_audit_seed1234_20260729 \
  --device cuda:0
```

For every one of 128 prefixes, the audit compares 64 two-path antithetic pairs
`Z,-Z` with 64 two-path IID pairs, crossed with four disjoint 256-path target
batches. At steps 64, 80, and 95, antithetic/IID R variance ratios are
`0.967 [0.890,1.049]`, `1.155 [1.037,1.281]`, and
`0.666 [0.578,0.769]`. Pair-half R correlations are respectively
`0.560/0.542`, `0.639/0.677`, and `0.836/0.765` for antithetic/IID. Thus the
method is useful late in the path, but provides no material reduction at the
primary split step and significantly worsens variance at step 80.

At step 64 both estimators retain a negative low-regime sigma change
(`-0.0891/-0.0915`), but the antithetic pair halves disagree in sign and the
four target batches do not agree. The predeclared gate therefore rejects an
antithetic training pilot. This closes the tested sequence of target pooling,
control variates, causal representations, nested branch averaging, and
antithetic sampling: no estimator tested so far gives a stable, material
low-regime R improvement at the control time without changing the MP problem.

To retain conditioning on the complete path without using the unstable ridge
teacher, train the same causal GRU directly against the Monte Carlo pathwise
adjoint samples:

```bash
python experiments/scripts/run_heston.py \
  --ce-target-mode direct \
  --ce-crossfit-folds 1 \
  --batch-size 1024 \
  --grad-clip-norm 5
```

Under squared error, the population regression functions are the conditional
adjoint moments given the full prefix. The final history record also contains a
fresh-batch `probe_adjoint_r2` and `probe_noise_adjoint_r2`; these are evaluated
after the last optimizer update on paths and noises not used by that update.

Direct targets can be preconditioned with fixed, independently calibrated
time/output RMS scales:

```bash
python experiments/scripts/run_heston.py \
  --ce-target-mode direct \
  --target-preconditioner timewise \
  --preconditioner-batches 8 \
  --preconditioner-min-scale 1e-3
```

The network predicts targets divided by the frozen scales; its outputs are
multiplied by the same scales before they reach the control map or diagnostics.
This is an invertible deterministic reparameterization, so it retains the
full-prefix conditional-expectation target. It uses RMS scaling without a mean
offset, keeping the zero-initialized heads equal to zero raw controls. The
calibration stream is distinct from training and does not change training
minibatch or noise seeds.

The noise-adjoint sample can use an unbiased control variate. `timewise`
calibrates the variance-minimizing deterministic baseline for each
time/state/noise coordinate; `adjoint` uses the detached causal prediction of
the ordinary adjoint:

```bash
python experiments/scripts/run_heston.py \
  --ce-target-mode direct \
  --target-preconditioner timewise \
  --noise-target-control-variate timewise
```

The resulting target is `(A - c) * epsilon`. Because either baseline is known
from the prefix before `epsilon` is drawn, its conditional expectation is the
same as that of `A * epsilon`. Training and fresh-probe metrics report the
controlled-to-raw target second-moment ratio.

To estimate how much conditional signal is actually present in the pathwise
targets, branch independent future-noise batches from fixed prefixes:

```bash
python experiments/scripts/run_heston.py \
  --conditional-signal-diagnostic \
  --signal-prefixes 64 \
  --signal-branches 8 \
  --signal-step-indices 16 32 48 63 \
  --signal-discrepancy-blocks volatility_law_conditional_correlation
```

Every branch is evaluated as a separate discrepancy minibatch, preserving the
Lions-gradient batch scaling. The report contains within-prefix noise,
between-prefix signal, an estimated reliability ratio (the fraction of target
variance attributable to the conditional mean), split-half agreement, and the
network's R-squared against nested branch averages. Raw and control-variate
noise targets are both retained. Summary metrics are written into
`metrics.json`; samples, nested means, predictions, and variance components are
saved in `conditional_signal_tensors.pt`.

When `--signal-discrepancy-blocks` is supplied, each named weighted block is
also differentiated in isolation on exactly the same paths and target batches.
The report compares its conditional (P) and (R) directions with the full
discrepancy and the full-minus-block remainder. Jacobian-vector products through
the configured control map report the resulting drift and volatility effects,
including attenuation from active control bounds. These isolated block signals
are diagnostic decompositions; the network prediction remains a prediction of
the full discrepancy signal. Signal-only alignment and control-effect RMS are
also estimated by applying the same calculations to two independent halves of
the future branches and taking their cross moment; this removes the positive
Monte Carlo noise bias present in a squared nested-mean norm.

FAISS is optional for large pure path-shadowing runs:

```bash
pip install -e ".[faiss]"
python experiments/scripts/run_heston.py --shadow-search faiss --sample-batch-size 50000
```

For a reusable top-k sweep with path-level bootstrap intervals and an optional
paired comparison bank, run:

```bash
python experiments/scripts/run_shadowing_sweep.py \
  --heldout-paths TEST_PATHS.pt \
  --validation-paths VALIDATION_PATHS.pt \
  --generated-paths PRIMARY_BANK.pt \
  --comparison-generated-paths BASELINE_BANK.pt \
  --comparison-label same_dgp \
  --prefix-length 65 \
  --future-steps 32 \
  --top-k 64 128 256 512 \
  --selection-metric cumulative_return_crps
```

The validation and held-out tensors must be distinct. The script selects
`top_k` only from validation CRPS, reports the selected configuration on the
held-out paths, and writes paired `primary - comparison` bootstrap intervals.
It saves aggregate results in `shadowing_sweep_metrics.json` and all per-path
metrics in `shadowing_per_path_metrics.pt`.

Before changing the generator or shadow features, diagnose whether a
realized-volatility mismatch is already present in the unconditional path law:

```bash
python experiments/scripts/run_volatility_diagnostics.py \
  --reference-paths HELDOUT_PATHS.pt \
  --reference-label target \
  --candidate mp=MP_BANK.pt \
  --candidate heston=HESTON_BANK.pt \
  --prefix-length 65 \
  --prefix-window 32 \
  --horizons 5 10 20 32 \
  --lags 1 2 5 10 20
```

The diagnostic uses target-defined prefix-volatility tertiles for every path
bank. It reports unconditional realized-volatility distribution ratios,
absolute- and squared-return ACF errors, and future volatility conditional on
low, middle, and high prefix volatility. Large candidate banks are
deterministically subsampled with `--max-candidate-paths` (100,000 by default).
The JSON report, path-level tensors, and summary figure are written under the
chosen ignored run directory.

Shadowing scores use three endpoint-aligned forecast representations:

- cumulative future returns relative to the observed prefix endpoint;
- future one-step increments;
- horizon realized volatility, defined as the square root of summed squared
  future increments.

The observed prefix endpoint is not scored. Absolute-level shadow fans remain
available for visualization after every selected continuation is anchored at
the observed endpoint.

Fast wiring checks are:

```bash
python experiments/scripts/run_heston.py \
  --run-dir /tmp/deep_mkv_path_dt_heston_smoke \
  --num-target-paths 32 \
  --num-heldout-paths 16 \
  --num-sample-paths 16 \
  --num-grid-steps 8 \
  --training-steps 2 \
  --batch-size 8 \
  --target-batch-size 8 \
  --hidden-dim 8 \
  --log-every 1 \
  --shadow-real-paths 4 \
  --shadow-top-k 3 \
  --no-scores \
  --no-plots \
  --no-save-tensors

python experiments/scripts/run_two_moons.py \
  --run-dir /tmp/deep_mkv_path_dt_two_moons_smoke \
  --num-target-paths 32 \
  --num-heldout-paths 16 \
  --num-sample-paths 16 \
  --num-grid-steps 8 \
  --training-steps 2 \
  --batch-size 8 \
  --target-batch-size 8 \
  --hidden-dim 8 \
  --log-every 1 \
  --no-scores \
  --no-plots \
  --no-save-tensors

python experiments/scripts/run_multivariate_diagonal.py \
  --run-dir /tmp/deep_mkv_path_dt_mvdiag_smoke \
  --num-target-paths 32 \
  --num-heldout-paths 16 \
  --num-sample-paths 16 \
  --num-grid-steps 8 \
  --training-steps 2 \
  --batch-size 8 \
  --target-batch-size 8 \
  --hidden-dim 8 \
  --log-every 1 \
  --state-dim 3 \
  --no-scores \
  --no-plots \
  --no-save-tensors
```

By default the launchers add a `benchmark_scores` block to `metrics.json`.
It contains:

- `discriminative_score = abs(accuracy - 0.5)`, where lower means generated
  paths are harder to distinguish from held-out target paths;
- `predictive_*`, where a one-step return predictor is trained on generated
  paths and evaluated on held-out target paths;
- `crps_*`, empirical CRPS for the unconditional generated ensemble, computed
  on path levels and returns.

Use `--score-steps`, `--score-batch-size`, `--score-hidden-dim`, and
`--score-lr` to control the evaluator training budget.
