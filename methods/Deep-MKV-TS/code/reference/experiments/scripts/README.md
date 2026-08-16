# Experiment Scripts

This folder is for thin launchers only.  Scripts should build a concrete
experiment from reusable utilities in `experiments/path_dt_experiments/` and
from the model package in `src/deep_mkv_gen_path_dt/`.

Do not put model logic here.  If a helper is shared by several experiments, put
it in `experiments/path_dt_experiments/` instead.

`run_volatility_diagnostics.py` compares named generated path banks with a
reference tensor at several forecast horizons, including volatility-clustering
ACFs and common low/middle/high prefix-volatility regimes.

`run_heston_stylized_comparison.py` puts several saved path banks on one common
Heston evaluation: return density, tail survival, target-versus-candidate QQ,
and absolute-/squared-return ACF panels. It also writes the corresponding
return-scale, tail, kurtosis, ACF, covariance, and path-SWD errors to JSON.

`run_heston.py --conditional-signal-diagnostic` performs nested future
branching from fixed prefixes after training and saves both aggregate
conditional-signal metrics and path-level diagnostic tensors. Pass
`--signal-discrepancy-blocks` to isolate named weighted composite blocks and
measure their (P/R) alignment and control-effective drift/volatility action.

`run_nested_representation_diagnostic.py` freezes a selected complete-MP
generator, constructs independent nested branch-averaged P/R targets, and
compares causal log-price, standardized-log-price, standardized-log-return,
and hybrid GRU inputs. It does not alter the log-price dynamics, running cost,
discrepancy, or generator checkpoint.

`run_objective_estimator_variance_audit.py` freezes the same generator and
crosses common prefixes/future shocks with disjoint target-law minibatches.
It decomposes the complete P/R teacher into running-cost, base-discrepancy, and
joint prefix/future-discrepancy terms; reports future, target-batch, and
interaction variance shares; and maps each component through the unchanged
Hamiltonian control to its marginal sigma effect. The average over target
partitions is explicitly reported as an estimator average, not as an exact
gradient of one pooled empirical discrepancy.

`run_antithetic_r_audit.py` is the subsequent equal-compute frozen gate. For
each prefix it compares complete-MP R averages from `Z,-Z` with averages from
two independent future shocks, using the same number of paths and the same
target batches. It reports bootstrap variance ratios, branch and split-half
correlations, conditional-mean agreement, and regime-specific sigma changes.
Gaussian symmetry makes the antithetic estimator unbiased; no objective,
running-cost, discrepancy, control, checkpoint, or neural loss is changed.

`run_heston.py --control-map entropy_barrier` replaces the fitted local
specific-entropy reference with the convex running cost
`0.5 * alpha**2 + 0.5 * kappa * sigma**2 - tau * log(sigma)`. The launcher
uses an explicitly supplied `--entropy-tau`, or calibrates only its global
zero-signal scale when that option is omitted.

`run_heston.py --conditional-volatility-discrepancy` adds a joint multiscale
prefix/future volatility MMD. Its windows are specified in original grid steps
and converted to observed-grid windows.

`run_heston.py --conditional-volatility-correlation-discrepancy` instead adds
the direct squared error between prefix/future realized-volatility
correlations. Generated standard deviations are floored at a detached fraction
of the corresponding target scales, configured by
`--conditional-volatility-correlation-scale-floor-fraction`, so the objective
is finite at collapsed dispersion and scale-invariant above the floor. The
default matched windows are 16 and 32 original grid steps;
`--conditional-volatility-correlation-full-matrix` enables every selected
pair. The launcher also reports each discrepancy block's local gradient along
timewise diffusion-scale perturbations; disable that report with
`--no-discrepancy-sensitivity`. The direct correlation has its own
`--conditional-volatility-correlation-weight`, which defaults to 0.025.

`run_heston.py --preset reduced_time_local` removes the redundant terminal,
increment, global-volatility, sigma-like, and ACF proxy blocks. It keeps the
observed-path MMD and adds local RMS-volatility marginal MMDs whose generated
features use the corresponding target-date mean and standard deviation. Use
`--time-marginal-volatility-window` and
`--time-marginal-volatility-endpoints` to configure them; both count returns on
the discrepancy grid, so under observed-only training they are observed-grid
indices. The direct correlation block remains optional and is the second axis
when enabled.

`run_heston.py --save-checkpoint` saves the network together with its fitted
timewise target scales and noise-control-variate baseline. A network
`state_dict` alone is insufficient to reproduce the fitted controls.

`run_heston.py --specific-entropy-r-ablation` performs a common-random-number
intervention after specific-entropy training. It compares the full learned
`P,R` policy with `R=0` and with `P=R=0`, and measures the direct change in
sigma caused by `R` while holding the full rollout's realized prefixes fixed.

`run_specific_entropy_joint_shadow.py` adds one target-standardized joint MMD
of compact causal prefix features and log future realized volatility to that
specific-entropy baseline. It leaves the running cost and adjoint regression
loss unchanged, evaluates post-update checkpoints on a separate validation
law, selects only by validation realized-volatility CRPS, and opens the source
held-out paths only after selection. It writes selected and terminal
checkpoints, fixed validation/test banks, per-path shadowing metrics, paired
bootstrap intervals, stylised-fact guards, and a separate overall gate. The
default command reproduces the seed-1234 pilot:

```bash
python experiments/scripts/run_specific_entropy_joint_shadow.py \
  --run-dir runs/heston_dt_specific_entropy_joint_shadow_seed1234_20260721 \
  --device cuda:3
```

`run_specific_entropy_shadow_frontier.py` repeats that exact deterministic
training trajectory but replaces the one-metric selector with a predeclared
validation frontier. Every 200-step checkpoint and fixed-seed 4,096-path bank
is saved. A checkpoint is eligible only if validation RV CRPS improves,
90% RV coverage moves toward 90%, cumulative-return CRPS stays within one half
of the source's bootstrap standard error, all four stylised-fact errors stay
within the corresponding half-standard-error budgets, and path SWD is no more
than 1.10 times the source value. The eligible checkpoint with the lowest RV
CRPS is selected; if none exists, step zero (the fitted source) is retained.
Only then does the script simulate a new Heston test law and report paired
bootstrap differences. It also writes a four-panel validation-frontier plot:

```bash
python experiments/scripts/run_specific_entropy_shadow_frontier.py \
  --run-dir runs/heston_dt_specific_entropy_shadow_frontier_seed1234_20260721 \
  --device cuda:3
```

For seed 1234, steps 1,200 and 1,400 are eligible and step 1,200 is selected.
On the newly simulated test law, RV CRPS changes from `0.022904` to `0.022645`
(paired 95% interval `[-0.000417, -0.000103]`), while all guarded stylised-fact
errors improve and path SWD falls from `0.02168` to `0.01251`. RV coverage
increases from `0.6816` to `0.6914`, although its paired interval includes
zero. Cumulative-return CRPS worsens slightly from `0.033306` to `0.033597`;
this is inside the predeclared tolerance but statistically detectable under
the paired bootstrap. The composite tolerance gate passes, but the outcome is
therefore a volatility-focused tradeoff rather than a Pareto improvement.

The same launcher can continue from a validation-selected specific-entropy
checkpoint and add formal dated volatility marginals without changing its
running cost, pre-existing joint discrepancy, nested solver, or neural
adjoint loss:

```bash
python experiments/scripts/run_specific_entropy_shadow_frontier.py \
  --initialization frontier \
  --frontier-run-dir runs/heston_dt_specific_entropy_complete_mp_nested_trust100_10x200_seed1234_20260722 \
  --dated-marginal-weight 1.0 \
  --solver nested \
  --nested-outer-steps 10 \
  --nested-inner-steps 200 \
  --nested-outer-relaxation-mode adjoint_blocks \
  --nested-outer-block-trust-fraction 1.0 \
  --grad-clip-norm none \
  --selection-objective coverage_error
```

By default, windows 4/8/16/32 are evaluated at every fourth fine-grid date.
Under observed-only training this becomes 117 valid window/date pairs over 31
running dates and one terminal date. `--dated-marginal-weight` is the total
weight of the entire dated family, divided uniformly over its dates; the
pre-existing complete objective retains weight one.

The locked seed-1234 weight-one continuation is a negative gate. None of its
ten checkpoints passes validation, so step zero is retained. The closest RV
CRPS checkpoint, step 1,000, changes validation RV CRPS from `0.022663` to
`0.022610` but lowers 90% coverage from `0.7871` to `0.7793` and violates SWD
and volatility-ACF guards. Consequently the untouched test metrics remain RV
CRPS `0.022267`, coverage `0.7578`, and cumulative-return CRPS `0.033431`.
Artifacts are under
`runs/heston_dt_specific_entropy_dated_marginal_w1_nested_trust100_10x200_seed1234_20260729/`.

`run_step1600_locked_comparison.py` revisits the earlier high-coverage step
1,600 checkpoint without using the new tests for selection. It freezes that
checkpoint in advance and compares it with the validation-selected joint
specific-entropy model and an independent equal-size Heston oracle. The
default protocol uses five new Heston laws, 4,096 scenarios per bank, 512
prefixes per law, `top_k=256`, and common random numbers between the two MP
models:

```bash
python experiments/scripts/run_step1600_locked_comparison.py \
  --run-dir runs/heston_dt_step1600_locked_fresh5_seed917001_20260729 \
  --device cuda:0
```

The frozen point rule requires coverage strictly closer to 90%, non-worsening
RV CRPS, and non-worsening cumulative-return CRPS. Step 1,600 passes that
literal rule in only one of five individual seeds and fails on the pooled
point estimate. Nevertheless, the pooled result establishes a strong
conditional-calibration response: RV coverage rises from `0.7602` to `0.8605`
versus `0.8797` for Heston, a `+0.1004` paired change with 95% interval
`[+0.0875,+0.1129]`. RV CRPS changes from `0.022217` to `0.021998`
(`-0.000219`, interval `[-0.000480,+0.000059]`). The literal failure is a
`+0.000024` return-CRPS change whose interval
`[-0.000066,+0.000119]` includes zero.

The diagnostic tradeoff also replicates. Mean generated excess kurtosis is
`4.863` versus `3.644` in the five Heston references, and tail-survival error,
path SWD, and squared-return ACF error are all worse than for the selected
specific-entropy checkpoint. The result therefore supports step 1,600 as a
coverage-focused checkpoint, not as an unconditional full-law improvement.

`run_specific_entropy_frontier_bank_sweep.py` performs the matched convergence
check required to interpret that 4,096-bank coverage. It generates nested
4K/16K/65K/262K/1M banks for the fitted source, the frontier-selected model,
and an independent Heston oracle, then evaluates all three on the same 512
fresh prefixes at `top_k=256`. The source and selected model use common random
numbers, and all comparisons receive paired bootstrap intervals:

```bash
python experiments/scripts/run_specific_entropy_frontier_bank_sweep.py \
  --run-dir runs/heston_dt_specific_entropy_frontier_bank_sweep_seed1234_20260721 \
  --device cuda:3
```

The selected model's RV coverage remains between `0.664` and `0.691` across
the sweep, while the Heston oracle remains between `0.863` and `0.887`. At one
million scenarios the values are `0.6738` and `0.8867`; the paired gap is
`-0.2129` with 95% interval `[-0.2500, -0.1777]`. The selected RV band is
`27.2%` narrower and RV CRPS is `0.00145` worse than the oracle. Increasing
the scenario count therefore rejects finite retrieval scarcity as the main
explanation and identifies structural conditional under-dispersion.

`run_rv_shadow_regime_diagnostic.py` consumes the saved matched banks and
decomposes horizon-RV coverage, lower/upper misses, width, CRPS, and forecast
bias by observed-prefix volatility tertile. Regimes are defined once from the
shared real prefixes, never from a candidate-generated bank:

```bash
python experiments/scripts/run_rv_shadow_regime_diagnostic.py \
  --bank-sweep-dir BANK_SWEEP_DIR \
  --real-paths FRONTIER_RUN/fresh_test_reference_paths.pt \
  --run-dir REGIME_DIAGNOSTIC_DIR
```

The shadow-frontier runner accepts `--regime-weight` to add smooth empirical
low/middle/high prefix-regime × future-log-RV moment and tail claims. Use
`--regime-conditional-moment-weight` to add explicit normalized conditional
mean/std errors. Both are fixed path-functional discrepancies; the running
cost, adjoint construction, normalized regression loss, and clipping setting
are unchanged.

`run_hamiltonian_regime_audit.py` is the matching dynamics-side audit. It
freezes source and selected specific-entropy checkpoints on identical initial
states and noise, reconstructs the complete ridge-projected P/R targets, and
reports by fixed target-prefix volatility tertile: discrepancy and running-
source signal, the isolated joint prefix/future block, current and
Hamiltonian-optimal alpha/sigma, entropy costs, stationarity gaps, eta
sensitivity, sigma-bound activity, and accepted source-to-selected control
changes. The saved outer-step row provides separate global line-search context.
No model parameter or neural loss is changed:

```bash
python experiments/scripts/run_hamiltonian_regime_audit.py \
  --frontier-run-dir FRONTIER_RUN \
  --run-dir HAMILTONIAN_AUDIT_DIR \
  --device cuda:3
```

`run_frozen_backward_learnability_audit.py` follows that audit without feeding
another candidate into the dynamics. It builds one fixed backward problem at
the selected law, applies the unchanged P/R loss and fixed-rate AdamW optimizer
for up to 5,000 unclipped steps, and evaluates the network on both the fitting
bank and an independent bank. The fit-bank ridge teacher is transferred to the
independent prefixes, while a second teacher is reconstructed independently;
this separates neural learnability/generalization from Monte Carlo teacher
stability. Metrics are reported globally and by prefix-volatility regime:

```bash
python experiments/scripts/run_frozen_backward_learnability_audit.py \
  --frontier-run-dir FRONTIER_RUN \
  --run-dir FROZEN_BACKWARD_AUDIT_DIR \
  --device cuda:3
```

`run_frozen_ridge_stability_sweep.py` reuses those exact frozen paths and raw
pathwise P/R samples. It independently fits the timewise ridge teacher on the
fit and heldout banks for each lambda, then reports cross-bank P/R agreement,
raw-target projection R², retained conditional variation, and agreement of the
resulting specific-entropy sigma controls. No network or forward dynamics are
trained during the sweep:

```bash
python experiments/scripts/run_frozen_ridge_stability_sweep.py \
  --frontier-run-dir FRONTIER_RUN \
  --learnability-run-dir FROZEN_BACKWARD_AUDIT_DIR \
  --run-dir RIDGE_STABILITY_DIR \
  --device cuda:3
```

`run_frozen_direct_backward_learnability_audit.py` then bypasses the ridge
teacher and fits the same causal network directly to the saved complete
pathwise P samples and unbiased control-variate R samples. Its unscaled arm is
ordinary direct MSE; its fixed timewise-RMS arm preserves each conditional-
expectation population minimizer while balancing time/output coordinates. The
two arms start from the identical selected checkpoint, use the identical
minibatch sequence, have no schedule or gradient clipping, and never feed a
trained diagnostic network back into the dynamics:

```bash
python experiments/scripts/run_frozen_direct_backward_learnability_audit.py \
  --frontier-run-dir FRONTIER_RUN \
  --frozen-bank-run-dir FROZEN_BACKWARD_AUDIT_DIR \
  --run-dir FROZEN_DIRECT_AUDIT_DIR \
  --device cuda:3
```

The report records full optimization trajectories, fit and independent-bank
raw-target metrics, held-out R-squared against fit-bank-only global and regime
baselines, target scales, gradient norms, and controlled-to-uncentered R
second-moment ratios. Saved networks are diagnostic artifacts, not generator
checkpoints.

Pass `--r-control-variate-mode crossfit_knn` to replace the selected-P control
variate with a prefix-dependent conditional variance-minimizer. The default
uses five deterministic folds, 128 neighbors, and rolling-volatility windows
4/8/16/32. Fit queries exclude their entire fold; independent queries use only
the fit bank. Configure these with `--control-variate-folds`,
`--control-variate-neighbors`, and
`--control-variate-volatility-windows`. This estimator changes the Monte Carlo
variance of R but not its conditional expectation.

`run_frozen_r_representation_isolation.py` consumes the saved cross-fitted
direct R targets and compares two no-feedback fits: the selected R head with
the GRU frozen, and an independent R-only copy of the selected GRU plus R head.
Neither arm has a P loss, schedule, or gradient clipping; both use the same
minibatch sequence and fixed timewise scale. Fit-loss selection and independent
R-squared trajectories distinguish head capacity, shared-gradient
interference, and representation overfit:

```bash
python experiments/scripts/run_frozen_r_representation_isolation.py \
  --frontier-run-dir FRONTIER_RUN \
  --frozen-bank-run-dir FROZEN_BACKWARD_AUDIT_DIR \
  --direct-target-run-dir FROZEN_DIRECT_CROSSFIT_R_DIR \
  --run-dir R_REPRESENTATION_ISOLATION_DIR \
  --device cuda:3
```

`run_rv_shadow_shock_diagnostic.py` determines whether that RV failure is an
artifact of one large return dominating each horizon. On the one-million-path
banks it reports ordinary RV, RV after removing the largest squared return,
bipower volatility, and maximum-absolute-return coverage at horizons 8, 16,
and 32, together with miss directions and largest-return shares:

```bash
python experiments/scripts/run_rv_shadow_shock_diagnostic.py \
  --run-dir runs/heston_dt_specific_entropy_rv_shock_diagnostic_1m_seed1234_20260721
```

At horizon 32, specific-entropy coverage is `0.6738` for ordinary RV, `0.6543`
after removing the largest return, and `0.6465` for bipower volatility; the
matching Heston values are `0.8867`, `0.8887`, and `0.8965`. Hence isolated
shocks do not explain the miss. The specific-entropy law also puts only 19.2%
of future squared variation in its largest return on average, versus 23.9% in
both the real continuations and Heston shadows. Its 101 lower misses and 66
upper misses show compression toward intermediate future volatility rather
than a one-sided failure to generate high volatility.

The model core now also differentiates the path-dependent running cost in the
adjoint source. Re-running the joint specific-entropy launcher under the new
core keeps the neural regression loss unchanged and writes complete-source
diagnostics:

```bash
python experiments/scripts/run_specific_entropy_joint_shadow.py \
  --device cuda:0 \
  --run-dir runs/heston_dt_specific_entropy_complete_mp_dt_joint_shadow_seed1234_20260721
```

The running callable retains the `dt` factor in the paper's stage cost even
though the displayed Hamiltonian control formula is divided by `dt`. The
selected step 1,000 gives matched 4,096-bank RV coverage `0.7285`, versus
`0.7480` for the old reduced joint-discrepancy run and `0.8789` for Heston.
The complete-minus-reduced RV-CRPS and coverage intervals both include zero.
Running-source RMS is only 0.0855% of discrepancy-source RMS at the selected
checkpoint, while clipping remains active on 99.55% of updates. Treat this run
as the faithful-formulation and stabilization baseline, not as evidence that
the added running derivative has already improved shadowing.

The validation-frontier runner accepts `--grad-clip-norm source|none|FLOAT`
and `--selection-objective coverage_error`.  A controlled seed-1234 sweep at
caps 5, 10, 25, and without clipping keeps the complete-MP objective, data,
initialization, banks, and guards fixed.  The cap-10 run is the only one with
an admissible checkpoint: validation selects step 1,000 with RV CRPS `0.022862`,
coverage `0.7832`, and excess-kurtosis error `0.0306`.  On the newly simulated
test law it improves RV CRPS from `0.022904` to `0.022346` and coverage from
`0.6816` to `0.7520`; paired 95% intervals are
`[-0.000822,-0.000314]` and `[0.04883,0.09180]`.  Return-CRPS change is not
significant, and every stylized guard passes.  Caps 5 and 25 and the uncapped
run select the source because no updated checkpoint passes every guard.

The cap sweep is a sensitivity diagnostic, not a principled way to choose the
learned law.  Cap 5 clips
99.55% of updates and retains a median 17.1% of raw gradient magnitude; cap 10
clips 93.2% and retains 34.9%; cap 25 clips 58.7% and retains 87.2%; the
uncapped trajectory remains finite but becomes inadmissible.  Raw gradients
are dominated by the ordinary adjoint head, whereas the largest AdamW update
norm is in the GRU.  Full-run parameter-update norms are similar and not
monotone in the cap.  Thus cap 10 happens to expose a valid checkpoint, but
does not supply a cap-independent training result.
Artifacts are in the four
`heston_dt_specific_entropy_complete_mp_dt_clip*_frontier_seed1234_20260721`
run directories.

`run_uncapped_trajectory_audit.py` analyzes the saved uncapped trajectory
without retraining.  It joins the logged NN, source, and control diagnostics
to the common-noise validation banks, then replays every checkpoint on one
additional common-noise bank to recover timewise control profiles:

```bash
python experiments/scripts/run_uncapped_trajectory_audit.py \
  --frontier-run-dir runs/heston_dt_specific_entropy_complete_mp_dt_clipnone_frontier_seed1234_20260721 \
  --run-dir runs/heston_dt_specific_entropy_complete_mp_dt_clipnone_trajectory_audit_seed1234_20260721 \
  --device cuda:3
```

The uncapped run is numerically finite but is not learning the projected
conditional adjoints.  Across checkpoints, the network's `P` RMS is only
0.092--0.486% of projected-`P` target RMS and `R` is 0.590--2.148%; both
relative RMSEs remain essentially one while parameter norm grows by 2.83x.
The running path source remains only 0.083--0.219% of discrepancy-source RMS.
No single checkpoint optimizes the reported laws: steps 600, 1,800, and 2,000
respectively minimize RV CRPS, kurtosis error, and distance to nominal
coverage.

This run has no optimizer gradient clipping, but its specific-entropy control
still has the declared admissible upper bound `sigma_max=0.6`.  The common
replay hits that bound at every checkpoint.  Upper-bound occupancy has 0.906
exploratory correlation with kurtosis error across the ten checkpoints; its
late-path component has correlation 0.902.  Adjacent common-noise return and
RV correlations remain near one, so the tail jumps reflect smooth movement of
where and how often the volatility control reaches its bound, not a numerical
explosion.  The next target is therefore conditional-adjoint learnability and
the interaction with the declared volatility admissible set, not selection of
another gradient cap.

`run_nested_directional_diagnostic.py` fits one genuinely frozen backward
problem without clipping, then evaluates the complete objective with common
random numbers across positive/negative parameter relaxations, separate `P`
and `R` interventions, sign reversals, and the plausible alternative `dt`
scalings:

```bash
python experiments/scripts/run_nested_directional_diagnostic.py \
  --device cuda:3 \
  --run-dir runs/heston_dt_specific_entropy_complete_mp_nested_directional_seed1234_20260722
```

It also differentiates the objective through an open-loop control replay and
compares that gradient directly with the reverse-summed pathwise targets. This
is the preferred audit for suspected off-by-one, sign, or Brownian-scaling
errors. The seed-1234 result finds exact drift-chain agreement and `5.2e-8`
relative noise-chain error. The implemented direction is locally descending,
while reversed `R` is ascending. `R` supplies about 82% of the initial descent;
the full-step failure is dominated by the fitted drift correction. The output
contains `metrics.json`, `diagnostic_artifacts.pt`, and
`objective_curves.png`.

`run_drawdown_adjoint_deployment_surface.py` applies the same audit to the
retained delayed-memory model but separates the old-to-candidate `P` and `R`
outputs directly, rather than only interpolating network parameters. It
records the complete objective, discrepancy, running-cost components, drift
change, log-volatility change, specific-entropy denominator quantiles, and
volatility-bound activity on a two-dimensional common-random-number surface.
It also sends the projected conditional targets directly through the frozen
Hamiltonian map, isolating local pointwise minimization from recursive law
refresh. `--target-estimator nested_branches --conditional-branches 128`
repeats the candidate fit with high-branch conditional targets, and
`--evaluation-batch-size` permits an independent larger confirmation bank.

`run_real_data_nested_signal_diagnostic.py --reference-only` audits the
zero-adjoint fitted reference under the same train-only branched-future
protocol. Passing the selected first-stage checkpoint instead gives a matched
reference-versus-MP comparison. Both stages use the unbiased adjoint `R`
control variate so that the reference audit does not depend on a baseline that
is calibrated only during MP training.

`run_specific_entropy_shadow_frontier.py --solver nested
--nested-outer-relaxation-mode adjoint_blocks` turns that diagnosis into a
numerical solver experiment. It keeps the frozen backward solve intact and
selects separate outer rates for the P head, R head, and shared GRU on the
unchanged complete empirical objective. The seed-1234 ten-sweep audit remains
finite without clipping and improves final validation RV coverage from `0.545`
under joint relaxation to `0.738`, but all blockwise checkpoints have worse RV
CRPS than the source and fail the declared gate. The accepted fixed-target loss
ratios are `0.9985`--`0.9999`, showing that stability is obtained by retaining
almost none of the learned backward correction rather than by reaching a useful
forward--backward fixed point.

`run_nested_holdout_diagnostic.py` separates outer-objective generalization
from path-law generalization. It reconstructs the exact frozen fitting bank,
creates an independent held-out path/target/noise bank, cross-scores joint and
P/R/shared relaxation curves, and evaluates common-noise scenario banks with
the existing shadowing/stylized gate. By default it warm-starts from the saved
specific-entropy checkpoint; `--initialization fresh` reproduces the earlier
from-scratch setup. On seed 1234 the full warm-start block update improves RV
CRPS and coverage but fails only kurtosis. A one-eighth fraction of the same
update lowers both complete objectives and passes every declared gate, with RV
CRPS `0.024296 -> 0.024219`, coverage `0.703 -> 0.717`, and kurtosis error
`0.057 -> 0.053`. The runner then locks `--locked-scale 0.125` before opening a
newly simulated Heston test law. On seed 1234, fresh RV CRPS improves
`0.022904 -> 0.022868` with a paired 95% interval excluding zero, coverage
improves `0.6816 -> 0.6895`, and the full fresh-test gate passes.

The same trust fraction is available directly in
`run_specific_entropy_shadow_frontier.py` through
`--nested-outer-block-trust-fraction`; combine it with `--initialization source`
and `--nested-outer-relaxation-mode adjoint_blocks`. Three-seed fixed one-step
runs pass all fresh gates and have paired RV-CRPS intervals below zero in every
seed. Ten-step validation-selected runs amplify the RV gains but pass only two
of three fresh gates because one late checkpoint loses kurtosis fidelity.
`summarize_nested_trust_multiseed.py` writes the aggregate JSON and comparison
plot from the per-seed run directories.

`run_heston.py --frozen-response-audit` evaluates every weighted discrepancy
block after training. It compares direct path-space descent with symmetric
common-noise perturbations along each block's additive
`J_theta^T(P_b, R_b)` target-forcing direction. The full neural row also
includes the non-discrepancy noise-control-variate and shared
prediction-shrinkage terms, so it equals the frozen adjoint-MSE descent
direction up to a common factor. The report contains the block-by-metric
response matrix, pairwise direction cosines, raw and row-normalized spectra,
and both JSON and tensor artifacts. When the training discrepancy does not
already contain the direct correlation block, the default audit adds it as an
untrained candidate; disable this with `--no-response-correlation-candidate`.

Use `run_frozen_response_audit.py` to repeat the diagnostic from a saved Heston
checkpoint without retraining. For example:

```bash
python experiments/scripts/run_frozen_response_audit.py \
  --source-run-dir runs/Heston_baseline \
  --run-dir runs/Heston_baseline_response \
  --device cuda:0 \
  --parameter-step-fraction 1e-5
```

The default parameter fraction `1e-5` is deliberately local. The response
should be repeated at a nearby value such as `3e-5`; agreement of the central
derivatives is the finite-difference linearity check.

`run_residual_r_persistence.py` is the go/no-go experiment that follows the
reduced-objective audit. It freezes the fitted base generator and its ordinary
adjoint output, adds a zero-initialized causal residual only to the
noise-adjoint `(R)` channel, and trains that residual against one fixed
full-training-bank prefix/future volatility-correlation target. The target
uses the same fine-grid horizon as the reported persistence metric, rather
than the correlation of aggregated observed-only increments. Checkpoints are
measured after their optimizer updates on a separately simulated validation
bank. Selection requires improved correlation without worsening h32 mean,
standard-deviation, or IQR error and limits SWD growth; the source held-out bank
is read only after selection.

In the seed-1234 gate, the best unconstrained validation checkpoint reduced the
h32 correlation error by 90.2% but collapsed the h32 RV standard-deviation and
IQR ratios from `0.960/0.796` to `0.660/0.615`. The constrained selector kept
step 0, so the experiment is a negative gate for an unconstrained residual and
positive evidence that the `R` channel can carry the requested signal.

The optional `--marginal-anchor-weight` adds fixed full-training-bank
intermediate volatility marginals for the horizons in
`--marginal-anchor-horizons`. For each horizon, prefix and future RMS-
volatility empirical quantile functions are compared independently with a
standardized Wasserstein discrepancy. The controlled equal-weight run
(`0.025` for both marginal and correlation blocks) also retained step 0: the
terminal h32 correlation moved from `0.7079` to `0.7367`, away from the Heston
value `0.5397`. The block targets are nearly orthogonal in noise-adjoint space
(cosine `-0.027`) but collapse in the residual GRU parameter space (gradient
cosine `0.974`). `block_direction_diagnostics.json` records this geometry for
the baseline, terminal, selected, and best-correlation checkpoints.

`run_constrained_feasibility_audit.py` performs the next no-training gate. It
forms end-to-end residual-parameter gradients for the h32 correlation error
and separate h32 RV mean, standard-deviation, and IQR errors, projects
correlation descent onto their linearized non-worsening half-spaces, and
verifies both signs with common-noise central finite differences. Use
`--audit-dtype float64` (the default) for the very small projected direction;
the Gaussian bank is always drawn in float32 before casting so precision does
not change the sample. The seed-1234--1238 audit retained only
`0.042%--0.241%` of unconstrained descent in every case, far below the `10%`
materiality gate, despite finite-difference agreement. The current residual
parameterization therefore has no meaningful local direction that improves
persistence while preserving those marginals.

At the exact zero-output initialization, the support report shows that only
the final residual readout is active: `97/37,921` parameters. A centered
zero-output residual with a frozen, input-differentiable anchor is available as
`--residual-parameterization centered_full_tangent`. It preserves the baseline
while making the complete GRU Jacobian available to the same audit without
changing the adjoint-MSE training loss.

All `37,921` trainable parameters are active under this parameterization, but
the five 2,048-path audits retain only `0.078%--0.402%` of correlation descent
(mean `0.251%`). Five centered initialization seeds on one fixed path bank
retain `0.321%--0.340%`. All finite differences pass and every local gate
fails. Consequently, only a one-step integration smoke test was run; a full
centered-residual training launch is not supported by the feasibility result.

`run_causal_r_basis_feasibility.py` bypasses that learned representation with
a direct model-free causal basis in `R`: rolling log realized volatility at
windows 4/8/16/32, quadratic terms, pairwise interactions, and an intercept,
all standardized on an unperturbed generated bank. The entropy-barrier control
and maximum-principle quantities remain unchanged. It reports exact
mean/std/IQR projection geometry, bootstrap-calibrated Pareto budgets, finite
differences, and time/feature localization.

Across seeds 1234--1238, exact non-worsening retains `74.35%--85.05%` of
correlation descent; a half-standard-error budget retains `93.85%--99.97%`,
and one standard error retains 100%. On seed 1234, a finite step removes 9.6%
of correlation error while changing mean error by only `+0.00034` and
improving standard-deviation and IQR errors. This is a positive control-space
gate: the bottleneck is the learned residual representation, not scalar
`R -> sigma` expressiveness.

`run_causal_r_basis_training.py` is the controlled signal-transmission test
that follows that feasibility result. It installs the same 1,920-parameter
basis as an additive `R` skip, freezes the complete fitted GRU base, and trains
only the basis coefficients with the existing normalized adjoint MSE. The
fixed correlation and time-marginal MP discrepancies are both weighted
`0.025`; the neural loss, entropy-barrier control, and maximum-principle
construction are unchanged. A fixed `2e-4` AdamW learning rate keeps the first
coefficient update near the finite-step scale checked by the audit; no
schedule or gradient clipping is used.

```bash
python experiments/scripts/run_causal_r_basis_training.py \
  --source-run-dir runs/heston_dt_entropy_barrier_k25_reduced_time_local_corr_w0p025_seed1234_20260718 \
  --run-dir runs/heston_dt_causal_r_basis_training_seed1234_20260721 \
  --device cuda:3 \
  --seed 1234
```

Selection is validation-only. A checkpoint must reduce h32 correlation error
by at least 10%, keep each h32 mean/std/IQR error increase below half of its
baseline bootstrap standard error, and keep SWD growth below 10%. The held-out
bank is evaluated only after selection. Rejected runs also save and report the
held-out response of the validation-best correlation checkpoint, but never use
it for selection.

Across seeds 1234--1238, the adjoint MSE falls to `28.7%--44.6%` of its first-
step value. The marginal/correlation parameter-gradient cosine is
`-0.353`--`-0.243` (mean `-0.296`), resolving the `+0.974` GRU representation
collapse. The best-correlation checkpoints reduce held-out correlation error
by `8.0%--29.4%` (mean `17.2%`) and satisfy the half-SE marginal constraints in
four of five seeds. Nevertheless, only seeds 1237 and 1238 pass both the
validation selection and held-out confirmation gates. This is positive
evidence that the unchanged MP signal reaches the useful basis, but not yet a
robust five-seed result.

Pass `--marginal-formulation multi_marginal` to replace the global eight-
statistic anchor by the formal family of dated running law costs. With the
defaults, local realized-volatility laws at windows 4/8/16/32 are fitted at
every fourth fine-grid date. This gives 31 non-terminal running costs and one
terminal marginal cost, covering 117 valid window/date pairs. The same total
marginal weight `0.025` is distributed uniformly over the 32 dates, while the
terminal prefix/future correlation weight remains `0.025`:

```bash
python experiments/scripts/run_causal_r_basis_training.py \
  --source-run-dir runs/heston_dt_entropy_barrier_k25_reduced_time_local_corr_w0p025_seed1234_20260718 \
  --run-dir runs/heston_dt_causal_r_basis_multimarginal_seed1234_20260721 \
  --device cuda:3 \
  --marginal-formulation multi_marginal \
  --seed 1234
```

The running blocks are evaluated only on their dated path prefixes. Their
Lions-gradient sources therefore enter the existing reverse cumulative
adjoint recursion at the correct dates. The normalized adjoint MSE and
entropy-barrier control remain unchanged. For an identical collection of
dated costs, this explicit formulation is mathematically equivalent to
differentiating their global sum; the empirical improvement comes from adding
the dense intermediate marginal information, not from a different autograd
identity.

The five-seed result passes the complete validation-plus-held-out gate in four
seeds. Selected held-out correlation-error reductions are `64.1%`, `24.6%`,
`63.8%`, `58.1%`, and `16.0%` (mean `45.3%`). Seed 1238 is retained as a
failure: its standard-deviation error increases by `0.00841` against a half-SE
allowance of `0.00801`. No gate or weight was changed after seeing the result.

`run_drawdown_memory_deep_mkv_ablation.py` is the controlled extension for the
delayed drawdown-memory toy law. It fits drawdown thresholds from training
quantiles, uses their decayed causal gates in both the specific-entropy
reference and the backward conditional-expectation basis, and optionally adds
a cross-fitted residual early-drawdown/future-volatility moment. Its three arms
separate the causal reference/CE basis, the residual moment under online
training, and the same objective under nested forward-backward training. The
neural adjoint-regression loss and the specific-entropy running cost are
unchanged. Model selection is evaluated on `disc.npy`; `test.npy` remains
unseen.

The optional `generic_ce_proximal_fixed_point` arm addresses finite
forward--backward deployment directly. At every outer iteration it recomputes
the ordinary projected MP targets under the current generated law, takes a
proximal step in actual control coordinates (linear in drift and geometric in
positive volatility), maps that step back to equivalent `P/R` regression
targets, and accepts it only on independent objective banks. The running cost,
path discrepancy, and MP stationary points are unchanged. For example:

```bash
python experiments/scripts/run_drawdown_memory_deep_mkv_ablation.py \
  --source-run-dir runs/paper_drawdown_fully_generic_4seed_20260801/seed_0 \
  --run-dir runs/drawdown_proximal_seed0 \
  --generic-reference \
  --arms generic_ce_proximal_fixed_point \
  --noise-target-control-variate adjoint \
  --proximal-alpha-fraction 0.03125 \
  --proximal-log-sigma-fraction 0.001953125 \
  --device cuda:0 \
  --seed 0
```

The optional `generic_ce_safeguarded_extragradient` arm instead treats the
coupled forward--backward equations as the fixed-point residual
`F(z)=z-T(z)`. It fits `T(z)` with the unchanged adjoint MSE, deploys a
control-impact-bounded predictor, refreshes the targets under the predictor
law, and applies the resulting corrector from the original iterate. Drift and
log-volatility have separate trust radii; the volatility step is also weighted
by a datewise split-sample reliability estimate for `R`. An update is accepted
only when an independent stationarity probe improves, the specific-entropy
denominator stays admissible, and common-random-number objective replays
remain safe. The complete objective is a safeguard, not a neural training
loss. For example:

```bash
python experiments/scripts/run_drawdown_memory_deep_mkv_ablation.py \
  --source-run-dir runs/paper_drawdown_fully_generic_4seed_20260801/seed_0 \
  --run-dir runs/drawdown_extragradient_seed0 \
  --generic-reference \
  --arms generic_ce_safeguarded_extragradient \
  --noise-target-control-variate adjoint \
  --extragradient-alpha-radius 0.05 \
  --extragradient-log-sigma-radius 0.0025 \
  --device cuda:0 \
  --seed 0
```

`generic_ce_continuation_extragradient` follows the same stationarity system
from the exactly known reference solution. It resets the two output heads so
that `P=R=0` at `beta=0`, then solves
`J_beta = running_cost + beta * lambda * discrepancy` on the locked schedule
`1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1`. Failed stages are bisected down to a
minimum increment of `1/4096`, and only completely accepted stages are carried
forward. Optimizer state is rebuilt at every stage. Stage acceptance uses a
common-probe relative stationarity residual and independent `J_beta` banks;
neither a proximal term nor downstream validation metrics enters the
financial objective. The runner does not evaluate a scenario bank unless the
continuation reaches `beta=1`.

```bash
python experiments/scripts/run_drawdown_memory_deep_mkv_ablation.py \
  --source-run-dir runs/paper_drawdown_fully_generic_4seed_20260801/seed_0 \
  --run-dir runs/drawdown_continuation_seed0 \
  --generic-reference \
  --arms generic_ce_continuation_extragradient \
  --noise-target-control-variate adjoint \
  --extragradient-alpha-radius 0.05 \
  --extragradient-log-sigma-radius 0.0025 \
  --continuation-stage-outer-steps 2 \
  --device cuda:0 \
  --seed 0
```

```bash
python experiments/scripts/run_drawdown_memory_deep_mkv_ablation.py \
  --dataset-root runs/drawdown_memory_seed0_comparison_20260730/data \
  --run-dir runs/drawdown_memory_deep_mkv_ablation_seed0_20260730 \
  --device cuda:2 \
  --seed 0
```

`run_heston_mixture_discrepancy_audit.py` and
`run_heston_mixture_deep_mkv_ablation.py` implement the corresponding
variable-parameter Heston experiment. The discrepancy is fitted on training
prices only and compares the joint law of return shape, multiscale
volatility-level/vol-of-vol summaries, pathwise volatility persistence,
leverage, terminal return, and drawdown. It never uses Heston parameters,
regime labels, or oracle probabilities. The audit must first show an
independent-mixture floor and finite generated-path gradients:

```bash
python experiments/scripts/run_heston_mixture_discrepancy_audit.py \
  --experiment-root runs/heston_parameter_mixture_seed0_20260730
```

The ablation reuses the frozen source checkpoint and separates four changes:
the new discrepancy under the original clipped online solver; removal of
clipping; the compact causal observable CE basis; and nested
forward-backward training. All arms retain the specific-entropy running cost,
the same MP adjoint construction, and the same neural regression loss.
Selection uses `disc.npy`, not `test.npy`.

```bash
python experiments/scripts/run_heston_mixture_deep_mkv_ablation.py \
  --experiment-root runs/heston_parameter_mixture_seed0_20260730 \
  --run-dir runs/heston_mixture_deep_mkv_ablation_seed0_20260730 \
  --device cuda:2 \
  --seed 0
```
# Frozen real-data generation

The real-data adapters consume only the training rows declared by
`real_data_protocol.py` and write normalized-price banks using the frozen
`<root>/<protocol>/<index>/seed_<seed>.npy` contract.

Deep-MKV-Gen uses the retained specific-entropy specification with the joint
prefix/future realized-volatility discrepancy.  Its 127 intraday increments
span one normalized trading session.  The absolute Heston volatility-
dispersion gate is replaced at the adapter boundary by a train-scale rule:
one quarter of the pooled training return standard deviation divided by
`sqrt(1/127)`.  No validation, test, or event row enters this fit or reference
calibration.

```bash
source /home/samer/venvs/mfc/bin/activate
python experiments/scripts/run_real_data_deep_mkv.py \
  --protocol primary_2026_holdout \
  --index ES \
  --seed 0 \
  --device cuda:1
```

SBTS calls the official external adapter without changing its formula:

```bash
source /home/samer/venvs/mfc/bin/activate
python experiments/scripts/run_real_data_sbts.py \
  --protocol primary_2026_holdout \
  --index ES \
  --seed 0 \
  --workers 32
```

Seeds 0--4 are independent fits for Deep-MKV-Gen.  Because SBTS has no fitted
parameters, its seeds are independent stochastic generation repetitions from
the same frozen training set.  The `us_iran_2026_war` protocol resolves to the
primary 2026 bank and therefore never triggers another fit.
The continuation arm starts from the exact zero-adjoint reference and follows
the locked discrepancy schedule to beta one.  Passing
`--extragradient-transition-kl-rho 1` enables the phase-2 conditional
transition-KL proximal proposal.  This KL is excluded from `J_beta` and from
all acceptance-objective values; it changes only the numerical proposal.

`run_terminal_fixed_point_diagnostic.py` freezes the last accepted continuation
checkpoint and evaluates the operator at the first rejected beta. It separates
conditional-target estimator noise from policy bias on common prefixes, fits
diagnostic-only GRU projections to estimate the attainable residual floor, and
estimates `J_T` in residual-informed P/R time blocks with common-random central
finite differences. The diagnostic never deploys its fitted projection models
and does not evaluate downstream output-law metrics.

`run_lifted_collocation_probe.py` is the no-update prerequisite for the lifted
law--adjoint consensus solver. It freezes a compact fit bank and an independent
confirmation bank containing data indices, base noise, and a chunk-invariant
conditional-branch seed schedule. Generated paths, candidate population paths,
and conditional targets are always recomputed under the evaluated causal
policy. The probe compares the accepted law checkpoint with the independently
fitted diagnostic adjoint network, records CE and control-consensus residuals,
and verifies checkpoint non-mutation. It performs no Gauss--Newton, dual, law,
or acceptance update.

```bash
python experiments/scripts/run_lifted_collocation_probe.py \
  --run-dir runs/lifted_collocation_probe_seed0 \
  --device cuda:0 \
  --paths 64 \
  --target-paths 256 \
  --branches 8 \
  --query-batch-size 4096
```
### Lifted directional active-set audit

`run_lifted_directional_active_set_audit.py` replays the locked seed-0 lifted
collocation banks and evaluates symmetric positive/negative perturbations in
three drift and three log-volatility time blocks.  It reports constrained
projected-KKT residuals, active-set flips, objective response, transition KL,
and independent fit/confirmation agreement.  It never deploys a candidate.

`run_lifted_multiradius_derivative_audit.py` refines that audit at decreasing
perturbation radii. It differentiates signed residual vectors, compares
one-sided columns under common randomness, and uses time-aligned signed
profiles for independent-bank agreement. It is a no-update gate for choosing
between an orthant model and a derivative-free trust region.

`run_lifted_poll_trust_region_cycle.py` performs the resulting one-update
experiment. It polls the twelve signed block coordinates, selects on a fit
bank, confirms independently, never shrinks below the locked radius floor,
and permits at most one law move. A confirmed law is saved separately, its
adjoint is refit with the unchanged P/R loss, and the complete lifted cycle is
audited on fresh banks without overwriting the source checkpoint.

`run_lifted_joint_deployment_surface.py` is the final no-update lifted-solver
diagnostic. On fixed generated laws it interpolates the denormalized outputs of
the old and refitted adjoints over a locked tau grid, reusing one conditional
target evaluation per law and bank. A small `(c, tau)` surface is evaluated
only when the accepted-law tau line contains a nonzero improvement on both
independent banks.

`run_hamiltonian_gradient_equal_kl_audit.py` compares full Hamiltonian,
raw-gradient, and exact transition-KL mirror-gradient actor moves at equal fit-bank KL
budgets. It covers early/middle/late drift and volatility blocks, applies the
fit parameter unchanged on confirmation, regenerates candidate-law targets,
and reports deployed-critic and refreshed-target KKT residuals separately.

`aggregate_hamiltonian_gradient_equal_kl_audit.py` verifies that independently
run direction families share banks and fitted adjoints, then compares objective
descent, refreshed stationarity, KL transfer, and direction geometry.

`run_early_volatility_complete_cycle.py` performs the single locked follow-up:
the full-control early-volatility move at the middle equal-KL budget, followed
by an unchanged-loss adjoint refit and a fresh two-bank complete lifted audit.

`run_curvature_matched_running_cost_audit.py` fixes the physical law and compares
fresh adjoints under specific entropy and a volatility-only Gaussian transition
KL with identical local log-volatility curvature. A locked phase-1 gate controls
whether spectral and complete-cycle diagnostics are permitted.

`run_conditional_target_channel_audit.py` runs the locked, no-update
prefix/population/conditional-future factorial decomposition at the rejected
continuation checkpoint. It uses the prespecified early-volatility proposal,
common random numbers within each bank, independent fit/confirmation banks,
componentwise Lions sources, and a predeclared 50% two-bank dominance rule.

`run_residual_volatility_signal_derivative_audit.py` is the locked first gate
for the redesigned volatility objective. It fits a generic and an
oracle-information task-aware causal reference on a calibration partition,
selects the two-dimensional residual-RV statistic or its predeclared generic
drawdown augmentation using a disjoint train-only audit partition, and checks
finite differences, frozen reference parameters, target detachment, and the
Lions/direct MMD source identity before any solver or training is allowed.

`run_residual_volatility_first_operator_audit.py` is the second locked gate. It
starts the current objective, the residual conditional-RV objective, and its
four-times volatility-penalty variant from one exact zero-adjoint reference,
fits fresh adjoints with the unchanged P/R loss, and compares conditional-fit
quality, induced volatility movement, bound activity, regenerated-target
alignment, KKT residuals, and transition KL on independent banks. The
locked decision uses target-relative P/R errors, nonnegative R explained
energy, and exact fitted-versus-target control consensus; raw adjoint or KKT
scales never determine acceptance. The generated innovations subtract the
frozen causal reference's fitted drift, while derivatives through both fitted
drift and volatility are retained on generated paths. The
`--variants` option permits the three arms to run concurrently; the partial
reports are combined by
`aggregate_residual_volatility_first_operator_audit.py`, which verifies the
common initialization, dataset, complete scenario-bank fingerprints, and
effective settings before applying the same locked gate.

`run_residual_volatility_reduced_spectral_audit.py` is available only for a
variant that passes the first-operator gate. It estimates the six-dimensional
early/middle/late P/R operator spectrum with replicated common-random central
differences. A failed first-operator gate forbids this diagnostic and all
subsequent model training.

### Exact finite-tree MP diagnostic

`run_exact_scenario_tree.py` removes sampling, nested conditional estimation,
regression, and neural approximation from the delayed-memory control problem.
It enumerates a complete binary tree, optimizes every node control
simultaneously, and verifies the discrete-time MP equations using an
independently implemented exact backward recursion. The report also includes
the reduced-control Hessian, the full equality-constrained KKT condition
number, volatility-bound activity, and delayed-memory recovery. Run the full
and volatility-only admissible-control problems separately:

```bash
python experiments/scripts/run_exact_scenario_tree.py \
  --run-dir runs/exact_scenario_tree_n8/full \
  --horizon 8 \
  --mode full \
  --device cuda:0

python experiments/scripts/run_exact_scenario_tree.py \
  --run-dir runs/exact_scenario_tree_n8/volatility_only \
  --horizon 8 \
  --mode volatility_only \
  --device cuda:1
```

The three locked starts are the fitted time-marginal reference, the exact
target-tree controls, and their midpoint. Horizon scaling is permitted only
after the `N=8` result identifies whether the obstruction is existence,
conditioning, or objective fidelity.

`evaluate_exact_scenario_tree.py` then evaluates reference, exact MP, and one
frozen-law MP stage on a prespecified panel of distributional, volatility,
dependence, delayed-memory, and local-calibration errors. It reports the
fraction of reference error removed by each correction. The one-stage law uses
one exact adjoint evaluation at the reference, the bounded Hamiltonian
best-response direction, and a logarithmic-grid-plus-local scalar objective
gate that explicitly includes the unchanged reference endpoint.

```bash
python experiments/scripts/evaluate_exact_scenario_tree.py \
  --run-dir runs/exact_scenario_tree_n8 \
  --device cuda:0
```

`run_exact_tree_scalable_comparison.py` performs the matched approximation
audit on that same enumerated `N=8` tree. A zero-output causal GRU exactly
reproduces the volatility reference. Nested MP trains it against exact
nodewise conditional `R` targets in eight frozen outer stages; direct pathwise
training differentiates the identical complete objective through the same GRU
and Hamiltonian map. Both arms use 1,600 AdamW steps, the same seed-specific
initial state, and exact population evaluation. The nested objective gate
includes the unchanged policy, so accepted objectives are monotone.

Run one seed with:

```bash
python experiments/scripts/run_exact_tree_scalable_comparison.py \
  --exact-run-dir runs/exact_scenario_tree_n8 \
  --run-dir runs/exact_tree_scalable_n8/seed_0 \
  --seed 0 \
  --device cuda:0
```

After running the locked seeds `0 1 2 3`, aggregate stability, resources, and
the fraction of exact reference-relative improvement captured with:

```bash
python experiments/scripts/summarize_exact_tree_scalable_comparison.py \
  --run-dir runs/exact_tree_scalable_n8 \
  --seeds 0 1 2 3
```
