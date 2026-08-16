# deep-mkv-gen-path-dt

Reusable discrete-time path-dependent maximum-principle model package.

This repository is intentionally focused on the model layer. Experiments,
benchmarks, plots, and launch scripts should live outside the model core.

## Developer Handover

The mathematical design, exact training data flow, extension points, experiment
presets, operational boundaries, and a runnable minimal example are documented
in [`notes/technical_note.pdf`](notes/technical_note.pdf). The editable source is
[`notes/technical_note.tex`](notes/technical_note.tex).

## Development

Install in editable mode from this directory:

```bash
python -m pip install -e ".[test]"
```

Run the package tests:

```bash
pytest -q
```

## Model Contract

The core model trains conditional adjoint moments for the complete
path-dependent discrete-time maximum-principle source. For supplied controls,
the backward target differentiates both the running cost and the path-law
discrepancy with respect to the generated path. Controls are held fixed in this
partial derivative; the neural network is still trained only by adjoint-moment
regression.

- `state_dim` is the path dimension.
- `noise_dim` is the dimension of the driving noise.
- `P_n` has dimension `state_dim`.
- `R_n` has dimension `state_dim` for diagonal controls when
  `noise_dim == state_dim`, or `state_dim * noise_dim` for full-matrix
  noise controls.
- If `adjoint_dim` is provided, it must equal `state_dim`.
- If `noise_adjoint_dim` is provided, it must satisfy the `R_n` convention
  above.

The model follows the dtype and device of its network parameters during
sampling and training.

`DiscreteTimeGrid` is a uniform model grid by default. Explicit nonuniform
`point_times` are allowed only with `allow_nonuniform=True`; this mode is
metadata for time-aware discrepancies. APIs that require a single uniform
time step, such as `dt` and Brownian increment scaling, reject nonuniform
metadata grids.

`SpecificEntropyDiagonalControl` leaves `sigma_max` optional, but an
unconstrained call raises if the Hamiltonian denominator is non-positive,
because then there is no finite volatility optimum. Provide `sigma_max` when
the desired behavior is capped volatility. The control requires a fitted
`LocalGaussianReferenceKernel`; even a constant reference used for a test should
be supplied through that fitted-kernel interface.

The supplied specific-entropy and entropy-barrier control maps expose their
matching running costs. `DiscreteMPModel` discovers these automatically. A
custom control can provide a `running_cost` callable using `RunningCostInputs`
and `RunningCostResult`, or callers can pass an explicit running-cost callable
to the model constructor.

`DiscreteMPModel.fit_nested` provides a two-timescale forward--backward solver.
Each outer sweep freezes one generated path law, constructs the complete
running-cost-plus-discrepancy adjoint targets once, and performs the configured
number of inner AdamW updates on that unchanged `P/R` regression problem. AdamW
state is reset at the next outer sweep. By default, the fully fitted parameter
displacement is backtracked against the same complete empirical objective; a
failed search restores the pre-sweep parameters. `DiscreteMPNestedTrainingConfig`
controls the two batch sizes, inner/outer iteration counts, fixed-point probe,
and outer line search. Set `outer_relaxation_mode="adjoint_blocks"` to select
separate outer interpolation rates for the fitted P head, R head, and shared
GRU parameters by deterministic coordinate search on that same objective.
`outer_block_trust_fraction` applies a fixed common shrinkage to those selected
rates and rechecks the complete objective before acceptance. This changes the
numerical solver only: the maximum-principle source, control
Hamiltonian, and neural regression loss are unchanged.

For an out-of-sample audit of that outer update, use
`experiments/scripts/run_nested_holdout_diagnostic.py`. It fits one frozen
backward problem, evaluates the same joint and blockwise relaxation surface on
the fitting bank and an independent path/target/noise bank, and then measures
common-noise path shadowing and stylized facts. The default warm-starts from
the saved specific-entropy checkpoint so that the comparison tests whether the
MP correction improves the actual reference generator.
