from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from deep_mkv_gen_path_dt.discrepancies import (  # noqa: E402
    CompositePathFunctionalDiscrepancy,
    MultiMarginalPathFunctionalDiscrepancy,
    TimeIndexedWeightedDiscrepancy,
    WeightedDiscrepancy,
)
from deep_mkv_gen_path_dt.noise import (  # noqa: E402
    derive_stream_seed,
    sample_standard_normals,
)
from path_dt_experiments.causal_r_feasibility import (  # noqa: E402
    CausalVolatilityBasisResidualNetwork,
)
from path_dt_experiments.constrained_feasibility import (  # noqa: E402
    bootstrap_volatility_gate_standard_errors,
)
from path_dt_experiments.data import simulate_heston_log_paths  # noqa: E402
from path_dt_experiments.residual_persistence import (  # noqa: E402
    FrozenBaseResidualRNetwork,
    ResidualRTrainingConfig,
    bootstrap_persistence_gate_decision,
    fit_fixed_target_time_marginal_volatility,
    fit_fixed_target_volatility_correlation,
    fit_residual_r_correction,
    persistence_gate_metrics,
    residual_block_direction_diagnostics,
)
from path_dt_experiments.runners import (  # noqa: E402
    ensure_run_dir,
    write_history_jsonl,
    write_json,
)
from run_residual_r_persistence import _build_source_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a fitted MP generator and train only a time-local causal "
            "realized-volatility basis added to its noise-adjoint R output."
        )
    )
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--windows", type=int, nargs="+", default=(4, 8, 16, 32))
    parser.add_argument("--num-calibration-paths", type=int, default=4096)
    parser.add_argument("--training-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--min-target-scale", type=float, default=1e-3)
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help=(
            "Fixed AdamW learning rate. With 1,920 initially active basis "
            "coefficients this gives a first update near the 0.01 parameter-"
            "norm scale validated by the direct feasibility audit."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--correlation-window", type=int, default=32)
    parser.add_argument("--correlation-weight", type=float, default=0.025)
    parser.add_argument("--scale-floor-fraction", type=float, default=0.25)
    parser.add_argument("--marginal-anchor-weight", type=float, default=0.025)
    parser.add_argument(
        "--marginal-formulation",
        choices=("global", "multi_marginal"),
        default="global",
        help=(
            "global reproduces the original eight-statistic path penalty; "
            "multi_marginal installs dated running law costs on path prefixes."
        ),
    )
    parser.add_argument(
        "--marginal-anchor-horizons",
        type=int,
        nargs="+",
        default=(5, 10, 20, 32),
    )
    parser.add_argument("--multi-marginal-endpoint-spacing", type=int, default=4)
    parser.add_argument(
        "--multi-marginal-windows",
        type=int,
        nargs="+",
        default=(4, 8, 16, 32),
    )
    parser.add_argument("--num-validation-paths", type=int, default=4096)
    parser.add_argument("--num-validation-generated-paths", type=int, default=4096)
    parser.add_argument("--num-heldout-generated-paths", type=int, default=8192)
    parser.add_argument("--bootstrap-replicates", type=int, default=200)
    parser.add_argument("--marginal-standard-error-budget", type=float, default=0.5)
    parser.add_argument(
        "--required-correlation-error-reduction",
        type=float,
        default=0.10,
    )
    parser.add_argument("--prefix-length", type=int, default=65)
    parser.add_argument("--prefix-window", type=int, default=32)
    parser.add_argument("--horizons", type=int, nargs="+", default=(5, 10, 20, 32))
    parser.add_argument("--lags", type=int, nargs="+", default=(1, 2, 5, 10, 20))
    parser.add_argument("--swd-projections", type=int, default=32)
    parser.add_argument("--swd-factor-limit", type=float, default=1.10)
    parser.add_argument("--direction-diagnostic-paths", type=int, default=1024)
    return parser.parse_args()


def _stream_seed(base_seed: int, offset: int) -> int:
    value = derive_stream_seed(base_seed=int(base_seed), stream_offset=int(offset))
    return int(value if value is not None else int(base_seed) + int(offset))


def _sample_metrics(
    *,
    model,
    reference_paths: torch.Tensor,
    num_paths: int,
    seed: int,
    x0: float,
    args: argparse.Namespace,
) -> tuple[dict[str, object], torch.Tensor]:
    with torch.no_grad():
        generated = model.sample(
            num_paths=int(num_paths),
            x0=torch.tensor([[float(x0)]], device=model.device, dtype=model.dtype),
            seed=int(seed),
        ).paths.detach().cpu()
    return (
        persistence_gate_metrics(
            generated,
            reference_paths,
            prefix_length=int(args.prefix_length),
            prefix_window=int(args.prefix_window),
            horizons=tuple(int(value) for value in args.horizons),
            lags=tuple(int(value) for value in args.lags),
            swd_projections=int(args.swd_projections),
        ),
        generated,
    )


def _marginal_pairs(
    *,
    num_steps: int,
    split_index: int,
    horizons: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    if len(horizons) == 0 or any(value < 1 for value in horizons):
        raise ValueError("marginal-anchor-horizons must contain positive integers")
    if len(set(horizons)) != len(horizons):
        raise ValueError("marginal-anchor-horizons must not contain duplicates")
    if any(
        split_index < horizon or split_index + horizon > int(num_steps)
        for horizon in horizons
    ):
        raise ValueError("every marginal anchor horizon must fit around the split")
    return tuple(
        dict.fromkeys(
            pair
            for horizon in horizons
            for pair in ((horizon, split_index), (horizon, split_index + horizon))
        )
    )


def _multi_marginal_pairs(
    *,
    num_steps: int,
    endpoint_spacing: int,
    windows: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    if int(endpoint_spacing) < 1:
        raise ValueError("multi-marginal-endpoint-spacing must be >= 1")
    if (
        len(windows) == 0
        or any(int(value) < 1 for value in windows)
        or tuple(sorted(set(int(value) for value in windows))) != windows
    ):
        raise ValueError(
            "multi-marginal-windows must be strictly increasing positive integers"
        )
    endpoints = list(
        range(int(endpoint_spacing), int(num_steps) + 1, int(endpoint_spacing))
    )
    if len(endpoints) == 0 or endpoints[-1] != int(num_steps):
        endpoints.append(int(num_steps))
    pairs = tuple(
        (int(window), int(endpoint))
        for endpoint in endpoints
        for window in windows
        if int(window) <= int(endpoint)
    )
    if len(pairs) == 0:
        raise ValueError("no multi-marginal window fits on the grid")
    return tuple(endpoints), pairs


def _coefficient_localization(
    basis: CausalVolatilityBasisResidualNetwork,
) -> dict[str, object]:
    coefficients = basis.coefficients.detach().cpu().double()
    time_norms = torch.linalg.vector_norm(coefficients, dim=(1, 2))
    feature_norms = torch.linalg.vector_norm(coefficients, dim=(0, 2))
    top_time_count = min(12, int(time_norms.numel()))
    top_feature_count = min(12, int(feature_norms.numel()))
    top_times = torch.topk(time_norms, top_time_count)
    top_features = torch.topk(feature_norms, top_feature_count)
    return {
        "coefficient_norm": float(torch.linalg.vector_norm(coefficients).item()),
        "nonzero_coefficient_fraction": float((coefficients != 0.0).double().mean().item()),
        "top_time_indices": [
            {"step": int(index.item()), "coefficient_norm": float(value.item())}
            for value, index in zip(top_times.values, top_times.indices)
        ],
        "top_features": [
            {
                "feature": basis.feature_names[int(index.item())],
                "coefficient_norm": float(value.item()),
            }
            for value, index in zip(top_features.values, top_features.indices)
        ],
    }


def main() -> None:
    args = parse_args()
    source = Path(args.source_run_dir)
    run_dir = ensure_run_dir(args.run_dir)
    model, source_training, stored_config, heston, target_paths, heldout_paths = (
        _build_source_model(source=source, device=torch.device(args.device))
    )
    if int(args.bootstrap_replicates) < 2:
        raise ValueError("bootstrap-replicates must be >= 2")
    if int(args.num_calibration_paths) < 2:
        raise ValueError("num-calibration-paths must be >= 2")

    fixed_target_paths = target_paths.to(device=model.device, dtype=model.dtype)
    split_index = int(model.grid.num_steps) // 2
    correlation_window = int(args.correlation_window)
    fixed_correlation = fit_fixed_target_volatility_correlation(
        fixed_target_paths,
        grid=model.grid,
        window=correlation_window,
        split_index=split_index,
        scale_floor_fraction=float(args.scale_floor_fraction),
    )
    marginal_formulation = str(args.marginal_formulation)
    marginal_horizons = tuple(int(value) for value in args.marginal_anchor_horizons)
    multi_marginal_windows = tuple(int(value) for value in args.multi_marginal_windows)
    if marginal_formulation == "global":
        marginal_endpoints: tuple[int, ...] = ()
        marginal_pairs = _marginal_pairs(
            num_steps=int(model.grid.num_steps),
            split_index=split_index,
            horizons=marginal_horizons,
        )
    else:
        marginal_endpoints, marginal_pairs = _multi_marginal_pairs(
            num_steps=int(model.grid.num_steps),
            endpoint_spacing=int(args.multi_marginal_endpoint_spacing),
            windows=multi_marginal_windows,
        )
    fixed_marginals = fit_fixed_target_time_marginal_volatility(
        fixed_target_paths,
        grid=model.grid,
        window_endpoint_pairs=marginal_pairs,
    )
    correlation_block = WeightedDiscrepancy(
        name="volatility_law_fixed_horizon_correlation",
        weight=float(args.correlation_weight),
        discrepancy=fixed_correlation,
    )
    if marginal_formulation == "global":
        marginal_block = WeightedDiscrepancy(
            name="volatility_law_fixed_time_marginals",
            weight=float(args.marginal_anchor_weight),
            discrepancy=fixed_marginals,
        )
        residual_blocks = (marginal_block, correlation_block)
        residual_discrepancy = CompositePathFunctionalDiscrepancy(
            blocks=residual_blocks
        )
    else:
        date_weight = float(args.marginal_anchor_weight) / float(
            len(marginal_endpoints)
        )
        pairs_by_endpoint = {
            endpoint: tuple(pair for pair in marginal_pairs if pair[1] == endpoint)
            for endpoint in marginal_endpoints
        }
        running_blocks = tuple(
            TimeIndexedWeightedDiscrepancy(
                name=f"volatility_marginal_t{endpoint}",
                endpoint_index=int(endpoint),
                weight=date_weight,
                discrepancy=fixed_marginals.select_pairs(pairs_by_endpoint[endpoint]),
            )
            for endpoint in marginal_endpoints
            if int(endpoint) < int(model.grid.num_steps)
        )
        terminal_marginal_block = WeightedDiscrepancy(
            name=f"volatility_marginal_t{int(model.grid.num_steps)}",
            weight=date_weight,
            discrepancy=fixed_marginals.select_pairs(
                pairs_by_endpoint[int(model.grid.num_steps)]
            ),
        )
        marginal_only_discrepancy = MultiMarginalPathFunctionalDiscrepancy(
            running_blocks=running_blocks,
            terminal_blocks=(terminal_marginal_block,),
        )
        correlation_only_discrepancy = MultiMarginalPathFunctionalDiscrepancy(
            terminal_blocks=(correlation_block,),
        )
        residual_discrepancy = MultiMarginalPathFunctionalDiscrepancy(
            running_blocks=running_blocks,
            terminal_blocks=(terminal_marginal_block, correlation_block),
        )
        # The direction diagnostic accepts named ordinary discrepancies.  An
        # outer unit weight keeps the already dated weights unchanged.
        residual_blocks = (
            WeightedDiscrepancy(
                name="volatility_law_multi_marginal_running",
                weight=1.0,
                discrepancy=marginal_only_discrepancy,
            ),
            WeightedDiscrepancy(
                name="volatility_law_fixed_horizon_correlation",
                weight=1.0,
                discrepancy=correlation_only_discrepancy,
            ),
        )

    calibration_seed = _stream_seed(int(args.seed), 130_000)
    with torch.no_grad():
        calibration_paths = model.sample(
            num_paths=int(args.num_calibration_paths),
            x0=torch.tensor(
                [[float(heston.x0)]],
                device=model.device,
                dtype=model.dtype,
            ),
            seed=calibration_seed,
        ).paths
    basis = CausalVolatilityBasisResidualNetwork(
        state_dim=int(model.architecture.state_dim),
        noise_adjoint_dim=int(model._default_noise_adjoint_dim(model.architecture)),
        num_steps=int(model.grid.num_steps),
        dt=float(model.grid.dt),
        windows=tuple(int(value) for value in args.windows),
    ).to(device=model.device, dtype=model.dtype)
    basis.fit_calibration(calibration_paths)
    del calibration_paths

    base_scale = model._noise_adjoint_target_scale
    if base_scale is None:
        raise ValueError("source checkpoint must contain a noise-adjoint target scale")
    combined_network = FrozenBaseResidualRNetwork(
        base_network=model.network,
        residual_network=basis,
        base_noise_scale=base_scale,
    ).to(device=model.device, dtype=model.dtype)
    model.network = combined_network
    model._validate_network_contract(
        network=combined_network,
        architecture=model.architecture,
    )
    base_state_before = {
        name: value.detach().cpu().clone()
        for name, value in combined_network.base_network.state_dict().items()
    }
    residual_source_training = replace(source_training, observed_only=False)

    validation_reference_seed = _stream_seed(int(args.seed), 131_000)
    validation_reference = simulate_heston_log_paths(
        num_paths=int(args.num_validation_paths),
        config=heston,
        seed=validation_reference_seed,
        device="cpu",
        dtype=model.dtype,
    )
    validation_generation_seed = _stream_seed(int(args.seed), 132_000)
    validation_generated: dict[int, torch.Tensor] = {}

    def evaluate_validation(step: int) -> dict[str, object]:
        metrics, generated = _sample_metrics(
            model=model,
            reference_paths=validation_reference,
            num_paths=int(args.num_validation_generated_paths),
            seed=validation_generation_seed,
            x0=float(heston.x0),
            args=args,
        )
        validation_generated[int(step)] = generated
        return metrics

    training_config = ResidualRTrainingConfig(
        num_steps=int(args.training_steps),
        batch_size=int(args.batch_size),
        calibration_batches=int(args.calibration_batches),
        min_scale=float(args.min_target_scale),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        eval_every=int(args.eval_every),
        seed=int(args.seed),
    )
    fit = fit_residual_r_correction(
        model=model,
        network=combined_network,
        target_paths=target_paths,
        discrepancy=residual_discrepancy,
        source_training=residual_source_training,
        config=training_config,
        evaluation_callback=evaluate_validation,
    )
    base_unchanged = all(
        torch.equal(value.detach().cpu(), base_state_before[name])
        for name, value in combined_network.base_network.state_dict().items()
    )
    if not base_unchanged:
        raise RuntimeError("the frozen base network changed during basis-only training")
    if not fit.evaluations or int(fit.evaluations[0]["step"]) != 0:
        raise RuntimeError("basis training did not record its validation baseline")

    baseline_validation = fit.evaluations[0]["metrics"]
    if not isinstance(baseline_validation, dict):
        raise RuntimeError("validation baseline metrics are malformed")
    validation_standard_errors = bootstrap_volatility_gate_standard_errors(
        validation_generated[0],
        validation_reference,
        dt=float(model.grid.dt),
        prefix_length=int(args.prefix_length),
        prefix_window=int(args.prefix_window),
        horizon=correlation_window,
        num_replicates=int(args.bootstrap_replicates),
        seed=_stream_seed(int(args.seed), 133_000),
    )
    validation_records: list[dict[str, object]] = []
    eligible: list[tuple[float, int]] = []
    all_candidates: list[tuple[float, int]] = []
    for record in fit.evaluations:
        step = int(record["step"])
        metrics = record["metrics"]
        if not isinstance(metrics, dict):
            raise RuntimeError("validation checkpoint metrics are malformed")
        decision = bootstrap_persistence_gate_decision(
            metrics,
            baseline_validation,
            standard_errors=validation_standard_errors,
            horizon=correlation_window,
            required_correlation_error_reduction=float(
                args.required_correlation_error_reduction
            ),
            marginal_standard_error_budget=float(args.marginal_standard_error_budget),
            swd_factor_limit=float(args.swd_factor_limit),
        )
        horizon_metrics = metrics["horizons"]
        assert isinstance(horizon_metrics, dict)
        selected_horizon = horizon_metrics[str(correlation_window)]
        assert isinstance(selected_horizon, dict)
        correlation_error = float(
            selected_horizon["prefix_future_correlation_abs_error"]
        )
        all_candidates.append((correlation_error, step))
        if bool(decision["constraints_pass"]):
            eligible.append((correlation_error, step))
        validation_records.append(
            {"step": step, "metrics": metrics, "decision": decision}
        )
    selected_step = min(eligible)[1] if eligible else 0
    best_correlation_step = min(all_candidates)[1]
    selected_validation = next(
        record for record in validation_records if int(record["step"]) == selected_step
    )
    best_correlation_validation = next(
        record
        for record in validation_records
        if int(record["step"]) == best_correlation_step
    )

    diagnostic_count = int(args.direction_diagnostic_paths)
    if diagnostic_count < 2:
        raise ValueError("direction-diagnostic-paths must be >= 2")
    diagnostic_noise = sample_standard_normals(
        grid=model.grid,
        batch_size=diagnostic_count,
        noise_dim=int(model.architecture.noise_dim),
        device=model.device,
        dtype=model.dtype,
        seed=_stream_seed(int(args.seed), 134_000),
    )
    diagnostic_x0 = torch.full(
        (diagnostic_count, int(model.architecture.state_dim)),
        float(heston.x0),
        device=model.device,
        dtype=model.dtype,
    )

    def direction_diagnostics(step: int) -> dict[str, object]:
        combined_network.residual_network.load_state_dict(fit.checkpoints[int(step)])
        with torch.no_grad():
            rollout = model.rollout(x0=diagnostic_x0, noise=diagnostic_noise)
        return residual_block_direction_diagnostics(
            model=model,
            generated_paths=rollout.paths,
            noise=diagnostic_noise,
            target_paths=target_paths,
            source_training=residual_source_training,
            blocks=residual_blocks,
        )

    direction_payload = {
        "baseline": direction_diagnostics(0),
        "selected": direction_diagnostics(selected_step),
        "best_correlation": direction_diagnostics(best_correlation_step),
        "terminal": direction_diagnostics(max(fit.checkpoints)),
    }
    write_json(run_dir / "block_direction_diagnostics.json", direction_payload)

    heldout_generation_seed = _stream_seed(int(args.seed), 135_000)
    combined_network.residual_network.load_state_dict(fit.checkpoints[0])
    heldout_baseline, heldout_baseline_generated = _sample_metrics(
        model=model,
        reference_paths=heldout_paths,
        num_paths=int(args.num_heldout_generated_paths),
        seed=heldout_generation_seed,
        x0=float(heston.x0),
        args=args,
    )
    heldout_standard_errors = bootstrap_volatility_gate_standard_errors(
        heldout_baseline_generated,
        heldout_paths,
        dt=float(model.grid.dt),
        prefix_length=int(args.prefix_length),
        prefix_window=int(args.prefix_window),
        horizon=correlation_window,
        num_replicates=int(args.bootstrap_replicates),
        seed=_stream_seed(int(args.seed), 136_000),
    )
    combined_network.residual_network.load_state_dict(fit.checkpoints[selected_step])
    heldout_selected, heldout_selected_generated = _sample_metrics(
        model=model,
        reference_paths=heldout_paths,
        num_paths=int(args.num_heldout_generated_paths),
        seed=heldout_generation_seed,
        x0=float(heston.x0),
        args=args,
    )
    heldout_decision = bootstrap_persistence_gate_decision(
        heldout_selected,
        heldout_baseline,
        standard_errors=heldout_standard_errors,
        horizon=correlation_window,
        required_correlation_error_reduction=float(
            args.required_correlation_error_reduction
        ),
        marginal_standard_error_budget=float(args.marginal_standard_error_budget),
        swd_factor_limit=float(args.swd_factor_limit),
    )
    if best_correlation_step == selected_step:
        heldout_best_correlation = heldout_selected
        heldout_best_correlation_generated = heldout_selected_generated
        heldout_best_correlation_decision = heldout_decision
    else:
        combined_network.residual_network.load_state_dict(
            fit.checkpoints[best_correlation_step]
        )
        (
            heldout_best_correlation,
            heldout_best_correlation_generated,
        ) = _sample_metrics(
            model=model,
            reference_paths=heldout_paths,
            num_paths=int(args.num_heldout_generated_paths),
            seed=heldout_generation_seed,
            x0=float(heston.x0),
            args=args,
        )
        heldout_best_correlation_decision = bootstrap_persistence_gate_decision(
            heldout_best_correlation,
            heldout_baseline,
            standard_errors=heldout_standard_errors,
            horizon=correlation_window,
            required_correlation_error_reduction=float(
                args.required_correlation_error_reduction
            ),
            marginal_standard_error_budget=float(
                args.marginal_standard_error_budget
            ),
            swd_factor_limit=float(args.swd_factor_limit),
        )
    validation_decision = selected_validation["decision"]
    assert isinstance(validation_decision, dict)
    overall_pass = bool(validation_decision["pass"] and heldout_decision["pass"])

    combined_network.residual_network.load_state_dict(fit.checkpoints[selected_step])
    selected_localization = _coefficient_localization(basis)
    combined_network.residual_network.load_state_dict(
        fit.checkpoints[best_correlation_step]
    )
    best_correlation_localization = _coefficient_localization(basis)
    checkpoint = {
        "format_version": 2,
        "source_run_dir": str(source),
        "selected_step": int(selected_step),
        "marginal_formulation": marginal_formulation,
        "multi_marginal_endpoints": tuple(marginal_endpoints),
        "multi_marginal_windows": tuple(multi_marginal_windows),
        "basis_architecture": {
            "state_dim": int(basis.state_dim),
            "noise_adjoint_dim": int(basis.noise_adjoint_dim),
            "num_steps": int(basis.num_steps),
            "dt": float(basis.dt),
            "windows": tuple(int(value) for value in basis.windows),
            "feature_names": tuple(basis.feature_names),
        },
        "basis_state_dict": fit.checkpoints[selected_step],
        "residual_noise_scale": fit.residual_noise_scale,
        "fixed_target_correlation": fixed_correlation.target_correlation,
        "fixed_prefix_scale_floor": fixed_correlation.prefix_scale_floor,
        "fixed_future_scale_floor": fixed_correlation.future_scale_floor,
        "fixed_marginal_window_endpoint_pairs": tuple(marginal_pairs),
        "fixed_marginal_target_mean": fixed_marginals.target_mean,
        "fixed_marginal_target_std": fixed_marginals.target_std,
        "fixed_marginal_target_iqr": fixed_marginals.target_iqr,
        "fixed_marginal_target_sorted_standardized": (
            fixed_marginals.target_sorted_standardized
        ),
    }
    torch.save(checkpoint, run_dir / "causal_r_basis_checkpoint.pt")
    torch.save(
        {
            "step": int(max(fit.checkpoints)),
            "basis_state_dict": fit.checkpoints[max(fit.checkpoints)],
            "residual_noise_scale": fit.residual_noise_scale,
        },
        run_dir / "causal_r_basis_terminal_checkpoint.pt",
    )
    torch.save(
        {
            "step": int(best_correlation_step),
            "basis_state_dict": fit.checkpoints[best_correlation_step],
            "residual_noise_scale": fit.residual_noise_scale,
            "selection_status": (
                "selected"
                if best_correlation_step == selected_step
                else "rejected_by_validation_gate"
            ),
        },
        run_dir / "causal_r_basis_best_correlation_checkpoint.pt",
    )
    torch.save(heldout_baseline_generated, run_dir / "heldout_baseline_generated_paths.pt")
    torch.save(heldout_selected_generated, run_dir / "heldout_selected_generated_paths.pt")
    if best_correlation_step != selected_step:
        torch.save(
            heldout_best_correlation_generated,
            run_dir / "heldout_best_correlation_generated_paths.pt",
        )
    write_history_jsonl(run_dir / "training_history.jsonl", fit.history)
    write_json(run_dir / "validation_history.json", {"records": validation_records})
    report = {
        "source_run_dir": str(source),
        "run_dir": str(run_dir),
        "experiment": {
            "question": (
                "Can the unchanged MP adjoint-MSE signal train a direct causal "
                "rolling-volatility basis in R while the fitted GRU base is frozen "
                "and volatility marginals enter as dated MP running costs?"
            ),
            "neural_loss_changed": False,
            "maximum_principle_formulation": (
                "dated_multi_marginal_running_costs"
                if marginal_formulation == "multi_marginal"
                else "global_path_functional"
            ),
            "control_map_changed": False,
            "base_network_frozen": True,
            "base_network_unchanged": bool(base_unchanged),
            "trainable_component": "causal_r_basis_coefficients_only",
            "trainable_parameter_count": int(basis.coefficients.numel()),
        },
        "config": {
            "stored_source_config": stored_config,
            "residual_training": asdict(training_config),
            "windows": list(basis.windows),
            "feature_names": list(basis.feature_names),
            "num_calibration_paths": int(args.num_calibration_paths),
            "calibration_seed": int(calibration_seed),
            "correlation_window": correlation_window,
            "correlation_weight": float(args.correlation_weight),
            "marginal_anchor_weight": float(args.marginal_anchor_weight),
            "marginal_formulation": marginal_formulation,
            "marginal_anchor_horizons": list(marginal_horizons),
            "marginal_window_endpoint_pairs": [list(pair) for pair in marginal_pairs],
            "multi_marginal_endpoints": list(marginal_endpoints),
            "multi_marginal_endpoint_spacing": int(
                args.multi_marginal_endpoint_spacing
            ),
            "multi_marginal_windows": list(multi_marginal_windows),
            "multi_marginal_date_weight": (
                None
                if marginal_formulation == "global"
                else float(args.marginal_anchor_weight)
                / float(len(marginal_endpoints))
            ),
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "marginal_standard_error_budget": float(
                args.marginal_standard_error_budget
            ),
            "required_correlation_error_reduction": float(
                args.required_correlation_error_reduction
            ),
            "validation_reference_seed": int(validation_reference_seed),
            "validation_generation_seed": int(validation_generation_seed),
            "heldout_generation_seed": int(heldout_generation_seed),
        },
        "selection": {
            "selected_step": int(selected_step),
            "best_correlation_step": int(best_correlation_step),
            "validation_standard_errors": validation_standard_errors,
            "heldout_standard_errors": heldout_standard_errors,
            "overall_pass": bool(overall_pass),
        },
        "validation_baseline": baseline_validation,
        "validation_selected": selected_validation,
        "validation_best_correlation": best_correlation_validation,
        "heldout_baseline": heldout_baseline,
        "heldout_selected": heldout_selected,
        "heldout_decision": heldout_decision,
        "heldout_best_correlation": heldout_best_correlation,
        "heldout_best_correlation_decision": heldout_best_correlation_decision,
        "basis_coefficients": {
            "selected": selected_localization,
            "best_correlation": best_correlation_localization,
        },
        "residual_scale": {
            "minimum": float(fit.residual_noise_scale.min().item()),
            "median": float(fit.residual_noise_scale.median().item()),
            "maximum": float(fit.residual_noise_scale.max().item()),
        },
        "block_direction_diagnostics": "block_direction_diagnostics.json",
    }
    write_json(run_dir / "metrics.json", report)

    baseline_horizon = heldout_baseline["horizons"]
    selected_horizon = heldout_selected["horizons"]
    assert isinstance(baseline_horizon, dict) and isinstance(selected_horizon, dict)
    baseline_h = baseline_horizon[str(correlation_window)]
    selected_h = selected_horizon[str(correlation_window)]
    assert isinstance(baseline_h, dict) and isinstance(selected_h, dict)
    print(f"wrote causal-R basis training run to {run_dir}")
    print(
        f"selected_step={selected_step} best_correlation_step={best_correlation_step} "
        f"overall_pass={overall_pass} heldout_corr_error="
        f"{float(baseline_h['prefix_future_correlation_abs_error']):.6f}->"
        f"{float(selected_h['prefix_future_correlation_abs_error']):.6f} "
        f"heldout_mean_error={float(baseline_h['future_rv_mean_abs_error']):.6f}->"
        f"{float(selected_h['future_rv_mean_abs_error']):.6f} "
        f"heldout_std_error={float(baseline_h['future_rv_std_abs_error']):.6f}->"
        f"{float(selected_h['future_rv_std_abs_error']):.6f} "
        f"heldout_iqr_error={float(baseline_h['future_rv_iqr_abs_error']):.6f}->"
        f"{float(selected_h['future_rv_iqr_abs_error']):.6f}"
    )


if __name__ == "__main__":
    main()
