from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from deep_mkv_gen_path_dt import (  # noqa: E402
    DiscreteMPArchitectureConfig,
    DiscreteMPModel,
    DiscreteMPTrainingConfig,
    DiscreteTimeGrid,
)
from deep_mkv_gen_path_dt.controls import EntropyBarrierDiagonalControl  # noqa: E402
from deep_mkv_gen_path_dt.discrepancies import (  # noqa: E402
    CompositePathFunctionalDiscrepancy,
    WeightedDiscrepancy,
)
from deep_mkv_gen_path_dt.noise import derive_stream_seed, sample_standard_normals  # noqa: E402
from path_dt_experiments.data import HestonConfig, simulate_heston_log_paths  # noqa: E402
from path_dt_experiments.discrepancies import build_heston_discrepancy  # noqa: E402
from path_dt_experiments.dynamics import DriftVolatilityEulerStep  # noqa: E402
from path_dt_experiments.residual_persistence import (  # noqa: E402
    CenteredNoiseAdjointResidualNetwork,
    FrozenBaseResidualRNetwork,
    NoiseAdjointResidualNetwork,
    ResidualRTrainingConfig,
    fit_fixed_target_volatility_correlation,
    fit_fixed_target_time_marginal_volatility,
    fit_residual_r_correction,
    persistence_gate_decision,
    persistence_gate_metrics,
    residual_block_direction_diagnostics,
)
from path_dt_experiments.runners import (  # noqa: E402
    ensure_run_dir,
    write_history_jsonl,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a validation-selected additive noise-adjoint correction on a frozen Heston model."
        )
    )
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--residual-hidden-dim", type=int, default=96)
    parser.add_argument("--residual-num-layers", type=int, default=1)
    parser.add_argument(
        "--residual-parameterization",
        choices=("zero_final", "centered_full_tangent"),
        default="zero_final",
        help=(
            "zero_final reproduces the original residual; centered_full_tangent "
            "subtracts a frozen initialized copy and activates the full network "
            "Jacobian while retaining exactly zero initial output."
        ),
    )
    parser.add_argument("--training-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--min-scale", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--correlation-window", type=int, default=32)
    parser.add_argument("--correlation-weight", type=float, default=0.025)
    parser.add_argument("--scale-floor-fraction", type=float, default=0.25)
    parser.add_argument("--marginal-anchor-weight", type=float, default=0.0)
    parser.add_argument(
        "--marginal-anchor-horizons",
        type=int,
        nargs="+",
        default=(5, 10, 20, 32),
    )
    parser.add_argument("--direction-diagnostic-paths", type=int, default=1024)
    parser.add_argument("--num-validation-paths", type=int, default=4096)
    parser.add_argument("--num-validation-generated-paths", type=int, default=4096)
    parser.add_argument("--num-heldout-generated-paths", type=int, default=8192)
    parser.add_argument("--prefix-length", type=int, default=65)
    parser.add_argument("--prefix-window", type=int, default=32)
    parser.add_argument("--horizons", type=int, nargs="+", default=(5, 10, 20, 32))
    parser.add_argument("--lags", type=int, nargs="+", default=(1, 2, 5, 10, 20))
    parser.add_argument("--swd-projections", type=int, default=32)
    parser.add_argument("--required-correlation-error-reduction", type=float, default=0.25)
    parser.add_argument("--scale-error-tolerance", type=float, default=0.0)
    parser.add_argument("--swd-factor-limit", type=float, default=1.10)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _observed_indices(*, num_steps: int, spacing: int) -> tuple[int, ...]:
    if spacing < 1 or num_steps % spacing != 0:
        raise ValueError("stored observed_every must divide num_steps")
    return tuple(range(0, int(num_steps) + 1, int(spacing)))


def _build_source_model(
    *,
    source: Path,
    device: torch.device,
) -> tuple[
    DiscreteMPModel,
    DiscreteMPTrainingConfig,
    dict[str, object],
    HestonConfig,
    torch.Tensor,
    torch.Tensor,
]:
    metrics = _load_json(source / "metrics.json")
    stored_config = metrics.get("config")
    if not isinstance(stored_config, dict):
        raise ValueError("source metrics.json is missing config")
    checkpoint = torch.load(source / "model_checkpoint.pt", map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("model_checkpoint.pt must contain a dictionary")
    architecture_payload = checkpoint.get("architecture")
    training_payload = checkpoint.get("training")
    if not isinstance(architecture_payload, dict) or not isinstance(training_payload, dict):
        raise ValueError("checkpoint is missing architecture or training")
    architecture = DiscreteMPArchitectureConfig(**architecture_payload)
    training = DiscreteMPTrainingConfig(**training_payload)
    heston_payload = stored_config.get("heston")
    if not isinstance(heston_payload, dict):
        raise ValueError("source config is missing heston")
    heston = HestonConfig(**heston_payload)
    observed_every = int(stored_config["observed_every"])
    grid = DiscreteTimeGrid(
        T=float(heston.T),
        num_steps=int(heston.num_steps),
        observed_indices=_observed_indices(
            num_steps=int(heston.num_steps),
            spacing=observed_every,
        ),
    )
    control_payload = stored_config.get("control_map")
    if not isinstance(control_payload, dict) or control_payload.get("name") != "entropy_barrier":
        raise ValueError("residual-R runner requires an entropy_barrier source checkpoint")
    control = EntropyBarrierDiagonalControl(
        dt=grid.dt,
        kappa=float(control_payload["kappa"]),
        tau=float(control_payload["tau"]),
        sigma_min=float(stored_config["sigma_min"]),
        sigma_max=float(stored_config["sigma_max"]),
    )
    discrepancy_steps = int(grid.observed_count) - 1 if bool(training.observed_only) else int(grid.num_steps)
    raw_correlation_windows = tuple(
        int(value)
        for value in stored_config.get("conditional_volatility_correlation_windows", (16, 32))
    )
    correlation_windows = (
        tuple(value // observed_every for value in raw_correlation_windows)
        if bool(training.observed_only)
        else raw_correlation_windows
    )
    stored_endpoints = stored_config.get("time_marginal_volatility_endpoints", ())
    if not isinstance(stored_endpoints, (list, tuple)):
        raise ValueError("stored time-marginal endpoints must be a sequence")
    discrepancy = build_heston_discrepancy(
        num_steps=discrepancy_steps,
        include_acf=bool(stored_config["include_acf"]),
        preset=str(stored_config["preset"]),
        lambda_scale=float(stored_config["lambda_scale"]),
        kappa_scale=float(stored_config["kappa_scale"]),
        include_conditional_volatility=bool(
            stored_config.get("conditional_volatility_discrepancy", False)
        ),
        include_conditional_volatility_correlation=bool(
            stored_config.get("conditional_volatility_correlation_discrepancy", False)
        ),
        conditional_volatility_windows=tuple(
            int(value)
            for value in stored_config.get("conditional_volatility_discrepancy_windows", ())
        ),
        conditional_volatility_correlation_windows=correlation_windows,
        conditional_volatility_weight=float(
            stored_config.get("conditional_volatility_weight", 1.0)
        ),
        conditional_volatility_correlation_weight=float(
            stored_config.get("conditional_volatility_correlation_weight", 0.025)
        ),
        conditional_volatility_bandwidths=tuple(
            float(value)
            for value in stored_config.get("conditional_volatility_bandwidths", (0.5, 1.0, 2.0))
        ),
        conditional_volatility_correlation_full_matrix=bool(
            stored_config.get("conditional_volatility_correlation_full_matrix", False)
        ),
        conditional_volatility_correlation_scale_floor_fraction=float(
            stored_config.get("conditional_volatility_correlation_scale_floor_fraction", 0.25)
        ),
        time_marginal_volatility_window=(
            None
            if stored_config.get("time_marginal_volatility_window") is None
            else int(stored_config["time_marginal_volatility_window"])
        ),
        time_marginal_volatility_endpoints=tuple(int(value) for value in stored_endpoints),
        time_marginal_volatility_weight=float(
            stored_config.get("time_marginal_volatility_weight", 1.0)
        ),
    )
    model = DiscreteMPModel(
        architecture=architecture,
        grid=grid,
        dynamics_step=DriftVolatilityEulerStep(dt=grid.dt),
        control_map=control,
        discrepancy=discrepancy,
    )
    dtype = torch.float64 if str(stored_config["dtype"]) == "float64" else torch.float32
    model.network.to(device=device, dtype=dtype)
    model.load_checkpoint_state(checkpoint)
    target_paths = torch.load(source / "target_paths.pt", map_location="cpu", weights_only=True)
    heldout_paths = torch.load(source / "heldout_paths.pt", map_location="cpu", weights_only=True)
    if not isinstance(target_paths, torch.Tensor) or not isinstance(heldout_paths, torch.Tensor):
        raise ValueError("source target and held-out artifacts must be tensors")
    return model, training, stored_config, heston, target_paths, heldout_paths


def _sample_metrics(
    *,
    model: DiscreteMPModel,
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
    metrics = persistence_gate_metrics(
        generated,
        reference_paths,
        prefix_length=int(args.prefix_length),
        prefix_window=int(args.prefix_window),
        horizons=tuple(int(value) for value in args.horizons),
        lags=tuple(int(value) for value in args.lags),
        swd_projections=int(args.swd_projections),
    )
    return metrics, generated


def _decision(
    candidate: dict[str, object],
    baseline: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    return persistence_gate_decision(
        candidate,
        baseline,
        horizon=int(args.correlation_window),
        required_correlation_error_reduction=float(
            args.required_correlation_error_reduction
        ),
        scale_error_tolerance=float(args.scale_error_tolerance),
        swd_factor_limit=float(args.swd_factor_limit),
    )


def main() -> None:
    args = parse_args()
    source = Path(args.source_run_dir)
    run_dir = ensure_run_dir(args.run_dir)
    device = torch.device(args.device)
    model, source_training, stored_config, heston, target_paths, heldout_paths = _build_source_model(
        source=source,
        device=device,
    )
    # The gate uses fine-grid realized volatility.  The residual target must use
    # that exact statistic rather than the correlation of aggregated increments
    # on the source model's observed-only discrepancy grid.
    fixed_target_paths = target_paths.to(device=model.device, dtype=model.dtype)
    discrepancy_grid = model.grid
    discrepancy_window = int(args.correlation_window)
    fixed_correlation = fit_fixed_target_volatility_correlation(
        fixed_target_paths,
        grid=discrepancy_grid,
        window=discrepancy_window,
        split_index=int(discrepancy_grid.num_steps) // 2,
        scale_floor_fraction=float(args.scale_floor_fraction),
    )
    split_index = int(discrepancy_grid.num_steps) // 2
    marginal_horizons = tuple(int(value) for value in args.marginal_anchor_horizons)
    if len(marginal_horizons) == 0 or any(value < 1 for value in marginal_horizons):
        raise ValueError("marginal-anchor-horizons must contain positive integers")
    if len(set(marginal_horizons)) != len(marginal_horizons):
        raise ValueError("marginal-anchor-horizons must not contain duplicates")
    if any(
        split_index < horizon or split_index + horizon > int(discrepancy_grid.num_steps)
        for horizon in marginal_horizons
    ):
        raise ValueError("every marginal anchor horizon must fit before and after the split")
    marginal_pairs = tuple(
        dict.fromkeys(
            pair
            for horizon in marginal_horizons
            for pair in ((horizon, split_index), (horizon, split_index + horizon))
        )
    )
    fixed_marginals = fit_fixed_target_time_marginal_volatility(
        fixed_target_paths,
        grid=discrepancy_grid,
        window_endpoint_pairs=marginal_pairs,
    )
    residual_blocks = (
        WeightedDiscrepancy(
            name="volatility_law_fixed_time_marginals",
            weight=float(args.marginal_anchor_weight),
            discrepancy=fixed_marginals,
        ),
        WeightedDiscrepancy(
            name="volatility_law_fixed_horizon_correlation",
            weight=float(args.correlation_weight),
            discrepancy=fixed_correlation,
        ),
    )
    residual_discrepancy = CompositePathFunctionalDiscrepancy(
        blocks=residual_blocks
    )
    base_scale = model._noise_adjoint_target_scale
    if base_scale is None:
        raise ValueError("source checkpoint must contain a timewise noise-adjoint target scale")
    residual_seed = derive_stream_seed(base_seed=int(args.seed), stream_offset=101_000)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(residual_seed if residual_seed is not None else args.seed))
        raw_residual_network = NoiseAdjointResidualNetwork(
            state_dim=int(model.architecture.state_dim),
            noise_adjoint_dim=int(model._default_noise_adjoint_dim(model.architecture)),
            hidden_dim=int(args.residual_hidden_dim),
            num_layers=int(args.residual_num_layers),
            zero_final=str(args.residual_parameterization) == "zero_final",
        )
    residual_network = (
        CenteredNoiseAdjointResidualNetwork(raw_residual_network)
        if str(args.residual_parameterization) == "centered_full_tangent"
        else raw_residual_network
    )
    residual_network.to(device=model.device, dtype=model.dtype)
    combined_network = FrozenBaseResidualRNetwork(
        base_network=model.network,
        residual_network=residual_network,
        base_noise_scale=base_scale,
    )
    combined_network.to(device=model.device, dtype=model.dtype)
    model.network = combined_network
    model._validate_network_contract(network=combined_network, architecture=model.architecture)
    residual_source_training = replace(source_training, observed_only=False)

    diagnostic_count = int(args.direction_diagnostic_paths)
    if diagnostic_count < 2:
        raise ValueError("direction-diagnostic-paths must be >= 2")
    diagnostic_x0 = target_paths[:diagnostic_count, 0, :].to(
        device=model.device,
        dtype=model.dtype,
    )
    if int(diagnostic_x0.shape[0]) < diagnostic_count:
        repeats = (diagnostic_count + int(diagnostic_x0.shape[0]) - 1) // int(
            diagnostic_x0.shape[0]
        )
        diagnostic_x0 = diagnostic_x0.repeat(repeats, 1)[:diagnostic_count]
    direction_seed = derive_stream_seed(base_seed=int(args.seed), stream_offset=105_000)
    diagnostic_noise = sample_standard_normals(
        grid=model.grid,
        batch_size=diagnostic_count,
        noise_dim=int(model.architecture.noise_dim),
        device=model.device,
        dtype=model.dtype,
        seed=int(direction_seed if direction_seed is not None else args.seed + 4),
    )

    def evaluate_block_directions() -> dict[str, object]:
        with torch.no_grad():
            diagnostic_rollout = model.rollout(
                x0=diagnostic_x0,
                noise=diagnostic_noise,
            )
        return residual_block_direction_diagnostics(
            model=model,
            generated_paths=diagnostic_rollout.paths,
            noise=diagnostic_noise,
            target_paths=target_paths,
            source_training=residual_source_training,
            blocks=residual_blocks,
        )

    validation_seed = derive_stream_seed(base_seed=int(args.seed), stream_offset=102_000)
    validation_reference = simulate_heston_log_paths(
        num_paths=int(args.num_validation_paths),
        config=heston,
        seed=int(validation_seed if validation_seed is not None else args.seed + 1),
        device="cpu",
        dtype=model.dtype,
    )
    validation_generation_seed = derive_stream_seed(
        base_seed=int(args.seed),
        stream_offset=103_000,
    )
    validation_generation_seed = int(
        validation_generation_seed if validation_generation_seed is not None else args.seed + 2
    )

    def evaluate_validation(step: int) -> dict[str, object]:
        del step
        metrics, _ = _sample_metrics(
            model=model,
            reference_paths=validation_reference,
            num_paths=int(args.num_validation_generated_paths),
            seed=validation_generation_seed,
            x0=float(heston.x0),
            args=args,
        )
        return metrics

    training_config = ResidualRTrainingConfig(
        num_steps=int(args.training_steps),
        batch_size=int(args.batch_size),
        calibration_batches=int(args.calibration_batches),
        min_scale=float(args.min_scale),
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
    combined_network.residual_network.load_state_dict(fit.checkpoints[0])
    baseline_direction_diagnostics = evaluate_block_directions()
    if not fit.evaluations or int(fit.evaluations[0]["step"]) != 0:
        raise RuntimeError("residual training did not record the validation baseline")
    baseline_validation = fit.evaluations[0]["metrics"]
    if not isinstance(baseline_validation, dict):
        raise RuntimeError("validation baseline metrics are malformed")
    validation_records: list[dict[str, object]] = []
    eligible: list[tuple[float, int]] = []
    for record in fit.evaluations:
        step = int(record["step"])
        metrics = record["metrics"]
        if not isinstance(metrics, dict):
            raise RuntimeError("validation metrics are malformed")
        decision = _decision(metrics, baseline_validation, args)
        validation_records.append({"step": step, "metrics": metrics, "decision": decision})
        horizon_metrics = metrics["horizons"]
        assert isinstance(horizon_metrics, dict)
        selected_horizon = horizon_metrics[str(int(args.correlation_window))]
        assert isinstance(selected_horizon, dict)
        if bool(decision["constraints_pass"]):
            eligible.append(
                (
                    float(selected_horizon["prefix_future_correlation_abs_error"]),
                    step,
                )
            )
    selected_step = min(eligible)[1] if eligible else 0
    unconstrained_candidates: list[tuple[float, int]] = []
    for record in validation_records:
        metrics = record["metrics"]
        assert isinstance(metrics, dict)
        horizons = metrics["horizons"]
        assert isinstance(horizons, dict)
        horizon_metrics = horizons[str(int(args.correlation_window))]
        assert isinstance(horizon_metrics, dict)
        unconstrained_candidates.append(
            (
                float(horizon_metrics["prefix_future_correlation_abs_error"]),
                int(record["step"]),
            )
        )
    best_correlation_step = min(unconstrained_candidates)[1]
    selected_validation = next(
        record for record in validation_records if int(record["step"]) == int(selected_step)
    )
    best_correlation_validation = next(
        record
        for record in validation_records
        if int(record["step"]) == int(best_correlation_step)
    )
    terminal_step = max(fit.checkpoints)
    combined_network.residual_network.load_state_dict(fit.checkpoints[int(terminal_step)])
    terminal_direction_diagnostics = evaluate_block_directions()
    combined_network.residual_network.load_state_dict(fit.checkpoints[int(selected_step)])
    selected_direction_diagnostics = evaluate_block_directions()
    combined_network.residual_network.load_state_dict(
        fit.checkpoints[int(best_correlation_step)]
    )
    best_correlation_direction_diagnostics = evaluate_block_directions()
    direction_diagnostics = {
        "probe_path_count": int(diagnostic_count),
        "probe_seed": int(direction_seed if direction_seed is not None else args.seed + 4),
        "baseline": baseline_direction_diagnostics,
        "terminal": terminal_direction_diagnostics,
        "selected": selected_direction_diagnostics,
        "best_correlation": best_correlation_direction_diagnostics,
    }
    write_json(run_dir / "block_direction_diagnostics.json", direction_diagnostics)
    combined_network.residual_network.load_state_dict(fit.checkpoints[int(selected_step)])

    heldout_generation_seed = derive_stream_seed(
        base_seed=int(args.seed),
        stream_offset=104_000,
    )
    heldout_generation_seed = int(
        heldout_generation_seed if heldout_generation_seed is not None else args.seed + 3
    )
    combined_network.residual_network.load_state_dict(fit.checkpoints[0])
    heldout_baseline, baseline_generated = _sample_metrics(
        model=model,
        reference_paths=heldout_paths,
        num_paths=int(args.num_heldout_generated_paths),
        seed=heldout_generation_seed,
        x0=float(heston.x0),
        args=args,
    )
    combined_network.residual_network.load_state_dict(fit.checkpoints[int(selected_step)])
    heldout_selected, selected_generated = _sample_metrics(
        model=model,
        reference_paths=heldout_paths,
        num_paths=int(args.num_heldout_generated_paths),
        seed=heldout_generation_seed,
        x0=float(heston.x0),
        args=args,
    )
    heldout_decision = _decision(heldout_selected, heldout_baseline, args)
    validation_decision = selected_validation["decision"]
    assert isinstance(validation_decision, dict)
    overall_pass = bool(validation_decision["pass"] and heldout_decision["pass"])

    selected_checkpoint = {
        "format_version": 2,
        "source_run_dir": str(source),
        "selected_step": int(selected_step),
        "residual_architecture": {
            "state_dim": int(model.architecture.state_dim),
            "noise_adjoint_dim": int(combined_network.noise_adjoint_dim),
            "hidden_dim": int(args.residual_hidden_dim),
            "num_layers": int(args.residual_num_layers),
            "parameterization": str(args.residual_parameterization),
            "initialization_seed": int(
                residual_seed if residual_seed is not None else args.seed
            ),
        },
        "residual_state_dict": fit.checkpoints[int(selected_step)],
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
        "marginal_anchor_weight": float(args.marginal_anchor_weight),
        "correlation_window_original_steps": int(args.correlation_window),
        "correlation_window_discrepancy_steps": int(discrepancy_window),
        "correlation_weight": float(args.correlation_weight),
    }
    torch.save(selected_checkpoint, run_dir / "residual_r_checkpoint.pt")
    torch.save(
        {
            "residual_state_dict": fit.checkpoints[max(fit.checkpoints)],
            "step": max(fit.checkpoints),
            "residual_noise_scale": fit.residual_noise_scale,
        },
        run_dir / "residual_r_terminal_checkpoint.pt",
    )
    torch.save(
        {
            "residual_state_dict": fit.checkpoints[int(best_correlation_step)],
            "step": int(best_correlation_step),
            "residual_noise_scale": fit.residual_noise_scale,
            "selection_status": "rejected_by_scale_or_swd_gate"
            if int(best_correlation_step) != int(selected_step)
            else "selected",
        },
        run_dir / "residual_r_best_correlation_checkpoint.pt",
    )
    torch.save(validation_reference, run_dir / "validation_paths.pt")
    torch.save(baseline_generated, run_dir / "heldout_baseline_generated_paths.pt")
    torch.save(selected_generated, run_dir / "heldout_selected_generated_paths.pt")
    write_history_jsonl(run_dir / "training_history.jsonl", fit.history)
    write_json(
        run_dir / "validation_history.json",
        {"records": validation_records},
    )
    report = {
        "source_run_dir": str(source),
        "run_dir": str(run_dir),
        "config": {
            "residual_training": asdict(training_config),
            "residual_parameterization": str(args.residual_parameterization),
            "residual_initialization_seed": int(
                residual_seed if residual_seed is not None else args.seed
            ),
            "correlation_window_original_steps": int(args.correlation_window),
            "correlation_window_discrepancy_steps": int(discrepancy_window),
            "correlation_uses_fine_grid": True,
            "correlation_weight": float(args.correlation_weight),
            "scale_floor_fraction": float(args.scale_floor_fraction),
            "marginal_anchor_weight": float(args.marginal_anchor_weight),
            "marginal_anchor_horizons": list(marginal_horizons),
            "marginal_window_endpoint_pairs": [list(pair) for pair in marginal_pairs],
            "direction_diagnostic_paths": int(diagnostic_count),
            "direction_diagnostic_seed": int(
                direction_seed if direction_seed is not None else args.seed + 4
            ),
            "num_validation_paths": int(args.num_validation_paths),
            "num_validation_generated_paths": int(args.num_validation_generated_paths),
            "num_heldout_generated_paths": int(args.num_heldout_generated_paths),
            "required_correlation_error_reduction": float(
                args.required_correlation_error_reduction
            ),
            "scale_error_tolerance": float(args.scale_error_tolerance),
            "swd_factor_limit": float(args.swd_factor_limit),
            "validation_seed": int(validation_seed),
            "validation_generation_seed": int(validation_generation_seed),
            "heldout_generation_seed": int(heldout_generation_seed),
        },
        "fixed_target": {
            "correlation": fixed_correlation.target_correlation,
            "prefix_scale_floor": fixed_correlation.prefix_scale_floor,
            "future_scale_floor": fixed_correlation.future_scale_floor,
            "marginal_window_endpoint_pairs": [list(pair) for pair in marginal_pairs],
            "marginal_target_mean": fixed_marginals.target_mean,
            "marginal_target_std": fixed_marginals.target_std,
            "marginal_target_iqr": fixed_marginals.target_iqr,
        },
        "block_direction_diagnostics": {
            "report": "block_direction_diagnostics.json",
            "baseline": baseline_direction_diagnostics,
            "terminal": terminal_direction_diagnostics,
            "selected": selected_direction_diagnostics,
            "best_correlation": best_correlation_direction_diagnostics,
        },
        "residual_scale": {
            "minimum": float(fit.residual_noise_scale.min().item()),
            "median": float(fit.residual_noise_scale.median().item()),
            "maximum": float(fit.residual_noise_scale.max().item()),
        },
        "selected_step": int(selected_step),
        "best_correlation_step": int(best_correlation_step),
        "validation_baseline": baseline_validation,
        "validation_selected": selected_validation,
        "validation_best_correlation": best_correlation_validation,
        "heldout_baseline": heldout_baseline,
        "heldout_selected": heldout_selected,
        "heldout_decision": heldout_decision,
        "overall_pass": overall_pass,
    }
    write_json(run_dir / "metrics.json", report)
    selected_h32 = heldout_selected["horizons"][str(int(args.correlation_window))]
    baseline_h32 = heldout_baseline["horizons"][str(int(args.correlation_window))]
    assert isinstance(selected_h32, dict) and isinstance(baseline_h32, dict)
    print(f"wrote residual-R persistence run to {run_dir}")
    print(
        "selected_step="
        f"{selected_step} best_correlation_step={best_correlation_step} "
        f"overall_pass={overall_pass} "
        "heldout_corr_error="
        f"{float(baseline_h32['prefix_future_correlation_abs_error']):.4f}->"
        f"{float(selected_h32['prefix_future_correlation_abs_error']):.4f} "
        "heldout_mean_ratio="
        f"{float(baseline_h32['future_rv_mean_ratio']):.4f}->"
        f"{float(selected_h32['future_rv_mean_ratio']):.4f} "
        "heldout_std_ratio="
        f"{float(baseline_h32['future_rv_std_ratio']):.4f}->"
        f"{float(selected_h32['future_rv_std_ratio']):.4f}"
    )


if __name__ == "__main__":
    main()
