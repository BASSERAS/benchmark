from __future__ import annotations

import argparse
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
from path_dt_experiments.discrepancies import build_heston_discrepancy  # noqa: E402
from path_dt_experiments.dynamics import DriftVolatilityEulerStep  # noqa: E402
from path_dt_experiments.frozen_response import (  # noqa: E402
    FrozenResponseAuditConfig,
    audit_frozen_block_responses,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the block-by-metric response audit from a saved Heston checkpoint.",
    )
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--audit-preset",
        choices=("simple", "old_fullv_w0p25", "reduced_time_local"),
        default=None,
    )
    parser.add_argument(
        "--add-correlation-candidate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--direction-batch-size", type=int, default=256)
    parser.add_argument("--path-gradient-batch-size", type=int, default=128)
    parser.add_argument("--target-batch-size", type=int, default=256)
    parser.add_argument("--metric-paths", type=int, default=512)
    parser.add_argument("--metric-batch-size", type=int, default=512)
    parser.add_argument("--parameter-step-fraction", type=float, default=1e-5)
    parser.add_argument("--path-return-step-fraction", type=float, default=1e-2)
    parser.add_argument("--prefix-length", type=int, default=None)
    parser.add_argument("--prefix-window", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--swd-projections", type=int, default=32)
    parser.add_argument("--time-marginal-volatility-window", type=int, default=None)
    parser.add_argument("--time-marginal-volatility-endpoints", type=int, nargs="+", default=None)
    parser.add_argument("--time-marginal-volatility-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _observed_indices(*, num_steps: int, spacing: int) -> tuple[int, ...]:
    if spacing < 1 or num_steps % spacing != 0:
        raise ValueError("stored observed_every must divide num_steps")
    return tuple(range(0, num_steps + 1, spacing))


def main() -> None:
    args = parse_args()
    source = Path(args.source_run_dir)
    run_dir = ensure_run_dir(args.run_dir)
    metrics = _load_json(source / "metrics.json")
    config = metrics.get("config")
    if not isinstance(config, dict):
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

    heston = config.get("heston")
    if not isinstance(heston, dict):
        raise ValueError("source config is missing heston")
    num_steps = int(heston["num_steps"])
    observed_every = int(config["observed_every"])
    grid = DiscreteTimeGrid(
        T=float(heston["T"]),
        num_steps=num_steps,
        observed_indices=_observed_indices(num_steps=num_steps, spacing=observed_every),
    )

    control_config = config.get("control_map")
    if not isinstance(control_config, dict) or control_config.get("name") != "entropy_barrier":
        raise ValueError("offline runner currently supports entropy_barrier Heston checkpoints")
    control_map = EntropyBarrierDiagonalControl(
        dt=grid.dt,
        kappa=float(control_config["kappa"]),
        tau=float(control_config["tau"]),
        sigma_min=float(config["sigma_min"]),
        sigma_max=float(config["sigma_max"]),
    )

    existing_correlation = bool(config.get("conditional_volatility_correlation_discrepancy", False))
    add_candidate = bool(args.add_correlation_candidate) and not existing_correlation
    correlation_windows_raw = tuple(
        int(value)
        for value in config.get("conditional_volatility_correlation_windows", (16, 32))
    )
    correlation_windows = (
        tuple(value // observed_every for value in correlation_windows_raw)
        if bool(training.observed_only)
        else correlation_windows_raw
    )
    if any(value < 1 for value in correlation_windows):
        raise ValueError("stored conditional-correlation windows do not fit the observed grid")
    discrepancy_steps = int(grid.observed_count) - 1 if bool(training.observed_only) else num_steps
    audit_preset = str(args.audit_preset) if args.audit_preset is not None else str(config["preset"])
    stored_marginal_endpoints = config.get("time_marginal_volatility_endpoints", ())
    if not isinstance(stored_marginal_endpoints, (list, tuple)):
        raise ValueError("stored time_marginal_volatility_endpoints must be a sequence")
    discrepancy = build_heston_discrepancy(
        num_steps=discrepancy_steps,
        include_acf=bool(config["include_acf"]),
        preset=audit_preset,
        lambda_scale=float(config["lambda_scale"]),
        kappa_scale=float(config["kappa_scale"]),
        include_conditional_volatility=bool(config.get("conditional_volatility_discrepancy", False)),
        include_conditional_volatility_correlation=existing_correlation or add_candidate,
        conditional_volatility_windows=tuple(
            int(value)
            for value in config.get("conditional_volatility_discrepancy_windows", ())
        ),
        conditional_volatility_correlation_windows=correlation_windows,
        conditional_volatility_weight=float(config.get("conditional_volatility_weight", 1.0)),
        conditional_volatility_correlation_weight=float(
            config.get("conditional_volatility_correlation_weight", 0.025)
        ),
        conditional_volatility_bandwidths=tuple(
            float(value)
            for value in config.get("conditional_volatility_bandwidths", (0.5, 1.0, 2.0))
        ),
        conditional_volatility_correlation_full_matrix=bool(
            config.get("conditional_volatility_correlation_full_matrix", False)
        ),
        conditional_volatility_correlation_scale_floor_fraction=float(
            config.get("conditional_volatility_correlation_scale_floor_fraction", 0.25)
        ),
        time_marginal_volatility_window=(
            int(args.time_marginal_volatility_window)
            if args.time_marginal_volatility_window is not None
            else (
                None
                if config.get("time_marginal_volatility_window") is None
                else int(config["time_marginal_volatility_window"])
            )
        ),
        time_marginal_volatility_endpoints=(
            tuple(int(value) for value in args.time_marginal_volatility_endpoints)
            if args.time_marginal_volatility_endpoints is not None
            else tuple(int(value) for value in stored_marginal_endpoints)
        ),
        time_marginal_volatility_weight=float(args.time_marginal_volatility_weight),
    )
    model = DiscreteMPModel(
        architecture=architecture,
        grid=grid,
        dynamics_step=DriftVolatilityEulerStep(dt=grid.dt),
        control_map=control_map,
        discrepancy=discrepancy,
    )
    dtype = torch.float64 if str(config["dtype"]) == "float64" else torch.float32
    model.network.to(device=torch.device(args.device), dtype=dtype)
    model.load_checkpoint_state(checkpoint)

    target_paths = torch.load(source / "target_paths.pt", map_location="cpu", weights_only=True)
    reference_paths = torch.load(source / "heldout_paths.pt", map_location="cpu", weights_only=True)
    result = audit_frozen_block_responses(
        model=model,
        discrepancy=discrepancy,
        target_paths=target_paths,
        reference_paths=reference_paths,
        x0=torch.tensor([[float(heston["x0"])]], device=model.device, dtype=model.dtype),
        training=training,
        config=FrozenResponseAuditConfig(
            direction_batch_size=int(args.direction_batch_size),
            path_gradient_batch_size=int(args.path_gradient_batch_size),
            target_batch_size=int(args.target_batch_size),
            metric_num_paths=int(args.metric_paths),
            metric_batch_size=int(args.metric_batch_size),
            parameter_step_fraction=float(args.parameter_step_fraction),
            path_return_step_fraction=float(args.path_return_step_fraction),
            prefix_length=args.prefix_length,
            prefix_window=args.prefix_window,
            horizon=args.horizon,
            swd_projections=int(args.swd_projections),
            seed=int(args.seed),
        ),
    )
    result.metrics["source_run_dir"] = str(source)
    result.metrics["audit_preset"] = audit_preset
    result.metrics["candidate_correlation_block_not_used_in_training"] = add_candidate
    write_json(run_dir / "frozen_response_audit.json", result.metrics)
    torch.save(result.tensors, run_dir / "frozen_response_tensors.pt")
    print(f"wrote frozen response audit to {run_dir}")


if __name__ == "__main__":
    main()
