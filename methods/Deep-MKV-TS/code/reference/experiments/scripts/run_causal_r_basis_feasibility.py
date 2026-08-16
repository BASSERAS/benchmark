from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from deep_mkv_gen_path_dt.noise import (  # noqa: E402
    derive_stream_seed,
    sample_standard_normals,
)
from path_dt_experiments.causal_r_feasibility import (  # noqa: E402
    CausalVolatilityBasisResidualNetwork,
)
from path_dt_experiments.constrained_feasibility import (  # noqa: E402
    ConstrainedFeasibilityConfig,
    audit_residual_constrained_feasibility,
)
from path_dt_experiments.residual_persistence import (  # noqa: E402
    FrozenBaseResidualRNetwork,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_residual_r_persistence import _build_source_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a direct causal realized-volatility basis in the R channel "
            "under exact and bootstrap-calibrated h32 marginal constraints."
        )
    )
    parser.add_argument("--residual-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--audit-dtype",
        choices=("source", "float64"),
        default="float64",
    )
    parser.add_argument("--num-paths", type=int, default=2048)
    parser.add_argument("--num-calibration-paths", type=int, default=4096)
    parser.add_argument("--windows", type=int, nargs="+", default=(4, 8, 16, 32))
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--prefix-length", type=int, default=65)
    parser.add_argument("--prefix-window", type=int, default=32)
    parser.add_argument(
        "--parameter-step-fractions",
        type=float,
        nargs="+",
        default=(1e-6, 3e-6, 1e-5),
    )
    parser.add_argument("--parameter-step-scale", type=float, default=1.0)
    parser.add_argument("--constraint-change-tolerance", type=float, default=1e-8)
    parser.add_argument("--minimum-retained-direction-fraction", type=float, default=0.10)
    parser.add_argument("--bootstrap-replicates", type=int, default=200)
    parser.add_argument(
        "--bootstrap-budget-multipliers",
        type=float,
        nargs="+",
        default=(0.5, 1.0, 2.0, 5.0),
    )
    parser.add_argument(
        "--pareto-target-correlation-reduction-fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    residual_run = Path(args.residual_run_dir)
    run_dir = ensure_run_dir(args.run_dir)
    residual_checkpoint = torch.load(
        residual_run / "residual_r_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(residual_checkpoint, dict):
        raise ValueError("residual_r_checkpoint.pt must contain a dictionary")
    source_value = residual_checkpoint.get("source_run_dir")
    residual_scale = residual_checkpoint.get("residual_noise_scale")
    if not isinstance(source_value, str):
        raise ValueError("residual checkpoint is missing source_run_dir")
    if not isinstance(residual_scale, torch.Tensor):
        raise ValueError("residual checkpoint is missing residual_noise_scale")
    source = Path(source_value)
    if not source.is_absolute():
        source = REPO_ROOT / source

    model, _, stored_config, heston, target_paths, heldout_paths = _build_source_model(
        source=source,
        device=torch.device(args.device),
    )
    audit_dtype = model.dtype if str(args.audit_dtype) == "source" else torch.float64
    model.network.to(device=model.device, dtype=audit_dtype)
    base_scale = model._noise_adjoint_target_scale
    if base_scale is None:
        raise ValueError("source checkpoint must contain a noise-adjoint target scale")

    basis = CausalVolatilityBasisResidualNetwork(
        state_dim=int(model.architecture.state_dim),
        noise_adjoint_dim=int(model._default_noise_adjoint_dim(model.architecture)),
        num_steps=int(model.grid.num_steps),
        dt=float(model.grid.dt),
        windows=tuple(int(value) for value in args.windows),
    )
    basis.to(device=model.device, dtype=model.dtype)
    combined_network = FrozenBaseResidualRNetwork(
        base_network=model.network,
        residual_network=basis,
        base_noise_scale=base_scale,
    )
    combined_network.to(device=model.device, dtype=model.dtype)
    combined_network.set_residual_noise_scale(residual_scale)
    model.network = combined_network
    model._validate_network_contract(
        network=combined_network,
        architecture=model.architecture,
    )

    calibration_seed = derive_stream_seed(
        base_seed=int(args.seed),
        stream_offset=119_000,
    )
    calibration_count = int(args.num_calibration_paths)
    if calibration_count < 2:
        raise ValueError("num-calibration-paths must be >= 2")
    calibration_x0 = torch.tensor(
        [[float(heston.x0)]],
        device=model.device,
        dtype=model.dtype,
    ).repeat(calibration_count, 1)
    calibration_noise = sample_standard_normals(
        grid=model.grid,
        batch_size=calibration_count,
        noise_dim=int(model.architecture.noise_dim),
        device=model.device,
        dtype=torch.float32,
        seed=int(
            calibration_seed if calibration_seed is not None else int(args.seed)
        ),
    ).to(dtype=model.dtype)
    with torch.no_grad():
        calibration_paths = model.rollout(
            x0=calibration_x0,
            noise=calibration_noise,
        ).paths
    basis.fit_calibration(calibration_paths)

    result = audit_residual_constrained_feasibility(
        model=model,
        training_reference_paths=target_paths,
        validation_reference_paths=heldout_paths,
        x0=torch.tensor(
            [[float(heston.x0)]],
            device=model.device,
            dtype=model.dtype,
        ),
        config=ConstrainedFeasibilityConfig(
            num_paths=int(args.num_paths),
            horizon=int(args.horizon),
            prefix_length=int(args.prefix_length),
            prefix_window=int(args.prefix_window),
            parameter_step_fractions=tuple(
                float(value) for value in args.parameter_step_fractions
            ),
            parameter_step_scale=float(args.parameter_step_scale),
            constraint_change_tolerance=float(args.constraint_change_tolerance),
            minimum_retained_direction_fraction=float(
                args.minimum_retained_direction_fraction
            ),
            bootstrap_replicates=int(args.bootstrap_replicates),
            bootstrap_budget_multipliers=tuple(
                float(value) for value in args.bootstrap_budget_multipliers
            ),
            pareto_target_correlation_reduction_fraction=float(
                args.pareto_target_correlation_reduction_fraction
            ),
            seed=int(args.seed),
        ),
    )
    split_index = int(args.prefix_length) - 1
    prefix_start = split_index - int(args.prefix_window)
    future_end = split_index + int(args.horizon)

    def direction_localization(direction: torch.Tensor) -> dict[str, object]:
        shaped = direction.detach().double().reshape_as(basis.coefficients)
        squared = shaped.pow(2)
        total = float(squared.sum().item())
        time_norms = torch.sqrt(squared.sum(dim=(1, 2)))
        feature_norms = torch.sqrt(squared.sum(dim=(0, 2)))
        top_count = min(12, int(time_norms.numel()))
        top_times = torch.topk(time_norms, top_count)

        def energy_fraction(start: int, end: int) -> float:
            if total <= torch.finfo(torch.float64).eps or end <= start:
                return 0.0
            return float(squared[start:end].sum().item() / total)

        return {
            "time_energy_fractions": {
                "before_prefix_window": energy_fraction(0, prefix_start),
                "prefix_window": energy_fraction(prefix_start, split_index),
                "future_window": energy_fraction(split_index, future_end),
                "after_future_window": energy_fraction(
                    future_end,
                    int(model.grid.num_steps),
                ),
            },
            "top_time_indices": [
                {
                    "step": int(index.item()),
                    "coefficient_norm": float(value.item()),
                }
                for value, index in zip(top_times.values, top_times.indices)
            ],
            "feature_norms": {
                name: float(value.item())
                for name, value in zip(basis.feature_names, feature_norms)
            },
        }

    localization = {
        "unconstrained_correlation_descent": direction_localization(
            result.tensors["unconstrained_direction"]
        ),
        "projected_feasible_descent": direction_localization(
            result.tensors["projected_direction"]
        ),
    }
    half_se_key = "direction_bootstrap_0.5se_budget"
    if half_se_key in result.tensors:
        localization["bootstrap_0.5se_budget"] = direction_localization(
            result.tensors[half_se_key]
        )
    result.metrics.update(
        {
            "source_run_dir": str(source),
            "residual_run_dir": str(residual_run),
            "source_dtype": str(stored_config["dtype"]),
            "audit_dtype": str(model.dtype),
            "audit_residual_parameterization": "causal_volatility_basis",
            "basis": {
                "windows": list(basis.windows),
                "feature_dim": int(basis.feature_dim),
                "feature_names": list(basis.feature_names),
                "coefficient_count": int(basis.coefficients.numel()),
                "calibration_path_count": calibration_count,
                "calibration_seed": int(
                    calibration_seed
                    if calibration_seed is not None
                    else int(args.seed)
                ),
                "uses_reference_paths_for_features": False,
                "causal": True,
            },
            "direction_localization": localization,
        }
    )
    write_json(run_dir / "causal_r_basis_feasibility.json", result.metrics)
    torch.save(result.tensors, run_dir / "causal_r_basis_feasibility_tensors.pt")
    torch.save(
        {
            "state_dict": basis.state_dict(),
            "windows": tuple(basis.windows),
            "feature_dim": int(basis.feature_dim),
        },
        run_dir / "causal_r_basis.pt",
    )

    exact = result.metrics["projection"]
    pareto = result.metrics["bootstrap_pareto_geometry"]
    pareto_summary = ", ".join(
        f"{name}={float(record['retained_direction_fraction']):.4f}"
        for name, record in pareto.items()
    )
    print(f"wrote causal-R basis audit to {run_dir}")
    print(
        "exact_retained="
        f"{float(exact['retained_direction_fraction']):.4f} "
        f"pareto=[{pareto_summary}]"
    )


if __name__ == "__main__":
    main()
