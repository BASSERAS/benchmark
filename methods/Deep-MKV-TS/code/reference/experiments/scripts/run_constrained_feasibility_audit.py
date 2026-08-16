from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from path_dt_experiments.constrained_feasibility import (  # noqa: E402
    ConstrainedFeasibilityConfig,
    audit_residual_constrained_feasibility,
)
from path_dt_experiments.residual_persistence import (  # noqa: E402
    CenteredNoiseAdjointResidualNetwork,
    FrozenBaseResidualRNetwork,
    NoiseAdjointResidualNetwork,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_residual_r_persistence import _build_source_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project the end-to-end h32 persistence direction onto separate "
            "mean/std/IQR non-worsening constraints and verify it by common-noise "
            "finite differences."
        )
    )
    parser.add_argument("--residual-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--audit-dtype",
        choices=("source", "float64"),
        default="float64",
        help=(
            "Numerical precision for gradients and finite differences. "
            "float64 avoids losing the very small projected response."
        ),
    )
    parser.add_argument(
        "--residual-parameterization",
        choices=("checkpoint", "centered_full_tangent"),
        default="checkpoint",
        help=(
            "Use the checkpoint parameterization, or replace an original "
            "zero-final residual by an exactly-zero centered residual whose "
            "full network tangent is active."
        ),
    )
    parser.add_argument("--centered-init-seed", type=int, default=1234)
    parser.add_argument("--num-paths", type=int, default=2048)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--prefix-length", type=int, default=65)
    parser.add_argument("--prefix-window", type=int, default=32)
    parser.add_argument(
        "--parameter-step-fractions",
        type=float,
        nargs="+",
        default=(1e-6, 3e-6, 1e-5),
    )
    parser.add_argument("--constraint-change-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--minimum-retained-direction-fraction",
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
    architecture_payload = residual_checkpoint.get("residual_architecture")
    residual_state = residual_checkpoint.get("residual_state_dict")
    residual_scale = residual_checkpoint.get("residual_noise_scale")
    if not isinstance(source_value, str):
        raise ValueError("residual checkpoint is missing source_run_dir")
    if not isinstance(architecture_payload, dict) or not isinstance(residual_state, dict):
        raise ValueError("residual checkpoint is missing its architecture or state")
    if not isinstance(residual_scale, torch.Tensor):
        raise ValueError("residual checkpoint is missing residual_noise_scale")

    source = Path(source_value)
    if not source.is_absolute():
        source = REPO_ROOT / source
    device = torch.device(args.device)
    model, _, stored_config, heston, target_paths, heldout_paths = _build_source_model(
        source=source,
        device=device,
    )
    audit_dtype = model.dtype if str(args.audit_dtype) == "source" else torch.float64
    model.network.to(device=model.device, dtype=audit_dtype)
    base_scale = model._noise_adjoint_target_scale
    if base_scale is None:
        raise ValueError("source checkpoint must contain a noise-adjoint target scale")
    stored_parameterization = str(
        architecture_payload.get("parameterization", "zero_final")
    )
    effective_parameterization = (
        stored_parameterization
        if str(args.residual_parameterization) == "checkpoint"
        else "centered_full_tangent"
    )
    raw_residual_network = NoiseAdjointResidualNetwork(
        state_dim=int(architecture_payload["state_dim"]),
        noise_adjoint_dim=int(architecture_payload["noise_adjoint_dim"]),
        hidden_dim=int(architecture_payload["hidden_dim"]),
        num_layers=int(architecture_payload["num_layers"]),
        zero_final=effective_parameterization == "zero_final",
    )
    if effective_parameterization == "centered_full_tangent":
        if stored_parameterization == "centered_full_tangent":
            residual_network = CenteredNoiseAdjointResidualNetwork(
                raw_residual_network
            )
            residual_network.load_state_dict(residual_state)
        else:
            raw_residual_network.load_state_dict(residual_state)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(args.centered_init_seed))
                raw_residual_network.reset_output_layer()
            residual_network = CenteredNoiseAdjointResidualNetwork(
                raw_residual_network
            )
    elif effective_parameterization == "zero_final":
        raw_residual_network.load_state_dict(residual_state)
        residual_network = raw_residual_network
    else:
        raise ValueError(
            f"unsupported residual parameterization: {effective_parameterization}"
        )
    residual_network.to(device=model.device, dtype=model.dtype)
    combined_network = FrozenBaseResidualRNetwork(
        base_network=model.network,
        residual_network=residual_network,
        base_noise_scale=base_scale,
    )
    combined_network.to(device=model.device, dtype=model.dtype)
    combined_network.set_residual_noise_scale(residual_scale)
    model.network = combined_network
    model._validate_network_contract(
        network=combined_network,
        architecture=model.architecture,
    )

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
            constraint_change_tolerance=float(args.constraint_change_tolerance),
            minimum_retained_direction_fraction=float(
                args.minimum_retained_direction_fraction
            ),
            seed=int(args.seed),
        ),
    )
    result.metrics["source_run_dir"] = str(source)
    result.metrics["residual_run_dir"] = str(residual_run)
    result.metrics["source_selected_step"] = int(
        residual_checkpoint.get("selected_step", -1)
    )
    result.metrics["source_dtype"] = str(stored_config["dtype"])
    result.metrics["audit_dtype"] = str(model.dtype)
    result.metrics["stored_residual_parameterization"] = stored_parameterization
    result.metrics["audit_residual_parameterization"] = effective_parameterization
    if effective_parameterization == "centered_full_tangent":
        result.metrics["centered_init_seed"] = int(args.centered_init_seed)
    write_json(run_dir / "constrained_feasibility.json", result.metrics)
    torch.save(result.tensors, run_dir / "constrained_feasibility_tensors.pt")

    decision = result.metrics["decision"]
    projection = result.metrics["projection"]
    print(f"wrote constrained feasibility audit to {run_dir}")
    print(
        "retained_fraction="
        f"{float(projection['retained_direction_fraction']):.4f} "
        "finite_difference_pass="
        f"{bool(decision['finite_difference_validation_pass'])} "
        f"local_feasibility_pass={bool(decision['local_feasibility_pass'])}"
    )


if __name__ == "__main__":
    main()
