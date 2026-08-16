#!/usr/bin/env python3
"""Verify the exact dual witness for one residual-volatility MMD block."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from path_dt_experiments.dual_mmd_witness import (  # noqa: E402
    FixedEmpiricalRBFMMDWitness,
)
from path_dt_experiments.extragradient_solver import _sample_inputs  # noqa: E402
from path_dt_experiments.lifted_collocation_solver import (  # noqa: E402
    AdjointNetworkControlPolicy,
    rollout_causal_policy,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from path_dt_experiments.validated_nested_solver import (  # noqa: E402
    _sample_target_batch,
)
from run_residual_volatility_reduced_spectral_audit import (  # noqa: E402
    _construct_model,
)
from run_residual_volatility_first_operator_audit import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_SIGNAL_RUN,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--signal-run-dir", type=Path, default=DEFAULT_SIGNAL_RUN)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calibration-paths", type=int, default=4096)
    parser.add_argument("--witness-population-paths", type=int, default=1024)
    parser.add_argument("--witness-target-paths", type=int, default=1024)
    parser.add_argument("--query-paths", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2_608_701)
    parser.add_argument("--network-init-seed", type=int, default=2_608_401)
    return parser.parse_args()


def _model_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        device=args.device,
        dataset_root=args.dataset_root,
        signal_run_dir=args.signal_run_dir,
        calibration_paths=int(args.calibration_paths),
        variant="residual_conditional_eta4",
        network_init_seed=int(args.network_init_seed),
    )


def _tensor_sha256(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    tensor = value.detach().cpu().contiguous()
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _residual_block(model) -> tuple[object, float, str]:
    matches = [
        block
        for block in model.discrepancy.blocks
        if block.name == "volatility_law_reference_residual_conditional_rv"
    ]
    if len(matches) != 1:
        raise RuntimeError("expected one residual conditional-volatility block")
    block = matches[0]
    return block.discrepancy, float(block.weight), str(block.name)


def main() -> None:
    args = parse_args()
    run_dir = ensure_run_dir(args.run_dir.resolve())
    model, paths = _construct_model(_model_args(args))
    residual, block_weight, block_name = _residual_block(model)
    policy = AdjointNetworkControlPolicy(model=model, network=model.network)

    x0, noise = _sample_inputs(
        model=model,
        initial_state_pool=paths,
        count=int(args.witness_population_paths),
        seed=int(args.seed) + 1,
    )
    reference_law = rollout_causal_policy(
        model=model,
        policy=policy,
        x0=x0,
        noise=noise,
    )
    if not torch.equal(
        reference_law.expected_adjoint_next,
        torch.zeros_like(reference_law.expected_adjoint_next),
    ) or not torch.equal(
        reference_law.expected_adjoint_noise_next,
        torch.zeros_like(reference_law.expected_adjoint_noise_next),
    ):
        raise RuntimeError("witness must be fitted at the exact zero-adjoint reference")
    target_support = _sample_target_batch(
        model=model,
        pool=paths,
        count=int(args.witness_target_paths),
        seed=int(args.seed) + 2,
    )
    population_features = residual.features(reference_law.paths, grid=model.grid).detach()
    target_features = residual.features(target_support, grid=model.grid).detach()
    witness = FixedEmpiricalRBFMMDWitness(
        residual_feature_map=residual,
        population_features=population_features,
        target_features=target_features,
        bandwidths=tuple(residual.bandwidths),
    )

    primal = residual(
        generated_paths=reference_law.paths,
        target_paths=target_support,
        grid=model.grid,
    ).value
    dual = witness(
        generated_paths=reference_law.paths,
        target_paths=target_support.flip(0),
        grid=model.grid,
    ).value
    value_absolute_error = float(torch.abs(primal - dual).item())
    value_relative_error = value_absolute_error / max(
        float(torch.abs(primal).item()), torch.finfo(torch.float64).eps
    )

    query_x0, query_noise = _sample_inputs(
        model=model,
        initial_state_pool=paths,
        count=int(args.query_paths),
        seed=int(args.seed) + 3,
    )
    query_law = rollout_causal_policy(
        model=model,
        policy=policy,
        x0=query_x0,
        noise=query_noise,
    )
    original_query = query_law.paths.detach().clone().requires_grad_(True)
    original_potential = residual.lions_potential(
        query_paths=original_query,
        population_paths=reference_law.paths,
        target_paths=target_support,
        grid=model.grid,
    )
    original_gradient = torch.autograd.grad(original_potential, original_query)[0]
    fixed_query = query_law.paths.detach().clone().requires_grad_(True)
    fixed_potential = witness.lions_potential(
        query_paths=fixed_query,
        population_paths=query_law.paths.flip(0),
        target_paths=target_support.flip(0),
        grid=model.grid,
    )
    fixed_gradient = torch.autograd.grad(fixed_potential, fixed_query)[0]
    potential_absolute_error = float(
        torch.abs(original_potential - fixed_potential).item()
    )
    gradient_difference = fixed_gradient - original_gradient
    gradient_relative_error = float(
        torch.linalg.vector_norm(gradient_difference.double())
        / torch.linalg.vector_norm(original_gradient.double()).clamp_min(
            torch.finfo(torch.float64).eps
        )
    )

    scales = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
    dual_curve: dict[str, float] = {}
    for scale in scales:
        scaled = FixedEmpiricalRBFMMDWitness(
            residual_feature_map=residual,
            population_features=population_features,
            target_features=target_features,
            bandwidths=tuple(residual.bandwidths),
            witness_scale=float(scale),
        )
        dual_curve[str(scale)] = float(
            scaled(
                generated_paths=reference_law.paths,
                target_paths=target_support,
                grid=model.grid,
            ).value.item()
        )
    maximizing_scale = float(max(dual_curve, key=dual_curve.__getitem__))
    exact_value = value_relative_error <= 1e-5
    exact_source = gradient_relative_error <= 1e-5
    exact_maximum = maximizing_scale == 1.0
    passed = exact_value and exact_source and exact_maximum
    artifact_path = run_dir / "fixed_residual_witness.pt"
    torch.save(
        {
            "witness": witness,
            "block_name": block_name,
            "block_weight": block_weight,
            "population_paths_sha256": _tensor_sha256(reference_law.paths),
            "target_paths_sha256": _tensor_sha256(target_support),
            "protocol_seed": int(args.seed),
        },
        artifact_path,
    )
    report = {
        "protocol": {
            "objective_variant": "residual_conditional_eta4",
            "witness_fitted_at_exact_zero_adjoint_reference": True,
            "population_support_is_frozen": True,
            "target_support_is_frozen": True,
            "exact_empirical_rbf_rkhs": True,
            "random_features_used": False,
            "biased_population_mmd": True,
            "test_paths_used": False,
            "seed": int(args.seed),
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "block": {"name": block_name, "weight": block_weight},
        "witness": witness.summary(),
        "fingerprints": {
            "population_paths_sha256": _tensor_sha256(reference_law.paths),
            "target_paths_sha256": _tensor_sha256(target_support),
            "population_features_sha256": _tensor_sha256(population_features),
            "target_features_sha256": _tensor_sha256(target_features),
        },
        "equivalence": {
            "primal_mmd_value": float(primal.item()),
            "maximized_dual_value": float(dual.item()),
            "value_absolute_error": value_absolute_error,
            "value_relative_error": value_relative_error,
            "original_lions_potential": float(original_potential.item()),
            "fixed_witness_lions_potential": float(fixed_potential.item()),
            "potential_absolute_error": potential_absolute_error,
            "source_gradient_relative_error": gradient_relative_error,
            "dual_scale_curve": dual_curve,
            "maximizing_scale": maximizing_scale,
        },
        "gate": {
            "exact_value": exact_value,
            "exact_source": exact_source,
            "exact_dual_maximizer": exact_maximum,
            "passed": passed,
            "next_phase": "fixed_witness_spectrum" if passed else "stop",
        },
        "artifact": str(artifact_path.resolve()),
    }
    write_json(run_dir / "report.json", report)
    (run_dir / "SUMMARY.md").write_text(
        "# Residual-volatility dual-witness identity audit\n\n"
        f"- Gate passed: `{passed}`\n"
        f"- Primal MMD / dual value: `{primal.item():.9g}` / `{dual.item():.9g}`\n"
        f"- Relative value error: `{value_relative_error:.3g}`\n"
        f"- Relative Lions-source gradient error: `{gradient_relative_error:.3g}`\n"
        f"- Maximizing witness scale on the locked curve: `{maximizing_scale:g}`\n",
        encoding="utf-8",
    )
    print(json.dumps(report["gate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
