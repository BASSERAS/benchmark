#!/usr/bin/env python3
"""Run one spectrum-damped law/direct-ridge-adjoint refresh cycle."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from path_dt_experiments.direct_ridge_adjoint import (  # noqa: E402
    BlockwiseAdjointInterpolationPolicy,
    DirectRidgeRAdjointPolicy,
    TimeSmoothedRidgeProjection,
)
from path_dt_experiments.lifted_collocation_solver import (  # noqa: E402
    AdjointNetworkControlPolicy,
    ScenarioDataContext,
    _bank_consensus_metrics,
    build_fit_confirmation_banks,
    rollout_policy_on_bank,
)
from path_dt_experiments.lifted_poll_solver import (  # noqa: E402
    FrozenLiftedMerit,
    evaluate_complete_lifted_cycle,
)
from path_dt_experiments.terminal_fixed_point_diagnostic import _time_blocks  # noqa: E402
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_residual_volatility_reduced_spectral_audit import _construct_model  # noqa: E402
from run_residual_volatility_first_operator_audit import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_SIGNAL_RUN,
)


DEFAULT_DIRECT_RUN = REPO_ROOT / "runs" / "direct_ridge_r_audit_seed0_20260803"
DEFAULT_SPECTRAL_RUN = REPO_ROOT / "runs" / "direct_ridge_r_spectral_20260803"
DEFAULT_CHECKPOINT_ROOT = (
    REPO_ROOT / "runs" / "residual_volatility_first_operator_corrected_parts_20260803"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--direct-run-dir", type=Path, default=DEFAULT_DIRECT_RUN)
    parser.add_argument("--spectral-run-dir", type=Path, default=DEFAULT_SPECTRAL_RUN)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--signal-run-dir", type=Path, default=DEFAULT_SIGNAL_RUN)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calibration-paths", type=int, default=4096)
    parser.add_argument("--fit-paths", type=int, default=256)
    parser.add_argument("--target-paths", type=int, default=1024)
    parser.add_argument("--branches", type=int, default=16)
    parser.add_argument("--audit-paths", type=int, default=64)
    parser.add_argument("--audit-target-paths", type=int, default=512)
    parser.add_argument("--audit-branches", type=int, default=16)
    parser.add_argument("--query-batch-size", type=int, default=4096)
    parser.add_argument("--relaxation", type=float, default=1.0 / 512.0)
    parser.add_argument("--seed", type=int, default=2_608_701)
    parser.add_argument("--network-init-seed", type=int, default=2_608_401)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < float(args.relaxation) < 1.0:
        raise ValueError("relaxation must lie in (0,1)")
    run_dir = ensure_run_dir(args.run_dir.resolve())
    model_args = argparse.Namespace(
        device=args.device,
        dataset_root=args.dataset_root,
        signal_run_dir=args.signal_run_dir,
        calibration_paths=int(args.calibration_paths),
        variant="residual_conditional_eta4",
        network_init_seed=int(args.network_init_seed),
    )
    model, paths = _construct_model(model_args)
    checkpoint = torch.load(
        args.checkpoint_root.resolve()
        / "residual_eta4"
        / "fresh_adjoint_residual_conditional_eta4.pt",
        map_location=model.device,
        weights_only=False,
    )
    p_network = copy.deepcopy(model.network)
    p_network.load_state_dict(checkpoint["state_dict"])
    p_network.eval()
    for module in p_network.modules():
        if isinstance(module, torch.nn.GRU):
            module.flatten_parameters()
    old_projection = torch.load(
        args.direct_run_dir.resolve() / "residual_aware_ridge.pt",
        map_location="cpu",
        weights_only=False,
    )
    base_policy = AdjointNetworkControlPolicy(model=model, network=model.network)
    pre_refit_policy = DirectRidgeRAdjointPolicy(
        model=model, p_network=p_network, r_projection=old_projection
    )
    blocks = _time_blocks(int(model.grid.num_steps), 3)
    theta = torch.full(
        (2 * len(blocks),), float(args.relaxation), dtype=torch.float64
    )
    moved_policy = BlockwiseAdjointInterpolationPolicy(
        model=model,
        base_policy=base_policy,
        candidate_policy=pre_refit_policy,
        theta=theta,
        blocks=blocks,
    )
    context = ScenarioDataContext(
        initial_state_pool=paths.detach().cpu(),
        target_path_pool=paths.detach().cpu(),
        dataset_id=json.dumps(
            {
                "path": str(args.dataset_root.resolve()),
                "split": "train.npy deterministic prefix",
                "count": int(args.calibration_paths),
            },
            sort_keys=True,
        ),
    )
    refit_fit, refit_validation = build_fit_confirmation_banks(
        model=model,
        context=context,
        path_count=int(args.fit_paths),
        target_count=int(args.target_paths),
        num_branches=int(args.branches),
        antithetic=True,
        beta=1.0,
        seed=int(args.seed),
    )
    audit_fit, audit_confirmation = build_fit_confirmation_banks(
        model=model,
        context=context,
        path_count=int(args.audit_paths),
        target_count=int(args.audit_target_paths),
        num_branches=int(args.audit_branches),
        antithetic=True,
        beta=1.0,
        seed=int(args.seed) + 20_000,
    )
    print("constructing moved-law ridge targets", flush=True)
    moved_fit = rollout_policy_on_bank(
        model=model,
        policy=moved_policy,
        bank=refit_fit,
        data_context=context,
        full_training=model.training,
        query_batch_size=int(args.query_batch_size),
    )
    moved_validation = rollout_policy_on_bank(
        model=model,
        policy=moved_policy,
        bank=refit_validation,
        data_context=context,
        full_training=model.training,
        query_batch_size=int(args.query_batch_size),
    )
    refreshed_projection = TimeSmoothedRidgeProjection.fit(
        feature_map=old_projection.feature_map,
        paths=moved_fit.paths,
        targets=moved_fit.target_r,
        ridge=float(old_projection.ridge),
    )
    post_refit_policy = DirectRidgeRAdjointPolicy(
        model=model, p_network=p_network, r_projection=refreshed_projection
    )
    refit_diagnostics = {
        "fit": refreshed_projection.diagnostics(
            paths=moved_fit.paths, targets=moved_fit.target_r
        ),
        "validation": refreshed_projection.diagnostics(
            paths=moved_validation.paths, targets=moved_validation.target_r
        ),
    }
    print("constructing frozen baseline merit", flush=True)
    baseline_evaluation = rollout_policy_on_bank(
        model=model,
        policy=base_policy,
        bank=audit_fit,
        data_context=context,
        full_training=model.training,
        query_batch_size=int(args.query_batch_size),
    )
    _, baseline_artifacts = _bank_consensus_metrics(
        model=model,
        evaluation=baseline_evaluation,
        adjoint_policy=pre_refit_policy,
    )
    merit = FrozenLiftedMerit.from_baseline(
        baseline_artifacts,
        p_estimator_noise_variance=0.0,
        r_estimator_noise_variance=0.0,
    )
    print("evaluating complete cycle on fresh banks", flush=True)
    cycle, artifacts = evaluate_complete_lifted_cycle(
        model=model,
        base_law_policy=base_policy,
        accepted_law_policy=moved_policy,
        pre_refit_adjoint_policy=pre_refit_policy,
        post_refit_adjoint_policy=post_refit_policy,
        data_context=context,
        fit_bank=audit_fit,
        confirmation_bank=audit_confirmation,
        full_training=model.training,
        query_batch_size=int(args.query_batch_size),
        merit=merit,
    )
    spectral = json.loads(
        (args.spectral_run_dir.resolve() / "report.json").read_text(encoding="utf-8")
    )
    report = {
        "protocol": {
            "single_complete_cycle": True,
            "relaxation": float(args.relaxation),
            "relaxation_rule": "locked conservative fraction below inverse measured spectral radius",
            "unchanged_P_network": True,
            "direct_R_ridge_refit": True,
            "test_paths_used": False,
        },
        "spectral_context": {
            "mean_radius": spectral["replicate_spectral_radius"]["mean"],
            "mean_matrix_spectrum": spectral["mean_spectrum"],
        },
        "scenario_banks": {
            "refit_fit": refit_fit.fingerprint,
            "refit_validation": refit_validation.fingerprint,
            "audit_fit": audit_fit.fingerprint,
            "audit_confirmation": audit_confirmation.fingerprint,
        },
        "refit_diagnostics": refit_diagnostics,
        "cycle": cycle,
    }
    write_json(run_dir / "report.json", report)
    torch.save(refreshed_projection, run_dir / "refreshed_residual_aware_ridge.pt")
    torch.save(artifacts, run_dir / "cycle_artifacts.pt")
    fit_improvement = cycle["fit"]["complete_cycle"]["merit_relative_improvement"]
    confirmation_improvement = cycle["confirmation"]["complete_cycle"][
        "merit_relative_improvement"
    ]
    (run_dir / "SUMMARY.md").write_text(
        "# Direct-ridge R complete cycle\n\n"
        f"- Relaxation: `{float(args.relaxation):.8f}`\n"
        f"- Measured spectral radius: `{spectral['replicate_spectral_radius']['mean']:.6g}`\n"
        f"- Fit merit improvement: `{fit_improvement:.3%}`\n"
        f"- Confirmation merit improvement: `{confirmation_improvement:.3%}`\n"
        f"- Complete-cycle success: `{cycle['cycle_success']}`\n",
        encoding="utf-8",
    )
    print(json.dumps(cycle, indent=2), flush=True)


if __name__ == "__main__":
    main()
