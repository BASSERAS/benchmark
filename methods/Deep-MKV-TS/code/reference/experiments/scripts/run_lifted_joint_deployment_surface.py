#!/usr/bin/env python3
"""No-update joint law/adjoint deployment diagnostic after the poll cycle."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from path_dt_experiments.lifted_collocation_solver import (  # noqa: E402
    AdjointNetworkControlPolicy,
    BlockwiseControlPerturbationPolicy,
    ScenarioDataContext,
    build_fit_confirmation_banks,
    model_checkpoint_sha256,
    network_sha256,
    rollout_policy_on_bank,
)
from path_dt_experiments.lifted_poll_solver import (  # noqa: E402
    FrozenLiftedMerit,
    interpolate_adjoint_outputs_on_evaluation,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_terminal_fixed_point_diagnostic import (  # noqa: E402
    DEFAULT_CONTINUATION,
    _build_frozen_model,
)


DEFAULT_POLL_CYCLE = (
    REPO_ROOT / "runs" / "lifted_poll_trust_region_cycle_seed0_20260803"
)
TAU_GRID = (0.0, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
C_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation-run-dir", type=Path, default=DEFAULT_CONTINUATION)
    parser.add_argument("--poll-cycle-run-dir", type=Path, default=DEFAULT_POLL_CYCLE)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2_608_111)
    parser.add_argument("--paths", type=int, default=64)
    parser.add_argument("--target-paths", type=int, default=256)
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=4_096)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _network_from_state(
    *, model, state: dict[str, torch.Tensor]
) -> torch.nn.Module:
    network = copy.deepcopy(model.network)
    network.load_state_dict(state)
    network.to(device=model.device, dtype=model.dtype)
    for module in network.modules():
        if isinstance(module, torch.nn.RNNBase):
            module.flatten_parameters()
    network.eval()
    return network


def _relative_improvement(before: float, after: float) -> float:
    scale = max(abs(float(before)), torch.finfo(torch.float64).eps)
    return (float(before) - float(after)) / scale


def _annotate_rows(
    *,
    rows: list[dict[str, object]],
    original_merit: float,
    stale_law_merit: float,
) -> None:
    for row in rows:
        value = float(row["merit"]["total"])
        row["merit_improvement_vs_original_pair"] = _relative_improvement(
            original_merit, value
        )
        row["merit_improvement_vs_stale_law_pair"] = _relative_improvement(
            stale_law_merit, value
        )


def _summary_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Joint law--adjoint deployment surface",
        "",
        "No candidate was deployed and no checkpoint was modified.",
        "",
        f"- Accepted law coordinate: `{report['accepted_coordinate']}`",
        f"- Accepted law coefficient: `{report['accepted_coefficient']:.6g}`",
        f"- Nonzero paired point remains better than original on both banks: `{report['tau_line']['nonzero_tau_beats_original_on_both_banks']}`",
        f"- Nonzero tau improves the accepted stale-law pair on both banks: `{report['tau_line']['nonzero_tau_improves_stale_pair_on_both_banks']}`",
        f"- Two-dimensional surface evaluated: `{report['joint_surface']['evaluated']}`",
        "",
        "## Fixed accepted-law tau line",
        "",
        "| Tau | Fit merit | Fit vs original | Fit vs stale law | Confirmation merit | Confirmation vs original | Confirmation vs stale law |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    fit_rows = {float(row["tau"]): row for row in report["tau_line"]["fit"]}
    confirmation_rows = {
        float(row["tau"]): row for row in report["tau_line"]["confirmation"]
    }
    for tau in TAU_GRID:
        fit = fit_rows[float(tau)]
        confirmation = confirmation_rows[float(tau)]
        lines.append(
            f"| {tau:.6g} | {fit['merit']['total']:.6g} | "
            f"{fit['merit_improvement_vs_original_pair']:.2%} | "
            f"{fit['merit_improvement_vs_stale_law_pair']:.2%} | "
            f"{confirmation['merit']['total']:.6g} | "
            f"{confirmation['merit_improvement_vs_original_pair']:.2%} | "
            f"{confirmation['merit_improvement_vs_stale_law_pair']:.2%} |"
        )
    joint = report["joint_surface"]
    if bool(joint["evaluated"]):
        lines.extend(
            [
                "",
                "## Conditional two-dimensional surface",
                "",
                f"- Fit-selected point: `{joint['fit_selected_point']}`",
                f"- Fit-selected point confirms: `{joint['fit_selected_point_confirms']}`",
                f"- Any jointly improving point: `{joint['jointly_improving_point_exists']}`",
                f"- Improvement limited to microscopic tau: `{joint['only_microscopic_tau_improves']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if int(args.paths) < 1 or int(args.target_paths) < 1:
        raise ValueError("paths and target-paths must be positive")
    if int(args.branches) < 2 or int(args.branches) % 2 != 0:
        raise ValueError("branches must be an even integer of at least two")
    if int(args.query_batch_size) < int(args.paths) * int(args.branches):
        raise ValueError("query-batch-size must hold one complete branch population")

    continuation = args.continuation_run_dir.resolve()
    poll_dir = args.poll_cycle_run_dir.resolve()
    run_dir = ensure_run_dir(args.run_dir.resolve())
    poll_report = _load_json(poll_dir / "poll_cycle_report.json")
    accepted_payload = torch.load(
        poll_dir / "accepted_law.pt", map_location="cpu", weights_only=True
    )
    refit_payload = torch.load(
        poll_dir / "refitted_adjoint.pt", map_location="cpu", weights_only=True
    )
    if not isinstance(accepted_payload, dict) or not isinstance(refit_payload, dict):
        raise ValueError("poll-cycle checkpoints must contain dictionaries")
    accepted = accepted_payload.get("accepted_candidate")
    if not isinstance(accepted, dict):
        raise ValueError("accepted law is missing its coordinate descriptor")

    model, target_pool, checkpoint_training, continuation_report, design = (
        _build_frozen_model(
            continuation=continuation,
            device=torch.device(args.device),
            model_seed=int(args.model_seed),
        )
    )
    stages = continuation_report.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("continuation report has no stages")
    terminal = stages[-1]
    operator_beta = float(terminal["beta_target"])
    full_lambda_scale = float(continuation_report["final_lambda_scale"])
    full_training = replace(
        checkpoint_training,
        lambda_scale=full_lambda_scale,
        grad_clip_norm=None,
        target_preconditioner="none",
        noise_target_control_variate="adjoint",
    )
    dataset = design.get("dataset")
    context = ScenarioDataContext(
        initial_state_pool=target_pool.detach().cpu(),
        target_path_pool=target_pool.detach().cpu(),
        dataset_id=(
            json.dumps(dataset, sort_keys=True)
            if isinstance(dataset, dict)
            else str(dataset)
        ),
    )
    base_state = accepted_payload.get("base_law_network_state_dict")
    old_state = accepted_payload.get("proposal_adjoint_network_state_dict")
    new_state = refit_payload.get("state_dict")
    if not all(isinstance(value, dict) for value in (base_state, old_state, new_state)):
        raise ValueError("joint surface requires all three network state dictionaries")
    base_network = _network_from_state(model=model, state=base_state)
    old_adjoint_network = _network_from_state(model=model, state=old_state)
    refitted_network = _network_from_state(model=model, state=new_state)
    base_law_policy = AdjointNetworkControlPolicy(model=model, network=base_network)
    old_adjoint_policy = AdjointNetworkControlPolicy(
        model=model, network=old_adjoint_network
    )
    refitted_adjoint_policy = AdjointNetworkControlPolicy(
        model=model, network=refitted_network
    )
    blocks = tuple(tuple(int(value) for value in row) for row in accepted["blocks"])
    accepted_coefficient = float(accepted["coefficient"])
    alpha_coefficients = tuple(float(value) for value in accepted["alpha_coefficients"])
    log_sigma_coefficients = tuple(
        float(value) for value in accepted["log_sigma_coefficients"]
    )
    accepted_law_policy = BlockwiseControlPerturbationPolicy(
        model=model,
        law_policy=base_law_policy,
        adjoint_policy=old_adjoint_policy,
        blocks=blocks,
        alpha_coefficients=alpha_coefficients,
        log_sigma_coefficients=log_sigma_coefficients,
    )
    merit_payload = poll_report["poll"]["merit"]
    merit = FrozenLiftedMerit(
        denominators={
            key: float(value) for key, value in merit_payload["denominators"].items()
        },
        weights={key: float(value) for key, value in merit_payload["weights"].items()},
        baseline_component_values={
            key: float(value)
            for key, value in merit_payload["baseline_component_values"].items()
        },
        estimator_noise_variances={
            key: float(value)
            for key, value in merit_payload["estimator_noise_variances"].items()
        },
    )
    model_hash = model_checkpoint_sha256(model)
    network_hashes = {
        "base": network_sha256(base_network),
        "old": network_sha256(old_adjoint_network),
        "refit": network_sha256(refitted_network),
    }
    fit_bank, confirmation_bank = build_fit_confirmation_banks(
        model=model,
        context=context,
        path_count=int(args.paths),
        target_count=int(args.target_paths),
        num_branches=int(args.branches),
        antithetic=True,
        beta=operator_beta,
        seed=int(args.seed),
    )

    evaluations: dict[str, dict[float, object]] = {"fit": {}, "confirmation": {}}
    for bank_name, bank in (("fit", fit_bank), ("confirmation", confirmation_bank)):
        print(f"building {bank_name} targets at c=0", flush=True)
        evaluations[bank_name][0.0] = rollout_policy_on_bank(
            model=model,
            policy=base_law_policy,
            bank=bank,
            data_context=context,
            full_training=full_training,
            query_batch_size=int(args.query_batch_size),
        )
        print(f"building {bank_name} targets at accepted c", flush=True)
        evaluations[bank_name][accepted_coefficient] = rollout_policy_on_bank(
            model=model,
            policy=accepted_law_policy,
            bank=bank,
            data_context=context,
            full_training=full_training,
            query_batch_size=int(args.query_batch_size),
        )

    artifacts: dict[str, object] = {"tau_line": {"fit": {}, "confirmation": {}}}
    original_rows = {}
    tau_rows = {"fit": [], "confirmation": []}
    for bank_name in ("fit", "confirmation"):
        original = interpolate_adjoint_outputs_on_evaluation(
            model=model,
            evaluation=evaluations[bank_name][0.0],
            old_adjoint_policy=old_adjoint_policy,
            refitted_adjoint_policy=refitted_adjoint_policy,
            tau=0.0,
            merit=merit,
        )
        original_rows[bank_name] = original
        accepted_old = interpolate_adjoint_outputs_on_evaluation(
            model=model,
            evaluation=evaluations[bank_name][accepted_coefficient],
            old_adjoint_policy=old_adjoint_policy,
            refitted_adjoint_policy=refitted_adjoint_policy,
            tau=0.0,
            merit=merit,
        )
        for tau in TAU_GRID:
            result = interpolate_adjoint_outputs_on_evaluation(
                model=model,
                evaluation=evaluations[bank_name][accepted_coefficient],
                old_adjoint_policy=old_adjoint_policy,
                refitted_adjoint_policy=refitted_adjoint_policy,
                tau=float(tau),
                merit=merit,
            )
            tau_rows[bank_name].append(result.report)
            artifacts["tau_line"][bank_name][f"{tau:.10g}"] = result.artifacts
        _annotate_rows(
            rows=tau_rows[bank_name],
            original_merit=float(original.report["merit"]["total"]),
            stale_law_merit=float(accepted_old.report["merit"]["total"]),
        )
    original_improving_taus = [
        float(fit["tau"])
        for fit, confirmation in zip(
            tau_rows["fit"], tau_rows["confirmation"], strict=True
        )
        if float(fit["tau"]) > 0.0
        and float(fit["merit_improvement_vs_original_pair"]) > 0.0
        and float(confirmation["merit_improvement_vs_original_pair"]) > 0.0
    ]
    stale_pair_improving_taus = [
        float(fit["tau"])
        for fit, confirmation in zip(
            tau_rows["fit"], tau_rows["confirmation"], strict=True
        )
        if float(fit["tau"]) > 0.0
        and float(fit["merit_improvement_vs_stale_law_pair"]) > 0.0
        and float(confirmation["merit_improvement_vs_stale_law_pair"]) > 0.0
    ]
    tau_line_report = {
        "fit": tau_rows["fit"],
        "confirmation": tau_rows["confirmation"],
        "nonzero_taus_beating_original_on_both_banks": original_improving_taus,
        "nonzero_tau_beats_original_on_both_banks": bool(
            original_improving_taus
        ),
        "nonzero_taus_improving_stale_pair_on_both_banks": (
            stale_pair_improving_taus
        ),
        "nonzero_tau_improves_stale_pair_on_both_banks": bool(
            stale_pair_improving_taus
        ),
        "targets_recomputed_per_tau": False,
    }

    joint_report: dict[str, object] = {"evaluated": False}
    if original_improving_taus:
        c_grid = tuple(accepted_coefficient * fraction for fraction in C_FRACTIONS)
        for c in c_grid[1:-1]:
            alpha = [0.0] * len(blocks)
            log_sigma = [0.0] * len(blocks)
            values = alpha if str(accepted["signal"]) == "alpha" else log_sigma
            values[int(accepted["block_index"])] = float(c)
            policy = BlockwiseControlPerturbationPolicy(
                model=model,
                law_policy=base_law_policy,
                adjoint_policy=old_adjoint_policy,
                blocks=blocks,
                alpha_coefficients=tuple(alpha),
                log_sigma_coefficients=tuple(log_sigma),
            )
            for bank_name, bank in (("fit", fit_bank), ("confirmation", confirmation_bank)):
                print(f"building {bank_name} targets at c={c:.6g}", flush=True)
                evaluations[bank_name][float(c)] = rollout_policy_on_bank(
                    model=model,
                    policy=policy,
                    bank=bank,
                    data_context=context,
                    full_training=full_training,
                    query_batch_size=int(args.query_batch_size),
                )
        surface_rows = {"fit": [], "confirmation": []}
        surface_artifacts: dict[str, object] = {"fit": {}, "confirmation": {}}
        for bank_name in ("fit", "confirmation"):
            original_merit = float(original_rows[bank_name].report["merit"]["total"])
            for c in c_grid:
                surface_artifacts[bank_name][f"{c:.10g}"] = {}
                for tau in TAU_GRID:
                    result = interpolate_adjoint_outputs_on_evaluation(
                        model=model,
                        evaluation=evaluations[bank_name][float(c)],
                        old_adjoint_policy=old_adjoint_policy,
                        refitted_adjoint_policy=refitted_adjoint_policy,
                        tau=float(tau),
                        merit=merit,
                    )
                    row = dict(result.report)
                    row["c"] = float(c)
                    row["merit_improvement_vs_original_pair"] = _relative_improvement(
                        original_merit, float(row["merit"]["total"])
                    )
                    surface_rows[bank_name].append(row)
                    surface_artifacts[bank_name][f"{c:.10g}"][f"{tau:.10g}"] = (
                        result.artifacts
                    )
            rows_by_c = {}
            for row in surface_rows[bank_name]:
                rows_by_c.setdefault(float(row["c"]), []).append(row)
            for c, rows in rows_by_c.items():
                tau_zero = next(row for row in rows if float(row["tau"]) == 0.0)
                stale_merit = float(tau_zero["merit"]["total"])
                for row in rows:
                    row["merit_improvement_vs_same_law_tau_zero"] = (
                        _relative_improvement(
                            stale_merit, float(row["merit"]["total"])
                        )
                    )
        fit_candidates = [
            row
            for row in surface_rows["fit"]
            if float(row["c"]) > 0.0
            and float(row["tau"]) > 0.0
            and float(row["merit_improvement_vs_original_pair"]) > 0.0
        ]
        fit_selected = (
            None
            if not fit_candidates
            else min(fit_candidates, key=lambda row: float(row["merit"]["total"]))
        )
        confirmation_by_point = {
            (float(row["c"]), float(row["tau"])): row
            for row in surface_rows["confirmation"]
        }
        selected_confirmation = (
            None
            if fit_selected is None
            else confirmation_by_point[
                (float(fit_selected["c"]), float(fit_selected["tau"]))
            ]
        )
        jointly_improving = [
            {
                "c": float(row["c"]),
                "tau": float(row["tau"]),
                "fit_improvement": float(row["merit_improvement_vs_original_pair"]),
                "confirmation_improvement": float(
                    confirmation_by_point[(float(row["c"]), float(row["tau"]))][
                        "merit_improvement_vs_original_pair"
                    ]
                ),
            }
            for row in fit_candidates
            if float(
                confirmation_by_point[(float(row["c"]), float(row["tau"]))][
                    "merit_improvement_vs_original_pair"
                ]
            )
            > 0.0
        ]
        same_law_deployment_improving = [
            {
                "c": float(row["c"]),
                "tau": float(row["tau"]),
                "fit_improvement": float(
                    row["merit_improvement_vs_same_law_tau_zero"]
                ),
                "confirmation_improvement": float(
                    confirmation_by_point[(float(row["c"]), float(row["tau"]))][
                        "merit_improvement_vs_same_law_tau_zero"
                    ]
                ),
            }
            for row in fit_candidates
            if float(row["merit_improvement_vs_same_law_tau_zero"]) > 0.0
            and float(
                confirmation_by_point[(float(row["c"]), float(row["tau"]))][
                    "merit_improvement_vs_same_law_tau_zero"
                ]
            )
            > 0.0
        ]
        fit_surface_optimum = min(
            surface_rows["fit"], key=lambda row: float(row["merit"]["total"])
        )
        selected_tau = None if fit_selected is None else float(fit_selected["tau"])
        joint_report = {
            "evaluated": True,
            "c_grid": list(c_grid),
            "tau_grid": list(TAU_GRID),
            "fit": surface_rows["fit"],
            "confirmation": surface_rows["confirmation"],
            "fit_selected_point": (
                None
                if fit_selected is None
                else {"c": fit_selected["c"], "tau": fit_selected["tau"]}
            ),
            "fit_selected_point_confirms": bool(
                selected_confirmation is not None
                and float(selected_confirmation["merit_improvement_vs_original_pair"])
                > 0.0
            ),
            "jointly_improving_points": jointly_improving,
            "jointly_improving_point_exists": bool(jointly_improving),
            "same_law_adjoint_deployment_improving_points": (
                same_law_deployment_improving
            ),
            "same_law_adjoint_deployment_improves_both_banks": bool(
                same_law_deployment_improving
            ),
            "fit_surface_optimum": {
                "c": fit_surface_optimum["c"],
                "tau": fit_surface_optimum["tau"],
            },
            "fit_surface_optimum_on_tau_zero_boundary": bool(
                float(fit_surface_optimum["tau"]) == 0.0
            ),
            "fit_selected_nonzero_tau_at_grid_floor": bool(
                selected_tau is not None and selected_tau == min(TAU_GRID[1:])
            ),
            "only_microscopic_tau_improves": bool(
                jointly_improving
                and not same_law_deployment_improving
                and selected_tau is not None
                and selected_tau <= 1 / 32
            ),
            "targets_recomputed_per_surface_point": False,
            "targets_recomputed_once_per_c_and_bank": True,
        }
        artifacts["joint_surface"] = surface_artifacts

    if model_checkpoint_sha256(model) != model_hash:
        raise RuntimeError("joint deployment diagnostic mutated the source model")
    if network_sha256(base_network) != network_hashes["base"]:
        raise RuntimeError("joint deployment diagnostic mutated the base law")
    if network_sha256(old_adjoint_network) != network_hashes["old"]:
        raise RuntimeError("joint deployment diagnostic mutated the old adjoint")
    if network_sha256(refitted_network) != network_hashes["refit"]:
        raise RuntimeError("joint deployment diagnostic mutated the refitted adjoint")
    report = {
        "protocol": {
            "diagnostic_only": True,
            "candidate_deployed": False,
            "denormalized_P_R_outputs_interpolated": True,
            "network_parameters_interpolated": False,
            "law_fixed_along_each_tau_line": True,
            "conditional_targets_computed_once_per_law_and_bank": True,
            "fit_bank_selects_surface_point": True,
            "confirmation_bank_does_not_select": True,
            "test_paths_used": False,
        },
        "poll_cycle_run_dir": str(poll_dir),
        "operator_beta": operator_beta,
        "effective_lambda_scale": operator_beta * full_lambda_scale,
        "accepted_coordinate": accepted["coordinate"],
        "accepted_coefficient": accepted_coefficient,
        "tau_grid": list(TAU_GRID),
        "tau_line": tau_line_report,
        "joint_surface": joint_report,
        "fit_bank_fingerprint": fit_bank.fingerprint,
        "confirmation_bank_fingerprint": confirmation_bank.fingerprint,
        "nonmutation": {
            "source_model_unchanged": True,
            "base_law_unchanged": True,
            "old_adjoint_unchanged": True,
            "refitted_adjoint_unchanged": True,
        },
    }
    write_json(run_dir / "joint_deployment_report.json", report)
    torch.save(artifacts, run_dir / "joint_deployment_tensors.pt")
    (run_dir / "SUMMARY.md").write_text(
        _summary_markdown(report), encoding="utf-8"
    )
    print(f"wrote joint deployment surface to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
