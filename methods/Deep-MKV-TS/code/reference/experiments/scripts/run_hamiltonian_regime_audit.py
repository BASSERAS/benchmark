from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from deep_mkv_gen_path_dt.discrepancies import (  # noqa: E402
    CompositePathFunctionalDiscrepancy,
)
from deep_mkv_gen_path_dt.noise import sample_standard_normals  # noqa: E402
from path_dt_experiments.hamiltonian_regime import (  # noqa: E402
    REGIME_NAMES,
    audit_specific_entropy_hamiltonian_by_regime,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from run_specific_entropy_joint_shadow import _build_models  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit complete-path P/R pressure, specific-entropy resistance, "
            "and accepted controls by fixed prefix-volatility regime."
        )
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=Path("runs/heston_dt_specific_entropy_r_ablation_seed1234_20260721"),
    )
    parser.add_argument("--frontier-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--audit-paths", type=int, default=1024)
    parser.add_argument("--joint-weight", type=float, default=1.0)
    parser.add_argument("--split-step", type=int, default=64)
    parser.add_argument("--future-steps", type=int, default=32)
    parser.add_argument("--prefix-window", type=int, default=16)
    parser.add_argument("--prefix-windows", type=int, nargs="+", default=(16, 32, 64))
    parser.add_argument("--recent-return-window", type=int, default=16)
    parser.add_argument("--bandwidths", type=float, nargs="+", default=(0.5, 1.0, 2.0))
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--regime-weight", type=float, default=0.0)
    parser.add_argument("--regime-prefix-window", type=int, default=16)
    parser.add_argument("--regime-gate-temperature-fraction", type=float, default=0.25)
    parser.add_argument("--regime-tail-temperature", type=float, default=0.15)
    parser.add_argument("--regime-conditional-moment-weight", type=float, default=0.0)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _selected_training_row(
    frontier_dir: Path,
    *,
    selected_step: int,
) -> dict[str, float] | None:
    path = frontier_dir / "training_history.jsonl"
    if not path.exists():
        return None
    selected = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(round(float(row.get("step", -1)))) == int(selected_step):
                selected = {
                    str(name): float(value)
                    for name, value in row.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
    return selected


def _line_search_context(row: dict[str, float] | None) -> dict[str, object] | None:
    if row is None:
        return None
    names = (
        "outer_objective_before",
        "outer_candidate_objective",
        "outer_accepted_objective",
        "outer_discrepancy_objective_before",
        "outer_candidate_discrepancy_objective",
        "outer_accepted_discrepancy_objective",
        "outer_running_cost_before",
        "outer_candidate_running_cost",
        "outer_accepted_running_cost",
        "outer_selected_relaxation_p",
        "outer_selected_relaxation_r",
        "outer_selected_relaxation_shared",
        "outer_block_trust_fraction",
        "outer_block_search_evaluations",
        "outer_backtracks",
        "outer_objective_change",
    )
    result: dict[str, object] = {
        name: float(row[name]) for name in names if name in row
    }
    before = row.get("outer_objective_before")
    candidate = row.get("outer_candidate_objective")
    accepted = row.get("outer_accepted_objective")
    if before is not None and candidate is not None and accepted is not None:
        raw_change = float(candidate) - float(before)
        accepted_change = float(accepted) - float(before)
        result["raw_candidate_objective_change"] = raw_change
        result["accepted_objective_change"] = accepted_change
        result["raw_candidate_was_acceptable"] = bool(raw_change <= 0.0)
        result["accepted_to_raw_objective_change_ratio"] = (
            None if raw_change == 0.0 else accepted_change / raw_change
        )
    result["scope"] = (
        "global saved outer-step context; regime-resolved quantities below "
        "measure the accepted policy and frozen Hamiltonian target"
    )
    return result


def _isolated_discrepancies(
    discrepancy: CompositePathFunctionalDiscrepancy,
) -> dict[str, CompositePathFunctionalDiscrepancy]:
    blocks = tuple(discrepancy.blocks)
    joint_name = "volatility_law_joint_prefix_future_rv"
    joint = tuple(block for block in blocks if block.name == joint_name)
    base = tuple(block for block in blocks if block.name != joint_name)
    if len(joint) != 1 or len(base) == 0:
        raise ValueError("candidate discrepancy must contain one joint block and base blocks")
    return {
        "joint_prefix_future_discrepancy": CompositePathFunctionalDiscrepancy(
            blocks=joint
        )
    }


def _compact_rows(summary: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for name in REGIME_NAMES:
        regime = summary["regimes"][name]
        window = regime["future_control_window"]
        controls = window["controls"]
        accepted_alpha = controls.get("accepted_alpha_change_from_source")
        accepted_sigma = controls.get("accepted_sigma_change_from_source")
        rows.append(
            {
                "regime": name,
                "count": int(regime["count"]),
                "future_rv_std_ratio": float(
                    regime["future_law"]["generated_to_target_rv_std_ratio"]
                ),
                "future_rv_iqr_ratio": float(
                    regime["future_law"]["generated_to_target_rv_iqr_ratio"]
                ),
                "p_fitted_target_rms_ratio": float(
                    window["p"]["fitted_to_target_rms_ratio"]
                ),
                "r_fitted_target_rms_ratio": float(
                    window["r"]["fitted_to_target_rms_ratio"]
                ),
                "p_relative_rmse": float(window["p"]["relative_rmse"]),
                "r_relative_rmse": float(window["r"]["relative_rmse"]),
                "running_to_discrepancy_p_ratio": float(
                    window["p"]["running_to_discrepancy_rms_ratio"]
                ),
                "running_to_discrepancy_r_ratio": float(
                    window["r"]["running_to_discrepancy_rms_ratio"]
                ),
                "joint_r_to_full_ratio": float(
                    window["r"]["components"][
                        "joint_prefix_future_discrepancy"
                    ]["to_full_target_rms_ratio"]
                ),
                "sigma_signal_transmission_fraction": controls[
                    "sigma_signal_transmission_fraction"
                ],
                "sigma_oracle_change_rms": float(
                    controls["oracle_sigma_change_from_current"]["rms"]
                ),
                "accepted_alpha_change_from_source_rms": (
                    None if accepted_alpha is None else float(accepted_alpha["rms"])
                ),
                "accepted_sigma_change_from_source_rms": (
                    None if accepted_sigma is None else float(accepted_sigma["rms"])
                ),
                "sigma_running_source_effect_rms": float(
                    controls["running_source_effect_on_oracle_sigma"]["rms"]
                ),
                "current_sigma_ratio": float(
                    controls["sigma_to_reference_ratio"]["mean"]
                ),
                "oracle_sigma_ratio": float(
                    controls["oracle_sigma_to_reference_ratio"]["mean"]
                ),
                "current_sigma_cap_fraction": float(
                    controls["sigma_cap_fraction"]
                ),
                "oracle_sigma_cap_fraction": float(
                    controls["oracle_sigma_cap_fraction"]
                ),
                "eta_half_sigma_change_rms": float(
                    controls["target_eta_sensitivity"]["eta_x0.5"][
                        "sigma_change_from_eta_1"
                    ]["rms"]
                ),
                "eta_double_sigma_change_rms": float(
                    controls["target_eta_sensitivity"]["eta_x2"][
                        "sigma_change_from_eta_1"
                    ]["rms"]
                ),
                "fitted_eta_half_sigma_change_rms": float(
                    controls["fitted_policy_eta_sensitivity"]["eta_x0.5"][
                        "sigma_change_from_eta_1"
                    ]["rms"]
                ),
                "fitted_eta_double_sigma_change_rms": float(
                    controls["fitted_policy_eta_sensitivity"]["eta_x2"][
                        "sigma_change_from_eta_1"
                    ]["rms"]
                ),
                "hamiltonian_gap": float(
                    window["hamiltonian_balance"]["current_minus_oracle_gap"]
                ),
            }
        )
    return rows


def _plot(rows_by_law: dict[str, list[dict[str, object]]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    positions = range(len(REGIME_NAMES))
    for law, rows in rows_by_law.items():
        axes[0, 0].plot(
            positions,
            [float(row["future_rv_std_ratio"]) for row in rows],
            marker="o",
            label=law,
        )
        axes[0, 1].plot(
            positions,
            [float(row["r_fitted_target_rms_ratio"]) for row in rows],
            marker="o",
            label=law,
        )
        axes[1, 0].plot(
            positions,
            [float(row["sigma_oracle_change_rms"]) for row in rows],
            marker="o",
            label=law,
        )
        axes[1, 1].plot(
            positions,
            [
                float(row["sigma_signal_transmission_fraction"])
                if row["sigma_signal_transmission_fraction"] is not None
                else float("nan")
                for row in rows
            ],
            marker="o",
            label=law,
        )
    titles = (
        "Future RV std / target",
        "Fitted R RMS / projected target R RMS",
        "RMS sigma change to Hamiltonian target",
        "Sigma correction transmission fraction",
    )
    for axis, title in zip(axes.reshape(-1), titles):
        axis.set_xticks(list(positions), REGIME_NAMES)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[0, 0].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if int(args.audit_paths) < 6:
        raise ValueError("audit-paths must be >= 6")
    if int(args.prefix_window) < 1 or int(args.prefix_window) > int(args.split_step):
        raise ValueError("prefix-window must fit before split-step")
    run_dir = ensure_run_dir(Path(args.run_dir))
    frontier_dir = Path(args.frontier_run_dir)
    frontier_metrics = _read_json(frontier_dir / "metrics.json")
    selected_step = int(frontier_metrics["selected_step"])

    (
        source_model,
        audit_model,
        _source_training,
        _stored_config,
        _heston,
        target_paths,
        _heldout_paths,
        _joint,
        _mapped,
    ) = _build_models(
        source=Path(args.source_run_dir),
        device=torch.device(args.device),
        args=args,
    )
    if int(target_paths.shape[0]) < int(args.audit_paths):
        raise ValueError("source target bank is smaller than audit-paths")
    target = target_paths[: int(args.audit_paths)].to(
        device=audit_model.device, dtype=audit_model.dtype
    )
    x0 = target[:, 0, :]
    noise = sample_standard_normals(
        grid=audit_model.grid,
        batch_size=int(args.audit_paths),
        noise_dim=int(audit_model.architecture.noise_dim),
        device=audit_model.device,
        dtype=audit_model.dtype,
        seed=int(args.seed) + 910_000,
    )
    isolated = _isolated_discrepancies(audit_model.discrepancy)

    print("auditing source checkpoint on the frozen common-noise problem", flush=True)
    audit_model.load_checkpoint_state(source_model.checkpoint_state())
    source_result = audit_specific_entropy_hamiltonian_by_regime(
        model=audit_model,
        target_paths=target,
        x0=x0,
        noise=noise,
        training=audit_model.training,
        split_step=int(args.split_step),
        prefix_window=int(args.prefix_window),
        future_horizon=int(args.future_steps),
        isolated_discrepancies=isolated,
    )

    selected_checkpoint = torch.load(
        frontier_dir / "selected_model_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(selected_checkpoint, dict):
        raise ValueError("selected checkpoint must be a dictionary")
    print(
        f"auditing selected step {selected_step} against the same source rollout",
        flush=True,
    )
    audit_model.load_checkpoint_state(selected_checkpoint)
    selected_result = audit_specific_entropy_hamiltonian_by_regime(
        model=audit_model,
        target_paths=target,
        x0=x0,
        noise=noise,
        training=audit_model.training,
        split_step=int(args.split_step),
        prefix_window=int(args.prefix_window),
        future_horizon=int(args.future_steps),
        isolated_discrepancies=isolated,
        baseline_rollout=source_result.rollout,
    )

    rows_by_law = {
        "source": _compact_rows(source_result.summary),
        "selected": _compact_rows(selected_result.summary),
    }
    training_row = _selected_training_row(
        frontier_dir, selected_step=selected_step
    )
    payload: dict[str, object] = {
        "run_dir": str(run_dir),
        "source_run_dir": str(args.source_run_dir),
        "frontier_run_dir": str(frontier_dir),
        "selected_step": selected_step,
        "config": {
            **vars(args),
            "source_run_dir": str(args.source_run_dir),
            "frontier_run_dir": str(frontier_dir),
            "run_dir": str(run_dir),
            "source_training": asdict(source_model.training),
            "selected_training": asdict(audit_model.training),
        },
        "line_search_context_at_selected_outer_step": _line_search_context(
            training_row
        ),
        "compact_comparison": rows_by_law,
        "source": source_result.summary,
        "selected": selected_result.summary,
    }
    write_json(run_dir / "metrics.json", payload)
    torch.save(
        {
            "source": source_result.artifacts,
            "selected": selected_result.artifacts,
        },
        run_dir / "diagnostic_artifacts.pt",
    )
    _plot(rows_by_law, run_dir / "regime_hamiltonian_audit.png")

    for regime_index, regime_name in enumerate(REGIME_NAMES):
        source_row = rows_by_law["source"][regime_index]
        selected_row = rows_by_law["selected"][regime_index]
        print(f"regime={regime_name}", flush=True)
        for law, row in (("source", source_row), ("selected", selected_row)):
            print(
                "  {law}: rv_std_ratio={rv:.3f} fitted_R/target_R={r:.3f} "
                "joint_R/full_R={joint:.3f} sigma_tx={tx} "
                "oracle_dsigma_rms={delta:.5f} eta_half_dsigma={eta:.5f}".format(
                    law=law,
                    rv=float(row["future_rv_std_ratio"]),
                    r=float(row["r_fitted_target_rms_ratio"]),
                    joint=float(row["joint_r_to_full_ratio"]),
                    tx=(
                        "n/a"
                        if row["sigma_signal_transmission_fraction"] is None
                        else f"{float(row['sigma_signal_transmission_fraction']):.3f}"
                    ),
                    delta=float(row["sigma_oracle_change_rms"]),
                    eta=float(row["eta_half_sigma_change_rms"]),
                ),
                flush=True,
            )
    print(f"wrote Hamiltonian regime audit to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
