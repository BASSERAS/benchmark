from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from deep_mkv_gen_path_dt.networks import AdjointMomentNetwork  # noqa: E402
from path_dt_experiments.frozen_sparse_teacher import (  # noqa: E402
    FrozenSparseTeacherConfig,
    audit_frozen_sparse_teacher_learnability,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402


DEFAULT_BALANCE_RUN = (
    REPO_ROOT
    / "runs"
    / "heston_mixture_objective_balance_audit_seed81031_20260731"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "runs"
    / "heston_mixture_deep_mkv_ablation_seed0_20260730"
    / "summary_online_uncapped_causal_ce"
    / "model_checkpoint.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the Heston-mixture generator and branch-averaged P/R "
            "teacher, then optimize only the unchanged adjoint network and "
            "raw regression loss."
        )
    )
    parser.add_argument(
        "--balance-artifacts",
        type=Path,
        default=DEFAULT_BALANCE_RUN / "objective_balance_artifacts.pt",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument(
        "--dtype",
        choices=("checkpoint", "float32", "float64"),
        default="checkpoint",
    )
    parser.add_argument(
        "--step-indices",
        type=int,
        nargs="+",
        default=(32, 64, 96),
    )
    parser.add_argument("--fit-prefixes", type=int, default=48)
    parser.add_argument(
        "--heldout-prefixes",
        type=int,
        default=None,
        help=(
            "Use the final N entries of the seeded prefix permutation as a "
            "fixed holdout; omitted means every non-fit prefix is held out."
        ),
    )
    parser.add_argument(
        "--validation-prefixes",
        type=int,
        default=0,
        help=(
            "Use N prefixes immediately before the fixed holdout for "
            "checkpoint selection."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=5_000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--plateau-patience", type=int, default=10)
    parser.add_argument(
        "--minimum-relative-improvement",
        type=float,
        default=1e-4,
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=91_031)
    return parser.parse_args()


def default_run_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return (
        REPO_ROOT
        / "runs"
        / f"heston_mixture_frozen_sparse_teacher_{timestamp}"
    )


def _load_mapping(
    payload: dict[str, object],
    *,
    steps: tuple[int, ...],
    suffix: str,
) -> dict[int, torch.Tensor]:
    result = {}
    for step in steps:
        key = f"step_{step}_{suffix}"
        value = payload.get(key)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{key} must be a tensor in the balance artifacts")
        result[step] = value
    return result


def _checkpoint_dtype(checkpoint: dict[str, object]) -> torch.dtype:
    state = checkpoint.get("network_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint must contain network_state_dict")
    tensor = next(
        (value for value in state.values() if isinstance(value, torch.Tensor)),
        None,
    )
    if tensor is None:
        raise ValueError("network_state_dict must contain tensors")
    return tensor.dtype


def _build_network(
    checkpoint: dict[str, object],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> AdjointMomentNetwork:
    architecture = checkpoint.get("architecture")
    state = checkpoint.get("network_state_dict")
    if not isinstance(architecture, dict) or not isinstance(state, dict):
        raise ValueError(
            "checkpoint must contain architecture and network_state_dict"
        )
    network = AdjointMomentNetwork(
        state_dim=int(architecture["state_dim"]),
        adjoint_dim=(
            None
            if architecture.get("adjoint_dim") is None
            else int(architecture["adjoint_dim"])
        ),
        hidden_dim=int(architecture["hidden_dim"]),
        num_layers=int(architecture["num_layers"]),
        noise_adjoint_dim=(
            None
            if architecture.get("noise_adjoint_dim") is None
            else int(architecture["noise_adjoint_dim"])
        ),
    )
    network.load_state_dict(state)
    return network.to(device=device, dtype=dtype)


def _assert_raw_loss_checkpoint(checkpoint: dict[str, object]) -> None:
    training = checkpoint.get("training")
    if not isinstance(training, dict):
        raise ValueError("checkpoint must contain training metadata")
    if training.get("target_preconditioner") != "none":
        raise ValueError("diagnostic requires the original raw P/R loss")
    if training.get("noise_target_control_variate") != "none":
        raise ValueError("diagnostic requires the original raw R target")
    for key in (
        "adjoint_target_mean",
        "adjoint_target_scale",
        "noise_adjoint_target_mean",
        "noise_adjoint_target_scale",
        "noise_target_timewise_baseline",
    ):
        if checkpoint.get(key) is not None:
            raise ValueError(f"{key} must be absent for a raw-loss checkpoint")


def _metric(
    evaluation: dict[str, object],
    split: str,
    axis: str,
    name: str,
) -> float | None:
    value = evaluation[split][axis][name]
    return None if value is None else float(value)


def compact_report(report: dict[str, object]) -> dict[str, object]:
    initial = report["initial_evaluation"]
    best = report["best_evaluation"]
    rows = []
    for label, evaluation in (("initial", initial), ("best", best)):
        row: dict[str, object] = {
            "evaluation": label,
            "optimizer_step": int(evaluation["step"]),
            "raw_training_loss": float(
                evaluation["training_loss"]["total"]
            ),
        }
        for split in ("fit", "heldout"):
            for axis in ("p", "r"):
                stem = f"{split}_{axis}"
                row[f"{stem}_relative_rmse"] = _metric(
                    evaluation,
                    split,
                    axis,
                    "relative_rmse",
                )
                row[f"{stem}_r2"] = _metric(
                    evaluation,
                    split,
                    axis,
                    "r2_against_fit_timewise_mean",
                )
                row[f"{stem}_correlation"] = _metric(
                    evaluation,
                    split,
                    axis,
                    "correlation",
                )
                row[f"{stem}_prediction_target_rms_ratio"] = _metric(
                    evaluation,
                    split,
                    axis,
                    "prediction_target_rms_ratio",
                )
        rows.append(row)
    by_step = {}
    for step, values in best["by_step"].items():
        by_step[step] = {
            split: {
                axis: {
                    name: (
                        None
                        if values[split][axis][name] is None
                        else float(values[split][axis][name])
                    )
                    for name in (
                        "relative_rmse",
                        "r2_against_fit_timewise_mean",
                        "correlation",
                        "prediction_target_rms_ratio",
                    )
                }
                for axis in ("p", "r")
            }
            for split in ("fit", "heldout")
        }
    return {
        "rows": rows,
        "best_by_step": by_step,
        "teacher_reliability": report["teacher_reliability"],
        "optimization": report["optimization"],
        "decision": report["decision"],
    }


def _plot(report: dict[str, object], path: Path) -> None:
    import matplotlib.pyplot as plt

    trajectory = list(report["trajectory"])
    steps = [int(row["step"]) for row in trajectory]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(
        steps,
        [float(row["training_loss"]["total"]) for row in trajectory],
        color="black",
    )
    axes[0].set_yscale("log")
    axes[0].set_title("Fixed-teacher raw P/R loss")
    axes[0].set_xlabel("optimizer step")
    for axis_name, axis in (("p", axes[1]), ("r", axes[2])):
        splits = [("fit", "-")]
        if "validation" in trajectory[0]:
            splits.append(("validation", ":"))
        splits.append(("heldout", "--"))
        for split, linestyle in splits:
            axis.plot(
                steps,
                [
                    float(row[split][axis_name]["relative_rmse"])
                    for row in trajectory
                ],
                linestyle=linestyle,
                label=split,
            )
        axis.set_title(f"{axis_name.upper()} relative RMSE")
        axis.set_xlabel("optimizer step")
        axis.legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    balance_path = Path(args.balance_artifacts)
    checkpoint_path = Path(args.checkpoint)
    if not balance_path.exists():
        raise FileNotFoundError(balance_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    run_dir = ensure_run_dir(
        Path(args.run_dir) if args.run_dir is not None else default_run_dir()
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    artifacts = torch.load(
        balance_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict) or not isinstance(artifacts, dict):
        raise ValueError("checkpoint and balance artifacts must be mappings")
    _assert_raw_loss_checkpoint(checkpoint)
    checkpoint_training = checkpoint["training"]
    dtype = (
        _checkpoint_dtype(checkpoint)
        if args.dtype == "checkpoint"
        else torch.float64
        if args.dtype == "float64"
        else torch.float32
    )
    device = torch.device(args.device)
    network = _build_network(
        checkpoint,
        device=device,
        dtype=dtype,
    )
    steps = tuple(int(value) for value in args.step_indices)
    config = FrozenSparseTeacherConfig(
        step_indices=steps,
        fit_prefixes=int(args.fit_prefixes),
        heldout_prefixes=(
            None
            if args.heldout_prefixes is None
            else int(args.heldout_prefixes)
        ),
        validation_prefixes=int(args.validation_prefixes),
        batch_size=int(args.batch_size),
        max_steps=int(args.max_steps),
        eval_every=int(args.eval_every),
        plateau_patience=int(args.plateau_patience),
        minimum_relative_improvement=float(
            args.minimum_relative_improvement
        ),
        learning_rate=(
            float(args.lr)
            if args.lr is not None
            else float(checkpoint_training["lr"])
        ),
        weight_decay=(
            float(args.weight_decay)
            if args.weight_decay is not None
            else float(checkpoint_training["weight_decay"])
        ),
        p_weight=float(checkpoint_training["adjoint_weight"]),
        r_weight=float(checkpoint_training["adjoint_noise_weight"]),
        seed=int(args.seed),
    )

    def log(row: dict[str, object]) -> None:
        validation = row.get("validation")
        validation_text = (
            ""
            if validation is None
            else (
                " validation(P/R)="
                f"{float(validation['p']['relative_rmse']):.4f}/"
                f"{float(validation['r']['relative_rmse']):.4f}"
            )
        )
        print(
            f"step={int(row['step']):5d} "
            f"loss={float(row['training_loss']['total']):.6g} "
            f"fit(P/R)={float(row['fit']['p']['relative_rmse']):.4f}/"
            f"{float(row['fit']['r']['relative_rmse']):.4f} "
            f"heldout(P/R)="
            f"{float(row['heldout']['p']['relative_rmse']):.4f}/"
            f"{float(row['heldout']['r']['relative_rmse']):.4f}"
            f"{validation_text}",
            flush=True,
        )

    result = audit_frozen_sparse_teacher_learnability(
        network=network,
        prefixes=_load_mapping(
            artifacts,
            steps=steps,
            suffix="prefix_paths",
        ),
        teacher_p=_load_mapping(
            artifacts,
            steps=steps,
            suffix="full_p_conditional_mean",
        ),
        teacher_r=_load_mapping(
            artifacts,
            steps=steps,
            suffix="full_r_conditional_mean",
        ),
        samples_p=_load_mapping(
            artifacts,
            steps=steps,
            suffix="full_p_samples",
        ),
        samples_r=_load_mapping(
            artifacts,
            steps=steps,
            suffix="full_r_samples",
        ),
        config=config,
        log_callback=log,
    )
    compact = compact_report(result.report)
    payload = {
        "protocol": result.report["protocol"],
        "interpretation_scope": {
            "price_paths_only": True,
            "heston_parameters_or_regime_labels_used": False,
            "latent_variance_used": False,
            "forward_generator_rerun_or_updated": False,
            "conditional_teacher_reestimated_during_training": False,
            "only_adjoint_network_parameters_optimized": True,
            "trained_network_is_not_a_candidate_generator_checkpoint": True,
        },
        "sources": {
            "checkpoint": str(checkpoint_path.resolve()),
            "balance_artifacts": str(balance_path.resolve()),
        },
        "compact": compact,
        "report": result.report,
    }
    report_path = run_dir / "frozen_sparse_teacher_report.json"
    artifact_path = run_dir / "frozen_sparse_teacher_artifacts.pt"
    write_json(report_path, payload)
    torch.save(result.artifacts, artifact_path)
    _plot(result.report, run_dir / "frozen_sparse_teacher_trajectory.png")
    print(json.dumps(compact, indent=2), flush=True)
    print(f"saved report to {report_path.resolve()}", flush=True)
    print(f"saved artifacts to {artifact_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
