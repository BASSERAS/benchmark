#!/usr/bin/env python3
"""Evaluate a frozen one-stage MP law and a validation-selected law mixture."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.stats import ks_2samp, wasserstein_distance


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "scripts"))

from path_dt_experiments.direct_ridge_adjoint import (  # noqa: E402
    DirectRidgeRAdjointPolicy,
)
from path_dt_experiments.extragradient_solver import _sample_inputs  # noqa: E402
from path_dt_experiments.lifted_collocation_solver import (  # noqa: E402
    RolloutResult,
    rollout_causal_policy,
)
from path_dt_experiments.metrics import stylized_fact_metrics  # noqa: E402
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from path_dt_experiments.terminal_fixed_point_diagnostic import (  # noqa: E402
    _time_blocks,
)
from path_dt_experiments.volatility_only_solver import (  # noqa: E402
    VolatilityOnlyBlockwiseRPolicy,
)
from evaluate_drawdown_memory import (  # noqa: E402
    drawdown_features,
    excess_kurtosis,
    pooled_acf,
    summarize_memory,
)
from run_direct_ridge_r_spectral_matrix import (  # noqa: E402
    DEFAULT_CHECKPOINT_ROOT,
)
from run_residual_volatility_first_operator_audit import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_SIGNAL_RUN,
)
from run_residual_volatility_reduced_spectral_audit import (  # noqa: E402
    _construct_model,
)
from run_volatility_only_first_operator_audit import (  # noqa: E402
    _restrict_to_volatility,
)
from run_volatility_only_r_spectral_matrix import (  # noqa: E402
    DEFAULT_FIRST_OPERATOR_RUN,
)


DEFAULT_NEWTON_RUN = (
    REPO_ROOT / "runs" / "volatility_only_residual_newton_20260803"
)
DEFAULT_BENCHMARK_METRICS = Path("/home/tbasseras/benchmark/metrics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--newton-run-dir", type=Path, default=DEFAULT_NEWTON_RUN)
    parser.add_argument(
        "--first-operator-run-dir", type=Path, default=DEFAULT_FIRST_OPERATOR_RUN
    )
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--signal-run-dir", type=Path, default=DEFAULT_SIGNAL_RUN)
    parser.add_argument("--benchmark-metrics-dir", type=Path, default=DEFAULT_BENCHMARK_METRICS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calibration-paths", type=int, default=4096)
    parser.add_argument("--component-pool-paths", type=int, default=4096)
    parser.add_argument("--validation-evaluation-paths", type=int, default=512)
    parser.add_argument("--validation-replicates", type=int, default=3)
    parser.add_argument("--test-paths", type=int, default=8192)
    parser.add_argument("--test-metric-subsample", type=int, default=1024)
    parser.add_argument(
        "--weight-grid",
        nargs="+",
        type=float,
        default=tuple(index / 20.0 for index in range(21)),
    )
    parser.add_argument("--seed", type=int, default=2_609_101)
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


def _load_price_split(root: Path, split: str) -> tuple[np.ndarray, torch.Tensor]:
    prices = np.asarray(np.load(root / f"{split}.npy"), dtype=np.float64)
    logs = np.log(prices / prices[:, :1])
    return prices, torch.from_numpy(logs).unsqueeze(-1).float()


def _generate_pair(
    *, model, paths: torch.Tensor, base_policy, candidate_policy, count: int, seed: int
) -> tuple[RolloutResult, RolloutResult]:
    x0, noise = _sample_inputs(
        model=model,
        initial_state_pool=paths,
        count=int(count),
        seed=int(seed),
    )
    with torch.no_grad():
        base = rollout_causal_policy(
            model=model, policy=base_policy, x0=x0, noise=noise
        )
        candidate = rollout_causal_policy(
            model=model, policy=candidate_policy, x0=x0, noise=noise
        )
    return base, candidate


def _mixture_indices(
    *, count: int, weight: float, pool_count: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    component_one = int(round(float(weight) * int(count)))
    component_zero = int(count) - component_one
    if component_zero > int(pool_count) or component_one > int(pool_count):
        raise ValueError("component pool is too small for the requested mixture")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    zero = torch.randperm(int(pool_count), generator=generator)[:component_zero]
    one = torch.randperm(int(pool_count), generator=generator)[:component_one]
    labels = torch.cat(
        (torch.zeros(component_zero, dtype=torch.int64), torch.ones(component_one, dtype=torch.int64))
    )
    order = torch.randperm(int(count), generator=generator)
    return zero, one, labels.index_select(0, order)


def _mix_rollouts(
    *, base: RolloutResult, candidate: RolloutResult, weight: float, count: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zero, one, shuffled_labels = _mixture_indices(
        count=int(count),
        weight=float(weight),
        pool_count=min(int(base.paths.shape[0]), int(candidate.paths.shape[0])),
        seed=int(seed),
    )
    paths = torch.cat((base.paths.index_select(0, zero.to(base.paths.device)), candidate.paths.index_select(0, one.to(candidate.paths.device))))
    controls = torch.cat((base.controls.index_select(0, zero.to(base.controls.device)), candidate.controls.index_select(0, one.to(candidate.controls.device))))
    # Apply the same deterministic shuffle encoded by the labels. Reconstruct
    # its order from component occurrence counters to avoid storing path IDs.
    zero_position = 0
    one_position = int(zero.shape[0])
    order_values = []
    for label in shuffled_labels.tolist():
        if label == 0:
            order_values.append(zero_position)
            zero_position += 1
        else:
            order_values.append(one_position)
            one_position += 1
    order = torch.tensor(order_values, device=paths.device, dtype=torch.long)
    return paths.index_select(0, order), controls.index_select(0, order), shuffled_labels


def _objective(
    *, model, paths: torch.Tensor, controls: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    with torch.no_grad():
        _, metrics = model._complete_objective_metrics(
            generated_path=paths,
            controls=controls,
            target_path=target.to(device=paths.device, dtype=paths.dtype),
            training=model.training,
        )
    return metrics


def _validation_curve(
    *,
    model,
    base: RolloutResult,
    candidate: RolloutResult,
    target_pool: torch.Tensor,
    weights: tuple[float, ...],
    count: int,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    rows = []
    for weight in weights:
        values = []
        details = []
        for replicate in range(int(replicates)):
            local_seed = int(seed) + replicate * 10_000
            paths, controls, _ = _mix_rollouts(
                base=base,
                candidate=candidate,
                weight=float(weight),
                count=int(count),
                seed=local_seed,
            )
            generator = torch.Generator(device="cpu").manual_seed(local_seed + 1)
            target_index = torch.randperm(
                int(target_pool.shape[0]), generator=generator
            )[: int(count)]
            metrics = _objective(
                model=model,
                paths=paths,
                controls=controls,
                target=target_pool.index_select(0, target_index),
            )
            values.append(float(metrics["complete_objective_value"]))
            details.append(metrics)
        rows.append(
            {
                "weight": float(weight),
                "complete_objective_values": values,
                "mean_complete_objective": float(np.mean(values)),
                "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "replicate_metrics": details,
            }
        )
    return {"rows": rows}


def _drawdown_errors(
    *, generated_log: np.ndarray, target_prices: np.ndarray, configuration: dict[str, object]
) -> dict[str, object]:
    generated_prices = float(configuration["s0"]) * np.exp(generated_log)
    kwargs = {
        "threshold": float(configuration["drawdown_threshold"]),
        "split": int(configuration["split_index"]),
        "history_cutoff": int(configuration["history_cutoff"]),
        "recent_window": int(configuration["recent_window"]),
        "horizon": int(configuration["future_horizon"]),
        "annualization": 1.0 / float(configuration["dt"]),
    }
    target_features = drawdown_features(target_prices, **kwargs)
    generated_features = drawdown_features(generated_prices, **kwargs)
    target_memory = summarize_memory(target_features)
    generated_memory = summarize_memory(generated_features)
    lags = np.arange(1, 51)
    target_returns = target_features["returns"]
    generated_returns = generated_features["returns"]
    return {
        "target_memory": target_memory,
        "generated_memory": generated_memory,
        "errors": {
            "return_std_error": abs(float(generated_returns.std() - target_returns.std())),
            "excess_kurtosis_error": abs(
                excess_kurtosis(generated_returns) - excess_kurtosis(target_returns)
            ),
            "abs_return_acf_rmse_lags_1_50": float(
                np.sqrt(
                    np.mean(
                        (
                            pooled_acf(np.abs(generated_returns), lags)
                            - pooled_acf(np.abs(target_returns), lags)
                        )
                        ** 2
                    )
                )
            ),
            "squared_return_acf_rmse_lags_1_50": float(
                np.sqrt(
                    np.mean(
                        (
                            pooled_acf(generated_returns**2, lags)
                            - pooled_acf(target_returns**2, lags)
                        )
                        ** 2
                    )
                )
            ),
            "terminal_log_price_ks": float(
                ks_2samp(
                    generated_features["log_paths"][:, -1],
                    target_features["log_paths"][:, -1],
                ).statistic
            ),
            "future_rv_wasserstein": float(
                wasserstein_distance(
                    generated_features["future_rv"], target_features["future_rv"]
                )
            ),
            "early_hit_rate_error": abs(
                generated_memory["early_hit_rate"] - target_memory["early_hit_rate"]
            ),
            "future_rv_hit_gap_error": abs(
                generated_memory["future_rv_hit_gap"]
                - target_memory["future_rv_hit_gap"]
            ),
            "early_history_incremental_r2_error": abs(
                generated_memory["early_history_incremental_r2"]
                - target_memory["early_history_incremental_r2"]
            ),
        },
    }


def main() -> None:
    args = parse_args()
    run_dir = ensure_run_dir(args.run_dir.resolve())
    weights = tuple(sorted(set(float(value) for value in args.weight_grid)))
    if not weights or weights[0] != 0.0 or weights[-1] != 1.0 or any(
        value < 0.0 or value > 1.0 for value in weights
    ):
        raise ValueError("weight grid must be unique in [0,1] and include 0 and 1")
    newton = json.loads(
        (args.newton_run_dir.resolve() / "report.json").read_text(encoding="utf-8")
    )
    theta = torch.tensor(
        newton["trust_region"]["accepted_theta"], dtype=torch.float64
    )
    model, train_paths = _construct_model(_model_args(args))
    _restrict_to_volatility(model)
    checkpoint_path = (
        args.checkpoint_root.resolve()
        / "residual_eta4"
        / "fresh_adjoint_residual_conditional_eta4.pt"
    )
    checkpoint = torch.load(
        checkpoint_path, map_location=model.device, weights_only=False
    )
    p_network = copy.deepcopy(model.network)
    p_network.load_state_dict(checkpoint["state_dict"])
    p_network.eval()
    projection_path = (
        args.first_operator_run_dir.resolve() / "residual_aware_ridge_r.pt"
    )
    projection = torch.load(projection_path, map_location="cpu", weights_only=False)
    ridge_policy = DirectRidgeRAdjointPolicy(
        model=model, p_network=p_network, r_projection=projection
    )
    blocks = _time_blocks(int(model.grid.num_steps), 3)
    reference_policy = VolatilityOnlyBlockwiseRPolicy(
        model=model,
        candidate_policy=ridge_policy,
        theta_r=torch.zeros_like(theta),
        blocks=blocks,
    )
    stage_policy = VolatilityOnlyBlockwiseRPolicy(
        model=model,
        candidate_policy=ridge_policy,
        theta_r=theta,
        blocks=blocks,
    )
    _, disc = _load_price_split(args.dataset_root.resolve(), "disc")
    test_prices, test = _load_price_split(args.dataset_root.resolve(), "test")
    disc_left = disc[: int(disc.shape[0]) // 2]
    disc_right = disc[int(disc.shape[0]) // 2 :]

    print("generating independent validation-selection components", flush=True)
    select_base, select_stage = _generate_pair(
        model=model,
        paths=train_paths,
        base_policy=reference_policy,
        candidate_policy=stage_policy,
        count=int(args.component_pool_paths),
        seed=int(args.seed),
    )
    print("generating independent validation-confirmation components", flush=True)
    confirm_base, confirm_stage = _generate_pair(
        model=model,
        paths=train_paths,
        base_policy=reference_policy,
        candidate_policy=stage_policy,
        count=int(args.component_pool_paths),
        seed=int(args.seed) + 100_000,
    )
    selection = _validation_curve(
        model=model,
        base=select_base,
        candidate=select_stage,
        target_pool=disc_left,
        weights=weights,
        count=int(args.validation_evaluation_paths),
        replicates=int(args.validation_replicates),
        seed=int(args.seed) + 200_000,
    )
    selected_row = min(
        selection["rows"],
        key=lambda row: (float(row["mean_complete_objective"]), float(row["weight"])),
    )
    selected_weight = float(selected_row["weight"])
    confirmation = _validation_curve(
        model=model,
        base=confirm_base,
        candidate=confirm_stage,
        target_pool=disc_right,
        weights=(0.0, selected_weight, 1.0),
        count=int(args.validation_evaluation_paths),
        replicates=int(args.validation_replicates),
        seed=int(args.seed) + 300_000,
    )
    confirmation_by_weight = {
        float(row["weight"]): row for row in confirmation["rows"]
    }
    reference_values = confirmation_by_weight[0.0]["complete_objective_values"]
    mixture_values = confirmation_by_weight[selected_weight]["complete_objective_values"]
    stage_values = confirmation_by_weight[1.0]["complete_objective_values"]
    stage_wins_over_reference = [
        float(stage) < float(reference)
        for stage, reference in zip(stage_values, reference_values)
    ]
    mixture_wins_over_reference = [
        float(mixture) < float(reference)
        for mixture, reference in zip(mixture_values, reference_values)
    ]
    mixture_wins_over_stage = [
        float(mixture) < float(stage)
        for mixture, stage in zip(mixture_values, stage_values)
    ]
    robust_stage_improvement = all(stage_wins_over_reference)
    robust_mixture_improvement = (
        0.0 < selected_weight < 1.0
        and all(mixture_wins_over_reference)
        and all(mixture_wins_over_stage)
    )

    print("generating untouched test component banks", flush=True)
    test_base, test_stage = _generate_pair(
        model=model,
        paths=train_paths,
        base_policy=reference_policy,
        candidate_policy=stage_policy,
        count=int(args.test_paths),
        seed=int(args.seed) + 400_000,
    )
    test_mixture_paths, test_mixture_controls, test_labels = _mix_rollouts(
        base=test_base,
        candidate=test_stage,
        weight=selected_weight,
        count=int(args.test_paths),
        seed=int(args.seed) + 500_000,
    )
    banks = {
        "reference": (test_base.paths, test_base.controls),
        "one_stage_mp": (test_stage.paths, test_stage.controls),
        "validation_selected_mixture": (test_mixture_paths, test_mixture_controls),
    }
    manifest = json.loads(
        (args.dataset_root.resolve() / "manifest.json").read_text(encoding="utf-8")
    )
    configuration = manifest["configuration"]
    target_test = test.to(device=model.device, dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed) + 600_000)
    metric_indices = torch.randperm(
        int(target_test.shape[0]), generator=generator
    )[: int(args.test_metric_subsample)]
    target_metric = target_test.index_select(0, metric_indices.to(target_test.device))

    sys.path.insert(0, str(args.benchmark_metrics_dir.resolve()))
    from metrics import (  # type: ignore[import-not-found]  # noqa: PLC0415
        increment_mmd2,
        mmd2,
        path_swd,
        terminal_mmd2,
        terminal_swd,
        volatility_mmd,
    )

    benchmark_functions = {
        "path_mmd2": mmd2,
        "terminal_mmd2": terminal_mmd2,
        "increment_mmd2": increment_mmd2,
        "terminal_swd": terminal_swd,
        "path_swd": path_swd,
        "volatility_mmd": volatility_mmd,
    }
    test_results = {}
    for index, (name, (paths, controls)) in enumerate(banks.items()):
        print(f"untouched test metrics: {name}", flush=True)
        state_dim = int(paths.shape[-1])
        if not bool(torch.all(controls[..., :state_dim] == 0.0).item()):
            raise RuntimeError("test bank violated the zero-drift admissible set")
        fake_generator = torch.Generator(device="cpu").manual_seed(
            int(args.seed) + 700_000
        )
        fake_indices = torch.randperm(
            int(paths.shape[0]), generator=fake_generator
        )[: int(args.test_metric_subsample)]
        fake_metric = paths.index_select(0, fake_indices.to(paths.device))
        target_np = target_metric.detach().cpu().numpy().astype(np.float64)
        fake_np = fake_metric.detach().cpu().numpy().astype(np.float64)
        benchmark = {
            metric: float(function(target_np, fake_np))
            for metric, function in benchmark_functions.items()
        }
        stylized = stylized_fact_metrics(
            fake_metric,
            target_metric,
            lags=(1, 5, 10, 20, 40),
            include_swd=True,
            swd_projections=128,
        )
        generated_log = paths.detach().cpu().numpy()[..., 0].astype(np.float64)
        drawdown = _drawdown_errors(
            generated_log=generated_log,
            target_prices=test_prices,
            configuration=configuration,
        )
        output_path = run_dir / name / "generated_paths_8192x128.npy"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, generated_log)
        test_results[name] = {
            "benchmark_path_metrics": benchmark,
            "stylized_fact_metrics": stylized,
            "drawdown_memory": drawdown,
            "generated_path": str(output_path.resolve()),
        }

    report = {
        "protocol": {
            "method": "finite-stage MP boosting with optional path-law mixture",
            "stationarity_claimed": False,
            "reference_and_stage_frozen_before_validation": True,
            "mixture_component_sampled_once_per_path": True,
            "weight_selected_on_disc_first_half_only": True,
            "weight_confirmed_on_disc_second_half": True,
            "test_used_once_after_weight_selection": True,
            "test_paths_used_for_selection": False,
            "candidate_source": str(args.newton_run_dir.resolve()),
        },
        "accepted_stage_coordinates": theta.tolist(),
        "validation": {
            "weight_grid": list(weights),
            "selection_curve": selection,
            "selected_weight": selected_weight,
            "confirmation_curve": confirmation,
            "stage_wins_over_reference": stage_wins_over_reference,
            "mixture_wins_over_reference": mixture_wins_over_reference,
            "mixture_wins_over_stage": mixture_wins_over_stage,
            "robust_stage_improvement": robust_stage_improvement,
            "robust_mixture_improvement": robust_mixture_improvement,
            "next_phase": (
                "evaluate_at_most_two_additional_components"
                if robust_mixture_improvement
                else "stop_stagewise_mixture_branch"
            ),
        },
        "test": {
            "selected_mixture_realized_component_one_fraction": float(
                test_labels.double().mean().item()
            ),
            "results": test_results,
        },
    }
    write_json(run_dir / "report.json", report)
    lines = [
        "# Frozen one-stage MP and path-law mixture evaluation",
        "",
        f"- Selected mixture weight: `{selected_weight:.3f}`",
        f"- One-stage wins over reference: `{stage_wins_over_reference}`",
        f"- Mixture wins over reference: `{mixture_wins_over_reference}`",
        f"- Mixture wins over one-stage: `{mixture_wins_over_stage}`",
        f"- Robust one-stage improvement: `{robust_stage_improvement}`",
        f"- Robust mixture improvement: `{robust_mixture_improvement}`",
        f"- Next phase: `{report['validation']['next_phase']}`",
        "",
        "| Test metric | Reference | One-stage MP | Mixture |",
        "|---|---:|---:|---:|",
    ]
    for metric in (
        "path_mmd2",
        "increment_mmd2",
        "path_swd",
        "terminal_swd",
        "volatility_mmd",
    ):
        lines.append(
            f"| {metric} | "
            f"{test_results['reference']['benchmark_path_metrics'][metric]:.6g} | "
            f"{test_results['one_stage_mp']['benchmark_path_metrics'][metric]:.6g} | "
            f"{test_results['validation_selected_mixture']['benchmark_path_metrics'][metric]:.6g} |"
        )
    for metric in (
        "abs_return_acf_rmse_lags_1_50",
        "future_rv_wasserstein",
        "future_rv_hit_gap_error",
        "early_history_incremental_r2_error",
    ):
        lines.append(
            f"| {metric} | "
            f"{test_results['reference']['drawdown_memory']['errors'][metric]:.6g} | "
            f"{test_results['one_stage_mp']['drawdown_memory']['errors'][metric]:.6g} | "
            f"{test_results['validation_selected_mixture']['drawdown_memory']['errors'][metric]:.6g} |"
        )
    (run_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["validation"], indent=2), flush=True)


if __name__ == "__main__":
    main()
