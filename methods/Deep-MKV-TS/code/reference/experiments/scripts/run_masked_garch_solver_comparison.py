#!/usr/bin/env python3
"""Compare sampled accuracy-gated and previous nested solvers on hidden GARCH truth."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys
import time

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from deep_mkv_gen_path_dt import (  # noqa: E402
    DiscreteMPNestedTrainingConfig,
    DiscreteMPTrainingConfig,
)
from deep_mkv_gen_path_dt.noise import (  # noqa: E402
    derive_stream_seed,
    sample_standard_normals,
)
from path_dt_experiments.accuracy_gated_nested import (  # noqa: E402
    FrozenCETrainingConfig,
    ReferenceControlPolicy,
    RelaxedLogVolatilityPolicy,
    ScaledAdjointControlPolicy,
    cpu_network_state,
    fit_frozen_sampled_ce,
    prediction_metrics,
)
from path_dt_experiments.causal_transformer_adjoint import (  # noqa: E402
    CausalTransformerAdjointNetwork,
)
from path_dt_experiments.frozen_garch_ce_gru import (  # noqa: E402
    collapse_leaf_controls_to_variables,
)
from path_dt_experiments.exact_garch_stationarity import (  # noqa: E402
    exact_garch_mp_stationarity,
)
from path_dt_experiments.garch_tree_evaluation import (  # noqa: E402
    GARCH_DEPENDENCE_METRICS,
    GARCH_DISTRIBUTION_METRICS,
    gap_closed,
    garch_tree_metric_errors,
    summarize_gap_closure,
)
from path_dt_experiments.garch_tree_oracle import GarchTreeConfig  # noqa: E402
from path_dt_experiments.garch_tree_production import (  # noqa: E402
    ProductionGarchTreeProblem,
    build_production_garch_components,
    enumerated_garch_noise,
    exact_garch_tree_mp_response,
    production_model_on_exact_tree,
)
from path_dt_experiments.garch_tree_sampling import (  # noqa: E402
    bank_with_garch_tree_noise,
    pool_garch_tree_prefix_targets,
    replay_garch_tree_branch_noise,
    sample_garch_tree_noise,
)
from path_dt_experiments.lifted_collocation_solver import (  # noqa: E402
    CausalControlPolicy,
    ScenarioDataContext,
    ScenarioCollocationBank,
    build_scenario_collocation_bank,
    rollout_causal_policy,
    rollout_policy_on_bank,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("new", "previous"), required=True)
    parser.add_argument("--exact-run-dir", type=Path, required=True)
    parser.add_argument("--online-report", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outer-steps", type=int, default=10)
    parser.add_argument("--fit-paths", type=int, default=512)
    parser.add_argument("--validation-paths", type=int, default=256)
    parser.add_argument("--fit-branches", type=int, default=64)
    parser.add_argument("--validation-branches", type=int, default=256)
    parser.add_argument("--population-paths", type=int, default=4096)
    parser.add_argument("--query-batch-size", type=int, default=65536)
    parser.add_argument("--inner-max-steps", type=int, default=3000)
    parser.add_argument("--previous-inner-steps", type=int, default=200)
    parser.add_argument("--inner-min-steps", type=int, default=200)
    parser.add_argument("--inner-batch-size", type=int, default=256)
    parser.add_argument("--inner-eval-every", type=int, default=100)
    parser.add_argument("--ce-relative-tolerance", type=float, default=0.15)
    parser.add_argument("--objective-paths", type=int, default=2048)
    parser.add_argument("--selection-banks", type=int, default=3)
    parser.add_argument("--confirmation-banks", type=int, default=3)
    parser.add_argument("--minimum-relative-improvement", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument(
        "--ce-network",
        choices=("gru", "causal_transformer"),
        default="gru",
        help="causal sequence model fitted directly to sampled CE labels",
    )
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--swd-projections", type=int, default=512)
    parser.add_argument(
        "--innovation-law",
        choices=("gaussian", "tree"),
        default="gaussian",
        help="law used for base paths, conditional branches, and objective banks",
    )
    parser.add_argument(
        "--operative-r-only-fit",
        action="store_true",
        help="fit only the volatility-relevant R head; P remains diagnostic",
    )
    return parser.parse_args()


def _load(
    args: argparse.Namespace,
) -> tuple[
    ProductionGarchTreeProblem,
    torch.Tensor,
    dict[str, object],
    GarchTreeConfig,
]:
    exact_dir = args.exact_run_dir.resolve()
    report = json.loads((exact_dir / "exact_report.json").read_text())
    protocol = report["protocol"]
    configuration = report["configuration"]
    config = GarchTreeConfig(
        horizon=int(configuration["horizon"]),
        dt=float(configuration["dt"]),
        long_run_sigma=float(configuration["long_run_sigma"]),
        arch=float(configuration["arch"]),
        garch=float(configuration["garch"]),
        sigma_min=float(configuration["sigma_bounds"][0]),
        sigma_max=float(configuration["sigma_bounds"][1]),
        eta=float(protocol["eta"]),
        lambda_scale=float(protocol["lambda_scale"]),
    )
    device = torch.device(args.device)
    problem = ProductionGarchTreeProblem(
        config,
        lambda_scale=float(protocol["lambda_scale"]),
        kappa_scale=float(protocol["kappa_scale"]),
        device=device,
        dtype=torch.float64,
    )
    exact = torch.load(
        exact_dir / "exact.pt", map_location=device, weights_only=True
    )["exact_variables"].to(device=device, dtype=torch.float64)
    return problem, exact, report, config


def _training(*, seed: int, steps: int) -> DiscreteMPTrainingConfig:
    return DiscreteMPTrainingConfig(
        batch_size=256,
        target_batch_size=256,
        num_steps=int(steps),
        lr=2e-3,
        weight_decay=1e-5,
        lambda_scale=50.0,
        adjoint_weight=1.0,
        adjoint_noise_weight=1.0,
        ce_target_mode="direct",
        ridge_lambda=1e-3,
        ce_crossfit_folds=1,
        target_preconditioner="timewise",
        preconditioner_batches=1,
        noise_target_control_variate="none",
        grad_clip_norm=5.0,
        log_every=100,
        seed=int(seed),
        observed_only=False,
    )


def _solver_noise(
    *,
    model,
    count: int,
    seed: int,
    config: GarchTreeConfig,
    innovation_law: str,
) -> torch.Tensor:
    if str(innovation_law) == "tree":
        return sample_garch_tree_noise(
            config,
            batch_size=int(count),
            seed=int(seed),
            device=model.device,
            dtype=model.dtype,
        )
    if int(count) % 2:
        raise ValueError("Gaussian antithetic path count must be even")
    half = sample_standard_normals(
        grid=model.grid,
        batch_size=int(count) // 2,
        noise_dim=int(model.architecture.noise_dim),
        device=model.device,
        dtype=model.dtype,
        seed=int(seed),
    )
    return torch.cat((half, -half), dim=0)


def _objective_banks(
    *,
    model,
    target_paths: torch.Tensor,
    count: int,
    paths_per_bank: int,
    seed: int,
    config: GarchTreeConfig,
    innovation_law: str,
) -> list[dict[str, torch.Tensor | int]]:
    banks = []
    for index in range(int(count)):
        bank_seed = derive_stream_seed(
            base_seed=int(seed), stream_offset=10_000 * index
        )
        assert bank_seed is not None
        noise = (
            enumerated_garch_noise(
                config, device=model.device, dtype=model.dtype
            )
            if str(innovation_law) == "tree"
            and int(paths_per_bank) == int(config.leaf_count)
            else _solver_noise(
                model=model,
                count=int(paths_per_bank),
                seed=bank_seed,
                config=config,
                innovation_law=str(innovation_law),
            )
        )
        x0 = torch.zeros(
            int(paths_per_bank),
            int(model.architecture.state_dim),
            device=model.device,
            dtype=model.dtype,
        )
        banks.append(
            {
                "seed": int(bank_seed),
                "x0": x0,
                "noise": noise,
                "target": target_paths,
            }
        )
    return banks


def _with_complete_target_bank(
    bank: ScenarioCollocationBank,
    *,
    target_count: int,
) -> ScenarioCollocationBank:
    """Use every observed target path once while retaining sampled source noise."""

    return ScenarioCollocationBank(
        initial_indices=bank.initial_indices,
        target_indices=torch.arange(int(target_count), dtype=torch.long),
        base_noise=bank.base_noise,
        branch_seed=int(bank.branch_seed),
        branch_stream_offset=int(bank.branch_stream_offset),
        num_branches=int(bank.num_branches),
        antithetic=bool(bank.antithetic),
        beta=float(bank.beta),
        dataset_fingerprint=bank.dataset_fingerprint,
        label=bank.label,
    )


def _evaluate_objective_banks(
    *,
    model,
    policy: CausalControlPolicy,
    banks: list[dict[str, torch.Tensor | int]],
    training: DiscreteMPTrainingConfig,
) -> list[float]:
    values = []
    with torch.no_grad():
        for bank in banks:
            rollout = rollout_causal_policy(
                model=model,
                policy=policy,
                x0=bank["x0"],
                noise=bank["noise"],
            )
            value, _ = model._complete_objective_metrics(
                generated_path=rollout.paths,
                controls=rollout.controls,
                target_path=bank["target"],
                training=training,
            )
            values.append(float(value.item()))
    return values


def _objective_comparison(
    baseline: list[float], candidate: list[float]
) -> dict[str, object]:
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("objective bank values must be nonempty and aligned")
    rows = []
    for index, (before, after) in enumerate(zip(baseline, candidate, strict=True)):
        rows.append(
            {
                "bank": index,
                "before": float(before),
                "after": float(after),
                "improvement": float(before - after),
                "relative_improvement": float(
                    (before - after) / max(abs(before), 1e-30)
                ),
                "improved": bool(after < before),
            }
        )
    return {
        "before_mean": float(sum(baseline) / len(baseline)),
        "after_mean": float(sum(candidate) / len(candidate)),
        "improvement_mean": float(
            sum(before - after for before, after in zip(baseline, candidate))
            / len(baseline)
        ),
        "relative_improvement_mean": float(
            sum(row["relative_improvement"] for row in rows) / len(rows)
        ),
        "improvement_fraction": float(
            sum(row["improved"] for row in rows) / len(rows)
        ),
        "all_improved": bool(all(row["improved"] for row in rows)),
        "rows": rows,
    }


def _replication_metrics(
    left: torch.Tensor,
    right: torch.Tensor,
) -> dict[str, float]:
    left64 = left.detach().double()
    right64 = right.detach().double()
    scale = torch.sqrt(torch.mean(0.5 * (left64.pow(2) + right64.pow(2)))).clamp_min(
        1e-30
    )
    centered_left = left64 - left64.mean(dim=0, keepdim=True)
    centered_right = right64 - right64.mean(dim=0, keepdim=True)
    denominator = (
        torch.linalg.vector_norm(centered_left.reshape(-1))
        * torch.linalg.vector_norm(centered_right.reshape(-1))
    ).clamp_min(1e-30)
    disagreement = torch.mean((left64 - right64).pow(2)).sqrt() / scale
    return {
        "symmetric_relative_disagreement": float(disagreement.item()),
        "estimated_average_label_noise_floor": float(0.5 * disagreement.item()),
        "correlation": float(
            (torch.sum(centered_left * centered_right) / denominator).item()
        ),
        "amplitude_ratio": float(
            (
                right64.pow(2).mean().sqrt()
                / left64.pow(2).mean().sqrt().clamp_min(1e-30)
            ).item()
        ),
    }


def _hidden_exact_evaluation(
    *,
    problem: ProductionGarchTreeProblem,
    exact: torch.Tensor,
    model,
    policy: CausalControlPolicy,
    reference_errors: dict[str, float],
    swd_projections: int,
) -> tuple[dict[str, object], torch.Tensor]:
    noise = enumerated_garch_noise(
        problem.config, device=model.device, dtype=model.dtype
    )
    x0 = torch.zeros(
        int(problem.config.leaf_count),
        1,
        device=model.device,
        dtype=model.dtype,
    )
    with torch.no_grad():
        rollout = rollout_causal_policy(
            model=model, policy=policy, x0=x0, noise=noise
        )
    variables, consensus = collapse_leaf_controls_to_variables(
        problem, rollout.controls.to(problem.device, problem.dtype)
    )
    objective = float(problem.objective(variables).item())
    reference = problem.reference_start()
    reference_objective = float(problem.objective(reference).item())
    exact_objective = float(problem.objective(exact).item())
    errors = garch_tree_metric_errors(
        problem,
        problem.solution(variables),
        swd_projections=int(swd_projections),
    )
    closure = gap_closed(reference_errors, errors)
    stationarity = exact_garch_mp_stationarity(
        problem,
        variables,
        gradient_audit=True,
    )
    return (
        {
            "objective": objective,
            "objective_gain_fraction_of_exact": (
                (reference_objective - objective)
                / (reference_objective - exact_objective)
            ),
            "nodewise_control_consensus_max_abs": float(consensus),
            "errors": errors,
            "gap_closed": closure,
            "distribution": summarize_gap_closure(
                closure, GARCH_DISTRIBUTION_METRICS
            ),
            "dependence": summarize_gap_closure(
                closure, GARCH_DEPENDENCE_METRICS
            ),
            "hidden_exact_stationarity": stationarity.metrics,
        },
        variables,
    )


def _run_new(
    *,
    args: argparse.Namespace,
    problem: ProductionGarchTreeProblem,
    exact: torch.Tensor,
    exact_report: dict[str, object],
    config: GarchTreeConfig,
    run_dir: Path,
) -> None:
    model, _, _ = build_production_garch_components(
        config,
        lambda_scale=50.0,
        kappa_scale=100.0,
        hidden_dim=int(args.hidden_dim),
        seed=int(args.seed),
        device=problem.device,
        dtype=torch.float32,
    )
    target_paths = problem.target.leaf_paths.unsqueeze(-1).to(
        model.device, model.dtype
    )
    context = ScenarioDataContext(
        initial_state_pool=target_paths,
        target_path_pool=target_paths,
        dataset_id="masked-garch-tree-target",
    )
    training = _training(
        seed=int(args.seed), steps=int(args.outer_steps) * int(args.inner_max_steps)
    )
    current_policy: CausalControlPolicy = ReferenceControlPolicy(model)
    warm_state: dict[str, torch.Tensor] | None = None
    network_template: torch.nn.Module | None = None
    if str(args.ce_network) == "causal_transformer":
        network_template = CausalTransformerAdjointNetwork(
            state_dim=int(model.architecture.state_dim),
            adjoint_dim=int(model.architecture.state_dim),
            noise_adjoint_dim=int(model.architecture.noise_dim),
            hidden_dim=int(args.hidden_dim),
            num_layers=int(args.transformer_layers),
            num_heads=int(args.transformer_heads),
            max_steps=int(model.grid.num_steps),
            input_mode=str(model.architecture.adjoint_input_mode),
        ).to(device=model.device, dtype=model.dtype)
    history: list[dict[str, object]] = []
    status = "maximum_outer_steps"
    started = time.perf_counter()
    relaxations = tuple(2.0 ** (-index) for index in range(13))

    def use_tree_noise(bank: ScenarioCollocationBank, *, seed: int) -> ScenarioCollocationBank:
        if str(args.innovation_law) != "tree":
            return bank
        if int(bank.initial_indices.numel()) == int(config.leaf_count):
            return replace(
                bank,
                base_noise=enumerated_garch_noise(
                    config, device=torch.device("cpu"), dtype=bank.base_noise.dtype
                ),
                antithetic=False,
            )
        return bank_with_garch_tree_noise(
            bank, config=config, seed=int(seed)
        )

    def branch_provider(bank: ScenarioCollocationBank):
        if str(args.innovation_law) != "tree":
            return None
        return lambda step: replay_garch_tree_branch_noise(
            config=config,
            bank=bank,
            base_noise=bank.base_noise,
            step_index=int(step),
        )

    for outer in range(int(args.outer_steps)):
        outer_seed = derive_stream_seed(
            base_seed=int(args.seed), stream_offset=100_000 * outer
        )
        assert outer_seed is not None
        fit_bank = build_scenario_collocation_bank(
            model=model,
            context=context,
            path_count=int(args.fit_paths),
            target_count=int(problem.config.leaf_count),
            num_branches=int(args.fit_branches),
            antithetic=str(args.innovation_law) == "gaussian",
            beta=1.0,
            seed=outer_seed + 1,
            label=f"fit_{outer}",
        )
        fit_bank = use_tree_noise(fit_bank, seed=outer_seed + 101)
        fit_bank = _with_complete_target_bank(
            fit_bank, target_count=int(problem.config.leaf_count)
        )
        validation_bank = build_scenario_collocation_bank(
            model=model,
            context=context,
            path_count=int(args.validation_paths),
            target_count=int(problem.config.leaf_count),
            num_branches=int(args.validation_branches),
            antithetic=str(args.innovation_law) == "gaussian",
            beta=1.0,
            seed=outer_seed + 2,
            label=f"validation_{outer}",
        )
        validation_bank = use_tree_noise(
            validation_bank, seed=outer_seed + 102
        )
        validation_bank = _with_complete_target_bank(
            validation_bank, target_count=int(problem.config.leaf_count)
        )
        validation_replication_bank = ScenarioCollocationBank(
            initial_indices=validation_bank.initial_indices,
            target_indices=validation_bank.target_indices,
            base_noise=validation_bank.base_noise,
            branch_seed=int(validation_bank.branch_seed) + 500_000,
            branch_stream_offset=int(validation_bank.branch_stream_offset),
            num_branches=int(validation_bank.num_branches),
            antithetic=bool(validation_bank.antithetic),
            beta=float(validation_bank.beta),
            dataset_fingerprint=validation_bank.dataset_fingerprint,
            label=f"validation_replication_{outer}",
        )
        population_noise = (
            enumerated_garch_noise(
                config, device=model.device, dtype=model.dtype
            )
            if str(args.innovation_law) == "tree"
            else _solver_noise(
                model=model,
                count=int(args.population_paths),
                seed=outer_seed + 500,
                config=config,
                innovation_law=str(args.innovation_law),
            )
        )
        population_x0 = torch.zeros(
            int(population_noise.shape[0]),
            int(model.architecture.state_dim),
            device=model.device,
            dtype=model.dtype,
        )
        with torch.no_grad():
            population_rollout = rollout_causal_policy(
                model=model,
                policy=current_policy,
                x0=population_x0,
                noise=population_noise,
            )
        frozen_population_paths = population_rollout.paths.detach()
        fit_law = rollout_policy_on_bank(
            model=model,
            policy=current_policy,
            bank=fit_bank,
            data_context=context,
            full_training=training,
            query_batch_size=int(args.query_batch_size),
            population_paths=frozen_population_paths,
            branch_noise_provider=branch_provider(fit_bank),
        )
        validation_law = rollout_policy_on_bank(
            model=model,
            policy=current_policy,
            bank=validation_bank,
            data_context=context,
            full_training=training,
            query_batch_size=int(args.query_batch_size),
            population_paths=frozen_population_paths,
            branch_noise_provider=branch_provider(validation_bank),
        )
        validation_replication_law = rollout_policy_on_bank(
            model=model,
            policy=current_policy,
            bank=validation_replication_bank,
            data_context=context,
            full_training=training,
            query_batch_size=int(args.query_batch_size),
            population_paths=frozen_population_paths,
            branch_noise_provider=branch_provider(
                validation_replication_bank
            ),
        )
        fit_target_p = fit_law.target_p
        fit_target_r = fit_law.target_r
        validation_first_p = validation_law.target_p
        validation_first_r = validation_law.target_r
        validation_second_p = validation_replication_law.target_p
        validation_second_r = validation_replication_law.target_r
        if str(args.innovation_law) == "tree":
            fit_target_p = pool_garch_tree_prefix_targets(
                config,
                base_noise=fit_bank.base_noise.to(fit_law.target_p),
                targets=fit_law.target_p,
            )
            fit_target_r = pool_garch_tree_prefix_targets(
                config,
                base_noise=fit_bank.base_noise.to(fit_law.target_r),
                targets=fit_law.target_r,
            )
            validation_first_p = pool_garch_tree_prefix_targets(
                config,
                base_noise=validation_bank.base_noise.to(validation_law.target_p),
                targets=validation_law.target_p,
            )
            validation_first_r = pool_garch_tree_prefix_targets(
                config,
                base_noise=validation_bank.base_noise.to(validation_law.target_r),
                targets=validation_law.target_r,
            )
            validation_second_p = pool_garch_tree_prefix_targets(
                config,
                base_noise=validation_replication_bank.base_noise.to(
                    validation_replication_law.target_p
                ),
                targets=validation_replication_law.target_p,
            )
            validation_second_r = pool_garch_tree_prefix_targets(
                config,
                base_noise=validation_replication_bank.base_noise.to(
                    validation_replication_law.target_r
                ),
                targets=validation_replication_law.target_r,
            )
        validation_target_p = 0.5 * (
            validation_first_p + validation_second_p
        )
        validation_target_r = 0.5 * (
            validation_first_r + validation_second_r
        )
        p_replication = _replication_metrics(
            validation_first_p, validation_second_p
        )
        r_replication = _replication_metrics(
            validation_first_r, validation_second_r
        )
        fit_config = FrozenCETrainingConfig(
            max_steps=int(args.inner_max_steps),
            minimum_steps=int(args.inner_min_steps),
            batch_size=int(args.inner_batch_size),
            eval_every=int(args.inner_eval_every),
            lr=2e-3,
            weight_decay=1e-5,
            grad_clip_norm=5.0,
            minimum_scale=1e-3,
            validation_relative_tolerance=float(args.ce_relative_tolerance),
            p_loss_weight=(0.0 if bool(args.operative_r_only_fit) else 1.0),
            r_loss_weight=1.0,
            seed=outer_seed + 3,
        )
        fit = fit_frozen_sampled_ce(
            model=model,
            fit_paths=fit_law.paths,
            fit_target_p=fit_target_p,
            fit_target_r=fit_target_r,
            validation_paths=validation_law.paths,
            validation_target_p=validation_target_p,
            validation_target_r=validation_target_r,
            config=fit_config,
            initial_network_state=warm_state,
            network_template=network_template,
        )
        warm_state = cpu_network_state(fit.network)
        p_allowed_error = max(
            float(args.ce_relative_tolerance),
            1.5 * float(p_replication["estimated_average_label_noise_floor"]),
        )
        r_allowed_error = max(
            float(args.ce_relative_tolerance),
            1.5 * float(r_replication["estimated_average_label_noise_floor"]),
        )
        validation_metrics = fit.validation_metrics
        r_gate = bool(
            validation_metrics["r_relative_error"] <= r_allowed_error
            and validation_metrics["r_explained_energy"] > 0.0
            and validation_metrics["r_correlation"] > 0.25
            and 0.25 <= validation_metrics["r_amplitude_ratio"] <= 2.0
        )
        p_gate = bool(
            validation_metrics["p_relative_error"] <= p_allowed_error
            and validation_metrics["p_explained_energy"] > 0.0
            and validation_metrics["p_correlation"] > 0.25
            and 0.25 <= validation_metrics["p_amplitude_ratio"] <= 2.0
        )
        noise_aware_ce_gate = bool(
            r_gate and (bool(args.operative_r_only_fit) or p_gate)
        )
        candidate_policy = ScaledAdjointControlPolicy(
            model=model,
            network=fit.network,
            p_scale=fit.p_scale,
            r_scale=fit.r_scale,
        )
        hidden_ce: dict[str, float] | None = None
        if str(args.innovation_law) == "tree":
            exact_noise = enumerated_garch_noise(
                config, device=model.device, dtype=model.dtype
            )
            exact_x0 = torch.zeros(
                int(config.leaf_count), 1, device=model.device, dtype=model.dtype
            )
            with torch.no_grad():
                current_tree = rollout_causal_policy(
                    model=model,
                    policy=current_policy,
                    x0=exact_x0,
                    noise=exact_noise,
                )
                normalized = fit.network(current_tree.paths)
                predicted_p = (
                    normalized.expected_adjoint_next
                    * fit.p_scale.unsqueeze(0)
                )
                predicted_r = (
                    normalized.expected_adjoint_noise_next
                    * fit.r_scale.unsqueeze(0)
                )
            current_variables, _ = collapse_leaf_controls_to_variables(
                problem,
                current_tree.controls.to(problem.device, problem.dtype),
            )
            exact_response = exact_garch_tree_mp_response(
                problem, current_variables
            )
            innovation_values = torch.tensor(
                config.innovations,
                device=validation_bank.base_noise.device,
                dtype=validation_bank.base_noise.dtype,
            )
            validation_digits = torch.argmin(
                torch.abs(
                    validation_bank.base_noise.squeeze(-1).unsqueeze(-1)
                    - innovation_values
                ),
                dim=-1,
            )
            validation_leaf_indices = torch.zeros(
                int(validation_digits.shape[0]),
                dtype=torch.long,
                device=validation_digits.device,
            )
            for step_index in range(int(config.horizon)):
                validation_leaf_indices = validation_leaf_indices + (
                    validation_digits[:, step_index]
                    * int(config.branching)
                    ** (int(config.horizon) - step_index - 1)
                )
            exact_validation_p = exact_response.expected_adjoint_next.index_select(
                0, validation_leaf_indices.to(problem.device)
            ).to(validation_target_p)
            exact_validation_r = exact_response.expected_adjoint_noise_next.index_select(
                0, validation_leaf_indices.to(problem.device)
            ).to(validation_target_r)
            hidden_ce = {
                **prediction_metrics(
                    predicted_p,
                    exact_response.expected_adjoint_next.to(predicted_p),
                    prefix="p",
                ),
                **prediction_metrics(
                    predicted_r,
                    exact_response.expected_adjoint_noise_next.to(predicted_r),
                    prefix="r",
                ),
                **prediction_metrics(
                    validation_target_p,
                    exact_validation_p,
                    prefix="sampled_target_p",
                ),
                **prediction_metrics(
                    validation_target_r,
                    exact_validation_r,
                    prefix="sampled_target_r",
                ),
            }
        row: dict[str, object] = {
            "outer_step": outer,
            "ce_accuracy_gate_passed": noise_aware_ce_gate,
            "absolute_ce_gate_passed": fit.accuracy_gate_passed,
            "fit_selected_step": fit.selected_step,
            "fit_metrics": fit.fit_metrics,
            "validation_metrics": fit.validation_metrics,
            "p_target_replication": p_replication,
            "r_target_replication": r_replication,
            "p_allowed_relative_error": p_allowed_error,
            "r_allowed_relative_error": r_allowed_error,
            "p_gate_passed": p_gate,
            "r_gate_passed": r_gate,
            "fit_bank_fingerprint": fit_bank.fingerprint,
            "validation_bank_fingerprint": validation_bank.fingerprint,
            "hidden_exact_ce_metrics_not_used_for_selection": hidden_ce,
        }
        if not noise_aware_ce_gate:
            row["update_accepted"] = False
            history.append(row)
            status = "ce_accuracy_gate_failed"
            print(json.dumps(row, sort_keys=True), flush=True)
            break

        selection_banks = _objective_banks(
            model=model,
            target_paths=target_paths,
            count=int(args.selection_banks),
            paths_per_bank=int(args.objective_paths),
            seed=outer_seed + 10_000,
            config=config,
            innovation_law=str(args.innovation_law),
        )
        confirmation_banks = _objective_banks(
            model=model,
            target_paths=target_paths,
            count=int(args.confirmation_banks),
            paths_per_bank=int(args.objective_paths),
            seed=outer_seed + 20_000,
            config=config,
            innovation_law=str(args.innovation_law),
        )
        selection_before = _evaluate_objective_banks(
            model=model,
            policy=current_policy,
            banks=selection_banks,
            training=training,
        )
        selection_candidates = []
        for rate in relaxations:
            policy = RelaxedLogVolatilityPolicy(
                model=model,
                current_policy=current_policy,
                candidate_policy=candidate_policy,
                relaxation=rate,
            )
            values = _evaluate_objective_banks(
                model=model,
                policy=policy,
                banks=selection_banks,
                training=training,
            )
            comparison = _objective_comparison(selection_before, values)
            if (
                comparison["all_improved"]
                and comparison["relative_improvement_mean"]
                >= float(args.minimum_relative_improvement)
            ):
                selection_candidates.append((comparison["after_mean"], rate, policy, comparison))
        selection_candidates.sort(key=lambda item: item[0])
        accepted = None
        confirmation_before = _evaluate_objective_banks(
            model=model,
            policy=current_policy,
            banks=confirmation_banks,
            training=training,
        )
        confirmation_report = None
        for _, rate, policy, selection_report in selection_candidates:
            confirmation_values = _evaluate_objective_banks(
                model=model,
                policy=policy,
                banks=confirmation_banks,
                training=training,
            )
            confirmation_report = _objective_comparison(
                confirmation_before, confirmation_values
            )
            if (
                confirmation_report["all_improved"]
                and confirmation_report["relative_improvement_mean"]
                >= float(args.minimum_relative_improvement)
            ):
                accepted = (rate, policy, selection_report, confirmation_report)
                break
        if accepted is None:
            row.update(
                {
                    "update_accepted": False,
                    "selection_candidate_count": len(selection_candidates),
                    "last_confirmation": confirmation_report,
                }
            )
            history.append(row)
            status = "objective_confirmation_rejected"
            print(json.dumps(row, sort_keys=True), flush=True)
            break
        rate, accepted_policy, selection_report, confirmation_report = accepted
        current_policy = accepted_policy
        row.update(
            {
                "update_accepted": True,
                "relaxation": float(rate),
                "selection_candidate_count": len(selection_candidates),
                "selection": selection_report,
                "confirmation": confirmation_report,
            }
        )
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        write_json(
            run_dir / "progress.json",
            {
                "status": "running",
                "history": history,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )

    reference_errors = garch_tree_metric_errors(
        problem,
        problem.solution(problem.reference_start()),
        swd_projections=int(args.swd_projections),
    )
    hidden, variables = _hidden_exact_evaluation(
        problem=problem,
        exact=exact,
        model=model,
        policy=current_policy,
        reference_errors=reference_errors,
        swd_projections=int(args.swd_projections),
    )
    online = json.loads(args.online_report.resolve().read_text())["selected_deep_mkv"]
    online_gain = float(online["objective_gain_fraction_of_exact"])
    success = bool(
        float(hidden["objective_gain_fraction_of_exact"]) > online_gain
        and int(hidden["distribution"]["improved_count"])
        >= int(online["summary"]["distribution"]["improved_count"])
        and int(hidden["dependence"]["improved_count"])
        >= int(online["summary"]["dependence"]["improved_count"])
    )
    report = {
        "status": status,
        "arm": "accuracy_gated_nested",
        "protocol": {
            "target_dataset": "sampled paths from the GARCH target bank",
            "exact_conditional_enumeration_used_for_training": False,
            "exact_tree_used_only_after_solver_stopped": bool(
                str(args.innovation_law) == "gaussian"
            ),
            "exact_tree_population_used": bool(
                str(args.innovation_law) == "tree"
            ),
            "exact_tree_objective_bank_used": bool(
                str(args.innovation_law) == "tree"
            ),
            "conditional_estimator": (
                "finite nested Gaussian branches"
                if str(args.innovation_law) == "gaussian"
                else "finite nested three-point branches matching the exact tree"
            ),
            "innovation_law": str(args.innovation_law),
            "operative_r_only_fit": bool(args.operative_r_only_fit),
            "ce_network": str(args.ce_network),
            "transformer_layers": int(args.transformer_layers),
            "transformer_heads": int(args.transformer_heads),
            "ce_accuracy_gate": "held-out prediction relative to independent label-replication noise",
            "accepted_law_separate_from_adjoint_gru": True,
            "law_update": "direct log-volatility interpolation",
            "objective_acceptance": (
                "complete enumerated tree objective"
                if str(args.innovation_law) == "tree"
                else "independent selection and confirmation banks"
            ),
            "fit_paths": int(args.fit_paths),
            "validation_paths": int(args.validation_paths),
            "population_paths": int(args.population_paths),
            "fit_branches": int(args.fit_branches),
            "validation_branches": int(args.validation_branches),
            "objective_paths": int(args.objective_paths),
            "selection_banks": int(args.selection_banks),
            "confirmation_banks": int(args.confirmation_banks),
            "ce_training": asdict(fit_config),
            "underlying_exact_protocol": exact_report["protocol"],
        },
        "history": history,
        "accepted_updates": sum(bool(row["update_accepted"]) for row in history),
        "hidden_exact_evaluation": hidden,
        "current_online_mp": {
            "objective_gain_fraction_of_exact": online_gain,
            "distribution": online["summary"]["distribution"],
            "dependence": online["summary"]["dependence"],
        },
        "locked_success_criterion_passed": success,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(run_dir / "report.json", report)
    torch.save(
        {
            "final_variables": variables.detach().cpu(),
            "last_adjoint_network_state": warm_state,
        },
        run_dir / "artifacts.pt",
    )
    (run_dir / "SUMMARY.md").write_text(
        "# Masked-tree accuracy-gated nested solver\n\n"
        f"- Status: `{status}`\n"
        f"- Accepted updates: {report['accepted_updates']}\n"
        f"- Exact objective gain captured: {hidden['objective_gain_fraction_of_exact']:.2%}\n"
        f"- Current online MP: {online_gain:.2%}\n"
        f"- Distribution metrics improved: {hidden['distribution']['improved_count']}/7\n"
        f"- Dependence metrics improved: {hidden['dependence']['improved_count']}/4\n"
        f"- Locked success criterion: {'PASS' if success else 'FAIL'}\n",
        encoding="utf-8",
    )


def _run_previous(
    *,
    args: argparse.Namespace,
    problem: ProductionGarchTreeProblem,
    exact: torch.Tensor,
    exact_report: dict[str, object],
    config: GarchTreeConfig,
    run_dir: Path,
) -> None:
    model, _, _ = build_production_garch_components(
        config,
        lambda_scale=50.0,
        kappa_scale=100.0,
        hidden_dim=int(args.hidden_dim),
        seed=int(args.seed),
        device=problem.device,
        dtype=torch.float32,
    )
    target_paths = problem.target.leaf_paths.unsqueeze(-1).to(
        model.device, model.dtype
    )
    training = _training(
        seed=int(args.seed),
        steps=int(args.outer_steps) * int(args.previous_inner_steps),
    )
    nested = DiscreteMPNestedTrainingConfig(
        outer_steps=int(args.outer_steps),
        inner_steps=int(args.previous_inner_steps),
        outer_batch_size=int(args.fit_paths),
        outer_target_batch_size=int(problem.config.leaf_count),
        outer_target_estimator="nested_branches",
        outer_conditional_branches=int(args.fit_branches),
        outer_conditional_antithetic=True,
        outer_conditional_query_batch_size=int(args.query_batch_size),
        outer_population_batch_size=int(args.fit_paths),
        outer_line_search_batch_size=int(args.objective_paths),
        inner_batch_size=int(args.inner_batch_size),
        fixed_point_probe_paths=int(args.validation_paths),
        log_every_outer=1,
        outer_objective_line_search=True,
        outer_relaxation_mode="adjoint_blocks",
        outer_max_backtracks=12,
        outer_backtrack_factor=0.5,
        outer_objective_tolerance=0.0,
        outer_block_coordinate_passes=2,
        outer_block_trust_fraction=1.0,
        seed=int(args.seed),
    )
    started = time.perf_counter()
    fit = model.fit_nested(target_paths, training=training, nested=nested)
    runtime = time.perf_counter() - started
    reference_errors = garch_tree_metric_errors(
        problem,
        problem.solution(problem.reference_start()),
        swd_projections=int(args.swd_projections),
    )
    solution, replay = production_model_on_exact_tree(model, problem)
    variables = collapse_leaf_controls_to_variables(
        problem, problem.controls_for_solution(solution)
    )[0]
    objective = float(problem.objective_components_for_solution(solution)[0].item())
    reference_objective = float(problem.objective(problem.reference_start()).item())
    exact_objective = float(problem.objective(exact).item())
    errors = garch_tree_metric_errors(
        problem, solution, swd_projections=int(args.swd_projections)
    )
    closure = gap_closed(reference_errors, errors)
    hidden = {
        "objective": objective,
        "objective_gain_fraction_of_exact": (
            (reference_objective - objective) / (reference_objective - exact_objective)
        ),
        "errors": errors,
        "gap_closed": closure,
        "distribution": summarize_gap_closure(closure, GARCH_DISTRIBUTION_METRICS),
        "dependence": summarize_gap_closure(closure, GARCH_DEPENDENCE_METRICS),
        "replay": replay,
    }
    report = {
        "status": "complete",
        "arm": "previous_nested",
        "protocol": {
            "nested": asdict(nested),
            "training": asdict(training),
            "underlying_exact_protocol": exact_report["protocol"],
        },
        "history": list(fit.history),
        "hidden_exact_evaluation": hidden,
        "runtime_seconds": runtime,
    }
    write_json(run_dir / "report.json", report)
    torch.save(
        {
            "checkpoint": model.checkpoint_state(),
            "final_variables": variables.detach().cpu(),
        },
        run_dir / "artifacts.pt",
    )
    (run_dir / "SUMMARY.md").write_text(
        "# Previous nested solver on masked GARCH target\n\n"
        f"- Exact objective gain captured: {hidden['objective_gain_fraction_of_exact']:.2%}\n"
        f"- Distribution metrics improved: {hidden['distribution']['improved_count']}/7\n"
        f"- Dependence metrics improved: {hidden['dependence']['improved_count']}/4\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    run_dir = ensure_run_dir(args.run_dir.resolve())
    problem, exact, exact_report, config = _load(args)
    if args.arm == "new":
        _run_new(
            args=args,
            problem=problem,
            exact=exact,
            exact_report=exact_report,
            config=config,
            run_dir=run_dir,
        )
    else:
        _run_previous(
            args=args,
            problem=problem,
            exact=exact,
            exact_report=exact_report,
            config=config,
            run_dir=run_dir,
        )


if __name__ == "__main__":
    run(parse_args())
