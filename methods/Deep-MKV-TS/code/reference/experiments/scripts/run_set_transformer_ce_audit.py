#!/usr/bin/env python3
"""Compare branch-mean and Set-Transformer CE teachers on hidden GARCH truth."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from deep_mkv_gen_path_dt import DiscreteMPTrainingConfig  # noqa: E402
from path_dt_experiments.accuracy_gated_nested import (  # noqa: E402
    FrozenCETrainingConfig,
    ReferenceControlPolicy,
    fit_frozen_sampled_ce,
    prediction_metrics,
)
from path_dt_experiments.garch_tree_oracle import GarchTreeConfig  # noqa: E402
from path_dt_experiments.garch_tree_production import (  # noqa: E402
    ProductionGarchTreeProblem,
    build_production_garch_components,
    enumerated_garch_noise,
    exact_garch_tree_mp_response,
)
from path_dt_experiments.garch_tree_sampling import (  # noqa: E402
    pool_garch_tree_prefix_targets,
    replay_garch_tree_branch_noise,
)
from path_dt_experiments.lifted_collocation_solver import (  # noqa: E402
    ScenarioCollocationBank,
    ScenarioDataContext,
    build_scenario_collocation_bank,
    rollout_causal_policy,
    rollout_policy_on_bank,
)
from path_dt_experiments.runners import ensure_run_dir, write_json  # noqa: E402
from path_dt_experiments.set_transformer_ce import (  # noqa: E402
    SetTransformerCEConfig,
    fit_set_transformer_ce,
    predict_set_transformer_ce,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--branches", type=int, default=128)
    parser.add_argument("--query-batch-size", type=int, default=131072)
    parser.add_argument("--set-steps", type=int, default=1500)
    parser.add_argument("--set-batch-size", type=int, default=64)
    parser.add_argument("--set-hidden-dim", type=int, default=64)
    parser.add_argument("--set-heads", type=int, default=4)
    parser.add_argument("--set-inducing-points", type=int, default=16)
    parser.add_argument("--gru-steps", type=int, default=3000)
    parser.add_argument("--gru-hidden-dim", type=int, default=96)
    return parser.parse_args()


def load_problem(args: argparse.Namespace) -> tuple[ProductionGarchTreeProblem, GarchTreeConfig]:
    report = json.loads((args.exact_run_dir / "exact_report.json").read_text())
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
    return (
        ProductionGarchTreeProblem(
            config,
            lambda_scale=float(protocol["lambda_scale"]),
            kappa_scale=float(protocol["kappa_scale"]),
            device=torch.device(args.device),
            dtype=torch.float64,
        ),
        config,
    )


def training_config(seed: int) -> DiscreteMPTrainingConfig:
    return DiscreteMPTrainingConfig(
        batch_size=256,
        target_batch_size=256,
        num_steps=3000,
        lr=2e-3,
        weight_decay=1e-5,
        lambda_scale=50.0,
        adjoint_weight=0.0,
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


def complete_bank(
    *,
    model,
    context: ScenarioDataContext,
    config: GarchTreeConfig,
    branches: int,
    seed: int,
    label: str,
) -> ScenarioCollocationBank:
    bank = build_scenario_collocation_bank(
        model=model,
        context=context,
        path_count=int(config.leaf_count),
        target_count=int(config.leaf_count),
        num_branches=int(branches),
        antithetic=False,
        beta=1.0,
        seed=int(seed),
        label=label,
    )
    return replace(
        bank,
        initial_indices=torch.arange(int(config.leaf_count), dtype=torch.long),
        target_indices=torch.arange(int(config.leaf_count), dtype=torch.long),
        base_noise=enumerated_garch_noise(
            config, device=torch.device("cpu"), dtype=bank.base_noise.dtype
        ),
        branch_seed=int(seed) + 10_000,
        antithetic=False,
    )


def collect_law(
    *,
    model,
    policy,
    bank: ScenarioCollocationBank,
    context: ScenarioDataContext,
    config: GarchTreeConfig,
    training: DiscreteMPTrainingConfig,
    population_paths: torch.Tensor,
    query_batch_size: int,
):
    rows: dict[int, torch.Tensor] = {}

    def observer(
        step: int,
        prefix: torch.Tensor,
        branch_p: torch.Tensor,
        branch_r: torch.Tensor,
    ) -> None:
        del prefix, branch_p
        if int(step) in rows:
            raise RuntimeError("branch observer received the same step twice")
        rows[int(step)] = branch_r.detach().clone()

    law = rollout_policy_on_bank(
        model=model,
        policy=policy,
        bank=bank,
        data_context=context,
        full_training=training,
        query_batch_size=int(query_batch_size),
        population_paths=population_paths,
        branch_noise_provider=lambda step: replay_garch_tree_branch_noise(
            config=config,
            bank=bank,
            base_noise=bank.base_noise,
            step_index=int(step),
        ),
        branch_target_observer=observer,
    )
    if tuple(sorted(rows)) != tuple(range(int(config.horizon))):
        raise RuntimeError("branch observer did not receive every decision time")
    return law, torch.stack([rows[index] for index in range(int(config.horizon))], dim=1)


def exact_gru_metrics(fit, *, paths: torch.Tensor, target_r: torch.Tensor) -> dict[str, float]:
    fit.network.eval()
    with torch.no_grad():
        normalized = fit.network(paths)
        prediction = normalized.expected_adjoint_noise_next * fit.r_scale.unsqueeze(0)
    return prediction_metrics(prediction, target_r, prefix="r")


def main() -> None:
    args = parse_args()
    run_dir = ensure_run_dir(args.run_dir)
    problem, config = load_problem(args)
    model, _, _ = build_production_garch_components(
        config,
        lambda_scale=float(problem.lambda_scale),
        kappa_scale=float(problem.kappa_scale),
        hidden_dim=int(args.gru_hidden_dim),
        seed=int(args.seed),
        device=problem.device,
        dtype=torch.float32,
    )
    target_paths = problem.target.leaf_paths.unsqueeze(-1).to(model.device, model.dtype)
    context = ScenarioDataContext(
        initial_state_pool=target_paths,
        target_path_pool=target_paths,
        dataset_id="set-transformer-garch-ce-audit",
    )
    policy = ReferenceControlPolicy(model)
    noise = enumerated_garch_noise(config, device=model.device, dtype=model.dtype)
    x0 = torch.zeros(int(config.leaf_count), 1, device=model.device, dtype=model.dtype)
    with torch.no_grad():
        population = rollout_causal_policy(
            model=model, policy=policy, x0=x0, noise=noise
        ).paths.detach()
    training = training_config(int(args.seed))
    banks = [
        complete_bank(
            model=model,
            context=context,
            config=config,
            branches=int(args.branches),
            seed=int(args.seed) + 1000 * (index + 1),
            label=f"ce_replication_{index}",
        )
        for index in range(4)
    ]
    laws = []
    raw_r = []
    for bank in banks:
        law, branch_r = collect_law(
            model=model,
            policy=policy,
            bank=bank,
            context=context,
            config=config,
            training=training,
            population_paths=population,
            query_batch_size=int(args.query_batch_size),
        )
        laws.append(law)
        raw_r.append(branch_r)

    def pooled(value: torch.Tensor, bank: ScenarioCollocationBank) -> torch.Tensor:
        return pool_garch_tree_prefix_targets(
            config,
            base_noise=bank.base_noise.to(value),
            targets=value,
        )

    branch_means = [pooled(values.mean(dim=2), bank) for values, bank in zip(raw_r, banks, strict=True)]
    exact_response = exact_garch_tree_mp_response(problem, problem.reference_start())
    exact_r = exact_response.expected_adjoint_noise_next.to(model.device, model.dtype)
    exact_p = exact_response.expected_adjoint_next.to(model.device, model.dtype)

    set_config = SetTransformerCEConfig(
        hidden_dim=int(args.set_hidden_dim),
        num_heads=int(args.set_heads),
        num_inducing_points=int(args.set_inducing_points),
        steps=int(args.set_steps),
        batch_size=int(args.set_batch_size),
        seed=int(args.seed) + 71,
    )
    set_fit = fit_set_transformer_ce(
        fit_paths=laws[0].paths,
        fit_branches=raw_r[0],
        fit_independent_target=branch_means[1],
        validation_paths=laws[2].paths,
        validation_branches=raw_r[2],
        validation_independent_target=branch_means[3],
        config=set_config,
    )
    set_fit_teacher = pooled(
        predict_set_transformer_ce(
            set_fit,
            paths=laws[0].paths,
            branches=raw_r[0],
            batch_size=int(args.set_batch_size),
        ),
        banks[0],
    )
    set_validation_teacher = pooled(
        predict_set_transformer_ce(
            set_fit,
            paths=laws[2].paths,
            branches=raw_r[2],
            batch_size=int(args.set_batch_size),
        ),
        banks[2],
    )
    teacher_metrics = {
        "branch_mean_fit": prediction_metrics(branch_means[0], exact_r, prefix="r"),
        "branch_mean_validation": prediction_metrics(branch_means[2], exact_r, prefix="r"),
        "set_transformer_fit": prediction_metrics(set_fit_teacher, exact_r, prefix="r"),
        "set_transformer_validation": prediction_metrics(set_validation_teacher, exact_r, prefix="r"),
        "independent_branch_mean_validation": prediction_metrics(branch_means[3], exact_r, prefix="r"),
    }
    teacher_gate = bool(
        teacher_metrics["set_transformer_validation"]["r_relative_error"]
        < teacher_metrics["branch_mean_validation"]["r_relative_error"]
    )

    gru_config = FrozenCETrainingConfig(
        max_steps=int(args.gru_steps),
        minimum_steps=min(200, int(args.gru_steps)),
        batch_size=256,
        eval_every=100,
        lr=2e-3,
        weight_decay=1e-5,
        grad_clip_norm=5.0,
        minimum_scale=1e-3,
        validation_relative_tolerance=0.3,
        p_loss_weight=0.0,
        r_loss_weight=1.0,
        seed=int(args.seed) + 91,
    )
    baseline_gru = fit_frozen_sampled_ce(
        model=model,
        fit_paths=laws[0].paths,
        fit_target_p=exact_p,
        fit_target_r=branch_means[0],
        validation_paths=laws[2].paths,
        validation_target_p=exact_p,
        validation_target_r=0.5 * (branch_means[2] + branch_means[3]),
        config=gru_config,
    )
    set_gru = fit_frozen_sampled_ce(
        model=model,
        fit_paths=laws[0].paths,
        fit_target_p=exact_p,
        fit_target_r=set_fit_teacher,
        validation_paths=laws[2].paths,
        validation_target_p=exact_p,
        validation_target_r=set_validation_teacher,
        config=gru_config,
    )
    gru_metrics = {
        "branch_mean_teacher": exact_gru_metrics(
            baseline_gru, paths=population, target_r=exact_r
        ),
        "set_transformer_teacher": exact_gru_metrics(
            set_gru, paths=population, target_r=exact_r
        ),
    }
    deployment_gate = bool(
        teacher_gate
        and gru_metrics["set_transformer_teacher"]["r_relative_error"]
        < gru_metrics["branch_mean_teacher"]["r_relative_error"]
    )
    report = {
        "protocol": {
            "exact_conditional_expectations_used_for_training": False,
            "exact_conditional_expectations_used_only_for_hidden_evaluation": True,
            "controlled_comparison": "CE teacher only; deployable GRU unchanged",
            "branch_count_per_replication": int(args.branches),
            "independent_replications": 4,
            "set_transformer": asdict(set_config),
            "gru": asdict(gru_config),
            "bank_fingerprints": [bank.fingerprint for bank in banks],
        },
        "teacher_metrics_against_hidden_exact_ce": teacher_metrics,
        "set_transformer_selected_step": int(set_fit.selected_step),
        "set_transformer_history": list(set_fit.history),
        "teacher_gate_passed": teacher_gate,
        "unchanged_gru_metrics_against_hidden_exact_ce": gru_metrics,
        "complete_deployment_gate_passed": deployment_gate,
    }
    write_json(run_dir / "report.json", report)
    torch.save(
        {
            "set_transformer_state": set_fit.network.state_dict(),
            "set_transformer_scale": set_fit.scale.cpu(),
            "baseline_gru_state": baseline_gru.network.state_dict(),
            "set_teacher_gru_state": set_gru.network.state_dict(),
        },
        run_dir / "artifacts.pt",
    )
    (run_dir / "SUMMARY.md").write_text(
        "# Set Transformer conditional-expectation audit\n\n"
        f"- Branch-mean teacher hidden error: {teacher_metrics['branch_mean_validation']['r_relative_error']:.2%}\n"
        f"- Set Transformer teacher hidden error: {teacher_metrics['set_transformer_validation']['r_relative_error']:.2%}\n"
        f"- Branch-mean + unchanged GRU hidden error: {gru_metrics['branch_mean_teacher']['r_relative_error']:.2%}\n"
        f"- Set Transformer + unchanged GRU hidden error: {gru_metrics['set_transformer_teacher']['r_relative_error']:.2%}\n"
        f"- Teacher gate: {'PASS' if teacher_gate else 'FAIL'}\n"
        f"- Complete deployment gate: {'PASS' if deployment_gate else 'FAIL'}\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
