from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Callable, Mapping

import torch

from deep_mkv_gen_path_dt.config import DiscreteMPTrainingConfig
from deep_mkv_gen_path_dt.controls import SpecificEntropyDiagonalControl
from deep_mkv_gen_path_dt.model import DiscreteMPModel
from path_dt_experiments.lifted_collocation_solver import (
    BlockwiseControlPerturbationPolicy,
    CausalControlPolicy,
    ScenarioCollocationBank,
    ScenarioDataContext,
    _bank_consensus_metrics,
    _control_safety,
    _directional_metrics,
    model_checkpoint_sha256,
    network_sha256,
    policy_moments_on_paths,
    rollout_policy_on_bank,
    time_block_bounds,
)


@dataclass(frozen=True)
class PollTrustRegionConfig:
    radii: tuple[float, ...] = (0.025, 0.0125, 0.00625)
    num_time_blocks: int = 3
    minimum_confirmation_merit_improvement: float = 1e-3
    maximum_objective_relative_increase: float = 5e-3
    maximum_target_kkt_relative_increase: float = 0.10
    minimum_denominator_margin: float = 1e-6
    maximum_transition_kl_at_initial_radius: float = 5e-4
    projected_kkt_gamma: float = 1.0

    def __post_init__(self) -> None:
        radii = tuple(float(value) for value in self.radii)
        if not radii or any(
            not math.isfinite(value) or value <= 0.0 for value in radii
        ):
            raise ValueError("radii must be positive and finite")
        if any(left <= right for left, right in zip(radii, radii[1:])):
            raise ValueError("radii must be strictly decreasing")
        object.__setattr__(self, "radii", radii)
        blocks = int(self.num_time_blocks)
        if blocks < 1:
            raise ValueError("num_time_blocks must be positive")
        object.__setattr__(self, "num_time_blocks", blocks)
        for name in (
            "minimum_confirmation_merit_improvement",
            "maximum_objective_relative_increase",
            "maximum_target_kkt_relative_increase",
            "minimum_denominator_margin",
            "maximum_transition_kl_at_initial_radius",
            "projected_kkt_gamma",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be nonnegative and finite")
            if name in (
                "minimum_denominator_margin",
                "maximum_transition_kl_at_initial_radius",
                "projected_kkt_gamma",
            ) and value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class AdjointRefitConfig:
    inner_batch_size: int = 128
    max_steps: int = 2_000
    eval_every: int = 50
    patience: int = 12
    minimum_relative_improvement: float = 1e-3
    seed: int = 2_608_071

    def __post_init__(self) -> None:
        for name in ("inner_batch_size", "max_steps", "eval_every", "patience"):
            value = int(getattr(self, name))
            if value < 1:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if int(self.eval_every) > int(self.max_steps):
            raise ValueError("eval_every cannot exceed max_steps")
        improvement = float(self.minimum_relative_improvement)
        if not math.isfinite(improvement) or not 0.0 < improvement < 1.0:
            raise ValueError("minimum_relative_improvement must lie in (0, 1)")
        object.__setattr__(self, "minimum_relative_improvement", improvement)
        object.__setattr__(self, "seed", int(self.seed))


@dataclass(frozen=True)
class FrozenLiftedMerit:
    denominators: dict[str, float]
    weights: dict[str, float]
    baseline_component_values: dict[str, float]
    estimator_noise_variances: dict[str, float]

    @classmethod
    def from_baseline(
        cls,
        artifacts: Mapping[str, torch.Tensor],
        *,
        p_estimator_noise_variance: float,
        r_estimator_noise_variance: float,
    ) -> FrozenLiftedMerit:
        names = {
            "p": "p_ce_residual",
            "r": "r_ce_residual",
            "alpha": "drift_consensus_residual",
            "log_sigma": "log_volatility_consensus_residual",
        }
        noise = {
            "p": float(p_estimator_noise_variance),
            "r": float(r_estimator_noise_variance),
            "alpha": 0.0,
            "log_sigma": 0.0,
        }
        if any(not math.isfinite(value) or value < 0.0 for value in noise.values()):
            raise ValueError("estimator-noise variances must be finite and nonnegative")
        baseline = {}
        denominators = {}
        weights = {}
        floor = 1e-12
        for short, artifact_name in names.items():
            values = artifacts[artifact_name].detach().double()
            mean_square = float(torch.mean(values.pow(2)).item())
            denominator = max(mean_square + noise[short], floor)
            baseline[short] = mean_square
            denominators[short] = denominator
            weights[short] = 1.0 / denominator
        return cls(
            denominators=denominators,
            weights=weights,
            baseline_component_values=baseline,
            estimator_noise_variances={"p": noise["p"], "r": noise["r"]},
        )

    def evaluate(
        self, artifacts: Mapping[str, torch.Tensor]
    ) -> dict[str, object]:
        names = {
            "p": "p_ce_residual",
            "r": "r_ce_residual",
            "alpha": "drift_consensus_residual",
            "log_sigma": "log_volatility_consensus_residual",
        }
        components = {}
        total = 0.0
        for short, artifact_name in names.items():
            mean_square = float(
                torch.mean(artifacts[artifact_name].detach().double().pow(2)).item()
            )
            weighted = mean_square * float(self.weights[short])
            components[short] = {
                "mean_square": mean_square,
                "weight": float(self.weights[short]),
                "weighted_value": weighted,
            }
            total += weighted
        return {"total": total, "components": components}


@dataclass(frozen=True)
class PollTrustRegionResult:
    report: dict[str, object]
    artifacts: dict[str, object]
    accepted_policy: BlockwiseControlPerturbationPolicy | None


def _relative_increase(candidate: float, baseline: float) -> float:
    scale = max(abs(float(baseline)), torch.finfo(torch.float64).eps)
    return (float(candidate) - float(baseline)) / scale


def _candidate_gate(
    *,
    metrics: Mapping[str, object],
    merit: Mapping[str, object],
    baseline_metrics: Mapping[str, object],
    baseline_merit: Mapping[str, object],
    radius: float,
    config: PollTrustRegionConfig,
    require_merit_improvement: bool,
) -> dict[str, object]:
    objective_increase = _relative_increase(
        float(metrics["complete_objective_value"]),
        float(baseline_metrics["complete_objective_value"]),
    )
    target_kkt_increase = _relative_increase(
        float(metrics["target_projected_kkt_residual_rms"]),
        float(baseline_metrics["target_projected_kkt_residual_rms"]),
    )
    baseline_merit_value = float(baseline_merit["total"])
    merit_improvement = (
        baseline_merit_value - float(merit["total"])
    ) / max(abs(baseline_merit_value), torch.finfo(torch.float64).eps)
    kl_limit = float(config.maximum_transition_kl_at_initial_radius) * (
        float(radius) / float(config.radii[0])
    ) ** 2
    safety = metrics["candidate_control_safety"]
    checks = {
        "merit_improves": bool(
            merit_improvement
            >= (
                float(config.minimum_confirmation_merit_improvement)
                if require_merit_improvement
                else 0.0
            )
        ),
        "objective_nonmaterial_increase": bool(
            objective_increase <= float(config.maximum_objective_relative_increase)
        ),
        "target_kkt_guardrail": bool(
            target_kkt_increase <= float(config.maximum_target_kkt_relative_increase)
        ),
        "denominator_admissible": bool(
            float(safety["specific_entropy_denominator_margin_min"])
            >= float(config.minimum_denominator_margin)
        ),
        "controls_finite": bool(float(safety["controls_finite"]) == 1.0),
        "bounds_admissible": bool(
            float(metrics["candidate_sigma_min"])
            >= float(metrics["admissible_sigma_min"]) * (1.0 - 1e-6)
            and float(metrics["candidate_sigma_max"])
            <= float(metrics["admissible_sigma_max"]) * (1.0 + 1e-6)
        ),
        "transition_kl_within_radius": bool(
            float(metrics["law_transition_kl_mean"]) <= kl_limit
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "merit_relative_improvement": merit_improvement,
        "objective_relative_increase": objective_increase,
        "target_kkt_relative_increase": target_kkt_increase,
        "transition_kl_limit": kl_limit,
    }


def select_fit_candidate(
    rows: list[dict[str, object]],
) -> dict[str, object] | None:
    """Select only among fit-gated candidates; confirmation is not consulted."""

    eligible = [row for row in rows if bool(row["fit_gate"]["passed"])]
    if not eligible:
        return None
    return min(eligible, key=lambda row: float(row["fit_merit"]["total"]))


def poll_one_law_update(
    *,
    model: DiscreteMPModel,
    law_policy: CausalControlPolicy,
    adjoint_policy: CausalControlPolicy,
    data_context: ScenarioDataContext,
    fit_bank: ScenarioCollocationBank,
    confirmation_bank: ScenarioCollocationBank,
    full_training: DiscreteMPTrainingConfig,
    query_batch_size: int,
    p_estimator_noise_variance: float,
    r_estimator_noise_variance: float,
    config: PollTrustRegionConfig,
    progress_callback: Callable[[str], None] | None = None,
) -> PollTrustRegionResult:
    """Poll finite coordinate moves and permit at most one confirmed update."""

    if not isinstance(model.control_map, SpecificEntropyDiagonalControl):
        raise TypeError("poll solver requires bounded specific entropy")
    if model.control_map.sigma_max is None:
        raise ValueError("poll solver requires a finite sigma_max")
    blocks = time_block_bounds(
        num_steps=int(model.grid.num_steps), num_blocks=int(config.num_time_blocks)
    )
    model_before = model_checkpoint_sha256(model)
    law_network = getattr(law_policy, "network", None)
    adjoint_network = getattr(adjoint_policy, "network", None)
    law_before = (
        None if not isinstance(law_network, torch.nn.Module) else network_sha256(law_network)
    )
    adjoint_before = (
        None
        if not isinstance(adjoint_network, torch.nn.Module)
        else network_sha256(adjoint_network)
    )
    baseline_evaluations = {}
    baseline_metrics = {}
    baseline_artifacts = {}
    for bank_name, bank in (("fit", fit_bank), ("confirmation", confirmation_bank)):
        evaluation = rollout_policy_on_bank(
            model=model,
            policy=law_policy,
            bank=bank,
            data_context=data_context,
            full_training=full_training,
            query_batch_size=query_batch_size,
        )
        metrics, artifacts = _directional_metrics(
            model=model,
            evaluation=evaluation,
            law_policy=law_policy,
            adjoint_policy=adjoint_policy,
            gamma=float(config.projected_kkt_gamma),
        )
        metrics["candidate_control_safety"] = evaluation.safety
        state_dim = int(model.architecture.state_dim)
        sigma = evaluation.controls[..., state_dim:]
        metrics.update(
            {
                "candidate_sigma_min": float(sigma.min().item()),
                "candidate_sigma_max": float(sigma.max().item()),
                "admissible_sigma_min": float(model.control_map.sigma_min),
                "admissible_sigma_max": float(model.control_map.sigma_max),
            }
        )
        baseline_evaluations[bank_name] = evaluation
        baseline_metrics[bank_name] = metrics
        baseline_artifacts[bank_name] = artifacts
    merit = FrozenLiftedMerit.from_baseline(
        baseline_artifacts["fit"],
        p_estimator_noise_variance=float(p_estimator_noise_variance),
        r_estimator_noise_variance=float(r_estimator_noise_variance),
    )
    baseline_merit = {
        bank: merit.evaluate(baseline_artifacts[bank])
        for bank in ("fit", "confirmation")
    }

    radius_reports = []
    poll_artifacts: dict[str, object] = {}
    accepted_policy = None
    accepted_descriptor = None
    confirmation_attempts = 0
    for radius in config.radii:
        radius_key = f"{radius:.10g}"
        rows = []
        poll_artifacts[radius_key] = {}
        for signal in ("alpha", "log_sigma"):
            for block_index in range(len(blocks)):
                for sign, side in ((-1.0, "minus"), (1.0, "plus")):
                    alpha = [0.0] * len(blocks)
                    log_sigma = [0.0] * len(blocks)
                    values = alpha if signal == "alpha" else log_sigma
                    values[block_index] = sign * float(radius)
                    candidate = BlockwiseControlPerturbationPolicy(
                        model=model,
                        law_policy=law_policy,
                        adjoint_policy=adjoint_policy,
                        blocks=blocks,
                        alpha_coefficients=tuple(alpha),
                        log_sigma_coefficients=tuple(log_sigma),
                    )
                    evaluation = rollout_policy_on_bank(
                        model=model,
                        policy=candidate,
                        bank=fit_bank,
                        data_context=data_context,
                        full_training=full_training,
                        query_batch_size=query_batch_size,
                    )
                    metrics, artifacts = _directional_metrics(
                        model=model,
                        evaluation=evaluation,
                        law_policy=law_policy,
                        adjoint_policy=adjoint_policy,
                        gamma=float(config.projected_kkt_gamma),
                    )
                    metrics["candidate_control_safety"] = evaluation.safety
                    state_dim = int(model.architecture.state_dim)
                    sigma = evaluation.controls[..., state_dim:]
                    metrics.update(
                        {
                            "candidate_sigma_min": float(sigma.min().item()),
                            "candidate_sigma_max": float(sigma.max().item()),
                            "admissible_sigma_min": float(model.control_map.sigma_min),
                            "admissible_sigma_max": float(model.control_map.sigma_max),
                        }
                    )
                    candidate_merit = merit.evaluate(artifacts)
                    fit_gate = _candidate_gate(
                        metrics=metrics,
                        merit=candidate_merit,
                        baseline_metrics=baseline_metrics["fit"],
                        baseline_merit=baseline_merit["fit"],
                        radius=float(radius),
                        config=config,
                        require_merit_improvement=False,
                    )
                    coordinate = f"{signal}_block_{block_index}_{side}"
                    row = {
                        "coordinate": coordinate,
                        "signal": signal,
                        "block_index": int(block_index),
                        "side": side,
                        "coefficient": sign * float(radius),
                        "fit_merit": candidate_merit,
                        "fit_metrics": metrics,
                        "fit_gate": fit_gate,
                    }
                    rows.append(row)
                    poll_artifacts[radius_key][coordinate] = {
                        "fit": artifacts,
                    }
                    if progress_callback is not None:
                        progress_callback(f"poll radius={radius_key} {coordinate}")
        selected = select_fit_candidate(rows)
        radius_report: dict[str, object] = {
            "radius": float(radius),
            "fit_candidates": rows,
            "fit_selected_coordinate": (
                None if selected is None else selected["coordinate"]
            ),
            "confirmation_evaluated": False,
            "accepted": False,
        }
        if selected is not None:
            confirmation_attempts += 1
            signal = str(selected["signal"])
            block_index = int(selected["block_index"])
            coefficient = float(selected["coefficient"])
            alpha = [0.0] * len(blocks)
            log_sigma = [0.0] * len(blocks)
            values = alpha if signal == "alpha" else log_sigma
            values[block_index] = coefficient
            candidate = BlockwiseControlPerturbationPolicy(
                model=model,
                law_policy=law_policy,
                adjoint_policy=adjoint_policy,
                blocks=blocks,
                alpha_coefficients=tuple(alpha),
                log_sigma_coefficients=tuple(log_sigma),
            )
            confirmation_evaluation = rollout_policy_on_bank(
                model=model,
                policy=candidate,
                bank=confirmation_bank,
                data_context=data_context,
                full_training=full_training,
                query_batch_size=query_batch_size,
            )
            confirmation_metrics, confirmation_artifacts = _directional_metrics(
                model=model,
                evaluation=confirmation_evaluation,
                law_policy=law_policy,
                adjoint_policy=adjoint_policy,
                gamma=float(config.projected_kkt_gamma),
            )
            confirmation_metrics["candidate_control_safety"] = (
                confirmation_evaluation.safety
            )
            state_dim = int(model.architecture.state_dim)
            sigma = confirmation_evaluation.controls[..., state_dim:]
            confirmation_metrics.update(
                {
                    "candidate_sigma_min": float(sigma.min().item()),
                    "candidate_sigma_max": float(sigma.max().item()),
                    "admissible_sigma_min": float(model.control_map.sigma_min),
                    "admissible_sigma_max": float(model.control_map.sigma_max),
                }
            )
            confirmation_merit = merit.evaluate(confirmation_artifacts)
            confirmation_gate = _candidate_gate(
                metrics=confirmation_metrics,
                merit=confirmation_merit,
                baseline_metrics=baseline_metrics["confirmation"],
                baseline_merit=baseline_merit["confirmation"],
                radius=float(radius),
                config=config,
                require_merit_improvement=True,
            )
            radius_report.update(
                {
                    "confirmation_evaluated": True,
                    "confirmation_merit": confirmation_merit,
                    "confirmation_metrics": confirmation_metrics,
                    "confirmation_gate": confirmation_gate,
                    "accepted": bool(confirmation_gate["passed"]),
                }
            )
            poll_artifacts[radius_key][str(selected["coordinate"])][
                "confirmation"
            ] = confirmation_artifacts
            if bool(confirmation_gate["passed"]):
                accepted_policy = candidate
                accepted_descriptor = {
                    "radius": float(radius),
                    "coordinate": selected["coordinate"],
                    "signal": signal,
                    "block_index": block_index,
                    "coefficient": coefficient,
                    "blocks": [list(block) for block in blocks],
                    "alpha_coefficients": alpha,
                    "log_sigma_coefficients": log_sigma,
                    "fit_merit": selected["fit_merit"],
                    "confirmation_merit": confirmation_merit,
                    "confirmation_gate": confirmation_gate,
                }
        radius_reports.append(radius_report)
        if accepted_policy is not None:
            break

    model_after = model_checkpoint_sha256(model)
    law_after = (
        None if not isinstance(law_network, torch.nn.Module) else network_sha256(law_network)
    )
    adjoint_after = (
        None
        if not isinstance(adjoint_network, torch.nn.Module)
        else network_sha256(adjoint_network)
    )
    nonmutation = {
        "model_unchanged": model_before == model_after,
        "law_policy_unchanged": law_before == law_after,
        "adjoint_policy_unchanged": adjoint_before == adjoint_after,
    }
    if not all(nonmutation.values()):
        raise RuntimeError("polling mutated a frozen model or policy")
    return PollTrustRegionResult(
        report={
            "protocol": {
                "maximum_accepted_updates": 1,
                "accepted_updates": int(accepted_policy is not None),
                "fit_bank_selects_candidate": True,
                "confirmation_bank_never_selects_candidate": True,
                "radii_strictly_decrease_to_floor": True,
                "no_radius_below_floor": True,
                "target_kkt_is_guardrail_not_merit": True,
                "merit_weights_frozen_before_polling": True,
            },
            "config": dict(config.__dict__),
            "blocks": [list(block) for block in blocks],
            "merit": {
                "denominators": merit.denominators,
                "weights": merit.weights,
                "baseline_component_values": merit.baseline_component_values,
                "estimator_noise_variances": merit.estimator_noise_variances,
            },
            "baseline": {
                "fit_metrics": baseline_metrics["fit"],
                "fit_merit": baseline_merit["fit"],
                "confirmation_metrics": baseline_metrics["confirmation"],
                "confirmation_merit": baseline_merit["confirmation"],
            },
            "radii": radius_reports,
            "confirmation_attempts": confirmation_attempts,
            "accepted": accepted_policy is not None,
            "accepted_candidate": accepted_descriptor,
            "stopped_at_radius_floor_without_update": bool(
                accepted_policy is None
                and len(radius_reports) == len(config.radii)
                and float(radius_reports[-1]["radius"]) == float(config.radii[-1])
            ),
            "nonmutation": nonmutation,
            "fit_bank_fingerprint": fit_bank.fingerprint,
            "confirmation_bank_fingerprint": confirmation_bank.fingerprint,
        },
        artifacts={
            "baseline": baseline_artifacts,
            "poll": poll_artifacts,
        },
        accepted_policy=accepted_policy,
    )


@dataclass(frozen=True)
class AdjointRefitResult:
    network: torch.nn.Module
    report: dict[str, object]
    state_dict: dict[str, torch.Tensor]


@dataclass(frozen=True)
class AdjointOutputInterpolationResult:
    report: dict[str, object]
    artifacts: dict[str, torch.Tensor]


def interpolate_adjoint_outputs_on_evaluation(
    *,
    model: DiscreteMPModel,
    evaluation,
    old_adjoint_policy: CausalControlPolicy,
    refitted_adjoint_policy: CausalControlPolicy,
    tau: float,
    merit: FrozenLiftedMerit,
) -> AdjointOutputInterpolationResult:
    """Interpolate denormalized P/R outputs on one fixed generated law."""

    fraction = float(tau)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("tau must lie in [0, 1]")
    if not isinstance(model.control_map, SpecificEntropyDiagonalControl):
        raise TypeError("adjoint output interpolation requires specific entropy")
    with torch.no_grad():
        old_p, old_r, _ = policy_moments_on_paths(
            model=model,
            policy=old_adjoint_policy,
            paths=evaluation.paths,
        )
        new_p, new_r, _ = policy_moments_on_paths(
            model=model,
            policy=refitted_adjoint_policy,
            paths=evaluation.paths,
        )
        p = (1.0 - fraction) * old_p + fraction * new_p
        r = (1.0 - fraction) * old_r + fraction * new_r
        controls = model.control_map.controls_from_moments(
            paths=evaluation.paths,
            expected_adjoint_next=p,
            expected_adjoint_noise_next=r,
        )
    state_dim = int(model.architecture.state_dim)
    residuals = {
        "p_ce_residual": (p - evaluation.target_p).detach(),
        "r_ce_residual": (r - evaluation.target_r).detach(),
        "drift_consensus_residual": (
            evaluation.controls[..., :state_dim] - controls[..., :state_dim]
        ).detach(),
        "log_volatility_consensus_residual": (
            torch.log(evaluation.controls[..., state_dim:])
            - torch.log(controls[..., state_dim:])
        ).detach(),
    }
    merit_row = merit.evaluate(residuals)
    transition_kl = model.control_map.transition_kernel_kl(
        controls=controls,
        current_controls=evaluation.controls,
    ).detach()
    safety = _control_safety(
        model=model,
        paths=evaluation.paths,
        controls=controls,
        moment_r=r,
    )
    report = {
        "tau": fraction,
        "merit": merit_row,
        "p_ce_residual_rms": float(
            torch.mean(residuals["p_ce_residual"].double().pow(2)).sqrt().item()
        ),
        "r_ce_residual_rms": float(
            torch.mean(residuals["r_ce_residual"].double().pow(2)).sqrt().item()
        ),
        "drift_consensus_rms": float(
            torch.mean(residuals["drift_consensus_residual"].double().pow(2))
            .sqrt()
            .item()
        ),
        "log_volatility_consensus_rms": float(
            torch.mean(
                residuals["log_volatility_consensus_residual"].double().pow(2)
            )
            .sqrt()
            .item()
        ),
        "transition_kl_mean": float(transition_kl.mean().item()),
        "transition_kl_q95": float(
            torch.quantile(transition_kl.reshape(-1), 0.95).item()
        ),
        "control_safety": safety,
        "objective_fixed_along_tau": float(
            evaluation.objective["complete_objective_value"]
        ),
    }
    return AdjointOutputInterpolationResult(
        report=report,
        artifacts={
            **{name: value.cpu() for name, value in residuals.items()},
            "interpolated_p": p.detach().cpu(),
            "interpolated_r": r.detach().cpu(),
            "interpolated_controls": controls.detach().cpu(),
            "transition_kl": transition_kl.cpu(),
        },
    )


def refit_adjoint_network(
    *,
    model: DiscreteMPModel,
    initial_network: torch.nn.Module,
    fit_paths: torch.Tensor,
    fit_target_p: torch.Tensor,
    fit_target_r: torch.Tensor,
    validation_paths: torch.Tensor,
    validation_target_p: torch.Tensor,
    validation_target_r: torch.Tensor,
    training: DiscreteMPTrainingConfig,
    config: AdjointRefitConfig,
    progress_callback: Callable[[str], None] | None = None,
) -> AdjointRefitResult:
    """Refit a separate adjoint network with the unchanged normalized P/R loss."""

    if int(config.inner_batch_size) > int(fit_paths.shape[0]):
        raise ValueError("inner_batch_size cannot exceed fit path count")
    network = copy.deepcopy(initial_network).to(device=model.device, dtype=model.dtype)
    initial_hash = network_sha256(initial_network)
    fit_p = model._normalize_adjoint_target(
        fit_target_p.to(model.device), noise_adjoint=False
    ).detach()
    fit_r = model._normalize_adjoint_target(
        fit_target_r.to(model.device), noise_adjoint=True
    ).detach()
    validation_p = model._normalize_adjoint_target(
        validation_target_p.to(model.device), noise_adjoint=False
    ).detach()
    validation_r = model._normalize_adjoint_target(
        validation_target_r.to(model.device), noise_adjoint=True
    ).detach()
    fit_paths = fit_paths.to(device=model.device, dtype=model.dtype)
    validation_paths = validation_paths.to(device=model.device, dtype=model.dtype)

    def losses(
        paths: torch.Tensor,
        target_p: torch.Tensor,
        target_r: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if indices is not None:
            paths = paths.index_select(0, indices)
            target_p = target_p.index_select(0, indices)
            target_r = target_r.index_select(0, indices)
        prediction = network(paths)
        p_loss = torch.mean((prediction.expected_adjoint_next - target_p).pow(2))
        r_loss = torch.mean(
            (prediction.expected_adjoint_noise_next - target_r).pow(2)
        )
        total = (
            float(training.adjoint_weight) * p_loss
            + float(training.adjoint_noise_weight) * r_loss
        )
        return total, p_loss, r_loss

    def evaluate(step: int) -> dict[str, object]:
        network.eval()
        with torch.no_grad():
            fit_total, fit_p_loss, fit_r_loss = losses(
                fit_paths, fit_p, fit_r
            )
            validation_total, validation_p_loss, validation_r_loss = losses(
                validation_paths, validation_p, validation_r
            )
        return {
            "step": int(step),
            "fit": {
                "total": float(fit_total.item()),
                "p": float(fit_p_loss.item()),
                "r": float(fit_r_loss.item()),
            },
            "validation": {
                "total": float(validation_total.item()),
                "p": float(validation_p_loss.item()),
                "r": float(validation_r_loss.item()),
            },
        }

    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=float(training.lr),
        weight_decay=float(training.weight_decay),
    )
    trajectory = [evaluate(0)]
    best_loss = float(trajectory[0]["validation"]["total"])
    meaningful_best = best_loss
    best_step = 0
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in network.state_dict().items()
    }
    patience = 0
    generator = torch.Generator(device="cpu").manual_seed(int(config.seed))
    gradient_norms = []
    for step in range(1, int(config.max_steps) + 1):
        indices = model._sample_indices(
            size=int(fit_paths.shape[0]),
            count=int(config.inner_batch_size),
            generator=generator,
        ).to(model.device)
        network.train()
        loss, _, _ = losses(fit_paths, fit_p, fit_r, indices)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        squared = torch.zeros((), dtype=torch.float64, device=model.device)
        for parameter in network.parameters():
            if parameter.grad is not None:
                squared = squared + parameter.grad.detach().double().pow(2).sum()
        gradient_norm = float(torch.sqrt(squared).item())
        if not math.isfinite(gradient_norm):
            raise RuntimeError("adjoint refit gradient became nonfinite")
        gradient_norms.append(gradient_norm)
        optimizer.step()
        if not all(
            bool(torch.isfinite(parameter).all().item())
            for parameter in network.parameters()
        ):
            raise RuntimeError("adjoint refit parameter became nonfinite")
        if step % int(config.eval_every) == 0 or step == int(config.max_steps):
            row = evaluate(step)
            trajectory.append(row)
            current = float(row["validation"]["total"])
            if current < best_loss:
                best_loss = current
                best_step = step
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in network.state_dict().items()
                }
            if current < meaningful_best * (
                1.0 - float(config.minimum_relative_improvement)
            ):
                meaningful_best = current
                patience = 0
            else:
                patience += 1
            if progress_callback is not None:
                progress_callback(
                    f"adjoint refit step={step} validation={current:.6g}"
                )
            if patience >= int(config.patience):
                break
    network.load_state_dict(best_state)
    network.eval()
    selected = evaluate(best_step)
    if network_sha256(initial_network) != initial_hash:
        raise RuntimeError("adjoint refit mutated its initialization network")
    gradient_tensor = torch.tensor(gradient_norms, dtype=torch.float64)
    return AdjointRefitResult(
        network=network,
        report={
            "protocol": {
                "loss": "unchanged_normalized_P_R_regression",
                "optimizer_moments_reset": True,
                "law_policy_frozen": True,
                "initial_adjoint_network_unchanged": True,
                "gradient_clipping": False,
            },
            "config": dict(config.__dict__),
            "initial": trajectory[0],
            "selected": selected,
            "best_step": best_step,
            "terminal_step": int(trajectory[-1]["step"]),
            "trajectory": trajectory,
            "gradient_norm_mean": float(gradient_tensor.mean().item()),
            "gradient_norm_max": float(gradient_tensor.max().item()),
        },
        state_dict=best_state,
    )


def evaluate_complete_lifted_cycle(
    *,
    model: DiscreteMPModel,
    base_law_policy: CausalControlPolicy,
    accepted_law_policy: CausalControlPolicy,
    pre_refit_adjoint_policy: CausalControlPolicy,
    post_refit_adjoint_policy: CausalControlPolicy,
    data_context: ScenarioDataContext,
    fit_bank: ScenarioCollocationBank,
    confirmation_bank: ScenarioCollocationBank,
    full_training: DiscreteMPTrainingConfig,
    query_batch_size: int,
    merit: FrozenLiftedMerit,
) -> tuple[dict[str, object], dict[str, object]]:
    """Compare center, moved law, and moved-law/refitted-adjoint on fresh banks."""

    report: dict[str, object] = {}
    artifacts: dict[str, object] = {}
    for bank_name, bank in (("fit", fit_bank), ("confirmation", confirmation_bank)):
        base_evaluation = rollout_policy_on_bank(
            model=model,
            policy=base_law_policy,
            bank=bank,
            data_context=data_context,
            full_training=full_training,
            query_batch_size=query_batch_size,
        )
        moved_evaluation = rollout_policy_on_bank(
            model=model,
            policy=accepted_law_policy,
            bank=bank,
            data_context=data_context,
            full_training=full_training,
            query_batch_size=query_batch_size,
        )
        bank_report = {}
        bank_artifacts = {}
        for label, evaluation, adjoint in (
            ("center_pre_refit", base_evaluation, pre_refit_adjoint_policy),
            ("moved_law_pre_refit", moved_evaluation, pre_refit_adjoint_policy),
            ("moved_law_post_refit", moved_evaluation, post_refit_adjoint_policy),
        ):
            metrics, residuals = _bank_consensus_metrics(
                model=model,
                evaluation=evaluation,
                adjoint_policy=adjoint,
            )
            bank_report[label] = {
                "metrics": metrics,
                "merit": merit.evaluate(residuals),
            }
            bank_artifacts[label] = residuals
        before = bank_report["center_pre_refit"]
        after = bank_report["moved_law_post_refit"]
        component_improvements = {}
        for component in ("p", "r", "alpha", "log_sigma"):
            initial = float(before["merit"]["components"][component]["mean_square"])
            terminal = float(after["merit"]["components"][component]["mean_square"])
            component_improvements[component] = (
                initial - terminal
            ) / max(abs(initial), torch.finfo(torch.float64).eps)
        initial_merit = float(before["merit"]["total"])
        terminal_merit = float(after["merit"]["total"])
        bank_report["complete_cycle"] = {
            "merit_relative_improvement": (
                initial_merit - terminal_merit
            ) / max(abs(initial_merit), torch.finfo(torch.float64).eps),
            "component_relative_improvements": component_improvements,
            "ce_both_improve": bool(
                component_improvements["p"] > 0.0
                and component_improvements["r"] > 0.0
            ),
            "consensus_both_improve": bool(
                component_improvements["alpha"] > 0.0
                and component_improvements["log_sigma"] > 0.0
            ),
        }
        report[bank_name] = bank_report
        artifacts[bank_name] = bank_artifacts
    report["cycle_success"] = bool(
        report["confirmation"]["complete_cycle"]["merit_relative_improvement"]
        > 0.0
        and report["confirmation"]["complete_cycle"]["ce_both_improve"]
        and report["confirmation"]["complete_cycle"]["consensus_both_improve"]
    )
    return report, artifacts


__all__ = [
    "AdjointRefitConfig",
    "AdjointRefitResult",
    "AdjointOutputInterpolationResult",
    "FrozenLiftedMerit",
    "PollTrustRegionConfig",
    "PollTrustRegionResult",
    "evaluate_complete_lifted_cycle",
    "interpolate_adjoint_outputs_on_evaluation",
    "poll_one_law_update",
    "refit_adjoint_network",
    "select_fit_candidate",
]
