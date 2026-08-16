from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import math

import torch

from path_dt_experiments.garch_tree_production import (
    ProductionGarchTreeProblem,
    build_production_garch_components,
    exact_garch_tree_mp_response,
)


@dataclass(frozen=True)
class FrozenExactCEFitConfig:
    steps: int = 3000
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-5
    grad_clip_norm: float = 5.0
    log_every: int = 100
    hidden_dim: int = 96
    seed: int = 0
    minimum_scale: float = 1e-3
    minimum_steps: int = 0
    early_stop_relative_error: float | None = None


@dataclass(frozen=True)
class FrozenExactCEFitResult:
    model: object
    paths: torch.Tensor
    target_p: torch.Tensor
    target_r: torch.Tensor
    prediction_p: torch.Tensor
    prediction_r: torch.Tensor
    p_scale: torch.Tensor
    r_scale: torch.Tensor
    history: tuple[dict[str, float], ...]
    selected_step: int
    selected_loss: float
    fingerprint: str


def tensor_fingerprint(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().contiguous().cpu()
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def objective_gated_exact_ce_update(
    problem: ProductionGarchTreeProblem,
    variables: torch.Tensor,
    proposal: torch.Tensor,
    *,
    max_backtracks: int = 24,
    objective_atol: float = 1e-12,
    objective_rtol: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Largest dyadic log-volatility move passing strict objective decrease."""

    current = variables.detach().to(problem.device, problem.dtype)
    proposed = proposal.detach().to(problem.device, problem.dtype)
    objective_before = float(problem.objective(current).detach().item())
    raw_objective = float(problem.objective(proposed).detach().item())
    required = max(
        float(objective_atol),
        float(objective_rtol) * max(abs(objective_before), 1.0),
    )
    direction = proposed - current
    for backtrack in range(int(max_backtracks)):
        relaxation = 2.0 ** (-backtrack)
        candidate = current + relaxation * direction
        candidate_objective = float(problem.objective(candidate).detach().item())
        if (
            math.isfinite(candidate_objective)
            and candidate_objective <= objective_before - required
        ):
            return candidate.detach(), {
                "accepted": 1.0,
                "relaxation": float(relaxation),
                "objective_before": objective_before,
                "raw_proposal_objective": raw_objective,
                "objective_after": candidate_objective,
                "objective_improvement": objective_before - candidate_objective,
            }
    return current.detach().clone(), {
        "accepted": 0.0,
        "relaxation": 0.0,
        "objective_before": objective_before,
        "raw_proposal_objective": raw_objective,
        "objective_after": objective_before,
        "objective_improvement": 0.0,
    }


def reconstruct_exact_ce_law(
    problem: ProductionGarchTreeProblem,
    *,
    accepted_updates: int,
) -> tuple[torch.Tensor, tuple[dict[str, float], ...]]:
    """Reconstruct a deterministic point on the locked exact-CE trajectory."""

    if int(accepted_updates) < 0:
        raise ValueError("accepted_updates must be nonnegative")
    current = problem.reference_start()
    history = []
    for outer in range(int(accepted_updates)):
        response = exact_garch_tree_mp_response(problem, current)
        candidate, metrics = objective_gated_exact_ce_update(
            problem,
            current,
            response.variables,
        )
        if metrics["accepted"] != 1.0:
            raise RuntimeError("the locked exact-CE trajectory rejected unexpectedly")
        history.append({"outer_step": float(outer), **metrics})
        current = candidate
    return current.detach(), tuple(history)


def _relative_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    prefix: str,
) -> dict[str, float]:
    prediction64 = prediction.detach().double()
    target64 = target.detach().double()
    error = prediction64 - target64
    target_energy = target64.pow(2).sum().clamp_min(1e-30)
    error_energy = error.pow(2).sum()
    centered_target = target64 - target64.mean(dim=0, keepdim=True)
    centered_prediction = prediction64 - prediction64.mean(dim=0, keepdim=True)
    baseline_energy = centered_target.pow(2).sum().clamp_min(1e-30)
    cosine_denominator = (
        torch.linalg.vector_norm(prediction64.reshape(-1))
        * torch.linalg.vector_norm(target64.reshape(-1))
    ).clamp_min(1e-30)
    centered_denominator = (
        torch.linalg.vector_norm(centered_prediction.reshape(-1))
        * torch.linalg.vector_norm(centered_target.reshape(-1))
    ).clamp_min(1e-30)
    metrics = {
        f"{prefix}_relative_error": float(
            torch.sqrt(error_energy / target_energy).item()
        ),
        f"{prefix}_explained_energy": float(
            (1.0 - error_energy / baseline_energy).item()
        ),
        f"{prefix}_cosine": float(
            (torch.sum(prediction64 * target64) / cosine_denominator).item()
        ),
        f"{prefix}_correlation": float(
            (
                torch.sum(centered_prediction * centered_target)
                / centered_denominator
            ).item()
        ),
        f"{prefix}_amplitude_ratio": float(
            (
                torch.linalg.vector_norm(prediction64.reshape(-1))
                / torch.linalg.vector_norm(target64.reshape(-1)).clamp_min(1e-30)
            ).item()
        ),
        f"{prefix}_target_rms": float(target64.pow(2).mean().sqrt().item()),
        f"{prefix}_prediction_rms": float(
            prediction64.pow(2).mean().sqrt().item()
        ),
    }
    for step in range(int(target.shape[1])):
        step_error = error[:, step]
        step_target = target64[:, step]
        metrics[f"{prefix}_time_{step}_relative_error"] = float(
            (
                torch.linalg.vector_norm(step_error.reshape(-1))
                / torch.linalg.vector_norm(step_target.reshape(-1)).clamp_min(1e-30)
            ).item()
        )
        metrics[f"{prefix}_time_{step}_amplitude_ratio"] = float(
            (
                torch.linalg.vector_norm(prediction64[:, step].reshape(-1))
                / torch.linalg.vector_norm(step_target.reshape(-1)).clamp_min(1e-30)
            ).item()
        )
    return metrics


def fit_frozen_exact_ce_gru(
    problem: ProductionGarchTreeProblem,
    variables: torch.Tensor,
    *,
    config: FrozenExactCEFitConfig,
    initial_network_state: dict[str, torch.Tensor] | None = None,
) -> FrozenExactCEFitResult:
    """Fit the production causal GRU to a fixed law's exact P/R labels."""

    if int(config.steps) < 1 or int(config.batch_size) < 1:
        raise ValueError("steps and batch_size must be positive")
    if not 0 <= int(config.minimum_steps) <= int(config.steps):
        raise ValueError("minimum_steps must lie between zero and steps")
    if (
        config.early_stop_relative_error is not None
        and float(config.early_stop_relative_error) <= 0.0
    ):
        raise ValueError("early_stop_relative_error must be positive")
    torch.manual_seed(int(config.seed))
    if problem.device.type == "cuda":
        torch.cuda.manual_seed_all(int(config.seed))
    model, _, _ = build_production_garch_components(
        problem.config,
        lambda_scale=float(problem.lambda_scale),
        kappa_scale=float(problem.kappa_scale),
        hidden_dim=int(config.hidden_dim),
        seed=int(config.seed),
        device=problem.device,
        dtype=torch.float32,
    )
    if initial_network_state is not None:
        model.network.load_state_dict(initial_network_state)
    response = exact_garch_tree_mp_response(problem, variables)
    paths = response.solution.leaf_paths.unsqueeze(-1).to(
        device=model.device, dtype=model.dtype
    )
    target_p = response.expected_adjoint_next.to(
        device=model.device, dtype=model.dtype
    )
    target_r = response.expected_adjoint_noise_next.to(
        device=model.device, dtype=model.dtype
    )
    p_scale = target_p.double().pow(2).mean(dim=0).sqrt().clamp_min(
        float(config.minimum_scale)
    ).to(dtype=model.dtype)
    r_scale = target_r.double().pow(2).mean(dim=0).sqrt().clamp_min(
        float(config.minimum_scale)
    ).to(dtype=model.dtype)
    normalized_p = target_p / p_scale.unsqueeze(0)
    normalized_r = target_r / r_scale.unsqueeze(0)
    frozen_fingerprint = tensor_fingerprint(paths, target_p, target_r)

    optimizer = torch.optim.AdamW(
        model.network.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config.seed))
    best_loss = math.inf
    best_step = 0
    best_state = copy.deepcopy(model.network.state_dict())
    history: list[dict[str, float]] = []
    leaf_count = int(paths.shape[0])

    def full_evaluation(step: int) -> tuple[float, dict[str, float]]:
        model.network.eval()
        with torch.no_grad():
            normalized_prediction = model.network(paths)
            p_loss = torch.mean(
                (
                    normalized_prediction.expected_adjoint_next
                    - normalized_p
                ).pow(2)
            )
            r_loss = torch.mean(
                (
                    normalized_prediction.expected_adjoint_noise_next
                    - normalized_r
                ).pow(2)
            )
            prediction_p = (
                normalized_prediction.expected_adjoint_next
                * p_scale.unsqueeze(0)
            )
            prediction_r = (
                normalized_prediction.expected_adjoint_noise_next
                * r_scale.unsqueeze(0)
            )
            total = p_loss + r_loss
            metrics = {
                "step": float(step),
                "full_total_loss": float(total.item()),
                "full_p_loss": float(p_loss.item()),
                "full_r_loss": float(r_loss.item()),
                **_relative_metrics(prediction_p, target_p, prefix="p"),
                **_relative_metrics(prediction_r, target_r, prefix="r"),
            }
        model.network.train()
        return float(total.item()), metrics

    initial_loss, initial_metrics = full_evaluation(0)
    history.append(initial_metrics)
    best_loss = initial_loss
    for step in range(1, int(config.steps) + 1):
        indices = torch.randint(
            leaf_count,
            (min(int(config.batch_size), leaf_count),),
            generator=generator,
            device="cpu",
        ).to(model.device)
        moments = model.network(paths.index_select(0, indices))
        p_loss = torch.mean(
            (
                moments.expected_adjoint_next
                - normalized_p.index_select(0, indices)
            ).pow(2)
        )
        r_loss = torch.mean(
            (
                moments.expected_adjoint_noise_next
                - normalized_r.index_select(0, indices)
            ).pow(2)
        )
        loss = p_loss + r_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        unclipped_gradient = torch.sqrt(
            sum(
                parameter.grad.detach().double().pow(2).sum()
                for parameter in model.network.parameters()
                if parameter.grad is not None
            )
        )
        torch.nn.utils.clip_grad_norm_(
            model.network.parameters(), float(config.grad_clip_norm)
        )
        optimizer.step()
        if step % int(config.log_every) == 0 or step == int(config.steps):
            full_loss, metrics = full_evaluation(step)
            metrics.update(
                {
                    "minibatch_total_loss": float(loss.detach().item()),
                    "minibatch_p_loss": float(p_loss.detach().item()),
                    "minibatch_r_loss": float(r_loss.detach().item()),
                    "unclipped_gradient_norm": float(
                        unclipped_gradient.item()
                    ),
                }
            )
            history.append(metrics)
            if full_loss < best_loss:
                best_loss = full_loss
                best_step = step
                best_state = copy.deepcopy(model.network.state_dict())
            tolerance = config.early_stop_relative_error
            if (
                tolerance is not None
                and step >= int(config.minimum_steps)
                and metrics["p_relative_error"] <= float(tolerance)
                and metrics["r_relative_error"] <= float(tolerance)
            ):
                break

    model.network.load_state_dict(best_state)
    model.network.eval()
    with torch.no_grad():
        normalized_prediction = model.network(paths)
        prediction_p = (
            normalized_prediction.expected_adjoint_next
            * p_scale.unsqueeze(0)
        )
        prediction_r = (
            normalized_prediction.expected_adjoint_noise_next
            * r_scale.unsqueeze(0)
        )
    if tensor_fingerprint(paths, target_p, target_r) != frozen_fingerprint:
        raise RuntimeError("frozen paths or exact conditional targets were mutated")
    return FrozenExactCEFitResult(
        model=model,
        paths=paths.detach(),
        target_p=target_p.detach(),
        target_r=target_r.detach(),
        prediction_p=prediction_p.detach(),
        prediction_r=prediction_r.detach(),
        p_scale=p_scale.detach(),
        r_scale=r_scale.detach(),
        history=tuple(history),
        selected_step=int(best_step),
        selected_loss=float(best_loss),
        fingerprint=frozen_fingerprint,
    )


def collapse_leaf_controls_to_variables(
    problem: ProductionGarchTreeProblem,
    controls: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    if tuple(controls.shape) != (
        int(problem.config.leaf_count),
        int(problem.config.horizon),
        2,
    ):
        raise ValueError("controls have the wrong tree shape")
    if not bool(torch.all(controls[..., 0] == 0.0).item()):
        raise ValueError("volatility-only controls must have zero drift")
    sigma_levels = []
    consensus = 0.0
    branching = int(problem.config.branching)
    horizon = int(problem.config.horizon)
    for level in range(horizon):
        nodes = branching**level
        descendants = branching ** (horizon - level)
        grouped = controls[:, level, 1].reshape(nodes, descendants)
        consensus = max(
            consensus,
            float((grouped - grouped[:, :1]).abs().max().item()),
        )
        sigma_levels.append(grouped[:, 0].to(problem.dtype))
    return problem.pack(tuple(sigma_levels)).detach(), consensus


def frozen_ce_fit_metrics(
    fit: FrozenExactCEFitResult,
) -> dict[str, float]:
    return {
        **_relative_metrics(fit.prediction_p, fit.target_p, prefix="p"),
        **_relative_metrics(fit.prediction_r, fit.target_r, prefix="r"),
        "selected_step": float(fit.selected_step),
        "selected_loss": float(fit.selected_loss),
    }


__all__ = [
    "FrozenExactCEFitConfig",
    "FrozenExactCEFitResult",
    "collapse_leaf_controls_to_variables",
    "fit_frozen_exact_ce_gru",
    "frozen_ce_fit_metrics",
    "objective_gated_exact_ce_update",
    "reconstruct_exact_ce_law",
    "tensor_fingerprint",
]
