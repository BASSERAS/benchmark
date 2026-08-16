from __future__ import annotations

from dataclasses import dataclass
import copy
import math

import torch

from deep_mkv_gen_path_dt.controls import HamiltonianControlInputs
from deep_mkv_gen_path_dt.model import DiscreteMPModel
from path_dt_experiments.lifted_collocation_solver import (
    CausalControlPolicy,
    PolicyStepResult,
)


@dataclass(frozen=True)
class ReferenceControlPolicy:
    """Causal adapter exposing the fitted reference as a physical policy."""

    model: DiscreteMPModel

    def initial_state(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        del batch_size, device, dtype
        return None

    def control_step(
        self,
        *,
        x_prefix: torch.Tensor,
        step_index: int,
        state: object,
    ) -> PolicyStepResult:
        del state
        reference = getattr(self.model.control_map, "reference_kernel", None)
        if reference is None:
            raise TypeError("reference policy requires a reference kernel")
        alpha, sigma = reference.evaluate(
            x_prefix=x_prefix,
            step_index=int(step_index),
        )
        control = torch.cat((alpha, sigma), dim=-1)
        state_dim = int(self.model.architecture.state_dim)
        p = torch.zeros_like(control[..., :state_dim])
        r = torch.zeros_like(p)
        return PolicyStepResult(
            control=control,
            state=None,
            expected_adjoint_next=p,
            expected_adjoint_noise_next=r,
        )


@dataclass(frozen=True)
class ScaledAdjointControlPolicy:
    """Immutable adjoint network with its own frozen output scales."""

    model: DiscreteMPModel
    network: torch.nn.Module
    p_scale: torch.Tensor
    r_scale: torch.Tensor

    def __post_init__(self) -> None:
        expected_p = (
            int(self.model.grid.num_steps),
            int(self.model.architecture.state_dim),
        )
        expected_r = (
            int(self.model.grid.num_steps),
            int(self.model.architecture.noise_dim),
        )
        if tuple(self.p_scale.shape) != expected_p:
            raise ValueError(f"p_scale must have shape {expected_p}")
        if tuple(self.r_scale.shape) != expected_r:
            raise ValueError(f"r_scale must have shape {expected_r}")

    def initial_state(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        del batch_size, device, dtype
        return None

    def control_step(
        self,
        *,
        x_prefix: torch.Tensor,
        step_index: int,
        state: object,
    ) -> PolicyStepResult:
        normalized_p, normalized_r, next_state = self.network.forward_step(
            x_t=x_prefix[:, -1, :], hidden=state
        )
        p = normalized_p * self.p_scale[int(step_index)].to(normalized_p)
        r = normalized_r * self.r_scale[int(step_index)].to(normalized_r)
        control = self.model.control_map(
            HamiltonianControlInputs(
                step_index=int(step_index),
                x_prefix=x_prefix,
                expected_adjoint_next=p,
                expected_adjoint_noise_next=r,
                parameters=self.model.parameters,
            )
        )
        if control.ndim == 1:
            control = control.unsqueeze(-1)
        return PolicyStepResult(
            control=control,
            state=next_state,
            expected_adjoint_next=p,
            expected_adjoint_noise_next=r,
        )


@dataclass(frozen=True)
class RelaxedLogVolatilityPolicy:
    """Direct causal law-space interpolation between two policies."""

    model: DiscreteMPModel
    current_policy: CausalControlPolicy
    candidate_policy: CausalControlPolicy
    relaxation: float

    def __post_init__(self) -> None:
        rate = float(self.relaxation)
        if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
            raise ValueError("relaxation must lie in [0, 1]")
        object.__setattr__(self, "relaxation", rate)

    def initial_state(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[object, object]:
        return (
            self.current_policy.initial_state(
                batch_size=batch_size, device=device, dtype=dtype
            ),
            self.candidate_policy.initial_state(
                batch_size=batch_size, device=device, dtype=dtype
            ),
        )

    def control_step(
        self,
        *,
        x_prefix: torch.Tensor,
        step_index: int,
        state: object,
    ) -> PolicyStepResult:
        if not isinstance(state, tuple) or len(state) != 2:
            raise TypeError("relaxed policy state must contain two policy states")
        current = self.current_policy.control_step(
            x_prefix=x_prefix,
            step_index=int(step_index),
            state=state[0],
        )
        candidate = self.candidate_policy.control_step(
            x_prefix=x_prefix,
            step_index=int(step_index),
            state=state[1],
        )
        state_dim = int(self.model.architecture.state_dim)
        current_alpha = current.control[..., :state_dim]
        candidate_alpha = candidate.control[..., :state_dim]
        current_sigma = current.control[..., state_dim:]
        candidate_sigma = candidate.control[..., state_dim:]
        rate = float(self.relaxation)
        alpha = current_alpha + rate * (candidate_alpha - current_alpha)
        log_sigma = torch.log(current_sigma) + rate * (
            torch.log(candidate_sigma) - torch.log(current_sigma)
        )
        control_map = self.model.control_map
        lower = getattr(control_map, "sigma_min", None)
        upper = getattr(control_map, "sigma_max", None)
        if lower is not None:
            log_sigma = log_sigma.clamp_min(math.log(float(lower)))
        if upper is not None:
            log_sigma = log_sigma.clamp_max(math.log(float(upper)))
        return PolicyStepResult(
            control=torch.cat((alpha, torch.exp(log_sigma)), dim=-1),
            state=(current.state, candidate.state),
        )


@dataclass(frozen=True)
class FrozenCETrainingConfig:
    max_steps: int = 3_000
    minimum_steps: int = 200
    batch_size: int = 256
    eval_every: int = 100
    lr: float = 2e-3
    weight_decay: float = 1e-5
    grad_clip_norm: float = 5.0
    minimum_scale: float = 1e-3
    validation_relative_tolerance: float = 0.15
    p_loss_weight: float = 1.0
    r_loss_weight: float = 1.0
    seed: int = 0


@dataclass(frozen=True)
class FrozenCETrainingResult:
    network: torch.nn.Module
    p_scale: torch.Tensor
    r_scale: torch.Tensor
    history: tuple[dict[str, float], ...]
    selected_step: int
    selected_validation_loss: float
    fit_metrics: dict[str, float]
    validation_metrics: dict[str, float]
    accuracy_gate_passed: bool


def prediction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    prefix: str,
) -> dict[str, float]:
    prediction64 = prediction.detach().double()
    target64 = target.detach().double()
    error = prediction64 - target64
    target_rms = target64.pow(2).mean().sqrt().clamp_min(1e-30)
    error_rms = error.pow(2).mean().sqrt()
    centered_prediction = prediction64 - prediction64.mean(dim=0, keepdim=True)
    centered_target = target64 - target64.mean(dim=0, keepdim=True)
    baseline = centered_target.pow(2).mean().clamp_min(1e-30)
    denominator = (
        torch.linalg.vector_norm(centered_prediction.reshape(-1))
        * torch.linalg.vector_norm(centered_target.reshape(-1))
    ).clamp_min(1e-30)
    return {
        f"{prefix}_relative_error": float((error_rms / target_rms).item()),
        f"{prefix}_explained_energy": float(
            (1.0 - error.pow(2).mean() / baseline).item()
        ),
        f"{prefix}_correlation": float(
            (
                torch.sum(centered_prediction * centered_target)
                / denominator
            ).item()
        ),
        f"{prefix}_amplitude_ratio": float(
            (
                prediction64.pow(2).mean().sqrt()
                / target_rms
            ).item()
        ),
        f"{prefix}_target_rms": float(target_rms.item()),
    }


def _evaluate_network(
    *,
    network: torch.nn.Module,
    paths: torch.Tensor,
    target_p: torch.Tensor,
    target_r: torch.Tensor,
    p_scale: torch.Tensor,
    r_scale: torch.Tensor,
    p_loss_weight: float = 1.0,
    r_loss_weight: float = 1.0,
) -> tuple[float, dict[str, float]]:
    network.eval()
    with torch.no_grad():
        normalized = network(paths)
        prediction_p = normalized.expected_adjoint_next * p_scale.unsqueeze(0)
        prediction_r = (
            normalized.expected_adjoint_noise_next * r_scale.unsqueeze(0)
        )
        normalized_p_target = target_p / p_scale.unsqueeze(0)
        normalized_r_target = target_r / r_scale.unsqueeze(0)
        p_loss = torch.mean(
            (normalized.expected_adjoint_next - normalized_p_target).pow(2)
        )
        r_loss = torch.mean(
            (
                normalized.expected_adjoint_noise_next
                - normalized_r_target
            ).pow(2)
        )
        total = (
            float(p_loss_weight) * p_loss
            + float(r_loss_weight) * r_loss
        )
        metrics = {
            "normalized_total_loss": float(total.item()),
            "normalized_p_loss": float(p_loss.item()),
            "normalized_r_loss": float(r_loss.item()),
            **prediction_metrics(prediction_p, target_p, prefix="p"),
            **prediction_metrics(prediction_r, target_r, prefix="r"),
        }
    network.train()
    return float(total.item()), metrics


def fit_frozen_sampled_ce(
    *,
    model: DiscreteMPModel,
    fit_paths: torch.Tensor,
    fit_target_p: torch.Tensor,
    fit_target_r: torch.Tensor,
    validation_paths: torch.Tensor,
    validation_target_p: torch.Tensor,
    validation_target_r: torch.Tensor,
    config: FrozenCETrainingConfig,
    initial_network_state: dict[str, torch.Tensor] | None = None,
    network_template: torch.nn.Module | None = None,
) -> FrozenCETrainingResult:
    """Fit one causal CE network while all laws and nested labels remain frozen."""

    if int(config.max_steps) < 1 or int(config.batch_size) < 1:
        raise ValueError("max_steps and batch_size must be positive")
    if not 0 <= int(config.minimum_steps) <= int(config.max_steps):
        raise ValueError("minimum_steps must lie between zero and max_steps")
    if int(config.eval_every) < 1:
        raise ValueError("eval_every must be positive")
    if float(config.validation_relative_tolerance) <= 0.0:
        raise ValueError("validation_relative_tolerance must be positive")
    if float(config.p_loss_weight) < 0.0 or float(config.r_loss_weight) < 0.0:
        raise ValueError("CE loss weights must be nonnegative")
    if float(config.p_loss_weight) + float(config.r_loss_weight) <= 0.0:
        raise ValueError("at least one CE loss weight must be positive")
    network = copy.deepcopy(
        model.network if network_template is None else network_template
    )
    if initial_network_state is not None:
        network.load_state_dict(initial_network_state)
    network.to(device=model.device, dtype=model.dtype)
    network.train()
    fit_paths = fit_paths.detach().to(model.device, model.dtype)
    validation_paths = validation_paths.detach().to(model.device, model.dtype)
    fit_target_p = fit_target_p.detach().to(model.device, model.dtype)
    fit_target_r = fit_target_r.detach().to(model.device, model.dtype)
    validation_target_p = validation_target_p.detach().to(model.device, model.dtype)
    validation_target_r = validation_target_r.detach().to(model.device, model.dtype)
    p_scale = fit_target_p.double().pow(2).mean(dim=0).sqrt().clamp_min(
        float(config.minimum_scale)
    ).to(model.dtype)
    r_scale = fit_target_r.double().pow(2).mean(dim=0).sqrt().clamp_min(
        float(config.minimum_scale)
    ).to(model.dtype)
    normalized_p = fit_target_p / p_scale.unsqueeze(0)
    normalized_r = fit_target_r / r_scale.unsqueeze(0)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    generator = torch.Generator(device="cpu").manual_seed(int(config.seed))
    history: list[dict[str, float]] = []
    best_loss = math.inf
    best_step = 0
    best_state = copy.deepcopy(network.state_dict())

    def evaluate(step: int) -> tuple[float, dict[str, float], dict[str, float]]:
        fit_loss, fit_metrics = _evaluate_network(
            network=network,
            paths=fit_paths,
            target_p=fit_target_p,
            target_r=fit_target_r,
            p_scale=p_scale,
            r_scale=r_scale,
            p_loss_weight=float(config.p_loss_weight),
            r_loss_weight=float(config.r_loss_weight),
        )
        validation_loss, validation_metrics = _evaluate_network(
            network=network,
            paths=validation_paths,
            target_p=validation_target_p,
            target_r=validation_target_r,
            p_scale=p_scale,
            r_scale=r_scale,
            p_loss_weight=float(config.p_loss_weight),
            r_loss_weight=float(config.r_loss_weight),
        )
        history.append(
            {
                "step": float(step),
                "fit_loss": fit_loss,
                "validation_loss": validation_loss,
                **{f"fit_{key}": value for key, value in fit_metrics.items()},
                **{
                    f"validation_{key}": value
                    for key, value in validation_metrics.items()
                },
            }
        )
        return validation_loss, fit_metrics, validation_metrics

    validation_loss, _, _ = evaluate(0)
    best_loss = validation_loss
    for step in range(1, int(config.max_steps) + 1):
        count = min(int(config.batch_size), int(fit_paths.shape[0]))
        indices = torch.randint(
            int(fit_paths.shape[0]),
            (count,),
            generator=generator,
            device="cpu",
        ).to(model.device)
        moments = network(fit_paths.index_select(0, indices))
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
        loss = (
            float(config.p_loss_weight) * p_loss
            + float(config.r_loss_weight) * r_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            network.parameters(), float(config.grad_clip_norm)
        )
        optimizer.step()
        if step % int(config.eval_every) == 0 or step == int(config.max_steps):
            validation_loss, _, validation_metrics = evaluate(step)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_step = step
                best_state = copy.deepcopy(network.state_dict())
            if (
                step >= int(config.minimum_steps)
                and (
                    float(config.p_loss_weight) == 0.0
                    or validation_metrics["p_relative_error"]
                    <= float(config.validation_relative_tolerance)
                )
                and validation_metrics["r_relative_error"]
                <= float(config.validation_relative_tolerance)
            ):
                break

    network.load_state_dict(best_state)
    _, fit_metrics = _evaluate_network(
        network=network,
        paths=fit_paths,
        target_p=fit_target_p,
        target_r=fit_target_r,
        p_scale=p_scale,
        r_scale=r_scale,
        p_loss_weight=float(config.p_loss_weight),
        r_loss_weight=float(config.r_loss_weight),
    )
    _, validation_metrics = _evaluate_network(
        network=network,
        paths=validation_paths,
        target_p=validation_target_p,
        target_r=validation_target_r,
        p_scale=p_scale,
        r_scale=r_scale,
        p_loss_weight=float(config.p_loss_weight),
        r_loss_weight=float(config.r_loss_weight),
    )
    network.eval()
    gate = bool(
        (
            float(config.p_loss_weight) == 0.0
            or validation_metrics["p_relative_error"]
            <= float(config.validation_relative_tolerance)
        )
        and validation_metrics["r_relative_error"]
        <= float(config.validation_relative_tolerance)
    )
    return FrozenCETrainingResult(
        network=network,
        p_scale=p_scale.detach(),
        r_scale=r_scale.detach(),
        history=tuple(history),
        selected_step=int(best_step),
        selected_validation_loss=float(best_loss),
        fit_metrics=fit_metrics,
        validation_metrics=validation_metrics,
        accuracy_gate_passed=bool(gate),
    )


def cpu_network_state(network: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in network.state_dict().items()
    }


__all__ = [
    "FrozenCETrainingConfig",
    "FrozenCETrainingResult",
    "ReferenceControlPolicy",
    "RelaxedLogVolatilityPolicy",
    "ScaledAdjointControlPolicy",
    "cpu_network_state",
    "fit_frozen_sampled_ce",
    "prediction_metrics",
]
