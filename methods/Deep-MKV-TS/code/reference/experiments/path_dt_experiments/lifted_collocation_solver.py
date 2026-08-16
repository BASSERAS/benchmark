from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from typing import Any, Callable, Mapping, Protocol

import torch

from deep_mkv_gen_path_dt.config import DiscreteMPTrainingConfig
from deep_mkv_gen_path_dt.controls import (
    DynamicsStepInputs,
    HamiltonianControlInputs,
    SpecificEntropyDiagonalControl,
)
from deep_mkv_gen_path_dt.model import DiscreteMPModel, RolloutResult
from deep_mkv_gen_path_dt.noise import derive_stream_seed, sample_standard_normals


_HASH_CHUNK_BYTES = 1 << 20


def _hash_tensor(digest: Any, value: torch.Tensor) -> None:
    tensor = value.detach().cpu().contiguous()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(json.dumps(tuple(int(v) for v in tensor.shape)).encode("utf-8"))
    raw = tensor.view(torch.uint8).reshape(-1)
    for left in range(0, int(raw.numel()), _HASH_CHUNK_BYTES):
        right = min(int(raw.numel()), left + _HASH_CHUNK_BYTES)
        digest.update(bytes(raw[left:right].tolist()))


def _hash_value(digest: Any, value: object) -> None:
    if isinstance(value, torch.Tensor):
        digest.update(b"tensor:")
        _hash_tensor(digest, value)
    elif isinstance(value, Mapping):
        digest.update(b"mapping{")
        for key in sorted(value, key=lambda item: str(item)):
            digest.update(str(key).encode("utf-8"))
            digest.update(b":")
            _hash_value(digest, value[key])
        digest.update(b"}")
    elif isinstance(value, (tuple, list)):
        digest.update(b"sequence[")
        for item in value:
            _hash_value(digest, item)
        digest.update(b"]")
    elif value is None or isinstance(value, (bool, int, float, str)):
        digest.update(
            json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
        )
    else:
        raise TypeError(f"cannot hash value of type {type(value).__name__}")


def stable_sha256(value: object) -> str:
    """Return a process-independent digest for tensors and JSON-like values."""

    digest = hashlib.sha256()
    _hash_value(digest, value)
    return digest.hexdigest()


def model_checkpoint_sha256(model: DiscreteMPModel) -> str:
    return stable_sha256(model.checkpoint_state())


def network_sha256(network: torch.nn.Module) -> str:
    return stable_sha256(network.state_dict())


@dataclass(frozen=True)
class ScenarioDataContext:
    """Immutable path pools addressed by compact scenario-bank indices."""

    initial_state_pool: torch.Tensor
    target_path_pool: torch.Tensor
    dataset_id: str
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        initial = self.initial_state_pool.detach().clone()
        target = self.target_path_pool.detach().clone()
        if initial.ndim != 3 or target.ndim != 3:
            raise ValueError("scenario data pools must have shape (B, N + 1, d)")
        if int(initial.shape[0]) < 1 or int(target.shape[0]) < 1:
            raise ValueError("scenario data pools cannot be empty")
        if tuple(initial.shape[1:]) != tuple(target.shape[1:]):
            raise ValueError("initial and target pools must share path dimensions")
        if not str(self.dataset_id):
            raise ValueError("dataset_id cannot be empty")
        object.__setattr__(self, "initial_state_pool", initial)
        object.__setattr__(self, "target_path_pool", target)
        object.__setattr__(self, "dataset_id", str(self.dataset_id))
        object.__setattr__(self, "fingerprint", self._compute_fingerprint())

    def _compute_fingerprint(self) -> str:
        return stable_sha256(
            {
                "dataset_id": self.dataset_id,
                "initial_state_pool": self.initial_state_pool,
                "target_path_pool": self.target_path_pool,
            }
        )

    def verify(self) -> None:
        if self._compute_fingerprint() != self.fingerprint:
            raise RuntimeError("scenario data context was mutated after construction")


@dataclass(frozen=True)
class ScenarioCollocationBank:
    """Frozen exogenous inputs for replaying one generated-law evaluation."""

    initial_indices: torch.Tensor
    target_indices: torch.Tensor
    base_noise: torch.Tensor
    branch_seed: int
    branch_stream_offset: int
    num_branches: int
    antithetic: bool
    beta: float
    dataset_fingerprint: str
    label: str
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        initial = self.initial_indices.detach().cpu().clone().to(torch.long)
        target = self.target_indices.detach().cpu().clone().to(torch.long)
        noise = self.base_noise.detach().cpu().clone()
        if initial.ndim != 1 or target.ndim != 1:
            raise ValueError("scenario bank indices must be one-dimensional")
        if int(initial.numel()) < 1 or int(target.numel()) < 1:
            raise ValueError("scenario bank indices cannot be empty")
        if noise.ndim != 3 or int(noise.shape[0]) != int(initial.numel()):
            raise ValueError("base_noise must have shape (len(initial_indices), N, q)")
        branches = int(self.num_branches)
        if branches < 2:
            raise ValueError("num_branches must be at least two")
        if bool(self.antithetic) and branches % 2 != 0:
            raise ValueError("antithetic replay requires an even branch count")
        beta = float(self.beta)
        if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
            raise ValueError("beta must lie in [0, 1]")
        if not str(self.dataset_fingerprint) or not str(self.label):
            raise ValueError("dataset_fingerprint and label cannot be empty")
        object.__setattr__(self, "initial_indices", initial)
        object.__setattr__(self, "target_indices", target)
        object.__setattr__(self, "base_noise", noise)
        object.__setattr__(self, "branch_seed", int(self.branch_seed))
        object.__setattr__(
            self, "branch_stream_offset", int(self.branch_stream_offset)
        )
        object.__setattr__(self, "num_branches", branches)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "dataset_fingerprint", str(self.dataset_fingerprint))
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "fingerprint", self._compute_fingerprint())

    def _compute_fingerprint(self) -> str:
        return stable_sha256(
            {
                "initial_indices": self.initial_indices,
                "target_indices": self.target_indices,
                "base_noise": self.base_noise,
                "branch_seed": self.branch_seed,
                "branch_stream_offset": self.branch_stream_offset,
                "num_branches": self.num_branches,
                "antithetic": self.antithetic,
                "beta": self.beta,
                "dataset_fingerprint": self.dataset_fingerprint,
                "label": self.label,
            }
        )

    def verify(self, context: ScenarioDataContext) -> None:
        context.verify()
        if self.dataset_fingerprint != context.fingerprint:
            raise RuntimeError("scenario bank does not address this data context")
        if self._compute_fingerprint() != self.fingerprint:
            raise RuntimeError("scenario bank was mutated after construction")
        if int(self.initial_indices.max().item()) >= int(
            context.initial_state_pool.shape[0]
        ):
            raise IndexError("scenario bank initial index is outside the data context")
        if int(self.target_indices.max().item()) >= int(
            context.target_path_pool.shape[0]
        ):
            raise IndexError("scenario bank target index is outside the data context")


def build_scenario_collocation_bank(
    *,
    model: DiscreteMPModel,
    context: ScenarioDataContext,
    path_count: int,
    target_count: int,
    num_branches: int,
    antithetic: bool,
    beta: float,
    seed: int,
    label: str,
    branch_stream_offset: int = 10_000,
) -> ScenarioCollocationBank:
    """Freeze indices and all exogenous randomness without caching a law."""

    context.verify()
    if int(path_count) < 1 or int(target_count) < 1:
        raise ValueError("path_count and target_count must be positive")
    initial_generator = torch.Generator(device="cpu").manual_seed(int(seed))
    target_seed = derive_stream_seed(base_seed=int(seed), stream_offset=2)
    assert target_seed is not None
    target_generator = torch.Generator(device="cpu").manual_seed(target_seed)
    initial_indices = model._sample_indices(
        size=int(context.initial_state_pool.shape[0]),
        count=int(path_count),
        generator=initial_generator,
    )
    target_indices = model._sample_indices(
        size=int(context.target_path_pool.shape[0]),
        count=int(target_count),
        generator=target_generator,
    )
    base_noise = sample_standard_normals(
        grid=model.grid,
        batch_size=int(path_count),
        noise_dim=int(model.architecture.noise_dim),
        device=torch.device("cpu"),
        dtype=model.dtype,
        seed=derive_stream_seed(base_seed=int(seed), stream_offset=1),
    )
    return ScenarioCollocationBank(
        initial_indices=initial_indices,
        target_indices=target_indices,
        base_noise=base_noise,
        branch_seed=int(seed),
        branch_stream_offset=int(branch_stream_offset),
        num_branches=int(num_branches),
        antithetic=bool(antithetic),
        beta=float(beta),
        dataset_fingerprint=context.fingerprint,
        label=str(label),
    )


def build_fit_confirmation_banks(
    *,
    model: DiscreteMPModel,
    context: ScenarioDataContext,
    path_count: int,
    target_count: int,
    num_branches: int,
    antithetic: bool,
    beta: float,
    seed: int,
) -> tuple[ScenarioCollocationBank, ScenarioCollocationBank]:
    fit = build_scenario_collocation_bank(
        model=model,
        context=context,
        path_count=path_count,
        target_count=target_count,
        num_branches=num_branches,
        antithetic=antithetic,
        beta=beta,
        seed=int(seed),
        label="fit",
    )
    confirmation_seed = derive_stream_seed(base_seed=int(seed), stream_offset=50_000)
    assert confirmation_seed is not None
    confirmation = build_scenario_collocation_bank(
        model=model,
        context=context,
        path_count=path_count,
        target_count=target_count,
        num_branches=num_branches,
        antithetic=antithetic,
        beta=beta,
        seed=confirmation_seed,
        label="confirmation",
    )
    if fit.fingerprint == confirmation.fingerprint:
        raise RuntimeError("fit and confirmation banks must be independent")
    return fit, confirmation


def replay_branch_noise(
    *,
    model: DiscreteMPModel,
    bank: ScenarioCollocationBank,
    base_noise: torch.Tensor,
    step_index: int,
) -> torch.Tensor:
    """Replay one branch bank independently of request order and chunking."""

    step = int(step_index)
    num_steps = int(model.grid.num_steps)
    if not 0 <= step < num_steps:
        raise IndexError("step_index is outside the model grid")
    batch_size = int(bank.initial_indices.numel())
    expected_base = (
        batch_size,
        num_steps,
        int(model.architecture.noise_dim),
    )
    if tuple(base_noise.shape) != expected_base:
        raise ValueError(f"base_noise must have shape {expected_base}")
    independent_count = (
        int(bank.num_branches) // 2
        if bool(bank.antithetic)
        else int(bank.num_branches)
    )
    independent = sample_standard_normals(
        grid=model.grid,
        batch_size=batch_size * independent_count,
        noise_dim=int(model.architecture.noise_dim),
        device=base_noise.device,
        dtype=base_noise.dtype,
        seed=derive_stream_seed(
            base_seed=int(bank.branch_seed),
            stream_offset=int(bank.branch_stream_offset) + step,
        ),
    ).reshape(
        batch_size,
        independent_count,
        num_steps,
        int(model.architecture.noise_dim),
    )
    branches = (
        torch.cat((independent, -independent), dim=1)
        if bool(bank.antithetic)
        else independent
    )
    if step > 0:
        branches[:, :, :step, :] = base_noise[:, None, :step, :]
    return branches


@dataclass(frozen=True)
class PolicyStepResult:
    control: torch.Tensor
    state: object
    expected_adjoint_next: torch.Tensor | None = None
    expected_adjoint_noise_next: torch.Tensor | None = None


class CausalControlPolicy(Protocol):
    def initial_state(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> object: ...

    def control_step(
        self,
        *,
        x_prefix: torch.Tensor,
        step_index: int,
        state: object,
    ) -> PolicyStepResult: ...


@dataclass(frozen=True)
class AdjointNetworkControlPolicy:
    """Functional adapter from an adjoint network to Hamiltonian controls."""

    model: DiscreteMPModel
    network: torch.nn.Module

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
        p = self.model._denormalize_adjoint_step(
            normalized_p, step_index=int(step_index), noise_adjoint=False
        )
        r = self.model._denormalize_adjoint_step(
            normalized_r, step_index=int(step_index), noise_adjoint=True
        )
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


def rollout_causal_policy(
    *,
    model: DiscreteMPModel,
    policy: CausalControlPolicy,
    x0: torch.Tensor,
    noise: torch.Tensor,
) -> RolloutResult:
    """Roll out a causal policy without storing episode state on the policy."""

    if x0.ndim != 2 or noise.ndim != 3:
        raise ValueError("x0 and noise must have shapes (B, d) and (B, N, q)")
    if tuple(noise.shape[:2]) != (int(x0.shape[0]), int(model.grid.num_steps)):
        raise ValueError("noise must align with x0 and the model grid")
    state = policy.initial_state(
        batch_size=int(x0.shape[0]), device=x0.device, dtype=x0.dtype
    )
    points = [x0]
    controls: list[torch.Tensor] = []
    moments_p: list[torch.Tensor] = []
    moments_r: list[torch.Tensor] = []
    complete_moments = True
    for step in range(int(model.grid.num_steps)):
        prefix = torch.stack(points, dim=1)
        result = policy.control_step(
            x_prefix=prefix, step_index=step, state=state
        )
        state = result.state
        control = result.control
        next_state = model.dynamics_step(
            DynamicsStepInputs(
                step_index=step,
                x_prefix=prefix,
                control=control,
                noise_next=noise[:, step, :],
                parameters=model.parameters,
            )
        )
        if next_state.ndim == 1:
            next_state = next_state.unsqueeze(-1)
        if tuple(next_state.shape) != tuple(x0.shape):
            raise ValueError("policy dynamics step returned an invalid state shape")
        points.append(next_state)
        controls.append(control)
        if (
            result.expected_adjoint_next is None
            or result.expected_adjoint_noise_next is None
        ):
            complete_moments = False
        else:
            moments_p.append(result.expected_adjoint_next)
            moments_r.append(result.expected_adjoint_noise_next)
    paths = torch.stack(points, dim=1)
    packed_controls = torch.stack(controls, dim=1)
    if complete_moments:
        packed_p = torch.stack(moments_p, dim=1)
        packed_r = torch.stack(moments_r, dim=1)
    else:
        inverse = getattr(model.control_map, "moments_from_controls", None)
        if not callable(inverse):
            raise TypeError(
                "policies without moment outputs require an invertible control map"
            )
        packed_p, packed_r = inverse(paths=paths, controls=packed_controls)
    return RolloutResult(
        paths=paths,
        controls=packed_controls,
        expected_adjoint_next=packed_p,
        expected_adjoint_noise_next=packed_r,
        noise=noise,
    )


def policy_moments_on_paths(
    *,
    model: DiscreteMPModel,
    policy: CausalControlPolicy,
    paths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate adapted moments and controls on externally supplied prefixes."""

    state = policy.initial_state(
        batch_size=int(paths.shape[0]), device=paths.device, dtype=paths.dtype
    )
    controls = []
    moments_p = []
    moments_r = []
    for step in range(int(model.grid.num_steps)):
        result = policy.control_step(
            x_prefix=paths[:, : step + 1, :], step_index=step, state=state
        )
        state = result.state
        if (
            result.expected_adjoint_next is None
            or result.expected_adjoint_noise_next is None
        ):
            raise TypeError("adjoint audit policy must expose P and R moments")
        controls.append(result.control)
        moments_p.append(result.expected_adjoint_next)
        moments_r.append(result.expected_adjoint_noise_next)
    return (
        torch.stack(moments_p, dim=1),
        torch.stack(moments_r, dim=1),
        torch.stack(controls, dim=1),
    )


def _control_safety(
    *,
    model: DiscreteMPModel,
    paths: torch.Tensor,
    controls: torch.Tensor,
    moment_r: torch.Tensor | None = None,
) -> dict[str, float]:
    report = {"controls_finite": float(bool(torch.isfinite(controls).all().item()))}
    if not isinstance(model.control_map, SpecificEntropyDiagonalControl):
        return report
    state_dim = int(model.architecture.state_dim)
    sigma = controls[..., state_dim:]
    if moment_r is None:
        _, r = model.control_map.moments_from_controls(
            paths=paths, controls=controls
        )
    else:
        expected_r_shape = (
            int(paths.shape[0]),
            int(model.grid.num_steps),
            int(model.architecture.state_dim),
        )
        if tuple(moment_r.shape) != expected_r_shape:
            raise ValueError(f"moment_r must have shape {expected_r_shape}")
        r = moment_r
    sigma_ref = model.control_map._reference_sigmas_all(paths)
    denominator = float(model.control_map.eta) / sigma_ref + r / math.sqrt(
        float(model.control_map.dt)
    )
    lower = float(model.control_map.sigma_min)
    upper = model.control_map.sigma_max
    report.update(
        {
            "specific_entropy_denominator_min": float(denominator.min().item()),
            "specific_entropy_denominator_margin_min": float(
                denominator.min().item() - float(model.control_map.denominator_eps)
            ),
            "sigma_lower_bound_activity": float(
                torch.mean((sigma <= lower * (1.0 + 1e-6)).float()).item()
            ),
            "sigma_upper_bound_activity": (
                0.0
                if upper is None
                else float(
                    torch.mean(
                        (sigma >= float(upper) * (1.0 - 1e-6)).float()
                    ).item()
                )
            ),
        }
    )
    return report


@dataclass(frozen=True)
class LawEvaluation:
    paths: torch.Tensor
    controls: torch.Tensor
    policy_p: torch.Tensor
    policy_r: torch.Tensor
    target_p: torch.Tensor
    target_r: torch.Tensor
    objective: dict[str, float]
    safety: dict[str, float]
    metadata: dict[str, object]


def rollout_policy_on_bank(
    *,
    model: DiscreteMPModel,
    policy: CausalControlPolicy,
    bank: ScenarioCollocationBank,
    data_context: ScenarioDataContext,
    full_training: DiscreteMPTrainingConfig,
    query_batch_size: int,
    population_paths: torch.Tensor | None = None,
    branch_noise_provider: Callable[[int], torch.Tensor] | None = None,
    branch_target_observer: Callable[
        [int, torch.Tensor, torch.Tensor, torch.Tensor], None
    ]
    | None = None,
) -> LawEvaluation:
    """Regenerate a law and its genuine conditional targets on frozen inputs."""

    bank.verify(data_context)
    if tuple(bank.base_noise.shape[1:]) != (
        int(model.grid.num_steps),
        int(model.architecture.noise_dim),
    ):
        raise ValueError("scenario bank noise does not align with the model")
    checkpoint_before = model_checkpoint_sha256(model)
    initial_paths = data_context.initial_state_pool.index_select(
        0, bank.initial_indices.to(data_context.initial_state_pool.device)
    )
    target_paths = data_context.target_path_pool.index_select(
        0, bank.target_indices.to(data_context.target_path_pool.device)
    )
    x0 = initial_paths[:, 0, :].to(device=model.device, dtype=model.dtype)
    target_paths = target_paths.to(device=model.device, dtype=model.dtype)
    base_noise = bank.base_noise.to(device=model.device, dtype=model.dtype)
    scaled_training = replace(
        full_training,
        lambda_scale=float(full_training.lambda_scale) * float(bank.beta),
    )

    def replay_rollout(*, x0: torch.Tensor, noise: torch.Tensor) -> RolloutResult:
        return rollout_causal_policy(
            model=model, policy=policy, x0=x0, noise=noise
        )

    with torch.no_grad():
        law = replay_rollout(x0=x0, noise=base_noise)
    frozen_population = (
        law.paths.detach()
        if population_paths is None
        else population_paths.detach().to(device=model.device, dtype=model.dtype)
    )
    if frozen_population.ndim != 3 or tuple(frozen_population.shape[1:]) != tuple(
        law.paths.shape[1:]
    ):
        raise ValueError("population_paths must align with the evaluated law grid")

    def branch_provider(step: int) -> torch.Tensor:
        if branch_noise_provider is not None:
            return branch_noise_provider(int(step))
        return replay_branch_noise(
            model=model,
            bank=bank,
            base_noise=base_noise,
            step_index=step,
        )

    target_p, target_r, target_metadata = (
        model._nested_branch_conditional_targets(
            x0=x0,
            base_noise=base_noise,
            base_paths=law.paths,
            population_paths=frozen_population,
            target_path=target_paths,
            training=scaled_training,
            num_branches=int(bank.num_branches),
            antithetic=bool(bank.antithetic),
            query_batch_size=int(query_batch_size),
            seed=int(bank.branch_seed),
            branch_noise_provider=branch_provider,
            rollout_fn=replay_rollout,
            branch_target_observer=branch_target_observer,
        )
    )
    _, objective = model._complete_objective_metrics(
        generated_path=law.paths,
        controls=law.controls,
        target_path=target_paths,
        training=scaled_training,
    )
    checkpoint_after = model_checkpoint_sha256(model)
    if checkpoint_after != checkpoint_before:
        raise RuntimeError("scenario-bank evaluation mutated the model checkpoint")
    return LawEvaluation(
        paths=law.paths.detach(),
        controls=law.controls.detach(),
        policy_p=law.expected_adjoint_next.detach(),
        policy_r=law.expected_adjoint_noise_next.detach(),
        target_p=target_p.detach(),
        target_r=target_r.detach(),
        objective=objective,
        safety=_control_safety(
            model=model,
            paths=law.paths,
            controls=law.controls,
            moment_r=law.expected_adjoint_noise_next,
        ),
        metadata={
            "bank_label": bank.label,
            "bank_fingerprint": bank.fingerprint,
            "dataset_fingerprint": data_context.fingerprint,
            "beta": float(bank.beta),
            "full_lambda_scale": float(full_training.lambda_scale),
            "effective_lambda_scale": float(scaled_training.lambda_scale),
            "checkpoint_sha256_before": checkpoint_before,
            "checkpoint_sha256_after": checkpoint_after,
            "conditional_target": target_metadata,
            "external_frozen_population": bool(population_paths is not None),
            "external_branch_noise_provider": bool(
                branch_noise_provider is not None
            ),
            "frozen_population_paths": int(frozen_population.shape[0]),
        },
    )


def _rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(value.detach().double().pow(2))).item())


def _bank_consensus_metrics(
    *,
    model: DiscreteMPModel,
    evaluation: LawEvaluation,
    adjoint_policy: CausalControlPolicy,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    with torch.no_grad():
        adjoint_p, adjoint_r, adjoint_controls = policy_moments_on_paths(
            model=model,
            policy=adjoint_policy,
            paths=evaluation.paths,
        )
    p_residual = adjoint_p - evaluation.target_p
    r_residual = adjoint_r - evaluation.target_r
    state_dim = int(model.architecture.state_dim)
    alpha_residual = (
        evaluation.controls[..., :state_dim]
        - adjoint_controls[..., :state_dim]
    )
    log_sigma_residual = (
        torch.log(evaluation.controls[..., state_dim:])
        - torch.log(adjoint_controls[..., state_dim:])
    )
    transition_kl: torch.Tensor | None = None
    if isinstance(model.control_map, SpecificEntropyDiagonalControl):
        transition_kl = model.control_map.transition_kernel_kl(
            controls=adjoint_controls,
            current_controls=evaluation.controls,
        ).detach()
    eps = torch.finfo(torch.float64).eps
    p_target_rms = _rms(evaluation.target_p)
    r_target_rms = _rms(evaluation.target_r)
    metrics: dict[str, object] = {
        "p_ce_residual_rms": _rms(p_residual),
        "p_ce_relative_residual": _rms(p_residual) / max(p_target_rms, eps),
        "p_target_rms": p_target_rms,
        "r_ce_residual_rms": _rms(r_residual),
        "r_ce_relative_residual": _rms(r_residual) / max(r_target_rms, eps),
        "r_target_rms": r_target_rms,
        "drift_consensus_rms": _rms(alpha_residual),
        "log_volatility_consensus_rms": _rms(log_sigma_residual),
        "complete_objective_value": float(
            evaluation.objective["complete_objective_value"]
        ),
        "discrepancy_objective_value": float(
            evaluation.objective["discrepancy_objective_value"]
        ),
        "running_cost_value": float(evaluation.objective["running_cost_value"]),
        "accepted_control_safety": evaluation.safety,
        "adjoint_control_safety": _control_safety(
            model=model,
            paths=evaluation.paths,
            controls=adjoint_controls,
            moment_r=adjoint_r,
        ),
        "transition_kl_mean": (
            0.0 if transition_kl is None else float(transition_kl.mean().item())
        ),
        "transition_kl_q95": (
            0.0
            if transition_kl is None
            else float(torch.quantile(transition_kl.reshape(-1), 0.95).item())
        ),
        "transition_kl_max": (
            0.0 if transition_kl is None else float(transition_kl.max().item())
        ),
    }
    artifacts = {
        "p_ce_residual": p_residual.detach().cpu(),
        "r_ce_residual": r_residual.detach().cpu(),
        "drift_consensus_residual": alpha_residual.detach().cpu(),
        "log_volatility_consensus_residual": log_sigma_residual.detach().cpu(),
        "accepted_controls": evaluation.controls.detach().cpu(),
        "adjoint_controls": adjoint_controls.detach().cpu(),
        "conditional_target_p": evaluation.target_p.detach().cpu(),
        "conditional_target_r": evaluation.target_r.detach().cpu(),
    }
    if transition_kl is not None:
        artifacts["transition_kl"] = transition_kl.cpu()
    return metrics, artifacts


def _replay_comparison(
    left: LawEvaluation,
    right: LawEvaluation,
    *,
    atol: float = 2e-6,
    rtol: float = 2e-6,
) -> dict[str, object]:
    rows: dict[str, object] = {}
    exact = True
    within_tolerance = True
    for name in ("paths", "controls", "target_p", "target_r"):
        first = getattr(left, name)
        second = getattr(right, name)
        same = bool(torch.equal(first, second))
        close = bool(torch.allclose(first, second, atol=float(atol), rtol=float(rtol)))
        exact = exact and same
        within_tolerance = within_tolerance and close
        rows[name] = {
            "exact": same,
            "within_tolerance": close,
            "max_abs_difference": float(
                torch.max(torch.abs(first.detach() - second.detach())).item()
            ),
        }
    rows["all_exact"] = exact
    rows["all_within_tolerance"] = within_tolerance
    rows["atol"] = float(atol)
    rows["rtol"] = float(rtol)
    return rows


@dataclass(frozen=True)
class LiftedConsensusAuditResult:
    report: dict[str, object]
    artifacts: dict[str, object]


def audit_lifted_consensus(
    *,
    model: DiscreteMPModel,
    law_policy: CausalControlPolicy,
    adjoint_policy: CausalControlPolicy,
    data_context: ScenarioDataContext,
    fit_bank: ScenarioCollocationBank,
    confirmation_bank: ScenarioCollocationBank,
    full_training: DiscreteMPTrainingConfig,
    query_batch_size: int,
) -> LiftedConsensusAuditResult:
    """No-update fit/confirmation audit of the lifted consensus residual."""

    if fit_bank.fingerprint == confirmation_bank.fingerprint:
        raise ValueError("fit and confirmation banks must be distinct")
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
    fit = rollout_policy_on_bank(
        model=model,
        policy=law_policy,
        bank=fit_bank,
        data_context=data_context,
        full_training=full_training,
        query_batch_size=query_batch_size,
    )
    fit_replay = rollout_policy_on_bank(
        model=model,
        policy=law_policy,
        bank=fit_bank,
        data_context=data_context,
        full_training=full_training,
        query_batch_size=query_batch_size,
    )
    replay = _replay_comparison(fit, fit_replay)
    replay_passed = (
        bool(replay["all_exact"])
        if fit.paths.device.type == "cpu"
        else bool(replay["all_within_tolerance"])
    )
    replay["device_type"] = fit.paths.device.type
    replay["required_mode"] = (
        "bitwise_exact" if fit.paths.device.type == "cpu" else "numeric_tolerance"
    )
    replay["passed"] = replay_passed
    if not replay_passed:
        raise RuntimeError("scenario-bank replay failed its device-specific gate")
    confirmation = rollout_policy_on_bank(
        model=model,
        policy=law_policy,
        bank=confirmation_bank,
        data_context=data_context,
        full_training=full_training,
        query_batch_size=query_batch_size,
    )
    fit_metrics, fit_artifacts = _bank_consensus_metrics(
        model=model, evaluation=fit, adjoint_policy=adjoint_policy
    )
    confirmation_metrics, confirmation_artifacts = _bank_consensus_metrics(
        model=model, evaluation=confirmation, adjoint_policy=adjoint_policy
    )
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
        "model_checkpoint_before": model_before,
        "model_checkpoint_after": model_after,
        "model_unchanged": model_before == model_after,
        "law_policy_before": law_before,
        "law_policy_after": law_after,
        "law_policy_unchanged": law_before == law_after,
        "adjoint_policy_before": adjoint_before,
        "adjoint_policy_after": adjoint_after,
        "adjoint_policy_unchanged": adjoint_before == adjoint_after,
    }
    if not all(
        bool(nonmutation[name])
        for name in (
            "model_unchanged",
            "law_policy_unchanged",
            "adjoint_policy_unchanged",
        )
    ):
        raise RuntimeError("lifted consensus audit mutated a checkpoint or policy")
    return LiftedConsensusAuditResult(
        report={
            "protocol": {
                "solver_update_applied": False,
                "generated_law_cached": False,
                "law_population_regenerated_each_evaluation": True,
                "conditional_targets_recomputed_under_evaluated_law": True,
                "branch_replay_chunk_independent": True,
                "effective_objective": "J_beta",
                "running_cost_scaled": False,
                "discrepancy_scaled_once": True,
                "fit_and_confirmation_banks_independent": True,
            },
            "dataset_fingerprint": data_context.fingerprint,
            "fit_bank_fingerprint": fit_bank.fingerprint,
            "confirmation_bank_fingerprint": confirmation_bank.fingerprint,
            "fit": fit_metrics,
            "confirmation": confirmation_metrics,
            "fit_replay": replay,
            "nonmutation": nonmutation,
            "fit_metadata": fit.metadata,
            "confirmation_metadata": confirmation.metadata,
        },
        artifacts={
            "fit": fit_artifacts,
            "confirmation": confirmation_artifacts,
            "fit_bank": {
                "initial_indices": fit_bank.initial_indices,
                "target_indices": fit_bank.target_indices,
                "base_noise": fit_bank.base_noise,
            },
            "confirmation_bank": {
                "initial_indices": confirmation_bank.initial_indices,
                "target_indices": confirmation_bank.target_indices,
                "base_noise": confirmation_bank.base_noise,
            },
        },
    )


@dataclass(frozen=True)
class DirectionalActiveSetAuditConfig:
    num_time_blocks: int = 3
    epsilon: float = 0.025
    projected_kkt_gamma: float = 1.0
    maximum_active_set_flip_fraction: float = 0.05
    maximum_kkt_slope_asymmetry: float = 0.25

    def __post_init__(self) -> None:
        blocks = int(self.num_time_blocks)
        if blocks < 1:
            raise ValueError("num_time_blocks must be positive")
        object.__setattr__(self, "num_time_blocks", blocks)
        for name in ("epsilon", "projected_kkt_gamma"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        for name in (
            "maximum_active_set_flip_fraction",
            "maximum_kkt_slope_asymmetry",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, value)


def time_block_bounds(
    *, num_steps: int, num_blocks: int
) -> tuple[tuple[int, int], ...]:
    steps = int(num_steps)
    blocks = int(num_blocks)
    if steps < 1 or blocks < 1 or blocks > steps:
        raise ValueError("time blocks require 1 <= num_blocks <= num_steps")
    result = []
    for index in range(blocks):
        left = (index * steps) // blocks
        right = ((index + 1) * steps) // blocks
        result.append((left, right))
    return tuple(result)


def _block_for_step(
    *, step_index: int, blocks: tuple[tuple[int, int], ...]
) -> int:
    step = int(step_index)
    for index, (left, right) in enumerate(blocks):
        if left <= step < right:
            return index
    raise IndexError("step_index is outside the declared time blocks")


@dataclass(frozen=True)
class BlockwiseControlPerturbationPolicy:
    """Causal physical-control path from a law policy toward an adjoint policy."""

    model: DiscreteMPModel
    law_policy: CausalControlPolicy
    adjoint_policy: CausalControlPolicy
    blocks: tuple[tuple[int, int], ...]
    alpha_coefficients: tuple[float, ...]
    log_sigma_coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        block_count = len(self.blocks)
        if block_count < 1:
            raise ValueError("blocks cannot be empty")
        if len(self.alpha_coefficients) != block_count or len(
            self.log_sigma_coefficients
        ) != block_count:
            raise ValueError("control coefficients must contain one value per block")
        if any(
            not math.isfinite(float(value))
            for value in self.alpha_coefficients + self.log_sigma_coefficients
        ):
            raise ValueError("control coefficients must be finite")
        expected_left = 0
        for left, right in self.blocks:
            if left != expected_left or right <= left:
                raise ValueError("blocks must be contiguous and nonempty")
            expected_left = right
        if expected_left != int(self.model.grid.num_steps):
            raise ValueError("blocks must partition the complete model grid")

    def initial_state(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[object, object]:
        return (
            self.law_policy.initial_state(
                batch_size=batch_size, device=device, dtype=dtype
            ),
            self.adjoint_policy.initial_state(
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
            raise TypeError("blockwise policy state must contain law and adjoint states")
        law_step = self.law_policy.control_step(
            x_prefix=x_prefix, step_index=step_index, state=state[0]
        )
        adjoint_step = self.adjoint_policy.control_step(
            x_prefix=x_prefix, step_index=step_index, state=state[1]
        )
        state_dim = int(self.model.architecture.state_dim)
        expected_control_dim = 2 * state_dim
        if int(law_step.control.shape[-1]) != expected_control_dim or int(
            adjoint_step.control.shape[-1]
        ) != expected_control_dim:
            raise ValueError(
                "blockwise physical perturbations require diagonal alpha/sigma controls"
            )
        block = _block_for_step(step_index=step_index, blocks=self.blocks)
        alpha_rate = float(self.alpha_coefficients[block])
        sigma_rate = float(self.log_sigma_coefficients[block])
        law_alpha = law_step.control[..., :state_dim]
        law_sigma = law_step.control[..., state_dim:]
        adjoint_alpha = adjoint_step.control[..., :state_dim]
        adjoint_sigma = adjoint_step.control[..., state_dim:]
        alpha = law_alpha + alpha_rate * (adjoint_alpha - law_alpha)
        log_sigma = torch.log(law_sigma) + sigma_rate * (
            torch.log(adjoint_sigma) - torch.log(law_sigma)
        )
        control_map = self.model.control_map
        if isinstance(control_map, SpecificEntropyDiagonalControl):
            log_sigma = log_sigma.clamp_min(math.log(float(control_map.sigma_min)))
            if control_map.sigma_max is not None:
                log_sigma = log_sigma.clamp_max(
                    math.log(float(control_map.sigma_max))
                )
        return PolicyStepResult(
            control=torch.cat((alpha, torch.exp(log_sigma)), dim=-1),
            state=(law_step.state, adjoint_step.state),
        )


@dataclass(frozen=True)
class ProjectedLogVolatilityKKT:
    residual: torch.Tensor
    gradient: torch.Tensor
    denominator: torch.Tensor
    target_active_set: torch.Tensor
    actual_active_set: torch.Tensor
    sign_violation: torch.Tensor


def projected_log_volatility_kkt(
    *,
    model: DiscreteMPModel,
    paths: torch.Tensor,
    sigma: torch.Tensor,
    moment_r: torch.Tensor,
    gamma: float,
    activity_tolerance: float = 1e-6,
) -> ProjectedLogVolatilityKKT:
    """Projected-gradient/KKT residual for bounded specific-entropy volatility."""

    if not isinstance(model.control_map, SpecificEntropyDiagonalControl):
        raise TypeError("projected volatility KKT requires specific entropy")
    if model.control_map.sigma_max is None:
        raise ValueError("projected volatility KKT requires an upper admissible bound")
    expected = (
        int(paths.shape[0]),
        int(model.grid.num_steps),
        int(model.architecture.state_dim),
    )
    if tuple(sigma.shape) != expected or tuple(moment_r.shape) != expected:
        raise ValueError(f"sigma and moment_r must have shape {expected}")
    gamma_value = float(gamma)
    if not math.isfinite(gamma_value) or gamma_value <= 0.0:
        raise ValueError("gamma must be positive and finite")
    lower = float(model.control_map.sigma_min)
    upper = float(model.control_map.sigma_max)
    sigma_ref = model.control_map._reference_sigmas_all(paths)
    denominator = float(model.control_map.eta) / sigma_ref + moment_r / math.sqrt(
        float(model.control_map.dt)
    )
    log_sigma = torch.log(sigma)
    gradient = sigma * denominator - float(model.control_map.eta)
    projected = (log_sigma - gamma_value * gradient).clamp(
        min=math.log(lower), max=math.log(upper)
    )
    residual = log_sigma - projected

    unconstrained = float(model.control_map.eta) / denominator.clamp_min(
        float(model.control_map.denominator_eps)
    )
    target_active = torch.zeros_like(sigma, dtype=torch.int8)
    target_active = torch.where(
        (denominator <= float(model.control_map.denominator_eps))
        | (unconstrained >= upper),
        torch.ones_like(target_active),
        target_active,
    )
    target_active = torch.where(
        (denominator > float(model.control_map.denominator_eps))
        & (unconstrained <= lower),
        -torch.ones_like(target_active),
        target_active,
    )
    actual_active = torch.zeros_like(sigma, dtype=torch.int8)
    actual_active = torch.where(
        sigma <= lower * (1.0 + float(activity_tolerance)),
        -torch.ones_like(actual_active),
        actual_active,
    )
    actual_active = torch.where(
        sigma >= upper * (1.0 - float(activity_tolerance)),
        torch.ones_like(actual_active),
        actual_active,
    )
    sign_violation = torch.where(
        actual_active < 0,
        torch.relu(-gradient),
        torch.where(actual_active > 0, torch.relu(gradient), torch.abs(gradient)),
    )
    return ProjectedLogVolatilityKKT(
        residual=residual.detach(),
        gradient=gradient.detach(),
        denominator=denominator.detach(),
        target_active_set=target_active.detach(),
        actual_active_set=actual_active.detach(),
        sign_violation=sign_violation.detach(),
    )


def _active_set_summary(values: torch.Tensor) -> dict[str, float]:
    return {
        "lower_fraction": float(torch.mean((values < 0).float()).item()),
        "interior_fraction": float(torch.mean((values == 0).float()).item()),
        "upper_fraction": float(torch.mean((values > 0).float()).item()),
    }


def _directional_metrics(
    *,
    model: DiscreteMPModel,
    evaluation: LawEvaluation,
    law_policy: CausalControlPolicy,
    adjoint_policy: CausalControlPolicy,
    gamma: float,
    baseline_active_sets: Mapping[str, torch.Tensor] | None = None,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    with torch.no_grad():
        adjoint_p, adjoint_r, adjoint_controls = policy_moments_on_paths(
            model=model, policy=adjoint_policy, paths=evaluation.paths
        )
        _, _, law_controls_on_candidate = policy_moments_on_paths(
            model=model, policy=law_policy, paths=evaluation.paths
        )
    state_dim = int(model.architecture.state_dim)
    candidate_sigma = evaluation.controls[..., state_dim:]
    adjoint_kkt = projected_log_volatility_kkt(
        model=model,
        paths=evaluation.paths,
        sigma=candidate_sigma,
        moment_r=adjoint_r,
        gamma=gamma,
    )
    target_kkt = projected_log_volatility_kkt(
        model=model,
        paths=evaluation.paths,
        sigma=candidate_sigma,
        moment_r=evaluation.target_r,
        gamma=gamma,
    )
    p_residual = adjoint_p - evaluation.target_p
    r_residual = adjoint_r - evaluation.target_r
    drift_consensus = (
        evaluation.controls[..., :state_dim]
        - adjoint_controls[..., :state_dim]
    )
    log_volatility_consensus = torch.log(candidate_sigma) - torch.log(
        adjoint_controls[..., state_dim:]
    )
    control = model.control_map
    assert isinstance(control, SpecificEntropyDiagonalControl)
    law_kl = control.transition_kernel_kl(
        controls=evaluation.controls,
        current_controls=law_controls_on_candidate,
    ).detach()
    adjoint_kl = control.transition_kernel_kl(
        controls=adjoint_controls,
        current_controls=evaluation.controls,
    ).detach()
    active_sets = {
        "adjoint_target": adjoint_kkt.target_active_set,
        "conditional_target": target_kkt.target_active_set,
        "actual_control": adjoint_kkt.actual_active_set,
    }
    flips = {}
    for name, values in active_sets.items():
        if baseline_active_sets is None:
            flips[f"{name}_flip_fraction"] = 0.0
        else:
            baseline = baseline_active_sets[name].to(values.device)
            if tuple(baseline.shape) != tuple(values.shape):
                raise ValueError("baseline active-set shape changed within a bank")
            flips[f"{name}_flip_fraction"] = float(
                torch.mean((values != baseline).float()).item()
            )
    metrics: dict[str, object] = {
        "p_ce_residual_rms": _rms(p_residual),
        "r_ce_residual_rms": _rms(r_residual),
        "drift_consensus_rms": _rms(drift_consensus),
        "log_volatility_consensus_rms": _rms(log_volatility_consensus),
        "adjoint_projected_kkt_residual_rms": _rms(adjoint_kkt.residual),
        "target_projected_kkt_residual_rms": _rms(target_kkt.residual),
        "adjoint_kkt_sign_violation_rms": _rms(adjoint_kkt.sign_violation),
        "target_kkt_sign_violation_rms": _rms(target_kkt.sign_violation),
        "adjoint_raw_denominator_min": float(adjoint_kkt.denominator.min().item()),
        "target_raw_denominator_min": float(target_kkt.denominator.min().item()),
        "complete_objective_value": float(
            evaluation.objective["complete_objective_value"]
        ),
        "discrepancy_objective_value": float(
            evaluation.objective["discrepancy_objective_value"]
        ),
        "running_cost_value": float(evaluation.objective["running_cost_value"]),
        "law_transition_kl_mean": float(law_kl.mean().item()),
        "law_transition_kl_q95": float(
            torch.quantile(law_kl.reshape(-1), 0.95).item()
        ),
        "adjoint_transition_kl_mean": float(adjoint_kl.mean().item()),
        "candidate_sigma_lower_activity": float(
            torch.mean(
                (
                    candidate_sigma
                    <= float(control.sigma_min) * (1.0 + 1e-6)
                ).float()
            ).item()
        ),
        "candidate_sigma_upper_activity": float(
            torch.mean(
                (
                    candidate_sigma
                    >= float(control.sigma_max) * (1.0 - 1e-6)
                ).float()
            ).item()
        ),
        "adjoint_target_active_set": _active_set_summary(
            adjoint_kkt.target_active_set
        ),
        "conditional_target_active_set": _active_set_summary(
            target_kkt.target_active_set
        ),
        "actual_control_active_set": _active_set_summary(
            adjoint_kkt.actual_active_set
        ),
        **flips,
    }
    artifacts = {
        "adjoint_target_active_set": adjoint_kkt.target_active_set.cpu(),
        "conditional_target_active_set": target_kkt.target_active_set.cpu(),
        "actual_control_active_set": adjoint_kkt.actual_active_set.cpu(),
        "adjoint_projected_kkt_residual": adjoint_kkt.residual.cpu(),
        "target_projected_kkt_residual": target_kkt.residual.cpu(),
        "adjoint_kkt_gradient": adjoint_kkt.gradient.cpu(),
        "target_kkt_gradient": target_kkt.gradient.cpu(),
        "adjoint_raw_denominator": adjoint_kkt.denominator.cpu(),
        "target_raw_denominator": target_kkt.denominator.cpu(),
        "p_ce_residual": p_residual.detach().cpu(),
        "r_ce_residual": r_residual.detach().cpu(),
        "drift_consensus_residual": drift_consensus.detach().cpu(),
        "log_volatility_consensus_residual": log_volatility_consensus.detach().cpu(),
        "candidate_controls": evaluation.controls.detach().cpu(),
        "law_transition_kl": law_kl.cpu(),
    }
    return metrics, artifacts


def _surface_slopes(
    *,
    baseline: Mapping[str, object],
    minus: Mapping[str, object],
    plus: Mapping[str, object],
    epsilon: float,
) -> dict[str, dict[str, float]]:
    metric_names = (
        "p_ce_residual_rms",
        "r_ce_residual_rms",
        "drift_consensus_rms",
        "log_volatility_consensus_rms",
        "adjoint_projected_kkt_residual_rms",
        "target_projected_kkt_residual_rms",
        "complete_objective_value",
        "law_transition_kl_mean",
    )
    result = {}
    floor = torch.finfo(torch.float64).eps
    for name in metric_names:
        zero = float(baseline[name])
        low = float(minus[name])
        high = float(plus[name])
        forward = (high - zero) / float(epsilon)
        backward = (zero - low) / float(epsilon)
        scale = max(abs(forward), abs(backward), floor)
        result[name] = {
            "central_slope": (high - low) / (2.0 * float(epsilon)),
            "forward_slope": forward,
            "backward_slope": backward,
            "slope_asymmetry": abs(forward - backward) / scale,
        }
    return result


@dataclass(frozen=True)
class DirectionalActiveSetAuditResult:
    report: dict[str, object]
    artifacts: dict[str, object]


def audit_directional_active_set_surface(
    *,
    model: DiscreteMPModel,
    law_policy: CausalControlPolicy,
    adjoint_policy: CausalControlPolicy,
    data_context: ScenarioDataContext,
    fit_bank: ScenarioCollocationBank,
    confirmation_bank: ScenarioCollocationBank,
    full_training: DiscreteMPTrainingConfig,
    query_batch_size: int,
    config: DirectionalActiveSetAuditConfig,
) -> DirectionalActiveSetAuditResult:
    """Evaluate twelve symmetric control perturbations without taking an update."""

    if not isinstance(model.control_map, SpecificEntropyDiagonalControl):
        raise TypeError("active-set surface audit requires specific entropy")
    if model.control_map.sigma_max is None:
        raise ValueError("active-set surface audit requires a bounded admissible set")
    blocks = time_block_bounds(
        num_steps=int(model.grid.num_steps),
        num_blocks=int(config.num_time_blocks),
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

    baseline_metrics = {}
    baseline_artifacts = {}
    for bank_name, bank in (
        ("fit", fit_bank),
        ("confirmation", confirmation_bank),
    ):
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
        baseline_metrics[bank_name] = metrics
        baseline_artifacts[bank_name] = artifacts

    coordinate_names = []
    for signal in ("alpha", "log_sigma"):
        for block_index in range(len(blocks)):
            coordinate_names.append((signal, block_index))
    surfaces: dict[str, dict[str, object]] = {
        "fit": {},
        "confirmation": {},
    }
    surface_artifacts: dict[str, dict[str, object]] = {
        "fit": {},
        "confirmation": {},
    }
    for signal, block_index in coordinate_names:
        coordinate = f"{signal}_block_{block_index}"
        for sign, sign_name in ((-1.0, "minus"), (1.0, "plus")):
            alpha = [0.0] * len(blocks)
            log_sigma = [0.0] * len(blocks)
            coefficients = alpha if signal == "alpha" else log_sigma
            coefficients[block_index] = sign * float(config.epsilon)
            policy = BlockwiseControlPerturbationPolicy(
                model=model,
                law_policy=law_policy,
                adjoint_policy=adjoint_policy,
                blocks=blocks,
                alpha_coefficients=tuple(alpha),
                log_sigma_coefficients=tuple(log_sigma),
            )
            for bank_name, bank in (
                ("fit", fit_bank),
                ("confirmation", confirmation_bank),
            ):
                evaluation = rollout_policy_on_bank(
                    model=model,
                    policy=policy,
                    bank=bank,
                    data_context=data_context,
                    full_training=full_training,
                    query_batch_size=query_batch_size,
                )
                baseline_sets = {
                    name: baseline_artifacts[bank_name][f"{name}_active_set"]
                    for name in (
                        "adjoint_target",
                        "conditional_target",
                        "actual_control",
                    )
                }
                metrics, artifacts = _directional_metrics(
                    model=model,
                    evaluation=evaluation,
                    law_policy=law_policy,
                    adjoint_policy=adjoint_policy,
                    gamma=float(config.projected_kkt_gamma),
                    baseline_active_sets=baseline_sets,
                )
                row = surfaces[bank_name].setdefault(
                    coordinate,
                    {
                        "coordinate": coordinate,
                        "signal": signal,
                        "block_index": int(block_index),
                        "block_steps": list(blocks[block_index]),
                    },
                )
                row[sign_name] = metrics
                artifact_row = surface_artifacts[bank_name].setdefault(
                    coordinate, {}
                )
                artifact_row[sign_name] = artifacts

    slope_metrics = {}
    for bank_name in ("fit", "confirmation"):
        slope_metrics[bank_name] = {}
        for coordinate, row in surfaces[bank_name].items():
            slopes = _surface_slopes(
                baseline=baseline_metrics[bank_name],
                minus=row["minus"],
                plus=row["plus"],
                epsilon=float(config.epsilon),
            )
            row["slopes"] = slopes
            slope_metrics[bank_name][coordinate] = slopes

    flip_names = (
        "adjoint_target_flip_fraction",
        "conditional_target_flip_fraction",
        "actual_control_flip_fraction",
    )
    maximum_flip = 0.0
    maximum_kkt_asymmetry = 0.0
    for bank_name in ("fit", "confirmation"):
        for row in surfaces[bank_name].values():
            for side in ("minus", "plus"):
                maximum_flip = max(
                    maximum_flip,
                    *(float(row[side][name]) for name in flip_names),
                )
            maximum_kkt_asymmetry = max(
                maximum_kkt_asymmetry,
                float(
                    row["slopes"]["adjoint_projected_kkt_residual_rms"][
                        "slope_asymmetry"
                    ]
                ),
                float(
                    row["slopes"]["target_projected_kkt_residual_rms"][
                        "slope_asymmetry"
                    ]
                ),
            )
    active_set_stable = maximum_flip <= float(
        config.maximum_active_set_flip_fraction
    )
    slopes_symmetric = maximum_kkt_asymmetry <= float(
        config.maximum_kkt_slope_asymmetry
    )
    central_jacobian_supported = active_set_stable and slopes_symmetric

    sign_agreement = {}
    for coordinate in surfaces["fit"]:
        sign_agreement[coordinate] = {}
        for metric in (
            "p_ce_residual_rms",
            "r_ce_residual_rms",
            "drift_consensus_rms",
            "adjoint_projected_kkt_residual_rms",
            "target_projected_kkt_residual_rms",
            "complete_objective_value",
        ):
            fit_slope = float(
                surfaces["fit"][coordinate]["slopes"][metric]["central_slope"]
            )
            confirmation_slope = float(
                surfaces["confirmation"][coordinate]["slopes"][metric][
                    "central_slope"
                ]
            )
            sign_agreement[coordinate][metric] = bool(
                fit_slope == 0.0
                or confirmation_slope == 0.0
                or math.copysign(1.0, fit_slope)
                == math.copysign(1.0, confirmation_slope)
            )

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
        "model_checkpoint_before": model_before,
        "model_checkpoint_after": model_after,
        "model_unchanged": model_before == model_after,
        "law_policy_before": law_before,
        "law_policy_after": law_after,
        "law_policy_unchanged": law_before == law_after,
        "adjoint_policy_before": adjoint_before,
        "adjoint_policy_after": adjoint_after,
        "adjoint_policy_unchanged": adjoint_before == adjoint_after,
    }
    if not all(
        bool(nonmutation[name])
        for name in (
            "model_unchanged",
            "law_policy_unchanged",
            "adjoint_policy_unchanged",
        )
    ):
        raise RuntimeError("directional active-set audit mutated model state")
    return DirectionalActiveSetAuditResult(
        report={
            "protocol": {
                "solver_update_applied": False,
                "directions": "three_time_blocks_in_alpha_and_log_sigma",
                "direction_target": "constrained_adjoint_induced_control",
                "uncapped_denominator_used_only_as_diagnostic": True,
                "candidate_law_and_conditional_targets_recomputed": True,
                "common_random_numbers": True,
                "fit_and_confirmation_banks_independent": True,
            },
            "config": {
                "num_time_blocks": int(config.num_time_blocks),
                "epsilon": float(config.epsilon),
                "projected_kkt_gamma": float(config.projected_kkt_gamma),
                "maximum_active_set_flip_fraction": float(
                    config.maximum_active_set_flip_fraction
                ),
                "maximum_kkt_slope_asymmetry": float(
                    config.maximum_kkt_slope_asymmetry
                ),
            },
            "blocks": [list(block) for block in blocks],
            "baseline": baseline_metrics,
            "surfaces": surfaces,
            "fit_confirmation_central_slope_sign_agreement": sign_agreement,
            "decision": {
                "maximum_active_set_flip_fraction": maximum_flip,
                "maximum_kkt_slope_asymmetry": maximum_kkt_asymmetry,
                "active_set_locally_stable": active_set_stable,
                "kkt_slopes_locally_symmetric": slopes_symmetric,
                "central_jacobian_supported": central_jacobian_supported,
                "recommended_next_linearization": (
                    "central_gauss_newton"
                    if central_jacobian_supported
                    else "semismooth_active_set"
                ),
            },
            "nonmutation": nonmutation,
            "dataset_fingerprint": data_context.fingerprint,
            "fit_bank_fingerprint": fit_bank.fingerprint,
            "confirmation_bank_fingerprint": confirmation_bank.fingerprint,
        },
        artifacts={
            "baseline": baseline_artifacts,
            "surfaces": surface_artifacts,
        },
    )


@dataclass(frozen=True)
class MultiRadiusDerivativeAuditConfig:
    epsilons: tuple[float, ...] = (0.025, 0.0125, 0.00625)
    num_time_blocks: int = 3
    projected_kkt_gamma: float = 1.0
    minimum_stable_cosine: float = 0.8
    maximum_stable_relative_difference: float = 0.5

    def __post_init__(self) -> None:
        radii = tuple(float(value) for value in self.epsilons)
        if len(radii) < 2 or any(
            not math.isfinite(value) or value <= 0.0 for value in radii
        ):
            raise ValueError("epsilons must contain at least two positive finite radii")
        if any(left <= right for left, right in zip(radii, radii[1:])):
            raise ValueError("epsilons must be strictly decreasing")
        object.__setattr__(self, "epsilons", radii)
        blocks = int(self.num_time_blocks)
        if blocks < 1:
            raise ValueError("num_time_blocks must be positive")
        object.__setattr__(self, "num_time_blocks", blocks)
        gamma = float(self.projected_kkt_gamma)
        if not math.isfinite(gamma) or gamma <= 0.0:
            raise ValueError("projected_kkt_gamma must be positive and finite")
        object.__setattr__(self, "projected_kkt_gamma", gamma)
        cosine = float(self.minimum_stable_cosine)
        difference = float(self.maximum_stable_relative_difference)
        if not -1.0 <= cosine <= 1.0:
            raise ValueError("minimum_stable_cosine must lie in [-1, 1]")
        if not math.isfinite(difference) or difference < 0.0:
            raise ValueError(
                "maximum_stable_relative_difference must be nonnegative and finite"
            )
        object.__setattr__(self, "minimum_stable_cosine", cosine)
        object.__setattr__(
            self, "maximum_stable_relative_difference", difference
        )


@dataclass(frozen=True)
class MultiRadiusDerivativeAuditResult:
    report: dict[str, object]
    artifacts: dict[str, object]


_SIGNED_RESIDUAL_COMPONENTS = (
    "p_ce_residual",
    "r_ce_residual",
    "drift_consensus_residual",
    "log_volatility_consensus_residual",
    "adjoint_projected_kkt_residual",
    "target_projected_kkt_residual",
)


def compare_signed_columns(
    left: torch.Tensor,
    right: torch.Tensor,
) -> dict[str, float]:
    """Compare signed derivative columns without discarding orientation."""

    first = left.detach().to(dtype=torch.float64).reshape(-1)
    second = right.detach().to(dtype=torch.float64).reshape(-1)
    if tuple(first.shape) != tuple(second.shape):
        raise ValueError("signed columns must have matching shapes")
    if not bool(torch.isfinite(first).all().item()) or not bool(
        torch.isfinite(second).all().item()
    ):
        raise ValueError("signed columns must be finite")
    first_norm = float(torch.linalg.vector_norm(first).item())
    second_norm = float(torch.linalg.vector_norm(second).item())
    floor = float(torch.finfo(torch.float64).eps)
    maximum_norm = max(first_norm, second_norm, floor)
    if first_norm <= floor and second_norm <= floor:
        cosine = 1.0
    elif first_norm <= floor or second_norm <= floor:
        cosine = 0.0
    else:
        cosine = float(
            torch.dot(first, second).item() / (first_norm * second_norm)
        )
        cosine = max(-1.0, min(1.0, cosine))
    difference_norm = float(torch.linalg.vector_norm(first - second).item())
    return {
        "cosine_similarity": cosine,
        "left_norm": first_norm,
        "right_norm": second_norm,
        "norm_ratio": first_norm / max(second_norm, floor),
        "relative_norm_change": abs(first_norm - second_norm) / maximum_norm,
        "relative_vector_difference": difference_norm / maximum_norm,
    }


def _residual_normalization_scales(
    fit_baseline: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    scales = {}
    for name in _SIGNED_RESIDUAL_COMPONENTS:
        value = fit_baseline[name].detach().to(dtype=torch.float64)
        scale = float(torch.mean(value.pow(2)).sqrt().item())
        scales[name] = max(scale, 1e-8)
    return scales


def _signed_residual_systems(
    artifacts: Mapping[str, torch.Tensor],
    *,
    scales: Mapping[str, float],
    profile: bool,
) -> dict[str, torch.Tensor]:
    components = {}
    for name in _SIGNED_RESIDUAL_COMPONENTS:
        value = artifacts[name].detach().to(dtype=torch.float64) / float(scales[name])
        if profile:
            if value.ndim != 3:
                raise ValueError("profile residuals must have shape (B, N, d)")
            value = value.mean(dim=(0, 2))
        components[name] = value.reshape(-1)
    components["lifted_consensus"] = torch.cat(
        [
            components["p_ce_residual"],
            components["r_ce_residual"],
            components["drift_consensus_residual"],
            components["log_volatility_consensus_residual"],
        ]
    )
    return components


def _one_sided_columns(
    *,
    baseline: Mapping[str, torch.Tensor],
    minus: Mapping[str, torch.Tensor],
    plus: Mapping[str, torch.Tensor],
    scales: Mapping[str, float],
    epsilon: float,
    profile: bool,
) -> dict[str, dict[str, torch.Tensor]]:
    zero_systems = _signed_residual_systems(
        baseline, scales=scales, profile=profile
    )
    minus_systems = _signed_residual_systems(
        minus, scales=scales, profile=profile
    )
    plus_systems = _signed_residual_systems(
        plus, scales=scales, profile=profile
    )
    return {
        "minus": {
            name: (zero_systems[name] - minus_systems[name]) / float(epsilon)
            for name in zero_systems
        },
        "plus": {
            name: (plus_systems[name] - zero_systems[name]) / float(epsilon)
            for name in zero_systems
        },
    }


def _comparison_gate(
    comparison: Mapping[str, float],
    *,
    config: MultiRadiusDerivativeAuditConfig,
) -> bool:
    return bool(
        float(comparison["cosine_similarity"])
        >= float(config.minimum_stable_cosine)
        and float(comparison["relative_vector_difference"])
        <= float(config.maximum_stable_relative_difference)
    )


def _aggregate_comparisons(
    values: list[Mapping[str, float]],
    *,
    config: MultiRadiusDerivativeAuditConfig,
) -> dict[str, object]:
    if not values:
        raise ValueError("cannot aggregate an empty comparison collection")
    return {
        "count": len(values),
        "minimum_cosine_similarity": min(
            float(value["cosine_similarity"]) for value in values
        ),
        "maximum_relative_vector_difference": max(
            float(value["relative_vector_difference"]) for value in values
        ),
        "maximum_relative_norm_change": max(
            float(value["relative_norm_change"]) for value in values
        ),
        "passing_count": sum(
            _comparison_gate(value, config=config) for value in values
        ),
        "all_pass": all(
            _comparison_gate(value, config=config) for value in values
        ),
    }


def audit_multiradius_signed_derivatives(
    *,
    model: DiscreteMPModel,
    law_policy: CausalControlPolicy,
    adjoint_policy: CausalControlPolicy,
    data_context: ScenarioDataContext,
    fit_bank: ScenarioCollocationBank,
    confirmation_bank: ScenarioCollocationBank,
    full_training: DiscreteMPTrainingConfig,
    query_batch_size: int,
    config: MultiRadiusDerivativeAuditConfig,
) -> MultiRadiusDerivativeAuditResult:
    """No-update one-sided signed-vector audit over decreasing radii."""

    if not isinstance(model.control_map, SpecificEntropyDiagonalControl):
        raise TypeError("multiradius derivative audit requires specific entropy")
    if model.control_map.sigma_max is None:
        raise ValueError("multiradius derivative audit requires bounded volatility")
    blocks = time_block_bounds(
        num_steps=int(model.grid.num_steps),
        num_blocks=int(config.num_time_blocks),
    )
    model_before = model_checkpoint_sha256(model)
    law_network = getattr(law_policy, "network", None)
    adjoint_network = getattr(adjoint_policy, "network", None)
    law_before = (
        None
        if not isinstance(law_network, torch.nn.Module)
        else network_sha256(law_network)
    )
    adjoint_before = (
        None
        if not isinstance(adjoint_network, torch.nn.Module)
        else network_sha256(adjoint_network)
    )

    baseline_metrics: dict[str, dict[str, object]] = {}
    baseline_artifacts: dict[str, dict[str, torch.Tensor]] = {}
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
        baseline_metrics[bank_name] = metrics
        baseline_artifacts[bank_name] = artifacts
    scales = _residual_normalization_scales(baseline_artifacts["fit"])

    coordinates = [
        (signal, block_index)
        for signal in ("alpha", "log_sigma")
        for block_index in range(len(blocks))
    ]
    surfaces: dict[str, dict[str, dict[str, object]]] = {
        "fit": {},
        "confirmation": {},
    }
    candidate_artifacts: dict[str, dict[str, dict[str, object]]] = {
        "fit": {},
        "confirmation": {},
    }
    derivative_artifacts: dict[str, dict[str, dict[str, object]]] = {
        "fit": {},
        "confirmation": {},
    }
    for epsilon in config.epsilons:
        radius = f"{float(epsilon):.10g}"
        for bank_name in surfaces:
            surfaces[bank_name][radius] = {}
            candidate_artifacts[bank_name][radius] = {}
            derivative_artifacts[bank_name][radius] = {}
        for signal, block_index in coordinates:
            coordinate = f"{signal}_block_{block_index}"
            for bank_name in surfaces:
                surfaces[bank_name][radius][coordinate] = {
                    "coordinate": coordinate,
                    "signal": signal,
                    "block_index": int(block_index),
                    "block_steps": list(blocks[block_index]),
                }
                candidate_artifacts[bank_name][radius][coordinate] = {}
            for sign, side in ((-1.0, "minus"), (1.0, "plus")):
                alpha = [0.0] * len(blocks)
                log_sigma = [0.0] * len(blocks)
                coefficients = alpha if signal == "alpha" else log_sigma
                coefficients[block_index] = sign * float(epsilon)
                policy = BlockwiseControlPerturbationPolicy(
                    model=model,
                    law_policy=law_policy,
                    adjoint_policy=adjoint_policy,
                    blocks=blocks,
                    alpha_coefficients=tuple(alpha),
                    log_sigma_coefficients=tuple(log_sigma),
                )
                for bank_name, bank in (
                    ("fit", fit_bank),
                    ("confirmation", confirmation_bank),
                ):
                    evaluation = rollout_policy_on_bank(
                        model=model,
                        policy=policy,
                        bank=bank,
                        data_context=data_context,
                        full_training=full_training,
                        query_batch_size=query_batch_size,
                    )
                    baseline_sets = {
                        name: baseline_artifacts[bank_name][f"{name}_active_set"]
                        for name in (
                            "adjoint_target",
                            "conditional_target",
                            "actual_control",
                        )
                    }
                    metrics, artifacts = _directional_metrics(
                        model=model,
                        evaluation=evaluation,
                        law_policy=law_policy,
                        adjoint_policy=adjoint_policy,
                        gamma=float(config.projected_kkt_gamma),
                        baseline_active_sets=baseline_sets,
                    )
                    surfaces[bank_name][radius][coordinate][side] = metrics
                    candidate_artifacts[bank_name][radius][coordinate][side] = artifacts
            for bank_name in surfaces:
                row = candidate_artifacts[bank_name][radius][coordinate]
                full_columns = _one_sided_columns(
                    baseline=baseline_artifacts[bank_name],
                    minus=row["minus"],
                    plus=row["plus"],
                    scales=scales,
                    epsilon=float(epsilon),
                    profile=False,
                )
                profile_columns = _one_sided_columns(
                    baseline=baseline_artifacts[bank_name],
                    minus=row["minus"],
                    plus=row["plus"],
                    scales=scales,
                    epsilon=float(epsilon),
                    profile=True,
                )
                derivative_artifacts[bank_name][radius][coordinate] = {
                    "full": full_columns,
                    "profile": profile_columns,
                }
                surfaces[bank_name][radius][coordinate]["one_sided_asymmetry"] = {
                    system: compare_signed_columns(
                        full_columns["minus"][system],
                        full_columns["plus"][system],
                    )
                    for system in full_columns["plus"]
                }

    radius_pairs = list(zip(config.epsilons, config.epsilons[1:]))
    radius_convergence: dict[str, dict[str, object]] = {
        "fit": {},
        "confirmation": {},
    }
    for bank_name in radius_convergence:
        for coordinate in surfaces[bank_name][f"{config.epsilons[0]:.10g}"]:
            radius_convergence[bank_name][coordinate] = {}
            for side in ("minus", "plus"):
                radius_convergence[bank_name][coordinate][side] = {}
                for system in derivative_artifacts[bank_name][
                    f"{config.epsilons[0]:.10g}"
                ][coordinate]["full"][side]:
                    comparisons = []
                    for larger, smaller in radius_pairs:
                        large_key = f"{larger:.10g}"
                        small_key = f"{smaller:.10g}"
                        comparison = compare_signed_columns(
                            derivative_artifacts[bank_name][large_key][coordinate][
                                "full"
                            ][side][system],
                            derivative_artifacts[bank_name][small_key][coordinate][
                                "full"
                            ][side][system],
                        )
                        comparison.update(
                            {
                                "larger_epsilon": float(larger),
                                "smaller_epsilon": float(smaller),
                                "passes_locked_gate": _comparison_gate(
                                    comparison, config=config
                                ),
                            }
                        )
                        comparisons.append(comparison)
                    radius_convergence[bank_name][coordinate][side][system] = comparisons

    bank_agreement: dict[str, dict[str, object]] = {}
    for epsilon in config.epsilons:
        radius = f"{epsilon:.10g}"
        bank_agreement[radius] = {}
        for coordinate in surfaces["fit"][radius]:
            bank_agreement[radius][coordinate] = {}
            for side in ("minus", "plus"):
                bank_agreement[radius][coordinate][side] = {}
                for system in derivative_artifacts["fit"][radius][coordinate][
                    "profile"
                ][side]:
                    comparison = compare_signed_columns(
                        derivative_artifacts["fit"][radius][coordinate]["profile"][
                            side
                        ][system],
                        derivative_artifacts["confirmation"][radius][coordinate][
                            "profile"
                        ][side][system],
                    )
                    comparison["passes_locked_gate"] = _comparison_gate(
                        comparison, config=config
                    )
                    bank_agreement[radius][coordinate][side][system] = comparison

    decision_systems = (
        "lifted_consensus",
        "adjoint_projected_kkt_residual",
        "target_projected_kkt_residual",
    )
    convergence_summary = {}
    bank_summary = {}
    final_pair_index = len(radius_pairs) - 1
    smallest_radius = f"{config.epsilons[-1]:.10g}"
    for system in decision_systems:
        convergence_values = [
            radius_convergence[bank][coordinate][side][system][final_pair_index]
            for bank in ("fit", "confirmation")
            for coordinate in radius_convergence[bank]
            for side in ("minus", "plus")
        ]
        bank_values = [
            bank_agreement[smallest_radius][coordinate][side][system]
            for coordinate in bank_agreement[smallest_radius]
            for side in ("minus", "plus")
        ]
        convergence_summary[system] = _aggregate_comparisons(
            convergence_values, config=config
        )
        bank_summary[system] = _aggregate_comparisons(bank_values, config=config)
    all_stable = all(
        bool(convergence_summary[system]["all_pass"])
        and bool(bank_summary[system]["all_pass"])
        for system in decision_systems
    )

    model_after = model_checkpoint_sha256(model)
    law_after = (
        None
        if not isinstance(law_network, torch.nn.Module)
        else network_sha256(law_network)
    )
    adjoint_after = (
        None
        if not isinstance(adjoint_network, torch.nn.Module)
        else network_sha256(adjoint_network)
    )
    nonmutation = {
        "model_checkpoint_before": model_before,
        "model_checkpoint_after": model_after,
        "model_unchanged": model_before == model_after,
        "law_policy_before": law_before,
        "law_policy_after": law_after,
        "law_policy_unchanged": law_before == law_after,
        "adjoint_policy_before": adjoint_before,
        "adjoint_policy_after": adjoint_after,
        "adjoint_policy_unchanged": adjoint_before == adjoint_after,
    }
    if not all(
        bool(nonmutation[name])
        for name in (
            "model_unchanged",
            "law_policy_unchanged",
            "adjoint_policy_unchanged",
        )
    ):
        raise RuntimeError("multiradius derivative audit mutated model state")
    return MultiRadiusDerivativeAuditResult(
        report={
            "protocol": {
                "solver_update_applied": False,
                "signed_residual_vectors": True,
                "full_columns_used_within_bank_across_radii": True,
                "timewise_mean_profiles_used_across_independent_banks": True,
                "candidate_law_and_conditional_targets_recomputed": True,
                "normalization_scales_frozen_from_fit_baseline": True,
                "common_random_numbers_within_bank": True,
                "fit_and_confirmation_banks_independent": True,
            },
            "config": {
                "epsilons": list(config.epsilons),
                "num_time_blocks": int(config.num_time_blocks),
                "projected_kkt_gamma": float(config.projected_kkt_gamma),
                "minimum_stable_cosine": float(config.minimum_stable_cosine),
                "maximum_stable_relative_difference": float(
                    config.maximum_stable_relative_difference
                ),
            },
            "normalization_scales": scales,
            "blocks": [list(block) for block in blocks],
            "baseline": baseline_metrics,
            "surfaces": surfaces,
            "radius_convergence": radius_convergence,
            "bank_agreement": bank_agreement,
            "decision": {
                "systems": list(decision_systems),
                "small_radius_convergence": convergence_summary,
                "smallest_radius_bank_agreement": bank_summary,
                "one_sided_columns_stable": all_stable,
                "recommended_next_solver": (
                    "orthant_enumerated_kl_regularized_least_squares"
                    if all_stable
                    else "derivative_free_trust_region"
                ),
                "recommendation_is_diagnostic_not_deployed": True,
            },
            "nonmutation": nonmutation,
            "dataset_fingerprint": data_context.fingerprint,
            "fit_bank_fingerprint": fit_bank.fingerprint,
            "confirmation_bank_fingerprint": confirmation_bank.fingerprint,
        },
        artifacts={
            "baseline": baseline_artifacts,
            "candidate_residuals": candidate_artifacts,
            "derivative_columns": derivative_artifacts,
        },
    )


__all__ = [
    "AdjointNetworkControlPolicy",
    "BlockwiseControlPerturbationPolicy",
    "CausalControlPolicy",
    "DirectionalActiveSetAuditConfig",
    "DirectionalActiveSetAuditResult",
    "LawEvaluation",
    "LiftedConsensusAuditResult",
    "MultiRadiusDerivativeAuditConfig",
    "MultiRadiusDerivativeAuditResult",
    "PolicyStepResult",
    "ScenarioCollocationBank",
    "ScenarioDataContext",
    "audit_directional_active_set_surface",
    "audit_lifted_consensus",
    "audit_multiradius_signed_derivatives",
    "build_fit_confirmation_banks",
    "build_scenario_collocation_bank",
    "compare_signed_columns",
    "model_checkpoint_sha256",
    "network_sha256",
    "policy_moments_on_paths",
    "projected_log_volatility_kkt",
    "replay_branch_noise",
    "rollout_causal_policy",
    "rollout_policy_on_bank",
    "stable_sha256",
    "time_block_bounds",
]
