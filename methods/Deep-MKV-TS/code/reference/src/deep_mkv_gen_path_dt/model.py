from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable

import torch

from deep_mkv_gen_path_dt.ce import ConditionalExpectationEstimator, RidgeConditionalExpectation
from deep_mkv_gen_path_dt.config import (
    DiscreteMPArchitectureConfig,
    DiscreteMPNestedTrainingConfig,
    DiscreteMPTrainingConfig,
    _require_int,
    _require_optional_int,
)
from deep_mkv_gen_path_dt.controls import (
    DynamicsStepInputs,
    DynamicsStepMap,
    HamiltonianControlInputs,
    HamiltonianControlMap,
    PathDependentRunningCost,
    RunningCostInputs,
    RunningCostResult,
)
from deep_mkv_gen_path_dt.discrepancies import PathFunctionalDiscrepancy
from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid
from deep_mkv_gen_path_dt.networks import AdjointMomentNetwork, AdjointMoments
from deep_mkv_gen_path_dt.noise import derive_stream_seed, sample_standard_normals
from deep_mkv_gen_path_dt.paths import future_source_sums, scatter_observed_sources


@dataclass(frozen=True)
class FitResult:
    history: list[dict[str, float]]
    final_metrics: dict[str, float]


@dataclass(frozen=True)
class SampleResult:
    paths: torch.Tensor
    controls: torch.Tensor
    expected_adjoint_next: torch.Tensor
    expected_adjoint_noise_next: torch.Tensor
    noise: torch.Tensor


@dataclass(frozen=True)
class RolloutResult:
    paths: torch.Tensor
    controls: torch.Tensor
    expected_adjoint_next: torch.Tensor
    expected_adjoint_noise_next: torch.Tensor
    noise: torch.Tensor


@dataclass(frozen=True)
class FitReplayBank:
    """Optional fixed empirical bank for a replayable :meth:`fit` run.

    The default production path continues to sample target indices and
    standard-Gaussian innovations at every optimizer step.  Supplying this
    bank instead freezes the empirical source/target populations and the
    exogenous innovations.  This is useful for finite-tree solver audits in
    which the learned model and the exact oracle must solve the same discrete
    control problem.
    """

    source_indices: torch.Tensor
    target_indices: torch.Tensor
    noise: torch.Tensor


@dataclass(frozen=True)
class ConditionalAdjointTargets:
    """Externally supplied conditional adjoint labels for a fitted rollout."""

    expected_adjoint_next: torch.Tensor
    expected_adjoint_noise_next: torch.Tensor
    metrics: dict[str, float]


class DiscreteMPModel:
    """
    Entry point for the path-dependent discrete-time maximum-principle model.

    The model owns the adjoint moment network and orchestrates training and
    sampling. Experiment-specific data generation, metrics, and plots are kept
    outside this package.
    """

    def __init__(
        self,
        *,
        architecture: DiscreteMPArchitectureConfig,
        grid: DiscreteTimeGrid,
        dynamics_step: DynamicsStepMap,
        control_map: HamiltonianControlMap,
        discrepancy: PathFunctionalDiscrepancy,
        running_cost: PathDependentRunningCost | None = None,
        ce_estimator: ConditionalExpectationEstimator | None = None,
        noise_ce_estimator: ConditionalExpectationEstimator | None = None,
        training: DiscreteMPTrainingConfig | None = None,
        network: AdjointMomentNetwork | None = None,
        init_seed: int | None = None,
        parameters: object | None = None,
    ) -> None:
        if int(architecture.state_dim) < 1:
            raise ValueError("architecture.state_dim must be >= 1")
        init_seed = _require_optional_int("init_seed", init_seed)
        self._validate_architecture_contract(architecture)
        if network is not None and init_seed is not None:
            raise ValueError("init_seed only applies when the model constructs the default network")
        self.architecture = architecture
        self.grid = grid
        self.dynamics_step = dynamics_step
        self.control_map = control_map
        self.discrepancy = discrepancy
        inferred_running_cost = getattr(control_map, "running_cost", None)
        self.running_cost = running_cost if running_cost is not None else inferred_running_cost
        if self.running_cost is not None and not callable(self.running_cost):
            raise ValueError("running_cost must be callable")
        self.ce_estimator = ce_estimator or RidgeConditionalExpectation()
        # When omitted, R uses the same conditional estimator as P.  Keeping
        # the override separate lets experiments improve the innovation-
        # weighted R projection without changing the drift/P projection.
        self.noise_ce_estimator = noise_ce_estimator
        self.training = training or DiscreteMPTrainingConfig()
        self.parameters = parameters
        if network is not None:
            self._validate_network_contract(network=network, architecture=architecture)
            self.network = network
        elif init_seed is None:
            self.network = self._make_network(architecture)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(init_seed))
                self.network = self._make_network(architecture)
        self._adjoint_target_mean: torch.Tensor | None = None
        self._adjoint_target_scale: torch.Tensor | None = None
        self._noise_adjoint_target_mean: torch.Tensor | None = None
        self._noise_adjoint_target_scale: torch.Tensor | None = None
        self._noise_target_timewise_baseline: torch.Tensor | None = None
        self._target_preconditioner_metrics: dict[str, float] = {}

    @staticmethod
    def _validate_architecture_contract(architecture: DiscreteMPArchitectureConfig) -> None:
        state_dim = int(architecture.state_dim)
        noise_dim = int(architecture.noise_dim)
        if architecture.adjoint_dim is not None and int(architecture.adjoint_dim) != state_dim:
            raise ValueError("training requires adjoint_dim to equal state_dim")
        if architecture.noise_adjoint_dim is None:
            return
        noise_adjoint_dim = int(architecture.noise_adjoint_dim)
        full_dim = state_dim * noise_dim
        if noise_adjoint_dim == full_dim:
            return
        if noise_dim == state_dim and noise_adjoint_dim == state_dim:
            return
        raise ValueError(
            "noise_adjoint_dim must be state_dim for diagonal controls with noise_dim == state_dim, "
            "or state_dim * noise_dim for full-matrix controls"
        )

    @staticmethod
    def _default_noise_adjoint_dim(architecture: DiscreteMPArchitectureConfig) -> int:
        if architecture.noise_adjoint_dim is not None:
            return int(architecture.noise_adjoint_dim)
        state_dim = int(architecture.state_dim)
        noise_dim = int(architecture.noise_dim)
        return state_dim if noise_dim == state_dim else state_dim * noise_dim

    @staticmethod
    def _validate_network_contract(*, network: AdjointMomentNetwork, architecture: DiscreteMPArchitectureConfig) -> None:
        state_dim = int(architecture.state_dim)
        noise_dim = int(architecture.noise_dim)
        if hasattr(network, "input_mode") and str(network.input_mode) != str(
            architecture.adjoint_input_mode
        ):
            raise ValueError("network input_mode must match architecture.adjoint_input_mode")
        if hasattr(network, "adjoint_dim") and int(network.adjoint_dim) != state_dim:
            raise ValueError("network adjoint_dim must equal state_dim")
        if not hasattr(network, "noise_adjoint_dim"):
            return
        noise_adjoint_dim = int(network.noise_adjoint_dim)
        full_dim = state_dim * noise_dim
        if architecture.noise_adjoint_dim is not None and noise_adjoint_dim != int(architecture.noise_adjoint_dim):
            raise ValueError("network noise_adjoint_dim must match architecture.noise_adjoint_dim")
        if noise_adjoint_dim == full_dim:
            return
        if noise_dim == state_dim and noise_adjoint_dim == state_dim:
            return
        raise ValueError(
            "network noise_adjoint_dim must be state_dim for diagonal controls with noise_dim == state_dim, "
            "or state_dim * noise_dim for full-matrix controls"
        )

    @staticmethod
    def _make_network(architecture: DiscreteMPArchitectureConfig) -> AdjointMomentNetwork:
        return AdjointMomentNetwork(
            state_dim=int(architecture.state_dim),
            adjoint_dim=int(architecture.state_dim),
            hidden_dim=int(architecture.hidden_dim),
            num_layers=int(architecture.num_layers),
            noise_adjoint_dim=DiscreteMPModel._default_noise_adjoint_dim(architecture),
            input_mode=str(architecture.adjoint_input_mode),
        )

    @property
    def device(self) -> torch.device:
        return next(self.network.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.network.parameters()).dtype

    def checkpoint_state(self) -> dict[str, object]:
        """Return the fitted state needed to reproduce this model's rollouts.

        The timewise target scales and noise-target control-variate baseline
        affect the adjoint moments passed to the control map.  Saving only the
        network ``state_dict`` would therefore not reproduce a fitted model.
        """

        def cpu_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.detach().cpu().clone()

        return {
            "format_version": 1,
            "architecture": asdict(self.architecture),
            "training": asdict(self.training),
            "network_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in self.network.state_dict().items()
            },
            "adjoint_target_mean": cpu_tensor(self._adjoint_target_mean),
            "adjoint_target_scale": cpu_tensor(self._adjoint_target_scale),
            "noise_adjoint_target_mean": cpu_tensor(self._noise_adjoint_target_mean),
            "noise_adjoint_target_scale": cpu_tensor(self._noise_adjoint_target_scale),
            "noise_target_timewise_baseline": cpu_tensor(self._noise_target_timewise_baseline),
            "target_preconditioner_metrics": dict(self._target_preconditioner_metrics),
            "adjoint_source": {
                "complete_path_dependent": True,
                "running_cost": self._running_cost_name(),
            },
        }

    def _running_cost_name(self) -> str | None:
        if self.running_cost is None:
            return None
        owner = getattr(self.running_cost, "__self__", None)
        if owner is not None:
            return f"{type(owner).__module__}.{type(owner).__qualname__}.running_cost"
        return f"{type(self.running_cost).__module__}.{type(self.running_cost).__qualname__}"

    def load_checkpoint_state(self, checkpoint: dict[str, object]) -> None:
        """Restore a payload produced by :meth:`checkpoint_state`.

        Model construction remains explicit because the dynamics, control map,
        discrepancy, and any experiment parameters are user-supplied objects.
        The architecture is checked before mutable fitted state is restored.
        """

        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint must be a dictionary")
        if checkpoint.get("format_version") != 1:
            raise ValueError("unsupported checkpoint format_version")
        architecture = checkpoint.get("architecture")
        # Format-version-one checkpoints written before the input representation
        # option implicitly used levels. Preserve their exact loading behavior.
        if isinstance(architecture, dict):
            architecture = dict(architecture)
            architecture.setdefault("adjoint_input_mode", "level")
        if architecture != asdict(self.architecture):
            raise ValueError("checkpoint architecture does not match model architecture")
        adjoint_source = checkpoint.get("adjoint_source")
        if adjoint_source is not None:
            if not isinstance(adjoint_source, dict):
                raise ValueError("checkpoint adjoint_source must be a dictionary")
            if not bool(adjoint_source.get("complete_path_dependent", False)):
                raise ValueError("checkpoint was not trained with the complete path-dependent adjoint source")
            if adjoint_source.get("running_cost") != self._running_cost_name():
                raise ValueError("checkpoint running cost does not match model running cost")
        network_state = checkpoint.get("network_state_dict")
        if not isinstance(network_state, dict):
            raise ValueError("checkpoint network_state_dict must be a dictionary")
        self.network.load_state_dict(network_state)

        training_payload = checkpoint.get("training")
        if not isinstance(training_payload, dict):
            raise ValueError("checkpoint training must be a dictionary")
        self.training = DiscreteMPTrainingConfig(**training_payload)

        def restore_tensor(name: str) -> torch.Tensor | None:
            value = checkpoint.get(name)
            if value is None:
                return None
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"checkpoint {name} must be a tensor or None")
            return value.detach().to(device=self.device, dtype=self.dtype).clone()

        self._adjoint_target_mean = restore_tensor("adjoint_target_mean")
        self._adjoint_target_scale = restore_tensor("adjoint_target_scale")
        self._noise_adjoint_target_mean = restore_tensor("noise_adjoint_target_mean")
        self._noise_adjoint_target_scale = restore_tensor("noise_adjoint_target_scale")
        self._noise_target_timewise_baseline = restore_tensor("noise_target_timewise_baseline")
        metrics = checkpoint.get("target_preconditioner_metrics", {})
        if not isinstance(metrics, dict):
            raise ValueError("checkpoint target_preconditioner_metrics must be a dictionary")
        self._target_preconditioner_metrics = {
            str(name): float(value)
            for name, value in metrics.items()
        }

    def _validate_target_paths(self, target_paths: torch.Tensor) -> tuple[int, int]:
        if target_paths.ndim != 3:
            raise ValueError("target_paths must have shape (B, N + 1, d)")
        if int(target_paths.shape[1]) != int(self.grid.num_steps) + 1:
            raise ValueError("target_paths time dimension must align with grid")
        if int(target_paths.shape[0]) < 1:
            raise ValueError("target_paths must contain at least one path")
        if int(target_paths.shape[2]) != int(self.architecture.state_dim):
            raise ValueError("target path state dimension must match architecture.state_dim")
        return int(target_paths.shape[0]), int(target_paths.shape[2])

    @staticmethod
    def _sample_indices(*, size: int, count: int, generator: torch.Generator) -> torch.Tensor:
        size = _require_int("size", size)
        count = _require_int("count", count)
        if size < 1:
            raise ValueError("size must be >= 1")
        if count < 1:
            raise ValueError("count must be >= 1")
        return torch.randint(size, (count,), generator=generator, dtype=torch.long, device="cpu")

    def _validated_fit_replay_bank(
        self,
        *,
        replay_bank: FitReplayBank,
        target_count: int,
        training: DiscreteMPTrainingConfig,
    ) -> FitReplayBank:
        if not isinstance(replay_bank, FitReplayBank):
            raise TypeError("replay_bank must be a FitReplayBank")

        def indices(name: str, value: torch.Tensor, expected_count: int) -> torch.Tensor:
            if not isinstance(value, torch.Tensor) or value.ndim != 1:
                raise ValueError(f"replay_bank.{name} must be a one-dimensional tensor")
            if value.dtype not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }:
                raise ValueError(f"replay_bank.{name} must contain integer indices")
            if int(value.numel()) != int(expected_count):
                raise ValueError(
                    f"replay_bank.{name} must contain exactly {expected_count} indices"
                )
            result = value.detach().to(device="cpu", dtype=torch.long).clone()
            if bool(torch.any(result < 0).item()) or bool(
                torch.any(result >= int(target_count)).item()
            ):
                raise ValueError(f"replay_bank.{name} contains an out-of-range index")
            return result

        source_indices = indices(
            "source_indices", replay_bank.source_indices, int(training.batch_size)
        )
        target_indices = indices(
            "target_indices",
            replay_bank.target_indices,
            int(training.target_batch_size),
        )
        noise = replay_bank.noise
        expected_noise_shape = (
            int(training.batch_size),
            int(self.grid.num_steps),
            int(self.architecture.noise_dim),
        )
        if not isinstance(noise, torch.Tensor) or tuple(noise.shape) != expected_noise_shape:
            raise ValueError(
                "replay_bank.noise must have shape "
                f"{expected_noise_shape}"
            )
        if not bool(torch.isfinite(noise).all().item()):
            raise ValueError("replay_bank.noise must be finite")
        return FitReplayBank(
            source_indices=source_indices,
            target_indices=target_indices,
            noise=noise.detach().to(device=self.device, dtype=self.dtype).clone(),
        )

    def _provided_conditional_adjoint_targets(
        self,
        *,
        provider: Callable[
            ["DiscreteMPModel", RolloutResult, torch.Tensor],
            ConditionalAdjointTargets,
        ],
        rollout: RolloutResult,
        target_paths: torch.Tensor,
    ) -> ConditionalAdjointTargets:
        result = provider(self, rollout, target_paths)
        if not isinstance(result, ConditionalAdjointTargets):
            raise TypeError(
                "conditional_target_provider must return ConditionalAdjointTargets"
            )
        expected_p_shape = tuple(rollout.expected_adjoint_next.shape)
        expected_r_shape = tuple(rollout.expected_adjoint_noise_next.shape)
        if tuple(result.expected_adjoint_next.shape) != expected_p_shape:
            raise ValueError(
                "provided expected_adjoint_next must have shape "
                f"{expected_p_shape}"
            )
        if tuple(result.expected_adjoint_noise_next.shape) != expected_r_shape:
            raise ValueError(
                "provided expected_adjoint_noise_next must have shape "
                f"{expected_r_shape}"
            )
        p = result.expected_adjoint_next.detach().to(
            device=self.device, dtype=self.dtype
        )
        r = result.expected_adjoint_noise_next.detach().to(
            device=self.device, dtype=self.dtype
        )
        if not bool(torch.isfinite(p).all().item()) or not bool(
            torch.isfinite(r).all().item()
        ):
            raise ValueError("provided conditional adjoint targets must be finite")
        return ConditionalAdjointTargets(
            expected_adjoint_next=p,
            expected_adjoint_noise_next=r,
            metrics={str(name): float(value) for name, value in result.metrics.items()},
        )

    @staticmethod
    def _grad_norm(parameters) -> float:
        total = 0.0
        for parameter in parameters:
            if parameter.grad is None:
                continue
            total += float(torch.sum(parameter.grad.detach().pow(2)).item())
        return total**0.5

    @staticmethod
    def _parameter_norm(parameters) -> float:
        total = 0.0
        for parameter in parameters:
            total += float(torch.sum(parameter.detach().pow(2)).item())
        return total**0.5

    @staticmethod
    def _update_norm(parameters, previous_values: list[torch.Tensor]) -> float:
        total = 0.0
        for parameter, previous in zip(parameters, previous_values):
            total += float(torch.sum((parameter.detach() - previous).pow(2)).item())
        return total**0.5

    def _component_parameters(self) -> dict[str, list[torch.nn.Parameter]]:
        groups: dict[str, list[torch.nn.Parameter]] = {
            "gru": [],
            "adjoint_head": [],
            "noise_adjoint_head": [],
            "other": [],
        }
        for name, parameter in self.network.named_parameters():
            if name.startswith("gru.") or ".gru." in name:
                group = "gru"
            elif name.startswith("expected_adjoint_next_head.") or (
                ".expected_adjoint_next_head." in name
            ) or name.startswith("expected_adjoint_next_heads.") or (
                ".expected_adjoint_next_heads." in name
            ):
                group = "adjoint_head"
            elif name.startswith("expected_adjoint_noise_next_head.") or (
                ".expected_adjoint_noise_next_head." in name
            ) or name.startswith("expected_adjoint_noise_next_heads.") or (
                ".expected_adjoint_noise_next_heads." in name
            ):
                group = "noise_adjoint_head"
            else:
                group = "other"
            groups[group].append(parameter)
        return {name: parameters for name, parameters in groups.items() if parameters}

    def _component_grad_norms(self, *, prefix: str = "grad_norm") -> dict[str, float]:
        return {
            f"{prefix}_{name}": self._grad_norm(parameters)
            for name, parameters in self._component_parameters().items()
        }

    def _component_update_norms(
        self,
        previous_values: dict[int, torch.Tensor],
    ) -> dict[str, float]:
        return {
            f"parameter_update_norm_{name}": self._update_norm(
                parameters,
                [previous_values[id(parameter)] for parameter in parameters],
            )
            for name, parameters in self._component_parameters().items()
        }

    @staticmethod
    def _nonfinite_fraction(tensors: list[torch.Tensor]) -> float:
        total = 0
        nonfinite = 0
        for tensor in tensors:
            values = tensor.detach()
            total += int(values.numel())
            nonfinite += int((~torch.isfinite(values)).sum().item())
        return 0.0 if total == 0 else float(nonfinite / total)

    @staticmethod
    def _moment_diagnostics(
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        prefix: str,
    ) -> dict[str, float]:
        prediction_detached = prediction.detach()
        target_detached = target.detach()
        prediction_rms = torch.sqrt(torch.mean(prediction_detached.pow(2)))
        target_rms = torch.sqrt(torch.mean(target_detached.pow(2)))
        error_rms = torch.sqrt(torch.mean((prediction_detached - target_detached).pow(2)))
        eps = torch.finfo(target_detached.dtype).eps
        metrics = {
            f"{prefix}_prediction_rms": float(prediction_rms.item()),
            f"{prefix}_target_rms": float(target_rms.item()),
            f"{prefix}_error_rms": float(error_rms.item()),
            f"{prefix}_relative_rmse": float((error_rms / target_rms.clamp_min(eps)).item()),
            f"{prefix}_prediction_target_rms_ratio": float(
                (prediction_rms / target_rms.clamp_min(eps)).item()
            ),
        }
        num_steps = int(target_detached.shape[1])
        boundaries = (0, num_steps // 3, (2 * num_steps) // 3, num_steps)
        for segment, start, stop in zip(("early", "middle", "late"), boundaries[:-1], boundaries[1:]):
            if stop <= start:
                continue
            segment_prediction = prediction_detached[:, start:stop, :]
            segment_target = target_detached[:, start:stop, :]
            segment_target_rms = torch.sqrt(torch.mean(segment_target.pow(2)))
            segment_error_rms = torch.sqrt(torch.mean((segment_prediction - segment_target).pow(2)))
            metrics[f"{prefix}_{segment}_target_rms"] = float(segment_target_rms.item())
            metrics[f"{prefix}_{segment}_relative_rmse"] = float(
                (segment_error_rms / segment_target_rms.clamp_min(eps)).item()
            )
        return metrics

    def _control_diagnostics(self, controls: torch.Tensor) -> dict[str, float]:
        state_dim = int(self.architecture.state_dim)
        if controls.ndim != 3 or int(controls.shape[-1]) != 2 * state_dim:
            return {}
        sigma = controls[..., state_dim:].detach()
        quantiles = torch.quantile(
            sigma.reshape(-1),
            torch.tensor([0.01, 0.50, 0.99], device=sigma.device, dtype=sigma.dtype),
        )
        metrics = {
            "sigma_mean": float(sigma.mean().item()),
            "sigma_std": float(sigma.std(unbiased=False).item()),
            "sigma_q01": float(quantiles[0].item()),
            "sigma_q50": float(quantiles[1].item()),
            "sigma_q99": float(quantiles[2].item()),
            "sigma_nonfinite_fraction": float((~torch.isfinite(sigma)).float().mean().item()),
        }
        for name in ("sigma_min", "sigma_max"):
            bound = getattr(self.control_map, name, None)
            if bound is None:
                continue
            tolerance = max(1e-7, abs(float(bound)) * 1e-5)
            if name == "sigma_min":
                active = sigma <= float(bound) + tolerance
            else:
                active = sigma >= float(bound) - tolerance
            metrics[f"{name}_active_fraction"] = float(active.float().mean().item())
        return metrics

    @staticmethod
    def _prediction_r2(prediction: torch.Tensor, target: torch.Tensor) -> float:
        prediction = prediction.detach()
        target = target.detach()
        residual_sum_squares = torch.sum((prediction - target).pow(2))
        timewise_mean = target.mean(dim=0, keepdim=True)
        baseline_sum_squares = torch.sum((target - timewise_mean).pow(2))
        eps = torch.finfo(target.dtype).eps
        return float((1.0 - residual_sum_squares / baseline_sum_squares.clamp_min(eps)).item())

    @staticmethod
    def _rolling_diagnostic_summary(values_by_name: dict[str, list[float]]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for name, values in values_by_name.items():
            if not values:
                continue
            tensor = torch.tensor(values, dtype=torch.float64)
            metrics[f"rolling_{name}_mean"] = float(tensor.mean().item())
            metrics[f"rolling_{name}_q50"] = float(torch.quantile(tensor, 0.50).item())
            metrics[f"rolling_{name}_q95"] = float(torch.quantile(tensor, 0.95).item())
            metrics[f"rolling_{name}_max"] = float(tensor.max().item())
        if "clip_active" in values_by_name and values_by_name["clip_active"]:
            metrics["rolling_grad_clip_fraction"] = float(
                sum(values_by_name["clip_active"]) / len(values_by_name["clip_active"])
            )
        metrics["rolling_window_steps"] = float(
            max((len(values) for values in values_by_name.values()), default=0)
        )
        return metrics

    @staticmethod
    def _abs_tensor_stats(tensor: torch.Tensor, *, prefix: str) -> dict[str, float]:
        values = tensor.detach().abs().reshape(-1)
        if int(values.numel()) == 0:
            return {}
        quantiles = torch.quantile(
            values,
            torch.tensor([0.5, 0.95, 0.99], device=values.device, dtype=values.dtype),
        )
        return {
            f"{prefix}_abs_mean": float(values.mean().item()),
            f"{prefix}_abs_rms": float(torch.sqrt(torch.mean(values.pow(2))).item()),
            f"{prefix}_abs_q50": float(quantiles[0].item()),
            f"{prefix}_abs_q95": float(quantiles[1].item()),
            f"{prefix}_abs_q99": float(quantiles[2].item()),
            f"{prefix}_abs_max": float(values.max().item()),
        }

    def _reset_target_preconditioner(self) -> None:
        self._adjoint_target_mean = None
        self._adjoint_target_scale = None
        self._noise_adjoint_target_mean = None
        self._noise_adjoint_target_scale = None
        self._noise_target_timewise_baseline = None
        self._target_preconditioner_metrics = {
            "target_preconditioner_timewise": 0.0,
            "noise_target_control_variate_timewise": 0.0,
        }

    def _target_preconditioner_tensors(
        self,
        *,
        noise_adjoint: bool,
        like: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if noise_adjoint:
            mean = self._noise_adjoint_target_mean
            scale = self._noise_adjoint_target_scale
        else:
            mean = self._adjoint_target_mean
            scale = self._adjoint_target_scale
        if mean is None or scale is None:
            return None
        return (
            mean.to(device=like.device, dtype=like.dtype),
            scale.to(device=like.device, dtype=like.dtype),
        )

    def _normalize_adjoint_target(
        self,
        target: torch.Tensor,
        *,
        noise_adjoint: bool,
    ) -> torch.Tensor:
        tensors = self._target_preconditioner_tensors(noise_adjoint=noise_adjoint, like=target)
        if tensors is None:
            return target
        mean, scale = tensors
        if tuple(target.shape[1:]) != tuple(mean.shape):
            raise ValueError(
                "target preconditioner shape does not match target: "
                f"expected (*, {tuple(mean.shape)}) but received {tuple(target.shape)}"
            )
        # A centered target would give the same squared residual after applying
        # the identical affine transform to prediction and target.  Scaling
        # only keeps a zero network output equal to a zero raw adjoint, which
        # avoids injecting the calibration mean into the control at startup.
        return target / scale.unsqueeze(0)

    def _denormalize_adjoint_prediction(
        self,
        prediction: torch.Tensor,
        *,
        noise_adjoint: bool,
    ) -> torch.Tensor:
        tensors = self._target_preconditioner_tensors(noise_adjoint=noise_adjoint, like=prediction)
        if tensors is None:
            return prediction
        mean, scale = tensors
        if tuple(prediction.shape[1:]) != tuple(mean.shape):
            raise ValueError(
                "target preconditioner shape does not match prediction: "
                f"expected (*, {tuple(mean.shape)}) but received {tuple(prediction.shape)}"
            )
        return scale.unsqueeze(0) * prediction

    def _denormalize_moments(self, moments: AdjointMoments) -> AdjointMoments:
        return AdjointMoments(
            expected_adjoint_next=self._denormalize_adjoint_prediction(
                moments.expected_adjoint_next,
                noise_adjoint=False,
            ),
            expected_adjoint_noise_next=self._denormalize_adjoint_prediction(
                moments.expected_adjoint_noise_next,
                noise_adjoint=True,
            ),
        )

    def _denormalize_adjoint_step(
        self,
        prediction: torch.Tensor,
        *,
        step_index: int,
        noise_adjoint: bool,
    ) -> torch.Tensor:
        tensors = self._target_preconditioner_tensors(noise_adjoint=noise_adjoint, like=prediction)
        if tensors is None:
            return prediction
        mean, scale = tensors
        if not 0 <= int(step_index) < int(mean.shape[0]):
            raise ValueError("step_index is outside the target preconditioner time dimension")
        mean_step = mean[int(step_index)]
        scale_step = scale[int(step_index)]
        if tuple(prediction.shape[1:]) != tuple(mean_step.shape):
            raise ValueError("target preconditioner output dimension does not match step prediction")
        return scale_step.unsqueeze(0) * prediction

    @staticmethod
    def _timewise_preconditioner_stats(
        *,
        mean: torch.Tensor,
        scale: torch.Tensor,
        prefix: str,
    ) -> dict[str, float]:
        flat_scale = scale.detach().reshape(-1)
        quantiles = torch.quantile(
            flat_scale,
            torch.tensor([0.01, 0.50, 0.99], device=flat_scale.device, dtype=flat_scale.dtype),
        )
        return {
            f"{prefix}_mean_rms": float(torch.sqrt(torch.mean(mean.detach().pow(2))).item()),
            f"{prefix}_scale_min": float(flat_scale.min().item()),
            f"{prefix}_scale_q01": float(quantiles[0].item()),
            f"{prefix}_scale_q50": float(quantiles[1].item()),
            f"{prefix}_scale_q99": float(quantiles[2].item()),
            f"{prefix}_scale_max": float(flat_scale.max().item()),
        }

    def _fit_provided_target_preconditioner(
        self,
        *,
        target_paths: torch.Tensor,
        training: DiscreteMPTrainingConfig,
        provider: Callable[
            ["DiscreteMPModel", RolloutResult, torch.Tensor],
            ConditionalAdjointTargets,
        ],
        replay_bank: FitReplayBank | None,
    ) -> None:
        """Calibrate fixed scales from externally supplied conditional labels."""

        self._reset_target_preconditioner()
        if str(training.noise_target_control_variate) != "none":
            raise ValueError(
                "a conditional_target_provider requires noise_target_control_variate='none'"
            )
        if str(training.target_preconditioner) == "none":
            self._target_preconditioner_metrics.update(
                {
                    "conditional_target_provider": 1.0,
                    "target_preconditioner_timewise": 0.0,
                    "noise_target_control_variate_timewise": 0.0,
                    "noise_target_control_variate_adjoint": 0.0,
                }
            )
            return
        if str(training.target_preconditioner) != "timewise":
            raise ValueError("unsupported target_preconditioner")

        target_count, _ = self._validate_target_paths(target_paths)
        calibration_seed = derive_stream_seed(
            base_seed=training.seed, stream_offset=70_000
        )
        if calibration_seed is None:
            calibration_seed = 70_000
        cpu_generator = torch.Generator(device="cpu")
        cpu_generator.manual_seed(int(calibration_seed))
        p_sum: torch.Tensor | None = None
        p_sum_squares: torch.Tensor | None = None
        r_sum: torch.Tensor | None = None
        r_sum_squares: torch.Tensor | None = None
        count = 0
        for batch_index in range(int(training.preconditioner_batches)):
            if replay_bank is None:
                source_indices = self._sample_indices(
                    size=target_count,
                    count=int(training.batch_size),
                    generator=cpu_generator,
                )
                target_indices = self._sample_indices(
                    size=target_count,
                    count=int(training.target_batch_size),
                    generator=cpu_generator,
                )
                noise = sample_standard_normals(
                    grid=self.grid,
                    batch_size=int(training.batch_size),
                    noise_dim=int(self.architecture.noise_dim),
                    device=self.device,
                    dtype=self.dtype,
                    seed=derive_stream_seed(
                        base_seed=calibration_seed,
                        stream_offset=batch_index + 1,
                    ),
                )
            else:
                source_indices = replay_bank.source_indices
                target_indices = replay_bank.target_indices
                noise = replay_bank.noise.detach().clone()
            x0 = target_paths.index_select(
                0, source_indices.to(target_paths.device)
            )[:, 0, :]
            target_batch = target_paths.index_select(
                0, target_indices.to(target_paths.device)
            )
            with torch.no_grad():
                rollout = self.rollout(x0=x0, noise=noise)
            supplied = self._provided_conditional_adjoint_targets(
                provider=provider,
                rollout=rollout,
                target_paths=target_batch,
            )
            p64 = supplied.expected_adjoint_next.to(dtype=torch.float64)
            r64 = supplied.expected_adjoint_noise_next.to(dtype=torch.float64)
            batch_p_sum = p64.sum(dim=0)
            batch_p_sum_squares = p64.pow(2).sum(dim=0)
            batch_r_sum = r64.sum(dim=0)
            batch_r_sum_squares = r64.pow(2).sum(dim=0)
            if p_sum is None:
                p_sum = batch_p_sum
                p_sum_squares = batch_p_sum_squares
                r_sum = batch_r_sum
                r_sum_squares = batch_r_sum_squares
            else:
                assert p_sum_squares is not None
                assert r_sum is not None
                assert r_sum_squares is not None
                p_sum = p_sum + batch_p_sum
                p_sum_squares = p_sum_squares + batch_p_sum_squares
                r_sum = r_sum + batch_r_sum
                r_sum_squares = r_sum_squares + batch_r_sum_squares
            count += int(p64.shape[0])
        if count < 1 or p_sum is None or p_sum_squares is None:
            raise RuntimeError("provided-target preconditioner produced no samples")
        assert r_sum is not None
        assert r_sum_squares is not None
        minimum_scale = float(training.preconditioner_min_scale)
        self._adjoint_target_mean = (p_sum / float(count)).to(
            device=self.device, dtype=self.dtype
        )
        self._noise_adjoint_target_mean = (r_sum / float(count)).to(
            device=self.device, dtype=self.dtype
        )
        self._adjoint_target_scale = torch.sqrt(
            p_sum_squares / float(count)
        ).clamp_min(minimum_scale).to(device=self.device, dtype=self.dtype)
        self._noise_adjoint_target_scale = torch.sqrt(
            r_sum_squares / float(count)
        ).clamp_min(minimum_scale).to(device=self.device, dtype=self.dtype)
        self._target_preconditioner_metrics = {
            "conditional_target_provider": 1.0,
            "target_preconditioner_timewise": 1.0,
            "noise_target_control_variate_timewise": 0.0,
            "noise_target_control_variate_adjoint": 0.0,
            "target_calibration_batches": float(training.preconditioner_batches),
            "target_calibration_samples": float(count),
            "target_calibration_seed": float(calibration_seed),
        }
        self._target_preconditioner_metrics.update(
            self._timewise_preconditioner_stats(
                mean=self._adjoint_target_mean,
                scale=self._adjoint_target_scale,
                prefix="target_preconditioner_adjoint",
            )
        )
        self._target_preconditioner_metrics.update(
            self._timewise_preconditioner_stats(
                mean=self._noise_adjoint_target_mean,
                scale=self._noise_adjoint_target_scale,
                prefix="target_preconditioner_noise_adjoint",
            )
        )

    def _fit_target_preconditioner(
        self,
        *,
        target_paths: torch.Tensor,
        training: DiscreteMPTrainingConfig,
        replay_bank: FitReplayBank | None = None,
    ) -> None:
        """Estimate fixed time/output target moments on an independent calibration stream."""
        self._reset_target_preconditioner()
        needs_preconditioner = str(training.target_preconditioner) == "timewise"
        needs_timewise_control_variate = str(training.noise_target_control_variate) == "timewise"
        if not needs_preconditioner and not needs_timewise_control_variate:
            self._target_preconditioner_metrics["noise_target_control_variate_adjoint"] = float(
                str(training.noise_target_control_variate) == "adjoint"
            )
            return
        if str(training.target_preconditioner) not in {"none", "timewise"}:
            raise ValueError("unsupported target_preconditioner")

        target_count, _ = self._validate_target_paths(target_paths)
        calibration_seed = derive_stream_seed(base_seed=training.seed, stream_offset=70_000)
        if calibration_seed is None:
            calibration_seed = 70_000
        cpu_generator = torch.Generator(device="cpu")
        cpu_generator.manual_seed(int(calibration_seed))
        adjoint_sum: torch.Tensor | None = None
        adjoint_sum_squares: torch.Tensor | None = None
        noise_sum_matrix: torch.Tensor | None = None
        noise_sum_squares_matrix: torch.Tensor | None = None
        adjoint_noise_squared_sum: torch.Tensor | None = None
        noise_sum: torch.Tensor | None = None
        noise_sum_squares: torch.Tensor | None = None
        moments_noise_target_dim: int | None = None
        count = 0

        for batch_index in range(int(training.preconditioner_batches)):
            if replay_bank is None:
                source_indices = self._sample_indices(
                    size=target_count,
                    count=int(training.batch_size),
                    generator=cpu_generator,
                )
                target_indices = self._sample_indices(
                    size=target_count,
                    count=int(training.target_batch_size),
                    generator=cpu_generator,
                )
                noise = sample_standard_normals(
                    grid=self.grid,
                    batch_size=int(training.batch_size),
                    noise_dim=int(self.architecture.noise_dim),
                    device=self.device,
                    dtype=self.dtype,
                    seed=derive_stream_seed(
                        base_seed=calibration_seed,
                        stream_offset=batch_index + 1,
                    ),
                )
            else:
                source_indices = replay_bank.source_indices
                target_indices = replay_bank.target_indices
                noise = replay_bank.noise.detach().clone()
            x0 = target_paths.index_select(0, source_indices.to(target_paths.device))[:, 0, :]
            target_batch = target_paths.index_select(0, target_indices.to(target_paths.device))
            with torch.no_grad():
                rollout = self.rollout(x0=x0, noise=noise)
            pathwise_target, _ = self._path_functional_adjoint_targets(
                generated_path=rollout.paths,
                controls=rollout.controls,
                target_path=target_batch,
                training=training,
            )
            values64 = pathwise_target.detach().to(dtype=torch.float64)
            noise64 = noise.detach().to(dtype=torch.float64)
            batch_adjoint_sum = values64.sum(dim=0)
            batch_adjoint_sum_squares = values64.pow(2).sum(dim=0)
            noise_matrix = values64.unsqueeze(-1) * noise64.unsqueeze(-2)
            batch_noise_sum_matrix = noise_matrix.sum(dim=0)
            batch_noise_sum_squares_matrix = noise_matrix.pow(2).sum(dim=0)
            batch_adjoint_noise_squared_sum = (
                values64.unsqueeze(-1) * noise64.pow(2).unsqueeze(-2)
            ).sum(dim=0)
            batch_noise_sum = noise64.sum(dim=0)
            batch_noise_sum_squares = noise64.pow(2).sum(dim=0)
            if adjoint_sum is None:
                adjoint_sum = batch_adjoint_sum
                adjoint_sum_squares = batch_adjoint_sum_squares
                noise_sum_matrix = batch_noise_sum_matrix
                noise_sum_squares_matrix = batch_noise_sum_squares_matrix
                adjoint_noise_squared_sum = batch_adjoint_noise_squared_sum
                noise_sum = batch_noise_sum
                noise_sum_squares = batch_noise_sum_squares
                moments_noise_target_dim = int(rollout.expected_adjoint_noise_next.shape[-1])
            else:
                adjoint_sum = adjoint_sum + batch_adjoint_sum
                assert adjoint_sum_squares is not None
                assert noise_sum_matrix is not None
                assert noise_sum_squares_matrix is not None
                assert adjoint_noise_squared_sum is not None
                assert noise_sum is not None
                assert noise_sum_squares is not None
                adjoint_sum_squares = adjoint_sum_squares + batch_adjoint_sum_squares
                noise_sum_matrix = noise_sum_matrix + batch_noise_sum_matrix
                noise_sum_squares_matrix = noise_sum_squares_matrix + batch_noise_sum_squares_matrix
                adjoint_noise_squared_sum = adjoint_noise_squared_sum + batch_adjoint_noise_squared_sum
                noise_sum = noise_sum + batch_noise_sum
                noise_sum_squares = noise_sum_squares + batch_noise_sum_squares
            count += int(pathwise_target.shape[0])

        if count < 1:
            raise RuntimeError("target preconditioner calibration produced no samples")
        minimum_scale = float(training.preconditioner_min_scale)

        assert adjoint_sum is not None
        assert adjoint_sum_squares is not None
        assert noise_sum_matrix is not None
        assert noise_sum_squares_matrix is not None
        assert adjoint_noise_squared_sum is not None
        assert noise_sum is not None
        assert noise_sum_squares is not None
        assert moments_noise_target_dim is not None

        adjoint_mean64 = adjoint_sum / float(count)
        adjoint_scale64 = torch.sqrt(adjoint_sum_squares / float(count)).clamp_min(minimum_scale)
        raw_noise_second_moment = noise_sum_squares_matrix / float(count)
        if needs_timewise_control_variate:
            baseline64 = adjoint_noise_squared_sum / noise_sum_squares.unsqueeze(-2).clamp_min(
                torch.finfo(torch.float64).eps
            )
            centered_noise_sum_matrix = noise_sum_matrix - baseline64 * noise_sum.unsqueeze(-2)
            centered_noise_sum_squares_matrix = (
                noise_sum_squares_matrix
                - 2.0 * baseline64 * adjoint_noise_squared_sum
                + baseline64.pow(2) * noise_sum_squares.unsqueeze(-2)
            ).clamp_min(0.0)
            self._noise_target_timewise_baseline = baseline64.to(device=self.device, dtype=self.dtype)
        else:
            baseline64 = torch.zeros_like(adjoint_noise_squared_sum)
            centered_noise_sum_matrix = noise_sum_matrix
            centered_noise_sum_squares_matrix = noise_sum_squares_matrix
        noise_mean_matrix64 = centered_noise_sum_matrix / float(count)
        noise_scale_matrix64 = torch.sqrt(centered_noise_sum_squares_matrix / float(count)).clamp_min(
            minimum_scale
        )
        noise_mean64 = self._format_noise_adjoint_matrix(
            noise_mean_matrix64.unsqueeze(0),
            moments_noise_target_dim=moments_noise_target_dim,
        )[0]
        noise_scale64 = self._format_noise_adjoint_matrix(
            noise_scale_matrix64.unsqueeze(0),
            moments_noise_target_dim=moments_noise_target_dim,
        )[0]

        self._adjoint_target_mean = adjoint_mean64.to(device=self.device, dtype=self.dtype)
        self._noise_adjoint_target_mean = noise_mean64.to(device=self.device, dtype=self.dtype)
        if needs_preconditioner:
            self._adjoint_target_scale = adjoint_scale64.to(device=self.device, dtype=self.dtype)
            self._noise_adjoint_target_scale = noise_scale64.to(device=self.device, dtype=self.dtype)
        self._target_preconditioner_metrics = {
            "target_preconditioner_timewise": float(needs_preconditioner),
            "noise_target_control_variate_timewise": float(needs_timewise_control_variate),
            "noise_target_control_variate_adjoint": float(
                str(training.noise_target_control_variate) == "adjoint"
            ),
            "target_calibration_batches": float(training.preconditioner_batches),
            "target_calibration_samples": float(count),
            "target_calibration_seed": float(calibration_seed),
        }
        if needs_preconditioner:
            assert self._adjoint_target_scale is not None
            assert self._noise_adjoint_target_scale is not None
            self._target_preconditioner_metrics.update(
                {
                    "target_preconditioner_batches": float(training.preconditioner_batches),
                    "target_preconditioner_samples": float(count),
                    "target_preconditioner_seed": float(calibration_seed),
                }
            )
            self._target_preconditioner_metrics.update(
                self._timewise_preconditioner_stats(
                    mean=self._adjoint_target_mean,
                    scale=self._adjoint_target_scale,
                    prefix="target_preconditioner_adjoint",
                )
            )
            self._target_preconditioner_metrics.update(
                self._timewise_preconditioner_stats(
                    mean=self._noise_adjoint_target_mean,
                    scale=self._noise_adjoint_target_scale,
                    prefix="target_preconditioner_noise_adjoint",
                )
            )
        if needs_timewise_control_variate:
            centered_noise_second_moment = centered_noise_sum_squares_matrix / float(count)
            raw_noise_second_moment_output = self._format_noise_adjoint_matrix(
                raw_noise_second_moment.unsqueeze(0),
                moments_noise_target_dim=moments_noise_target_dim,
            )[0]
            centered_noise_second_moment_output = self._format_noise_adjoint_matrix(
                centered_noise_second_moment.unsqueeze(0),
                moments_noise_target_dim=moments_noise_target_dim,
            )[0]
            baseline_output = self._format_noise_adjoint_matrix(
                baseline64.unsqueeze(0),
                moments_noise_target_dim=moments_noise_target_dim,
            )[0]
            self._target_preconditioner_metrics.update(
                {
                    "noise_target_control_variate_baseline_rms": float(
                        torch.sqrt(torch.mean(baseline_output.pow(2))).item()
                    ),
                    "noise_target_control_variate_calibration_second_moment_ratio": float(
                        (
                            centered_noise_second_moment_output.mean()
                            / raw_noise_second_moment_output.mean().clamp_min(torch.finfo(torch.float64).eps)
                        ).item()
                    ),
                }
            )

    @staticmethod
    def _running_cost_result_and_path_gradient(
        *,
        running_cost: PathDependentRunningCost,
        inputs: RunningCostInputs,
        create_graph: bool,
    ) -> tuple[RunningCostResult, torch.Tensor | None]:
        """Evaluate a running cost and its partial path derivative.

        Running-cost owners may expose an exact first-order fast path.  The
        generic autograd route remains the fallback and is always retained
        when higher-order derivatives are requested.
        """

        owner = getattr(running_cost, "__self__", None)
        combined_fast_path = getattr(
            owner,
            "running_cost_value_and_path_gradient",
            None,
        )
        use_explicit_vjp = bool(
            getattr(owner, "use_explicit_running_cost_vjp", False)
        )
        if (
            callable(combined_fast_path)
            and use_explicit_vjp
            and not bool(create_graph)
        ):
            combined = combined_fast_path(inputs)
            if combined is not None:
                result, gradient = combined
                if result.value.ndim != 0:
                    raise ValueError("running cost must return a scalar value")
                if tuple(gradient.shape) != tuple(inputs.paths.shape):
                    raise ValueError(
                        "fast running-cost gradient must match the path shape"
                    )
                if not bool(torch.isfinite(result.value).item()):
                    raise ValueError("running cost must be finite")
                if not bool(torch.isfinite(gradient).all().item()):
                    raise ValueError("fast running-cost gradient must be finite")
                return result, gradient.detach()

        result = running_cost(inputs)
        if result.value.ndim != 0:
            raise ValueError("running cost must return a scalar value")
        if not bool(torch.isfinite(result.value.detach()).item()):
            raise ValueError("running cost must be finite")
        gradient = None
        if result.value.requires_grad:
            gradient = torch.autograd.grad(
                result.value,
                inputs.paths,
                allow_unused=True,
                create_graph=bool(create_graph),
            )[0]
            if gradient is not None and not bool(create_graph):
                gradient = gradient.detach()
        return result, gradient

    def _drift_aware_pathwise_adjoint(
        self,
        *,
        paths: torch.Tensor,
        sources: torch.Tensor,
        backend: str,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Propagate direct path sources through a fixed path-dependent drift.

        The realized volatility controls are held fixed.  Only the state
        dependence of the frozen reference drift is differentiated.  The
        ``none`` backend preserves the historical reverse cumulative sum.
        """

        mode = str(backend).strip().lower()
        baseline = future_source_sums(sources)
        if mode == "none":
            return baseline, {
                "drift_adjoint_applied": 0.0,
                "drift_adjoint_autograd_replay": 0.0,
                "drift_adjoint_analytical_reference": 0.0,
                "drift_adjoint_relative_change": 0.0,
                "drift_adjoint_cosine_with_cumulative_sum": 1.0,
                "drift_adjoint_replay_max_abs_error": 0.0,
            }

        if tuple(paths.shape) != tuple(sources.shape):
            raise ValueError("paths and sources must have the same shape")
        if int(paths.shape[1]) != int(self.grid.num_steps) + 1:
            raise ValueError("paths must align with the model grid")
        reference_kernel = getattr(self.control_map, "reference_kernel", None)
        if reference_kernel is None:
            raise TypeError(
                "a drift-aware adjoint requires a control map with a reference_kernel"
            )
        times = self.grid.time_points(device=paths.device, dtype=paths.dtype)
        step_sizes = times[1:] - times[:-1]
        replay_error = 0.0

        if mode == "autograd_replay":
            with torch.no_grad():
                base_drifts = torch.stack(
                    [
                        reference_kernel.evaluate(
                            paths[:, : step + 1, :],
                            step_index=step,
                        )[0]
                        for step in range(int(self.grid.num_steps))
                    ],
                    dim=1,
                )
                additive_increments = (
                    paths[:, 1:, :]
                    - paths[:, :-1, :]
                    - base_drifts * step_sizes.view(1, -1, 1)
                ).detach()
            additive_increments = additive_increments.clone().requires_grad_(True)
            replay_points = [paths[:, 0, :].detach()]
            for step in range(int(self.grid.num_steps)):
                replay_prefix = torch.stack(replay_points, dim=1)
                replay_drift = reference_kernel.evaluate(
                    replay_prefix,
                    step_index=step,
                )[0]
                replay_points.append(
                    replay_points[-1]
                    + replay_drift * step_sizes[step]
                    + additive_increments[:, step, :]
                )
            replay_paths = torch.stack(replay_points, dim=1)
            linearized_objective = torch.sum(sources.detach() * replay_paths)
            corrected = torch.autograd.grad(
                linearized_objective,
                additive_increments,
                create_graph=False,
            )[0].detach()
            replay_error = float(
                torch.max(torch.abs(replay_paths.detach() - paths.detach())).item()
            )
        elif mode == "analytical_reference":
            drift_vjp = getattr(reference_kernel, "drift_step_path_vjp", None)
            if not callable(drift_vjp):
                raise TypeError(
                    "the selected reference does not provide drift_step_path_vjp"
                )
            with torch.no_grad():
                state_cotangent = sources.detach().clone()
                corrected = torch.zeros_like(baseline)
                for step in range(int(self.grid.num_steps) - 1, -1, -1):
                    next_cotangent = state_cotangent[:, step + 1, :].clone()
                    corrected[:, step, :] = next_cotangent
                    state_cotangent[:, step, :].add_(next_cotangent)
                    prefix_cotangent = drift_vjp(
                        paths[:, : step + 1, :],
                        next_cotangent * step_sizes[step],
                        step_index=step,
                    )
                    state_cotangent[:, : step + 1, :].add_(prefix_cotangent)
        else:
            raise ValueError(
                "drift adjoint backend must be none, autograd_replay, or analytical_reference"
            )

        corrected_flat = corrected.reshape(-1).double()
        baseline_flat = baseline.reshape(-1).double()
        corrected_norm = torch.linalg.vector_norm(corrected_flat)
        baseline_norm = torch.linalg.vector_norm(baseline_flat)
        denominator = max(
            float(baseline_norm.item()),
            torch.finfo(torch.float64).eps,
        )
        relative_change = float(
            torch.linalg.vector_norm(corrected_flat - baseline_flat).item()
        ) / denominator
        if float(corrected_norm.item()) > 0.0 and float(baseline_norm.item()) > 0.0:
            cosine = float(
                (
                    torch.dot(corrected_flat, baseline_flat)
                    / (corrected_norm * baseline_norm)
                ).item()
            )
        else:
            cosine = 1.0 if torch.equal(corrected, baseline) else 0.0
        return corrected, {
            "drift_adjoint_applied": 1.0,
            "drift_adjoint_autograd_replay": float(mode == "autograd_replay"),
            "drift_adjoint_analytical_reference": float(
                mode == "analytical_reference"
            ),
            "drift_adjoint_relative_change": relative_change,
            "drift_adjoint_cosine_with_cumulative_sum": cosine,
            "drift_adjoint_replay_max_abs_error": replay_error,
        }

    def _path_functional_adjoint_targets(
        self,
        *,
        generated_path: torch.Tensor,
        controls: torch.Tensor | None = None,
        target_path: torch.Tensor,
        training: DiscreteMPTrainingConfig,
        discrepancy: PathFunctionalDiscrepancy | None = None,
        include_running_cost: bool = True,
        create_graph: bool = False,
        preserve_generated_path_graph: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if generated_path.ndim != 3:
            raise ValueError("generated_path must have shape (B, N + 1, d)")
        if bool(preserve_generated_path_graph) and not bool(create_graph):
            raise ValueError(
                "preserve_generated_path_graph requires create_graph=True"
            )
        if bool(preserve_generated_path_graph) and not generated_path.requires_grad:
            raise ValueError(
                "generated_path must require gradients when preserving its graph"
            )
        analytical_backend = str(training.path_derivative_backend) == "analytical"
        if analytical_backend and (
            bool(create_graph) or bool(preserve_generated_path_graph)
        ):
            raise ValueError(
                "the analytical path-derivative backend provides first derivatives "
                "only and cannot preserve or create an autograd graph"
            )
        discrepancy_used = self.discrepancy if discrepancy is None else discrepancy
        if bool(training.observed_only):
            observed_indices = self.grid.observed_point_indices(device=generated_path.device)
            observed_times = self.grid.time_points(
                device=generated_path.device,
                dtype=torch.float64,
            ).index_select(0, observed_indices)
            discrepancy_grid = DiscreteTimeGrid(
                T=float(self.grid.T),
                num_steps=int(self.grid.observed_count) - 1,
                observed_indices=None,
                point_times=tuple(float(value) for value in observed_times.detach().cpu().tolist()),
                allow_nonuniform=True,
            )
            generated_req = generated_path.index_select(1, observed_indices)
            if not bool(preserve_generated_path_graph):
                generated_req = generated_req.detach().clone()
                if not analytical_backend:
                    generated_req.requires_grad_(True)
            target_used = target_path.detach().index_select(1, observed_indices)
            if analytical_backend:
                evaluator = getattr(
                    discrepancy_used,
                    "analytic_value_and_path_gradient",
                    None,
                )
                if not callable(evaluator):
                    raise TypeError(
                        "the selected discrepancy does not provide analytical path sources"
                    )
                discrepancy_result, base_gradient = evaluator(
                    generated_paths=generated_req,
                    target_paths=target_used,
                    grid=discrepancy_grid,
                )
                total_objective = float(training.lambda_scale) * discrepancy_result.value
                empirical_grad = float(training.lambda_scale) * base_gradient
            else:
                discrepancy_result = discrepancy_used(
                    generated_paths=generated_req,
                    target_paths=target_used,
                    grid=discrepancy_grid,
                )
                total_objective = float(training.lambda_scale) * discrepancy_result.value
                empirical_grad = torch.autograd.grad(
                    total_objective,
                    generated_req,
                    create_graph=bool(create_graph),
                )[0]
                if not bool(create_graph):
                    empirical_grad = empirical_grad.detach()
            observed_sources = float(generated_req.shape[0]) * empirical_grad
            discrepancy_sources = scatter_observed_sources(observed_sources, grid=self.grid)
        else:
            generated_req = generated_path if bool(preserve_generated_path_graph) else generated_path.detach().clone()
            if not bool(preserve_generated_path_graph) and not analytical_backend:
                generated_req.requires_grad_(True)
            if analytical_backend:
                evaluator = getattr(
                    discrepancy_used,
                    "analytic_value_and_path_gradient",
                    None,
                )
                if not callable(evaluator):
                    raise TypeError(
                        "the selected discrepancy does not provide analytical path sources"
                    )
                discrepancy_result, base_gradient = evaluator(
                    generated_paths=generated_req,
                    target_paths=target_path.detach(),
                    grid=self.grid,
                )
                total_objective = float(training.lambda_scale) * discrepancy_result.value
                empirical_grad = float(training.lambda_scale) * base_gradient
            else:
                discrepancy_result = discrepancy_used(
                    generated_paths=generated_req,
                    target_paths=target_path.detach(),
                    grid=self.grid,
                )
                total_objective = float(training.lambda_scale) * discrepancy_result.value
                empirical_grad = torch.autograd.grad(
                    total_objective,
                    generated_req,
                    create_graph=bool(create_graph),
                )[0]
                if not bool(create_graph):
                    empirical_grad = empirical_grad.detach()
            discrepancy_sources = float(generated_req.shape[0]) * empirical_grad

        running_sources = torch.zeros_like(discrepancy_sources)
        running_metrics: dict[str, float] = {}
        running_cost_value = 0.0
        running_cost_used = self.running_cost if bool(include_running_cost) else None
        if running_cost_used is not None:
            if controls is None:
                raise ValueError(
                    "controls are required when constructing the complete path-dependent "
                    "adjoint target with a running cost"
                )
            expected_control_prefix = (int(generated_path.shape[0]), int(self.grid.num_steps))
            if controls.ndim != 3 or tuple(controls.shape[:2]) != expected_control_prefix:
                raise ValueError("controls must have shape (B, N, control_dim) and align with generated_path")
            running_path_req = generated_path if bool(preserve_generated_path_graph) else generated_path.detach().clone()
            if not bool(preserve_generated_path_graph) and not analytical_backend:
                running_path_req.requires_grad_(True)
            running_inputs = RunningCostInputs(
                paths=running_path_req,
                controls=controls.detach(),
                grid=self.grid,
                parameters=self.parameters,
            )
            if analytical_backend:
                running_cost_owner = getattr(running_cost_used, "__self__", None)
                combined = getattr(
                    running_cost_owner,
                    "running_cost_value_and_path_gradient",
                    None,
                )
                evaluated = combined(running_inputs) if callable(combined) else None
                if evaluated is None:
                    raise TypeError(
                        "the selected running cost/reference does not provide an "
                        "analytical path derivative"
                    )
                running_result, running_grad = evaluated
            else:
                running_result, running_grad = self._running_cost_result_and_path_gradient(
                    running_cost=running_cost_used,
                    inputs=running_inputs,
                    create_graph=bool(create_graph),
                )
            if running_grad is not None:
                running_sources = float(running_path_req.shape[0]) * running_grad
            running_metrics = {key: float(value) for key, value in running_result.metrics.items()}
            running_cost_value = float(running_result.value.detach().item())

        full_sources = discrepancy_sources + running_sources

        targets, drift_adjoint_metrics = self._drift_aware_pathwise_adjoint(
            paths=generated_path,
            sources=full_sources,
            backend=training.drift_adjoint_backend,
        )
        metrics = {key: float(value) for key, value in discrepancy_result.metrics.items()}
        metrics["lambda_scale"] = float(training.lambda_scale)
        metrics["batch_lions_scale"] = float(generated_req.shape[0])
        metrics["observed_only"] = float(bool(training.observed_only))
        metrics["analytical_path_derivative"] = float(analytical_backend)
        metrics["running_cost_included"] = float(running_cost_used is not None)
        metrics["running_cost_value"] = running_cost_value
        metrics["discrepancy_objective_value"] = float(total_objective.detach().item())
        metrics["complete_objective_value"] = float(total_objective.detach().item()) + running_cost_value
        metrics["discrepancy_source_rms"] = float(
            torch.sqrt(torch.mean(discrepancy_sources.pow(2))).item()
        )
        metrics["running_source_rms"] = float(torch.sqrt(torch.mean(running_sources.pow(2))).item())
        discrepancy_norm = torch.linalg.vector_norm(discrepancy_sources.detach().reshape(-1).double())
        running_norm = torch.linalg.vector_norm(running_sources.detach().reshape(-1).double())
        if float(discrepancy_norm.item()) > 0.0 and float(running_norm.item()) > 0.0:
            source_cosine = torch.dot(
                discrepancy_sources.detach().reshape(-1).double(),
                running_sources.detach().reshape(-1).double(),
            ) / (discrepancy_norm * running_norm)
            metrics["running_discrepancy_source_cosine"] = float(source_cosine.item())
        else:
            metrics["running_discrepancy_source_cosine"] = 0.0
        metrics["running_to_discrepancy_source_rms_ratio"] = metrics["running_source_rms"] / max(
            metrics["discrepancy_source_rms"],
            torch.finfo(torch.float64).eps,
        )
        metrics.update(running_metrics)
        metrics["source_rms"] = float(torch.sqrt(torch.mean(full_sources.pow(2))).item())
        metrics["target_rms"] = float(torch.sqrt(torch.mean(targets.pow(2))).item())
        metrics.update(drift_adjoint_metrics)
        return targets, metrics

    def _frozen_law_path_functional_adjoint_targets(
        self,
        *,
        query_path: torch.Tensor,
        population_path: torch.Tensor,
        controls: torch.Tensor,
        target_path: torch.Tensor,
        training: DiscreteMPTrainingConfig,
        discrepancy: object | None = None,
        include_running_cost: bool = True,
        create_graph: bool = False,
        preserve_query_path_graph: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Evaluate complete-MP sources at queries under a frozen law.

        Conditional future branching changes the distribution of the query
        paths.  The Lions derivative must nevertheless be evaluated at the
        unconditional outer forward law.  Discrepancies used here therefore
        provide a ``lions_potential`` whose population argument is detached;
        differentiating its query average and multiplying by the query count
        gives the required per-path functional derivative.
        """

        if query_path.ndim != 3 or population_path.ndim != 3:
            raise ValueError("query and population paths must have shape (B, N + 1, d)")
        if tuple(query_path.shape[1:]) != tuple(population_path.shape[1:]):
            raise ValueError("query and population path dimensions must match")
        if controls.ndim != 3 or tuple(controls.shape[:2]) != (
            int(query_path.shape[0]),
            int(self.grid.num_steps),
        ):
            raise ValueError("controls must align with the query paths")
        discrepancy_functional = self.discrepancy if discrepancy is None else discrepancy
        analytical_backend = str(training.path_derivative_backend) == "analytical"
        potential = getattr(discrepancy_functional, "lions_potential", None)
        analytical_potential = getattr(
            discrepancy_functional,
            "analytic_lions_potential_and_path_gradient",
            None,
        )
        if analytical_backend and not callable(analytical_potential):
            raise TypeError(
                "nested conditional targets require a discrepancy with an "
                "analytical frozen-law Lions source"
            )
        if not analytical_backend and not callable(potential):
            raise TypeError(
                "nested conditional targets require a discrepancy with a "
                "frozen-law lions_potential"
            )

        if bool(preserve_query_path_graph) and not bool(create_graph):
            raise ValueError(
                "preserve_query_path_graph requires create_graph=True"
            )
        if bool(preserve_query_path_graph) and not query_path.requires_grad:
            raise ValueError(
                "query_path must require gradients when preserving its graph"
            )
        if analytical_backend and (
            bool(create_graph) or bool(preserve_query_path_graph)
        ):
            raise ValueError(
                "the analytical path-derivative backend provides first derivatives "
                "only and cannot preserve or create an autograd graph"
            )
        query_req = query_path if bool(preserve_query_path_graph) else query_path.detach().clone()
        if not bool(preserve_query_path_graph) and not analytical_backend:
            query_req.requires_grad_(True)
        population_detached = population_path.detach()
        target_detached = target_path.detach().to(
            device=query_req.device,
            dtype=query_req.dtype,
        )
        if bool(training.observed_only):
            observed_indices = self.grid.observed_point_indices(
                device=query_req.device
            )
            observed_times = self.grid.time_points(
                device=query_req.device,
                dtype=torch.float64,
            ).index_select(0, observed_indices)
            discrepancy_grid = DiscreteTimeGrid(
                T=float(self.grid.T),
                num_steps=int(self.grid.observed_count) - 1,
                point_times=tuple(
                    float(value)
                    for value in observed_times.detach().cpu().tolist()
                ),
                allow_nonuniform=True,
            )
            query_used = query_req.index_select(1, observed_indices)
            population_used = population_detached.index_select(
                1, observed_indices
            )
            target_used = target_detached.index_select(1, observed_indices)
        else:
            discrepancy_grid = self.grid
            query_used = query_req
            population_used = population_detached
            target_used = target_detached

        if analytical_backend:
            potential_value, used_gradient = analytical_potential(
                query_paths=query_used,
                population_paths=population_used,
                target_paths=target_used,
                grid=discrepancy_grid,
            )
            discrepancy_potential = float(training.lambda_scale) * potential_value
            if bool(training.observed_only):
                query_gradient = scatter_observed_sources(
                    float(training.lambda_scale) * used_gradient,
                    grid=self.grid,
                )
            else:
                query_gradient = float(training.lambda_scale) * used_gradient
        else:
            potential_value = potential(
                query_paths=query_used,
                population_paths=population_used,
                target_paths=target_used,
                grid=discrepancy_grid,
            )
            discrepancy_potential = float(training.lambda_scale) * potential_value
            query_gradient = torch.autograd.grad(
                discrepancy_potential,
                query_req,
                create_graph=bool(create_graph),
            )[0]
            if not bool(create_graph):
                query_gradient = query_gradient.detach()
        discrepancy_sources = float(query_req.shape[0]) * query_gradient

        running_sources = torch.zeros_like(discrepancy_sources)
        running_cost_value = 0.0
        running_metrics: dict[str, float] = {}
        if self.running_cost is not None and bool(include_running_cost):
            running_path_req = query_path if bool(preserve_query_path_graph) else query_path.detach().clone()
            if not bool(preserve_query_path_graph) and not analytical_backend:
                running_path_req.requires_grad_(True)
            running_inputs = RunningCostInputs(
                paths=running_path_req,
                controls=controls.detach(),
                grid=self.grid,
                parameters=self.parameters,
            )
            if analytical_backend:
                running_cost_owner = getattr(self.running_cost, "__self__", None)
                combined = getattr(
                    running_cost_owner,
                    "running_cost_value_and_path_gradient",
                    None,
                )
                evaluated = combined(running_inputs) if callable(combined) else None
                if evaluated is None:
                    raise TypeError(
                        "the selected running cost/reference does not provide an "
                        "analytical path derivative"
                    )
                running_result, running_gradient = evaluated
            else:
                running_result, running_gradient = self._running_cost_result_and_path_gradient(
                    running_cost=self.running_cost,
                    inputs=running_inputs,
                    create_graph=bool(create_graph),
                )
            if running_gradient is not None:
                running_sources = float(query_path.shape[0]) * running_gradient
            running_cost_value = float(running_result.value.detach().item())
            running_metrics = {
                str(name): float(value)
                for name, value in running_result.metrics.items()
            }

        full_sources = discrepancy_sources + running_sources
        targets, drift_adjoint_metrics = self._drift_aware_pathwise_adjoint(
            paths=query_path,
            sources=full_sources,
            backend=training.drift_adjoint_backend,
        )
        discrepancy_norm = torch.linalg.vector_norm(
            discrepancy_sources.reshape(-1).double()
        )
        running_norm = torch.linalg.vector_norm(
            running_sources.reshape(-1).double()
        )
        if float(discrepancy_norm.item()) > 0.0 and float(
            running_norm.item()
        ) > 0.0:
            source_cosine = torch.dot(
                discrepancy_sources.reshape(-1).double(),
                running_sources.reshape(-1).double(),
            ) / (discrepancy_norm * running_norm)
        else:
            source_cosine = discrepancy_sources.new_tensor(0.0).double()
        discrepancy_rms = float(
            torch.sqrt(torch.mean(discrepancy_sources.pow(2))).item()
        )
        running_rms = float(
            torch.sqrt(torch.mean(running_sources.pow(2))).item()
        )
        metrics = {
            "frozen_law_lions_derivative": 1.0,
            "frozen_law_population_paths": float(population_path.shape[0]),
            "frozen_law_query_paths": float(query_path.shape[0]),
            "batch_lions_scale": float(query_path.shape[0]),
            "lambda_scale": float(training.lambda_scale),
            "observed_only": float(bool(training.observed_only)),
            "analytical_path_derivative": float(analytical_backend),
            "running_cost_included": float(
                self.running_cost is not None and bool(include_running_cost)
            ),
            "running_cost_value": running_cost_value,
            "discrepancy_lions_potential_value": float(
                discrepancy_potential.detach().item()
            ),
            "discrepancy_source_rms": discrepancy_rms,
            "running_source_rms": running_rms,
            "running_discrepancy_source_cosine": float(source_cosine.item()),
            "running_to_discrepancy_source_rms_ratio": running_rms
            / max(discrepancy_rms, torch.finfo(torch.float64).eps),
            "source_rms": float(torch.sqrt(torch.mean(full_sources.pow(2))).item()),
            "target_rms": float(torch.sqrt(torch.mean(targets.pow(2))).item()),
            **drift_adjoint_metrics,
            **running_metrics,
        }
        return targets, metrics

    def _stein_adjoint_noise_target(
        self,
        *,
        differentiable_adjoint_target: torch.Tensor,
        noise: torch.Tensor,
        moments_noise_target_dim: int,
        num_probes: int = 1,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Estimate the Gaussian-Stein representation of the noise adjoint.

        For standard-normal innovations and a differentiable pathwise adjoint
        ``A``, Gaussian integration by parts gives

        ``E[A * epsilon | F_n] = E[d A / d epsilon | F_n]``.

        A Hutchinson diagonal estimator recovers the same-sample, same-time
        Jacobian blocks without materializing the full path Jacobian.  It is an
        unbiased numerical estimator of the derivative representation; it does
        not alter the maximum-principle target or the neural regression loss.
        """

        if differentiable_adjoint_target.ndim != 3:
            raise ValueError(
                "differentiable_adjoint_target must have shape (B, N, d)"
            )
        if noise.ndim != 3:
            raise ValueError("noise must have shape (B, N, q)")
        if tuple(differentiable_adjoint_target.shape[:2]) != tuple(noise.shape[:2]):
            raise ValueError("adjoint target and noise must align in batch and time")
        if not differentiable_adjoint_target.requires_grad:
            raise ValueError("differentiable_adjoint_target must require gradients")
        if not noise.requires_grad:
            raise ValueError("noise must require gradients")
        probe_count = int(num_probes)
        if probe_count < 1:
            raise ValueError("num_probes must be >= 1")

        batch_size, num_steps, state_dim = differentiable_adjoint_target.shape
        noise_dim = int(noise.shape[-1])
        cpu_generator = torch.Generator(device="cpu")
        if seed is None:
            cpu_generator.seed()
        else:
            cpu_generator.manual_seed(int(seed))
        estimate = differentiable_adjoint_target.new_zeros(
            int(batch_size), int(num_steps), int(state_dim), noise_dim
        )
        total_reverse_calls = probe_count * int(state_dim)
        reverse_call = 0
        for _ in range(probe_count):
            signs = torch.randint(
                0,
                2,
                (int(batch_size), int(num_steps)),
                generator=cpu_generator,
                device="cpu",
                dtype=torch.int64,
            )
            signs = signs.to(
                device=differentiable_adjoint_target.device,
                dtype=differentiable_adjoint_target.dtype,
            ).mul_(2.0).sub_(1.0)
            for state_index in range(int(state_dim)):
                reverse_call += 1
                scalar = torch.sum(
                    differentiable_adjoint_target[..., state_index] * signs
                )
                derivative = torch.autograd.grad(
                    scalar,
                    noise,
                    retain_graph=reverse_call < total_reverse_calls,
                    create_graph=False,
                )[0]
                estimate[..., state_index, :] += derivative * signs.unsqueeze(-1)
        estimate = estimate / float(probe_count)
        return self._format_noise_adjoint_matrix(
            estimate,
            moments_noise_target_dim=int(moments_noise_target_dim),
        ).detach()

    def _complete_objective_metrics(
        self,
        *,
        generated_path: torch.Tensor,
        controls: torch.Tensor,
        target_path: torch.Tensor,
        training: DiscreteMPTrainingConfig,
        differentiate_controls: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Evaluate the complete empirical objective.

        ``differentiate_controls`` is false for MP line-search evaluation.  A
        direct-objective diagnostic sets it true so that the running cost's
        control derivative is included in backpropagation through the policy.
        """

        if generated_path.ndim != 3:
            raise ValueError("generated_path must have shape (B, N + 1, d)")
        if controls.ndim != 3 or tuple(controls.shape[:2]) != (
            int(generated_path.shape[0]),
            int(self.grid.num_steps),
        ):
            raise ValueError("controls must align with generated_path")
        if bool(training.observed_only):
            observed_indices = self.grid.observed_point_indices(device=generated_path.device)
            observed_times = self.grid.time_points(
                device=generated_path.device,
                dtype=torch.float64,
            ).index_select(0, observed_indices)
            discrepancy_grid = DiscreteTimeGrid(
                T=float(self.grid.T),
                num_steps=int(self.grid.observed_count) - 1,
                point_times=tuple(
                    float(value) for value in observed_times.detach().cpu().tolist()
                ),
                allow_nonuniform=True,
            )
            generated_used = generated_path.index_select(1, observed_indices)
            target_used = target_path.index_select(1, observed_indices)
        else:
            discrepancy_grid = self.grid
            generated_used = generated_path
            target_used = target_path
        discrepancy_result = self.discrepancy(
            generated_paths=generated_used,
            target_paths=target_used,
            grid=discrepancy_grid,
        )
        discrepancy_value = float(training.lambda_scale) * discrepancy_result.value
        running_value = discrepancy_value.new_tensor(0.0)
        running_metrics: dict[str, float] = {}
        if self.running_cost is not None:
            running_result = self.running_cost(
                RunningCostInputs(
                    paths=generated_path,
                    controls=controls,
                    grid=self.grid,
                    parameters=self.parameters,
                    differentiate_controls=bool(differentiate_controls),
                )
            )
            running_value = running_result.value
            running_metrics = {
                str(name): float(value)
                for name, value in running_result.metrics.items()
            }
        complete = discrepancy_value + running_value
        metrics = {
            "complete_objective_value": float(complete.detach().item()),
            "discrepancy_objective_value": float(discrepancy_value.detach().item()),
            "running_cost_value": float(running_value.detach().item()),
            **{str(name): float(value) for name, value in discrepancy_result.metrics.items()},
            **running_metrics,
        }
        return complete, metrics

    @staticmethod
    def _check_same_shape(*, actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
        if tuple(actual.shape) != tuple(expected.shape):
            raise ValueError(
                f"{name} shape mismatch: network produced {tuple(actual.shape)} "
                f"but target has {tuple(expected.shape)}"
            )

    def _project_pathwise_target(
        self,
        *,
        frozen_generated_path: torch.Tensor,
        pathwise_target: torch.Tensor,
        training: DiscreteMPTrainingConfig,
        noise_adjoint: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if frozen_generated_path.ndim != 3:
            raise ValueError("frozen_generated_path must have shape (B, N + 1, d)")
        if pathwise_target.ndim != 3:
            raise ValueError("pathwise_target must have shape (B, N, m)")
        if int(frozen_generated_path.shape[0]) != int(pathwise_target.shape[0]):
            raise ValueError("path and target batch sizes must match")
        if int(frozen_generated_path.shape[1]) != int(pathwise_target.shape[1]) + 1:
            raise ValueError("pathwise_target must have one fewer time point than frozen path")

        if str(training.ce_target_mode) == "direct":
            direct_target = pathwise_target.detach()
            return direct_target, {
                "ce_projected_target_rms": float(torch.sqrt(torch.mean(direct_target.pow(2))).item()),
                "ce_training_target_rms": float(torch.sqrt(torch.mean(direct_target.pow(2))).item()),
                "ce_target_is_direct": 1.0,
                "ce_crossfit_folds": 1.0,
                "ce_projection_is_crossfit": 0.0,
            }

        fold_count = int(training.ce_crossfit_folds)
        batch_size = int(frozen_generated_path.shape[0])
        if fold_count > batch_size:
            raise ValueError("ce_crossfit_folds cannot exceed the generated training batch size")

        ce_estimator = (
            self.noise_ce_estimator
            if bool(noise_adjoint) and self.noise_ce_estimator is not None
            else self.ce_estimator
        )
        if isinstance(ce_estimator, RidgeConditionalExpectation):
            ce_estimator = RidgeConditionalExpectation(
                ridge=float(training.ridge_lambda),
                eps=float(ce_estimator.eps),
            )
        features_all = getattr(ce_estimator, "features_all", None)
        fit_predict_features = getattr(
            ce_estimator,
            "fit_predict_features",
            None,
        )
        precomputed_features = (
            features_all(frozen_generated_path)
            if callable(features_all) and callable(fit_predict_features)
            else None
        )
        if precomputed_features is not None and tuple(
            precomputed_features.shape[:2]
        ) != tuple(pathwise_target.shape[:2]):
            raise ValueError(
                "precomputed CE features must align with the pathwise target"
            )

        projected_steps = []
        fit_rms_values = []
        feature_dims = []
        residual_sum_squares = pathwise_target.new_tensor(0.0)
        baseline_sum_squares = pathwise_target.new_tensor(0.0)
        projection_count = 0
        fold_train_sizes = []
        fold_eval_sizes = []
        all_indices = torch.arange(batch_size, device=frozen_generated_path.device)
        if fold_count == 1:
            fold_specs = ((all_indices, all_indices),)
        else:
            fold_membership = all_indices.remainder(fold_count)
            fold_specs = tuple(
                (
                    all_indices[fold_membership != fold_index],
                    all_indices[fold_membership == fold_index],
                )
                for fold_index in range(fold_count)
            )
        pooled_time = bool(getattr(ce_estimator, "pool_time", False))
        if pooled_time:
            if precomputed_features is None or not callable(fit_predict_features):
                raise ValueError(
                    "pooled-time CE requires precomputed causal features"
                )
            projected_target = torch.empty_like(pathwise_target)
            for train_indices, eval_indices in fold_specs:
                train_features = precomputed_features.index_select(
                    0, train_indices
                ).reshape(-1, int(precomputed_features.shape[-1]))
                targets_train = pathwise_target.index_select(
                    0, train_indices
                ).reshape(-1, int(pathwise_target.shape[-1]))
                eval_features = precomputed_features.index_select(
                    0, eval_indices
                ).reshape(-1, int(precomputed_features.shape[-1]))
                result = fit_predict_features(
                    features_train=train_features,
                    targets_train=targets_train,
                    features_eval=(
                        None if fold_count == 1 else eval_features
                    ),
                )
                prediction = result.prediction.reshape(
                    int(eval_indices.numel()),
                    int(pathwise_target.shape[1]),
                    int(pathwise_target.shape[2]),
                )
                projected_target.index_copy_(0, eval_indices, prediction)
                targets_eval = pathwise_target.index_select(0, eval_indices)
                residual_sum_squares = residual_sum_squares + torch.sum((prediction - targets_eval).pow(2))
                baseline = targets_train.mean(dim=0).reshape(1, 1, -1)
                baseline_sum_squares = baseline_sum_squares + torch.sum((targets_eval - baseline).pow(2))
                projection_count += int(targets_eval.numel())
                fold_train_sizes.append(int(train_indices.numel()))
                fold_eval_sizes.append(int(eval_indices.numel()))
                if "ce_fit_rms" in result.metrics:
                    fit_rms_values.append(float(result.metrics["ce_fit_rms"]))
                if "ce_feature_dim" in result.metrics:
                    feature_dims.append(float(result.metrics["ce_feature_dim"]))
        else:
            for step_index in range(int(pathwise_target.shape[1])):
                prefixes = frozen_generated_path[:, : int(step_index) + 1, :]
                targets = pathwise_target[:, step_index, :]

                projected_step = torch.empty_like(targets)
                for train_indices, eval_indices in fold_specs:
                    prefixes_train = prefixes.index_select(0, train_indices)
                    targets_train = targets.index_select(0, train_indices)
                    prefixes_eval = None if fold_count == 1 else prefixes.index_select(0, eval_indices)
                    if precomputed_features is None:
                        result = ce_estimator.fit_predict(
                            prefixes_train=prefixes_train,
                            targets_train=targets_train,
                            prefixes_eval=prefixes_eval,
                        )
                    else:
                        step_features = precomputed_features[:, step_index, :]
                        result = fit_predict_features(
                            features_train=step_features.index_select(
                                0,
                                train_indices,
                            ),
                            targets_train=targets_train,
                            features_eval=(
                                None
                                if fold_count == 1
                                else step_features.index_select(0, eval_indices)
                            ),
                        )
                    prediction = result.prediction
                    projected_step.index_copy_(0, eval_indices, prediction)
                    targets_eval = targets.index_select(0, eval_indices)
                    residual_sum_squares = residual_sum_squares + torch.sum((prediction - targets_eval).pow(2))
                    baseline = targets_train.mean(dim=0, keepdim=True)
                    baseline_sum_squares = baseline_sum_squares + torch.sum((targets_eval - baseline).pow(2))
                    projection_count += int(targets_eval.numel())
                    fold_train_sizes.append(int(train_indices.numel()))
                    fold_eval_sizes.append(int(eval_indices.numel()))
                    if "ce_fit_rms" in result.metrics:
                        fit_rms_values.append(float(result.metrics["ce_fit_rms"]))
                    if "ce_feature_dim" in result.metrics:
                        feature_dims.append(float(result.metrics["ce_feature_dim"]))
                projected_steps.append(projected_step)

            projected_target = torch.stack(projected_steps, dim=1)
        metadata = {
            "ce_projected_target_rms": float(torch.sqrt(torch.mean(projected_target.detach().pow(2))).item()),
            "ce_training_target_rms": float(torch.sqrt(torch.mean(projected_target.detach().pow(2))).item()),
            "ce_target_is_direct": 0.0,
            "ridge_lambda": float(training.ridge_lambda),
            "ce_crossfit_folds": float(fold_count),
            "ce_projection_is_crossfit": float(fold_count > 1),
            "ce_pooled_time_regression": float(pooled_time),
        }
        if projection_count > 0:
            projection_mse = residual_sum_squares / float(projection_count)
            metadata["ce_projection_rms"] = float(torch.sqrt(projection_mse).item())
            baseline_mse = baseline_sum_squares / float(projection_count)
            metadata["ce_projection_r2"] = float(
                (1.0 - projection_mse / baseline_mse.clamp_min(torch.finfo(baseline_mse.dtype).eps)).item()
            )
        if fold_train_sizes:
            metadata["ce_fold_train_size_min"] = float(min(fold_train_sizes))
            metadata["ce_fold_eval_size_min"] = float(min(fold_eval_sizes))
        if feature_dims:
            metadata["ce_feature_dim_max"] = float(max(feature_dims))
        if fit_rms_values:
            metadata["ce_fit_rms_mean"] = float(sum(fit_rms_values) / len(fit_rms_values))
            metadata["ce_fit_rms_max"] = float(max(fit_rms_values))
        return projected_target, metadata

    @staticmethod
    def _format_noise_adjoint_matrix(
        values: torch.Tensor,
        *,
        moments_noise_target_dim: int,
    ) -> torch.Tensor:
        if values.ndim != 4:
            raise ValueError("noise-adjoint matrix must have shape (B, N, d, q)")
        state_dim = int(values.shape[-2])
        noise_dim = int(values.shape[-1])
        if int(moments_noise_target_dim) == state_dim and noise_dim == state_dim:
            return torch.diagonal(values, dim1=-2, dim2=-1)
        if int(moments_noise_target_dim) == state_dim * noise_dim:
            return values.reshape(int(values.shape[0]), int(values.shape[1]), state_dim * noise_dim)
        raise ValueError(
            "noise adjoint head width must be either state_dim for diagonal controls "
            "or state_dim * noise_dim for full-matrix controls"
        )

    def _adjoint_noise_target(
        self,
        *,
        pathwise_adjoint_target: torch.Tensor,
        moments_noise_target_dim: int,
        noise: torch.Tensor,
        control_variate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pathwise_adjoint_target.ndim != 3:
            raise ValueError("pathwise_adjoint_target must have shape (B, N, d)")
        if noise.ndim != 3:
            raise ValueError("noise must have shape (B, N, q)")
        state_dim = int(pathwise_adjoint_target.shape[-1])
        noise_dim = int(noise.shape[-1])
        if int(noise.shape[0]) != int(pathwise_adjoint_target.shape[0]) or int(noise.shape[1]) != int(
            pathwise_adjoint_target.shape[1]
        ):
            raise ValueError("noise and adjoint target must have matching batch/time dimensions")
        centered = pathwise_adjoint_target.unsqueeze(-1)
        if control_variate is not None:
            baseline = control_variate.detach().to(
                device=pathwise_adjoint_target.device,
                dtype=pathwise_adjoint_target.dtype,
            )
            if baseline.ndim == 2:
                if tuple(baseline.shape) != tuple(pathwise_adjoint_target.shape[1:]):
                    raise ValueError("two-dimensional control variate must have shape (N, d)")
                baseline = baseline.unsqueeze(0).unsqueeze(-1)
            elif baseline.ndim == 3:
                if tuple(baseline.shape) != tuple(pathwise_adjoint_target.shape):
                    raise ValueError("three-dimensional control variate must have shape (B, N, d)")
                baseline = baseline.unsqueeze(-1)
            elif baseline.ndim == 4:
                expected = (
                    int(pathwise_adjoint_target.shape[1]),
                    state_dim,
                    noise_dim,
                )
                if int(baseline.shape[0]) not in {1, int(pathwise_adjoint_target.shape[0])} or tuple(
                    baseline.shape[1:]
                ) != expected:
                    raise ValueError("four-dimensional control variate must have shape (1 or B, N, d, q)")
            else:
                raise ValueError("control variate must have shape (N, d), (B, N, d), or (1 or B, N, d, q)")
            centered = centered - baseline
        target_matrix = centered * noise.detach().unsqueeze(-2)
        return self._format_noise_adjoint_matrix(
            target_matrix,
            moments_noise_target_dim=int(moments_noise_target_dim),
        )

    def _noise_target_control_variate(
        self,
        *,
        training: DiscreteMPTrainingConfig,
        raw_adjoint_prediction: torch.Tensor,
    ) -> torch.Tensor | None:
        mode = str(training.noise_target_control_variate)
        if mode == "none":
            return None
        if mode == "adjoint":
            return raw_adjoint_prediction.detach()
        if mode == "timewise":
            if self._noise_target_timewise_baseline is None:
                raise RuntimeError("timewise noise-target control variate has not been calibrated")
            return self._noise_target_timewise_baseline.unsqueeze(0)
        raise ValueError("unsupported noise_target_control_variate")

    def rollout(
        self,
        *,
        x0: torch.Tensor,
        noise: torch.Tensor,
    ) -> RolloutResult:
        if x0.ndim != 2:
            raise ValueError("x0 must have shape (B, d)")
        if int(x0.shape[1]) != int(self.architecture.state_dim):
            raise ValueError("x0 state dimension must match architecture.state_dim")
        if noise.ndim != 3:
            raise ValueError("noise must have shape (B, N, q)")
        if int(noise.shape[0]) != int(x0.shape[0]):
            raise ValueError("x0 and noise must have the same batch size")
        if int(noise.shape[1]) != int(self.grid.num_steps):
            raise ValueError("noise time dimension must align with grid")
        if int(noise.shape[2]) != int(self.architecture.noise_dim):
            raise ValueError("noise dimension must match architecture.noise_dim")

        x_points = [x0]
        controls = []
        expected_adjoint_next_values = []
        expected_adjoint_noise_next_values = []
        hidden = None
        for step_index in range(int(self.grid.num_steps)):
            x_prefix = torch.stack(x_points, dim=1)
            normalized_adjoint_next, normalized_adjoint_noise_next, hidden = self.network.forward_step(
                x_t=x_points[-1],
                hidden=hidden,
            )
            expected_adjoint_next = self._denormalize_adjoint_step(
                normalized_adjoint_next,
                step_index=step_index,
                noise_adjoint=False,
            )
            expected_adjoint_noise_next = self._denormalize_adjoint_step(
                normalized_adjoint_noise_next,
                step_index=step_index,
                noise_adjoint=True,
            )
            control = self.control_map(
                HamiltonianControlInputs(
                    step_index=int(step_index),
                    x_prefix=x_prefix,
                    expected_adjoint_next=expected_adjoint_next,
                    expected_adjoint_noise_next=expected_adjoint_noise_next,
                    parameters=self.parameters,
                )
            )
            if control.ndim == 1:
                control = control.unsqueeze(-1)
            x_next = self.dynamics_step(
                DynamicsStepInputs(
                    step_index=int(step_index),
                    x_prefix=x_prefix,
                    control=control,
                    noise_next=noise[:, step_index, :],
                    parameters=self.parameters,
                )
            )
            if x_next.ndim == 1:
                x_next = x_next.unsqueeze(-1)
            if tuple(x_next.shape) != tuple(x0.shape):
                raise ValueError("dynamics_step must return shape (B, d)")
            x_points.append(x_next)
            controls.append(control)
            expected_adjoint_next_values.append(expected_adjoint_next)
            expected_adjoint_noise_next_values.append(expected_adjoint_noise_next)

        return RolloutResult(
            paths=torch.stack(x_points, dim=1),
            controls=torch.stack(controls, dim=1),
            expected_adjoint_next=torch.stack(expected_adjoint_next_values, dim=1),
            expected_adjoint_noise_next=torch.stack(expected_adjoint_noise_next_values, dim=1),
            noise=noise,
        )

    def sample(
        self,
        *,
        num_paths: int,
        x0: torch.Tensor,
        seed: int | None = None,
    ) -> SampleResult:
        num_paths = _require_int("num_paths", num_paths)
        if num_paths < 1:
            raise ValueError("num_paths must be >= 1")
        x0_device = x0.to(device=self.device, dtype=self.dtype)
        if int(x0_device.shape[0]) == 1 and num_paths > 1:
            x0_device = x0_device.repeat(num_paths, 1)
        if int(x0_device.shape[0]) != num_paths:
            raise ValueError("x0 must have either one row or num_paths rows")
        noise = sample_standard_normals(
            grid=self.grid,
            batch_size=num_paths,
            noise_dim=int(self.architecture.noise_dim),
            device=self.device,
            dtype=self.dtype,
            seed=seed,
        )
        rollout = self.rollout(x0=x0_device, noise=noise)
        return SampleResult(
            paths=rollout.paths,
            controls=rollout.controls,
            expected_adjoint_next=rollout.expected_adjoint_next,
            expected_adjoint_noise_next=rollout.expected_adjoint_noise_next,
            noise=rollout.noise,
        )

    def fit(
        self,
        target_paths: torch.Tensor,
        *,
        training: DiscreteMPTrainingConfig | None = None,
        log_callback: Callable[["DiscreteMPModel", dict[str, float]], None] | None = None,
        replay_bank: FitReplayBank | None = None,
        conditional_target_provider: Callable[
            ["DiscreteMPModel", RolloutResult, torch.Tensor],
            ConditionalAdjointTargets,
        ]
        | None = None,
    ) -> FitResult:
        """Fit the adjoint-moment network.

        ``log_callback`` is invoked after each logged optimizer update.  It is
        intended for read-only validation and checkpoint selection; the
        default training path is unchanged when no callback is supplied.
        ``replay_bank`` optionally fixes the source/target empirical indices
        and exogenous innovations for finite-population solver audits.  When
        omitted, the standard stochastic Gaussian training path is unchanged.
        ``conditional_target_provider`` replaces only the numerical
        conditional-expectation labels; the GRU, adjoint MSE loss, rollout and
        Hamiltonian control map are unchanged.
        """
        training_config = training or self.training
        self.training = training_config
        target_count, _ = self._validate_target_paths(target_paths)
        target_paths_device = target_paths.to(device=self.device, dtype=self.dtype)
        validated_replay_bank = (
            None
            if replay_bank is None
            else self._validated_fit_replay_bank(
                replay_bank=replay_bank,
                target_count=target_count,
                training=training_config,
            )
        )
        cpu_generator = torch.Generator(device="cpu")
        if training_config.seed is not None:
            cpu_generator.manual_seed(int(training_config.seed))

        if int(training_config.ce_crossfit_folds) > int(training_config.batch_size):
            raise ValueError("ce_crossfit_folds cannot exceed batch_size")
        if conditional_target_provider is None:
            self._fit_target_preconditioner(
                target_paths=target_paths_device,
                training=training_config,
                replay_bank=validated_replay_bank,
            )
        else:
            self._fit_provided_target_preconditioner(
                target_paths=target_paths_device,
                training=training_config,
                provider=conditional_target_provider,
                replay_bank=validated_replay_bank,
            )
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=float(training_config.lr),
            weight_decay=float(training_config.weight_decay),
        )

        history: list[dict[str, float]] = []
        final_metrics: dict[str, float] | None = None
        rolling_diagnostics: dict[str, list[float]] = {}
        run_diagnostics: dict[str, list[float]] = {}
        for step in range(int(training_config.num_steps)):
            if validated_replay_bank is None:
                source_indices = self._sample_indices(
                    size=target_count,
                    count=int(training_config.batch_size),
                    generator=cpu_generator,
                )
                target_indices = self._sample_indices(
                    size=target_count,
                    count=int(training_config.target_batch_size),
                    generator=cpu_generator,
                )
                noise = sample_standard_normals(
                    grid=self.grid,
                    batch_size=int(training_config.batch_size),
                    noise_dim=int(self.architecture.noise_dim),
                    device=self.device,
                    dtype=self.dtype,
                    seed=derive_stream_seed(
                        base_seed=training_config.seed,
                        stream_offset=step + 1,
                    ),
                )
            else:
                source_indices = validated_replay_bank.source_indices
                target_indices = validated_replay_bank.target_indices
                noise = validated_replay_bank.noise.detach().clone()
            x0 = target_paths_device.index_select(0, source_indices.to(self.device))[:, 0, :]
            target_batch = target_paths_device.index_select(0, target_indices.to(self.device))
            use_stein_target = bool(
                conditional_target_provider is None
                and str(training_config.noise_target_estimator) == "stein"
                and float(training_config.adjoint_noise_weight) > 0.0
            )
            provided_noise_target = None
            if conditional_target_provider is not None:
                with torch.no_grad():
                    rollout = self.rollout(x0=x0, noise=noise)
                supplied = self._provided_conditional_adjoint_targets(
                    provider=conditional_target_provider,
                    rollout=rollout,
                    target_paths=target_batch,
                )
                adjoint_target = supplied.expected_adjoint_next
                provided_noise_target = supplied.expected_adjoint_noise_next
                target_metadata = dict(supplied.metrics)
                target_metadata.setdefault(
                    "target_rms",
                    float(torch.sqrt(torch.mean(adjoint_target.pow(2))).item()),
                )
                target_metadata.setdefault(
                    "source_rms", float(target_metadata["target_rms"])
                )
                target_metadata["conditional_target_provider"] = 1.0
                stein_adjoint_noise_target = None
            elif use_stein_target:
                noise = noise.detach().requires_grad_(True)
                rollout = self.rollout(x0=x0, noise=noise)
                adjoint_target, target_metadata = self._path_functional_adjoint_targets(
                    generated_path=rollout.paths,
                    controls=rollout.controls,
                    target_path=target_batch,
                    training=training_config,
                    create_graph=True,
                    preserve_generated_path_graph=True,
                )
                stein_adjoint_noise_target = self._stein_adjoint_noise_target(
                    differentiable_adjoint_target=adjoint_target,
                    noise=noise,
                    moments_noise_target_dim=int(
                        self._default_noise_adjoint_dim(self.architecture)
                    ),
                    num_probes=int(training_config.stein_num_probes),
                    seed=derive_stream_seed(
                        base_seed=training_config.seed,
                        stream_offset=1_000_000 + step,
                    ),
                )
            else:
                stein_adjoint_noise_target = None
                with torch.no_grad():
                    rollout = self.rollout(x0=x0, noise=noise)
                adjoint_target, target_metadata = self._path_functional_adjoint_targets(
                    generated_path=rollout.paths,
                    controls=rollout.controls,
                    target_path=target_batch,
                    training=training_config,
                )
            frozen_generated_path = rollout.paths.detach()
            normalized_moments = self.network(frozen_generated_path)
            moments = self._denormalize_moments(normalized_moments)
            pathwise_adjoint_target = adjoint_target.detach()
            if conditional_target_provider is None:
                with torch.no_grad():
                    projected_adjoint_target, ce_metadata = self._project_pathwise_target(
                        frozen_generated_path=frozen_generated_path,
                        pathwise_target=pathwise_adjoint_target,
                        training=training_config,
                    )
            else:
                projected_adjoint_target = pathwise_adjoint_target
                ce_metadata = {
                    "ce_oracle_conditional_targets": 1.0,
                    "ce_projected_target_rms": float(
                        torch.sqrt(
                            torch.mean(projected_adjoint_target.pow(2))
                        ).item()
                    ),
                }
            normalized_adjoint_target = self._normalize_adjoint_target(
                projected_adjoint_target,
                noise_adjoint=False,
            )
            self._check_same_shape(
                actual=normalized_moments.expected_adjoint_next,
                expected=normalized_adjoint_target,
                name="expected_adjoint_next",
            )
            adjoint_loss = torch.mean(
                (normalized_moments.expected_adjoint_next - normalized_adjoint_target).pow(2)
            )
            raw_adjoint_loss = torch.mean(
                (moments.expected_adjoint_next - projected_adjoint_target).pow(2)
            )

            noise_loss = adjoint_loss.new_tensor(0.0)
            adjoint_noise_target = None
            uncentered_adjoint_noise_target = None
            noise_control_variate = None
            projected_adjoint_noise_target = None
            normalized_adjoint_noise_target = None
            raw_noise_loss = adjoint_loss.new_tensor(0.0)
            noise_ce_metadata: dict[str, float] = {}
            if float(training_config.adjoint_noise_weight) > 0.0:
                if provided_noise_target is not None:
                    adjoint_noise_target = provided_noise_target.detach()
                elif stein_adjoint_noise_target is not None:
                    adjoint_noise_target = stein_adjoint_noise_target
                else:
                    uncentered_adjoint_noise_target = self._adjoint_noise_target(
                        pathwise_adjoint_target=pathwise_adjoint_target,
                        moments_noise_target_dim=int(normalized_moments.expected_adjoint_noise_next.shape[-1]),
                        noise=noise,
                    )
                    noise_control_variate = self._noise_target_control_variate(
                        training=training_config,
                        raw_adjoint_prediction=moments.expected_adjoint_next,
                    )
                    if noise_control_variate is None:
                        adjoint_noise_target = uncentered_adjoint_noise_target
                    else:
                        adjoint_noise_target = self._adjoint_noise_target(
                            pathwise_adjoint_target=pathwise_adjoint_target,
                            moments_noise_target_dim=int(normalized_moments.expected_adjoint_noise_next.shape[-1]),
                            noise=noise,
                            control_variate=noise_control_variate,
                        )
                if provided_noise_target is None:
                    with torch.no_grad():
                        projected_adjoint_noise_target, raw_noise_ce_metadata = self._project_pathwise_target(
                            frozen_generated_path=frozen_generated_path,
                            pathwise_target=adjoint_noise_target,
                            training=training_config,
                            noise_adjoint=True,
                        )
                    noise_ce_metadata = {
                        f"noise_{key}": float(value)
                        for key, value in raw_noise_ce_metadata.items()
                    }
                else:
                    projected_adjoint_noise_target = adjoint_noise_target
                    noise_ce_metadata = {
                        "noise_ce_oracle_conditional_targets": 1.0,
                        "noise_ce_projected_target_rms": float(
                            torch.sqrt(
                                torch.mean(
                                    projected_adjoint_noise_target.pow(2)
                                )
                            ).item()
                        ),
                    }
                normalized_adjoint_noise_target = self._normalize_adjoint_target(
                    projected_adjoint_noise_target,
                    noise_adjoint=True,
                )
                self._check_same_shape(
                    actual=normalized_moments.expected_adjoint_noise_next,
                    expected=normalized_adjoint_noise_target,
                    name="expected_adjoint_noise_next",
                )
                noise_loss = torch.mean(
                    (normalized_moments.expected_adjoint_noise_next - normalized_adjoint_noise_target).pow(2)
                )
                raw_noise_loss = torch.mean(
                    (moments.expected_adjoint_noise_next - projected_adjoint_noise_target).pow(2)
                )

            total_loss = (
                float(training_config.adjoint_weight) * adjoint_loss
                + float(training_config.adjoint_noise_weight) * noise_loss
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            parameters = list(self.network.parameters())
            parameter_norm_before = self._parameter_norm(parameters)
            parameter_values_before = [parameter.detach().clone() for parameter in parameters]
            parameter_values_before_by_id = {
                id(parameter): previous
                for parameter, previous in zip(parameters, parameter_values_before)
            }
            unclipped_grad_norm = self._grad_norm(parameters)
            component_grad_norms = self._component_grad_norms()
            clip_active = float(
                training_config.grad_clip_norm is not None
                and unclipped_grad_norm > float(training_config.grad_clip_norm)
            )
            if training_config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters, float(training_config.grad_clip_norm))
            clipped_grad_norm = self._grad_norm(parameters)
            clipped_component_grad_norms = self._component_grad_norms(
                prefix="grad_norm_clipped"
            )
            grad_clip_scale = (
                clipped_grad_norm / unclipped_grad_norm
                if unclipped_grad_norm > 0.0
                else 1.0
            )
            optimizer.step()
            update_norm = self._update_norm(parameters, parameter_values_before)
            component_update_norms = self._component_update_norms(
                parameter_values_before_by_id
            )
            update_to_parameter_norm = update_norm / max(parameter_norm_before, torch.finfo(torch.float64).eps)

            adjoint_target_rms = float(torch.sqrt(torch.mean(projected_adjoint_target.detach().pow(2))).item())
            adjoint_prediction_rms = float(
                torch.sqrt(torch.mean(moments.expected_adjoint_next.detach().pow(2))).item()
            )
            adjoint_relative_rmse = float(torch.sqrt(raw_adjoint_loss.detach()).item()) / max(
                adjoint_target_rms,
                torch.finfo(torch.float64).eps,
            )
            normalized_adjoint_target_rms = float(
                torch.sqrt(torch.mean(normalized_adjoint_target.detach().pow(2))).item()
            )
            normalized_adjoint_relative_rmse = float(torch.sqrt(adjoint_loss.detach()).item()) / max(
                normalized_adjoint_target_rms,
                torch.finfo(torch.float64).eps,
            )
            if projected_adjoint_noise_target is None:
                noise_target_rms = 0.0
                noise_relative_rmse = 0.0
                normalized_noise_target_rms = 0.0
                normalized_noise_relative_rmse = 0.0
            else:
                noise_target_rms = float(
                    torch.sqrt(torch.mean(projected_adjoint_noise_target.detach().pow(2))).item()
                )
                noise_prediction_rms = float(
                    torch.sqrt(torch.mean(moments.expected_adjoint_noise_next.detach().pow(2))).item()
                )
                noise_relative_rmse = float(torch.sqrt(raw_noise_loss.detach()).item()) / max(
                    noise_target_rms,
                    torch.finfo(torch.float64).eps,
                )
                assert normalized_adjoint_noise_target is not None
                normalized_noise_target_rms = float(
                    torch.sqrt(torch.mean(normalized_adjoint_noise_target.detach().pow(2))).item()
                )
                normalized_noise_relative_rmse = float(torch.sqrt(noise_loss.detach()).item()) / max(
                    normalized_noise_target_rms,
                    torch.finfo(torch.float64).eps,
                )
            if projected_adjoint_noise_target is None:
                noise_prediction_rms = 0.0
            step_diagnostics = {
                "grad_norm": float(unclipped_grad_norm),
                "grad_norm_clipped": float(clipped_grad_norm),
                "clip_active": clip_active,
                "grad_clip_scale": float(grad_clip_scale),
                "train_total_loss": float(total_loss.detach().item()),
                "adjoint_relative_rmse": adjoint_relative_rmse,
                "noise_adjoint_relative_rmse": noise_relative_rmse,
                "normalized_adjoint_relative_rmse": normalized_adjoint_relative_rmse,
                "normalized_noise_adjoint_relative_rmse": normalized_noise_relative_rmse,
                "adjoint_prediction_target_rms_ratio": adjoint_prediction_rms
                / max(adjoint_target_rms, torch.finfo(torch.float64).eps),
                "noise_adjoint_prediction_target_rms_ratio": noise_prediction_rms
                / max(noise_target_rms, torch.finfo(torch.float64).eps),
                "parameter_update_norm": float(update_norm),
                "update_to_parameter_norm": float(update_to_parameter_norm),
                "gradient_nonfinite_fraction": self._nonfinite_fraction(
                    [parameter.grad for parameter in parameters if parameter.grad is not None]
                ),
                "parameter_nonfinite_fraction": self._nonfinite_fraction(
                    [parameter for parameter in parameters]
                ),
                "target_rms": float(target_metadata["target_rms"]),
                "ce_projected_target_rms": float(ce_metadata["ce_projected_target_rms"]),
                "noise_ce_projected_target_rms": float(
                    noise_ce_metadata.get("noise_ce_projected_target_rms", 0.0)
                ),
                "noise_target_estimator_stein": float(use_stein_target),
                "stein_num_probes": float(
                    training_config.stein_num_probes if use_stein_target else 0
                ),
            }
            if uncentered_adjoint_noise_target is not None and adjoint_noise_target is not None:
                uncentered_second_moment = torch.mean(
                    uncentered_adjoint_noise_target.detach().pow(2)
                ).clamp_min(torch.finfo(uncentered_adjoint_noise_target.dtype).eps)
                step_diagnostics["noise_target_control_variate_second_moment_ratio"] = float(
                    (torch.mean(adjoint_noise_target.detach().pow(2)) / uncentered_second_moment).item()
                )
            if "ce_projection_r2" in ce_metadata:
                step_diagnostics["ce_projection_r2"] = float(ce_metadata["ce_projection_r2"])
            if "noise_ce_projection_r2" in noise_ce_metadata:
                step_diagnostics["noise_ce_projection_r2"] = float(
                    noise_ce_metadata["noise_ce_projection_r2"]
                )
            step_diagnostics.update(component_grad_norms)
            step_diagnostics.update(clipped_component_grad_norms)
            step_diagnostics.update(component_update_norms)
            for name, value in step_diagnostics.items():
                rolling_diagnostics.setdefault(name, []).append(float(value))
                run_diagnostics.setdefault(name, []).append(float(value))

            if (
                step == 0
                or (step + 1) % int(training_config.log_every) == 0
                or step == int(training_config.num_steps) - 1
            ):
                metrics = dict(target_metadata)
                metrics.update(ce_metadata)
                metrics.update(noise_ce_metadata)
                metrics.update(self._target_preconditioner_metrics)
                metrics["fit_replay_bank"] = float(
                    validated_replay_bank is not None
                )
                metrics["train_total_loss"] = float(total_loss.detach().item())
                metrics["train_adjoint_consistency_loss"] = float(adjoint_loss.detach().item())
                metrics["train_adjoint_raw_mse"] = float(raw_adjoint_loss.detach().item())
                metrics["train_adjoint_consistency_weighted"] = float(
                    (float(training_config.adjoint_weight) * adjoint_loss.detach()).item()
                )
                metrics["train_adjoint_noise_consistency_loss"] = float(noise_loss.detach().item())
                metrics["train_adjoint_noise_raw_mse"] = float(raw_noise_loss.detach().item())
                metrics["train_adjoint_noise_consistency_weighted"] = float(
                    (float(training_config.adjoint_noise_weight) * noise_loss.detach()).item()
                )
                metrics["step"] = float(step + 1)
                metrics["grad_norm"] = float(unclipped_grad_norm)
                metrics["grad_norm_clipped"] = float(clipped_grad_norm)
                metrics["grad_clip_active"] = clip_active
                metrics["grad_clip_scale"] = float(grad_clip_scale)
                metrics["grad_clip_norm"] = (
                    float(training_config.grad_clip_norm)
                    if training_config.grad_clip_norm is not None
                    else 0.0
                )
                metrics.update(component_grad_norms)
                metrics.update(clipped_component_grad_norms)
                metrics["parameter_norm_before_update"] = float(parameter_norm_before)
                metrics["parameter_update_norm"] = float(update_norm)
                metrics.update(component_update_norms)
                metrics["update_to_parameter_norm"] = float(update_to_parameter_norm)
                metrics["control_rms"] = float(torch.sqrt(torch.mean(rollout.controls.detach().pow(2))).item())
                metrics["expected_adjoint_rms"] = float(
                    torch.sqrt(torch.mean(moments.expected_adjoint_next.detach().pow(2))).item()
                )
                metrics["expected_adjoint_noise_rms"] = float(
                    torch.sqrt(torch.mean(moments.expected_adjoint_noise_next.detach().pow(2))).item()
                )
                metrics["normalized_adjoint_target_rms"] = normalized_adjoint_target_rms
                metrics["normalized_adjoint_relative_rmse"] = normalized_adjoint_relative_rmse
                metrics["normalized_noise_adjoint_target_rms"] = normalized_noise_target_rms
                metrics["normalized_noise_adjoint_relative_rmse"] = normalized_noise_relative_rmse
                metrics.update(
                    self._moment_diagnostics(
                        moments.expected_adjoint_next,
                        projected_adjoint_target,
                        prefix="adjoint",
                    )
                )
                if projected_adjoint_noise_target is not None:
                    metrics.update(
                        self._moment_diagnostics(
                            moments.expected_adjoint_noise_next,
                            projected_adjoint_noise_target,
                            prefix="noise_adjoint",
                        )
                    )
                metrics.update(self._control_diagnostics(rollout.controls))
                metrics.update(self._rolling_diagnostic_summary(rolling_diagnostics))
                if step == int(training_config.num_steps) - 1:
                    metrics.update(
                        {
                            key.replace("rolling_", "run_", 1): value
                            for key, value in self._rolling_diagnostic_summary(run_diagnostics).items()
                        }
                    )
                metrics.update(self._abs_tensor_stats(pathwise_adjoint_target, prefix="pathwise_adjoint_target"))
                metrics.update(
                    self._abs_tensor_stats(projected_adjoint_target, prefix="projected_adjoint_target")
                )
                if adjoint_noise_target is not None:
                    metrics.update(self._abs_tensor_stats(adjoint_noise_target, prefix="adjoint_noise_target"))
                if uncentered_adjoint_noise_target is not None:
                    metrics.update(
                        self._abs_tensor_stats(
                            uncentered_adjoint_noise_target,
                            prefix="uncentered_adjoint_noise_target",
                        )
                    )
                if noise_control_variate is not None:
                    metrics.update(
                        self._abs_tensor_stats(
                            noise_control_variate,
                            prefix="noise_target_control_variate",
                        )
                    )
                if projected_adjoint_noise_target is not None:
                    metrics.update(
                        self._abs_tensor_stats(
                            projected_adjoint_noise_target,
                            prefix="projected_adjoint_noise_target",
                        )
                    )
                history.append(metrics)
                final_metrics = metrics
                if log_callback is not None:
                    log_callback(self, dict(metrics))
                rolling_diagnostics.clear()

        if final_metrics is None:
            raise RuntimeError("training produced no metrics")
        final_metrics.update(
            self._fresh_regression_probe(
                target_paths=target_paths_device,
                training=training_config,
                replay_bank=validated_replay_bank,
                conditional_target_provider=conditional_target_provider,
            )
        )
        return FitResult(history=history, final_metrics=final_metrics)

    def _fit_timewise_linear_residual_lstsq(
        self,
        *,
        frozen_paths: torch.Tensor,
        normalized_adjoint_target: torch.Tensor,
        normalized_adjoint_noise_target: torch.Tensor | None,
    ) -> None:
        """Exactly solve the affine residual's frozen regression problem.

        This is a numerical solver for the same squared conditional-moment
        regression used by :meth:`fit_nested`.  It does not alter the MP
        target or add a penalty.  A Moore--Penrose solution is used because
        causal features are necessarily rank deficient at the first few time
        steps (for example, all lagged-return features are then zero).
        """

        combined = self.network
        base = getattr(combined, "base_network", None)
        residual = getattr(combined, "residual_network", None)
        if base is None or residual is None:
            raise ValueError(
                "timewise_linear_lstsq requires a frozen-base residual network"
            )
        p_head = getattr(residual, "expected_adjoint_next_head", None)
        r_head = getattr(
            residual,
            "expected_adjoint_noise_next_head",
            None,
        )
        feature_map = getattr(residual, "feature_map", None)
        standardize = getattr(residual, "_standardize", None)
        if (
            p_head is None
            or r_head is None
            or feature_map is None
            or not callable(standardize)
            or not hasattr(p_head, "weight")
            or not hasattr(p_head, "bias")
            or not hasattr(r_head, "weight")
            or not hasattr(r_head, "bias")
        ):
            raise ValueError(
                "timewise_linear_lstsq requires timewise affine P/R heads"
            )
        if normalized_adjoint_noise_target is None:
            raise ValueError(
                "timewise_linear_lstsq requires a noise-adjoint target"
            )

        with torch.no_grad():
            base_moments = base(frozen_paths)
            raw_features = feature_map.features_all(frozen_paths)
            features = standardize(raw_features)
            ones = torch.ones(
                int(features.shape[0]),
                int(features.shape[1]),
                1,
                device=features.device,
                dtype=features.dtype,
            )
            # torch.linalg.pinv handles the unavoidable early-time rank
            # deficiency and returns the minimum-norm least-squares solution.
            design = torch.cat((ones, features), dim=2).transpose(0, 1)
            design_pinv = torch.linalg.pinv(design, rtol=1e-5)

            def solve_head(
                head: object,
                target: torch.Tensor,
                base_prediction: torch.Tensor,
            ) -> None:
                response = (target - base_prediction).transpose(0, 1)
                coefficients = torch.matmul(design_pinv, response)
                head.bias.copy_(coefficients[:, 0, :])
                head.weight.copy_(coefficients[:, 1:, :].transpose(1, 2))

            solve_head(
                p_head,
                normalized_adjoint_target,
                base_moments.expected_adjoint_next,
            )
            solve_head(
                r_head,
                normalized_adjoint_noise_target,
                base_moments.expected_adjoint_noise_next,
            )

    def _nested_branch_conditional_targets(
        self,
        *,
        x0: torch.Tensor,
        base_noise: torch.Tensor,
        base_paths: torch.Tensor,
        population_paths: torch.Tensor | None = None,
        target_path: torch.Tensor,
        training: DiscreteMPTrainingConfig,
        num_branches: int,
        antithetic: bool,
        query_batch_size: int,
        seed: int | None,
        branch_noise_provider: Callable[[int], torch.Tensor] | None = None,
        rollout_fn: Callable[..., RolloutResult] | None = None,
        branch_target_observer: Callable[
            [int, torch.Tensor, torch.Tensor, torch.Tensor], None
        ]
        | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Estimate the complete-MP conditional moments by future branching.

        For every decision time, independent futures are simulated from the
        same observable prefixes.  Averaging ``A_n`` and
        ``A_n times epsilon_{n+1}`` over those futures directly estimates
        the two conditional expectations in the stochastic maximum
        principle.  Antithetic futures preserve the Gaussian marginal law
        and remove any branch-common causal baseline from the score average.
        No objective term or neural regression loss is changed.
        """

        branch_count = int(num_branches)
        if branch_count < 2:
            raise ValueError("nested conditional targets require at least two branches")
        if bool(antithetic) and branch_count % 2 != 0:
            raise ValueError("antithetic nested conditional targets require an even branch count")
        max_query_paths = int(query_batch_size)
        if max_query_paths < 1:
            raise ValueError("nested conditional query_batch_size must be positive")
        if x0.ndim != 2 or base_noise.ndim != 3 or base_paths.ndim != 3:
            raise ValueError("nested conditional target inputs have invalid ranks")
        batch_size = int(x0.shape[0])
        num_steps = int(self.grid.num_steps)
        if tuple(base_noise.shape[:2]) != (batch_size, num_steps):
            raise ValueError("base_noise must align with x0 and the model grid")
        if tuple(base_paths.shape[:2]) != (batch_size, num_steps + 1):
            raise ValueError("base_paths must align with x0 and the model grid")
        frozen_population_paths = (
            base_paths if population_paths is None else population_paths
        )
        if frozen_population_paths.ndim != 3 or tuple(
            frozen_population_paths.shape[1:]
        ) != tuple(base_paths.shape[1:]):
            raise ValueError("population_paths must align with the model grid")

        state_dim = int(self.architecture.state_dim)
        noise_target_dim = int(self._default_noise_adjoint_dim(self.architecture))
        nested_p = base_paths.new_empty((batch_size, num_steps, state_dim))
        nested_r = base_paths.new_empty((batch_size, num_steps, noise_target_dim))
        metadata_rows: list[dict[str, float]] = []
        prefix_errors: list[float] = []

        queries_per_step = batch_size * branch_count
        if max_query_paths < queries_per_step:
            raise ValueError(
                "nested conditional query_batch_size must hold at least one "
                "complete branch population"
            )
        steps_per_chunk = max(1, max_query_paths // queries_per_step)
        independent_count = branch_count // 2 if bool(antithetic) else branch_count
        flat_x0_one_step = x0[:, None, :].expand(
            batch_size,
            branch_count,
            int(x0.shape[-1]),
        ).reshape(queries_per_step, int(x0.shape[-1]))

        for chunk_start in range(0, num_steps, steps_per_chunk):
            chunk_steps = tuple(
                range(chunk_start, min(num_steps, chunk_start + steps_per_chunk))
            )
            chunk_noise_rows: list[torch.Tensor] = []
            for step_index in chunk_steps:
                if branch_noise_provider is None:
                    independent_noise = sample_standard_normals(
                        grid=self.grid,
                        batch_size=batch_size * independent_count,
                        noise_dim=int(self.architecture.noise_dim),
                        device=self.device,
                        dtype=self.dtype,
                        seed=derive_stream_seed(
                            base_seed=seed,
                            stream_offset=10_000 + step_index,
                        ),
                    ).reshape(
                        batch_size,
                        independent_count,
                        num_steps,
                        int(self.architecture.noise_dim),
                    )
                    branch_noise = (
                        torch.cat((independent_noise, -independent_noise), dim=1)
                        if bool(antithetic)
                        else independent_noise
                    )
                    if step_index > 0:
                        branch_noise[:, :, :step_index, :] = base_noise[
                            :, None, :step_index, :
                        ]
                else:
                    branch_noise = branch_noise_provider(int(step_index))
                    expected_branch_shape = (
                        batch_size,
                        branch_count,
                        num_steps,
                        int(self.architecture.noise_dim),
                    )
                    if tuple(branch_noise.shape) != expected_branch_shape:
                        raise ValueError(
                            "branch_noise_provider must return shape "
                            f"{expected_branch_shape}"
                        )
                    branch_noise = branch_noise.to(
                        device=self.device,
                        dtype=self.dtype,
                    )
                chunk_noise_rows.append(
                    branch_noise.reshape(
                        queries_per_step,
                        num_steps,
                        int(self.architecture.noise_dim),
                    )
                )
            flat_noise = torch.cat(chunk_noise_rows, dim=0)
            flat_x0 = flat_x0_one_step.repeat(len(chunk_steps), 1)
            with torch.no_grad():
                branch_rollout = (
                    self.rollout(x0=flat_x0, noise=flat_noise)
                    if rollout_fn is None
                    else rollout_fn(x0=flat_x0, noise=flat_noise)
                )
            pathwise_target, metadata = self._frozen_law_path_functional_adjoint_targets(
                query_path=branch_rollout.paths,
                population_path=frozen_population_paths,
                controls=branch_rollout.controls,
                target_path=target_path,
                training=training,
            )
            score_target = self._adjoint_noise_target(
                pathwise_adjoint_target=pathwise_target.detach(),
                moments_noise_target_dim=noise_target_dim,
                noise=flat_noise,
            )
            for local_index, step_index in enumerate(chunk_steps):
                row_start = local_index * queries_per_step
                row_stop = row_start + queries_per_step
                branch_p = pathwise_target[
                    row_start:row_stop, step_index, :
                ].reshape(batch_size, branch_count, state_dim)
                branch_r = score_target[
                    row_start:row_stop, step_index, :
                ].reshape(batch_size, branch_count, noise_target_dim)
                if branch_target_observer is not None:
                    branch_target_observer(
                        int(step_index),
                        base_paths[:, : int(step_index) + 1, :].detach(),
                        branch_p.detach(),
                        branch_r.detach(),
                    )
                nested_p[:, step_index, :] = branch_p.mean(dim=1)
                nested_r[:, step_index, :] = branch_r.mean(dim=1)
                if step_index > 0:
                    branch_prefix = branch_rollout.paths[
                        row_start:row_stop
                    ].reshape(
                        batch_size,
                        branch_count,
                        num_steps + 1,
                        state_dim,
                    )[:, :, : step_index + 1, :]
                    expected_prefix = base_paths[:, None, : step_index + 1, :]
                    prefix_errors.append(
                        float(
                            torch.max(torch.abs(branch_prefix - expected_prefix))
                            .detach()
                            .item()
                        )
                    )
            metadata_rows.append(metadata)

        common_names = set.intersection(*(set(row) for row in metadata_rows))
        averaged_metadata = {
            name: float(
                sum(float(row[name]) for row in metadata_rows)
                / len(metadata_rows)
            )
            for name in common_names
        }
        averaged_metadata.update(
            {
                "nested_conditional_target": 1.0,
                "nested_conditional_branches": float(branch_count),
                "nested_conditional_antithetic": float(bool(antithetic)),
                "nested_conditional_query_batch_size": float(max_query_paths),
                "nested_conditional_query_chunks": float(len(metadata_rows)),
                "nested_conditional_steps_per_chunk": float(steps_per_chunk),
                "nested_conditional_external_branch_replay": float(
                    branch_noise_provider is not None
                ),
                "nested_conditional_external_rollout": float(
                    rollout_fn is not None
                ),
                "nested_conditional_prefix_max_abs_error": (
                    max(prefix_errors) if prefix_errors else 0.0
                ),
                "nested_conditional_p_rms": float(
                    torch.sqrt(torch.mean(nested_p.detach().pow(2))).item()
                ),
                "nested_conditional_r_rms": float(
                    torch.sqrt(torch.mean(nested_r.detach().pow(2))).item()
                ),
            }
        )
        return nested_p.detach(), nested_r.detach(), averaged_metadata

    def fit_nested(
        self,
        target_paths: torch.Tensor,
        *,
        training: DiscreteMPTrainingConfig | None = None,
        nested: DiscreteMPNestedTrainingConfig | None = None,
        log_callback: Callable[["DiscreteMPModel", dict[str, float]], None] | None = None,
    ) -> FitResult:
        """Solve the forward/backward fixed point on two explicit timescales.

        The forward path bank, complete pathwise adjoint source, and projected
        conditional targets are frozen for an entire inner loop.  The inner
        optimizer minimizes the same weighted P/R regression loss as
        :meth:`fit`; only after those updates is the forward law regenerated.
        The optional ``adjoint_blocks`` outer relaxation uses the same
        candidate and complete empirical objective, but selects separate
        interpolation rates for its P head, R head, and shared parameters.
        """

        training_config = training or self.training
        nested_config = nested or DiscreteMPNestedTrainingConfig(seed=training_config.seed)
        self.training = training_config
        target_count, _ = self._validate_target_paths(target_paths)
        target_paths_device = target_paths.to(device=self.device, dtype=self.dtype)
        if int(training_config.ce_crossfit_folds) > int(nested_config.outer_batch_size):
            raise ValueError("ce_crossfit_folds cannot exceed nested outer_batch_size")
        self._fit_target_preconditioner(
            target_paths=target_paths_device,
            training=training_config,
        )
        parameters = list(self.network.parameters())
        outer_parameter_blocks = []
        for name, _parameter in self.network.named_parameters():
            if name.startswith("expected_adjoint_next_head.") or (
                ".expected_adjoint_next_head." in name
            ) or name.startswith("expected_adjoint_next_heads.") or (
                ".expected_adjoint_next_heads." in name
            ):
                outer_parameter_blocks.append("p")
            elif name.startswith("expected_adjoint_noise_next_head.") or (
                ".expected_adjoint_noise_next_head." in name
            ) or name.startswith("expected_adjoint_noise_next_heads.") or (
                ".expected_adjoint_noise_next_heads." in name
            ):
                outer_parameter_blocks.append("r")
            else:
                outer_parameter_blocks.append("shared")
        if len(outer_parameter_blocks) != len(parameters):
            raise RuntimeError("nested outer parameter-block mapping is inconsistent")
        cpu_generator = torch.Generator(device="cpu")
        nested_seed = nested_config.seed
        if nested_seed is None:
            cpu_generator.seed()
        else:
            cpu_generator.manual_seed(int(nested_seed))

        probe_generator = torch.Generator(device="cpu")
        probe_seed = derive_stream_seed(base_seed=nested_seed, stream_offset=500_000)
        if probe_seed is None:
            probe_generator.seed()
        else:
            probe_generator.manual_seed(int(probe_seed))
        probe_indices = self._sample_indices(
            size=target_count,
            count=int(nested_config.fixed_point_probe_paths),
            generator=probe_generator,
        )
        probe_x0 = target_paths_device.index_select(
            0,
            probe_indices.to(target_paths_device.device),
        )[:, 0, :]
        probe_noise = sample_standard_normals(
            grid=self.grid,
            batch_size=int(nested_config.fixed_point_probe_paths),
            noise_dim=int(self.architecture.noise_dim),
            device=self.device,
            dtype=self.dtype,
            seed=derive_stream_seed(base_seed=probe_seed, stream_offset=1),
        )
        with torch.no_grad():
            previous_probe = self.rollout(x0=probe_x0, noise=probe_noise)

        history: list[dict[str, float]] = []
        final_metrics: dict[str, float] | None = None
        global_inner_step = 0
        for outer_index in range(int(nested_config.outer_steps)):
            # The inner optimizer belongs to one frozen backward problem.  Its
            # moments must not leak into the next outer problem, especially
            # after an accepted parameter relaxation.
            optimizer = torch.optim.AdamW(
                self.network.parameters(),
                lr=float(training_config.lr),
                weight_decay=float(training_config.weight_decay),
            )
            outer_parameter_values = [
                parameter.detach().clone() for parameter in parameters
            ]
            use_stein_target = (
                str(training_config.noise_target_estimator) == "stein"
                and float(training_config.adjoint_noise_weight) > 0.0
            )
            use_nested_conditional_target = (
                str(nested_config.outer_target_estimator) == "nested_branches"
            )
            if use_nested_conditional_target and use_stein_target:
                raise ValueError(
                    "nested branch targets use the unbiased score identity and "
                    "cannot be combined with the Stein estimator"
                )
            if use_stein_target and int(nested_config.outer_backward_replicates) > 1:
                raise ValueError(
                    "replicated backward banks currently require the score R estimator"
                )
            regression_paths = []
            regression_noises = []
            regression_targets = []
            regression_noise_targets = []
            replicate_metadata_rows: list[dict[str, float]] = []
            line_objective_rows: list[dict[str, float]] = []
            line_search_specs: list[
                tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ] = []
            rollout = None
            outer_noise = None
            x0 = None
            target_batch = None
            stein_adjoint_noise_target = None
            for replicate in range(int(nested_config.outer_backward_replicates)):
                source_indices = self._sample_indices(
                    size=target_count,
                    count=int(nested_config.outer_batch_size),
                    generator=cpu_generator,
                )
                if (
                    use_nested_conditional_target
                    and int(nested_config.outer_target_batch_size) == target_count
                ):
                    target_indices = torch.randperm(
                        target_count,
                        generator=cpu_generator,
                        device="cpu",
                    )
                else:
                    target_indices = self._sample_indices(
                        size=target_count,
                        count=int(nested_config.outer_target_batch_size),
                        generator=cpu_generator,
                    )
                replicate_x0 = target_paths_device.index_select(
                    0, source_indices.to(self.device)
                )[:, 0, :]
                replicate_target_batch = target_paths_device.index_select(
                    0, target_indices.to(self.device)
                )
                replicate_noise = sample_standard_normals(
                    grid=self.grid,
                    batch_size=int(nested_config.outer_batch_size),
                    noise_dim=int(self.architecture.noise_dim),
                    device=self.device,
                    dtype=self.dtype,
                    seed=derive_stream_seed(
                        base_seed=nested_seed,
                        stream_offset=(
                            200_000
                            + outer_index
                            * int(nested_config.outer_backward_replicates)
                            + replicate
                        ),
                    ),
                )
                replicate_population_paths = None
                if (
                    not use_nested_conditional_target
                    and nested_config.outer_population_batch_size is not None
                ):
                    population_batch_size = int(
                        nested_config.outer_population_batch_size
                    )
                    population_indices = self._sample_indices(
                        size=target_count,
                        count=population_batch_size,
                        generator=cpu_generator,
                    )
                    population_x0 = target_paths_device.index_select(
                        0, population_indices.to(self.device)
                    )[:, 0, :]
                    population_noise = sample_standard_normals(
                        grid=self.grid,
                        batch_size=population_batch_size,
                        noise_dim=int(self.architecture.noise_dim),
                        device=self.device,
                        dtype=self.dtype,
                        seed=derive_stream_seed(
                            base_seed=nested_seed,
                            stream_offset=(
                                900_000
                                + outer_index
                                * int(nested_config.outer_backward_replicates)
                                + replicate
                            ),
                        ),
                    )
                    with torch.no_grad():
                        replicate_population_paths = self.rollout(
                            x0=population_x0,
                            noise=population_noise,
                        ).paths.detach()
                if use_nested_conditional_target:
                    with torch.no_grad():
                        replicate_rollout = self.rollout(
                            x0=replicate_x0,
                            noise=replicate_noise,
                        )
                    population_batch_size = (
                        int(nested_config.outer_batch_size)
                        if nested_config.outer_population_batch_size is None
                        else int(nested_config.outer_population_batch_size)
                    )
                    if population_batch_size == int(
                        nested_config.outer_batch_size
                    ):
                        replicate_population_paths = (
                            replicate_rollout.paths.detach()
                        )
                    else:
                        population_indices = self._sample_indices(
                            size=target_count,
                            count=population_batch_size,
                            generator=cpu_generator,
                        )
                        population_x0 = target_paths_device.index_select(
                            0, population_indices.to(self.device)
                        )[:, 0, :]
                        population_noise = sample_standard_normals(
                            grid=self.grid,
                            batch_size=population_batch_size,
                            noise_dim=int(self.architecture.noise_dim),
                            device=self.device,
                            dtype=self.dtype,
                            seed=derive_stream_seed(
                                base_seed=nested_seed,
                                stream_offset=(
                                    900_000
                                    + outer_index
                                    * int(
                                        nested_config.outer_backward_replicates
                                    )
                                    + replicate
                                ),
                            ),
                        )
                        with torch.no_grad():
                            replicate_population_paths = self.rollout(
                                x0=population_x0,
                                noise=population_noise,
                            ).paths.detach()
                    (
                        replicate_target,
                        replicate_noise_target,
                        replicate_metadata,
                    ) = self._nested_branch_conditional_targets(
                        x0=replicate_x0,
                        base_noise=replicate_noise,
                        base_paths=replicate_rollout.paths.detach(),
                        population_paths=replicate_population_paths,
                        target_path=replicate_target_batch,
                        training=training_config,
                        num_branches=int(
                            nested_config.outer_conditional_branches
                        ),
                        antithetic=bool(
                            nested_config.outer_conditional_antithetic
                        ),
                        query_batch_size=int(
                            nested_config.outer_conditional_query_batch_size
                        ),
                        seed=derive_stream_seed(
                            base_seed=nested_seed,
                            stream_offset=(
                                1_500_000
                                + outer_index
                                * int(nested_config.outer_backward_replicates)
                                + replicate
                            ),
                        ),
                    )
                    regression_noise_targets.append(
                        replicate_noise_target.detach()
                    )
                elif use_stein_target:
                    replicate_noise = replicate_noise.detach().requires_grad_(True)
                    replicate_rollout = self.rollout(
                        x0=replicate_x0, noise=replicate_noise
                    )
                    if replicate_population_paths is None:
                        replicate_target, replicate_metadata = (
                            self._path_functional_adjoint_targets(
                                generated_path=replicate_rollout.paths,
                                controls=replicate_rollout.controls,
                                target_path=replicate_target_batch,
                                training=training_config,
                                create_graph=True,
                                preserve_generated_path_graph=True,
                            )
                        )
                    else:
                        replicate_target, replicate_metadata = (
                            self._frozen_law_path_functional_adjoint_targets(
                                query_path=replicate_rollout.paths,
                                population_path=replicate_population_paths,
                                controls=replicate_rollout.controls,
                                target_path=replicate_target_batch,
                                training=training_config,
                                create_graph=True,
                                preserve_query_path_graph=True,
                            )
                        )
                    stein_adjoint_noise_target = self._stein_adjoint_noise_target(
                        differentiable_adjoint_target=replicate_target,
                        noise=replicate_noise,
                        moments_noise_target_dim=int(
                            self._default_noise_adjoint_dim(self.architecture)
                        ),
                        num_probes=int(training_config.stein_num_probes),
                        seed=derive_stream_seed(
                            base_seed=nested_seed,
                            stream_offset=1_200_000 + outer_index,
                        ),
                    )
                else:
                    with torch.no_grad():
                        replicate_rollout = self.rollout(
                            x0=replicate_x0, noise=replicate_noise
                        )
                    if replicate_population_paths is None:
                        replicate_target, replicate_metadata = (
                            self._path_functional_adjoint_targets(
                                generated_path=replicate_rollout.paths,
                                controls=replicate_rollout.controls,
                                target_path=replicate_target_batch,
                                training=training_config,
                            )
                        )
                    else:
                        replicate_target, replicate_metadata = (
                            self._frozen_law_path_functional_adjoint_targets(
                                query_path=replicate_rollout.paths,
                                population_path=replicate_population_paths,
                                controls=replicate_rollout.controls,
                                target_path=replicate_target_batch,
                                training=training_config,
                            )
                        )
                if replicate == 0:
                    rollout = replicate_rollout
                    outer_noise = replicate_noise
                    x0 = replicate_x0
                    target_batch = replicate_target_batch
                regression_paths.append(replicate_rollout.paths.detach())
                regression_noises.append(replicate_noise.detach())
                regression_targets.append(replicate_target.detach())
                replicate_metadata_rows.append(replicate_metadata)
                line_x0 = replicate_x0.detach()
                line_noise = replicate_noise.detach()
                line_rollout = replicate_rollout
                if (
                    nested_config.outer_line_search_batch_size is not None
                    and int(nested_config.outer_line_search_batch_size)
                    != int(nested_config.outer_batch_size)
                ):
                    line_source_indices = self._sample_indices(
                        size=target_count,
                        count=int(nested_config.outer_line_search_batch_size),
                        generator=cpu_generator,
                    )
                    line_x0 = target_paths_device.index_select(
                        0,
                        line_source_indices.to(self.device),
                    )[:, 0, :]
                    line_noise = sample_standard_normals(
                        grid=self.grid,
                        batch_size=int(
                            nested_config.outer_line_search_batch_size
                        ),
                        noise_dim=int(self.architecture.noise_dim),
                        device=self.device,
                        dtype=self.dtype,
                        seed=derive_stream_seed(
                            base_seed=nested_seed,
                            stream_offset=(
                                1_700_000
                                + outer_index
                                * int(nested_config.outer_backward_replicates)
                                + replicate
                            ),
                        ),
                    )
                    with torch.no_grad():
                        line_rollout = self.rollout(
                            x0=line_x0,
                            noise=line_noise,
                        )
                if (
                    use_nested_conditional_target
                    or "complete_objective_value" not in replicate_metadata
                ):
                    with torch.no_grad():
                        _, line_metrics = self._complete_objective_metrics(
                            generated_path=line_rollout.paths,
                            controls=line_rollout.controls,
                            target_path=replicate_target_batch,
                            training=training_config,
                        )
                    line_objective_rows.append(line_metrics)
                line_search_specs.append(
                    (
                        line_x0,
                        line_noise,
                        replicate_target_batch.detach(),
                    )
                )
            assert rollout is not None
            assert outer_noise is not None
            assert x0 is not None
            assert target_batch is not None
            common_metric_names = set.intersection(
                *(set(row) for row in replicate_metadata_rows)
            )
            target_metadata = {
                name: float(
                    sum(float(row[name]) for row in replicate_metadata_rows)
                    / len(replicate_metadata_rows)
                )
                for name in common_metric_names
            }
            target_metadata["outer_backward_replicates"] = float(
                nested_config.outer_backward_replicates
            )
            target_metadata["outer_backward_regression_paths"] = float(
                int(nested_config.outer_backward_replicates)
                * int(nested_config.outer_batch_size)
            )
            target_metadata["outer_target_estimator_nested_branches"] = float(
                use_nested_conditional_target
            )
            target_metadata["outer_line_search_batch_size"] = float(
                nested_config.outer_batch_size
                if nested_config.outer_line_search_batch_size is None
                else nested_config.outer_line_search_batch_size
            )
            if line_objective_rows:
                for objective_name in (
                    "complete_objective_value",
                    "discrepancy_objective_value",
                    "running_cost_value",
                ):
                    target_metadata[objective_name] = float(
                        sum(float(row[objective_name]) for row in line_objective_rows)
                        / len(line_objective_rows)
                    )
            frozen_paths = torch.cat(regression_paths, dim=0)
            regression_noise = torch.cat(regression_noises, dim=0)
            pathwise_adjoint_target = torch.cat(regression_targets, dim=0)
            with torch.no_grad():
                pre_inner_normalized_moments = self.network(frozen_paths)
                pre_inner_moments = self._denormalize_moments(pre_inner_normalized_moments)
            with torch.no_grad():
                projected_adjoint_target, ce_metadata = self._project_pathwise_target(
                    frozen_generated_path=frozen_paths,
                    pathwise_target=pathwise_adjoint_target.detach(),
                    training=training_config,
                )
            ce_metadata["ce_nested_branch_conditional_mean"] = float(
                use_nested_conditional_target
            )
            normalized_adjoint_target = self._normalize_adjoint_target(
                projected_adjoint_target,
                noise_adjoint=False,
            ).detach()
            self._check_same_shape(
                actual=pre_inner_normalized_moments.expected_adjoint_next,
                expected=normalized_adjoint_target,
                name="nested expected_adjoint_next",
            )

            projected_adjoint_noise_target: torch.Tensor | None = None
            normalized_adjoint_noise_target: torch.Tensor | None = None
            noise_ce_metadata: dict[str, float] = {}
            if float(training_config.adjoint_noise_weight) > 0.0:
                if use_nested_conditional_target:
                    if not regression_noise_targets:
                        raise RuntimeError(
                            "nested conditional R targets were not constructed"
                        )
                    pathwise_adjoint_noise_target = torch.cat(
                        regression_noise_targets,
                        dim=0,
                    ).detach()
                elif stein_adjoint_noise_target is not None:
                    pathwise_adjoint_noise_target = stein_adjoint_noise_target
                else:
                    noise_control_variate = self._noise_target_control_variate(
                        training=training_config,
                        raw_adjoint_prediction=pre_inner_moments.expected_adjoint_next,
                    )
                    pathwise_adjoint_noise_target = self._adjoint_noise_target(
                        pathwise_adjoint_target=pathwise_adjoint_target.detach(),
                        moments_noise_target_dim=int(
                            pre_inner_normalized_moments.expected_adjoint_noise_next.shape[-1]
                        ),
                        noise=regression_noise,
                        control_variate=noise_control_variate,
                    )
                with torch.no_grad():
                    projected_adjoint_noise_target, raw_noise_ce_metadata = (
                        self._project_pathwise_target(
                            frozen_generated_path=frozen_paths,
                            pathwise_target=pathwise_adjoint_noise_target,
                            training=training_config,
                            noise_adjoint=True,
                        )
                    )
                noise_ce_metadata = {
                    f"noise_{name}": float(value)
                    for name, value in raw_noise_ce_metadata.items()
                }
                noise_ce_metadata[
                    "noise_ce_nested_branch_conditional_mean"
                ] = float(use_nested_conditional_target)
                normalized_adjoint_noise_target = self._normalize_adjoint_target(
                    projected_adjoint_noise_target,
                    noise_adjoint=True,
                ).detach()
                self._check_same_shape(
                    actual=pre_inner_normalized_moments.expected_adjoint_noise_next,
                    expected=normalized_adjoint_noise_target,
                    name="nested expected_adjoint_noise_next",
                )

            def fixed_losses(
                normalized_moments: AdjointMoments,
                indices: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                adjoint_target_used = (
                    normalized_adjoint_target
                    if indices is None
                    else normalized_adjoint_target.index_select(0, indices)
                )
                adjoint_loss = torch.mean(
                    (
                        normalized_moments.expected_adjoint_next
                        - adjoint_target_used
                    ).pow(2)
                )
                noise_loss = adjoint_loss.new_tensor(0.0)
                if normalized_adjoint_noise_target is not None:
                    noise_target_used = (
                        normalized_adjoint_noise_target
                        if indices is None
                        else normalized_adjoint_noise_target.index_select(0, indices)
                    )
                    noise_loss = torch.mean(
                        (
                            normalized_moments.expected_adjoint_noise_next
                            - noise_target_used
                        ).pow(2)
                    )
                total = (
                    float(training_config.adjoint_weight) * adjoint_loss
                    + float(training_config.adjoint_noise_weight) * noise_loss
                )
                return total, adjoint_loss, noise_loss

            with torch.no_grad():
                start_total, start_adjoint, start_noise = fixed_losses(
                    pre_inner_normalized_moments
                )
            inner_grad_norms: list[float] = []
            inner_update_norms: list[float] = []
            inner_clip_active: list[float] = []
            inner_losses: list[float] = []
            if str(nested_config.inner_solver) == "timewise_linear_lstsq":
                parameter_values_before = [
                    parameter.detach().clone() for parameter in parameters
                ]
                self._fit_timewise_linear_residual_lstsq(
                    frozen_paths=frozen_paths,
                    normalized_adjoint_target=normalized_adjoint_target,
                    normalized_adjoint_noise_target=(
                        normalized_adjoint_noise_target
                    ),
                )
                with torch.no_grad():
                    solved_moments = self.network(frozen_paths)
                    solved_total, _, _ = fixed_losses(solved_moments)
                inner_grad_norms.append(0.0)
                inner_update_norms.append(
                    self._update_norm(parameters, parameter_values_before)
                )
                inner_clip_active.append(0.0)
                inner_losses.append(float(solved_total.item()))
                global_inner_step += 1
            else:
                for _ in range(int(nested_config.inner_steps)):
                    inner_indices_cpu = self._sample_indices(
                        size=int(frozen_paths.shape[0]),
                        count=int(nested_config.inner_batch_size),
                        generator=cpu_generator,
                    )
                    inner_indices = inner_indices_cpu.to(self.device)
                    normalized_moments = self.network(
                        frozen_paths.index_select(0, inner_indices)
                    )
                    total_loss, _, _ = fixed_losses(normalized_moments, inner_indices)
                    optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                    parameter_values_before = [
                        parameter.detach().clone() for parameter in parameters
                    ]
                    grad_norm = self._grad_norm(parameters)
                    clip_active = float(
                        training_config.grad_clip_norm is not None
                        and grad_norm > float(training_config.grad_clip_norm)
                    )
                    if training_config.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            parameters,
                            float(training_config.grad_clip_norm),
                        )
                    optimizer.step()
                    inner_grad_norms.append(float(grad_norm))
                    inner_update_norms.append(
                        self._update_norm(parameters, parameter_values_before)
                    )
                    inner_clip_active.append(clip_active)
                    inner_losses.append(float(total_loss.detach().item()))
                    global_inner_step += 1

            with torch.no_grad():
                candidate_parameter_values = [
                    parameter.detach().clone() for parameter in parameters
                ]
                candidate_normalized_moments = self.network(frozen_paths)
                candidate_moments = self._denormalize_moments(
                    candidate_normalized_moments
                )
                candidate_total, candidate_adjoint, candidate_noise = fixed_losses(
                    candidate_normalized_moments
                )

                def evaluate_outer_banks() -> tuple[
                    RolloutResult,
                    float,
                    float,
                    float,
                    bool,
                ]:
                    first_rollout: RolloutResult | None = None
                    objective_values = []
                    discrepancy_values = []
                    running_values = []
                    all_finite = True
                    for bank_x0, bank_noise, bank_target in line_search_specs:
                        bank_rollout = self.rollout(
                            x0=bank_x0,
                            noise=bank_noise,
                        )
                        if first_rollout is None:
                            first_rollout = bank_rollout
                        bank_is_finite = bool(
                            torch.isfinite(bank_rollout.paths).all().item()
                            and torch.isfinite(bank_rollout.controls).all().item()
                        )
                        if not bank_is_finite:
                            all_finite = False
                            break
                        bank_objective, bank_metrics = (
                            self._complete_objective_metrics(
                                generated_path=bank_rollout.paths,
                                controls=bank_rollout.controls,
                                target_path=bank_target,
                                training=training_config,
                            )
                        )
                        objective_values.append(float(bank_objective.item()))
                        discrepancy_values.append(
                            float(bank_metrics["discrepancy_objective_value"])
                        )
                        running_values.append(
                            float(bank_metrics["running_cost_value"])
                        )
                    assert first_rollout is not None
                    if not all_finite:
                        return (
                            first_rollout,
                            float("inf"),
                            float("inf"),
                            float("inf"),
                            False,
                        )
                    return (
                        first_rollout,
                        float(sum(objective_values) / len(objective_values)),
                        float(sum(discrepancy_values) / len(discrepancy_values)),
                        float(sum(running_values) / len(running_values)),
                        True,
                    )

                objective_before = float(target_metadata["complete_objective_value"])
                discrepancy_before = float(
                    target_metadata["discrepancy_objective_value"]
                )
                running_before = float(target_metadata["running_cost_value"])
                objective_tolerance = float(
                    nested_config.outer_objective_tolerance
                ) * max(1.0, abs(objective_before))
                accepted_relaxation = 1.0
                accepted_backtracks = 0
                accepted_objective = objective_before
                candidate_objective = float("nan")
                accepted_discrepancy = discrepancy_before
                candidate_discrepancy = float("nan")
                accepted_running = running_before
                candidate_running = float("nan")
                accepted_rollout = rollout
                objective_acceptance = 1.0
                accepted_block_relaxations = {
                    "p": 1.0,
                    "r": 1.0,
                    "shared": 1.0,
                }
                selected_block_relaxations = dict(accepted_block_relaxations)
                block_search_evaluations = 0
                block_search_passes = 0

                if bool(nested_config.outer_objective_line_search) and str(
                    nested_config.outer_relaxation_mode
                ) == "adjoint_blocks":
                    (
                        candidate_rollout,
                        candidate_objective,
                        candidate_discrepancy,
                        candidate_running,
                        candidate_is_finite,
                    ) = evaluate_outer_banks()
                    block_search_evaluations += 1

                    def apply_block_relaxations(
                        relaxations: dict[str, float],
                    ) -> None:
                        for (
                            parameter,
                            old_value,
                            candidate_value,
                            block,
                        ) in zip(
                            parameters,
                            outer_parameter_values,
                            candidate_parameter_values,
                            outer_parameter_blocks,
                        ):
                            parameter.copy_(
                                old_value
                                + float(relaxations[block])
                                * (candidate_value - old_value)
                            )

                    rate_candidates = sorted(
                        {
                            0.0,
                            *(
                                float(nested_config.outer_backtrack_factor)
                                ** backtrack_index
                                for backtrack_index in range(
                                    int(nested_config.outer_max_backtracks) + 1
                                )
                            ),
                        }
                    )
                    current_rates = {"p": 0.0, "r": 0.0, "shared": 0.0}
                    apply_block_relaxations(current_rates)
                    accepted_rollout = rollout
                    accepted_objective = objective_before
                    accepted_discrepancy = discrepancy_before
                    accepted_running = running_before
                    improvement_epsilon = max(
                        torch.finfo(torch.float64).eps,
                        1e-12 * max(1.0, abs(objective_before)),
                    )
                    for coordinate_pass in range(
                        int(nested_config.outer_block_coordinate_passes)
                    ):
                        pass_changed = False
                        for block in ("r", "p", "shared"):
                            coordinate_rate = float(current_rates[block])
                            coordinate_objective = accepted_objective
                            coordinate_discrepancy = accepted_discrepancy
                            coordinate_running = accepted_running
                            coordinate_rollout = accepted_rollout
                            for rate in rate_candidates:
                                if rate == float(current_rates[block]):
                                    continue
                                trial_rates = dict(current_rates)
                                trial_rates[block] = float(rate)
                                apply_block_relaxations(trial_rates)
                                (
                                    trial_rollout,
                                    trial_objective,
                                    trial_discrepancy,
                                    trial_running,
                                    trial_is_finite,
                                ) = evaluate_outer_banks()
                                block_search_evaluations += 1
                                if (
                                    math.isfinite(trial_objective)
                                    and trial_objective
                                    < coordinate_objective
                                    - improvement_epsilon
                                ):
                                    coordinate_rate = float(rate)
                                    coordinate_objective = trial_objective
                                    coordinate_discrepancy = trial_discrepancy
                                    coordinate_running = trial_running
                                    coordinate_rollout = trial_rollout
                            if coordinate_rate != float(current_rates[block]):
                                pass_changed = True
                            current_rates[block] = coordinate_rate
                            apply_block_relaxations(current_rates)
                            accepted_objective = coordinate_objective
                            accepted_discrepancy = coordinate_discrepancy
                            accepted_running = coordinate_running
                            accepted_rollout = coordinate_rollout
                        block_search_passes = coordinate_pass + 1
                        if not pass_changed:
                            break
                    apply_block_relaxations(current_rates)
                    selected_block_relaxations = dict(current_rates)
                    trust_fraction = float(
                        nested_config.outer_block_trust_fraction
                    )
                    trusted_rates = {
                        block: trust_fraction * float(rate)
                        for block, rate in current_rates.items()
                    }
                    if trusted_rates != current_rates:
                        apply_block_relaxations(trusted_rates)
                        (
                            trusted_rollout,
                            trusted_objective,
                            trusted_discrepancy,
                            trusted_running,
                            trusted_is_finite,
                        ) = evaluate_outer_banks()
                        if trusted_is_finite:
                            accepted_objective = trusted_objective
                            accepted_discrepancy = trusted_discrepancy
                            accepted_running = trusted_running
                            accepted_rollout = trusted_rollout
                        else:
                            accepted_objective = float("inf")
                            accepted_discrepancy = float("inf")
                            accepted_running = float("inf")
                        block_search_evaluations += 1
                    accepted_block_relaxations = dict(trusted_rates)
                    objective_acceptance = float(
                        any(value > 0.0 for value in trusted_rates.values())
                        and math.isfinite(accepted_objective)
                        and accepted_objective
                        <= objective_before + objective_tolerance
                    )
                    if objective_acceptance == 0.0:
                        current_rates = {"p": 0.0, "r": 0.0, "shared": 0.0}
                        accepted_block_relaxations = dict(current_rates)
                        apply_block_relaxations(current_rates)
                        accepted_rollout = rollout
                        accepted_objective = objective_before
                        accepted_discrepancy = discrepancy_before
                        accepted_running = running_before
                    accepted_relaxation = min(
                        accepted_block_relaxations.values()
                    )
                    accepted_backtracks = 0
                elif bool(nested_config.outer_objective_line_search):
                    objective_acceptance = 0.0
                    for backtrack_index in range(
                        int(nested_config.outer_max_backtracks) + 1
                    ):
                        relaxation = float(
                            nested_config.outer_backtrack_factor
                        ) ** backtrack_index
                        for parameter, old_value, candidate_value in zip(
                            parameters,
                            outer_parameter_values,
                            candidate_parameter_values,
                        ):
                            parameter.copy_(
                                old_value + relaxation * (candidate_value - old_value)
                            )
                        (
                            trial_rollout,
                            trial_objective,
                            trial_discrepancy,
                            trial_running,
                            trial_is_finite,
                        ) = evaluate_outer_banks()
                        if backtrack_index == 0:
                            candidate_objective = trial_objective
                            if trial_is_finite:
                                candidate_discrepancy = trial_discrepancy
                                candidate_running = trial_running
                        if math.isfinite(trial_objective) and trial_objective <= (
                            objective_before + objective_tolerance
                        ):
                            accepted_relaxation = relaxation
                            accepted_backtracks = backtrack_index
                            accepted_objective = trial_objective
                            accepted_discrepancy = trial_discrepancy
                            accepted_running = trial_running
                            accepted_rollout = trial_rollout
                            objective_acceptance = 1.0
                            break
                    if objective_acceptance == 0.0:
                        accepted_relaxation = 0.0
                        accepted_backtracks = int(
                            nested_config.outer_max_backtracks
                        ) + 1
                        for parameter, old_value in zip(
                            parameters,
                            outer_parameter_values,
                        ):
                            parameter.copy_(old_value)
                        accepted_rollout = rollout
                        accepted_objective = objective_before
                else:
                    (
                        accepted_rollout,
                        candidate_objective,
                        candidate_discrepancy,
                        candidate_running,
                        _,
                    ) = evaluate_outer_banks()
                    accepted_objective = candidate_objective
                    accepted_discrepancy = candidate_discrepancy
                    accepted_running = candidate_running

                if not (
                    bool(nested_config.outer_objective_line_search)
                    and str(nested_config.outer_relaxation_mode)
                    == "adjoint_blocks"
                ):
                    accepted_block_relaxations = {
                        "p": float(accepted_relaxation),
                        "r": float(accepted_relaxation),
                        "shared": float(accepted_relaxation),
                    }
                    selected_block_relaxations = dict(
                        accepted_block_relaxations
                    )
                accepted_normalized_moments = self.network(frozen_paths)
                accepted_moments = self._denormalize_moments(
                    accepted_normalized_moments
                )
                accepted_total, accepted_adjoint, accepted_noise = fixed_losses(
                    accepted_normalized_moments
                )
                current_probe = self.rollout(x0=probe_x0, noise=probe_noise)

            tensor_grad_norms = torch.tensor(inner_grad_norms, dtype=torch.float64)
            tensor_update_norms = torch.tensor(inner_update_norms, dtype=torch.float64)
            tensor_inner_losses = torch.tensor(inner_losses, dtype=torch.float64)
            metrics = dict(target_metadata)
            metrics.update(ce_metadata)
            metrics.update(noise_ce_metadata)
            metrics.update(self._target_preconditioner_metrics)
            metrics.update(
                self._moment_diagnostics(
                    accepted_moments.expected_adjoint_next,
                    projected_adjoint_target,
                    prefix="fixed_adjoint",
                )
            )
            metrics.update(
                self._moment_diagnostics(
                    candidate_moments.expected_adjoint_next,
                    projected_adjoint_target,
                    prefix="candidate_fixed_adjoint",
                )
            )
            if projected_adjoint_noise_target is not None:
                metrics.update(
                    self._moment_diagnostics(
                        accepted_moments.expected_adjoint_noise_next,
                        projected_adjoint_noise_target,
                        prefix="fixed_noise_adjoint",
                    )
                )
                metrics.update(
                    self._moment_diagnostics(
                        candidate_moments.expected_adjoint_noise_next,
                        projected_adjoint_noise_target,
                        prefix="candidate_fixed_noise_adjoint",
                    )
                )
            metrics.update(self._control_diagnostics(current_probe.controls))
            metrics.update(
                {
                    "outer_step": float(outer_index + 1),
                    "step": float(global_inner_step),
                    "nested_inner_steps": float(nested_config.inner_steps),
                    "nested_inner_solver_timewise_linear_lstsq": float(
                        str(nested_config.inner_solver)
                        == "timewise_linear_lstsq"
                    ),
                    "nested_outer_batch_size": float(nested_config.outer_batch_size),
                    "nested_outer_backward_replicates": float(
                        nested_config.outer_backward_replicates
                    ),
                    "nested_outer_target_estimator_nested_branches": float(
                        str(nested_config.outer_target_estimator)
                        == "nested_branches"
                    ),
                    "nested_outer_conditional_branches": float(
                        nested_config.outer_conditional_branches
                    ),
                    "nested_outer_conditional_antithetic": float(
                        nested_config.outer_conditional_antithetic
                    ),
                    "nested_outer_line_search_batch_size": float(
                        nested_config.outer_batch_size
                        if nested_config.outer_line_search_batch_size is None
                        else nested_config.outer_line_search_batch_size
                    ),
                    "nested_inner_batch_size": float(nested_config.inner_batch_size),
                    "inner_start_total_loss": float(start_total.item()),
                    "inner_start_adjoint_loss": float(start_adjoint.item()),
                    "inner_start_noise_adjoint_loss": float(start_noise.item()),
                    "inner_final_total_loss": float(candidate_total.item()),
                    "inner_final_adjoint_loss": float(candidate_adjoint.item()),
                    "inner_final_noise_adjoint_loss": float(candidate_noise.item()),
                    "inner_loss_ratio": float(
                        candidate_total.div(
                            start_total.clamp_min(torch.finfo(start_total.dtype).eps)
                        ).item()
                    ),
                    "accepted_fixed_total_loss": float(accepted_total.item()),
                    "accepted_fixed_adjoint_loss": float(accepted_adjoint.item()),
                    "accepted_fixed_noise_adjoint_loss": float(accepted_noise.item()),
                    "accepted_fixed_loss_ratio": float(
                        accepted_total.div(
                            start_total.clamp_min(torch.finfo(start_total.dtype).eps)
                        ).item()
                    ),
                    "outer_objective_before": objective_before,
                    "outer_candidate_objective": candidate_objective,
                    "outer_accepted_objective": accepted_objective,
                    "outer_objective_change": accepted_objective - objective_before,
                    "outer_discrepancy_objective_before": discrepancy_before,
                    "outer_candidate_discrepancy_objective": candidate_discrepancy,
                    "outer_accepted_discrepancy_objective": accepted_discrepancy,
                    "outer_running_cost_before": running_before,
                    "outer_candidate_running_cost": candidate_running,
                    "outer_accepted_running_cost": accepted_running,
                    "outer_relaxation": accepted_relaxation,
                    "outer_relaxation_p": float(
                        accepted_block_relaxations["p"]
                    ),
                    "outer_relaxation_r": float(
                        accepted_block_relaxations["r"]
                    ),
                    "outer_relaxation_shared": float(
                        accepted_block_relaxations["shared"]
                    ),
                    "outer_selected_relaxation_p": float(
                        selected_block_relaxations["p"]
                    ),
                    "outer_selected_relaxation_r": float(
                        selected_block_relaxations["r"]
                    ),
                    "outer_selected_relaxation_shared": float(
                        selected_block_relaxations["shared"]
                    ),
                    "outer_block_trust_fraction": float(
                        nested_config.outer_block_trust_fraction
                    ),
                    "outer_relaxation_mode_adjoint_blocks": float(
                        str(nested_config.outer_relaxation_mode)
                        == "adjoint_blocks"
                    ),
                    "outer_block_search_evaluations": float(
                        block_search_evaluations
                    ),
                    "outer_block_search_passes": float(block_search_passes),
                    "outer_backtracks": float(accepted_backtracks),
                    "outer_objective_accepted": objective_acceptance,
                    "noise_target_estimator_stein": float(use_stein_target),
                    "stein_num_probes": float(
                        training_config.stein_num_probes if use_stein_target else 0
                    ),
                    "inner_sampled_loss_mean": float(tensor_inner_losses.mean().item()),
                    "inner_sampled_loss_q95": float(
                        torch.quantile(tensor_inner_losses, 0.95).item()
                    ),
                    "inner_grad_norm_mean": float(tensor_grad_norms.mean().item()),
                    "inner_grad_norm_q95": float(
                        torch.quantile(tensor_grad_norms, 0.95).item()
                    ),
                    "inner_grad_norm_max": float(tensor_grad_norms.max().item()),
                    "inner_update_norm_mean": float(tensor_update_norms.mean().item()),
                    "inner_update_norm_q95": float(
                        torch.quantile(tensor_update_norms, 0.95).item()
                    ),
                    "inner_clip_fraction": float(
                        sum(inner_clip_active) / len(inner_clip_active)
                    ),
                    "parameter_norm": float(self._parameter_norm(parameters)),
                    "fixed_point_path_rms_change": float(
                        torch.sqrt(
                            torch.mean(
                                (current_probe.paths - previous_probe.paths).pow(2)
                            )
                        ).item()
                    ),
                    "fixed_point_control_rms_change": float(
                        torch.sqrt(
                            torch.mean(
                                (current_probe.controls - previous_probe.controls).pow(2)
                            )
                        ).item()
                    ),
                    "fixed_point_path_rms": float(
                        torch.sqrt(torch.mean(current_probe.paths.pow(2))).item()
                    ),
                    "fixed_point_control_rms": float(
                        torch.sqrt(torch.mean(current_probe.controls.pow(2))).item()
                    ),
                    "gradient_nonfinite_fraction": self._nonfinite_fraction(
                        [
                            parameter.grad
                            for parameter in parameters
                            if parameter.grad is not None
                        ]
                    ),
                    "parameter_nonfinite_fraction": self._nonfinite_fraction(parameters),
                }
            )
            history.append(metrics)
            final_metrics = metrics
            previous_probe = current_probe
            if (
                (outer_index + 1) % int(nested_config.log_every_outer) == 0
                or outer_index == int(nested_config.outer_steps) - 1
            ) and log_callback is not None:
                log_callback(self, dict(metrics))

        if final_metrics is None:
            raise RuntimeError("nested training produced no metrics")
        final_metrics.update(
            self._fresh_regression_probe(
                target_paths=target_paths_device,
                training=training_config,
            )
        )
        return FitResult(history=history, final_metrics=final_metrics)

    def _fresh_regression_probe(
        self,
        *,
        target_paths: torch.Tensor,
        training: DiscreteMPTrainingConfig,
        replay_bank: FitReplayBank | None = None,
        conditional_target_provider: Callable[
            ["DiscreteMPModel", RolloutResult, torch.Tensor],
            ConditionalAdjointTargets,
        ]
        | None = None,
    ) -> dict[str, float]:
        """Evaluate the fitted moment network on a deterministic fresh batch."""
        target_count, _ = self._validate_target_paths(target_paths)
        probe_seed = derive_stream_seed(base_seed=training.seed, stream_offset=90_000)
        cpu_generator = torch.Generator(device="cpu")
        if probe_seed is None:
            cpu_generator.seed()
        else:
            cpu_generator.manual_seed(int(probe_seed))
        if replay_bank is None:
            source_indices = self._sample_indices(
                size=target_count,
                count=int(training.batch_size),
                generator=cpu_generator,
            )
            target_indices = self._sample_indices(
                size=target_count,
                count=int(training.target_batch_size),
                generator=cpu_generator,
            )
            noise = sample_standard_normals(
                grid=self.grid,
                batch_size=int(training.batch_size),
                noise_dim=int(self.architecture.noise_dim),
                device=self.device,
                dtype=self.dtype,
                seed=derive_stream_seed(base_seed=probe_seed, stream_offset=1),
            )
        else:
            source_indices = replay_bank.source_indices
            target_indices = replay_bank.target_indices
            noise = replay_bank.noise.detach().clone()
        x0 = target_paths.index_select(0, source_indices.to(target_paths.device))[:, 0, :]
        target_batch = target_paths.index_select(0, target_indices.to(target_paths.device))
        use_stein_target = bool(
            conditional_target_provider is None
            and str(training.noise_target_estimator) == "stein"
            and float(training.adjoint_noise_weight) > 0.0
        )
        provided_noise_target = None
        if conditional_target_provider is not None:
            with torch.no_grad():
                rollout = self.rollout(x0=x0, noise=noise)
            supplied = self._provided_conditional_adjoint_targets(
                provider=conditional_target_provider,
                rollout=rollout,
                target_paths=target_batch,
            )
            pathwise_target = supplied.expected_adjoint_next
            provided_noise_target = supplied.expected_adjoint_noise_next
            target_metadata = dict(supplied.metrics)
            stein_pathwise_noise_target = None
        elif use_stein_target:
            noise = noise.detach().requires_grad_(True)
            rollout = self.rollout(x0=x0, noise=noise)
            pathwise_target, target_metadata = self._path_functional_adjoint_targets(
                generated_path=rollout.paths,
                controls=rollout.controls,
                target_path=target_batch,
                training=training,
                create_graph=True,
                preserve_generated_path_graph=True,
            )
            stein_pathwise_noise_target = self._stein_adjoint_noise_target(
                differentiable_adjoint_target=pathwise_target,
                noise=noise,
                moments_noise_target_dim=int(
                    self._default_noise_adjoint_dim(self.architecture)
                ),
                num_probes=int(training.stein_num_probes),
                seed=derive_stream_seed(
                    base_seed=probe_seed,
                    stream_offset=1_300_000,
                ),
            )
        else:
            stein_pathwise_noise_target = None
            with torch.no_grad():
                rollout = self.rollout(x0=x0, noise=noise)
            pathwise_target, target_metadata = self._path_functional_adjoint_targets(
                generated_path=rollout.paths,
                controls=rollout.controls,
                target_path=target_batch,
                training=training,
            )
        frozen_path = rollout.paths.detach()
        with torch.no_grad():
            normalized_moments = self.network(frozen_path)
            moments = self._denormalize_moments(normalized_moments)
            if conditional_target_provider is None:
                training_target, ce_metadata = self._project_pathwise_target(
                    frozen_generated_path=frozen_path,
                    pathwise_target=pathwise_target.detach(),
                    training=training,
                )
            else:
                training_target = pathwise_target.detach()
                ce_metadata = {"ce_oracle_conditional_targets": 1.0}
            normalized_training_target = self._normalize_adjoint_target(
                training_target,
                noise_adjoint=False,
            )
        metrics = {
            "probe_seed": float(probe_seed) if probe_seed is not None else -1.0,
            "probe_target_mode_direct": float(str(training.ce_target_mode) == "direct"),
            "probe_fit_replay_bank": float(replay_bank is not None),
            "probe_conditional_target_provider": float(
                conditional_target_provider is not None
            ),
            "probe_adjoint_r2": self._prediction_r2(moments.expected_adjoint_next, training_target),
            "probe_normalized_adjoint_r2": self._prediction_r2(
                normalized_moments.expected_adjoint_next,
                normalized_training_target,
            ),
        }
        metrics.update(
            {
                f"probe_{key}": float(value)
                for key, value in target_metadata.items()
                if key
                in {
                    "discrepancy_source_rms",
                    "running_discrepancy_source_cosine",
                    "running_source_rms",
                    "running_to_discrepancy_source_rms_ratio",
                    "source_rms",
                    "target_rms",
                }
            }
        )
        metrics.update(
            {
                f"probe_{key}": float(value)
                for key, value in ce_metadata.items()
            }
        )
        metrics.update(
            self._moment_diagnostics(
                moments.expected_adjoint_next,
                training_target,
                prefix="probe_adjoint",
            )
        )
        metrics.update(
            {
                f"probe_{key}": value
                for key, value in self._control_diagnostics(rollout.controls).items()
            }
        )

        if float(training.adjoint_noise_weight) > 0.0:
            if provided_noise_target is not None:
                pathwise_noise_target = provided_noise_target.detach()
            elif stein_pathwise_noise_target is not None:
                pathwise_noise_target = stein_pathwise_noise_target
                metrics["probe_noise_target_estimator_stein"] = 1.0
                metrics["probe_stein_num_probes"] = float(
                    training.stein_num_probes
                )
            else:
                uncentered_pathwise_noise_target = self._adjoint_noise_target(
                    pathwise_adjoint_target=pathwise_target.detach(),
                    moments_noise_target_dim=int(moments.expected_adjoint_noise_next.shape[-1]),
                    noise=noise,
                )
                probe_control_variate = self._noise_target_control_variate(
                    training=training,
                    raw_adjoint_prediction=moments.expected_adjoint_next,
                )
                if probe_control_variate is None:
                    pathwise_noise_target = uncentered_pathwise_noise_target
                else:
                    pathwise_noise_target = self._adjoint_noise_target(
                        pathwise_adjoint_target=pathwise_target.detach(),
                        moments_noise_target_dim=int(moments.expected_adjoint_noise_next.shape[-1]),
                        noise=noise,
                        control_variate=probe_control_variate,
                    )
                uncentered_second_moment = torch.mean(
                    uncentered_pathwise_noise_target.pow(2)
                ).clamp_min(torch.finfo(uncentered_pathwise_noise_target.dtype).eps)
                metrics["probe_noise_target_control_variate_second_moment_ratio"] = float(
                    (torch.mean(pathwise_noise_target.pow(2)) / uncentered_second_moment).item()
                )
            with torch.no_grad():
                if conditional_target_provider is None:
                    noise_training_target, noise_ce_metadata = self._project_pathwise_target(
                        frozen_generated_path=frozen_path,
                        pathwise_target=pathwise_noise_target,
                        training=training,
                        noise_adjoint=True,
                    )
                else:
                    noise_training_target = pathwise_noise_target
                    noise_ce_metadata = {
                        "ce_oracle_conditional_targets": 1.0
                    }
                normalized_noise_training_target = self._normalize_adjoint_target(
                    noise_training_target,
                    noise_adjoint=True,
                )
            metrics["probe_noise_adjoint_r2"] = self._prediction_r2(
                moments.expected_adjoint_noise_next,
                noise_training_target,
            )
            metrics["probe_normalized_noise_adjoint_r2"] = self._prediction_r2(
                normalized_moments.expected_adjoint_noise_next,
                normalized_noise_training_target,
            )
            metrics.update(
                {
                    f"probe_noise_{key}": float(value)
                    for key, value in noise_ce_metadata.items()
                }
            )
            metrics.update(
                self._moment_diagnostics(
                    moments.expected_adjoint_noise_next,
                    noise_training_target,
                    prefix="probe_noise_adjoint",
                )
            )
        return metrics

    def sample_training_noise(self, *, batch_size: int, step: int) -> torch.Tensor:
        batch_size = _require_int("batch_size", batch_size)
        step = _require_int("step", step)
        return sample_standard_normals(
            grid=self.grid,
            batch_size=batch_size,
            noise_dim=int(self.architecture.noise_dim),
            device=self.device,
            dtype=self.dtype,
            seed=derive_stream_seed(base_seed=self.training.seed, stream_offset=step + 1),
        )
