from __future__ import annotations

from dataclasses import dataclass
import math
import operator


def _require_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _require_optional_int(name: str, value: object | None) -> int | None:
    if value is None:
        return None
    return _require_int(name, value)


def _require_finite_float(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite float")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_optional_finite_float(name: str, value: object | None) -> float | None:
    if value is None:
        return None
    return _require_finite_float(name, value)


def _require_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


@dataclass(frozen=True)
class DiscreteMPArchitectureConfig:
    state_dim: int
    noise_dim: int
    hidden_dim: int = 96
    num_layers: int = 1
    adjoint_dim: int | None = None
    noise_adjoint_dim: int | None = None
    adjoint_input_mode: str = "level"

    def __post_init__(self) -> None:
        state_dim = _require_int("state_dim", self.state_dim)
        noise_dim = _require_int("noise_dim", self.noise_dim)
        hidden_dim = _require_int("hidden_dim", self.hidden_dim)
        num_layers = _require_int("num_layers", self.num_layers)
        adjoint_dim = _require_optional_int("adjoint_dim", self.adjoint_dim)
        noise_adjoint_dim = _require_optional_int("noise_adjoint_dim", self.noise_adjoint_dim)
        if not isinstance(self.adjoint_input_mode, str):
            raise ValueError("adjoint_input_mode must be level or increment")
        adjoint_input_mode = self.adjoint_input_mode.strip().lower()
        if state_dim < 1:
            raise ValueError("state_dim must be >= 1")
        if noise_dim < 1:
            raise ValueError("noise_dim must be >= 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if adjoint_dim is not None and adjoint_dim < 1:
            raise ValueError("adjoint_dim must be >= 1 when provided")
        if noise_adjoint_dim is not None and noise_adjoint_dim < 1:
            raise ValueError("noise_adjoint_dim must be >= 1 when provided")
        if adjoint_input_mode not in {"level", "increment"}:
            raise ValueError("adjoint_input_mode must be level or increment")
        object.__setattr__(self, "state_dim", state_dim)
        object.__setattr__(self, "noise_dim", noise_dim)
        object.__setattr__(self, "hidden_dim", hidden_dim)
        object.__setattr__(self, "num_layers", num_layers)
        object.__setattr__(self, "adjoint_dim", adjoint_dim)
        object.__setattr__(self, "noise_adjoint_dim", noise_adjoint_dim)
        object.__setattr__(self, "adjoint_input_mode", adjoint_input_mode)


@dataclass(frozen=True)
class DiscreteMPTrainingConfig:
    batch_size: int = 256
    target_batch_size: int = 256
    num_steps: int = 2000
    lr: float = 2e-3
    weight_decay: float = 1e-5
    lambda_scale: float = 50.0
    adjoint_weight: float = 1.0
    adjoint_noise_weight: float = 1.0
    ce_target_mode: str = "ridge"
    ridge_lambda: float = 1e-3
    ce_crossfit_folds: int = 1
    target_preconditioner: str = "none"
    preconditioner_batches: int = 8
    preconditioner_min_scale: float = 1e-3
    noise_target_control_variate: str = "none"
    noise_target_estimator: str = "score"
    stein_num_probes: int = 8
    grad_clip_norm: float | None = 5.0
    log_every: int = 100
    seed: int | None = 1234
    observed_only: bool = True
    path_derivative_backend: str = "autograd"
    drift_adjoint_backend: str = "none"

    def __post_init__(self) -> None:
        batch_size = _require_int("batch_size", self.batch_size)
        target_batch_size = _require_int("target_batch_size", self.target_batch_size)
        num_steps = _require_int("num_steps", self.num_steps)
        log_every = _require_int("log_every", self.log_every)
        seed = _require_optional_int("seed", self.seed)
        lr = _require_finite_float("lr", self.lr)
        weight_decay = _require_finite_float("weight_decay", self.weight_decay)
        lambda_scale = _require_finite_float("lambda_scale", self.lambda_scale)
        adjoint_weight = _require_finite_float("adjoint_weight", self.adjoint_weight)
        adjoint_noise_weight = _require_finite_float("adjoint_noise_weight", self.adjoint_noise_weight)
        if not isinstance(self.ce_target_mode, str):
            raise ValueError("ce_target_mode must be direct or ridge")
        ce_target_mode = self.ce_target_mode.strip().lower()
        ridge_lambda = _require_finite_float("ridge_lambda", self.ridge_lambda)
        ce_crossfit_folds = _require_int("ce_crossfit_folds", self.ce_crossfit_folds)
        if not isinstance(self.target_preconditioner, str):
            raise ValueError("target_preconditioner must be none or timewise")
        target_preconditioner = self.target_preconditioner.strip().lower()
        preconditioner_batches = _require_int("preconditioner_batches", self.preconditioner_batches)
        preconditioner_min_scale = _require_finite_float(
            "preconditioner_min_scale", self.preconditioner_min_scale
        )
        if not isinstance(self.noise_target_control_variate, str):
            raise ValueError("noise_target_control_variate must be none, timewise, or adjoint")
        noise_target_control_variate = self.noise_target_control_variate.strip().lower()
        if not isinstance(self.noise_target_estimator, str):
            raise ValueError("noise_target_estimator must be score or stein")
        noise_target_estimator = self.noise_target_estimator.strip().lower()
        stein_num_probes = _require_int("stein_num_probes", self.stein_num_probes)
        grad_clip_norm = _require_optional_finite_float("grad_clip_norm", self.grad_clip_norm)
        observed_only = _require_bool("observed_only", self.observed_only)
        if not isinstance(self.path_derivative_backend, str):
            raise ValueError("path_derivative_backend must be autograd or analytical")
        path_derivative_backend = self.path_derivative_backend.strip().lower()
        if not isinstance(self.drift_adjoint_backend, str):
            raise ValueError(
                "drift_adjoint_backend must be none, autograd_replay, or analytical_reference"
            )
        drift_adjoint_backend = self.drift_adjoint_backend.strip().lower()
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if target_batch_size < 1:
            raise ValueError("target_batch_size must be >= 1")
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if lr <= 0.0:
            raise ValueError("lr must be > 0")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be >= 0")
        if lambda_scale < 0.0:
            raise ValueError("lambda_scale must be >= 0")
        if adjoint_weight < 0.0:
            raise ValueError("adjoint_weight must be >= 0")
        if adjoint_noise_weight < 0.0:
            raise ValueError("adjoint_noise_weight must be >= 0")
        if not (adjoint_weight > 0.0 or adjoint_noise_weight > 0.0):
            raise ValueError("at least one adjoint consistency weight must be > 0")
        if ce_target_mode not in {"direct", "ridge"}:
            raise ValueError("ce_target_mode must be direct or ridge")
        if ridge_lambda < 0.0:
            raise ValueError("ridge_lambda must be >= 0")
        if ce_crossfit_folds < 1:
            raise ValueError("ce_crossfit_folds must be >= 1")
        if ce_target_mode == "direct" and ce_crossfit_folds != 1:
            raise ValueError("direct ce_target_mode requires ce_crossfit_folds=1")
        if target_preconditioner not in {"none", "timewise"}:
            raise ValueError("target_preconditioner must be none or timewise")
        if target_preconditioner == "timewise" and ce_target_mode != "direct":
            raise ValueError("timewise target_preconditioner requires ce_target_mode='direct'")
        if preconditioner_batches < 1:
            raise ValueError("preconditioner_batches must be >= 1")
        if preconditioner_min_scale <= 0.0:
            raise ValueError("preconditioner_min_scale must be > 0")
        if noise_target_control_variate not in {"none", "timewise", "adjoint"}:
            raise ValueError("noise_target_control_variate must be none, timewise, or adjoint")
        if noise_target_estimator not in {"score", "stein"}:
            raise ValueError("noise_target_estimator must be score or stein")
        if stein_num_probes < 1:
            raise ValueError("stein_num_probes must be >= 1")
        if noise_target_estimator == "stein" and noise_target_control_variate != "none":
            raise ValueError(
                "the Stein noise target already removes the score-product baseline; "
                "noise_target_control_variate must be none"
            )
        if grad_clip_norm is not None and grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be > 0 when provided")
        if log_every < 1:
            raise ValueError("log_every must be >= 1")
        if path_derivative_backend not in {"autograd", "analytical"}:
            raise ValueError("path_derivative_backend must be autograd or analytical")
        if drift_adjoint_backend not in {
            "none",
            "autograd_replay",
            "analytical_reference",
        }:
            raise ValueError(
                "drift_adjoint_backend must be none, autograd_replay, or analytical_reference"
            )
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "target_batch_size", target_batch_size)
        object.__setattr__(self, "num_steps", num_steps)
        object.__setattr__(self, "lr", lr)
        object.__setattr__(self, "weight_decay", weight_decay)
        object.__setattr__(self, "lambda_scale", lambda_scale)
        object.__setattr__(self, "adjoint_weight", adjoint_weight)
        object.__setattr__(self, "adjoint_noise_weight", adjoint_noise_weight)
        object.__setattr__(self, "ce_target_mode", ce_target_mode)
        object.__setattr__(self, "ridge_lambda", ridge_lambda)
        object.__setattr__(self, "ce_crossfit_folds", ce_crossfit_folds)
        object.__setattr__(self, "target_preconditioner", target_preconditioner)
        object.__setattr__(self, "preconditioner_batches", preconditioner_batches)
        object.__setattr__(self, "preconditioner_min_scale", preconditioner_min_scale)
        object.__setattr__(self, "noise_target_control_variate", noise_target_control_variate)
        object.__setattr__(self, "noise_target_estimator", noise_target_estimator)
        object.__setattr__(self, "stein_num_probes", stein_num_probes)
        object.__setattr__(self, "grad_clip_norm", grad_clip_norm)
        object.__setattr__(self, "log_every", log_every)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "observed_only", observed_only)
        object.__setattr__(self, "path_derivative_backend", path_derivative_backend)
        object.__setattr__(self, "drift_adjoint_backend", drift_adjoint_backend)


@dataclass(frozen=True)
class DiscreteMPNestedTrainingConfig:
    """Two-timescale solver configuration for the coupled forward/backward law.

    One outer iteration freezes a generated path bank and its complete-MP
    conditional targets.  ``inner_steps`` optimizer updates then fit the
    unchanged adjoint-regression loss before the forward law is refreshed.
    ``outer_relaxation_mode='adjoint_blocks'`` applies the same fitted
    candidate with separately selected outer rates for the P head, R head,
    and shared recurrent parameters. ``outer_block_trust_fraction`` can then
    shrink all three selected rates by a fixed amount before the final
    objective safety check. This is numerical fixed-point relaxation only; it
    does not alter the regression loss or MP equations.
    """

    outer_steps: int = 10
    inner_steps: int = 200
    outer_batch_size: int = 1024
    outer_target_batch_size: int = 1024
    outer_backward_replicates: int = 1
    outer_target_estimator: str = "projected_score"
    outer_conditional_branches: int = 16
    outer_conditional_antithetic: bool = True
    outer_conditional_query_batch_size: int = 16384
    outer_population_batch_size: int | None = None
    outer_line_search_batch_size: int | None = None
    inner_batch_size: int = 256
    fixed_point_probe_paths: int = 512
    log_every_outer: int = 1
    outer_objective_line_search: bool = True
    outer_relaxation_mode: str = "joint"
    outer_max_backtracks: int = 10
    outer_backtrack_factor: float = 0.5
    outer_objective_tolerance: float = 0.0
    outer_block_coordinate_passes: int = 2
    outer_block_trust_fraction: float = 1.0
    inner_solver: str = "adam"
    seed: int | None = 1234

    def __post_init__(self) -> None:
        outer_steps = _require_int("outer_steps", self.outer_steps)
        inner_steps = _require_int("inner_steps", self.inner_steps)
        outer_batch_size = _require_int("outer_batch_size", self.outer_batch_size)
        outer_target_batch_size = _require_int(
            "outer_target_batch_size", self.outer_target_batch_size
        )
        outer_backward_replicates = _require_int(
            "outer_backward_replicates", self.outer_backward_replicates
        )
        if not isinstance(self.outer_target_estimator, str):
            raise ValueError(
                "outer_target_estimator must be projected_score or nested_branches"
            )
        outer_target_estimator = self.outer_target_estimator.strip().lower()
        outer_conditional_branches = _require_int(
            "outer_conditional_branches", self.outer_conditional_branches
        )
        outer_conditional_antithetic = _require_bool(
            "outer_conditional_antithetic", self.outer_conditional_antithetic
        )
        outer_conditional_query_batch_size = _require_int(
            "outer_conditional_query_batch_size",
            self.outer_conditional_query_batch_size,
        )
        outer_population_batch_size = (
            None
            if self.outer_population_batch_size is None
            else _require_int(
                "outer_population_batch_size",
                self.outer_population_batch_size,
            )
        )
        outer_line_search_batch_size = (
            None
            if self.outer_line_search_batch_size is None
            else _require_int(
                "outer_line_search_batch_size",
                self.outer_line_search_batch_size,
            )
        )
        inner_batch_size = _require_int("inner_batch_size", self.inner_batch_size)
        fixed_point_probe_paths = _require_int(
            "fixed_point_probe_paths", self.fixed_point_probe_paths
        )
        log_every_outer = _require_int("log_every_outer", self.log_every_outer)
        outer_objective_line_search = _require_bool(
            "outer_objective_line_search", self.outer_objective_line_search
        )
        if not isinstance(self.outer_relaxation_mode, str):
            raise ValueError("outer_relaxation_mode must be joint or adjoint_blocks")
        outer_relaxation_mode = self.outer_relaxation_mode.strip().lower()
        outer_max_backtracks = _require_int(
            "outer_max_backtracks", self.outer_max_backtracks
        )
        outer_backtrack_factor = _require_finite_float(
            "outer_backtrack_factor", self.outer_backtrack_factor
        )
        outer_objective_tolerance = _require_finite_float(
            "outer_objective_tolerance", self.outer_objective_tolerance
        )
        outer_block_coordinate_passes = _require_int(
            "outer_block_coordinate_passes", self.outer_block_coordinate_passes
        )
        outer_block_trust_fraction = _require_finite_float(
            "outer_block_trust_fraction", self.outer_block_trust_fraction
        )
        if not isinstance(self.inner_solver, str):
            raise ValueError(
                "inner_solver must be adam or timewise_linear_lstsq"
            )
        inner_solver = self.inner_solver.strip().lower()
        seed = _require_optional_int("seed", self.seed)
        for name, value in (
            ("outer_steps", outer_steps),
            ("inner_steps", inner_steps),
            ("outer_batch_size", outer_batch_size),
            ("outer_target_batch_size", outer_target_batch_size),
            ("outer_backward_replicates", outer_backward_replicates),
            ("outer_conditional_branches", outer_conditional_branches),
            (
                "outer_conditional_query_batch_size",
                outer_conditional_query_batch_size,
            ),
            ("inner_batch_size", inner_batch_size),
            ("fixed_point_probe_paths", fixed_point_probe_paths),
            ("log_every_outer", log_every_outer),
            ("outer_block_coordinate_passes", outer_block_coordinate_passes),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
        if inner_batch_size > outer_batch_size:
            raise ValueError("inner_batch_size cannot exceed outer_batch_size")
        if outer_target_estimator not in {"projected_score", "nested_branches"}:
            raise ValueError(
                "outer_target_estimator must be projected_score or nested_branches"
            )
        if outer_target_estimator == "nested_branches" and outer_conditional_branches < 2:
            raise ValueError(
                "nested_branches requires outer_conditional_branches >= 2"
            )
        if (
            outer_population_batch_size is not None
            and outer_population_batch_size < 2
        ):
            raise ValueError("outer_population_batch_size must be >= 2")
        if (
            outer_target_estimator == "nested_branches"
            and outer_conditional_antithetic
            and outer_conditional_branches % 2 != 0
        ):
            raise ValueError(
                "antithetic nested branches require an even branch count"
            )
        if (
            outer_line_search_batch_size is not None
            and outer_line_search_batch_size < 2
        ):
            raise ValueError("outer_line_search_batch_size must be >= 2")
        if outer_relaxation_mode not in {"joint", "adjoint_blocks"}:
            raise ValueError("outer_relaxation_mode must be joint or adjoint_blocks")
        if outer_max_backtracks < 0:
            raise ValueError("outer_max_backtracks must be >= 0")
        if not 0.0 < outer_backtrack_factor < 1.0:
            raise ValueError("outer_backtrack_factor must lie in (0, 1)")
        if outer_objective_tolerance < 0.0:
            raise ValueError("outer_objective_tolerance must be >= 0")
        if not 0.0 < outer_block_trust_fraction <= 1.0:
            raise ValueError("outer_block_trust_fraction must lie in (0, 1]")
        if inner_solver not in {"adam", "timewise_linear_lstsq"}:
            raise ValueError(
                "inner_solver must be adam or timewise_linear_lstsq"
            )
        if outer_block_trust_fraction < 1.0 and (
            outer_relaxation_mode != "adjoint_blocks"
            or not outer_objective_line_search
        ):
            raise ValueError(
                "outer_block_trust_fraction < 1 requires an enabled "
                "adjoint_blocks outer line search"
            )
        object.__setattr__(self, "outer_steps", outer_steps)
        object.__setattr__(self, "inner_steps", inner_steps)
        object.__setattr__(self, "outer_batch_size", outer_batch_size)
        object.__setattr__(self, "outer_target_batch_size", outer_target_batch_size)
        object.__setattr__(
            self,
            "outer_backward_replicates",
            outer_backward_replicates,
        )
        object.__setattr__(self, "outer_target_estimator", outer_target_estimator)
        object.__setattr__(
            self,
            "outer_conditional_branches",
            outer_conditional_branches,
        )
        object.__setattr__(
            self,
            "outer_conditional_antithetic",
            outer_conditional_antithetic,
        )
        object.__setattr__(
            self,
            "outer_conditional_query_batch_size",
            outer_conditional_query_batch_size,
        )
        object.__setattr__(
            self,
            "outer_population_batch_size",
            outer_population_batch_size,
        )
        object.__setattr__(
            self,
            "outer_line_search_batch_size",
            outer_line_search_batch_size,
        )
        object.__setattr__(self, "inner_batch_size", inner_batch_size)
        object.__setattr__(self, "fixed_point_probe_paths", fixed_point_probe_paths)
        object.__setattr__(self, "log_every_outer", log_every_outer)
        object.__setattr__(
            self,
            "outer_objective_line_search",
            outer_objective_line_search,
        )
        object.__setattr__(self, "outer_relaxation_mode", outer_relaxation_mode)
        object.__setattr__(self, "outer_max_backtracks", outer_max_backtracks)
        object.__setattr__(self, "outer_backtrack_factor", outer_backtrack_factor)
        object.__setattr__(
            self,
            "outer_objective_tolerance",
            outer_objective_tolerance,
        )
        object.__setattr__(
            self,
            "outer_block_coordinate_passes",
            outer_block_coordinate_passes,
        )
        object.__setattr__(
            self,
            "outer_block_trust_fraction",
            outer_block_trust_fraction,
        )
        object.__setattr__(self, "inner_solver", inner_solver)
        object.__setattr__(self, "seed", seed)
