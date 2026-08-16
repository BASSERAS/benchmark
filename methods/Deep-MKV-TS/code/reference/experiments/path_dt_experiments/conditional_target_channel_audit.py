from __future__ import annotations

from dataclasses import dataclass, replace
import itertools
import math
from typing import Mapping, Sequence

import torch

from deep_mkv_gen_path_dt.controls import RunningCostInputs
from deep_mkv_gen_path_dt.discrepancies import CompositePathFunctionalDiscrepancy
from deep_mkv_gen_path_dt.model import DiscreteMPModel
from deep_mkv_gen_path_dt.paths import future_source_sums
from path_dt_experiments.lifted_collocation_solver import (
    CausalControlPolicy,
    PolicyStepResult,
    ScenarioCollocationBank,
    ScenarioDataContext,
    projected_log_volatility_kkt,
    replay_branch_noise,
    rollout_causal_policy,
)


CHANNEL_NAMES = ("query_prefix", "population_law", "conditional_future")
BASELINE_CELL = "q0_m0_c0"
REFRESHED_CELL = "q1_m1_c1"


def cell_name(query: int, population: int, conditional: int) -> str:
    values = (int(query), int(population), int(conditional))
    if any(value not in (0, 1) for value in values):
        raise ValueError("factorial cell coordinates must be binary")
    return f"q{values[0]}_m{values[1]}_c{values[2]}"


@dataclass(frozen=True)
class DecisionSwitchPolicy:
    """Use one policy before each row's decision time and another afterwards."""

    prefix_policy: CausalControlPolicy
    future_policy: CausalControlPolicy
    decision_steps: torch.Tensor

    def __post_init__(self) -> None:
        steps = self.decision_steps.detach().clone().to(torch.long)
        if steps.ndim != 1 or int(steps.numel()) < 1:
            raise ValueError("decision_steps must be a nonempty vector")
        if bool(torch.any(steps < 0).item()):
            raise ValueError("decision_steps cannot be negative")
        object.__setattr__(self, "decision_steps", steps)

    def initial_state(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[object, object]:
        if int(batch_size) != int(self.decision_steps.numel()):
            raise ValueError("decision_steps must contain one entry per rollout row")
        return (
            self.prefix_policy.initial_state(
                batch_size=batch_size, device=device, dtype=dtype
            ),
            self.future_policy.initial_state(
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
            raise TypeError("switch-policy state must contain two policy states")
        prefix = self.prefix_policy.control_step(
            x_prefix=x_prefix, step_index=step_index, state=state[0]
        )
        future = self.future_policy.control_step(
            x_prefix=x_prefix, step_index=step_index, state=state[1]
        )
        mask = (
            int(step_index)
            < self.decision_steps.to(device=x_prefix.device)
        ).unsqueeze(-1)
        control = torch.where(mask, prefix.control, future.control)
        return PolicyStepResult(
            control=control,
            state=(prefix.state, future.state),
        )


def _component_family(name: str) -> str:
    if name == "running_cost":
        return "running_cost"
    if name == "path_law_terminal" or "terminal_return" in name:
        return "terminal"
    if name == "path_law_observed_path":
        return "path"
    if name == "path_law_increments":
        return "returns"
    if "prefix_future" in name:
        return "prefix_future"
    if name.endswith("_acf"):
        return "dependence"
    if name.startswith("volatility_law_"):
        return "realized_volatility"
    return "other"


def _weighted_discrepancy_blocks(
    model: DiscreteMPModel,
) -> tuple[tuple[str, float, object], ...]:
    discrepancy = model.discrepancy
    if isinstance(discrepancy, CompositePathFunctionalDiscrepancy):
        rows = tuple(
            (str(block.name), float(block.weight), block.discrepancy)
            for block in tuple(discrepancy.blocks)
        )
    else:
        rows = (("discrepancy", 1.0, discrepancy),)
    names = [row[0] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("discrepancy component names must be unique")
    return rows


def component_pathwise_targets(
    *,
    model: DiscreteMPModel,
    query_paths: torch.Tensor,
    population_paths: torch.Tensor,
    controls: torch.Tensor,
    target_paths: torch.Tensor,
    lambda_scale: float,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Compute additive pathwise adjoint targets for every discrepancy block."""

    if query_paths.ndim != 3 or population_paths.ndim != 3:
        raise ValueError("query and population paths must have rank three")
    if tuple(query_paths.shape[1:]) != tuple(population_paths.shape[1:]):
        raise ValueError("query and population paths must share path dimensions")
    if tuple(controls.shape[:2]) != (
        int(query_paths.shape[0]),
        int(model.grid.num_steps),
    ):
        raise ValueError("controls must align with query paths")
    scale = float(lambda_scale)
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("lambda_scale must be finite and nonnegative")

    query = query_paths.detach().clone().requires_grad_(True)
    population = population_paths.detach().to(device=query.device, dtype=query.dtype)
    target = target_paths.detach().to(device=query.device, dtype=query.dtype)
    if bool(model.training.observed_only):
        observed = model.grid.observed_point_indices(device=query.device)
        query_used = query.index_select(1, observed)
        population_used = population.index_select(1, observed)
        target_used = target.index_select(1, observed)
        times = model.grid.time_points(
            device=query.device, dtype=torch.float64
        ).index_select(0, observed)
        from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid

        discrepancy_grid = DiscreteTimeGrid(
            T=float(model.grid.T),
            num_steps=int(model.grid.observed_count) - 1,
            point_times=tuple(float(value) for value in times.cpu().tolist()),
            allow_nonuniform=True,
        )
    else:
        query_used = query
        population_used = population
        target_used = target
        discrepancy_grid = model.grid

    blocks = _weighted_discrepancy_blocks(model)
    source_rows: dict[str, torch.Tensor] = {}
    source_rms: dict[str, float] = {}
    for name, weight, discrepancy in blocks:
        potential = getattr(discrepancy, "lions_potential", None)
        if not callable(potential):
            raise TypeError(f"discrepancy block {name!r} has no Lions potential")
        value = scale * weight * potential(
            query_paths=query_used,
            population_paths=population_used,
            target_paths=target_used,
            grid=discrepancy_grid,
        )
        gradient = torch.autograd.grad(
            value,
            query,
            retain_graph=False,
            allow_unused=False,
        )[0]
        source = float(query.shape[0]) * gradient.detach()
        source_rows[name] = source
        source_rms[name] = float(source.double().pow(2).mean().sqrt().item())

    running_source = torch.zeros_like(query)
    if model.running_cost is not None:
        running_result = model.running_cost(
            RunningCostInputs(
                paths=query,
                controls=controls.detach(),
                grid=model.grid,
                parameters=model.parameters,
            )
        )
        if running_result.value.requires_grad:
            running_gradient = torch.autograd.grad(
                running_result.value,
                query,
                allow_unused=True,
            )[0]
            if running_gradient is not None:
                running_source = float(query.shape[0]) * running_gradient.detach()
    source_rows["running_cost"] = running_source
    source_rms["running_cost"] = float(
        running_source.double().pow(2).mean().sqrt().item()
    )
    targets = {
        name: future_source_sums(source)
        for name, source in source_rows.items()
    }
    total_source = sum(source_rows.values(), torch.zeros_like(query))
    targets["__total__"] = future_source_sums(total_source)
    reconstructed = sum(
        (value for name, value in targets.items() if name != "__total__"),
        torch.zeros_like(targets["__total__"]),
    )
    metadata = {
        **{f"{name}_source_rms": value for name, value in source_rms.items()},
        "component_reconstruction_max_abs": float(
            (reconstructed - targets["__total__"]).abs().max().item()
        ),
    }
    return targets, metadata


@dataclass(frozen=True)
class FactorialTargetBankResult:
    cells: dict[str, dict[str, dict[str, torch.Tensor]]]
    baseline_paths: torch.Tensor
    baseline_controls: torch.Tensor
    candidate_paths: torch.Tensor
    candidate_controls: torch.Tensor
    component_families: dict[str, str]
    metadata: dict[str, object]


def factorial_conditional_targets_on_bank(
    *,
    model: DiscreteMPModel,
    baseline_policy: CausalControlPolicy,
    candidate_policy: CausalControlPolicy,
    bank: ScenarioCollocationBank,
    data_context: ScenarioDataContext,
    full_training,
    query_batch_size: int,
) -> FactorialTargetBankResult:
    """Evaluate all prefix/population/future refresh combinations without updates."""

    bank.verify(data_context)
    initial = data_context.initial_state_pool.index_select(
        0, bank.initial_indices
    ).to(device=model.device, dtype=model.dtype)
    target_paths = data_context.target_path_pool.index_select(
        0, bank.target_indices
    ).to(device=model.device, dtype=model.dtype)
    x0 = initial[:, 0, :]
    base_noise = bank.base_noise.to(device=model.device, dtype=model.dtype)
    scaled_training = replace(
        full_training,
        lambda_scale=float(full_training.lambda_scale) * float(bank.beta),
    )
    with torch.no_grad():
        baseline_law = rollout_causal_policy(
            model=model, policy=baseline_policy, x0=x0, noise=base_noise
        )
        candidate_law = rollout_causal_policy(
            model=model, policy=candidate_policy, x0=x0, noise=base_noise
        )
    laws = (baseline_law, candidate_law)
    policies = (baseline_policy, candidate_policy)
    batch_size = int(x0.shape[0])
    branches = int(bank.num_branches)
    num_steps = int(model.grid.num_steps)
    state_dim = int(model.architecture.state_dim)
    noise_dim = int(model.architecture.noise_dim)
    noise_target_dim = int(model._default_noise_adjoint_dim(model.architecture))
    queries_per_step = batch_size * branches
    if int(query_batch_size) < queries_per_step:
        raise ValueError("query_batch_size must hold one complete branch population")
    steps_per_chunk = max(1, int(query_batch_size) // queries_per_step)
    component_names = tuple(name for name, _, _ in _weighted_discrepancy_blocks(model)) + (
        "running_cost",
        "__total__",
    )
    cells: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for query, population, conditional in itertools.product((0, 1), repeat=3):
        key = cell_name(query, population, conditional)
        cells[key] = {
            name: {
                "p": torch.empty(
                    batch_size, num_steps, state_dim,
                    device=model.device, dtype=model.dtype,
                ),
                "r": torch.empty(
                    batch_size, num_steps, noise_target_dim,
                    device=model.device, dtype=model.dtype,
                ),
            }
            for name in component_names
        }

    prefix_errors: dict[str, float] = {}
    reconstruction_errors: list[float] = []
    flat_x0_one_step = x0[:, None, :].expand(
        batch_size, branches, state_dim
    ).reshape(queries_per_step, state_dim)
    for query, conditional in itertools.product((0, 1), repeat=2):
        qf_label = f"q{query}_c{conditional}"
        max_prefix_error = 0.0
        for chunk_start in range(0, num_steps, steps_per_chunk):
            chunk_steps = tuple(
                range(chunk_start, min(num_steps, chunk_start + steps_per_chunk))
            )
            noise_rows = []
            for step in chunk_steps:
                noise_rows.append(
                    replay_branch_noise(
                        model=model,
                        bank=bank,
                        base_noise=base_noise,
                        step_index=step,
                    ).reshape(queries_per_step, num_steps, noise_dim)
                )
            flat_noise = torch.cat(noise_rows, dim=0)
            flat_x0 = flat_x0_one_step.repeat(len(chunk_steps), 1)
            decision_steps = torch.tensor(
                chunk_steps, dtype=torch.long, device=model.device
            ).repeat_interleave(queries_per_step)
            hybrid = DecisionSwitchPolicy(
                prefix_policy=policies[query],
                future_policy=policies[conditional],
                decision_steps=decision_steps,
            )
            with torch.no_grad():
                branch_law = rollout_causal_policy(
                    model=model,
                    policy=hybrid,
                    x0=flat_x0,
                    noise=flat_noise,
                )
            for local, step in enumerate(chunk_steps):
                left = local * queries_per_step
                right = left + queries_per_step
                if step > 0:
                    actual = branch_law.paths[left:right].reshape(
                        batch_size, branches, num_steps + 1, state_dim
                    )[:, :, : step + 1, :]
                    expected = laws[query].paths[:, None, : step + 1, :]
                    max_prefix_error = max(
                        max_prefix_error,
                        float((actual - expected).abs().max().item()),
                    )
            for population in (0, 1):
                pathwise, metadata = component_pathwise_targets(
                    model=model,
                    query_paths=branch_law.paths,
                    population_paths=laws[population].paths,
                    controls=branch_law.controls,
                    target_paths=target_paths,
                    lambda_scale=float(scaled_training.lambda_scale),
                )
                reconstruction_errors.append(
                    float(metadata["component_reconstruction_max_abs"])
                )
                for name, target in pathwise.items():
                    score = model._adjoint_noise_target(
                        pathwise_adjoint_target=target,
                        moments_noise_target_dim=noise_target_dim,
                        noise=flat_noise,
                    )
                    for local, step in enumerate(chunk_steps):
                        left = local * queries_per_step
                        right = left + queries_per_step
                        key = cell_name(query, population, conditional)
                        cells[key][name]["p"][:, step, :] = target[
                            left:right, step, :
                        ].reshape(batch_size, branches, state_dim).mean(dim=1)
                        cells[key][name]["r"][:, step, :] = score[
                            left:right, step, :
                        ].reshape(batch_size, branches, noise_target_dim).mean(dim=1)
        prefix_errors[qf_label] = max_prefix_error

    detached_cells = {
        key: {
            name: {
                signal: value.detach().cpu()
                for signal, value in signals.items()
            }
            for name, signals in components.items()
        }
        for key, components in cells.items()
    }
    families = {
        name: _component_family(name)
        for name in component_names
        if name != "__total__"
    }
    return FactorialTargetBankResult(
        cells=detached_cells,
        baseline_paths=baseline_law.paths.detach().cpu(),
        baseline_controls=baseline_law.controls.detach().cpu(),
        candidate_paths=candidate_law.paths.detach().cpu(),
        candidate_controls=candidate_law.controls.detach().cpu(),
        component_families=families,
        metadata={
            "bank_fingerprint": bank.fingerprint,
            "effective_lambda_scale": float(scaled_training.lambda_scale),
            "steps_per_chunk": steps_per_chunk,
            "prefix_max_abs_errors": prefix_errors,
            "component_reconstruction_max_abs": max(reconstruction_errors),
            "conditional_operator": "direct_antithetic_future_branch_average",
            "conditional_freeze": "future_continuation_policy",
        },
    )


def _rms(value: torch.Tensor) -> float:
    return float(value.detach().double().pow(2).mean().sqrt().item())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    x = left.detach().double().reshape(-1)
    y = right.detach().double().reshape(-1)
    denominator = x.norm() * y.norm()
    if float(denominator.item()) <= torch.finfo(torch.float64).eps:
        return 0.0
    return float((torch.dot(x, y) / denominator).item())


def joint_target_vector(
    *, p: torch.Tensor, r: torch.Tensor, p_scale: float, r_scale: float
) -> torch.Tensor:
    return torch.cat(
        (
            p.detach().double().reshape(-1) / float(p_scale),
            r.detach().double().reshape(-1) / float(r_scale),
        )
    )


def shapley_vectors(
    values: Mapping[tuple[int, ...], torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Exact vector-valued Shapley decomposition on a complete Boolean cube."""

    if not values:
        raise ValueError("factorial values cannot be empty")
    dimension = len(next(iter(values)))
    expected = set(itertools.product((0, 1), repeat=dimension))
    if set(values) != expected:
        raise ValueError("factorial values must contain the complete Boolean cube")
    shape = tuple(next(iter(values.values())).shape)
    if any(tuple(value.shape) != shape for value in values.values()):
        raise ValueError("all factorial value tensors must have matching shapes")
    result = []
    factorial = math.factorial
    for index in range(dimension):
        contribution = torch.zeros_like(next(iter(values.values())))
        for subset_size in range(dimension):
            weight = (
                factorial(subset_size)
                * factorial(dimension - subset_size - 1)
                / factorial(dimension)
            )
            for active in itertools.combinations(
                [item for item in range(dimension) if item != index],
                subset_size,
            ):
                base = [0] * dimension
                for item in active:
                    base[item] = 1
                extended = list(base)
                extended[index] = 1
                contribution = contribution + weight * (
                    values[tuple(extended)] - values[tuple(base)]
                )
        result.append(contribution)
    return tuple(result)


def _vector_attribution(
    contributions: Mapping[str, torch.Tensor], total: torch.Tensor
) -> dict[str, dict[str, float]]:
    epsilon = torch.finfo(torch.float64).eps
    total_norm = float(total.norm().item())
    norm_sum = sum(float(value.norm().item()) for value in contributions.values())
    denominator = max(float(torch.dot(total, total).item()), epsilon)
    report = {}
    for name, value in contributions.items():
        norm = float(value.norm().item())
        report[name] = {
            "norm": norm,
            "magnitude_share": norm / max(norm_sum, epsilon),
            "alignment_fraction": float(torch.dot(value, total).item()) / denominator,
            "cosine_to_total": (
                0.0 if norm == 0.0 or total_norm == 0.0 else _cosine(value, total)
            ),
        }
    return report


def summarize_factorial_bank(
    *, model: DiscreteMPModel, result: FactorialTargetBankResult
) -> dict[str, object]:
    baseline = result.cells[BASELINE_CELL]["__total__"]
    refreshed = result.cells[REFRESHED_CELL]["__total__"]
    p_scale = max(_rms(baseline["p"]), torch.finfo(torch.float64).eps)
    r_scale = max(_rms(baseline["r"]), torch.finfo(torch.float64).eps)
    vectors: dict[tuple[int, ...], torch.Tensor] = {}
    for query, population, conditional in itertools.product((0, 1), repeat=3):
        cell = result.cells[cell_name(query, population, conditional)]["__total__"]
        vectors[(query, population, conditional)] = joint_target_vector(
            p=cell["p"], r=cell["r"], p_scale=p_scale, r_scale=r_scale
        )
    total_delta = vectors[(1, 1, 1)] - vectors[(0, 0, 0)]
    shapley = shapley_vectors(vectors)
    channel_vectors = dict(zip(CHANNEL_NAMES, shapley))
    channels = _vector_attribution(channel_vectors, total_delta)
    shapley_reconstruction = sum(shapley, torch.zeros_like(total_delta))
    head_channels = {}
    for signal in ("p", "r"):
        signal_values = {
            (query, population, conditional): result.cells[
                cell_name(query, population, conditional)
            ]["__total__"][signal].detach().double().reshape(-1)
            for query, population, conditional in itertools.product((0, 1), repeat=3)
        }
        signal_total = signal_values[(1, 1, 1)] - signal_values[(0, 0, 0)]
        signal_shapley = shapley_vectors(signal_values)
        signal_attribution = _vector_attribution(
            dict(zip(CHANNEL_NAMES, signal_shapley)), signal_total
        )
        for channel in CHANNEL_NAMES:
            head_channels.setdefault(channel, {})[signal] = signal_attribution[channel]

    component_vectors = {}
    for component in result.component_families:
        before = result.cells[BASELINE_CELL][component]
        after = result.cells[REFRESHED_CELL][component]
        component_vectors[component] = joint_target_vector(
            p=after["p"] - before["p"],
            r=after["r"] - before["r"],
            p_scale=p_scale,
            r_scale=r_scale,
        )
    components = _vector_attribution(component_vectors, total_delta)
    family_vectors: dict[str, torch.Tensor] = {}
    for component, vector in component_vectors.items():
        family = result.component_families[component]
        family_vectors[family] = family_vectors.get(
            family, torch.zeros_like(vector)
        ) + vector
    families = _vector_attribution(family_vectors, total_delta)
    component_reconstruction = sum(
        component_vectors.values(), torch.zeros_like(total_delta)
    )

    control = model.control_map
    controls_from_moments = getattr(control, "controls_from_moments", None)
    if not callable(controls_from_moments):
        raise TypeError("channel audit requires controls_from_moments")
    cell_report = {}
    baseline_induced = None
    induced_by_cell = {}
    baseline_vector = vectors[(0, 0, 0)]
    for query, population, conditional in itertools.product((0, 1), repeat=3):
        key = cell_name(query, population, conditional)
        target = result.cells[key]["__total__"]
        paths = result.baseline_paths if query == 0 else result.candidate_paths
        current_controls = (
            result.baseline_controls if query == 0 else result.candidate_controls
        )
        induced = controls_from_moments(
            paths=paths.to(model.device),
            expected_adjoint_next=target["p"].to(model.device),
            expected_adjoint_noise_next=target["r"].to(model.device),
        ).detach().cpu()
        if baseline_induced is None:
            baseline_induced = induced
        induced_by_cell[key] = induced
        state_dim = int(model.architecture.state_dim)
        kkt = projected_log_volatility_kkt(
            model=model,
            paths=paths.to(model.device),
            sigma=current_controls[..., state_dim:].to(model.device),
            moment_r=target["r"].to(model.device),
            gamma=1.0,
        )
        vector = vectors[(query, population, conditional)]
        delta = vector - baseline_vector
        cell_report[key] = {
            "p_target_rms": _rms(target["p"]),
            "r_target_rms": _rms(target["r"]),
            "joint_change_rms": float(delta.pow(2).mean().sqrt().item()),
            "cosine_to_complete_target_change": _cosine(delta, total_delta),
            "target_kkt_rms_at_cell_law": _rms(kkt.residual),
            "induced_drift_change_rms": _rms(
                induced[..., :state_dim] - baseline_induced[..., :state_dim]
            ),
            "induced_log_volatility_change_rms": _rms(
                torch.log(induced[..., state_dim:])
                - torch.log(baseline_induced[..., state_dim:])
            ),
        }

    assert baseline_induced is not None
    control_channels: dict[str, dict[str, dict[str, float]]] = {
        channel: {} for channel in CHANNEL_NAMES
    }
    state_dim = int(model.architecture.state_dim)
    for signal, transform in (
        ("drift", lambda value: value[..., :state_dim]),
        ("log_volatility", lambda value: torch.log(value[..., state_dim:])),
    ):
        control_values = {
            (query, population, conditional): transform(
                induced_by_cell[cell_name(query, population, conditional)]
            ).double().reshape(-1)
            for query, population, conditional in itertools.product((0, 1), repeat=3)
        }
        control_total = control_values[(1, 1, 1)] - control_values[(0, 0, 0)]
        control_shapley = shapley_vectors(control_values)
        attribution = _vector_attribution(
            dict(zip(CHANNEL_NAMES, control_shapley)), control_total
        )
        for channel in CHANNEL_NAMES:
            control_channels[channel][signal] = attribution[channel]
    law_drift_change = (
        result.candidate_controls[..., :state_dim]
        - result.baseline_controls[..., :state_dim]
    )
    law_log_volatility_change = (
        torch.log(result.candidate_controls[..., state_dim:])
        - torch.log(result.baseline_controls[..., state_dim:])
    )
    target_drift_change = (
        induced_by_cell[REFRESHED_CELL][..., :state_dim]
        - induced_by_cell[BASELINE_CELL][..., :state_dim]
    )
    target_log_volatility_change = (
        torch.log(induced_by_cell[REFRESHED_CELL][..., state_dim:])
        - torch.log(induced_by_cell[BASELINE_CELL][..., state_dim:])
    )

    top_channel = max(channels, key=lambda name: channels[name]["magnitude_share"])
    top_family = max(families, key=lambda name: families[name]["magnitude_share"])
    top_component = max(
        components, key=lambda name: components[name]["magnitude_share"]
    )
    return {
        "scales": {"p_target_rms": p_scale, "r_target_rms": r_scale},
        "complete_change": {
            "joint_norm": float(total_delta.norm().item()),
            "p_rms": _rms(refreshed["p"] - baseline["p"]),
            "r_rms": _rms(refreshed["r"] - baseline["r"]),
        },
        "channels": channels,
        "head_channels": head_channels,
        "control_channels": control_channels,
        "control_reversal": {
            "target_vs_law_drift_cosine": _cosine(
                target_drift_change, law_drift_change
            ),
            "target_vs_law_log_volatility_cosine": _cosine(
                target_log_volatility_change, law_log_volatility_change
            ),
            "law_drift_change_rms": _rms(law_drift_change),
            "law_log_volatility_change_rms": _rms(law_log_volatility_change),
            "target_drift_change_rms": _rms(target_drift_change),
            "target_log_volatility_change_rms": _rms(
                target_log_volatility_change
            ),
        },
        "top_channel": top_channel,
        "components": components,
        "top_component": top_component,
        "families": families,
        "top_family": top_family,
        "cells": cell_report,
        "reconstruction": {
            "shapley_relative_error": float(
                (shapley_reconstruction - total_delta).norm().item()
                / max(float(total_delta.norm().item()), torch.finfo(torch.float64).eps)
            ),
            "component_relative_error": float(
                (component_reconstruction - total_delta).norm().item()
                / max(float(total_delta.norm().item()), torch.finfo(torch.float64).eps)
            ),
        },
        "metadata": result.metadata,
    }


def locked_channel_decision(
    banks: Mapping[str, Mapping[str, object]],
    *,
    dominance_threshold: float = 0.50,
) -> dict[str, object]:
    threshold = float(dominance_threshold)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("dominance_threshold must lie in (0, 1]")
    if set(banks) != {"fit", "confirmation"}:
        raise ValueError("locked decision requires fit and confirmation banks")

    def stable_dominant(group: str, top_name: str) -> tuple[str | None, dict[str, bool]]:
        top = [str(banks[bank][top_name]) for bank in ("fit", "confirmation")]
        checks = {
            "same_top": top[0] == top[1],
            "fit_share": float(
                banks["fit"][group][top[0]]["magnitude_share"]
            ) >= threshold,
            "confirmation_share": float(
                banks["confirmation"][group][top[1]]["magnitude_share"]
            ) >= threshold,
            "fit_positive_alignment": float(
                banks["fit"][group][top[0]]["alignment_fraction"]
            ) > 0.0,
            "confirmation_positive_alignment": float(
                banks["confirmation"][group][top[1]]["alignment_fraction"]
            ) > 0.0,
        }
        return (top[0] if all(checks.values()) else None), checks

    channel, channel_checks = stable_dominant("channels", "top_channel")
    family, family_checks = stable_dominant("families", "top_family")
    if channel == "population_law":
        next_experiment = "mixture_of_complete_laws"
    elif channel == "query_prefix":
        next_experiment = "multiple_shooting"
    elif channel == "conditional_future":
        next_experiment = "persistent_scenario_tree"
    elif family is not None:
        next_experiment = "componentwise_continuation"
    else:
        next_experiment = "distributed_feedback_no_targeted_solver"
    return {
        "dominance_threshold": threshold,
        "dominant_channel": channel,
        "dominant_discrepancy_family": family,
        "channel_checks": channel_checks,
        "family_checks": family_checks,
        "next_experiment": next_experiment,
    }


__all__ = [
    "BASELINE_CELL",
    "CHANNEL_NAMES",
    "DecisionSwitchPolicy",
    "FactorialTargetBankResult",
    "REFRESHED_CELL",
    "cell_name",
    "component_pathwise_targets",
    "factorial_conditional_targets_on_bank",
    "joint_target_vector",
    "locked_channel_decision",
    "shapley_vectors",
    "summarize_factorial_bank",
]
