from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class HestonConfig:
    """Parameters for the synthetic Heston log-price generator."""

    T: float = 1.0
    num_steps: int = 64
    mu: float = 0.0
    kappa: float = 3.0
    theta: float = 0.04
    vol_of_vol: float = 0.6
    rho: float = -0.7
    v0: float = 0.04
    x0: float = 0.0
    variance_floor: float = 1e-8

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def simulate_heston_log_paths(
    *,
    num_paths: int,
    config: HestonConfig,
    seed: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Simulate one-dimensional Heston log-price paths.

    The returned tensor has shape ``(num_paths, num_steps + 1, 1)``.  The
    variance path is latent and is used only to generate the target log-price
    law.
    """
    if num_paths < 1:
        raise ValueError("num_paths must be >= 1")
    if config.T <= 0.0:
        raise ValueError("T must be > 0")
    if config.num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if not (-1.0 <= float(config.rho) <= 1.0):
        raise ValueError("rho must lie in [-1, 1]")
    if config.theta <= 0.0 or config.v0 <= 0.0:
        raise ValueError("theta and v0 must be > 0")
    if config.variance_floor <= 0.0:
        raise ValueError("variance_floor must be > 0")

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed))
    z_price = torch.randn(
        int(num_paths),
        int(config.num_steps),
        generator=generator,
        device="cpu",
        dtype=dtype,
    ).to(device=device)
    z_variance_independent = torch.randn(
        int(num_paths),
        int(config.num_steps),
        generator=generator,
        device="cpu",
        dtype=dtype,
    ).to(device=device)
    z_var = float(config.rho) * z_price + math.sqrt(max(0.0, 1.0 - float(config.rho) ** 2)) * z_variance_independent

    dt = float(config.T) / float(config.num_steps)
    sqrt_dt = math.sqrt(dt)
    x = torch.full((int(num_paths),), float(config.x0), device=device, dtype=dtype)
    v = torch.full((int(num_paths),), float(config.v0), device=device, dtype=dtype)
    values = [x]
    for step in range(int(config.num_steps)):
        v_pos = v.clamp_min(float(config.variance_floor))
        x = x + (float(config.mu) - 0.5 * v_pos) * dt + torch.sqrt(v_pos) * sqrt_dt * z_price[:, step]
        v = (
            v
            + float(config.kappa) * (float(config.theta) - v_pos) * dt
            + float(config.vol_of_vol) * torch.sqrt(v_pos) * sqrt_dt * z_var[:, step]
        ).clamp_min(float(config.variance_floor))
        values.append(x)
    return torch.stack(values, dim=1).unsqueeze(-1)


@dataclass(frozen=True)
class RoutedTwoMoonsConfig:
    """Parameters for the path-dependent Gaussian-to-two-moons generator."""

    T: float = 1.0
    num_steps: int = 32
    source_std: float = 0.35
    terminal_noise_std: float = 0.05
    bridge_noise_std: float = 0.03

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class RoutedTwoMoonsBatch:
    x_path: torch.Tensor
    labels: torch.Tensor
    terminal: torch.Tensor
    x0: torch.Tensor


def _make_cpu_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def sample_two_moons_terminal(
    *,
    num_paths: int,
    noise_std: float = 0.05,
    seed: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(num_paths) < 1:
        raise ValueError("num_paths must be >= 1")
    if float(noise_std) < 0.0:
        raise ValueError("noise_std must be >= 0")
    generator = _make_cpu_generator(seed)
    labels = torch.randint(0, 2, (int(num_paths),), generator=generator, device="cpu")
    theta = math.pi * torch.rand(int(num_paths), generator=generator, device="cpu", dtype=dtype)
    upper = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    lower = torch.stack([1.0 - torch.cos(theta), -torch.sin(theta) - 0.5], dim=1)
    terminal = torch.where(labels.view(-1, 1).bool(), lower, upper)
    terminal = terminal - torch.tensor([0.5, 0.25], dtype=dtype)
    terminal = 1.15 * terminal
    if float(noise_std) > 0.0:
        terminal = terminal + float(noise_std) * torch.randn(
            int(num_paths),
            2,
            generator=generator,
            device="cpu",
            dtype=dtype,
        )
    return terminal.to(device=device), labels.to(device=device)


def sample_routed_two_moons_paths(
    *,
    num_paths: int,
    config: RoutedTwoMoonsConfig,
    seed: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> RoutedTwoMoonsBatch:
    if int(num_paths) < 1:
        raise ValueError("num_paths must be >= 1")
    if float(config.T) <= 0.0:
        raise ValueError("T must be > 0")
    if int(config.num_steps) < 1:
        raise ValueError("num_steps must be >= 1")
    if float(config.source_std) <= 0.0:
        raise ValueError("source_std must be > 0")
    if float(config.terminal_noise_std) < 0.0:
        raise ValueError("terminal_noise_std must be >= 0")
    if float(config.bridge_noise_std) < 0.0:
        raise ValueError("bridge_noise_std must be >= 0")

    generator = _make_cpu_generator(seed)
    x0 = float(config.source_std) * torch.randn(int(num_paths), 2, generator=generator, device="cpu", dtype=dtype)
    terminal, labels = sample_two_moons_terminal(
        num_paths=int(num_paths),
        noise_std=float(config.terminal_noise_std),
        seed=None if seed is None else int(seed) + 17,
        device="cpu",
        dtype=dtype,
    )
    labels_cpu = labels.to(device="cpu")
    route_sign = torch.where(labels_cpu == 0, 1.0, -1.0).to(dtype=dtype).view(-1, 1)
    p0 = x0
    p3 = terminal.to(device="cpu", dtype=dtype)
    p1 = torch.cat(
        [
            -0.95 * torch.ones(int(num_paths), 1, device="cpu", dtype=dtype),
            1.25 * route_sign,
        ],
        dim=1,
    )
    p2 = torch.cat(
        [
            0.95 * torch.ones(int(num_paths), 1, device="cpu", dtype=dtype),
            1.05 * route_sign,
        ],
        dim=1,
    )
    times = torch.linspace(0.0, float(config.T), int(config.num_steps) + 1, device="cpu", dtype=dtype)
    normalized_times = (times / float(config.T)).view(1, -1, 1)
    one_minus = 1.0 - normalized_times
    path = (
        one_minus.pow(3) * p0.unsqueeze(1)
        + 3.0 * one_minus.pow(2) * normalized_times * p1.unsqueeze(1)
        + 3.0 * one_minus * normalized_times.pow(2) * p2.unsqueeze(1)
        + normalized_times.pow(3) * p3.unsqueeze(1)
    )
    if float(config.bridge_noise_std) > 0.0:
        bridge_scale = torch.sqrt((normalized_times * (1.0 - normalized_times)).clamp_min(0.0))
        path = path + float(config.bridge_noise_std) * bridge_scale * torch.randn(
            int(num_paths),
            int(config.num_steps) + 1,
            2,
            generator=generator,
            device="cpu",
            dtype=dtype,
        )
    path[:, 0, :] = p0
    path[:, -1, :] = p3
    return RoutedTwoMoonsBatch(
        x_path=path.to(device=device),
        labels=labels_cpu.to(device=device),
        terminal=p3.to(device=device),
        x0=p0.to(device=device),
    )


@dataclass(frozen=True)
class MultivariateDiagonalConfig:
    """Synthetic diagonal-diffusion benchmark parameters."""

    T: float = 1.0
    num_steps: int = 32
    state_dim: int = 3
    drift: Sequence[float] | None = None
    base_sigma: Sequence[float] | None = None
    mean_reversion: float = 0.0
    state_sensitivity: float = 0.0
    cross_sensitivity: float = 0.0
    sigma_min: float = 0.02
    sigma_max: float = 0.6
    initial_mean: Sequence[float] | None = None
    initial_std: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MultivariateDiagonalBatch:
    x_path: torch.Tensor
    effective_sigma_path: torch.Tensor
    drift_path: torch.Tensor
    x0: torch.Tensor


def _as_parameter_vector(
    values: Sequence[float] | None,
    *,
    default: float,
    state_dim: int,
    name: str,
) -> tuple[float, ...]:
    if values is None:
        return tuple(float(default) for _ in range(int(state_dim)))
    vector = tuple(float(value) for value in values)
    if len(vector) == 1:
        return tuple(vector[0] for _ in range(int(state_dim)))
    if len(vector) != int(state_dim):
        raise ValueError(f"{name} must have length 1 or state_dim")
    return vector


def _parameter_tensor(
    values: Sequence[float] | None,
    *,
    default: float,
    state_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.tensor(
        _as_parameter_vector(values, default=default, state_dim=int(state_dim), name="parameter"),
        dtype=dtype,
        device=device,
    ).view(1, int(state_dim))


def _diagonal_oracle_drift_and_sigma(
    x: torch.Tensor,
    *,
    config: MultivariateDiagonalConfig,
    drift_vector: torch.Tensor,
    base_sigma_vector: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    drift = drift_vector - float(config.mean_reversion) * x
    if float(config.state_sensitivity) == 0.0 and float(config.cross_sensitivity) == 0.0:
        sigma = base_sigma_vector.expand_as(x)
    else:
        own_signal = float(config.state_sensitivity) * torch.tanh(x)
        cross_signal = float(config.cross_sensitivity) * torch.tanh(x.mean(dim=1, keepdim=True))
        sigma = base_sigma_vector * torch.exp(own_signal + cross_signal)
    sigma = torch.clamp(sigma, min=float(config.sigma_min), max=float(config.sigma_max))
    return drift, sigma


def sample_multivariate_diagonal_paths(
    *,
    num_paths: int,
    config: MultivariateDiagonalConfig,
    seed: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> MultivariateDiagonalBatch:
    if int(num_paths) < 1:
        raise ValueError("num_paths must be >= 1")
    if float(config.T) <= 0.0:
        raise ValueError("T must be > 0")
    if int(config.num_steps) < 1:
        raise ValueError("num_steps must be >= 1")
    if int(config.state_dim) < 1:
        raise ValueError("state_dim must be >= 1")
    if float(config.sigma_min) <= 0.0:
        raise ValueError("sigma_min must be > 0")
    if float(config.sigma_max) <= float(config.sigma_min):
        raise ValueError("sigma_max must be > sigma_min")
    if float(config.initial_std) < 0.0:
        raise ValueError("initial_std must be >= 0")

    generator = _make_cpu_generator(seed)
    cpu = torch.device("cpu")
    state_dim = int(config.state_dim)
    drift_vector = _parameter_tensor(config.drift, default=0.0, state_dim=state_dim, device=cpu, dtype=dtype)
    base_sigma_vector = _parameter_tensor(config.base_sigma, default=0.2, state_dim=state_dim, device=cpu, dtype=dtype)
    initial_mean = _parameter_tensor(config.initial_mean, default=0.0, state_dim=state_dim, device=cpu, dtype=dtype)
    if float(config.initial_std) > 0.0:
        x = initial_mean + float(config.initial_std) * torch.randn(
            int(num_paths),
            state_dim,
            generator=generator,
            dtype=dtype,
            device=cpu,
        )
    else:
        x = initial_mean.expand(int(num_paths), state_dim).clone()

    noise = torch.randn(
        int(num_paths),
        int(config.num_steps),
        state_dim,
        generator=generator,
        dtype=dtype,
        device=cpu,
    )
    x_points = [x]
    drift_values = []
    sigma_values = []
    dt = float(config.T) / float(config.num_steps)
    sqrt_dt = math.sqrt(dt)
    for _ in range(int(config.num_steps)):
        drift, sigma = _diagonal_oracle_drift_and_sigma(
            x,
            config=config,
            drift_vector=drift_vector,
            base_sigma_vector=base_sigma_vector,
        )
        x = x + drift * dt + sigma * sqrt_dt * noise[:, len(drift_values), :]
        drift_values.append(drift)
        sigma_values.append(sigma)
        x_points.append(x)

    return MultivariateDiagonalBatch(
        x_path=torch.stack(x_points, dim=1).to(device=device),
        effective_sigma_path=torch.stack(sigma_values, dim=1).to(device=device),
        drift_path=torch.stack(drift_values, dim=1).to(device=device),
        x0=x_points[0].to(device=device),
    )
