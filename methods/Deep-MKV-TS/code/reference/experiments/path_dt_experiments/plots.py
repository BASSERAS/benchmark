from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Sequence

import torch

from path_dt_experiments.metrics import acf, path_returns, return_correlation
from path_dt_experiments.volatility_diagnostics import REGIME_NAMES, VolatilityPathStatistics


def _to_numpy(value: torch.Tensor):
    return value.detach().cpu().numpy()


def _maybe_save(fig, output_path: str | Path | None) -> None:
    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")


def plot_heston_diagnostics(
    *,
    target_paths: torch.Tensor,
    generated_paths: torch.Tensor,
    output_path: str | Path | None = None,
    asset: int = 0,
    lags: Sequence[int] = tuple(range(1, 21)),
    bins: int = 80,
):
    """
    Create the four standard one-asset diagnostic plots.

    The panels are return density, tail survival of absolute returns, QQ plot,
    and ACF of absolute and squared returns.
    """
    import matplotlib.pyplot as plt

    target_returns = path_returns(target_paths)[..., asset].reshape(-1)
    generated_returns = path_returns(generated_paths)[..., asset].reshape(-1)
    target_abs = target_returns.abs()
    generated_abs = generated_returns.abs()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].hist(_to_numpy(target_returns), bins=bins, density=True, alpha=0.45, label="target")
    axes[0, 0].hist(_to_numpy(generated_returns), bins=bins, density=True, alpha=0.45, label="generated")
    axes[0, 0].set_title("Return density")
    axes[0, 0].legend()

    thresholds = torch.quantile(
        target_abs,
        torch.linspace(0.50, 0.995, 80, device=target_abs.device, dtype=target_abs.dtype),
    )
    target_survival = torch.stack([(target_abs > threshold).float().mean() for threshold in thresholds])
    generated_survival = torch.stack([(generated_abs > threshold).float().mean() for threshold in thresholds])
    axes[0, 1].semilogy(_to_numpy(thresholds), _to_numpy(target_survival), label="target")
    axes[0, 1].semilogy(_to_numpy(thresholds), _to_numpy(generated_survival), label="generated")
    axes[0, 1].set_title("Tail survival")
    axes[0, 1].legend()

    quantiles = torch.linspace(0.01, 0.99, 99, device=target_returns.device, dtype=target_returns.dtype)
    target_q = torch.quantile(target_returns, quantiles)
    generated_q = torch.quantile(generated_returns, quantiles)
    axes[1, 0].plot(_to_numpy(target_q), _to_numpy(generated_q), marker=".", linestyle="none")
    low = float(torch.minimum(target_q.min(), generated_q.min()).item())
    high = float(torch.maximum(target_q.max(), generated_q.max()).item())
    axes[1, 0].plot([low, high], [low, high], color="black", linewidth=1)
    axes[1, 0].set_title("QQ plot")
    axes[1, 0].set_xlabel("target quantile")
    axes[1, 0].set_ylabel("generated quantile")

    target_return_tensor = path_returns(target_paths)[..., asset : asset + 1]
    generated_return_tensor = path_returns(generated_paths)[..., asset : asset + 1]
    lag_values = tuple(int(lag) for lag in lags)
    target_abs_acf = acf(target_return_tensor.abs(), lags=lag_values)[:, 0]
    generated_abs_acf = acf(generated_return_tensor.abs(), lags=lag_values)[:, 0]
    target_sq_acf = acf(target_return_tensor.pow(2), lags=lag_values)[:, 0]
    generated_sq_acf = acf(generated_return_tensor.pow(2), lags=lag_values)[:, 0]
    axes[1, 1].plot(lag_values, _to_numpy(target_abs_acf), label="target |r|")
    axes[1, 1].plot(lag_values, _to_numpy(generated_abs_acf), label="generated |r|")
    axes[1, 1].plot(lag_values, _to_numpy(target_sq_acf), linestyle="--", label="target r^2")
    axes[1, 1].plot(lag_values, _to_numpy(generated_sq_acf), linestyle="--", label="generated r^2")
    axes[1, 1].set_title("Volatility ACF")
    axes[1, 1].set_xlabel("lag")
    axes[1, 1].legend()

    fig.tight_layout()
    _maybe_save(fig, output_path)
    return fig


def plot_heston_diagnostics_comparison(
    *,
    target_paths: torch.Tensor,
    candidate_paths: Mapping[str, torch.Tensor],
    output_path: str | Path | None = None,
    target_label: str = "Heston target",
    asset: int = 0,
    lags: Sequence[int] = tuple(range(1, 21)),
    bins: int = 100,
):
    """Compare several path banks with one target in the standard four panels."""
    import matplotlib.pyplot as plt

    if len(candidate_paths) == 0:
        raise ValueError("candidate_paths must not be empty")
    labels = [str(target_label), *(str(label) for label in candidate_paths)]
    if any(not label.strip() for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("target and candidate labels must be non-empty and unique")
    if int(bins) < 2:
        raise ValueError("bins must be >= 2")

    target_return_tensor = path_returns(target_paths)[..., asset : asset + 1]
    target_returns = target_return_tensor.reshape(-1)
    candidate_return_tensors = {
        str(label): path_returns(paths)[..., asset : asset + 1]
        for label, paths in candidate_paths.items()
    }
    candidate_returns = {
        label: returns.reshape(-1)
        for label, returns in candidate_return_tensors.items()
    }
    for label, paths in candidate_paths.items():
        if int(paths.shape[1]) != int(target_paths.shape[1]):
            raise ValueError(f"{label!r} has a different time dimension from the target")
        if int(paths.shape[2]) <= int(asset):
            raise ValueError(f"asset {asset} is unavailable in {label!r}")

    colors = {
        label: plt.get_cmap("tab10")(index % 10)
        for index, label in enumerate(candidate_returns)
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    all_returns = [target_returns, *candidate_returns.values()]
    density_low = min(float(torch.quantile(values, 0.001).item()) for values in all_returns)
    density_high = max(float(torch.quantile(values, 0.999).item()) for values in all_returns)
    axes[0, 0].hist(
        _to_numpy(target_returns),
        bins=int(bins),
        range=(density_low, density_high),
        density=True,
        histtype="step",
        color="black",
        linewidth=2.0,
        label=str(target_label),
    )
    for label, returns in candidate_returns.items():
        axes[0, 0].hist(
            _to_numpy(returns),
            bins=int(bins),
            range=(density_low, density_high),
            density=True,
            histtype="step",
            color=colors[label],
            linewidth=1.4,
            label=label,
        )
    axes[0, 0].set_title("Return density")
    axes[0, 0].legend(fontsize="small")

    target_abs = target_returns.abs()
    thresholds = torch.quantile(
        target_abs,
        torch.linspace(0.50, 0.995, 80, device=target_abs.device, dtype=target_abs.dtype),
    )
    target_survival = torch.stack([(target_abs > threshold).float().mean() for threshold in thresholds])
    axes[0, 1].semilogy(
        _to_numpy(thresholds),
        _to_numpy(target_survival),
        color="black",
        linewidth=2.0,
        label=str(target_label),
    )
    for label, returns in candidate_returns.items():
        absolute_returns = returns.abs()
        survival = torch.stack(
            [(absolute_returns > threshold).float().mean() for threshold in thresholds]
        )
        axes[0, 1].semilogy(
            _to_numpy(thresholds),
            _to_numpy(survival),
            color=colors[label],
            linewidth=1.4,
            label=label,
        )
    axes[0, 1].set_title("Tail survival")
    axes[0, 1].legend(fontsize="small")

    quantiles = torch.linspace(0.01, 0.99, 99, device=target_returns.device, dtype=target_returns.dtype)
    target_q = torch.quantile(target_returns, quantiles)
    qq_values = [target_q]
    for label, returns in candidate_returns.items():
        candidate_q = torch.quantile(returns, quantiles)
        qq_values.append(candidate_q)
        axes[1, 0].plot(
            _to_numpy(target_q),
            _to_numpy(candidate_q),
            color=colors[label],
            marker=".",
            markersize=3,
            linewidth=0.8,
            label=label,
        )
    low = min(float(values.min().item()) for values in qq_values)
    high = max(float(values.max().item()) for values in qq_values)
    axes[1, 0].plot([low, high], [low, high], color="black", linewidth=1, label="ideal")
    axes[1, 0].set_title("QQ plot against Heston target")
    axes[1, 0].set_xlabel("Heston target quantile")
    axes[1, 0].set_ylabel("candidate quantile")
    axes[1, 0].legend(fontsize="small")

    lag_values = tuple(int(lag) for lag in lags)
    target_abs_acf = acf(target_return_tensor.abs(), lags=lag_values)[:, 0]
    target_sq_acf = acf(target_return_tensor.pow(2), lags=lag_values)[:, 0]
    axes[1, 1].plot(
        lag_values,
        _to_numpy(target_abs_acf),
        color="black",
        linewidth=2.0,
        label=f"{target_label} |r|",
    )
    axes[1, 1].plot(
        lag_values,
        _to_numpy(target_sq_acf),
        color="black",
        linewidth=2.0,
        linestyle="--",
        label=f"{target_label} r²",
    )
    for label, returns in candidate_return_tensors.items():
        axes[1, 1].plot(
            lag_values,
            _to_numpy(acf(returns.abs(), lags=lag_values)[:, 0]),
            color=colors[label],
            linewidth=1.4,
            label=f"{label} |r|",
        )
        axes[1, 1].plot(
            lag_values,
            _to_numpy(acf(returns.pow(2), lags=lag_values)[:, 0]),
            color=colors[label],
            linewidth=1.4,
            linestyle="--",
            label=f"{label} r²",
        )
    axes[1, 1].set_title("Volatility clustering")
    axes[1, 1].set_xlabel("lag")
    axes[1, 1].legend(fontsize="x-small", ncol=2)

    fig.tight_layout()
    _maybe_save(fig, output_path)
    return fig


def plot_correlation_heatmaps(
    *,
    target_paths: torch.Tensor,
    generated_paths: torch.Tensor,
    output_path: str | Path | None = None,
):
    """Plot target, generated, and generated-minus-target return correlations."""
    import matplotlib.pyplot as plt

    target_corr = return_correlation(target_paths)
    generated_corr = return_correlation(generated_paths)
    error = generated_corr - target_corr
    matrices = [target_corr, generated_corr, error]
    titles = ["Target correlation", "Generated correlation", "Error"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for axis, matrix, title in zip(axes, matrices, titles):
        image = axis.imshow(_to_numpy(matrix), vmin=-1.0, vmax=1.0, cmap="coolwarm")
        axis.set_title(title)
        axis.set_xticks(range(int(matrix.shape[0])))
        axis.set_yticks(range(int(matrix.shape[0])))
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8)
    _maybe_save(fig, output_path)
    return fig


def plot_shadow_fan(
    *,
    real_paths: torch.Tensor,
    generated_paths: torch.Tensor,
    selected_indices: torch.Tensor,
    prefix_length: int,
    real_index: int = 0,
    asset: int = 0,
    output_path: str | Path | None = None,
):
    """Plot selected shadow continuations anchored at the observed endpoint."""
    import matplotlib.pyplot as plt

    if real_paths.ndim != 3 or generated_paths.ndim != 3:
        raise ValueError("paths must have shape (B, N + 1, d)")
    if selected_indices.ndim != 2:
        raise ValueError("selected_indices must have shape (B_real, top_k)")
    if real_index < 0 or real_index >= int(real_paths.shape[0]):
        raise ValueError("real_index out of range")
    if asset < 0 or asset >= int(real_paths.shape[2]):
        raise ValueError("asset out of range")

    shadows = generated_paths[selected_indices[real_index], prefix_length - 1 :, asset]
    generated_anchors = shadows[:, :1]
    observed_anchor = real_paths[real_index, prefix_length - 1, asset]
    shadows = observed_anchor + (shadows - generated_anchors)
    continuation_times = torch.arange(prefix_length - 1, int(real_paths.shape[1]))
    full_times = torch.arange(0, int(real_paths.shape[1]))
    shadow_mean = shadows.mean(dim=0)
    lower_90 = torch.quantile(shadows, 0.05, dim=0)
    upper_90 = torch.quantile(shadows, 0.95, dim=0)
    lower_50 = torch.quantile(shadows, 0.25, dim=0)
    upper_50 = torch.quantile(shadows, 0.75, dim=0)

    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(_to_numpy(full_times), _to_numpy(real_paths[real_index, :, asset]), color="black", linewidth=2, label="real")
    axis.axvline(prefix_length - 1, color="black", linewidth=1, alpha=0.5)
    axis.fill_between(_to_numpy(continuation_times), _to_numpy(lower_90), _to_numpy(upper_90), alpha=0.18, label="90% band")
    axis.fill_between(_to_numpy(continuation_times), _to_numpy(lower_50), _to_numpy(upper_50), alpha=0.28, label="50% band")
    for shadow in shadows:
        axis.plot(_to_numpy(continuation_times), _to_numpy(shadow), color="tab:blue", alpha=0.08, linewidth=0.8)
    axis.plot(_to_numpy(continuation_times), _to_numpy(shadow_mean), color="tab:blue", linewidth=2, label="shadow mean")
    axis.set_title("Path shadowing")
    axis.set_xlabel("time index")
    axis.legend()
    fig.tight_layout()
    _maybe_save(fig, output_path)
    return fig


def plot_volatility_diagnostic_summary(
    *,
    statistics_by_label: dict[str, VolatilityPathStatistics],
    reference_label: str,
    output_path: str | Path | None = None,
):
    """Plot ACF, realized-volatility distribution, and regime persistence diagnostics."""
    import matplotlib.pyplot as plt

    if reference_label not in statistics_by_label:
        raise ValueError("reference_label is not present in statistics_by_label")
    reference = statistics_by_label[reference_label]
    for label, statistics in statistics_by_label.items():
        if statistics.horizons != reference.horizons or statistics.lags != reference.lags:
            raise ValueError(f"{label} statistics do not use the reference horizons and lags")

    horizon = max(reference.horizons)
    regime_thresholds = torch.quantile(
        reference.prefix_rms_volatility.double(),
        torch.tensor([1.0 / 3.0, 2.0 / 3.0], dtype=torch.float64),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    for label, statistics in statistics_by_label.items():
        axes[0, 0].plot(statistics.lags, _to_numpy(statistics.absolute_return_acf), marker="o", label=label)
        axes[0, 1].plot(statistics.lags, _to_numpy(statistics.squared_return_acf), marker="o", label=label)
    axes[0, 0].set_title("Absolute-return ACF")
    axes[0, 1].set_title("Squared-return ACF")
    for axis in axes[0]:
        axis.set_xlabel("lag")
        axis.set_ylabel("correlation")
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.legend()

    for label, statistics in statistics_by_label.items():
        values = torch.sort(statistics.future_realized_volatility[horizon].double()).values
        probabilities = torch.arange(1, int(values.numel()) + 1, dtype=torch.float64) / float(values.numel())
        axes[1, 0].plot(_to_numpy(values), _to_numpy(probabilities), label=label)
    axes[1, 0].set_title(f"Future realized volatility ECDF (h={horizon})")
    axes[1, 0].set_xlabel("realized volatility")
    axes[1, 0].set_ylabel("empirical probability")
    axes[1, 0].legend()

    regime_positions = torch.arange(len(REGIME_NAMES))
    for label, statistics in statistics_by_label.items():
        regime_index = torch.bucketize(statistics.prefix_rms_volatility.double(), regime_thresholds)
        future = statistics.future_realized_volatility[horizon]
        means = [
            float(future[regime_index == index].mean().item())
            if bool((regime_index == index).any())
            else float("nan")
            for index in range(len(REGIME_NAMES))
        ]
        axes[1, 1].plot(_to_numpy(regime_positions), means, marker="o", label=label)
    axes[1, 1].set_xticks(_to_numpy(regime_positions), REGIME_NAMES)
    axes[1, 1].set_title(f"Future volatility by prefix regime (h={horizon})")
    axes[1, 1].set_xlabel("reference-defined prefix-volatility regime")
    axes[1, 1].set_ylabel("mean future realized volatility")
    axes[1, 1].legend()

    fig.tight_layout()
    _maybe_save(fig, output_path)
    return fig


def _bezier_route_points(*, sign: float, samples: int = 120) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, int(samples), dtype=torch.float32).view(-1, 1)
    one_minus = 1.0 - t
    p0 = torch.tensor([0.0, 0.0], dtype=torch.float32)
    p1 = torch.tensor([-0.95, 1.25 * float(sign)], dtype=torch.float32)
    p2 = torch.tensor([0.95, 1.05 * float(sign)], dtype=torch.float32)
    p3 = torch.tensor([-0.575, 0.8625], dtype=torch.float32)
    if float(sign) < 0.0:
        p3 = torch.tensor([0.575, -2.0125], dtype=torch.float32)
    return (
        one_minus.pow(3) * p0.view(1, 2)
        + 3.0 * one_minus.pow(2) * t * p1.view(1, 2)
        + 3.0 * one_minus * t.pow(2) * p2.view(1, 2)
        + t.pow(3) * p3.view(1, 2)
    )


def _add_two_moons_route_overlay(axis) -> None:
    import matplotlib.patches as patches

    obstacle = patches.Circle(
        (0.0, 0.0),
        radius=0.48,
        facecolor="#8E6C8A",
        edgecolor="#5E3C58",
        alpha=0.12,
        linewidth=1.0,
        zorder=0,
    )
    axis.add_patch(obstacle)
    for sign, color in ((1.0, "#2B8CBE"), (-1.0, "#C44E52")):
        route = _bezier_route_points(sign=sign)
        axis.plot(_to_numpy(route[:, 0]), _to_numpy(route[:, 1]), linestyle="--", linewidth=1.1, color=color, alpha=0.75)


def plot_two_moons_summary(
    *,
    target_paths: torch.Tensor,
    generated_paths: torch.Tensor,
    output_path: str | Path | None = None,
    max_points: int = 1200,
    max_trajectories: int = 80,
):
    """Plot terminal clouds, routed path samples, and generated marginals."""
    import matplotlib.pyplot as plt

    count = min(int(max_points), int(target_paths.shape[0]), int(generated_paths.shape[0]))
    traj_count = min(int(max_trajectories), count)
    target = target_paths[:count].detach().cpu()
    generated = generated_paths[:count].detach().cpu()
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].scatter(_to_numpy(target[:, 0, 0]), _to_numpy(target[:, 0, 1]), s=5, alpha=0.45, color="#4C78A8")
    axes[0, 0].set_title("source Gaussian")
    axes[0, 1].scatter(_to_numpy(target[:, -1, 0]), _to_numpy(target[:, -1, 1]), s=5, alpha=0.45, color="#54A24B")
    axes[0, 1].set_title("target terminal")
    axes[0, 2].scatter(_to_numpy(generated[:, -1, 0]), _to_numpy(generated[:, -1, 1]), s=5, alpha=0.45, color="#E45756")
    axes[0, 2].set_title("generated terminal")

    for idx in range(traj_count):
        axes[1, 0].plot(_to_numpy(target[idx, :, 0]), _to_numpy(target[idx, :, 1]), color="#54A24B", alpha=0.12, linewidth=0.8)
        axes[1, 1].plot(_to_numpy(generated[idx, :, 0]), _to_numpy(generated[idx, :, 1]), color="#E45756", alpha=0.12, linewidth=0.8)
    _add_two_moons_route_overlay(axes[1, 0])
    _add_two_moons_route_overlay(axes[1, 1])
    axes[1, 0].set_title("target routed paths")
    axes[1, 1].set_title("generated paths")

    time_indices = [
        0,
        int(generated.shape[1]) // 4,
        int(generated.shape[1]) // 2,
        3 * int(generated.shape[1]) // 4,
        int(generated.shape[1]) - 1,
    ]
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#B279A2", "#E45756"]
    for time_index, color in zip(time_indices, colors):
        axes[1, 2].scatter(
            _to_numpy(generated[:, time_index, 0]),
            _to_numpy(generated[:, time_index, 1]),
            s=4,
            alpha=0.35,
            color=color,
            label=f"k={time_index}",
        )
    axes[1, 2].set_title("generated marginals")
    axes[1, 2].legend(markerscale=2, fontsize=8)
    for axis in axes.ravel():
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
    _maybe_save(fig, output_path)
    return fig


def plot_two_moons_animation(
    *,
    target_paths: torch.Tensor,
    generated_paths: torch.Tensor,
    output_path: str | Path,
    max_points: int = 1000,
) -> None:
    """Write a GIF comparing target and generated two-moons path marginals."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = min(int(max_points), int(target_paths.shape[0]), int(generated_paths.shape[0]))
    target = target_paths[:count].detach().cpu()
    generated = generated_paths[:count].detach().cpu()
    x_values = torch.cat([generated[:, :, 0].reshape(-1), target[:, :, 0].reshape(-1)])
    y_values = torch.cat([generated[:, :, 1].reshape(-1), target[:, :, 1].reshape(-1)])
    x_pad = 0.15 * float((x_values.max() - x_values.min()).clamp_min(1e-6).item())
    y_pad = 0.15 * float((y_values.max() - y_values.min()).clamp_min(1e-6).item())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    target_scatter = axes[0].scatter([], [], s=5, alpha=0.45, color="#54A24B")
    generated_scatter = axes[1].scatter([], [], s=5, alpha=0.45, color="#E45756")
    for axis, title in zip(axes, ("target path law", "generated path law")):
        _add_two_moons_route_overlay(axis)
        axis.set_xlim(float(x_values.min().item()) - x_pad, float(x_values.max().item()) + x_pad)
        axis.set_ylim(float(y_values.min().item()) - y_pad, float(y_values.max().item()) + y_pad)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
        axis.set_title(title)
    time_text = fig.text(0.5, 0.02, "", ha="center", va="center")

    def update(frame: int):
        target_scatter.set_offsets(_to_numpy(target[:, frame, :]))
        generated_scatter.set_offsets(_to_numpy(generated[:, frame, :]))
        time_text.set_text(f"k = {frame} / {int(target.shape[1]) - 1}")
        return target_scatter, generated_scatter, time_text

    animation = FuncAnimation(fig, update, frames=int(target.shape[1]), interval=140, blit=False)
    animation.save(output, writer=PillowWriter(fps=8))
    plt.close(fig)
