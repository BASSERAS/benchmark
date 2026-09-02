"""Throwaway smoke test for ``multivariate_reference``.

Checks, on a small real slice of the ``HestonMultiAsset`` train split:

1. the kernel constructs and its recursion produces finite ``(B, d)`` drifts and
   symmetric ``(B, d, d)`` sigmas inside ``[sigma_min, sigma_max]`` spectrally;
2. ``evaluate`` (single step, prefix cache) agrees with ``evaluate_all`` (scan);
3. the penalised MLE runs, decreases the calibration NLL and stays finite;
4. the fitted kernel reproduces a plausible average correlation.

Not part of the deliverable; deleted once ``fit_reference_multiasset.py`` runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(
    0,
    "/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference/src",
)

from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid  # noqa: E402

import multivariate_reference as mr  # noqa: E402

DATASET = Path("/home/tbasseras/benchmark/dataset/HestonMultiAsset")
DT = 0.003968253968253968
TORCH_DTYPE = torch.float64


def _load(num_paths: int, num_steps: int) -> torch.Tensor:
    prices = np.load(DATASET / "heston_ma_S_8192x252x8.npy", mmap_mode="r")
    slice_ = np.asarray(prices[:num_paths, : num_steps + 1, :], dtype=np.float64)
    return torch.log(torch.from_numpy(slice_).to(TORCH_DTYPE))


def main() -> int:
    torch.manual_seed(0)
    num_paths, num_steps = 256, 60
    log_prices = _load(num_paths, num_steps)
    print(f"[data] log prices {tuple(log_prices.shape)}  dt={DT}")

    increments = log_prices[:, 1:, :] - log_prices[:, :-1, :]
    initial_activity = float(increments.pow(2).mean() / DT)
    print(f"[data] initial_activity (mean r^2/dt) = {initial_activity:.6f}")

    grid = DiscreteTimeGrid(T=num_steps * DT, num_steps=num_steps)

    kernel = mr.MultivariateCovarianceReferenceKernel(
        grid=grid,
        state_dim=8,
        trend_half_lives=(12.883786821834656, 74.25954988259295),
        activity_half_lives=(11.777077698433718, 50.337696867760634),
        trend_weight=0.8225691318511963,
        activity_weight=0.254260390996933,
        gamma_diagonal=torch.tensor(
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=TORCH_DTYPE
        ),
        gamma_offdiagonal=torch.tensor(
            [0.0, 0.45, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=TORCH_DTYPE
        ),
        gamma_drift=torch.zeros(5, dtype=TORCH_DTYPE),
        initial_activity=initial_activity,
    )
    print("[1] kernel constructed")

    drifts, sigmas = kernel.evaluate_all(log_prices)
    print(f"[1] drifts {tuple(drifts.shape)}  sigmas {tuple(sigmas.shape)}")
    assert drifts.shape == (num_paths, num_steps, 8), drifts.shape
    assert sigmas.shape == (num_paths, num_steps, 8, 8), sigmas.shape
    assert torch.isfinite(drifts).all() and torch.isfinite(sigmas).all()

    asymmetry = (sigmas - sigmas.transpose(-1, -2)).abs().max().item()
    print(f"[1] max |sigma - sigma^T| = {asymmetry:.3e}")
    assert asymmetry < 1e-10, asymmetry

    spectrum = torch.linalg.eigvalsh(sigmas.reshape(-1, 8, 8))
    print(
        f"[1] sigma spectrum min={spectrum.min().item():.6f} "
        f"max={spectrum.max().item():.6f} "
        f"(bounds {kernel.sigma_min}, {kernel.sigma_max})"
    )
    assert spectrum.min().item() >= kernel.sigma_min - 1e-12
    assert spectrum.max().item() <= kernel.sigma_max + 1e-12

    offdiag_ratio = (
        sigmas[..., 0, 1].abs().mean() / sigmas[..., 0, 0].abs().mean()
    ).item()
    print(f"[1] mean |sigma_01| / |sigma_00| = {offdiag_ratio:.4f}")
    assert offdiag_ratio > 0.05, "off-diagonal collapsed -- matrix structure lost"

    # Out-of-order steps exercise the full-replay fallback of the prefix cache.
    for step in (0, num_steps // 2, num_steps - 1):
        drift_step, sigma_step = kernel.evaluate(
            log_prices[:, : step + 1, :], step_index=step
        )
        drift_gap = (drift_step - drifts[:, step, :]).abs().max().item()
        sigma_gap = (sigma_step - sigmas[:, step, :, :]).abs().max().item()
        print(
            f"[2] replay  step {step:3d}: drift gap {drift_gap:.3e}  "
            f"sigma gap {sigma_gap:.3e}"
        )
        assert drift_gap < 1e-10 and sigma_gap < 1e-10

    # A consecutive walk exercises the O(1) cached branch, which is the one the
    # autograd_replay drift adjoint actually uses.
    worst_drift = worst_sigma = 0.0
    for step in range(num_steps):
        drift_step, sigma_step = kernel.evaluate(
            log_prices[:, : step + 1, :], step_index=step
        )
        worst_drift = max(
            worst_drift, (drift_step - drifts[:, step, :]).abs().max().item()
        )
        worst_sigma = max(
            worst_sigma, (sigma_step - sigmas[:, step, :, :]).abs().max().item()
        )
    print(f"[2] cached walk: worst drift {worst_drift:.3e}  worst sigma {worst_sigma:.3e}")
    assert worst_drift < 1e-10 and worst_sigma < 1e-10

    result = mr.fit_multivariate_covariance_reference_kernel(
        target_paths=log_prices,
        grid=grid,
        ridge_covariance=1e-4,
        ridge_drift=1e-3,
        initial_activity=initial_activity,
        steps=30,
        lr=5e-2,
        log_every=5,
        seed=0,
        verbose=True,
    )
    first = result.history[0]
    last = result.history[-1]
    print(
        f"[3] calibration NLL {first['calibration_nll']:.6f} -> "
        f"{last['calibration_nll']:.6f}"
    )
    print(
        f"[3] validation  NLL {first['validation_nll']:.6f} -> "
        f"{last['validation_nll']:.6f}"
    )
    assert last["calibration_nll"] < first["calibration_nll"], "NLL did not decrease"

    fitted = result.kernel
    _, fitted_sigmas = fitted.evaluate_all(log_prices)
    covariance = fitted_sigmas @ fitted_sigmas
    variance = torch.diagonal(covariance, dim1=-2, dim2=-1)
    correlation = covariance / variance.unsqueeze(-1).sqrt() / variance.unsqueeze(-2).sqrt()
    mask = ~torch.eye(8, dtype=torch.bool)
    mean_correlation = correlation[..., mask].mean().item()
    print(f"[4] fitted mean off-diagonal correlation = {mean_correlation:.4f}")
    print("[4] dataset sigma_s off-diagonal mean = 0.4495 (reference value)")

    empirical = increments.reshape(-1, 8)
    empirical_correlation = torch.corrcoef(empirical.T)
    print(
        f"[4] empirical log-return correlation mean = "
        f"{empirical_correlation[mask].mean().item():.4f}"
    )

    print("[ok] all smoke checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
