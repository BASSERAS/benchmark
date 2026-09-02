"""Validate and time ``sbts_reference.SBTSReferenceKernel`` before any training.

Three independent checks, none of which touches the validation or test split:

1. DRIFT vs an independent float64 numpy recomputation of the SBTS Markovian
   weights (direct product over the last K lags -- no rolling division, no log
   trick), on a prefix taken from the TRAIN split.  This is the check that the
   log-weight accumulation and the ``||a-b||^2`` expansion did not change the
   answer, and that the bank index shift (dummy zero row) is right.

2. JACOBIAN vs central finite differences of the drift.  The analytic Jacobian
   is what the custom ``autograd.Function`` sends backward, so if it is wrong
   the model trains on a wrong gradient silently.

3. TIMING + WEIGHT HEALTH at the real training shapes
   (B = 256, M = 8192, N = 251, d = 8), reporting seconds per full forward
   rollout and the fraction of (row, step) pairs whose weights all underflow to
   zero (drift forced to 0).  Reported, not assumed.

Run:
    /home/tbasseras/gpu-venv/bin/python validate_sbts_reference.py --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import sbts_reference as sr

DATASET = Path("/home/tbasseras/benchmark/dataset/HestonMultiAsset")
TRAIN = DATASET / "heston_ma_S_8192x252x8.npy"
DT = 1.0 / 252.0
H = 0.31
K = 20


class _Grid:
    """Minimal stand-in for ``DiscreteTimeGrid`` (only ``dt``/``num_steps`` used)."""

    def __init__(self, dt: float, num_steps: int) -> None:
        self.dt = float(dt)
        self.num_steps = int(num_steps)
        self.T = float(dt) * int(num_steps)


# --------------------------------------------------------------------------- #
# 1. independent numpy recomputation
# --------------------------------------------------------------------------- #
def numpy_drift(
    bank_scaled: np.ndarray,
    bank_raw: np.ndarray,
    prefix_returns_scaled: np.ndarray,
    *,
    step_index: int,
    h: float,
    markov_order: int,
    dt: float,
) -> np.ndarray:
    """``b^ref`` for ONE prefix, by direct float64 product of the kernels.

    ``prefix_returns_scaled`` is ``(step_index, d)``: rt_1 .. rt_{step_index}.
    """

    lags = min(int(step_index), int(markov_order))
    if lags == 0:
        return bank_raw[:, step_index, :].mean(axis=0) / dt

    weights = np.ones(bank_scaled.shape[0], dtype=np.float64)
    for j in range(lags):
        date = step_index - lags + 1 + j          # l
        diff = bank_scaled[:, date - 1, :] - prefix_returns_scaled[date - 1]
        norm = np.sqrt((diff ** 2).sum(axis=1))
        weights *= np.where(norm < h, (h ** 2 - norm ** 2) ** 2, 0.0)

    total = weights.sum()
    if total <= 0.0:
        return np.zeros(bank_raw.shape[-1], dtype=np.float64)
    return (weights[:, None] * bank_raw[:, step_index, :]).sum(axis=0) / total / dt


def check_drift(prices: np.ndarray, *, n_bank: int, batch: int, steps: tuple[int, ...]):
    grid = _Grid(DT, int(prices.shape[1]) - 1)

    kernel = sr.build_sbts_reference_kernel(
        train_prices=torch.from_numpy(prices[:n_bank]),
        grid=grid,
        h=H,
        markov_order=K,
        weight_grad_mode="detached",
        device="cpu",
        dtype=torch.float64,
    )
    bank_scaled = kernel.bank_scaled.numpy()
    bank_raw = kernel.bank_raw.numpy()
    sigma_assets = kernel.sigma_assets.numpy()

    # Prefixes: log-prices of bank paths that are NOT the first n_bank, so the
    # conditioning is not a trivial self-match.
    log_prices = np.log(prices[n_bank : n_bank + batch])
    x_prefix_full = torch.from_numpy(log_prices)

    worst = 0.0
    sigma_err = 0.0
    rows = []
    for step in steps:
        drift_t, sigma_t = kernel.evaluate(
            x_prefix_full[:, : step + 1, :], step_index=step
        )
        prefix_returns = (
            np.diff(log_prices[:, : step + 1, :], axis=1) * math.sqrt(DT) / sigma_assets
        )  # (batch, step, d)
        for b in range(batch):
            expected = numpy_drift(
                bank_scaled,
                bank_raw,
                prefix_returns[b],
                step_index=step,
                h=H,
                markov_order=K,
                dt=DT,
            )
            got = drift_t[b].numpy()
            scale = max(float(np.abs(expected).max()), 1e-12)
            worst = max(worst, float(np.abs(got - expected).max() / scale))
        sigma_expected = np.diag(sigma_assets / math.sqrt(DT))
        sigma_err = max(sigma_err, float(np.abs(sigma_t[0].numpy() - sigma_expected).max()))
        rows.append((step, float(np.abs(drift_t.numpy()).max())))

    return worst, sigma_err, rows, kernel


def check_jacobian(kernel, prices: np.ndarray, *, n_bank: int, batch: int, step: int):
    """Central finite differences of ``b^ref`` w.r.t. EVERY state it reads.

    ``b^ref_i`` reads the last ``K`` returns, so it depends on ``K + 1`` states,
    not two.  Perturbing only the newest state and comparing against the newest
    block alone would pass even if the other ``K - 1`` blocks were garbage --
    which is exactly the bug this check now has to be able to see.

    So we assemble the analytic derivative the same way ``_SBTSDrift.backward``
    does, by scattering each per-lag block onto the two states that bracket its
    return,

        d b / d x_window[t] = jac[t - 1] - jac[t]

    (each term present only when in range), and finite-difference every one of
    the ``K + 1`` states.  That validates the per-lag Jacobians and the
    accumulation together; a sign error or an off-by-one in the scatter shows up
    here and nowhere else.
    """

    log_prices = np.log(prices[n_bank : n_bank + batch, : step + 1, :])
    x = torch.from_numpy(log_prices).clone()
    d = int(x.shape[-1])
    lags = min(step, K)

    returns = kernel._scaled_returns_of_prefix(x, lags=lags)
    log_w = kernel._log_weights(returns, step_index=step, lags=lags)
    p, alive = kernel._normalised_weights(log_w)
    target_raw = kernel.bank_raw[:, step, :]
    jac = kernel._drift_jacobian(
        returns=returns, p=p, alive=alive, target_raw=target_raw, step_index=step
    ).numpy()  # (B, J, d, d) = d b_a / d rt_j, chained by d rt / d x

    n_jac = jac.shape[1]
    # (B, J + 1, d, d): total derivative w.r.t. each state of the window.
    analytic = np.zeros((jac.shape[0], n_jac + 1, d, d), dtype=jac.dtype)
    analytic[:, 1:, :, :] += jac
    analytic[:, :-1, :, :] -= jac

    eps = 1e-7
    numeric = np.zeros_like(analytic)
    for t in range(n_jac + 1):
        col = x.shape[1] - (n_jac + 1) + t  # window position -> prefix column
        for c in range(d):
            for sign in (+1, -1):
                xp = x.clone()
                xp[:, col, c] += sign * eps
                drift, _ = kernel.evaluate(xp, step_index=step)
                numeric[:, t, :, c] += sign * drift.numpy() / (2 * eps)

    scale = max(float(np.abs(numeric).max()), 1e-9)
    return float(np.abs(analytic - numeric).max() / scale), float(np.abs(analytic).max())


# --------------------------------------------------------------------------- #
# 3. timing + weight health at production shapes
# --------------------------------------------------------------------------- #
def time_rollout(prices: np.ndarray, *, device: str, batch: int, npi: int, dtype):
    grid = _Grid(DT, int(prices.shape[1]) - 1)
    kernel = sr.build_sbts_reference_kernel(
        train_prices=torch.from_numpy(prices),
        grid=grid,
        h=H,
        markov_order=K,
        npi=npi,
        weight_grad_mode="detached",
        device=device,
        dtype=dtype,
    )
    dev = torch.device(device)
    generator = torch.Generator(device="cpu").manual_seed(0)

    # Walk forward with the reference dynamics so the prefix distribution is
    # realistic; a frozen real path would make every weight vector identical
    # across the batch and hide the real cost.
    x = torch.full((batch, 1, kernel.state_dim), math.log(100.0), dtype=dtype, device=dev)
    dead = 0
    total_rows = 0

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(grid.num_steps):
        drift, sig = kernel.evaluate(x, step_index=step)
        noise = torch.randn(
            batch, kernel.state_dim, generator=generator, dtype=dtype
        ).to(dev)
        nxt = x[:, -1, :] + drift * DT + noise @ sig[0].t() * math.sqrt(DT)
        x = torch.cat((x, nxt.unsqueeze(1)), dim=1)
        dead += int((drift.abs().sum(dim=1) == 0).sum().item())
        total_rows += batch
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    peak = (
        torch.cuda.max_memory_allocated() / 2**30 if device.startswith("cuda") else 0.0
    )
    return dict(
        device=device,
        batch=batch,
        npi=npi,
        dtype=str(dtype),
        seconds_per_forward=round(elapsed, 3),
        dead_row_fraction=round(dead / max(total_rows, 1), 6),
        peak_gib=round(peak, 3),
        final_log_price_std=round(float(x[:, -1, :].double().std().item()), 5),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--out", default="../losses/reference_validation.json")
    args = parser.parse_args()

    prices = np.load(TRAIN)
    print(f"[data] {TRAIN.name} {prices.shape} {prices.dtype}", flush=True)

    print("[1/3] drift vs independent float64 numpy recomputation ...", flush=True)
    rel, sigma_err, rows, kernel = check_drift(
        prices, n_bank=512, batch=4, steps=(0, 1, 5, 19, 20, 21, 120, 250)
    )
    print(f"      max relative drift error = {rel:.3e}")
    print(f"      max sigma^ref error      = {sigma_err:.3e}")
    for step, mx in rows:
        print(f"      step {step:4d}  max|b^ref| = {mx:.4f}")

    print("[2/3] analytic Jacobian vs central finite differences ...", flush=True)
    jac_rel, jac_mag = check_jacobian(kernel, prices, n_bank=512, batch=4, step=120)
    print(f"      max relative Jacobian error = {jac_rel:.3e}  (|J|max = {jac_mag:.3e})")

    report = {
        "drift_max_rel_error": rel,
        "sigma_max_abs_error": sigma_err,
        "jacobian_max_rel_error": jac_rel,
        "jacobian_max_abs": jac_mag,
        "h": H,
        "K": K,
        "dt": DT,
        "timings": [],
    }

    if not args.skip_timing:
        print("[3/3] timing at production shapes ...", flush=True)
        for npi in (1, 5, 50):
            try:
                stats = time_rollout(
                    prices,
                    device=args.device,
                    batch=args.batch,
                    npi=npi,
                    dtype=torch.float32,
                )
            except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover
                stats = {"npi": npi, "error": f"OOM: {exc}"}
            print("      " + json.dumps(stats), flush=True)
            report["timings"].append(stats)
            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()

    out = (Path(__file__).resolve().parent / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
