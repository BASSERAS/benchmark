"""Independent finite-difference proof that the adjoint is ``d b^ref / d x_j``.

Why this exists.  The analytic Jacobian is derived w.r.t. the SCALED RETURNS
``rt_l``, then chained to the states ``x_j`` in two separate places:

  * ``_drift_jacobian`` multiplies by ``d rt / d x = sqrt(dt) / sigma`` (the
    ``chain`` line), which converts a return-derivative into a state-derivative
    but says nothing about WHICH state;
  * ``_SBTSDrift.backward`` scatters each lag block onto the two states that
    bracket it, ``+g`` right and ``-g`` left, which is where the identity
    ``rt_l = (x_l - x_{l-1}) sqrt(dt)/sigma`` actually enters.

Either half can be right while the other is wrong, and the failure is silent:
a mis-scaled or mis-placed block still produces a finite, plausible gradient.
So this script never inspects the Jacobian.  It perturbs ONE COORDINATE OF ONE
STATE at a time and compares the resulting change in ``b^ref`` against what
autograd claims, i.e. it tests exactly the composite object that training uses.

Central differences in float64, one column of the full ``d b / d x`` at a time.
The check covers EVERY state in the prefix, not just the K that carry a lag --
a state older than the window must come back with a gradient of exactly zero,
and confirming that is what distinguishes "the adjoint is complete" from "the
adjoint is large".

Run:
    /home/tbasseras/gpu-venv/bin/python probe_dbref_dx.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import sbts_reference as sr

DATASET = Path("/home/tbasseras/benchmark/dataset/HestonMultiAsset")
TRAIN = DATASET / "heston_ma_S_8192x252x8.npy"
DT = 1.0 / 252.0
H = 0.36
K = 20
BANK_PATHS = 2048
BATCH = 4
# A prefix longer than K, so the test also sees states the drift must IGNORE.
STEP = 26
EPS = 1e-6


class _Grid:
    def __init__(self, dt: float, num_steps: int) -> None:
        self.dt = float(dt)
        self.num_steps = int(num_steps)
        self.T = float(dt) * int(num_steps)


def _build(jacobian_lags: int, prices: np.ndarray, grid: _Grid):
    return sr.build_sbts_reference_kernel(
        train_prices=torch.from_numpy(prices),
        grid=grid,
        h=H,
        markov_order=K,
        npi=1,
        weight_grad_mode="analytic",
        device="cpu",
        dtype=torch.float64,
        jacobian_lags=jacobian_lags,
    )


def _analytic(kernel, x0: torch.Tensor, seed_vec: torch.Tensor) -> torch.Tensor:
    """``d (v . b^ref) / d x`` from autograd, shape ``(B, STEP+1, d)``."""

    x = x0.clone().requires_grad_(True)
    drift, _ = kernel.evaluate(x, step_index=STEP)
    (drift * seed_vec).sum().backward()
    assert x.grad is not None
    return x.grad.detach().clone()


def _numeric(kernel, x0: torch.Tensor, seed_vec: torch.Tensor) -> torch.Tensor:
    """The same object by central differences on the FORWARD pass only."""

    out = torch.zeros_like(x0)
    n_t, n_d = int(x0.shape[1]), int(x0.shape[2])
    for t in range(n_t):
        for c in range(n_d):
            xp = x0.clone()
            xp[:, t, c] += EPS
            xm = x0.clone()
            xm[:, t, c] -= EPS
            with torch.no_grad():
                bp, _ = kernel.evaluate(xp, step_index=STEP)
                bm, _ = kernel.evaluate(xm, step_index=STEP)
            out[:, t, c] = ((bp - bm) * seed_vec).sum(dim=1) / (2.0 * EPS)
    return out


def _report(name: str, ana: torch.Tensor, num: torch.Tensor) -> None:
    a = ana.reshape(ana.shape[0], -1)
    n = num.reshape(num.shape[0], -1)
    denom = n.norm(dim=1).clamp_min(1e-30)
    rel = ((a - n).norm(dim=1) / denom).numpy()

    # Which STATES receive a gradient, analytically and numerically.
    ana_t = (ana.abs().sum(dim=(0, 2)) > 1e-12).numpy()
    num_t = (num.abs().sum(dim=(0, 2)) > 1e-9).numpy()
    n_t = int(ana.shape[1])

    print(f"  {name}")
    print(f"     relative error per row   max={rel.max():.3e}  med={np.median(rel):.3e}")
    print(f"     states with grad         analytic {int(ana_t.sum())}/{n_t}"
          f"   numeric {int(num_t.sum())}/{n_t}")
    if not np.array_equal(ana_t, num_t):
        miss = np.where(num_t & ~ana_t)[0]
        extra = np.where(ana_t & ~num_t)[0]
        if miss.size:
            print(f"     MISSING (true grad, adjoint says 0): t={miss.tolist()}")
        if extra.size:
            print(f"     SPURIOUS (adjoint invents a grad):   t={extra.tolist()}")
    else:
        print("     support MATCHES exactly")


def main() -> None:
    all_prices = np.load(TRAIN)
    grid = _Grid(DT, int(all_prices.shape[1]) - 1)

    # Bank and queries disjoint: a query that is its own bank member sits at
    # distance 0, dominates every weight, and makes the covariance degenerate --
    # a finite-difference test would then pass on an object nobody trains on.
    prices = np.ascontiguousarray(all_prices[:BANK_PATHS])
    queries = np.ascontiguousarray(
        all_prices[BANK_PATHS : BANK_PATHS + BATCH, : STEP + 1]
    )
    x0 = torch.from_numpy(np.log(queries)).to(torch.float64)

    g = torch.Generator().manual_seed(0)
    # Contract b^ref against a fixed random covector, so the test exercises
    # every OUTPUT component at once instead of only component 0.
    seed_vec = torch.randn(1, int(x0.shape[2]), generator=g, dtype=torch.float64)

    print(f"[setup] step={STEP} K={K} h={H} bank={BANK_PATHS} batch={BATCH} "
          f"eps={EPS} float64 CPU")
    print(f"[setup] prefix has {STEP + 1} states; only the last {K + 1} can matter")
    print()

    print("=" * 72)
    print("d b^ref / d x  : autograd vs central differences")
    print("=" * 72)

    for jl, label in ((-1, "jacobian_lags = -1  (all K lags -- what now runs)"),
                      (1, "jacobian_lags =  1  (newest lag only -- the old bug)")):
        kernel = _build(jl, prices, grid)
        ana = _analytic(kernel, x0, seed_vec)
        num = _numeric(kernel, x0, seed_vec)
        _report(label, ana, num)
        print()

    print("reading: relative error ~1e-9 or below means the adjoint IS the")
    print("derivative of b^ref with respect to the STATES x_j.  A truncated")
    print("adjoint has relative error ~1 because it drops K-1 of the K blocks.")


if __name__ == "__main__":
    main()
