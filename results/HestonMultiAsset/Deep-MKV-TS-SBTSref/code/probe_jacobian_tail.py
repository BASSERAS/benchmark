"""Measure the magnitude distribution of ``d b^ref / d rt`` on real prefixes.

Why this exists.  After the adjoint was corrected to carry all ``K`` lags, the
training loss started swinging over four orders of magnitude between logged
steps while the *objective* stayed flat.  A loss that moves like that with a
stable forward pass is a property of the TARGET, and the target is built from
the adjoint, so the adjoint magnitude is the thing to measure.

The suspicion is a pole.  The SBTS kernel is ``Kh(u) = (h^2 - ||u||^2)^2`` on
``||u|| < h``, so

    d log Kh / du = -4 u / (h^2 - ||u||^2)

which diverges as a bank path approaches the kernel edge ``||u|| -> h``.  The
weight ``p_m`` of such a path vanishes like ``(h^2 - ||u||^2)^2``, so the
PRODUCT ``p * g`` stays finite -- the pole is integrable and each individual
lag is well behaved.  What is not obviously bounded is the sum over ``K = 20``
lags once only a handful of bank paths survive the joint support.

So this script reports, on prefixes drawn from the TRAIN split:

  * ``n_eff = 1 / sum_m p_m^2``, the effective number of contributing bank
    paths -- the occupancy that the ``M p^K`` model predicts;
  * the per-lag Frobenius norm of the Jacobian block, and its distribution
    across (row, step) pairs, with the upper tail spelled out;
  * the ratio ``||sum of all K blocks|| / ||newest block||``, i.e. exactly how
    much bigger the corrected adjoint is than the truncated one it replaced.

Nothing here is a fix.  It is the measurement that says whether a fix (a norm
cap on the Jacobian, a floor on ``h^2 - ||u||^2``, or a smaller learning rate)
is even addressing the right quantity.

Run:
    /home/tbasseras/gpu-venv/bin/python probe_jacobian_tail.py
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
STEPS = (25, 60, 120, 180, 250)
BATCH = 256
# Bank size.  Held-out query paths are taken from beyond this index, so the bank
# and the queries never overlap; see the note in main().
BANK_PATHS = 4096
# Occupancy is swept over h and K separately, because the question "is n_eff = 1
# an artefact of the chosen h" cannot be answered at a single h.
H_GRID = (0.36, 0.5, 0.7, 1.0, 1.5, 2.0)
K_GRID = (1, 2, 5, 10, 20)


class _Grid:
    def __init__(self, dt: float, num_steps: int) -> None:
        self.dt = float(dt)
        self.num_steps = int(num_steps)
        self.T = float(dt) * int(num_steps)


def _pct(x: np.ndarray, qs=(50, 90, 99, 99.9, 100)) -> str:
    if x.size == 0:
        return "(empty)"
    return "  ".join(f"p{q}={np.percentile(x, q):.4g}" for q in qs)


def main() -> None:
    all_prices = np.load(TRAIN)
    grid = _Grid(DT, int(all_prices.shape[1]) - 1)

    # THE BANK AND THE QUERIES MUST BE DISJOINT.
    #
    # A first version of this probe built the bank from every path and then drew
    # the query prefixes from that same array.  Every query was therefore its own
    # nearest neighbour at distance exactly zero, so it survived any bandwidth --
    # even h = 0.05, where a genuine 20-lag match in 8 dimensions is impossible --
    # and it dominated the weights.  That reported ``n_eff = 1`` at every h and
    # was read as "the kernel has collapsed to a nearest neighbour lookup", which
    # was an artefact of the probe, not a property of the kernel.  With the split
    # below the same measurement gives n_eff = 12.7 at h = 0.36, i.e. a small but
    # perfectly real average.
    #
    # The split also matches training: there the prefixes are MODEL ROLLOUTS,
    # which are never bank members, so self-matching cannot occur.  If anything
    # these numbers stay optimistic, because a rollout drifts further off the
    # bank manifold than a held-out real path does.
    prices = np.ascontiguousarray(all_prices[:BANK_PATHS])
    queries = np.ascontiguousarray(all_prices[BANK_PATHS : BANK_PATHS + BATCH])

    kernel = sr.build_sbts_reference_kernel(
        train_prices=torch.from_numpy(prices),
        grid=grid,
        h=H,
        markov_order=K,
        npi=1,
        weight_grad_mode="analytic",
        device="cpu",
        dtype=torch.float64,
        jacobian_lags=-1,
    )

    x_full = torch.from_numpy(np.log(queries))

    print(f"[data] {TRAIN.name} {prices.shape}  h={H} K={K} dt={DT:.8f}")
    print(f"[note] bank M={kernel.bank_scaled.shape[0]}  batch={BATCH}  float64 CPU")
    print()

    for step in STEPS:
        lags = min(step, K)
        x = x_full[:, : step + 1, :]

        returns = kernel._scaled_returns_of_prefix(x, lags=lags)
        log_w = kernel._log_weights(returns, step_index=step, lags=lags)
        p, alive = kernel._normalised_weights(log_w)
        target_raw = kernel.bank_raw[:, step, :]

        jac = kernel._drift_jacobian(
            returns=returns, p=p, alive=alive,
            target_raw=target_raw, step_index=step,
        )  # (B, J, d, d), oldest lag first

        n_eff = (1.0 / p.pow(2).sum(dim=1).clamp(min=1e-300)).numpy()
        live = alive.numpy()
        n_eff = n_eff[live]

        # per-lag block norms, and the two adjoints being compared
        per_lag = jac.norm(dim=(-2, -1)).numpy()            # (B, J)
        newest = jac[:, -1].norm(dim=(-2, -1)).numpy()      # truncated adjoint
        summed = jac.sum(dim=1).norm(dim=(-2, -1)).numpy()  # corrected adjoint

        ratio = summed[live] / np.maximum(newest[live], 1e-30)

        print(f"step {step:3d}  lags={lags:2d}  alive={int(live.sum())}/{BATCH}")
        print(f"   n_eff bank paths   {_pct(n_eff)}")
        print(f"   ||J_lag||          {_pct(per_lag[live].ravel())}")
        print(f"   ||J_newest||       {_pct(newest[live])}")
        print(f"   ||sum_lags J||     {_pct(summed[live])}")
        print(f"   ratio sum/newest   {_pct(ratio)}")
        print()

    _occupancy_grid(prices, grid, x_full)
    _small_h_endpoint(prices, grid, x_full)


def _small_h_endpoint(prices, grid, x_full) -> None:
    """Where the kernel dies, at ``K = 20``.

    Below some bandwidth NO bank path lies inside the joint support of all
    ``K`` lags, every weight underflows, and the kernel returns ``b^ref = 0``.
    An arm trained there is not "a bad bandwidth", it is a model trained
    against a reference drift of exactly zero, and its score means nothing.
    This table is what makes that distinction visible instead of inferred.
    """

    step = 120
    print("=" * 72)
    print(f"small-h endpoint at step {step}, K=20: where does the kernel die?")
    print("=" * 72)
    print(f"{'h':>6} {'alive':>10} {'med n_eff':>11}  verdict")
    for h in (0.05, 0.10, 0.20, 0.31, 0.36):
        kern = sr.build_sbts_reference_kernel(
            train_prices=torch.from_numpy(prices),
            grid=grid, h=h, markov_order=20, npi=1,
            weight_grad_mode="analytic", device="cpu",
            dtype=torch.float64, jacobian_lags=-1,
        )
        returns = kern._scaled_returns_of_prefix(x_full[:, : step + 1, :], lags=20)
        p, alive = kern._normalised_weights(
            kern._log_weights(returns, step_index=step, lags=20)
        )
        n_alive = int(alive.sum())
        total = int(alive.numel())
        if n_alive == 0:
            print(f"{h:>6.2f} {f'{n_alive}/{total}':>10} {'--':>11}  DEAD: b^ref := 0 on every row")
            continue
        n_eff = (1.0 / p.pow(2).sum(dim=1).clamp(min=1e-300)).numpy()[alive.numpy()]
        med = float(np.median(n_eff))
        verdict = "1-NN lookup" if med < 1.1 else (
            "near-degenerate" if med < 10.0 else "averaging"
        )
        dead = total - n_alive
        extra = f" ({dead} rows have zero drift)" if dead else ""
        print(f"{h:>6.2f} {f'{n_alive}/{total}':>10} {med:>11.2f}  {verdict}{extra}")


def _occupancy_grid(prices, grid, x_full) -> None:
    """``n_eff`` across (h, K) at one mid-path step.

    The point is to separate two explanations of ``n_eff = 1``: either the
    bandwidth is simply too small, in which case raising ``h`` recovers a real
    average, or the joint support over ``K`` lags in ``d = 8`` dimensions is so
    thin that NO usable ``h`` gives one -- in which case ``b^ref`` is a nearest
    neighbour lookup by construction and no tuning changes that.
    """

    step = 120
    print("=" * 72)
    print(f"occupancy grid at step {step}: median n_eff (and % rows with n_eff<1.1)")
    print("=" * 72)
    header = "  K \\ h  " + "".join(f"{h:>11.2f}" for h in H_GRID)
    print(header)

    for k in K_GRID:
        cells = []
        for h in H_GRID:
            kern = sr.build_sbts_reference_kernel(
                train_prices=torch.from_numpy(prices),
                grid=grid, h=h, markov_order=k, npi=1,
                weight_grad_mode="analytic", device="cpu",
                dtype=torch.float64, jacobian_lags=-1,
            )
            lags = min(step, k)
            returns = kern._scaled_returns_of_prefix(x_full[:, : step + 1, :], lags=lags)
            log_w = kern._log_weights(returns, step_index=step, lags=lags)
            p, alive = kern._normalised_weights(log_w)
            n_eff = (1.0 / p.pow(2).sum(dim=1).clamp(min=1e-300)).numpy()[alive.numpy()]
            if n_eff.size == 0:
                cells.append("     DEAD  ")
                continue
            med = float(np.median(n_eff))
            frac = float((n_eff < 1.1).mean()) * 100.0
            cells.append(f" {med:7.1f}/{frac:3.0f}%")
        print(f"  {k:2d}     " + "".join(cells))

    print()
    print("reading: 'n_eff/pct' = median effective bank paths / share of rows")
    print("that are effectively one-hot (n_eff < 1.1).  DEAD = every weight")
    print("underflowed, drift forced to zero.")


if __name__ == "__main__":
    main()
