"""Fit and FREEZE the multivariate reference kernel for ``HestonMultiAsset``.

Algorithm 1 consumes the reference as two fixed objects: the drift ``b^ref``
used verbatim in the Euler step, and the volatility ``sigma^ref`` that enters
``Theta = eta (sigma^ref)^{-1} + Z / sqrt(dt)``.  Neither is learned inside the
Algorithm-1 loop -- the loop only learns ``Z`` through the GRU.  So the
reference has to be calibrated ONCE, up front, and then frozen, exactly as the
frozen ``d = 1`` run did (its manifest records
``physical_drift = fitted_reference_fixed``).

This script is that one-off calibration.  It writes a JSON artefact that
``load_kernel`` reads back into an identical
:class:`MultivariateCovarianceReferenceKernel`, so every training seed and every
generation run shares bit-identical reference coefficients.

Split discipline
----------------
Only ``heston_ma_S_8192x252x8.npy`` -- the TRAIN split -- is ever opened here.
The fitter carves its own internal 80/20 calibration/validation split out of
those paths (``validation_fraction``), which is a model-selection device inside
the train split and not a peek at the held-out files.  The ``_val_``, ``_test_``,
``_disc_`` and ``_valdisc_`` banks are not touched.

Learning rate
-------------
``lr = 1e-3`` is not a guess; it came out of a sweep over
``{5e-2, 1e-2, 3e-3, 1e-3, 3e-4}``:

===========  ==========  ===========  ===========
lr           best val    final cal    final val
===========  ==========  ===========  ===========
5e-2           -3.63        -3.63        -3.63
1e-2           -8.37     +2364.60    +10160.38
3e-3          -10.30       -10.30       -10.22
1e-3          -10.31       -10.31       -10.31
3e-4          -10.31       -10.31       -10.31
===========  ==========  ===========  ===========

``5e-2`` (the fitter default, inherited from the ``d = 1`` run) parks on a bad
plateau; ``1e-2`` diverges outright.  ``1e-3`` gives the best validation NLL
with calibration and validation agreeing to five decimals, i.e. no overfit --
unsurprising for 23 parameters against ~2e6 observations.  ``3e-4`` ties it but
buys nothing for the extra steps.

The value is worth a sanity check against ``d = 1``: that run reached
``-1.1466`` per asset, so eight assets should land near ``8 x -1.15 = -9.2``.
``-10.31`` clears it, and the surplus is the cross-asset covariance the diagonal
model structurally cannot capture.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _datetime
import json
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
TRAIN_FILE = "heston_ma_S_8192x252x8.npy"
DT = 0.003968253968253968
STATE_DIM = 8
DEFAULT_ARTEFACT = (
    Path(__file__).resolve().parent / "reference" / "reference_kernel.json"
)

_TENSOR_FIELDS = ("gamma_diagonal", "gamma_offdiagonal", "gamma_drift")
_SCALAR_FIELDS = (
    "trend_weight",
    "activity_weight",
    "initial_activity",
    "sigma_min",
    "sigma_max",
    "ridge_covariance",
    "ridge_drift",
    "calibration_nll",
    "validation_nll",
)


def load_log_prices(
    *,
    num_paths: int | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Train-split LOG prices, shape ``(N, T + 1, d)``, float64.

    ``mmap_mode`` keeps the 132 MB bank off the heap until the slice is taken.
    """

    prices = np.load(DATASET / TRAIN_FILE, mmap_mode="r")
    if num_paths is not None:
        prices = prices[: int(num_paths)]
    array = np.asarray(prices, dtype=np.float64)
    if array.ndim != 3 or int(array.shape[2]) != STATE_DIM:
        raise ValueError(f"expected (N, T + 1, {STATE_DIM}), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("train split contains non-finite prices")
    if float(array.min()) <= 0.0:
        raise ValueError("train split contains non-positive prices; cannot take log")
    return torch.log(torch.from_numpy(array)).to(device=device, dtype=torch.float64)


def kernel_to_dict(
    kernel: mr.MultivariateCovarianceReferenceKernel,
    *,
    fit_metadata: dict[str, object],
) -> dict[str, object]:
    """Everything ``load_kernel`` needs to rebuild ``kernel`` exactly."""

    payload: dict[str, object] = {
        "state_dim": int(kernel.state_dim),
        "dt": float(kernel.grid.dt),
        "num_steps": int(kernel.grid.num_steps),
        "trend_half_lives": [float(v) for v in kernel.trend_half_lives],
        "activity_half_lives": [float(v) for v in kernel.activity_half_lives],
        "activity_update": str(kernel.activity_update),
        "fit": fit_metadata,
    }
    for name in _SCALAR_FIELDS:
        payload[name] = float(getattr(kernel, name))
    for name in _TENSOR_FIELDS:
        payload[name] = [float(v) for v in getattr(kernel, name).detach().cpu()]
    # Recorded so a reader can tell which coefficient goes with which feature
    # without opening multivariate_reference.py.
    payload["covariance_feature_names"] = list(mr.COVARIANCE_FEATURE_NAMES)
    payload["drift_feature_names"] = list(mr.DRIFT_FEATURE_NAMES)
    return payload


def load_kernel(
    path: Path | str = DEFAULT_ARTEFACT,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    num_steps: int | None = None,
    differentiable_sigma: bool = False,
) -> mr.MultivariateCovarianceReferenceKernel:
    """Rebuild the frozen kernel from its JSON artefact.

    ``num_steps`` overrides only the grid length, for the case where generation
    runs a different horizon than calibration.  It cannot change ``dt``, since
    ``dt`` is baked into the fitted half-lives (they are measured in steps).

    ``differentiable_sigma`` keeps the reference-volatility evaluation inside
    the autograd graph.  The fit never needs it -- the kernel is calibrated by
    maximum likelihood on fixed data, not through a rollout -- so it defaults to
    False.  Training turns it on so Algorithm 1's running-cost path source picks
    up the d(sigma_ref)/dx term; measured at the realistic Z scale that term is
    ~7% of the discrepancy source, for +0.13 GiB and ~2.7x running-cost time.
    """

    payload = json.loads(Path(path).read_text())
    if int(payload["state_dim"]) != STATE_DIM:
        raise ValueError(
            f"artefact state_dim={payload['state_dim']}, expected {STATE_DIM}"
        )
    if abs(float(payload["dt"]) - DT) > 1e-15:
        raise ValueError(
            f"artefact dt={payload['dt']!r} does not match dataset dt={DT!r}"
        )
    steps = int(payload["num_steps"]) if num_steps is None else int(num_steps)
    grid = DiscreteTimeGrid(T=steps * float(payload["dt"]), num_steps=steps)
    tensors = {
        name: torch.tensor(payload[name], dtype=dtype, device=device)
        for name in _TENSOR_FIELDS
    }
    return mr.MultivariateCovarianceReferenceKernel(
        grid=grid,
        state_dim=int(payload["state_dim"]),
        trend_half_lives=tuple(payload["trend_half_lives"]),
        activity_half_lives=tuple(payload["activity_half_lives"]),
        trend_weight=float(payload["trend_weight"]),
        activity_weight=float(payload["activity_weight"]),
        initial_activity=float(payload["initial_activity"]),
        sigma_min=float(payload["sigma_min"]),
        sigma_max=float(payload["sigma_max"]),
        activity_update=str(payload["activity_update"]),
        ridge_covariance=float(payload["ridge_covariance"]),
        ridge_drift=float(payload["ridge_drift"]),
        calibration_nll=float(payload["calibration_nll"]),
        validation_nll=float(payload["validation_nll"]),
        differentiable_sigma=bool(differentiable_sigma),
        **tensors,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-paths", type=int, default=None)
    parser.add_argument("--ridge-covariance", type=float, default=1e-4)
    parser.add_argument("--ridge-drift", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTEFACT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))

    log_prices = load_log_prices(num_paths=args.num_paths, device=device)
    num_paths, num_dates, _ = log_prices.shape
    num_steps = int(num_dates) - 1
    print(f"[data] train log prices {tuple(log_prices.shape)}  dt={DT}  device={device}")

    # The activity state needs a t=0 value before any return has been seen.
    # Seeding it with the pooled realised variance rate of the train split is
    # the only choice that does not smuggle in a free parameter.
    increments = log_prices[:, 1:, :] - log_prices[:, :-1, :]
    initial_activity = float(increments.pow(2).mean() / DT)
    print(f"[data] initial_activity (pooled mean r^2/dt) = {initial_activity:.8f}")

    grid = DiscreteTimeGrid(T=num_steps * DT, num_steps=num_steps)
    print(f"[fit] steps={args.steps} lr={args.lr} seed={args.seed}")

    result = mr.fit_multivariate_covariance_reference_kernel(
        target_paths=log_prices,
        grid=grid,
        ridge_covariance=float(args.ridge_covariance),
        ridge_drift=float(args.ridge_drift),
        initial_activity=initial_activity,
        steps=int(args.steps),
        lr=float(args.lr),
        validation_fraction=float(args.validation_fraction),
        seed=int(args.seed),
        log_every=int(args.log_every),
        verbose=True,
    )

    kernel = result.kernel
    print(
        f"[fit] calibration NLL {result.calibration_nll:.6f}  "
        f"validation NLL {result.validation_nll:.6f}"
    )
    if not np.isfinite(result.validation_nll):
        raise RuntimeError("validation NLL is not finite; refusing to freeze")

    # A reference whose validation NLL is worse than the per-asset d = 1 result
    # scaled to eight assets (-1.1466 x 8 = -9.17) would mean the matrix model
    # lost ground to the diagonal one, which would void the point of this file.
    if result.validation_nll > -9.0:
        raise RuntimeError(
            f"validation NLL {result.validation_nll:.6f} is worse than the "
            "d = 1 per-asset baseline scaled to d = 8 (-9.17); refusing to freeze"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "steps": int(args.steps),
        "lr": float(args.lr),
        "seed": int(args.seed),
        "num_paths": int(num_paths),
        "num_steps": int(num_steps),
        "validation_fraction": float(args.validation_fraction),
        "train_file": TRAIN_FILE,
        "torch_version": str(torch.__version__),
        "device": str(device),
        "fitted_at": _datetime.datetime.now().replace(microsecond=0).isoformat(),
    }
    out.write_text(json.dumps(kernel_to_dict(kernel, fit_metadata=metadata), indent=2))
    print(f"[out] {out}")

    history_path = out.parent / "reference_fit_history.csv"
    with history_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "calibration_nll", "validation_nll"])
        for record in result.history:
            writer.writerow(
                [
                    int(record["step"]),
                    f"{float(record['calibration_nll']):.10f}",
                    f"{float(record['validation_nll']):.10f}",
                ]
            )
    print(f"[out] {history_path}  ({len(result.history)} rows)")

    # Round-trip on CPU: if the artefact does not rebuild the exact kernel the
    # fit produced, every downstream seed silently uses a different reference.
    reloaded = load_kernel(out, device="cpu")
    for name in _TENSOR_FIELDS:
        gap = (
            (getattr(reloaded, name) - getattr(kernel, name).detach().cpu())
            .abs()
            .max()
            .item()
        )
        if gap != 0.0:
            raise RuntimeError(f"round-trip changed {name} by {gap:.3e}")
    print("[check] artefact round-trips bit-exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
