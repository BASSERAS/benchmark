"""Fit and FREEZE the multivariate reference kernel for ``TrueDataset``.

Port of ``results/HestonMultiAsset/Deep-MKV-TS/code/fit_reference_multiasset.py``.
The exported surface (``DATASET``, ``DT``, ``STATE_DIM``, ``load_kernel``,
``load_log_prices``, ``kernel_to_dict``) is kept name-for-name so the two method
trees stay diffable; everything that differs is a dataset fact, not a taste.

Algorithm 1 consumes the reference as two fixed objects: the drift ``b^ref`` used
verbatim in the Euler step, and ``sigma^ref`` which enters
``Theta = eta (sigma^ref)^{-1} + Z / sqrt(dt)``.  Neither is learned inside the
Algorithm-1 loop, so the reference is calibrated ONCE here and frozen.

WHAT CHANGED FROM THE HESTON PORT, AND WHY
------------------------------------------
1.  ``DT = 9.512937595129376e-07`` years -- 30-second bars, 1,051,200 bars per
    year.  Heston used 1/252.  Carrying the Heston dt forward would inflate
    every volatility quantity by ``sqrt(1051200 / 252) ~ 64.6x``; the fit would
    still converge, on the wrong units.

2.  ``--sigma-max`` is REQUIRED and has NO default.  Heston froze it at 0.6.
    The crypto panel's annualised covariance has spectrum
    ``[0.231, 0.387, 0.481, 0.600, 0.648, 0.757, 0.804, 1.802]``: the top
    eigenvalue is 3x the Heston ceiling and 5 of 8 assets exceed 0.6
    individually.  ``spectral_sigma`` clamps into ``[sigma_min, sigma_max]``, so
    a ceiling of 0.6 makes the market volatility physically unreachable -- and,
    critically, the fit STILL CONVERGES and STILL reports a finite NLL.  A
    silent default is therefore the dangerous option; the caller must name the
    ceiling on the command line.  ``sigma_min = 1e-3`` is left at the Heston
    value: the smallest market eigenvalue is 0.231, three orders above it, so
    the lower clip is inactive.

3.  Heston's freeze guard ``validation_nll > -9.0`` is DELETED, not adapted.
    That number is ``8 x -1.1466``, the d = 1 Heston per-asset result -- it is a
    Heston constant and has no meaning on crypto at a different dt.  It is
    replaced by ``iid_gaussian_nll``: the NLL of the same Euler increments under
    a CONSTANT clipped covariance estimated from this split.  That is the
    honest question the path-dependent memories have to answer -- if the fitted
    kernel cannot beat a constant matrix, the memories are decoration and
    freezing it would be fabrication.  The baseline is computed from the data at
    hand under the SAME sigma_max, so it moves with the sweep instead of being
    a hardcoded relic.

4.  ``clip_saturation`` is new and is the quantity that SELECTS ``sigma_max``.
    It reports the fraction of the fitted sigma spectrum sitting on either clip
    bound.  A grid point whose high-clip fraction is non-negligible has censored
    the target, whatever its NLL says.  NLL alone cannot see this: censoring
    removes variance, which a likelihood can reward.

5.  ``--report-only`` / ``--json-out`` let ``sweep_reference_true.py`` score a
    grid point without writing the frozen artefact.  Only the selected point is
    ever frozen.

Split discipline
----------------
Only ``true_S_6144x128x8.npy`` -- the TRAIN split -- is opened here.  The fitter
carves its own 80/20 calibration/validation split out of those paths
(``validation_fraction``); that is model selection inside the train split, not a
peek at held-out data.  The ``_val_``, ``_valdisc_``, ``_disc_`` and ``_test_``
banks are never touched by this file.

Learning rate
-------------
``lr = 1e-3`` is inherited from the Heston sweep over
``{5e-2, 1e-2, 3e-3, 1e-3, 3e-4}``, where ``5e-2`` parked on a plateau and
``1e-2`` diverged.  It is a starting point here, NOT a verified optimum: the
loss surface moved with dt.  ``sweep_reference_true.py`` re-checks it.
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

DATASET = Path(
    "/home/tbasseras/benchmark/dataset/TrueDataset/variants/om_2022-07_N6144"
)
SEQ_TAG = "6144x128x8"
TRAIN_FILE = f"true_S_{SEQ_TAG}.npy"

# 30-second bars: 1,051,200 bars per year -> dt = 1 / 1051200.  Spelled out
# rather than written as a division so the artefact and the four mandatory
# --dt flags downstream can be byte-compared against this literal.
DT = 9.512937595129376e-07
STATE_DIM = 8
SIGMA_MIN = 1e-3

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

    ``mmap_mode`` keeps the 48 MB bank off the heap until the slice is taken.
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


def iid_gaussian_nll(
    log_prices: torch.Tensor, *, sigma_min: float, sigma_max: float
) -> float:
    """NLL of the Euler increments under a CONSTANT clipped covariance.

    This is the floor the path-dependent memories must beat.  Everything is done
    exactly as the fitter does it -- the same ``spectral_sigma`` clip, the same
    ``dt`` scaling, the same ``logdet + 0.5 r' Sigma^-1 r / dt`` form -- with the
    single difference that the covariance is one constant matrix instead of a
    function of the path.  Any gap is therefore attributable to the memories and
    to nothing else.
    """

    increments = log_prices[:, 1:, :] - log_prices[:, :-1, :]
    flat = increments.reshape(-1, int(increments.shape[2]))
    mean = flat.mean(dim=0, keepdim=True)
    centred = flat - mean
    covariance = (centred.T @ centred) / float(centred.shape[0]) / float(DT)
    sigma, _vectors, _values = mr.spectral_sigma(
        covariance.unsqueeze(0), sigma_min=float(sigma_min), sigma_max=float(sigma_max)
    )
    factor = torch.linalg.cholesky(sigma)
    solved = torch.cholesky_solve(
        centred.unsqueeze(-1), factor.expand(int(centred.shape[0]), -1, -1)
    ).squeeze(-1)
    log_determinant = (
        2.0 * torch.log(torch.diagonal(factor, dim1=-2, dim2=-1)).sum(dim=-1)
    )
    per_increment = log_determinant + 0.5 * solved.pow(2).sum(dim=-1) / float(DT)
    return float(per_increment.mean())


# cuSOLVER's `cusolverDnXsyevBatched` takes the batch count in a 16-bit field.
# Measured on this box (torch 2.13.0+cu130, CUDA 13.0, A100): a batch of 65535
# symmetric 8x8 float64 matrices decomposes fine, 65536 raises
# CUSOLVER_STATUS_INTERNAL_ERROR. Reproduced with synthetic SPD matrices on an
# idle GPU, so it is a library limit, not memory pressure and not our data.
# 32768 is the chunk the rest of this tree already uses (train_true.MAX_EIGH_BATCH).
MAX_EIGH_BATCH = 32768


def _eigvalsh_chunked(matrices: torch.Tensor) -> torch.Tensor:
    """``torch.linalg.eigvalsh`` over a flattened batch, in cuSOLVER-sized chunks.

    Eigenvalues of a batch are computed independently per matrix, so splitting
    the batch dimension is an identity, not an approximation: the returned
    tensor is bit-for-bit what a single call would give if cuSOLVER accepted the
    batch. Only the batch is chunked; every matrix is decomposed whole.
    """

    tail = matrices.shape[-2:]
    flat = matrices.reshape(-1, *tail)
    total = int(flat.shape[0])
    if total <= MAX_EIGH_BATCH:
        return torch.linalg.eigvalsh(flat).reshape(*matrices.shape[:-1])
    pieces = [
        torch.linalg.eigvalsh(flat[start : start + MAX_EIGH_BATCH])
        for start in range(0, total, MAX_EIGH_BATCH)
    ]
    return torch.cat(pieces, dim=0).reshape(*matrices.shape[:-1])


def clip_saturation(
    kernel: mr.MultivariateCovarianceReferenceKernel,
    log_prices: torch.Tensor,
    *,
    num_paths: int = 512,
) -> dict[str, float]:
    """Fraction of the fitted sigma spectrum pinned on either clip bound.

    THIS is the diagnostic that selects ``sigma_max``, not the NLL.  If the
    ceiling is below the market's top eigenvalue the fit converges anyway and
    reports a perfectly ordinary likelihood -- censoring removes variance, and a
    Gaussian likelihood is happy to be handed less variance.  The censoring is
    only visible as a pile-up of eigenvalues exactly on the bound.

    ``num_paths`` is a subsample: ``evaluate_sigmas_all`` materialises
    ``(N, T, d, d)``, which is 6144 x 127 x 8 x 8 x 8 bytes = 3.2 GiB at full N.
    """

    subset = log_prices[: int(num_paths)]
    with torch.no_grad():
        sigmas = kernel.evaluate_sigmas_all(subset)
        spectrum = _eigvalsh_chunked(sigmas.double())
    high = float((spectrum >= float(kernel.sigma_max) * (1.0 - 1e-9)).double().mean())
    low = float((spectrum <= float(kernel.sigma_min) * (1.0 + 1e-9)).double().mean())
    return {
        "clip_saturation_high": high,
        "clip_saturation_low": low,
        "spectrum_min": float(spectrum.min()),
        "spectrum_mean": float(spectrum.mean()),
        "spectrum_p99": float(torch.quantile(spectrum.reshape(-1), 0.99)),
        "spectrum_max": float(spectrum.max()),
        "saturation_num_paths": int(subset.shape[0]),
    }


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

    ``differentiable_sigma`` keeps the reference-volatility evaluation inside the
    autograd graph.  The fit never needs it, so it defaults to False; training
    turns it on so Algorithm 1's running-cost path source picks up the
    ``d(sigma_ref)/dx`` term.

    The ``dt`` tolerance is ``1e-18``, not the Heston ``1e-15``: this dataset's
    ``dt`` is ~9.5e-07, so 1e-15 would accept a relative error of 1e-9 and let a
    subtly wrong bar convention through.
    """

    payload = json.loads(Path(path).read_text())
    if int(payload["state_dim"]) != STATE_DIM:
        raise ValueError(
            f"artefact state_dim={payload['state_dim']}, expected {STATE_DIM}"
        )
    if abs(float(payload["dt"]) - DT) > 1e-18:
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
    # No default, on purpose -- see item 2 of the module docstring.  Inheriting
    # the Heston 0.6 would censor 5 of 8 assets and still converge.
    parser.add_argument(
        "--sigma-max",
        type=float,
        required=True,
        help="spectral clip ceiling (annualised vol units); NO default, "
        "the crypto panel's top eigenvalue is 1.802",
    )
    parser.add_argument("--sigma-min", type=float, default=SIGMA_MIN)
    parser.add_argument("--ridge-covariance", type=float, default=1e-4)
    parser.add_argument("--ridge-drift", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--saturation-paths", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTEFACT)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="score this grid point and exit without writing the frozen "
        "artefact or enforcing the freeze guard (used by sweep_reference_true)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="write the scorecard (NLLs + clip saturation) to this path",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))

    log_prices = load_log_prices(num_paths=args.num_paths, device=device)
    num_paths, num_dates, _ = log_prices.shape
    num_steps = int(num_dates) - 1
    print(
        f"[data] train log prices {tuple(log_prices.shape)}  dt={DT!r}  "
        f"num_steps={num_steps}  device={device}"
    )

    # The activity state needs a t=0 value before any return has been seen.
    # Seeding it with the pooled realised variance rate of the train split is
    # the only choice that does not smuggle in a free parameter.
    increments = log_prices[:, 1:, :] - log_prices[:, :-1, :]
    initial_activity = float(increments.pow(2).mean() / DT)
    print(f"[data] initial_activity (pooled mean r^2/dt) = {initial_activity:.8f}")

    baseline_nll = iid_gaussian_nll(
        log_prices, sigma_min=float(args.sigma_min), sigma_max=float(args.sigma_max)
    )
    print(
        f"[base] constant-covariance NLL = {baseline_nll:.6f}  "
        f"(sigma clip [{args.sigma_min:g}, {args.sigma_max:g}])"
    )

    grid = DiscreteTimeGrid(T=num_steps * DT, num_steps=num_steps)
    print(
        f"[fit] steps={args.steps} lr={args.lr} seed={args.seed} "
        f"ridge_cov={args.ridge_covariance:g} ridge_drift={args.ridge_drift:g} "
        f"sigma_max={args.sigma_max:g}"
    )

    result = mr.fit_multivariate_covariance_reference_kernel(
        target_paths=log_prices,
        grid=grid,
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
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
        f"validation NLL {result.validation_nll:.6f}  "
        f"(baseline {baseline_nll:.6f}, gain {baseline_nll - result.validation_nll:+.6f})"
    )

    saturation = clip_saturation(
        kernel, log_prices, num_paths=int(args.saturation_paths)
    )
    print(
        "[clip] on-ceiling {clip_saturation_high:.4%}  on-floor "
        "{clip_saturation_low:.4%}  spectrum [{spectrum_min:.4f}, "
        "{spectrum_max:.4f}] mean {spectrum_mean:.4f} p99 "
        "{spectrum_p99:.4f}".format(**saturation)
    )

    scorecard: dict[str, object] = {
        "sigma_min": float(args.sigma_min),
        "sigma_max": float(args.sigma_max),
        "ridge_covariance": float(args.ridge_covariance),
        "ridge_drift": float(args.ridge_drift),
        "lr": float(args.lr),
        "steps": int(args.steps),
        "seed": int(args.seed),
        "num_paths": int(num_paths),
        "num_steps": int(num_steps),
        "initial_activity": initial_activity,
        "baseline_nll": baseline_nll,
        "calibration_nll": float(result.calibration_nll),
        "validation_nll": float(result.validation_nll),
        "validation_gain_over_baseline": baseline_nll - float(result.validation_nll),
        **saturation,
    }
    if args.json_out is not None:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(scorecard, indent=2))
        print(f"[out] {args.json_out}")

    if not np.isfinite(result.validation_nll):
        raise RuntimeError("validation NLL is not finite; refusing to freeze")

    if args.report_only:
        print("[report-only] not writing the frozen artefact")
        return 0

    # Heston compared against a hardcoded -9.0 (its own d = 1 result x 8).  That
    # constant means nothing here.  The question that DOES transfer is whether
    # the path-dependent memories earn their keep against a constant covariance
    # fitted to the same data under the same clip.
    if result.validation_nll >= baseline_nll:
        raise RuntimeError(
            f"validation NLL {result.validation_nll:.6f} is no better than the "
            f"constant-covariance baseline {baseline_nll:.6f}; the memories are "
            "not earning their keep -- refusing to freeze"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": str(DATASET),
        "seq_tag": SEQ_TAG,
        "train_file": TRAIN_FILE,
        "dt": DT,
        "steps": int(args.steps),
        "lr": float(args.lr),
        "seed": int(args.seed),
        "num_paths": int(num_paths),
        "num_steps": int(num_steps),
        "validation_fraction": float(args.validation_fraction),
        "baseline_nll": baseline_nll,
        "validation_gain_over_baseline": baseline_nll - float(result.validation_nll),
        **saturation,
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
    if float(reloaded.sigma_max) != float(args.sigma_max):
        raise RuntimeError("round-trip lost sigma_max")
    print("[check] artefact round-trips bit-exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
