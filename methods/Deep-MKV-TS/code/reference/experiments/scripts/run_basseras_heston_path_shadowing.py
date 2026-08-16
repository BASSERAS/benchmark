from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from deep_mkv_gen_path_dt import (  # noqa: E402
    DiscreteMPArchitectureConfig,
    DiscreteMPModel,
    DiscreteMPTrainingConfig,
    DiscreteTimeGrid,
    calibrate_shrunk_causal_reference_kernel,
    fit_causal_volatility_reference_kernel,
    fit_guyon_lekeufack_price_realized_volatility_reference_kernel,
    fit_local_gaussian_reference_kernel,
)
from deep_mkv_gen_path_dt.controls import (  # noqa: E402
    GaussianRelativeEntropyDiagonalControl,
    ReferenceDriftGaussianRelativeEntropyDiagonalControl,
    VolatilityOnlyGaussianRelativeEntropyDiagonalControl,
    ReferenceDriftSpecificEntropyDiagonalControl,
    SpecificEntropyDiagonalControl,
)
from deep_mkv_gen_path_dt.discrepancies import (  # noqa: E402
    CompositePathFunctionalDiscrepancy,
    PathFunctionalDiscrepancy,
    PathFunctionalDiscrepancyResult,
    WeightedDiscrepancy,
)
from deep_mkv_gen_path_dt.noise import derive_stream_seed  # noqa: E402
from path_dt_experiments.discrepancies import (  # noqa: E402
    build_heston_discrepancy,
    build_real_data_standardized_discrepancy,
    fit_fixed_target_joint_prefix_future_volatility_mmd,
)
from path_dt_experiments.dynamics import DriftVolatilityEulerStep  # noqa: E402
from path_dt_experiments.runners import (  # noqa: E402
    ensure_run_dir,
    write_history_jsonl,
    write_json,
)


DT = 1.0 / 250.0
S0 = 100.0
SEQ_LEN = 128
NUM_STEPS = SEQ_LEN - 1
BANK_SIZE = 1_000_000
BANK_FILENAME = f"generated_bank_seed0_{BANK_SIZE}x{SEQ_LEN}.npy"


@dataclass(frozen=True)
class SBTSLogReturnTransform:
    """The benchmark's train-fitted log-return transform, without an extra wrapper."""

    sigma: float
    dt: float = DT
    s0: float = S0

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.sigma)) or float(self.sigma) <= 0.0:
            raise ValueError("sigma must be a positive finite float")
        if not math.isfinite(float(self.dt)) or float(self.dt) <= 0.0:
            raise ValueError("dt must be a positive finite float")
        if not math.isfinite(float(self.s0)) or float(self.s0) <= 0.0:
            raise ValueError("s0 must be a positive finite float")

    @classmethod
    def fit(cls, prices: np.ndarray, *, dt: float = DT, s0: float = S0) -> "SBTSLogReturnTransform":
        prices64 = _validate_prices(prices)
        returns = np.log(prices64[:, 1:] / prices64[:, :-1])
        return cls(sigma=float(returns.std(ddof=0)), dt=float(dt), s0=float(s0))

    def forward(self, prices: np.ndarray) -> np.ndarray:
        prices64 = _validate_prices(prices)
        returns = np.log(prices64[:, 1:] / prices64[:, :-1])
        scaled = returns * math.sqrt(float(self.dt)) / float(self.sigma)
        return np.concatenate(
            [np.zeros((prices64.shape[0], 1), dtype=np.float64), scaled],
            axis=1,
        )

    def inverse(self, return_states: np.ndarray, *, dtype: np.dtype = np.float64) -> np.ndarray:
        values = np.asarray(return_states, dtype=np.float64)
        if values.ndim != 2 or int(values.shape[1]) != SEQ_LEN:
            raise ValueError(f"return_states must have shape (B, {SEQ_LEN})")
        if not np.isfinite(values).all():
            raise ValueError("return_states must be finite")
        raw_returns = values[:, 1:] * float(self.sigma) / math.sqrt(float(self.dt))
        prices = np.empty(values.shape, dtype=np.float64)
        prices[:, 0] = float(self.s0)
        prices[:, 1:] = float(self.s0) * np.exp(np.cumsum(raw_returns, axis=1))
        if not np.isfinite(prices).all() or np.any(prices <= 0.0):
            raise ValueError("inverse transform produced invalid prices")
        return prices.astype(dtype, copy=False)


@dataclass(frozen=True)
class CumulativeReturnPathDiscrepancy:
    """Evaluate a log-price-path discrepancy induced by scaled return states.

    The BASSERAS input sequence is ``[0, r_tilde_1, ..., r_tilde_127]``. Its
    cumulative sum, multiplied by the frozen inverse scale
    ``sigma / sqrt(dt)``, is the corresponding cumulative raw log-price path.
    Applying the discrepancy after this differentiable map preserves the
    exact benchmark input convention while making every path-law block act on
    financial returns, not on second differences or arbitrarily scaled paths.
    """

    discrepancy: PathFunctionalDiscrepancy
    path_scale: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.path_scale)) or float(self.path_scale) <= 0.0:
            raise ValueError("path_scale must be a positive finite float")

    def __call__(
        self,
        *,
        generated_paths: torch.Tensor,
        target_paths: torch.Tensor,
        grid: DiscreteTimeGrid,
    ) -> PathFunctionalDiscrepancyResult:
        return self.discrepancy(
            generated_paths=float(self.path_scale) * torch.cumsum(generated_paths, dim=1),
            target_paths=float(self.path_scale) * torch.cumsum(target_paths, dim=1),
            grid=grid,
        )


def _validate_prices(prices: np.ndarray) -> np.ndarray:
    values = np.asarray(prices, dtype=np.float64)
    if values.ndim != 2 or int(values.shape[1]) != SEQ_LEN:
        raise ValueError(f"prices must have shape (B, {SEQ_LEN})")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("prices must be finite and strictly positive")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Deep-MKV-Gen on a declared BASSERAS Heston split, build a "
            "price-path bank, and optionally invoke its PDF evaluator unchanged."
        )
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("/home/samer/scenarios/BASSERAS-benchmark"),
    )
    parser.add_argument(
        "--custom-dataset-root",
        type=Path,
        default=None,
        help=(
            "Optional directory containing train.npy, test.npy, and disc.npy. "
            "This preserves the benchmark adapter while allowing controlled "
            "synthetic-law experiments."
        ),
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--stage", choices=("all", "train", "bank", "evaluate"), default="all")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-layout",
        choices=("path_shadowing_4096", "benchmark_8192"),
        default="path_shadowing_4096",
        help=(
            "path_shadowing_4096 preserves the existing path-shadowing protocol; "
            "benchmark_8192 uses the A/B comparison kit's exact training split"
        ),
    )
    parser.add_argument(
        "--state-representation",
        choices=("log_price", "scaled_log_return"),
        default="log_price",
        help=(
            "log_price applies the benchmark transform and its fixed cumulative inverse "
            "before the MP dynamics; scaled_log_return evolves the transformed returns directly"
        ),
    )
    parser.add_argument("--source-training-steps", type=int, default=2_000)
    parser.add_argument("--joint-training-steps", type=int, default=1_600)
    parser.add_argument(
        "--joint-weight",
        type=float,
        default=1.0,
        help="Weight of the joint prefix/future realized-volatility discrepancy.",
    )
    parser.add_argument(
        "--time-horizon",
        type=float,
        default=NUM_STEPS * DT,
        help=(
            "Physical horizon represented by the 127 increments. The historical "
            "Heston benchmark default is 127/250; a normalized intraday session "
            "uses 1.0."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--target-batch-size", type=int, default=256)
    parser.add_argument("--sample-batch-size", type=int, default=50_000)
    parser.add_argument("--bank-size", type=int, default=BANK_SIZE)
    parser.add_argument(
        "--bank-seed",
        type=int,
        default=0,
        help="Base seed for generated paths; independent of the training seed.",
    )
    parser.add_argument(
        "--bank-output",
        type=Path,
        default=None,
        help=(
            "Optional exact .npy output path. By default the existing nested "
            "1M path-shadowing bank location and filename are preserved."
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lambda-scale", type=float, default=50.0)
    parser.add_argument("--kappa-scale", type=float, default=50.0)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--reference-ridge", type=float, default=1e-3)
    parser.add_argument("--reference-variance-shrinkage", type=float, default=0.1)
    parser.add_argument("--reference-log-variance-bias-correction", type=float, default=0.6)
    parser.add_argument(
        "--reference-kind",
        choices=("local", "calibrated_causal"),
        default="local",
        help="Volatility reference used by the specific-entropy running cost.",
    )
    parser.add_argument(
        "--causal-rolling-windows",
        type=int,
        nargs="+",
        default=(5, 10, 20, 32),
    )
    parser.add_argument(
        "--causal-ewma-half-lives",
        type=int,
        nargs="+",
        default=(5, 20, 60),
    )
    parser.add_argument(
        "--causal-downside-windows",
        type=int,
        nargs="+",
        default=(10, 32),
    )
    parser.add_argument(
        "--causal-variance-ridges",
        type=float,
        nargs="+",
        default=(1e-5, 1e-4, 1e-3),
    )
    parser.add_argument("--causal-crossfit-folds", type=int, default=3)
    parser.add_argument("--causal-crossfit-seed", type=int, default=1701)
    parser.add_argument(
        "--causal-calibration-weights",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.4, 0.55, 0.7, 0.75, 0.8, 0.85, 1.0),
    )
    parser.add_argument("--causal-calibration-paths", type=int, default=4096)
    parser.add_argument("--causal-calibration-seed", type=int, default=271828)
    parser.add_argument("--causal-calibration-offset-steps", type=int, default=3)
    parser.add_argument(
        "--causal-calibration-return-std-tolerance",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--causal-calibration-sigma-std-floor",
        type=float,
        default=0.045,
    )
    parser.add_argument(
        "--causal-calibration-sigma-cap-fraction-limit",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--causal-calibration-clustering-lags",
        type=int,
        nargs="+",
        default=(1, 5, 20),
    )
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--sigma-min", type=float, default=1e-3)
    parser.add_argument(
        "--sigma-max",
        type=float,
        default=0.6,
        help="Control/reference volatility cap in the selected MP state units.",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def _default_run_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "runs" / f"basseras_heston_specific_entropy_joint_seed0_{timestamp}"


def _data_paths(
    benchmark_root: Path,
    data_layout: str = "path_shadowing_4096",
    custom_dataset_root: Path | None = None,
) -> dict[str, Path]:
    if custom_dataset_root is not None:
        data_dir = Path(custom_dataset_root)
        return {
            "train": data_dir / "train.npy",
            "test": data_dir / "test.npy",
            "disc": data_dir / "disc.npy",
        }
    if str(data_layout) == "path_shadowing_4096":
        data_dir = (
            benchmark_root
            / "dataset"
            / "Heston"
            / "preprocessing_with_log_returns"
        )
        return {
            "train": data_dir / "heston_S_4096x128.npy",
            "test": data_dir / "heston_S_test_4096x128.npy",
            "disc": data_dir / "heston_S_disc_4096x128.npy",
            "ps": data_dir / "heston_S_ps_512x128.npy",
        }
    if str(data_layout) == "benchmark_8192":
        data_dir = benchmark_root / "dataset" / "Heston"
        return {
            "train": data_dir / "heston_S_8192x128.npy",
            "test": data_dir / "heston_S_test_8192x128.npy",
            "disc": data_dir / "heston_S_disc_8192x128.npy",
        }
    raise ValueError(f"unsupported data_layout {data_layout!r}")


def _benchmark_commit(benchmark_root: Path) -> str:
    head = benchmark_root / ".git" / "HEAD"
    if not head.exists():
        return "unknown"
    head_value = head.read_text(encoding="utf-8").strip()
    if head_value.startswith("ref: "):
        ref_path = benchmark_root / ".git" / head_value.removeprefix("ref: ")
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()
    return head_value


def _load_training_data(
    benchmark_root: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    state_representation: str = "log_price",
    data_layout: str = "path_shadowing_4096",
    custom_dataset_root: Path | None = None,
    model_dt: float = DT,
) -> tuple[torch.Tensor, SBTSLogReturnTransform, dict[str, float]]:
    paths = _data_paths(
        benchmark_root,
        data_layout,
        custom_dataset_root=custom_dataset_root,
    )
    train_prices = np.load(paths["train"])
    if not math.isfinite(float(model_dt)) or float(model_dt) <= 0.0:
        raise ValueError("model_dt must be a positive finite float")
    transform = SBTSLogReturnTransform.fit(train_prices, dt=float(model_dt))
    return_states = transform.forward(train_prices)
    reconstructed = transform.inverse(return_states)
    roundtrip_error = float(np.max(np.abs(reconstructed - train_prices)))
    if roundtrip_error > 1e-10:
        raise RuntimeError(f"benchmark preprocessing round trip failed: {roundtrip_error}")
    expected_sigmas = {
        "path_shadowing_4096": 0.0126316296,
        "benchmark_8192": 0.0126466634,
    }
    if custom_dataset_root is None:
        expected_sigma = expected_sigmas[str(data_layout)]
        if abs(float(transform.sigma) - expected_sigma) > 5e-9:
            raise RuntimeError(
                f"unexpected train sigma {transform.sigma:.10f}; "
                f"expected approximately {expected_sigma}"
            )
    if str(state_representation) == "log_price":
        model_states = (
            float(transform.sigma)
            / math.sqrt(float(transform.dt))
            * np.cumsum(return_states, axis=1)
        )
    elif str(state_representation) == "scaled_log_return":
        model_states = return_states
    else:
        raise ValueError("unsupported state_representation")
    tensor = torch.from_numpy(model_states).unsqueeze(-1).to(device=device, dtype=dtype)
    diagnostics = {
        "train_paths": float(train_prices.shape[0]),
        "sequence_length": float(train_prices.shape[1]),
        "pooled_log_return_sigma": float(transform.sigma),
        "scaled_return_std": float(return_states[:, 1:].std(ddof=0)),
        "expected_scaled_return_std": math.sqrt(float(model_dt)),
        "roundtrip_max_abs_error": roundtrip_error,
        "state_representation": str(state_representation),
        "model_dt": float(model_dt),
    }
    return tensor, transform, diagnostics


def _base_discrepancy() -> CompositePathFunctionalDiscrepancy:
    return build_heston_discrepancy(
        num_steps=NUM_STEPS,
        include_acf=True,
        preset="old_fullv_w0p25",
        lambda_scale=50.0,
        kappa_scale=50.0,
        include_conditional_volatility=False,
        include_conditional_volatility_correlation=False,
    )


def _make_training(args: argparse.Namespace, *, num_steps: int, seed: int) -> DiscreteMPTrainingConfig:
    clip = None if float(args.grad_clip_norm) <= 0.0 else float(args.grad_clip_norm)
    return DiscreteMPTrainingConfig(
        batch_size=int(args.batch_size),
        target_batch_size=int(args.target_batch_size),
        num_steps=int(num_steps),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        lambda_scale=float(args.lambda_scale),
        ce_target_mode="ridge",
        ridge_lambda=float(args.ridge_lambda),
        ce_crossfit_folds=1,
        target_preconditioner="none",
        noise_target_control_variate="none",
        grad_clip_norm=clip,
        log_every=int(args.log_every),
        seed=int(seed),
        observed_only=False,
    )


def _build_components(
    args: argparse.Namespace,
    *,
    target_paths: torch.Tensor,
    path_scale: float,
) -> tuple[
    DiscreteTimeGrid,
    DiscreteMPArchitectureConfig,
    SpecificEntropyDiagonalControl | GaussianRelativeEntropyDiagonalControl,
    CompositePathFunctionalDiscrepancy,
    PathFunctionalDiscrepancy,
]:
    time_horizon = float(getattr(args, "time_horizon", NUM_STEPS * DT))
    if not math.isfinite(time_horizon) or time_horizon <= 0.0:
        raise ValueError("time_horizon must be a positive finite float")
    grid = DiscreteTimeGrid(T=time_horizon, num_steps=NUM_STEPS)
    architecture = DiscreteMPArchitectureConfig(
        state_dim=1,
        noise_dim=1,
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
    )
    reference_sigma_max = float(
        getattr(args, "reference_sigma_max", args.sigma_max)
    )
    local_reference = fit_local_gaussian_reference_kernel(
        target_paths=target_paths,
        grid=grid,
        ridge=float(args.reference_ridge),
        variance_ridge=float(args.reference_ridge),
        variance_shrinkage=float(args.reference_variance_shrinkage),
        log_variance_bias_correction=float(args.reference_log_variance_bias_correction),
        sigma_min=float(args.sigma_min),
        sigma_max=reference_sigma_max,
    )
    reference_kind = str(getattr(args, "reference_kind", "local"))
    if reference_kind == "local":
        reference = local_reference
    elif reference_kind == "calibrated_causal":
        causal_reference = fit_causal_volatility_reference_kernel(
            target_paths=target_paths,
            grid=grid,
            rolling_windows=tuple(
                int(value) for value in args.causal_rolling_windows
            ),
            ewma_half_lives=tuple(
                int(value) for value in args.causal_ewma_half_lives
            ),
            downside_windows=tuple(
                int(value) for value in args.causal_downside_windows
            ),
            include_expanding_moments=bool(
                getattr(args, "causal_include_expanding_moments", False)
            ),
            persistence_lags=tuple(
                int(value)
                for value in getattr(args, "causal_persistence_lags", ())
            ),
            leverage_lags=tuple(
                int(value)
                for value in getattr(args, "causal_leverage_lags", ())
            ),
            local_volatility_windows=tuple(
                int(value)
                for value in getattr(
                    args,
                    "causal_local_volatility_windows",
                    (),
                )
            ),
            standardized_feature_clip=getattr(
                args,
                "causal_standardized_feature_clip",
                None,
            ),
            drift_ridge=float(args.reference_ridge),
            variance_ridges=tuple(
                float(value) for value in args.causal_variance_ridges
            ),
            crossfit_folds=int(args.causal_crossfit_folds),
            crossfit_seed=int(args.causal_crossfit_seed),
            sigma_min=float(args.sigma_min),
            sigma_max=reference_sigma_max,
        )
        reference, calibration = calibrate_shrunk_causal_reference_kernel(
            target_paths=target_paths,
            causal_reference=causal_reference,
            anchor_reference=local_reference,
            causal_weights=tuple(
                float(value) for value in args.causal_calibration_weights
            ),
            calibration_paths=int(args.causal_calibration_paths),
            calibration_seed=int(args.causal_calibration_seed),
            offset_steps=int(args.causal_calibration_offset_steps),
            offset_relaxation=float(
                getattr(args, "causal_calibration_offset_relaxation", 1.0)
            ),
            return_std_tolerance=float(
                args.causal_calibration_return_std_tolerance
            ),
            sigma_std_floor=float(
                args.causal_calibration_sigma_std_floor
            ),
            sigma_cap_fraction_limit=float(
                args.causal_calibration_sigma_cap_fraction_limit
            ),
            path_rv_std_ratio_min=float(
                getattr(args, "causal_calibration_path_rv_std_ratio_min", 0.0)
            ),
            path_rv_std_ratio_max=float(
                getattr(args, "causal_calibration_path_rv_std_ratio_max", math.inf)
            ),
            path_rv_quantile_spread_ratio_min=float(
                getattr(
                    args,
                    "causal_calibration_path_rv_quantile_spread_ratio_min",
                    0.0,
                )
            ),
            path_rv_quantile_spread_ratio_max=float(
                getattr(
                    args,
                    "causal_calibration_path_rv_quantile_spread_ratio_max",
                    math.inf,
                )
            ),
            clustering_lags=tuple(
                int(value)
                for value in args.causal_calibration_clustering_lags
            ),
            sigma_min=float(args.sigma_min),
            sigma_max=reference_sigma_max,
        )
        selected = calibration["selected"]
        print(
            "calibrated causal reference "
            f"weight={selected['causal_log_variance_weight']:.3f} "
            f"offset={selected['log_variance_offset']:.6f} "
            f"return_std={selected['return_std']:.6f} "
            f"sigma_std={selected['sigma_std']:.6f}",
            flush=True,
        )
    elif reference_kind == "guyon_lekeufack_price_realized_volatility":
        reference = fit_guyon_lekeufack_price_realized_volatility_reference_kernel(
            target_paths=target_paths,
            grid=grid,
            realized_volatility_window=int(
                getattr(args, "guyon_realized_volatility_window", 20)
            ),
            ridge=float(args.reference_ridge),
            variance_ridge=float(args.reference_ridge),
            variance_shrinkage=float(args.reference_variance_shrinkage),
            log_variance_bias_correction=float(
                args.reference_log_variance_bias_correction
            ),
            sigma_min=float(args.sigma_min),
            sigma_max=reference_sigma_max,
        )
    else:
        raise ValueError(f"unsupported reference_kind {reference_kind!r}")
    running_cost = str(getattr(args, "running_cost", "specific_entropy"))
    use_reference_drift = bool(getattr(args, "use_reference_drift", False))
    drift_control_mode = str(getattr(args, "drift_control_mode", "full"))
    if drift_control_mode not in {"full", "volatility_only"}:
        raise ValueError(f"unsupported drift_control_mode {drift_control_mode!r}")
    if running_cost == "specific_entropy":
        control_class = (
            ReferenceDriftSpecificEntropyDiagonalControl
            if use_reference_drift
            else SpecificEntropyDiagonalControl
        )
    elif running_cost == "gaussian_relative_entropy":
        if use_reference_drift:
            control_class = ReferenceDriftGaussianRelativeEntropyDiagonalControl
        elif drift_control_mode == "volatility_only":
            control_class = VolatilityOnlyGaussianRelativeEntropyDiagonalControl
        else:
            control_class = GaussianRelativeEntropyDiagonalControl
    else:
        raise ValueError(f"unsupported running cost {running_cost!r}")
    control_kwargs = dict(
        dt=grid.dt,
        eta=float(args.eta),
        reference_kernel=reference,
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
    )
    if control_class in {
        GaussianRelativeEntropyDiagonalControl,
        ReferenceDriftGaussianRelativeEntropyDiagonalControl,
        VolatilityOnlyGaussianRelativeEntropyDiagonalControl,
    }:
        control_kwargs["drift_penalty"] = float(
            getattr(args, "drift_penalty", 1.0)
        )
    if use_reference_drift:
        control_kwargs["allow_drift_correction"] = (
            drift_control_mode == "full"
        )
    control = control_class(**control_kwargs)
    representation = str(getattr(args, "state_representation", "log_price"))
    if representation == "log_price":
        discrepancy_target = target_paths
    elif representation == "scaled_log_return":
        discrepancy_target = float(path_scale) * torch.cumsum(target_paths, dim=1)
    else:
        raise ValueError("unsupported state_representation")
    discrepancy_preset = str(
        getattr(args, "discrepancy_preset", "heston_old_fullv_w0p25")
    )
    if discrepancy_preset == "heston_old_fullv_w0p25":
        base = _base_discrepancy()
    elif discrepancy_preset == "real_train_standardized_v1":
        base = build_real_data_standardized_discrepancy(
            discrepancy_target,
            grid=grid,
            bandwidths=(0.5, 1.0, 2.0),
            acf_lags=tuple(
                int(value)
                for value in getattr(
                    args,
                    "discrepancy_acf_lags",
                    (1, 2, 5, 10),
                )
            ),
        )
    else:
        raise ValueError(f"unsupported discrepancy_preset {discrepancy_preset!r}")
    joint = fit_fixed_target_joint_prefix_future_volatility_mmd(
        discrepancy_target,
        grid=grid,
        split_index=64,
        future_horizon=32,
        prefix_windows=(16, 32, 64),
        recent_return_window=16,
        bandwidths=(0.5, 1.0, 2.0),
    )
    composite = CompositePathFunctionalDiscrepancy(
        blocks=tuple(base.blocks)
        + (
            WeightedDiscrepancy(
                name="volatility_law_joint_prefix_future_rv",
                weight=float(getattr(args, "joint_weight", 1.0)),
                discrepancy=joint,
            ),
        )
    )
    if representation == "log_price":
        final_discrepancy: PathFunctionalDiscrepancy = composite
    else:
        final_discrepancy = CumulativeReturnPathDiscrepancy(
            composite,
            path_scale=float(path_scale),
        )
    return grid, architecture, control, base, final_discrepancy


def _build_model(
    *,
    architecture: DiscreteMPArchitectureConfig,
    grid: DiscreteTimeGrid,
    control: SpecificEntropyDiagonalControl,
    discrepancy: PathFunctionalDiscrepancy,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> DiscreteMPModel:
    model = DiscreteMPModel(
        architecture=architecture,
        grid=grid,
        dynamics_step=DriftVolatilityEulerStep(dt=grid.dt),
        control_map=control,
        discrepancy=discrepancy,
        init_seed=int(seed),
    )
    model.network.to(device=device, dtype=dtype)
    return model


def _write_scaler(path: Path, transform: SBTSLogReturnTransform) -> None:
    write_json(
        path,
        {
            "name": "SBTS_log_return_preprocessing",
            "sigma": float(transform.sigma),
            "dt": float(transform.dt),
            "s0": float(transform.s0),
            "unit_variance_wrapper": False,
        },
    )


def _load_scaler(path: Path) -> SBTSLogReturnTransform:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SBTSLogReturnTransform(
        sigma=float(payload["sigma"]),
        dt=float(payload["dt"]),
        s0=float(payload["s0"]),
    )


def train(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    model_dt = float(args.time_horizon) / NUM_STEPS
    target_paths, transform, preprocessing = _load_training_data(
        Path(args.benchmark_root),
        device=device,
        dtype=dtype,
        state_representation=str(args.state_representation),
        data_layout=str(args.data_layout),
        custom_dataset_root=args.custom_dataset_root,
        model_dt=model_dt,
    )
    grid, architecture, control, base, joint_discrepancy = _build_components(
        args,
        target_paths=target_paths,
        path_scale=float(transform.sigma) / math.sqrt(float(transform.dt)),
    )
    source_training = _make_training(
        args,
        num_steps=int(args.source_training_steps),
        seed=int(args.seed),
    )
    joint_training = _make_training(
        args,
        num_steps=int(args.joint_training_steps),
        seed=int(args.seed) + 1,
    )

    def progress(stage: str):
        def report(_model: DiscreteMPModel, metrics: dict[str, float]) -> None:
            print(
                "{stage} step={step:>4.0f} loss={loss:.5g} "
                "sigma_mean={sigma:.4f} grad={grad:.3g} clip_scale={clip:.3f}".format(
                    stage=stage,
                    step=float(metrics["step"]),
                    loss=float(metrics["train_total_loss"]),
                    sigma=float(metrics.get("sigma_mean", float("nan"))),
                    grad=float(metrics.get("grad_norm", float("nan"))),
                    clip=float(metrics.get("grad_clip_scale", float("nan"))),
                ),
                flush=True,
            )

        return report

    print(
        "training fixed source checkpoint "
        f"({source_training.num_steps} updates; benchmark test/ps splits unseen)",
        flush=True,
    )
    source_discrepancy: PathFunctionalDiscrepancy
    if str(args.state_representation) == "log_price":
        source_discrepancy = base
    else:
        source_discrepancy = CumulativeReturnPathDiscrepancy(
            base,
            path_scale=float(transform.sigma) / math.sqrt(float(transform.dt)),
        )
    source = _build_model(
        architecture=architecture,
        grid=grid,
        control=control,
        discrepancy=source_discrepancy,
        device=device,
        dtype=dtype,
        seed=int(args.seed),
    )
    source_fit = source.fit(
        target_paths,
        training=source_training,
        log_callback=progress("source"),
    )
    source_checkpoint = source.checkpoint_state()
    torch.save(source_checkpoint, run_dir / "source_model_checkpoint.pt")
    write_history_jsonl(run_dir / "source_history.jsonl", source_fit.history)

    print(
        "training fixed joint-discrepancy terminal checkpoint "
        f"({joint_training.num_steps} updates; no checkpoint selection)",
        flush=True,
    )
    candidate = _build_model(
        architecture=architecture,
        grid=grid,
        control=control,
        discrepancy=joint_discrepancy,
        device=device,
        dtype=dtype,
        seed=int(args.seed),
    )
    candidate.load_checkpoint_state(source_checkpoint)
    joint_fit = candidate.fit(
        target_paths,
        training=joint_training,
        log_callback=progress("joint"),
    )
    torch.save(candidate.checkpoint_state(), run_dir / "model_checkpoint.pt")
    write_history_jsonl(run_dir / "joint_history.jsonl", joint_fit.history)
    _write_scaler(run_dir / "preprocessing.json", transform)

    manifest = {
        "benchmark": {
            "root": str(Path(args.benchmark_root).resolve()),
            "commit": _benchmark_commit(Path(args.benchmark_root)),
            "dataset": {
                name: str(path.resolve())
                for name, path in _data_paths(
                    Path(args.benchmark_root),
                    str(args.data_layout),
                    custom_dataset_root=args.custom_dataset_root,
                ).items()
            },
            "data_layout": str(args.data_layout),
            "only_train_split_seen_by_generator": True,
            "test_or_ps_checkpoint_selection": False,
        },
        "preprocessing": preprocessing,
        "time_grid": {
            "horizon": float(args.time_horizon),
            "num_steps": NUM_STEPS,
            "dt": model_dt,
        },
        "representation": {
            "benchmark_external_sequence": (
                "[dummy zero, volatility-scaled one-step log returns]"
            ),
            "model_state": (
                "cumulative raw log return (log price relative to S0)"
                if str(args.state_representation) == "log_price"
                else "[dummy zero, volatility-scaled one-step log returns]"
            ),
            "fixed_model_adapter": (
                "sigma/sqrt(dt) times cumulative sum"
                if str(args.state_representation) == "log_price"
                else "identity"
            ),
            "path_discrepancy_input": "cumulative raw log returns",
            "inverse_to_price": "benchmark SBTS inverse",
        },
        "model": {
            "name": "Deep-MKV-Gen",
            "running_cost": "specific_entropy",
            "reference_kind": str(args.reference_kind),
            "reference_summary": control.reference_kernel.summary(),
            "discrepancy": "old_fullv_w0p25 + joint prefix/future realized-volatility MMD",
            "architecture": asdict(architecture),
            "source_training": asdict(source_training),
            "joint_training": asdict(joint_training),
            "fixed_terminal_checkpoint": True,
            "joint_weight": float(args.joint_weight),
            "joint_split": 64,
            "joint_horizon": 32,
            "joint_prefix_windows": [16, 32, 64],
        },
        "source_fit_final": source_fit.final_metrics,
        "joint_fit_final": joint_fit.final_metrics,
    }
    write_json(run_dir / "training_manifest.json", manifest)
    print(f"saved fixed terminal checkpoint to {run_dir / 'model_checkpoint.pt'}", flush=True)


def _reconstruct_model(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[DiscreteMPModel, SBTSLogReturnTransform]:
    model_dt = float(getattr(args, "time_horizon", NUM_STEPS * DT)) / NUM_STEPS
    target_paths, fitted_transform, _ = _load_training_data(
        Path(args.benchmark_root),
        device=device,
        dtype=dtype,
        state_representation=str(getattr(args, "state_representation", "log_price")),
        data_layout=str(getattr(args, "data_layout", "path_shadowing_4096")),
        custom_dataset_root=getattr(args, "custom_dataset_root", None),
        model_dt=model_dt,
    )
    transform = _load_scaler(run_dir / "preprocessing.json")
    if abs(float(transform.sigma) - float(fitted_transform.sigma)) > 1e-12:
        raise RuntimeError("persisted preprocessing sigma does not match the benchmark train split")
    grid, architecture, control, _base, discrepancy = _build_components(
        args,
        target_paths=target_paths,
        path_scale=float(transform.sigma) / math.sqrt(float(transform.dt)),
    )
    model = _build_model(
        architecture=architecture,
        grid=grid,
        control=control,
        discrepancy=discrepancy,
        device=device,
        dtype=dtype,
        seed=int(args.seed),
    )
    checkpoint = torch.load(
        run_dir / "model_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_checkpoint_state(checkpoint)
    model.network.eval()
    return model, transform


def build_bank(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> Path:
    model, transform = _reconstruct_model(
        args,
        run_dir=run_dir,
        device=device,
        dtype=dtype,
    )
    bank_size = int(getattr(args, "bank_size", BANK_SIZE))
    if bank_size <= 0:
        raise ValueError("bank_size must be positive")
    bank_output = getattr(args, "bank_output", None)
    if bank_output is None:
        bank_dir = run_dir / "path_shadowing" / "bank"
        bank_path = bank_dir / BANK_FILENAME
    else:
        bank_path = Path(bank_output)
        bank_dir = bank_path.parent
    bank_dir.mkdir(parents=True, exist_ok=True)
    bank = np.lib.format.open_memmap(
        bank_path,
        mode="w+",
        dtype=np.float32,
        shape=(bank_size, SEQ_LEN),
    )
    x0 = torch.zeros((1, 1), device=device, dtype=dtype)
    batch_size = int(args.sample_batch_size)
    with torch.no_grad():
        for batch_index, start in enumerate(range(0, bank_size, batch_size)):
            count = min(batch_size, bank_size - start)
            batch_seed = derive_stream_seed(
                base_seed=int(getattr(args, "bank_seed", 0)),
                stream_offset=10_000 + int(batch_index),
            )
            sample = model.sample(num_paths=count, x0=x0, seed=batch_seed)
            model_states = sample.paths[..., 0].detach().cpu().numpy()
            if str(args.state_representation) == "log_price":
                return_states = np.zeros_like(model_states)
                return_states[:, 1:] = (
                    np.diff(model_states, axis=1)
                    * math.sqrt(float(transform.dt))
                    / float(transform.sigma)
                )
            else:
                return_states = model_states
            bank[start : start + count] = transform.inverse(
                return_states,
                dtype=np.float32,
            )
            bank.flush()
            print(
                f"bank {start + count:>7}/{bank_size} "
                f"({100.0 * (start + count) / bank_size:5.1f}%)",
                flush=True,
            )
            del sample, model_states, return_states
            if device.type == "cuda":
                torch.cuda.empty_cache()
    del bank
    saved = np.load(bank_path, mmap_mode="r")
    if saved.shape != (bank_size, SEQ_LEN):
        raise RuntimeError(f"saved bank has unexpected shape {saved.shape}")
    if not np.isfinite(np.asarray(saved[: min(4096, bank_size)])).all():
        raise RuntimeError("saved bank contains nonfinite values")
    print(f"saved benchmark bank to {bank_path}", flush=True)
    return bank_path


def _load_pdf_evaluator(benchmark_root: Path):
    evaluator_path = (
        benchmark_root
        / "results"
        / "Heston"
        / "preprocessing_with_log_returns"
        / "CSDI"
        / "path_shadowing"
        / "path_shadowing_pdf.py"
    )
    spec = importlib.util.spec_from_file_location("basseras_path_shadowing_pdf", evaluator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import benchmark evaluator {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, evaluator_path


def evaluate(args: argparse.Namespace, *, run_dir: Path) -> Path:
    benchmark_root = Path(args.benchmark_root)
    evaluator, evaluator_path = _load_pdf_evaluator(benchmark_root)
    output_dir = run_dir / "path_shadowing"
    bank_dir = output_dir / "bank"
    bank_path = bank_dir / BANK_FILENAME
    if not bank_path.exists():
        raise FileNotFoundError(bank_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator.HERE = str(output_dir)
    evaluator.BANK_DIR = str(bank_dir)
    evaluator.PLOT_DIR = str(output_dir / "plots")
    print(
        f"running unchanged evaluator {evaluator_path} "
        f"(sha256 recorded after completion)",
        flush=True,
    )
    evaluator.main()
    summary_path = output_dir / "pdf_summary.json"
    if not summary_path.exists():
        raise RuntimeError("benchmark evaluator did not write pdf_summary.json")

    import hashlib

    digest = hashlib.sha256(evaluator_path.read_bytes()).hexdigest()
    audit = {
        "evaluator_path": str(evaluator_path.resolve()),
        "evaluator_sha256": digest,
        "benchmark_commit": _benchmark_commit(benchmark_root),
        "evaluator_source_modified": False,
        "runtime_overrides": {
            "HERE": str(output_dir.resolve()),
            "BANK_DIR": str(bank_dir.resolve()),
            "PLOT_DIR": str((output_dir / "plots").resolve()),
        },
    }
    write_json(output_dir / "evaluator_audit.json", audit)
    return summary_path


def main() -> None:
    args = _parse_args()
    run_dir = ensure_run_dir(args.run_dir or _default_run_dir())
    benchmark_root = Path(args.benchmark_root)
    if not benchmark_root.exists():
        raise FileNotFoundError(benchmark_root)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    if args.stage in {"all", "train"}:
        train(args, run_dir=run_dir, device=device, dtype=dtype)
    if args.stage in {"all", "bank"}:
        build_bank(args, run_dir=run_dir, device=device, dtype=dtype)
    if args.stage in {"all", "evaluate"}:
        summary = evaluate(args, run_dir=run_dir)
        print(f"completed exact BASSERAS path-shadowing evaluation: {summary}", flush=True)


if __name__ == "__main__":
    main()
