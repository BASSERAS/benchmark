from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from deep_mkv_gen_path_dt import (  # noqa: E402
    ConditionalSignalDiagnosticConfig,
    DiscreteMPArchitectureConfig,
    DiscreteMPModel,
    DiscreteMPTrainingConfig,
    DiscreteTimeGrid,
    diagnose_conditional_adjoint_signal,
    fit_guyon_lekeufack_likelihood_reference_kernel,
    fit_guyon_lekeufack_price_realized_volatility_reference_kernel,
    fit_guyon_lekeufack_reference_kernel,
    fit_local_gaussian_reference_kernel,
    guyon_lekeufack_reference_kernel_from_parameters,
)
from deep_mkv_gen_path_dt.controls import (  # noqa: E402
    EntropyBarrierDiagonalControl,
    ReferenceDriftSpecificEntropyDiagonalControl,
    SpecificEntropyDiagonalControl,
)
from deep_mkv_gen_path_dt.noise import derive_stream_seed  # noqa: E402
from path_dt_experiments.data import HestonConfig, simulate_heston_log_paths  # noqa: E402
from path_dt_experiments.control_ablation import (  # noqa: E402
    AdjointComponentAblationControl,
    specific_entropy_r_effect_on_realized_prefixes,
)
from path_dt_experiments.discrepancy_diagnostics import (  # noqa: E402
    discrepancy_diffusion_scale_sensitivity,
)
from path_dt_experiments.discrepancies import build_heston_discrepancy  # noqa: E402
from path_dt_experiments.dynamics import DriftVolatilityEulerStep  # noqa: E402
from path_dt_experiments.frozen_response import (  # noqa: E402
    FrozenResponseAuditConfig,
    audit_frozen_block_responses,
)
from path_dt_experiments.metrics import stylized_fact_metrics  # noqa: E402
from path_dt_experiments.observed_reference_fine_grid import (  # noqa: E402
    FineGridVolatilityOnlySpecificEntropyControl,
    ObservedReferenceFineGridLift,
)
from path_dt_experiments.runners import default_run_dir, ensure_run_dir, write_history_jsonl, write_json  # noqa: E402
from path_dt_experiments.scores import BenchmarkScoreConfig, benchmark_scores  # noqa: E402
from path_dt_experiments.shadowing import evaluate_path_shadowing  # noqa: E402


def sample_model_paths_in_batches(
    model: DiscreteMPModel,
    *,
    num_paths: int,
    x0: torch.Tensor,
    seed: int | None,
    batch_size: int,
) -> torch.Tensor:
    num_paths = int(num_paths)
    batch_size = int(batch_size)
    if num_paths < 1:
        raise ValueError("num_paths must be >= 1")
    if batch_size < 1:
        raise ValueError("sample_batch_size must be >= 1")

    chunks = []
    for batch_index, start in enumerate(range(0, num_paths, batch_size)):
        current = min(batch_size, num_paths - start)
        chunk_seed = None
        if seed is not None:
            chunk_seed = derive_stream_seed(base_seed=int(seed), stream_offset=10_000 + int(batch_index))
        sample = model.sample(
            num_paths=current,
            x0=x0,
            seed=chunk_seed,
        )
        chunks.append(sample.paths.detach().cpu())
        if model.device.type == "cuda":
            torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0)


def observed_indices_from_every(*, num_steps: int, observed_every: int) -> tuple[int, ...]:
    if int(observed_every) < 1:
        raise ValueError("observed_every must be >= 1")
    if int(num_steps) % int(observed_every) != 0:
        raise ValueError("num_grid_steps must be divisible by observed_every")
    return tuple(range(0, int(num_steps) + 1, int(observed_every)))


def global_diffusion_scale(paths: torch.Tensor, *, dt: float) -> float:
    if paths.ndim != 3 or int(paths.shape[1]) < 2:
        raise ValueError("paths must have shape (B, N + 1, d) with N >= 1")
    if not math.isfinite(float(dt)) or float(dt) <= 0.0:
        raise ValueError("dt must be a positive finite float")
    increments = paths[:, 1:, :] - paths[:, :-1, :]
    centered = increments - increments.mean(dim=0, keepdim=True)
    return float(torch.sqrt(torch.mean(centered.pow(2)) / float(dt)).item())


def conditional_volatility_windows_for_discrepancy(
    windows: tuple[int, ...] | list[int],
    *,
    observed_every: int,
    observed_only: bool,
) -> tuple[int, ...]:
    values = tuple(int(value) for value in windows)
    if len(values) == 0 or any(value < 1 for value in values):
        raise ValueError("conditional volatility windows must be non-empty positive integers")
    if len(set(values)) != len(values):
        raise ValueError("conditional volatility windows must not contain duplicates")
    if not bool(observed_only):
        return values
    spacing = int(observed_every)
    if spacing < 1:
        raise ValueError("observed_every must be >= 1")
    if any(value % spacing != 0 for value in values):
        raise ValueError("conditional volatility windows must be divisible by observed_every")
    observed_windows = tuple(value // spacing for value in values)
    if len(set(observed_windows)) != len(observed_windows):
        raise ValueError("conditional volatility windows collapse to duplicates on the observed grid")
    return observed_windows


def discrepancy_evaluation_paths(
    generated_paths: torch.Tensor,
    target_paths: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
    observed_only: bool,
) -> tuple[torch.Tensor, torch.Tensor, DiscreteTimeGrid]:
    generated = generated_paths.detach().cpu()
    target = target_paths.detach().cpu()
    if not bool(observed_only):
        return generated, target, grid
    indices = grid.observed_point_indices(device=torch.device("cpu"))
    observed_times = grid.time_points(device=torch.device("cpu"), dtype=torch.float64).index_select(0, indices)
    observed_grid = DiscreteTimeGrid(
        T=float(grid.T),
        num_steps=int(grid.observed_count) - 1,
        point_times=tuple(float(value) for value in observed_times.tolist()),
        allow_nonuniform=True,
    )
    return generated.index_select(1, indices), target.index_select(1, indices), observed_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the discrete MP model on synthetic Heston paths.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-target-paths", type=int, default=8192)
    parser.add_argument("--num-heldout-paths", type=int, default=2048)
    parser.add_argument("--num-sample-paths", type=int, default=2048)
    parser.add_argument("--sample-batch-size", type=int, default=50_000)

    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--num-grid-steps", type=int, default=64)
    parser.add_argument(
        "--target-simulation-refinement",
        type=int,
        default=1,
        help=(
            "Simulate the Heston target with this many Euler transitions per "
            "model transition, then retain only the model-grid points. This "
            "supports matched coarse-versus-fine discretization comparisons."
        ),
    )
    parser.add_argument("--observed-every", type=int, default=1)
    parser.add_argument(
        "--observed-reference-fine-substeps",
        type=int,
        default=1,
        help=(
            "Experimental isolated fine-grid mode. Fit the local reference only "
            "at observed dates, lift it between observations, and recompute the "
            "MP volatility at every fine step. A value above one must equal "
            "--observed-every. Defaults to one, leaving the established runner unchanged."
        ),
    )
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--kappa", type=float, default=3.0)
    parser.add_argument("--theta", type=float, default=0.04)
    parser.add_argument("--vol-of-vol", type=float, default=0.6)
    parser.add_argument("--rho", type=float, default=-0.7)
    parser.add_argument("--v0", type=float, default=0.04)
    parser.add_argument("--x0", type=float, default=0.0)

    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument(
        "--adjoint-input-mode",
        choices=("level", "increment"),
        default="level",
        help=(
            "Input representation for the causal adjoint GRU. The increment "
            "mode supplies zero initially and the latest log return thereafter; "
            "it does not change the simulated log-price state or MP objective."
        ),
    )
    parser.add_argument("--training-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--target-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lambda-scale", type=float, default=50.0)
    parser.add_argument("--kappa-scale", type=float, default=50.0)
    parser.add_argument("--ce-target-mode", choices=("ridge", "direct"), default="ridge")
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--ce-crossfit-folds", type=int, default=1)
    parser.add_argument("--target-preconditioner", choices=("none", "timewise"), default="none")
    parser.add_argument("--preconditioner-batches", type=int, default=8)
    parser.add_argument("--preconditioner-min-scale", type=float, default=1e-3)
    parser.add_argument(
        "--noise-target-control-variate",
        choices=("none", "timewise", "adjoint"),
        default="none",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--observed-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--path-derivative-backend",
        choices=("autograd", "analytical"),
        default="autograd",
        help=(
            "Compute empirical discrepancy and running-cost path sources with "
            "PyTorch autograd or with the closed-form first-derivative backend."
        ),
    )

    parser.add_argument(
        "--control-map",
        choices=("specific_entropy", "entropy_barrier"),
        default="specific_entropy",
    )
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--entropy-kappa", type=float, default=12.5)
    parser.add_argument("--entropy-tau", type=float, default=None)
    parser.add_argument("--sigma-min", type=float, default=1e-4)
    parser.add_argument("--sigma-max", type=float, default=2.0)
    parser.add_argument("--reference-ridge", type=float, default=1e-3)
    parser.add_argument(
        "--reference-kind",
        choices=(
            "local_gaussian",
            "guyon_lekeufack",
            "guyon_lekeufack_likelihood",
            "guyon_lekeufack_structural_likelihood",
            "guyon_lekeufack_price_realized_volatility",
            "guyon_lekeufack_official",
        ),
        default="local_gaussian",
        help=(
            "Causal volatility reference used by specific entropy. The default "
            "preserves the established datewise ridge reference; guyon_lekeufack "
            "fits two exponential return and squared-return filters; the two "
            "likelihood variants fit every volatility parameter from returns, "
            "with either realized-return or structural-variance activity; "
            "the price-realized-volatility variant fits the empirical factor "
            "regression to a forward volatility label built from prices; "
            "guyon_lekeufack_official loads parameters calibrated by the authors' "
            "public implementation."
        ),
    )
    parser.add_argument(
        "--guyon-realized-volatility-window",
        type=int,
        default=20,
        help="Forward return window used by the price-only empirical Guyon calibration.",
    )
    parser.add_argument(
        "--guyon-official-parameters",
        type=Path,
        default=None,
        help=(
            "JSON file containing beta_0, beta_1, beta_2, theta_1, theta_2, "
            "lambda_1, and lambda_2 from the official two-exponential calibration."
        ),
    )
    parser.add_argument("--reference-variance-ridge", type=float, default=None)
    parser.add_argument("--reference-variance-shrinkage", type=float, default=0.1)
    parser.add_argument("--reference-log-variance-bias-correction", type=float, default=0.6)
    parser.add_argument("--reference-sigma-max", type=float, default=0.6)
    parser.add_argument(
        "--fixed-reference-drift-volatility-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Retain the fitted causal reference drift exactly and let MP control "
            "only volatility. This is available with specific entropy only."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=("simple", "old_fullv_w0p25", "reduced_time_local"),
        default="simple",
    )
    parser.add_argument("--include-acf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--conditional-volatility-discrepancy",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--conditional-volatility-correlation-discrepancy",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--conditional-volatility-correlation-full-matrix",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--conditional-volatility-windows", type=int, nargs="+", default=(4, 8, 16, 32))
    parser.add_argument(
        "--conditional-volatility-correlation-windows",
        type=int,
        nargs="+",
        default=(16, 32),
    )
    parser.add_argument("--conditional-volatility-weight", type=float, default=1.0)
    parser.add_argument("--conditional-volatility-correlation-weight", type=float, default=0.025)
    parser.add_argument(
        "--conditional-volatility-correlation-scale-floor-fraction",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--conditional-volatility-bandwidths",
        type=float,
        nargs="+",
        default=(0.5, 1.0, 2.0),
    )
    parser.add_argument("--time-marginal-volatility-window", type=int, default=None)
    parser.add_argument("--time-marginal-volatility-endpoints", type=int, nargs="+", default=None)
    parser.add_argument("--time-marginal-volatility-weight", type=float, default=1.0)
    parser.add_argument(
        "--discrepancy-sensitivity",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--sensitivity-paths", type=int, default=256)

    parser.add_argument("--shadowing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shadow-top-k", type=int, default=32)
    parser.add_argument("--shadow-real-paths", type=int, default=128)
    parser.add_argument("--shadow-prefix-length", type=int, default=None)
    parser.add_argument("--shadow-future-steps", type=int, default=None)
    parser.add_argument("--shadow-search", choices=("auto", "exact", "faiss"), default="auto")
    parser.add_argument("--shadow-exact-batch-size", type=int, default=64)
    parser.add_argument("--scores", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--score-steps", type=int, default=300)
    parser.add_argument("--score-batch-size", type=int, default=128)
    parser.add_argument("--score-hidden-dim", type=int, default=32)
    parser.add_argument("--score-lr", type=float, default=1e-3)

    parser.add_argument(
        "--conditional-signal-diagnostic",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--signal-prefixes", type=int, default=64)
    parser.add_argument("--signal-branches", type=int, default=8)
    parser.add_argument("--signal-target-batch-size", type=int, default=256)
    parser.add_argument("--signal-step-indices", type=int, nargs="+", default=None)
    parser.add_argument("--signal-discrepancy-blocks", type=str, nargs="+", default=None)
    parser.add_argument("--signal-seed", type=int, default=None)

    parser.add_argument(
        "--frozen-response-audit",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--response-correlation-candidate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the stabilized correlation block in the audit even when it was not used for training.",
    )
    parser.add_argument("--response-direction-batch-size", type=int, default=256)
    parser.add_argument("--response-path-gradient-batch-size", type=int, default=128)
    parser.add_argument("--response-target-batch-size", type=int, default=256)
    parser.add_argument("--response-metric-paths", type=int, default=512)
    parser.add_argument("--response-metric-batch-size", type=int, default=512)
    parser.add_argument("--response-parameter-step-fraction", type=float, default=1e-5)
    parser.add_argument("--response-path-return-step-fraction", type=float, default=1e-2)
    parser.add_argument("--response-prefix-length", type=int, default=None)
    parser.add_argument("--response-prefix-window", type=int, default=None)
    parser.add_argument("--response-horizon", type=int, default=None)
    parser.add_argument("--response-swd-projections", type=int, default=32)
    parser.add_argument("--response-seed", type=int, default=None)

    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--save-tensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-checkpoint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--specific-entropy-r-ablation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After fitting specific entropy, compare the full rollout with R=0 "
            "and with P=R=0 under common random numbers."
        ),
    )
    parser.add_argument("--specific-entropy-ablation-diagnostic-paths", type=int, default=2048)
    return parser.parse_args()


def load_official_guyon_parameters(path: Path) -> dict[str, object]:
    """Load the rate pairs as emitted by the verified official-code wrapper."""
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    values = document.get("parameters", document)
    if not isinstance(values, dict):
        raise ValueError("official Guyon parameter JSON must contain an object")
    required = (
        "beta_0",
        "beta_1",
        "beta_2",
        "theta_1",
        "theta_2",
        "lambda_1",
        "lambda_2",
    )
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"official Guyon parameter JSON is missing {missing}")
    lambda_1 = tuple(float(value) for value in values["lambda_1"])
    lambda_2 = tuple(float(value) for value in values["lambda_2"])
    if len(lambda_1) != 2 or len(lambda_2) != 2:
        raise ValueError("lambda_1 and lambda_2 must each contain two rates")
    return {
        "beta_0": float(values["beta_0"]),
        "beta_1": float(values["beta_1"]),
        "beta_2": float(values["beta_2"]),
        "trend_weight": float(values["theta_1"]),
        "activity_weight": float(values["theta_2"]),
        "trend_lambdas": lambda_1,
        "activity_lambdas": lambda_2,
    }


def main() -> None:
    args = parse_args()
    official_parameter_path = args.guyon_official_parameters
    if str(args.reference_kind) == "guyon_lekeufack_official":
        if official_parameter_path is None:
            raise ValueError(
                "guyon_lekeufack_official requires --guyon-official-parameters"
            )
        official_parameter_path = official_parameter_path.resolve()
        if not official_parameter_path.is_file():
            raise FileNotFoundError(official_parameter_path)
    elif official_parameter_path is not None:
        raise ValueError(
            "--guyon-official-parameters is valid only with "
            "--reference-kind guyon_lekeufack_official"
        )
    if int(args.target_simulation_refinement) < 1:
        raise ValueError("target-simulation-refinement must be >= 1")
    fine_substeps = int(args.observed_reference_fine_substeps)
    if fine_substeps < 1:
        raise ValueError("observed-reference-fine-substeps must be >= 1")
    if fine_substeps > 1:
        if int(args.observed_every) != fine_substeps:
            raise ValueError(
                "fine substeps must equal observed-every so the reference is fitted "
                "exactly at discrepancy dates"
            )
        if int(args.target_simulation_refinement) != 1:
            raise ValueError(
                "fine-grid MP already uses the target simulation grid; "
                "target-simulation-refinement must be one"
            )
        if not bool(args.observed_only):
            raise ValueError("fine-grid MP requires observed-only discrepancies")
        if not bool(args.fixed_reference_drift_volatility_only):
            raise ValueError(
                "fine-grid MP requires fixed-reference-drift-volatility-only"
            )
        if str(args.control_map) != "specific_entropy":
            raise ValueError("fine-grid MP currently requires specific entropy")
        if str(args.reference_kind) != "local_gaussian":
            raise ValueError(
                "the Guyon--Lekeufack reference is currently available on the "
                "established single model grid only"
            )
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    run_dir = ensure_run_dir(args.run_dir or default_run_dir(prefix="heston_dt"))

    heston_config = HestonConfig(
        T=float(args.T),
        num_steps=(
            int(args.num_grid_steps) * int(args.target_simulation_refinement)
        ),
        mu=float(args.mu),
        kappa=float(args.kappa),
        theta=float(args.theta),
        vol_of_vol=float(args.vol_of_vol),
        rho=float(args.rho),
        v0=float(args.v0),
        x0=float(args.x0),
    )
    grid = DiscreteTimeGrid(
        T=float(args.T),
        num_steps=int(args.num_grid_steps),
        observed_indices=observed_indices_from_every(
            num_steps=int(args.num_grid_steps),
            observed_every=int(args.observed_every),
        ),
    )
    conditional_discrepancy_count = int(bool(args.conditional_volatility_discrepancy)) + int(
        bool(args.conditional_volatility_correlation_discrepancy)
    )
    if conditional_discrepancy_count > 1:
        raise ValueError("choose at most one conditional volatility discrepancy")
    conditional_volatility_windows = (
        conditional_volatility_windows_for_discrepancy(
            (
                args.conditional_volatility_correlation_windows
                if bool(args.conditional_volatility_correlation_discrepancy)
                else args.conditional_volatility_windows
            ),
            observed_every=int(args.observed_every),
            observed_only=bool(args.observed_only),
        )
        if conditional_discrepancy_count == 1
        else ()
    )
    discrepancy_steps = (
        int(grid.observed_count) - 1 if bool(args.observed_only) else int(args.num_grid_steps)
    )
    if conditional_discrepancy_count == 1 and max(conditional_volatility_windows) > discrepancy_steps // 2:
        raise ValueError("largest conditional volatility window must fit before and after the midpoint")

    target_paths = simulate_heston_log_paths(
        num_paths=int(args.num_target_paths),
        config=heston_config,
        seed=int(args.seed),
        device=device,
        dtype=dtype,
    )
    evaluation_seed = derive_stream_seed(base_seed=int(args.seed), stream_offset=20) or int(args.seed)
    heldout_seed = derive_stream_seed(base_seed=evaluation_seed, stream_offset=1)
    sample_seed = derive_stream_seed(base_seed=evaluation_seed, stream_offset=2)
    heldout_paths = simulate_heston_log_paths(
        num_paths=int(args.num_heldout_paths),
        config=heston_config,
        seed=heldout_seed,
        device=device,
        dtype=dtype,
    )
    if int(args.target_simulation_refinement) > 1:
        target_indices = torch.arange(
            0,
            int(heston_config.num_steps) + 1,
            int(args.target_simulation_refinement),
            device=device,
        )
        target_paths = target_paths.index_select(1, target_indices)
        heldout_paths = heldout_paths.index_select(1, target_indices)
    expected_target_points = int(args.num_grid_steps) + 1
    if int(target_paths.shape[1]) != expected_target_points:
        raise RuntimeError(
            "refined Heston target did not collapse to the requested model grid"
        )
    global_sigma = global_diffusion_scale(target_paths, dt=grid.dt)
    if str(args.control_map) == "specific_entropy":
        reference_target_paths = target_paths
        reference_grid = grid
        if fine_substeps > 1:
            reference_indices = grid.observed_point_indices(device=device)
            reference_target_paths = target_paths.index_select(1, reference_indices)
            reference_grid = DiscreteTimeGrid(
                T=float(grid.T),
                num_steps=int(grid.observed_count) - 1,
            )
        reference_kwargs = {
            "target_paths": reference_target_paths,
            "grid": reference_grid,
            "ridge": float(args.reference_ridge),
            "variance_ridge": args.reference_variance_ridge,
            "variance_shrinkage": float(args.reference_variance_shrinkage),
            "log_variance_bias_correction": float(args.reference_log_variance_bias_correction),
            "sigma_min": float(args.sigma_min),
            "sigma_max": (
                float(args.reference_sigma_max)
                if args.reference_sigma_max is not None
                else None
            ),
        }
        if str(args.reference_kind) == "guyon_lekeufack_official":
            reference_kernel = guyon_lekeufack_reference_kernel_from_parameters(
                **reference_kwargs,
                **load_official_guyon_parameters(official_parameter_path),
            )
        elif str(args.reference_kind) == "guyon_lekeufack":
            reference_kernel = fit_guyon_lekeufack_reference_kernel(
                **reference_kwargs,
            )
        elif str(args.reference_kind) in {
            "guyon_lekeufack_likelihood",
            "guyon_lekeufack_structural_likelihood",
        }:
            reference_kernel = fit_guyon_lekeufack_likelihood_reference_kernel(
                **reference_kwargs,
                activity_update=(
                    "structural_variance"
                    if str(args.reference_kind) == "guyon_lekeufack_structural_likelihood"
                    else "realized_returns"
                ),
            )
        elif str(args.reference_kind) == "guyon_lekeufack_price_realized_volatility":
            reference_kernel = (
                fit_guyon_lekeufack_price_realized_volatility_reference_kernel(
                    **reference_kwargs,
                    realized_volatility_window=int(args.guyon_realized_volatility_window),
                )
            )
        else:
            reference_kernel = fit_local_gaussian_reference_kernel(
                **reference_kwargs,
            )
        if fine_substeps > 1:
            reference_lift = ObservedReferenceFineGridLift(
                observed_reference=reference_kernel,
                fine_grid=grid,
                substeps=fine_substeps,
            )
            control_map = FineGridVolatilityOnlySpecificEntropyControl(
                dt=float(grid.dt),
                reference_lift=reference_lift,
                eta=float(args.eta),
                sigma_min=float(args.sigma_min),
                sigma_max=float(args.sigma_max),
            )
        else:
            control_kwargs = dict(
                dt=grid.dt,
                eta=float(args.eta),
                reference_kernel=reference_kernel,
                sigma_min=float(args.sigma_min),
                sigma_max=float(args.sigma_max),
            )
            control_map = (
                ReferenceDriftSpecificEntropyDiagonalControl(
                    **control_kwargs,
                    allow_drift_correction=False,
                )
                if bool(args.fixed_reference_drift_volatility_only)
                else SpecificEntropyDiagonalControl(**control_kwargs)
            )
        control_summary: dict[str, object] = {
            "name": (
                "observed_reference_fine_grid_volatility_only_specific_entropy"
                if fine_substeps > 1
                else (
                    "fixed_reference_drift_volatility_only_specific_entropy"
                    if bool(args.fixed_reference_drift_volatility_only)
                    else "specific_entropy"
                )
            ),
            "eta": float(args.eta),
            "reference_kind": str(args.reference_kind),
            "official_parameter_file": (
                str(official_parameter_path)
                if official_parameter_path is not None
                else None
            ),
            "reference": (
                reference_lift.summary()
                if fine_substeps > 1
                else reference_kernel.summary()
            )
            | {
                "reference_ridge": float(args.reference_ridge),
                "reference_variance_ridge": (
                    float(args.reference_variance_ridge)
                    if args.reference_variance_ridge is not None
                    else float(args.reference_ridge)
                ),
                "reference_variance_shrinkage": float(args.reference_variance_shrinkage),
                "reference_log_variance_bias_correction": float(args.reference_log_variance_bias_correction),
            },
        }
    else:
        if bool(args.fixed_reference_drift_volatility_only):
            raise ValueError(
                "fixed-reference-drift-volatility-only requires --control-map specific_entropy"
            )
        reference_kernel = None
        entropy_kappa = float(args.entropy_kappa)
        entropy_tau = (
            float(args.entropy_tau)
            if args.entropy_tau is not None
            else entropy_kappa * float(global_sigma) ** 2
        )
        control_map = EntropyBarrierDiagonalControl(
            dt=grid.dt,
            kappa=entropy_kappa,
            tau=entropy_tau,
            sigma_min=float(args.sigma_min),
            sigma_max=float(args.sigma_max),
        )
        control_summary = {
            "name": "entropy_barrier",
            "kappa": entropy_kappa,
            "tau": entropy_tau,
            "baseline_sigma": float(control_map.baseline_sigma),
            "baseline_source": "explicit_tau" if args.entropy_tau is not None else "target_global_diffusion_scale",
            "reference": None,
        }

    discrepancy_kwargs = {
        "num_steps": discrepancy_steps,
        "include_acf": bool(args.include_acf),
        "preset": str(args.preset),
        "lambda_scale": float(args.lambda_scale),
        "kappa_scale": float(args.kappa_scale),
        "include_conditional_volatility": bool(args.conditional_volatility_discrepancy),
        "include_conditional_volatility_correlation": bool(
            args.conditional_volatility_correlation_discrepancy
        ),
        "conditional_volatility_windows": (
            conditional_volatility_windows
            if bool(args.conditional_volatility_discrepancy)
            else ()
        ),
        "conditional_volatility_correlation_windows": (
            conditional_volatility_windows
            if bool(args.conditional_volatility_correlation_discrepancy)
            else ()
        ),
        "conditional_volatility_weight": float(args.conditional_volatility_weight),
        "conditional_volatility_correlation_weight": float(
            args.conditional_volatility_correlation_weight
        ),
        "conditional_volatility_bandwidths": tuple(
            float(value) for value in args.conditional_volatility_bandwidths
        ),
        "conditional_volatility_correlation_full_matrix": bool(
            args.conditional_volatility_correlation_full_matrix
        ),
        "conditional_volatility_correlation_scale_floor_fraction": float(
            args.conditional_volatility_correlation_scale_floor_fraction
        ),
        "time_marginal_volatility_window": args.time_marginal_volatility_window,
        "time_marginal_volatility_endpoints": (
            ()
            if args.time_marginal_volatility_endpoints is None
            else tuple(int(value) for value in args.time_marginal_volatility_endpoints)
        ),
        "time_marginal_volatility_weight": float(args.time_marginal_volatility_weight),
    }
    discrepancy = build_heston_discrepancy(**discrepancy_kwargs)
    model = DiscreteMPModel(
        architecture=DiscreteMPArchitectureConfig(
            state_dim=1,
            noise_dim=1,
            hidden_dim=int(args.hidden_dim),
            num_layers=int(args.num_layers),
            adjoint_input_mode=str(args.adjoint_input_mode),
        ),
        grid=grid,
        dynamics_step=DriftVolatilityEulerStep(dt=grid.dt),
        control_map=control_map,
        discrepancy=discrepancy,
        init_seed=int(args.seed),
    )
    model.network.to(device=device, dtype=dtype)

    training_config = DiscreteMPTrainingConfig(
        batch_size=int(args.batch_size),
        target_batch_size=int(args.target_batch_size),
        num_steps=int(args.training_steps),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        lambda_scale=float(args.lambda_scale),
        ce_target_mode=str(args.ce_target_mode),
        ridge_lambda=float(args.ridge_lambda),
        ce_crossfit_folds=int(args.ce_crossfit_folds),
        target_preconditioner=str(args.target_preconditioner),
        preconditioner_batches=int(args.preconditioner_batches),
        preconditioner_min_scale=float(args.preconditioner_min_scale),
        noise_target_control_variate=str(args.noise_target_control_variate),
        grad_clip_norm=float(args.grad_clip_norm) if args.grad_clip_norm > 0.0 else None,
        log_every=int(args.log_every),
        seed=int(args.seed),
        observed_only=bool(args.observed_only),
        path_derivative_backend=str(args.path_derivative_backend),
    )
    fit_result = model.fit(target_paths, training=training_config)

    if bool(args.save_checkpoint):
        torch.save(model.checkpoint_state(), run_dir / "model_checkpoint.pt")

    if bool(args.frozen_response_audit):
        response_adds_candidate = bool(args.response_correlation_candidate) and not bool(
            args.conditional_volatility_correlation_discrepancy
        )
        if response_adds_candidate:
            if bool(args.conditional_volatility_discrepancy):
                raise ValueError(
                    "response correlation candidate cannot be combined with the conditional-volatility MMD block"
                )
            response_windows = conditional_volatility_windows_for_discrepancy(
                args.conditional_volatility_correlation_windows,
                observed_every=int(args.observed_every),
                observed_only=bool(args.observed_only),
            )
            if max(response_windows) > discrepancy_steps // 2:
                raise ValueError("largest response correlation window must fit before and after the midpoint")
            response_kwargs = dict(discrepancy_kwargs)
            response_kwargs["include_conditional_volatility_correlation"] = True
            response_kwargs["conditional_volatility_correlation_windows"] = response_windows
            response_discrepancy = build_heston_discrepancy(**response_kwargs)
        else:
            response_discrepancy = discrepancy
        response_seed = (
            int(args.response_seed)
            if args.response_seed is not None
            else derive_stream_seed(base_seed=int(args.seed), stream_offset=96_000)
        )
        if response_seed is None:
            response_seed = int(args.seed)
        frozen_response = audit_frozen_block_responses(
            model=model,
            discrepancy=response_discrepancy,
            target_paths=target_paths,
            reference_paths=heldout_paths,
            x0=torch.tensor([[float(args.x0)]], device=device, dtype=dtype),
            training=training_config,
            config=FrozenResponseAuditConfig(
                direction_batch_size=int(args.response_direction_batch_size),
                path_gradient_batch_size=int(args.response_path_gradient_batch_size),
                target_batch_size=int(args.response_target_batch_size),
                metric_num_paths=int(args.response_metric_paths),
                metric_batch_size=int(args.response_metric_batch_size),
                parameter_step_fraction=float(args.response_parameter_step_fraction),
                path_return_step_fraction=float(args.response_path_return_step_fraction),
                prefix_length=args.response_prefix_length,
                prefix_window=args.response_prefix_window,
                horizon=args.response_horizon,
                swd_projections=int(args.response_swd_projections),
                seed=int(response_seed),
            ),
        )
        frozen_response.metrics["candidate_correlation_block_not_used_in_training"] = response_adds_candidate
        write_json(run_dir / "frozen_response_audit.json", frozen_response.metrics)
        torch.save(frozen_response.tensors, run_dir / "frozen_response_tensors.pt")
    else:
        frozen_response = None

    if bool(args.conditional_signal_diagnostic):
        signal_seed = (
            int(args.signal_seed)
            if args.signal_seed is not None
            else derive_stream_seed(base_seed=int(args.seed), stream_offset=95_000)
        )
        if signal_seed is None:
            signal_seed = int(args.seed)
        conditional_signal = diagnose_conditional_adjoint_signal(
            model=model,
            target_paths=target_paths,
            training=training_config,
            config=ConditionalSignalDiagnosticConfig(
                num_prefixes=int(args.signal_prefixes),
                num_branches=int(args.signal_branches),
                target_batch_size=int(args.signal_target_batch_size),
                step_indices=(
                    None
                    if args.signal_step_indices is None
                    else tuple(int(value) for value in args.signal_step_indices)
                ),
                discrepancy_block_names=(
                    ()
                    if args.signal_discrepancy_blocks is None
                    else tuple(str(value) for value in args.signal_discrepancy_blocks)
                ),
                seed=int(signal_seed),
            ),
        )
        torch.save(conditional_signal.tensors, run_dir / "conditional_signal_tensors.pt")
    else:
        conditional_signal = None

    sample_batch_size = min(int(args.sample_batch_size), int(args.num_sample_paths))
    generated_paths = sample_model_paths_in_batches(
        model,
        num_paths=int(args.num_sample_paths),
        x0=torch.tensor([[float(args.x0)]], device=device, dtype=dtype),
        seed=sample_seed,
        batch_size=sample_batch_size,
    )

    # For the fixed-reference-drift ablation, retain a common-random-number
    # reference bank as well.  Zero adjoint heads recover the fitted drift and
    # fitted reference volatility exactly, so the learned correction can be
    # compared with the law it is intended to improve.
    reference_generated_paths: torch.Tensor | None = None
    if bool(args.fixed_reference_drift_volatility_only):
        trained_checkpoint = model.checkpoint_state()
        with torch.no_grad():
            for parameter in model.network.parameters():
                parameter.zero_()
        reference_generated_paths = sample_model_paths_in_batches(
            model,
            num_paths=int(args.num_sample_paths),
            x0=torch.tensor([[float(args.x0)]], device=device, dtype=dtype),
            seed=sample_seed,
            batch_size=sample_batch_size,
        )
        model.load_checkpoint_state(trained_checkpoint)

    specific_entropy_ablation_paths: dict[str, torch.Tensor] = {}
    specific_entropy_r_effect: dict[str, float] | None = None
    if bool(args.specific_entropy_r_ablation):
        if str(args.control_map) != "specific_entropy":
            raise ValueError("specific-entropy-r-ablation requires --control-map specific_entropy")
        if int(args.specific_entropy_ablation_diagnostic_paths) < 1:
            raise ValueError("specific-entropy-ablation-diagnostic-paths must be >= 1")
        base_control = model.control_map
        diagnostic_paths = min(
            int(args.specific_entropy_ablation_diagnostic_paths),
            int(args.num_sample_paths),
        )
        with torch.no_grad():
            diagnostic_rollout = model.sample(
                num_paths=diagnostic_paths,
                x0=torch.tensor([[float(args.x0)]], device=device, dtype=dtype),
                seed=derive_stream_seed(base_seed=int(sample_seed), stream_offset=50_000),
            )
            specific_entropy_r_effect = specific_entropy_r_effect_on_realized_prefixes(
                base_control=base_control,
                rollout=diagnostic_rollout,
                state_dim=int(model.architecture.state_dim),
            )
        try:
            model.control_map = AdjointComponentAblationControl(
                base_control=base_control,
                zero_r=True,
            )
            specific_entropy_ablation_paths["r_zero"] = sample_model_paths_in_batches(
                model,
                num_paths=int(args.num_sample_paths),
                x0=torch.tensor([[float(args.x0)]], device=device, dtype=dtype),
                seed=sample_seed,
                batch_size=sample_batch_size,
            )
            model.control_map = AdjointComponentAblationControl(
                base_control=base_control,
                zero_p=True,
                zero_r=True,
            )
            specific_entropy_ablation_paths["p_r_zero"] = sample_model_paths_in_batches(
                model,
                num_paths=int(args.num_sample_paths),
                x0=torch.tensor([[float(args.x0)]], device=device, dtype=dtype),
                seed=sample_seed,
                batch_size=sample_batch_size,
            )
        finally:
            model.control_map = base_control

    lags = tuple(lag for lag in (1, 2, 5, 10, 20) if lag < int(args.num_grid_steps))
    observed_generated, observed_heldout, _ = discrepancy_evaluation_paths(
        generated_paths,
        heldout_paths,
        grid=grid,
        observed_only=bool(args.observed_only),
    )
    observed_lags = tuple(
        lag
        for lag in (1, 2, 5, 10, 20)
        if lag < int(observed_generated.shape[1]) - 1
    )
    metrics = {
        "run_dir": str(run_dir),
        "config": {
            "heston": heston_config.to_dict(),
            "model_grid": {
                "T": float(grid.T),
                "num_steps": int(grid.num_steps),
                "target_simulation_refinement": int(
                    args.target_simulation_refinement
                ),
                "observed_reference_fine_substeps": int(fine_substeps),
            },
            "seed": int(args.seed),
            "evaluation_seed": int(evaluation_seed),
            "heldout_seed": int(heldout_seed) if heldout_seed is not None else None,
            "sample_seed": int(sample_seed) if sample_seed is not None else None,
            "num_sample_paths": int(args.num_sample_paths),
            "sample_batch_size": int(sample_batch_size),
            "device": str(device),
            "dtype": args.dtype,
            "include_acf": bool(args.include_acf),
            "adjoint_input_mode": str(args.adjoint_input_mode),
            "preset": str(args.preset),
            "conditional_volatility_discrepancy": bool(args.conditional_volatility_discrepancy),
            "conditional_volatility_correlation_discrepancy": bool(
                args.conditional_volatility_correlation_discrepancy
            ),
            "conditional_volatility_correlation_full_matrix": bool(
                args.conditional_volatility_correlation_full_matrix
            ),
            "conditional_volatility_windows": [int(value) for value in args.conditional_volatility_windows],
            "conditional_volatility_correlation_windows": [
                int(value) for value in args.conditional_volatility_correlation_windows
            ],
            "conditional_volatility_discrepancy_windows": [int(value) for value in conditional_volatility_windows],
            "conditional_volatility_weight": float(args.conditional_volatility_weight),
            "conditional_volatility_correlation_weight": float(
                args.conditional_volatility_correlation_weight
            ),
            "conditional_volatility_correlation_scale_floor_fraction": float(
                args.conditional_volatility_correlation_scale_floor_fraction
            ),
            "conditional_volatility_bandwidths": [
                float(value) for value in args.conditional_volatility_bandwidths
            ],
            "time_marginal_volatility_window": (
                None
                if args.time_marginal_volatility_window is None
                else int(args.time_marginal_volatility_window)
            ),
            "time_marginal_volatility_endpoints": (
                []
                if args.time_marginal_volatility_endpoints is None
                else [int(value) for value in args.time_marginal_volatility_endpoints]
            ),
            "time_marginal_volatility_weight": float(args.time_marginal_volatility_weight),
            "observed_every": int(args.observed_every),
            "observed_count": int(grid.observed_count),
            "shadow_search": str(args.shadow_search),
            "lambda_scale": float(args.lambda_scale),
            "kappa_scale": float(args.kappa_scale),
            "ce_target_mode": str(args.ce_target_mode),
            "ridge_lambda": float(args.ridge_lambda),
            "ce_crossfit_folds": int(args.ce_crossfit_folds),
            "target_preconditioner": str(args.target_preconditioner),
            "preconditioner_batches": int(args.preconditioner_batches),
            "preconditioner_min_scale": float(args.preconditioner_min_scale),
            "noise_target_control_variate": str(args.noise_target_control_variate),
            "grad_clip_norm": float(args.grad_clip_norm) if args.grad_clip_norm > 0.0 else None,
            "control_map": control_summary,
            "fixed_reference_drift_volatility_only": bool(
                args.fixed_reference_drift_volatility_only
            ),
            "global_diffusion_scale": float(global_sigma),
            "eta": float(args.eta) if str(args.control_map) == "specific_entropy" else None,
            "sigma_min": float(args.sigma_min),
            "sigma_max": float(args.sigma_max),
            "reference": control_summary["reference"],
            "save_checkpoint": bool(args.save_checkpoint),
            "frozen_response_audit": bool(args.frozen_response_audit),
            "specific_entropy_r_ablation": bool(args.specific_entropy_r_ablation),
        },
        "fit_final": fit_result.final_metrics,
        "stylized": stylized_fact_metrics(
            generated_paths,
            heldout_paths,
            lags=lags or (1,),
            tail_quantiles=(0.90, 0.95, 0.99),
            include_swd=True,
            swd_projections=128,
        ),
        "stylized_observed_grid": stylized_fact_metrics(
            observed_generated,
            observed_heldout,
            lags=observed_lags or (1,),
            tail_quantiles=(0.90, 0.95, 0.99),
            include_swd=True,
            swd_projections=128,
        ),
    }

    if reference_generated_paths is not None:
        observed_reference, _, _ = discrepancy_evaluation_paths(
            reference_generated_paths,
            heldout_paths,
            grid=grid,
            observed_only=bool(args.observed_only),
        )
        metrics["reference_stylized"] = stylized_fact_metrics(
            reference_generated_paths,
            heldout_paths,
            lags=lags or (1,),
            tail_quantiles=(0.90, 0.95, 0.99),
            include_swd=True,
            swd_projections=128,
        )
        metrics["reference_stylized_observed_grid"] = stylized_fact_metrics(
            observed_reference,
            observed_heldout,
            lags=observed_lags or (1,),
            tail_quantiles=(0.90, 0.95, 0.99),
            include_swd=True,
            swd_projections=128,
        )

    if specific_entropy_r_effect is not None:
        metrics["specific_entropy_r_ablation"] = {
            "interpretation": {
                "full": "learned P and R",
                "r_zero": "learned P retained; R forced to zero",
                "p_r_zero": "both P and R forced to zero; fitted sigma_ref only",
                "common_random_numbers": True,
            },
            "r_effect_on_full_realized_prefixes": specific_entropy_r_effect,
            "stylized": {
                "full": metrics["stylized"],
                **{
                    name: stylized_fact_metrics(
                        paths,
                        heldout_paths,
                        lags=lags or (1,),
                        tail_quantiles=(0.90, 0.95, 0.99),
                        include_swd=True,
                        swd_projections=128,
                    )
                    for name, paths in specific_entropy_ablation_paths.items()
                },
            },
        }

    if bool(args.discrepancy_sensitivity):
        sensitivity_count = min(
            int(args.sensitivity_paths),
            int(generated_paths.shape[0]),
            int(heldout_paths.shape[0]),
        )
        if sensitivity_count < 2:
            raise ValueError("sensitivity_paths must allow at least two generated and target paths")
        sensitivity_generated, sensitivity_target, sensitivity_grid = discrepancy_evaluation_paths(
            generated_paths[:sensitivity_count],
            heldout_paths[:sensitivity_count],
            grid=grid,
            observed_only=bool(args.observed_only),
        )
        metrics["discrepancy_diffusion_scale_sensitivity"] = discrepancy_diffusion_scale_sensitivity(
            discrepancy=discrepancy,
            generated_paths=sensitivity_generated,
            target_paths=sensitivity_target,
            grid=sensitivity_grid,
        )

    if conditional_signal is not None:
        metrics["conditional_signal_diagnostic"] = conditional_signal.metrics

    if frozen_response is not None:
        metrics["frozen_response_audit"] = {
            "report": "frozen_response_audit.json",
            "tensors": "frozen_response_tensors.pt",
            "candidate_correlation_block_not_used_in_training": bool(
                frozen_response.metrics["candidate_correlation_block_not_used_in_training"]
            ),
        }

    if bool(args.scores):
        metrics["benchmark_scores"] = benchmark_scores(
            generated_paths,
            heldout_paths,
            config=BenchmarkScoreConfig(
                train_steps=int(args.score_steps),
                batch_size=int(args.score_batch_size),
                hidden_dim=int(args.score_hidden_dim),
                lr=float(args.score_lr),
                seed=int(evaluation_seed) + 1000,
            ),
        )

    if bool(args.shadowing):
        prefix_length = (
            int(args.shadow_prefix_length)
            if args.shadow_prefix_length is not None
            else max(2, int(grid.observed_count) // 2 + 1 if bool(args.observed_only) else int(args.num_grid_steps) // 2 + 1)
        )
        real_count = min(int(args.shadow_real_paths), int(heldout_paths.shape[0]))
        shadow_result = evaluate_path_shadowing(
            real_paths=heldout_paths[:real_count],
            generated_paths=generated_paths,
            prefix_length=prefix_length,
            top_k=min(int(args.shadow_top_k), int(generated_paths.shape[0])),
            future_horizon=None if args.shadow_future_steps is None else int(args.shadow_future_steps),
            search_backend=str(args.shadow_search),
            exact_batch_size=int(args.shadow_exact_batch_size),
        )
        metrics["shadowing"] = shadow_result.metrics
    else:
        shadow_result = None

    write_json(run_dir / "metrics.json", metrics)
    write_history_jsonl(run_dir / "history.jsonl", fit_result.history)

    if bool(args.save_tensors):
        torch.save(target_paths.detach().cpu(), run_dir / "target_paths.pt")
        torch.save(heldout_paths.detach().cpu(), run_dir / "heldout_paths.pt")
        torch.save(generated_paths.detach().cpu(), run_dir / "generated_paths.pt")
        if reference_generated_paths is not None:
            torch.save(
                reference_generated_paths.detach().cpu(),
                run_dir / "reference_generated_paths.pt",
            )
        for name, paths in specific_entropy_ablation_paths.items():
            torch.save(paths.detach().cpu(), run_dir / f"generated_{name}_paths.pt")

    if not bool(args.no_plots):
        try:
            from path_dt_experiments.plots import plot_heston_diagnostics, plot_shadow_fan

            fig = plot_heston_diagnostics(
                target_paths=heldout_paths.detach().cpu(),
                generated_paths=generated_paths.detach().cpu(),
                output_path=run_dir / "heston_diagnostics.png",
                lags=lags or (1,),
            )
            import matplotlib.pyplot as plt

            plt.close(fig)
            if specific_entropy_ablation_paths:
                from path_dt_experiments.plots import plot_heston_diagnostics_comparison

                fig = plot_heston_diagnostics_comparison(
                    target_paths=heldout_paths.detach().cpu(),
                    candidate_paths={
                        "Full P,R": generated_paths.detach().cpu(),
                        "R=0": specific_entropy_ablation_paths["r_zero"].detach().cpu(),
                        "P=R=0 (sigma ref only)": (
                            specific_entropy_ablation_paths["p_r_zero"].detach().cpu()
                        ),
                    },
                    output_path=run_dir / "specific_entropy_r_ablation.png",
                    lags=lags or (1,),
                )
                plt.close(fig)
            if shadow_result is not None:
                fig = plot_shadow_fan(
                    real_paths=heldout_paths[: min(int(args.shadow_real_paths), int(heldout_paths.shape[0]))]
                    .detach()
                    .cpu(),
                    generated_paths=generated_paths.detach().cpu(),
                    selected_indices=shadow_result.selected_indices.detach().cpu(),
                    prefix_length=(
                        int(args.shadow_prefix_length)
                        if args.shadow_prefix_length is not None
                        else max(
                            2,
                            int(grid.observed_count) // 2 + 1
                            if bool(args.observed_only)
                            else int(args.num_grid_steps) // 2 + 1,
                        )
                    ),
                    output_path=run_dir / "shadow_fan.png",
                )
                plt.close(fig)
        except ImportError as exc:
            write_json(run_dir / "plot_error.json", {"error": str(exc)})

    print(f"wrote Heston run to {run_dir}")
    print(f"stylized metrics: {metrics['stylized']}")
    if "benchmark_scores" in metrics:
        print(f"benchmark scores: {metrics['benchmark_scores']}")
    if "shadowing" in metrics:
        print(f"shadowing metrics: {metrics['shadowing']}")


if __name__ == "__main__":
    main()
