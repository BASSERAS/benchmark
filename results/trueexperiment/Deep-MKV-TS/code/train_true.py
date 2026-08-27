#!/usr/bin/env python3
"""Deep-MKV-TS generator training on TrueDataset (om_2022-07_N6144, d = 8).

This is the published Heston trainer
(``results/HestonMultiAsset/Deep-MKV-TS/code/train_multiasset.py``) ported to
real 30-second crypto bars.  The exported surface, the constant names, the
callback structure, the seed lock and the checkpoint grid are kept name for
name so the two trees stay diffable.  Algorithm 1 is unchanged: the same
frozen reference package, the same control map, the same 13-block discrepancy,
the same architecture, the same optimiser.

WHAT CHANGED, AND WHY -- there are exactly four departures.

1.  ``DT`` and the sequence length come from ``fit_reference_true``:
    ``dt = 9.512937595129376e-07`` years (30-second bars, 1,051,200 per year)
    and ``num_steps = 127``.  Heston is ``dt = 1/252`` with 251 steps.  Nothing
    else in the file knows about the panel.

2.  ``sigma_max`` is NOT a module constant.  Heston hardcodes ``SIGMA_MAX =
    0.6``; on this panel five of the eight assets have annualised volatility
    above 0.6 and the top covariance eigenvalue is 1.80, so 0.6 would clamp the
    market away and STILL train to completion with a smooth loss curve.  The
    ceiling is read from the frozen ``reference_kernel.json``, which is where
    ``sweep_reference_true.py`` recorded the value it selected.  The control map
    and the reference kernel then clip with the SAME bound by construction,
    instead of by two constants that agree until someone edits one of them.

3.  The MMD bandwidths are regauged by ``calibrate_bandwidths_true.py``.  The
    published bandwidths are ABSOLUTE and were tuned at Heston's scale; on this
    panel the price features shrink by ~27x and the realized-variance features
    grow by ~42x, which drives the price blocks' gradient three orders of
    magnitude below the volatility blocks'.  The gauge multiplies each block's
    bandwidths by that block's measured pcRMS ratio, restoring the Heston
    balance.  Same features, same weights, same geometry -- only the ruler
    changes.  See ``losses/bandwidth_gauge.json``.

4.  ``ridge_lambda`` is re-selected for this panel by
    ``sweep_ridge_lambda_true.py`` on the VALIDATION bank, exactly as Heston
    re-selected it for d = 8.  ``HESTON_RIDGE_LAMBDA`` is kept as a named
    constant so ``retuned_for_true`` in the config is derived from the
    comparison and never hand-asserted.

Everything else -- eta, sigma_min, lambda_scale, kappa_scale, the ACF weights,
hidden dim, layers, batch sizes, lr, weight decay, grad clip, the CE ridge, the
adjoint weights, the two backends, the checkpoint grid, float32 -- is the
committed Heston value.

Usage
-----
    R=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 \
    PYTHONPATH="$R/src:$R/experiments" taskset -c 0-7 \
    /home/tbasseras/gpu-venv/bin/python train_true.py \
        --seed 0 --device cuda:0 --steps 3000
"""

from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import json
import platform
import os
import sys
import time
from datetime import date
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
METHOD_ROOT = HERE.parent
REFERENCE = Path("/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference")
for _path in (HERE, REFERENCE / "src", REFERENCE / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from deep_mkv_gen_path_dt.ce.ridge import RidgeConditionalExpectation  # noqa: E402
from deep_mkv_gen_path_dt.config import (  # noqa: E402
    DiscreteMPArchitectureConfig,
    DiscreteMPTrainingConfig,
)
from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid  # noqa: E402
from deep_mkv_gen_path_dt.model import DiscreteMPModel  # noqa: E402
from path_dt_experiments.discrepancies import build_heston_discrepancy  # noqa: E402

import matrix_control_multiasset as mc  # noqa: E402
import eigh_fallback  # noqa: E402
import eigh_float64  # noqa: E402
from calibrate_bandwidths_true import (  # noqa: E402
    DEFAULT_GAUGE,
    apply_bandwidth_gauge,
    load_bandwidth_gauge,
)
from fit_reference_true import (  # noqa: E402
    DATASET,
    DT,
    SEQ_TAG,
    SIGMA_MIN,
    STATE_DIM,
    load_kernel,
    load_log_prices,
)

# Batched cuSOLVER `eigh` on BATCH_SIZE * num_steps symmetric 8x8 matrices per
# outer iteration can fail to converge; the shim retries the SAME decomposition
# on CPU LAPACK and raises when Theta is non-finite. No mathematics changes.
eigh_fallback.install()
# The eigh BACKWARD carries 1/(lambda_i - lambda_j). Measured on the frozen
# kernel, 0.891% of the real sigma_ref matrices have a relative eigenvalue gap
# at or below float32 eps, so float32 rounds the two eigenvalues to the same
# bits, the gap evaluates to exactly 0.0 and the gradient returns NaN -- which
# is what killed training at optimizer step 3. In float64 the gaps resolve
# (0/910336 pairs below 1e-9) and max|grad| is IDENTICAL to float32's, so this
# promotes precision only: no clipping, no jitter, no damping, no setting moves.
# Installed AFTER the fallback so float64 is the outer layer.
eigh_float64.install()

# ---- paper settings, byte-for-byte the committed Heston d = 8 run ---------- #
ETA = 1.0
LAMBDA_SCALE = 50.0
KAPPA_SCALE = 100.0
ABS_RETURN_ACF_WEIGHT = 0.25
SQUARED_RETURN_ACF_WEIGHT = 0.125
HIDDEN_DIM = 96
NUM_LAYERS = 1
BATCH_SIZE = 256
TARGET_BATCH_SIZE = 256
LR = 2e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 5.0
CE_RIDGE = 1e-3
LOG_EVERY = 100
CHECKPOINT_STEPS = (500, 1000, 1500, 2000, 2500, 3000)
MAX_EIGH_BATCH = 32768
DTYPE = torch.float32
DISCREPANCY_PRESET = "old_fullv_w0p25"

# The committed Heston d = 8 value.  Kept as a named constant, exactly as that
# file kept ``D1_RIDGE_LAMBDA``, so ``retuned_for_true`` in the config is
# derived from a comparison instead of hand-asserted.
HESTON_RIDGE_LAMBDA = 1000.0
# Same role for the learning rate: LR above IS this number, so a retuned run has
# to be detected by comparison rather than by trusting the launcher.
HESTON_LR = 2e-3
# Provisional: the Heston value, used until sweep_ridge_lambda_true.py
# re-selects it on the VALIDATION bank for this panel.  Selection there is on
# |log NNratio| subject to the real-vs-real envelope, per the TrueDataset
# guideline section 7 -- never on goodness of fit.
RIDGE_LAMBDA = 1000.0

# Heston's hardcoded ceiling.  Present only so the config can record that this
# panel does not use it and why; the live value comes from the frozen kernel.
HESTON_SIGMA_MAX = 0.6

LOSS_FIELDS = (
    "step",
    "phase",
    "loss_total",
    "objective",
    "grad_norm",
    "control_rms",
    "elapsed_sec",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build_model(
    *,
    grid: DiscreteTimeGrid,
    device: torch.device,
    seed: int,
    kernel_steps: int,
    gauge_path: Path = DEFAULT_GAUGE,
) -> tuple[DiscreteMPModel, float]:
    """Assemble Algorithm 1's generator, control and objective on TrueDataset.

    Returns the model and the ``sigma_max`` actually in force, which is read
    from the frozen kernel rather than from a constant in this file so that the
    control map cannot drift away from the kernel it was fitted with.
    """

    kernel = load_kernel(
        device=device,
        dtype=DTYPE,
        num_steps=kernel_steps,
        differentiable_sigma=True,
    )
    sigma_max = float(kernel.sigma_max)
    kernel_sigma_min = float(kernel.sigma_min)
    if kernel_sigma_min != float(SIGMA_MIN):
        raise SystemExit(
            f"frozen kernel was fitted with sigma_min={kernel_sigma_min} but this "
            f"module uses {SIGMA_MIN}; the control map and the kernel would clip "
            "differently"
        )
    if sigma_max <= float(SIGMA_MIN):
        raise SystemExit(
            f"frozen kernel has a degenerate ceiling sigma_max={sigma_max}"
        )
    print(
        f"[control] sigma clip [{SIGMA_MIN}, {sigma_max}] read from the frozen "
        "reference kernel",
        flush=True,
    )

    control = mc.MultivariateReferenceDriftSpecificEntropyMatrixControl(
        dt=grid.dt,
        eta=ETA,
        reference_kernel=kernel,
        sigma_min=SIGMA_MIN,
        sigma_max=sigma_max,
        allow_drift_correction=False,
        max_eigh_batch=MAX_EIGH_BATCH,
    )
    discrepancy = build_heston_discrepancy(
        num_steps=int(grid.num_steps),
        include_acf=True,
        preset=DISCREPANCY_PRESET,
        lambda_scale=LAMBDA_SCALE,
        kappa_scale=KAPPA_SCALE,
        include_conditional_volatility=False,
        include_conditional_volatility_correlation=False,
        abs_return_acf_weight=ABS_RETURN_ACF_WEIGHT,
        squared_return_acf_weight=SQUARED_RETURN_ACF_WEIGHT,
    )
    # Absolute bandwidths tuned at Heston's scale are meaningless here; regauge
    # before the objective is ever evaluated. apply_bandwidth_gauge raises if
    # the gauge and the stack disagree about a single block, rather than
    # silently leaving Heston bandwidths in place.
    discrepancy = apply_bandwidth_gauge(discrepancy, load_bandwidth_gauge(gauge_path))
    print(f"[discrepancy] bandwidths regauged from {gauge_path}", flush=True)

    model = DiscreteMPModel(
        architecture=DiscreteMPArchitectureConfig(
            state_dim=STATE_DIM,
            noise_dim=STATE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            adjoint_dim=STATE_DIM,
            noise_adjoint_dim=STATE_DIM * STATE_DIM,
        ),
        grid=grid,
        dynamics_step=mc.MatrixDriftVolatilityEulerStep(dt=grid.dt),
        control_map=control,
        discrepancy=discrepancy,
        ce_estimator=RidgeConditionalExpectation(ridge=CE_RIDGE),
        init_seed=int(seed),
    )
    model.network.to(device=device, dtype=DTYPE)
    return model, sigma_max


def build_training(
    *, seed: int, steps: int, ridge_lambda: float | None = None,
    lr: float | None = None,
) -> DiscreteMPTrainingConfig:
    """The Heston ``build_training(...)`` block, unchanged.

    ``ridge_lambda`` defaults to the module constant.  It is exposed as a
    keyword only so ``sweep_ridge_lambda_true.py`` can re-select it on the
    VALIDATION bank without forking this file.  The production campaign never
    passes it, so it always uses the constant.

    ``lr`` is exposed for the same reason and defaults to the same constant.
    Why it needs to be exposed at all: ``LR`` is declared above under "paper
    settings, byte-for-byte the committed Heston d = 8 run", and that is the
    problem.  The control is ``Theta = eta*sigma_ref^-1 + Zhat/sqrt(dt)``, so
    ``1/sqrt(dt)`` is a gain on the network output: 15.87 at Heston's
    dt = 1/252, but 1025.28 at this panel's dt = 9.51e-07, a factor of 64.6.
    The optimiser is AdamW, whose per-coordinate step is ~lr regardless of
    gradient scale, so the Theta displacement per step is lr/sqrt(dt) --
    0.032 on Heston, 2.05 here.  Measured consequence: the learned term is
    194x the reference term, i.e. sigma_ref is erased and Theta is pure
    noise-head output.  Nothing in the checkpoint grid escapes it (even step
    500 is 72-477x too strong), because the broken axis is control MAGNITUDE
    and selection only searched checkpoint STEP.
    """

    ridge = RIDGE_LAMBDA if ridge_lambda is None else float(ridge_lambda)
    return DiscreteMPTrainingConfig(
        batch_size=BATCH_SIZE,
        target_batch_size=TARGET_BATCH_SIZE,
        num_steps=int(steps),
        lr=LR if lr is None else float(lr),
        weight_decay=WEIGHT_DECAY,
        lambda_scale=LAMBDA_SCALE,
        adjoint_weight=0.0,
        adjoint_noise_weight=1.0,
        ce_target_mode="ridge",
        ridge_lambda=ridge,
        ce_crossfit_folds=1,
        target_preconditioner="none",
        noise_target_control_variate="none",
        grad_clip_norm=GRAD_CLIP_NORM,
        log_every=LOG_EVERY,
        seed=int(seed),
        observed_only=False,
        path_derivative_backend="autograd",
        drift_adjoint_backend="autograd_replay",
    )


def _progress(
    *,
    model: DiscreteMPModel,
    checkpoint_dir: Path,
    checkpoint_steps: tuple[int, ...],
    started: float,
    rows: list[dict[str, float]],
    losses_path: Path | None = None,
):
    """Print one line per logged step and dump the requested checkpoints.

    ``losses_path`` is rewritten on every logged step and ``latest.pt`` is a
    rolling checkpoint, so a crash costs at most ``LOG_EVERY`` steps instead of
    the whole run.  ``CHECKPOINT_STEPS`` is the SELECTION grid and is
    deliberately untouched: ``select_checkpoint_true.py`` chooses among those
    steps, so adding to it would change which checkpoints compete and therefore
    change the method.  The rolling file has a different name and is never a
    selection candidate.
    """

    wanted = {int(step) for step in checkpoint_steps}

    def _save_atomic(payload, path: Path) -> None:
        # A crash midway through torch.save leaves a truncated file that loads
        # as garbage. Write beside the target, then rename -- rename is atomic
        # on the same filesystem, so the visible file is always complete.
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def report(_model: DiscreteMPModel, metrics: dict[str, float]) -> None:
        step = int(metrics.get("step", 0))
        loss = float(metrics.get("train_total_loss", float("nan")))
        objective = float(metrics.get("complete_objective_value", float("nan")))
        grad = float(metrics.get("grad_norm", float("nan")))
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "step": step,
                "phase": "train",
                "loss_total": loss,
                "objective": objective,
                "grad_norm": grad,
                "control_rms": float(metrics.get("control_rms", float("nan"))),
                "elapsed_sec": elapsed,
            }
        )
        print(
            f"step={step:5d} loss={loss:.6g} objective={objective:.6g} "
            f"grad={grad:.4g} elapsed={elapsed / 60.0:.1f}min",
            flush=True,
        )
        # Non-finite loss means the control has diverged. Report the STEP INDEX:
        # a wall-clock crash time does not say where it died, and the exception
        # raised downstream blames the eigensolver instead.
        if loss != loss or objective != objective:
            print(
                f"[diverged] step={step} produced a non-finite value "
                f"(loss={loss}, objective={objective}, grad={grad}). The control "
                "has blown up; the eigensolver will fail next with a misleading "
                "'ill-conditioned' message.",
                file=sys.stderr,
                flush=True,
            )

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if step in wanted:
            _save_atomic(
                model.checkpoint_state(), checkpoint_dir / f"step_{step:04d}.pt"
            )
        _save_atomic(
            {"step": step, "state": model.checkpoint_state()},
            checkpoint_dir / "latest.pt",
        )
        if losses_path is not None:
            _write_losses(losses_path, rows)

    return report


def _acquire_seed_lock(run_dir: Path, seed: int) -> Path:
    """Refuse to start if a LIVE process already owns this seed's output dir.

    Two trainers on one seed interleave writes into the same ``latest.pt`` and
    the same ``losses/seed_N_losses.csv``.  That does not fail loudly -- it
    produces a loadable checkpoint and a plausible loss curve that are both
    silently wrong.  Liveness is tested with ``os.kill(pid, 0)`` so a lock left
    by a crashed run is reclaimed instead of blocking every future attempt.
    """

    lock = run_dir / ".lock"
    if lock.exists():
        try:
            owner = int(lock.read_text().split()[0])
        except (ValueError, IndexError):
            owner = -1
        alive = False
        if owner > 0:
            try:
                os.kill(owner, 0)
            except ProcessLookupError:
                alive = False
            except PermissionError:
                # Exists but belongs to another user: treat as live, never steal.
                alive = True
            else:
                alive = True
        if alive:
            sys.exit(
                f"[lock] REFUSING TO START: seed {seed} is already being trained "
                f"by PID {owner} (lock file {lock}). Two trainers on one seed "
                "corrupt both outputs without failing loudly. If that PID is "
                "demonstrably not a trainer, delete the lock file and retry."
            )
        print(f"[lock] reclaiming stale lock left by dead PID {owner}", flush=True)

    lock.write_text(f"{os.getpid()}\n", encoding="utf-8")
    atexit.register(lambda: lock.unlink(missing_ok=True))
    print(f"[lock] seed {seed} owned by PID {os.getpid()}", flush=True)
    return lock


def _write_losses(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LOSS_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def _write_config(
    path: Path,
    *,
    seed: int,
    steps: int,
    num_paths: int,
    num_steps: int,
    parameters: int,
    elapsed: float,
    device: torch.device,
    sigma_max: float,
    ridge_lambda: float,
    gauge_path: Path,
    lr: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Derived, never hand-asserted: a value is listed as retuned only if it
    # actually differs from the committed Heston d = 8 value.
    retuned: list[str] = []
    if float(ridge_lambda) != HESTON_RIDGE_LAMBDA:
        retuned.append("ridge_lambda")
    # LR is itself the committed Heston value, so the comparison is against the
    # constant. A run that lowers it is doing the dt correction and must say so
    # in the artefact, not only in the launcher.
    if float(lr) != HESTON_LR:
        retuned.append("lr")
    if float(sigma_max) != HESTON_SIGMA_MAX:
        retuned.append("sigma_max")
    retuned.append("mmd_bandwidths")
    kernel_path = HERE / "reference" / "reference_kernel.json"
    payload = {
        "method": "Deep-MKV-TS",
        "dataset": "TrueDataset",
        "data_dir": str(DATASET),
        "seq_tag": SEQ_TAG,
        "seed": int(seed),
        "d": STATE_DIM,
        "joint_or_per_asset": "joint",
        "retuned_for_true": sorted(set(retuned)),
        "state_dim": STATE_DIM,
        "dt": DT,
        "seq_len": int(num_steps) + 1,
        "n_train": int(num_paths),
        "n_steps": int(steps),
        "n_parameters": int(parameters),
        "batch_size": BATCH_SIZE,
        "target_batch_size": TARGET_BATCH_SIZE,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "adjoint_dim": STATE_DIM,
        "noise_adjoint_dim": STATE_DIM * STATE_DIM,
        "lr": float(lr),
        "heston_lr": HESTON_LR,
        # The dimensionless quantity the lr is actually chosen to control:
        # AdamW moves each coordinate by ~lr per step, and the control divides
        # the network output by sqrt(dt), so this is the Theta displacement per
        # optimiser step. Heston's value is 0.032; anything near 2.0 means the
        # reference SDE is being erased.
        "theta_step_per_optimiser_step": float(lr) / (DT ** 0.5),
        "weight_decay": WEIGHT_DECAY,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "eta": ETA,
        "sigma_min": SIGMA_MIN,
        "sigma_max": float(sigma_max),
        "sigma_max_source": "frozen reference_kernel.json (selected by sweep)",
        "heston_sigma_max": HESTON_SIGMA_MAX,
        "lambda_scale": LAMBDA_SCALE,
        "kappa_scale": KAPPA_SCALE,
        "discrepancy_preset": DISCREPANCY_PRESET,
        "abs_return_acf_weight": ABS_RETURN_ACF_WEIGHT,
        "squared_return_acf_weight": SQUARED_RETURN_ACF_WEIGHT,
        "bandwidth_gauge": str(gauge_path),
        "bandwidth_gauge_sha256_16": _digest(gauge_path),
        "adjoint_weight": 0.0,
        "adjoint_noise_weight": 1.0,
        "ce_target_mode": "ridge",
        "ridge_lambda": float(ridge_lambda),
        "heston_ridge_lambda": HESTON_RIDGE_LAMBDA,
        "ce_ridge": CE_RIDGE,
        "path_derivative_backend": "autograd",
        "drift_adjoint_backend": "autograd_replay",
        "differentiable_sigma": True,
        "max_eigh_batch": MAX_EIGH_BATCH,
        # How many times batched cuSOLVER `eigh` failed on this seed's Theta and
        # the decomposition had to be redone on CPU LAPACK. Zero is expected;
        # non-zero does not invalidate the run (same decomposition, other
        # backend) but belongs in the audit record, not only in an uncommitted
        # log file.
        "eigh_fallback": eigh_fallback.counters(),
        # The eigh BACKWARD divides by the eigenvalue gap, and 0.891% of the
        # frozen kernel's real sigma_ref matrices have a relative gap at or
        # below float32 eps, so float32 sees an exact tie and returns NaN. These
        # counts say how many calls each of the three sites ran in float64
        # instead. max|grad| was measured identical in the two precisions, so
        # this is a precision change only -- no setting in this config moves.
        "eigh_float64": eigh_float64.counters(),
        "eigh_float64_rationale": (
            "LinalgEighBackward0 returned NaN at specific_entropy_matrix.py:133 "
            "on float32 eigenvalue ties; float64 resolves the gaps "
            "(0/910336 pairs below 1e-9) with identical gradients"
        ),
        "control": "MultivariateReferenceDriftSpecificEntropyMatrixControl",
        "reference_kernel": "code/reference/reference_kernel.json",
        "reference_kernel_sha256_16": (
            _digest(kernel_path) if kernel_path.is_file() else None
        ),
        "paper_hyperparams": False,
        "dtype": "float32",
        "train_time_sec": float(elapsed),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor()
        ),
        "torch": torch.__version__,
        "date": date.today().isoformat(),
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-paths", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument(
        "--ridge-lambda",
        type=float,
        default=None,
        help="override RIDGE_LAMBDA; used only by sweep_ridge_lambda_true.py",
    )
    parser.add_argument("--gauge", type=Path, default=DEFAULT_GAUGE)
    parser.add_argument(
        "--checkpoint-steps", type=int, nargs="*", default=list(CHECKPOINT_STEPS)
    )
    parser.add_argument("--run-root", type=Path, default=HERE / "runs")
    parser.add_argument(
        "--lr", type=float, default=None,
        help=f"AdamW learning rate (default: the Heston constant {LR:g}). "
             "Theta moves ~lr/sqrt(dt) per step, so at this panel's dt the "
             "Heston value is 64.6x too large.",
    )
    parser.add_argument(
        "--out-root", type=Path, default=None,
        help="directory to hold weights/ and losses/ (default: the published "
             "Deep-MKV-TS method root). A retune MUST pass this, otherwise it "
             "overwrites the committed seed_<S>_model.pt.",
    )
    parser.add_argument(
        "--time-only",
        type=int,
        default=0,
        help="run this many steps, report seconds/step, write nothing",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    device = torch.device(args.device)
    ridge_lambda = (
        RIDGE_LAMBDA if args.ridge_lambda is None else float(args.ridge_lambda)
    )

    target = load_log_prices(num_paths=args.num_paths, device=device).to(DTYPE)
    if args.num_steps is not None:
        target = target[:, : int(args.num_steps) + 1, :].contiguous()
    num_paths = int(target.shape[0])
    num_steps = int(target.shape[1]) - 1
    grid = DiscreteTimeGrid(T=num_steps * DT, num_steps=num_steps)
    print(
        f"[data] train log prices {tuple(target.shape)} dt={DT} device={device}",
        flush=True,
    )

    model, sigma_max = build_model(
        grid=grid,
        device=device,
        seed=int(args.seed),
        kernel_steps=num_steps,
        gauge_path=Path(args.gauge),
    )
    parameters = sum(p.numel() for p in model.network.parameters())
    print(f"[model] parameters={parameters} ridge_lambda={ridge_lambda}", flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if int(args.time_only) > 0:
        training = build_training(
            seed=int(args.seed),
            steps=int(args.time_only),
            ridge_lambda=ridge_lambda,
        )
        started = time.perf_counter()
        model.fit(target, training=training)
        elapsed = time.perf_counter() - started
        per_step = elapsed / float(args.time_only)
        print(
            f"[time] {args.time_only} steps in {elapsed:.1f}s = {per_step:.3f} s/step "
            f"-> {per_step * float(args.steps) / 60.0:.1f} min for {args.steps} steps",
            flush=True,
        )
        if device.type == "cuda":
            print(
                "[time] peak GPU memory "
                f"{torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB",
                flush=True,
            )
        return

    run_dir = Path(args.run_root) / f"seed_{int(args.seed)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _acquire_seed_lock(run_dir, int(args.seed))
    checkpoint_dir = run_dir / "training_checkpoints"
    rows: list[dict[str, float]] = []
    lr = LR if args.lr is None else float(args.lr)
    # Every write below goes here. Defaults to METHOD_ROOT so the production
    # campaign is unchanged; a retune passes --out-root and cannot touch the
    # committed weights even by accident.
    out_root = Path(args.out_root) if args.out_root is not None else METHOD_ROOT
    print(
        f"[lr] {lr:g} (Heston constant {HESTON_LR:g}); "
        f"Theta displacement per optimiser step = lr/sqrt(dt) = "
        f"{lr / (DT ** 0.5):.4g} (Heston: {HESTON_LR / (1.0 / 252.0) ** 0.5:.4g})",
        flush=True,
    )
    training = build_training(
        seed=int(args.seed), steps=int(args.steps), ridge_lambda=ridge_lambda,
        lr=lr,
    )

    started = time.perf_counter()
    fit = model.fit(
        target,
        training=training,
        log_callback=_progress(
            model=model,
            checkpoint_dir=checkpoint_dir,
            checkpoint_steps=tuple(int(s) for s in args.checkpoint_steps),
            started=started,
            rows=rows,
            # Flushed every logged step so a crash still leaves the loss history.
            losses_path=out_root / "losses" / f"seed_{int(args.seed)}_losses.csv",
        ),
    )
    elapsed = time.perf_counter() - started

    missing = [
        int(step)
        for step in args.checkpoint_steps
        if not (checkpoint_dir / f"step_{int(step):04d}.pt").is_file()
    ]
    if missing:
        raise RuntimeError(f"training finished without checkpoints {missing}")

    torch.save(model.checkpoint_state(), run_dir / "model_checkpoint.pt")
    (run_dir / "history.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in fit.history),
        encoding="utf-8",
    )

    torch.save(
        model.network.state_dict(),
        out_root / "weights" / f"seed_{int(args.seed)}_model.pt",
    )
    _write_losses(out_root / "losses" / f"seed_{int(args.seed)}_losses.csv", rows)
    _write_config(
        out_root / "weights" / f"seed_{int(args.seed)}_config.json",
        seed=int(args.seed),
        steps=int(args.steps),
        num_paths=num_paths,
        num_steps=num_steps,
        parameters=parameters,
        elapsed=elapsed,
        device=device,
        sigma_max=sigma_max,
        ridge_lambda=ridge_lambda,
        gauge_path=Path(args.gauge),
        lr=lr,
    )
    (run_dir / "COMPLETE.json").write_text(
        json.dumps(
            {
                "seed": int(args.seed),
                "steps": int(args.steps),
                "train_time_sec": float(elapsed),
                "final_loss": float(rows[-1]["loss_total"]) if rows else float("nan"),
                "peak_gpu_gib": (
                    float(torch.cuda.max_memory_allocated(device) / 2**30)
                    if device.type == "cuda"
                    else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[done] seed={args.seed} {elapsed / 60.0:.1f} min", flush=True)


if __name__ == "__main__":
    main()
