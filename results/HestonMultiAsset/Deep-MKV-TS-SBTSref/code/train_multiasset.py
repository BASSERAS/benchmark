#!/usr/bin/env python3
"""Deep-MKV-TS Algorithm 1 with the **SBTS** reference model, at d = 8.

This is the sibling ``results/HestonMultiAsset/Deep-MKV-TS/code/train_multiasset.py``
with exactly ONE thing changed: the reference model.  Everything else -- the
control, the Euler step, the discrepancy, the architecture, the optimiser, the
checkpoint grid -- is the published Deep-MKV-TS configuration, so any difference
in the benchmark tables is attributable to the reference, not to a re-tuned
pipeline.

What changed
------------
``fit_reference_multiasset.load_kernel`` (the Guyon-Lekeufack-style cross-asset
covariance kernel) is replaced by ``sbts_reference.build_sbts_reference_kernel``.
The new kernel supplies

    b^ref_i = E_w[R^m_{i+1}] / dt        kernel-weighted mean RAW log-return of
                                         the training bank at the date being
                                         generated
    sigma^ref = diag(sigma_j / sqrt(dt)) constant in time and across paths

where ``w`` are the SBTS Markovian weights over the last ``K`` realised scaled
returns of the prefix.  The derivation, the log-weight accumulation (a naive
product underflows float32 for EVERY bank path at K = 20, h = 0.31), the
matmul distance expansion and the analytic Jacobian all live in
``sbts_reference.py``; see its module docstring.

``sigma^ref`` here is the SBTS *corrected* volatility -- the one that actually
generated the SBTS paths after the divide-by-vol / re-multiply-by-vol trick --
not the unit diffusion SBTS uses in its internal scaled-return coordinate.

What is IDENTICAL to the sibling
--------------------------------
    eta 1.0 | sigma_min 1e-3 | sigma_max 0.6 | lambda_scale 50 | kappa_scale 100
    discrepancy build_heston_discrepancy(preset="old_fullv_w0p25", include_acf=True,
        abs_return_acf_weight=0.25, squared_return_acf_weight=0.125,
        include_conditional_volatility=False,
        include_conditional_volatility_correlation=False)
    ce estimator RidgeConditionalExpectation(ridge=1e-3)
    architecture GRU hidden 96, 1 layer, adjoint_dim 8, noise_adjoint_dim 64
    batch_size 256 | target_batch_size 256 | num_steps 3000
    weight_decay 1e-5 | grad_clip_norm 5.0
    ce_target_mode "ridge" | ce_crossfit_folds 1
    target_preconditioner "none" | noise_target_control_variate "none"
    adjoint_weight 0.0 | adjoint_noise_weight 1.0 | observed_only False
    path_derivative_backend "autograd" | drift_adjoint_backend "autograd_replay"
    log_every 100 | checkpoints 500..3000 | dtype float32
    max_eigh_batch 32768 | control MultivariateReferenceDriftSpecificEntropyMatrixControl

The frozen package under ``methods/Deep-MKV-TS/code/reference`` is imported and
is not edited BY THIS DIRECTORY.  It is not pristine, though, and the earlier
wording ("never edited") was wrong: it carries an uncommitted matrix-valued
control, ``controls/specific_entropy_matrix.py``, plus eight re-export lines in
``controls/__init__.py`` -- both dated 2026-08-24, i.e. after the 2026-08-16
vendoring commit 0e09c637, and both shared by every Deep-MKV-TS variant.  See
README.md "Other deviations" item 3: because they are untracked, a fresh clone
cannot import any of those methods.

``matrix_control_multiasset.py`` and ``eigh_fallback.py`` in this directory are
byte-identical copies of the sibling's (md5 bd164b07dab15f0cbccb5c8157a49b7f for
the control, verified across all five variants).

Bank vs target
--------------
``--num-paths`` subsets the DISCREPANCY TARGET only.  The SBTS bank is always
the full 8192-path TRAIN split, because shrinking the bank changes the reference
model itself and would make a small-subset sweep measure the wrong thing.

Swept, not published
--------------------
``--lr``, ``--ridge-lambda``, ``--h``, ``--markov-order``, ``--npi``,
``--weight-grad-mode`` and ``--allow-drift-correction`` all default to the
values below and are exposed so ``sweep_hyperparams.py`` can re-select them on
the VALIDATION bank without forking this file.  Training only ever sees
``heston_ma_S_8192x252x8.npy``; the test split is not opened until
``metrics/compute_all_multiasset.py`` runs.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-7 \
    /home/tbasseras/gpu-venv/bin/python train_multiasset.py \
        --seed 0 --device cuda:0 --steps 3000
"""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import math
import os
import platform
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
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
import sbts_reference as sr  # noqa: E402

# Batched cuSOLVER `eigh` on BATCH_SIZE * 251 symmetric 8x8 matrices per outer
# iteration failed to converge once during the sibling's lambda sweep. The shim
# diagnoses the failure and retries the SAME decomposition on CPU LAPACK, and
# raises rather than retrying when Theta is non-finite. It changes no
# mathematics; see eigh_fallback.py.
eigh_fallback.install()

# ---- dataset -------------------------------------------------------------- #
DATASET = Path("/home/tbasseras/benchmark/dataset/HestonMultiAsset")
TRAIN_FILE = "heston_ma_S_8192x252x8.npy"
DT = 0.003968253968253968
STATE_DIM = 8

# ---- paper settings, byte-for-byte the sibling ---------------------------- #
ETA = 1.0
SIGMA_MIN = 1e-3
SIGMA_MAX = 0.6
LAMBDA_SCALE = 50.0
KAPPA_SCALE = 100.0
ABS_RETURN_ACF_WEIGHT = 0.25
SQUARED_RETURN_ACF_WEIGHT = 0.125
HIDDEN_DIM = 96
NUM_LAYERS = 1
BATCH_SIZE = 256
TARGET_BATCH_SIZE = 256
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 5.0
CE_RIDGE = 1e-3
LOG_EVERY = 100
CHECKPOINT_STEPS = (500, 1000, 1500, 2000, 2500, 3000)
MAX_EIGH_BATCH = 32768
DTYPE = torch.float32

# ---- swept defaults ------------------------------------------------------- #
# These are the SWEEP OUTCOME, not its starting point, and they must stay in
# lockstep with sweep/incumbent.json.  The guideline's canonical launch line is
# a bare `python code/train_multiasset.py --seed $i` with no knobs, so whatever
# is written here is what a reproduction actually runs.  The pre-sweep values
# were lr=2e-3 (80x too large), ridge_lambda=1000 and h=0.31; leaving them in
# place would have silently reproduced the configuration the sweep spent
# 9.8 GPU-hours rejecting.  See SWEEP.md for the per-knob evidence.
DEFAULT_LR = 2.5e-5
DEFAULT_RIDGE_LAMBDA = 10.0
DEFAULT_H = 0.36
DEFAULT_MARKOV_ORDER = 20
DEFAULT_NPI = 1
# "analytic" carries d b_ref / d x through the backward pass; "detached" drops
# it (sbts_reference.py:420).  The method is specified to carry it, so this is
# a correctness default, not a tuned one.
DEFAULT_WEIGHT_GRAD_MODE = "analytic"

LOSS_FIELDS = (
    "step",
    "phase",
    "loss_total",
    "objective",
    "grad_norm",
    "control_rms",
    "elapsed_sec",
)


def load_prices(*, num_paths: int | None = None) -> np.ndarray:
    """Train-split RAW prices, shape ``(N, T + 1, d)``, float64.

    Raw prices, not log prices: ``sbts_reference`` needs the prices to rebuild
    the SBTS bank (raw log-returns, their per-asset std, and the scaled
    returns) exactly as ``sbts_generate_multiasset.py`` does.  ``mmap_mode``
    keeps the 132 MB bank off the heap until the slice is taken.
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
    return array


def build_model(
    *,
    grid: DiscreteTimeGrid,
    device: torch.device,
    seed: int,
    bank_prices: np.ndarray,
    h: float,
    markov_order: int,
    npi: int,
    weight_grad_mode: str,
    jacobian_lags: int,
    allow_drift_correction: bool,
) -> tuple[DiscreteMPModel, sr.SBTSReferenceKernel]:
    """Assemble Algorithm 1 with the SBTS reference kernel."""

    kernel = sr.build_sbts_reference_kernel(
        train_prices=torch.from_numpy(bank_prices),
        grid=grid,
        h=float(h),
        markov_order=int(markov_order),
        npi=int(npi),
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
        weight_grad_mode=str(weight_grad_mode),
        jacobian_lags=int(jacobian_lags),
        device=device,
        dtype=DTYPE,
    )
    control = mc.MultivariateReferenceDriftSpecificEntropyMatrixControl(
        dt=grid.dt,
        eta=ETA,
        reference_kernel=kernel,
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
        allow_drift_correction=bool(allow_drift_correction),
        max_eigh_batch=MAX_EIGH_BATCH,
    )
    discrepancy = build_heston_discrepancy(
        num_steps=int(grid.num_steps),
        include_acf=True,
        preset="old_fullv_w0p25",
        lambda_scale=LAMBDA_SCALE,
        kappa_scale=KAPPA_SCALE,
        include_conditional_volatility=False,
        include_conditional_volatility_correlation=False,
        abs_return_acf_weight=ABS_RETURN_ACF_WEIGHT,
        squared_return_acf_weight=SQUARED_RETURN_ACF_WEIGHT,
    )
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
    return model, kernel


def build_training(
    *,
    seed: int,
    steps: int,
    lr: float,
    ridge_lambda: float,
) -> DiscreteMPTrainingConfig:
    """The sibling's ``build_training(...)``, with lr exposed for the sweep."""

    return DiscreteMPTrainingConfig(
        batch_size=BATCH_SIZE,
        target_batch_size=TARGET_BATCH_SIZE,
        num_steps=int(steps),
        lr=float(lr),
        weight_decay=WEIGHT_DECAY,
        lambda_scale=LAMBDA_SCALE,
        adjoint_weight=0.0,
        adjoint_noise_weight=1.0,
        ce_target_mode="ridge",
        ridge_lambda=float(ridge_lambda),
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


def _write_losses(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LOSS_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def _progress(
    *,
    model: DiscreteMPModel,
    checkpoint_dir: Path,
    checkpoint_steps: tuple[int, ...],
    started: float,
    rows: list[dict[str, float]],
    losses_path: Path | None = None,
    step_offset: int = 0,
):
    """Print one line per logged step and dump the requested checkpoints.

    ``losses_path`` is rewritten on **every** logged step and ``latest.pt`` is a
    rolling checkpoint, both for crash survival on a multi-hour run.
    ``CHECKPOINT_STEPS`` is the SELECTION grid and is deliberately untouched;
    the rolling file has a different name and is never a selection candidate.

    ``step_offset`` is the step already completed by an earlier run under
    ``--resume``. The trainer always counts from 1, so without the offset a
    resumed run would write a ``step_0500.pt`` holding global step 2000 and
    silently corrupt the selection grid.
    """

    wanted = {int(step) for step in checkpoint_steps}

    def _save_atomic(payload, path: Path) -> None:
        # A crash midway through torch.save leaves a truncated file that loads as
        # garbage. Write beside the target, then rename -- rename is atomic on the
        # same filesystem, so the visible file is always a complete checkpoint.
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def report(_model: DiscreteMPModel, metrics: dict[str, float]) -> None:
        step = int(metrics.get("step", 0)) + step_offset
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
        # Non-finite loss means the control has diverged. Report the STEP INDEX --
        # a wall-clock crash time does not tell you where it died, and the
        # exception raised further downstream blames the eigensolver.
        if loss != loss or objective != objective:
            print(
                f"[diverged] step={step} produced a non-finite value "
                f"(loss={loss}, objective={objective}, grad={grad}). The control has "
                "blown up; the eigensolver will fail next with a misleading "
                "'ill-conditioned' message.",
                file=sys.stderr,
                flush=True,
            )

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if step in wanted:
            _save_atomic(model.checkpoint_state(), checkpoint_dir / f"step_{step:04d}.pt")
        _save_atomic(
            {"step": step, "state": model.checkpoint_state()},
            checkpoint_dir / "latest.pt",
        )
        if losses_path is not None:
            _write_losses(losses_path, rows)

    return report


def _resume(
    *,
    model: DiscreteMPModel,
    checkpoint_dir: Path,
    history_path: Path,
    device: torch.device,
    target_steps: int,
) -> tuple[int, list[dict[str, float]]]:
    """Load ``latest.pt`` into ``model``; return (steps already done, prior loss rows).

    Only the network weights survive: ``model.fit`` builds its own Adam inside the
    frozen package, so the optimiser moments and the RNG stream restart. A resumed
    curve therefore has a visible transient at the join -- that is honest, and far
    cheaper than re-running the hours already spent.

    Prior loss rows are read back because ``_progress`` rewrites the whole CSV on
    every logged step; starting from an empty list would truncate the history.
    """

    latest = checkpoint_dir / "latest.pt"
    if not latest.is_file():
        raise SystemExit(f"ABORT: --resume needs {latest}, which does not exist")

    payload = torch.load(latest, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or "step" not in payload or "state" not in payload:
        raise SystemExit(
            f"ABORT: {latest} is not a rolling checkpoint "
            "({'step': int, 'state': checkpoint_state()})"
        )
    start_step = int(payload["step"])
    if start_step >= target_steps:
        raise SystemExit(
            f"ABORT: {latest} is already at step {start_step} >= --steps {target_steps}; "
            "nothing left to run"
        )
    model.load_checkpoint_state(payload["state"])

    rows: list[dict[str, float]] = []
    if history_path.is_file():
        with history_path.open(newline="", encoding="utf-8") as handle:
            rows = [r for r in csv.DictReader(handle) if int(float(r["step"])) <= start_step]

    print(
        f"[resume] {latest} at step {start_step}; running {target_steps - start_step} "
        f"more to reach {target_steps} | {len(rows)} prior loss rows kept",
        flush=True,
    )
    return start_step, rows


def _acquire_seed_lock(run_dir: Path, seed: int) -> Path:
    """Refuse to start if a LIVE process already owns this seed's output directory.

    Two trainers on one seed interleave writes into the same ``latest.pt`` and the
    same loss CSV.  That does not produce an obvious failure -- it produces a
    loadable checkpoint and a plausible loss curve that are both silently wrong.
    Liveness is tested with ``os.kill(pid, 0)`` so a lock left by a crashed run is
    reclaimed instead of becoming a landmine.
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
                f"[lock] REFUSING TO START: seed {seed} is already being trained by "
                f"PID {owner} (lock file {lock}). Two trainers on one seed corrupt "
                "both outputs without failing loudly. If that PID is demonstrably "
                "not a trainer, delete the lock file and retry."
            )
        print(f"[lock] reclaiming stale lock left by dead PID {owner}", flush=True)

    lock.write_text(f"{os.getpid()}\n", encoding="utf-8")
    atexit.register(lambda: lock.unlink(missing_ok=True))
    print(f"[lock] seed {seed} owned by PID {os.getpid()}", flush=True)
    return lock


def _write_config(
    path: Path,
    *,
    args: argparse.Namespace,
    kernel: sr.SBTSReferenceKernel,
    seed: int,
    steps: int,
    num_paths: int,
    num_steps: int,
    bank_paths: int,
    parameters: int,
    elapsed: float,
    device: torch.device,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sigma_ref = (kernel.sigma_assets / math.sqrt(DT)).double().tolist()
    payload = {
        "method": "Deep-MKV-TS-SBTSref",
        "dataset": "HestonMultiAsset",
        "seed": int(seed),
        "d": STATE_DIM,
        "joint_or_per_asset": "joint",
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
        "lr": float(args.lr),
        "weight_decay": WEIGHT_DECAY,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "eta": ETA,
        "sigma_min": SIGMA_MIN,
        "sigma_max": SIGMA_MAX,
        "lambda_scale": LAMBDA_SCALE,
        "kappa_scale": KAPPA_SCALE,
        "discrepancy_preset": "old_fullv_w0p25",
        "abs_return_acf_weight": ABS_RETURN_ACF_WEIGHT,
        "squared_return_acf_weight": SQUARED_RETURN_ACF_WEIGHT,
        "adjoint_weight": 0.0,
        "adjoint_noise_weight": 1.0,
        "ce_target_mode": "ridge",
        "ridge_lambda": float(args.ridge_lambda),
        "ce_ridge": CE_RIDGE,
        "path_derivative_backend": "autograd",
        "drift_adjoint_backend": "autograd_replay",
        "max_eigh_batch": MAX_EIGH_BATCH,
        "control": "MultivariateReferenceDriftSpecificEntropyMatrixControl",
        "allow_drift_correction": bool(args.allow_drift_correction),
        # ---- the one thing that differs from the sibling ------------------- #
        "reference_kernel": "code/sbts_reference.py",
        "reference_family": "SBTS-Markovian",
        "reference_h": float(args.h),
        "reference_markov_order": int(args.markov_order),
        "reference_npi": int(args.npi),
        "reference_weight_grad_mode": str(args.weight_grad_mode),
        "reference_jacobian_lags": int(args.jacobian_lags),
        "reference_bank_paths": int(bank_paths),
        "reference_bank_split": "train",
        "reference_sigma_constant": sigma_ref,
        # How many times batched cuSOLVER `eigh` failed on this seed's Theta and
        # the decomposition had to be redone on CPU LAPACK. Zero is expected;
        # non-zero does not invalidate the run (same decomposition, other
        # backend) but it belongs in the audit record.
        "eigh_fallback": eigh_fallback.counters(),
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
    parser.add_argument(
        "--num-paths",
        type=int,
        default=None,
        help="subset the DISCREPANCY TARGET only; the SBTS bank stays full",
    )
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--ridge-lambda", type=float, default=DEFAULT_RIDGE_LAMBDA)
    parser.add_argument("--h", type=float, default=DEFAULT_H)
    parser.add_argument("--markov-order", type=int, default=DEFAULT_MARKOV_ORDER)
    parser.add_argument(
        "--jacobian-lags",
        type=int,
        default=-1,
        help="how many trailing lags of d b^ref/dx enter the adjoint; -1 (default) "
        "= all markov_order of them, the exact gradient. 1 is the truncated "
        "ablation and is WRONG by 100%% relative error at K=20 -- kept only so the "
        "pre-fix behaviour stays reproducible.",
    )
    parser.add_argument("--npi", type=int, default=DEFAULT_NPI)
    parser.add_argument(
        "--weight-grad-mode",
        choices=("analytic", "detached"),
        default=DEFAULT_WEIGHT_GRAD_MODE,
        help="analytic sends the exact db^ref/dx backward; detached sends zero",
    )
    parser.add_argument("--allow-drift-correction", action="store_true")
    parser.add_argument(
        "--checkpoint-steps", type=int, nargs="*", default=list(CHECKPOINT_STEPS)
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from runs/seed_N/training_checkpoints/latest.pt; --steps stays "
             "the GLOBAL target, so only the remaining steps are run",
    )
    parser.add_argument("--run-root", type=Path, default=HERE / "runs")
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="run-directory suffix, so sweep arms do not collide on seed_N",
    )
    parser.add_argument(
        "--no-artefacts",
        action="store_true",
        help="skip weights/ and losses/ writes; sweep arms are not publications",
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

    # The bank is ALWAYS the full train split: shrinking it changes the
    # reference model itself. --num-paths subsets the discrepancy target only.
    bank_prices = load_prices()
    target_prices = bank_prices if args.num_paths is None else bank_prices[: int(args.num_paths)]
    target = torch.log(torch.from_numpy(np.ascontiguousarray(target_prices))).to(
        device=device, dtype=DTYPE
    )
    if args.num_steps is not None:
        target = target[:, : int(args.num_steps) + 1, :].contiguous()
        bank_prices = bank_prices[:, : int(args.num_steps) + 1, :]
    num_paths = int(target.shape[0])
    num_steps = int(target.shape[1]) - 1
    grid = DiscreteTimeGrid(T=num_steps * DT, num_steps=num_steps)
    print(
        f"[data] train log prices {tuple(target.shape)} dt={DT} device={device}",
        flush=True,
    )

    model, kernel = build_model(
        grid=grid,
        device=device,
        seed=int(args.seed),
        bank_prices=bank_prices,
        h=float(args.h),
        markov_order=int(args.markov_order),
        npi=int(args.npi),
        weight_grad_mode=str(args.weight_grad_mode),
        jacobian_lags=int(args.jacobian_lags),
        allow_drift_correction=bool(args.allow_drift_correction),
    )
    parameters = sum(p.numel() for p in model.network.parameters())
    print(f"[model] parameters={parameters}", flush=True)
    spectrum = (kernel.sigma_assets / math.sqrt(DT)).double()
    print(
        f"[reference] SBTS-Markovian h={args.h} K={args.markov_order} "
        f"npi={args.npi} grad={args.weight_grad_mode} "
        f"jac_lags={args.jacobian_lags} "
        f"bank={int(bank_prices.shape[0])} paths | "
        f"sigma^ref in [{float(spectrum.min()):.4f}, {float(spectrum.max()):.4f}]",
        flush=True,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if int(args.time_only) > 0:
        training = build_training(
            seed=int(args.seed),
            steps=int(args.time_only),
            lr=float(args.lr),
            ridge_lambda=float(args.ridge_lambda),
        )
        started = time.perf_counter()
        model.fit(target, training=training)
        elapsed = time.perf_counter() - started
        per_step = elapsed / float(args.time_only)
        print(
            f"[time] {args.time_only} steps in {elapsed:.1f}s = {per_step:.3f} s/step "
            f"-> {per_step * 3000.0 / 60.0:.1f} min for 3000 steps",
            flush=True,
        )
        if device.type == "cuda":
            print(
                "[time] peak GPU memory "
                f"{torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB",
                flush=True,
            )
        return

    name = f"seed_{int(args.seed)}" + (f"_{args.tag}" if args.tag else "")
    run_dir = Path(args.run_root) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    _acquire_seed_lock(run_dir, int(args.seed))
    checkpoint_dir = run_dir / "training_checkpoints"
    rows: list[dict[str, float]] = []

    losses_path = (
        None
        if args.no_artefacts
        else METHOD_ROOT / "losses" / f"seed_{int(args.seed)}_losses.csv"
    )
    history_path = losses_path if losses_path is not None else run_dir / "losses.csv"

    start_step = 0
    if args.resume:
        start_step, rows = _resume(
            model=model,
            checkpoint_dir=checkpoint_dir,
            history_path=history_path,
            device=device,
            target_steps=int(args.steps),
        )

    training = build_training(
        seed=int(args.seed),
        steps=int(args.steps) - start_step,
        lr=float(args.lr),
        ridge_lambda=float(args.ridge_lambda),
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
            losses_path=history_path,
            step_offset=start_step,
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

    if not args.no_artefacts:
        torch.save(
            model.network.state_dict(),
            METHOD_ROOT / "weights" / f"seed_{int(args.seed)}_model.pt",
        )
        _write_losses(
            METHOD_ROOT / "losses" / f"seed_{int(args.seed)}_losses.csv", rows
        )
        _write_config(
            METHOD_ROOT / "weights" / f"seed_{int(args.seed)}_config.json",
            args=args,
            kernel=kernel,
            seed=int(args.seed),
            steps=int(args.steps),
            num_paths=num_paths,
            num_steps=num_steps,
            bank_paths=int(bank_prices.shape[0]),
            parameters=parameters,
            elapsed=elapsed,
            device=device,
        )
    # Always written, sweep arms included: the run directory must be
    # self-describing even when it produces no publication artefact.
    _write_config(
        run_dir / "config.json",
        args=args,
        kernel=kernel,
        seed=int(args.seed),
        steps=int(args.steps),
        num_paths=num_paths,
        num_steps=num_steps,
        bank_paths=int(bank_prices.shape[0]),
        parameters=parameters,
        elapsed=elapsed,
        device=device,
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
