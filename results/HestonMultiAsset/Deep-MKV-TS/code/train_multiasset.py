#!/usr/bin/env python3
"""Deep-MKV-TS Algorithm 1 on the d = 8 HestonMultiAsset dataset.

This is the d = 1 Heston reproduction, lifted to d = 8 with the smallest set of
changes the paper's own formulation forces.  Everything that can stay identical
stays identical, so any difference in the benchmark tables is attributable to
the dimension, not to a re-tuned pipeline.

What is IDENTICAL to the committed d = 1 run
--------------------------------------------
Read off ``methods/Deep-MKV-TS/paper_reimplementation/metric/run_reproduction.sh``
and ``runs/seed_0/volatility_only_online_mp/run_manifest.json``:

    eta                       1.0
    sigma_min / sigma_max     1e-3 / 0.6
    lambda_scale              50
    kappa_scale               100
    discrepancy               build_heston_discrepancy(preset="old_fullv_w0p25",
                                include_acf=True, abs_return_acf_weight=0.25,
                                squared_return_acf_weight=0.125,
                                include_conditional_volatility=False,
                                include_conditional_volatility_correlation=False)
    joint volatility weight   0   (block never appended, as under --source-only)
    ce estimator              RidgeConditionalExpectation(ridge=1e-3)
    architecture              GRU, hidden 96, 1 layer
    batch_size                256
    target_batch_size         256
    num_steps                 3000
    lr                        2e-3
    weight_decay              1e-5
    grad_clip_norm            5.0
    ce_target_mode            "ridge",   ridge_lambda 1e-3, crossfit folds 1
    target_preconditioner     "none",    preconditioner_batches 8
    noise_target_control_var  "none",    estimator "score", stein probes 8
    adjoint_weight            0.0
    adjoint_noise_weight      1.0
    observed_only             False
    path_derivative_backend   "autograd"
    log_every                 100
    checkpoint steps          500 1000 1500 2000 2500 3000
    dtype                     float32

The discrepancy stack is dimension-agnostic: every feature map flattens
``(B, T, d)`` before the MMD kernel, so the exact same 13 blocks are used at
d = 8 (verified: finite value and finite path gradient at d = 8).

What HAD to change, and why
---------------------------
1.  **Matrix control instead of diagonal control.**  ``Theta`` is a symmetric
    ``d x d`` matrix and the paper clips its EIGENVALUES.  The shipped
    ``ReferenceDriftSpecificEntropyDiagonalControl`` clips diagonal ENTRIES,
    which coincides with the paper only at d = 1.  We use
    ``MultivariateReferenceDriftSpecificEntropyMatrixControl``
    (``matrix_control_multiasset.py``), which is the shipped
    ``SpecificEntropyMatrixControl`` with its d > 1 plumbing repaired.

2.  **Matrix Euler step.**  ``x + b dt + sqrt(dt) * sigma @ eps`` rather than the
    elementwise product ``DriftVolatilityEulerStep`` performs.

3.  **``noise_adjoint_dim = d * d = 64``.**  The GRU must emit a full symmetric
    ``Z``, not a ``d``-vector.

4.  **Multivariate reference kernel.**  No multivariate Guyon-Lekeufack kernel
    exists in the frozen package.  ``multivariate_reference.py`` supplies a
    GL-style cross-asset kernel on exactly the 7 covariance features the
    package's own d > 1 smoke test uses, fitted by maximum likelihood.  The
    frozen artefact is ``reference/reference_kernel.json``
    (calibration NLL -10.3362, validation NLL -10.3847).

5.  **``drift_adjoint_backend = "autograd_replay"`` instead of
    ``"analytical_reference"``.**  Not a preference: ``analytical_reference``
    requires the reference kernel to expose ``drift_step_path_vjp``, which the
    multivariate kernel does not have, AND the matrix control deliberately
    returns ``None`` from ``running_cost_value_and_path_gradient`` (its fast VJP
    assumes a diagonal sigma).  ``autograd_replay`` computes the SAME
    ``db^ref/dx`` term by differentiating the replayed drift; it is slower, not
    weaker.

6.  **``differentiable_sigma = True`` on the reference kernel.**  At d = 1 the
    running-cost path source was identically zero because ``sigma_ref`` was
    evaluated under ``no_grad``.  Measured here, the ``d(sigma_ref)/dx`` term is
    ~7% of the discrepancy source at the realistic ``Z`` scale -- the same order
    as ``db^ref/dx`` (~8.6%) -- so dropping it is not free.  Turned on by
    explicit instruction.

7.  **``max_eigh_batch = 32768``.**  cuSOLVER's batched ``syev`` fails above
    ~64k matrices, and the running cost at ``batch_size=256, num_steps=251``
    asks for 64,256 of them -- inside the failure band.  The control chunks the
    running cost over the batch; every chunk size tested reproduces the
    unchunked value, gradient and metrics BIT-IDENTICALLY.

Model selection
---------------
Checkpoints are written at steps 500...3000.  Which one is reported is decided
afterwards on the VALIDATION split ``heston_ma_S_val_8192x252x8.npy``, which
``MULTIASSET_GUIDELINE.md`` section 0.2 names as the selection split.  Training
itself only ever sees ``heston_ma_S_8192x252x8.npy``, and the test split is not
opened until ``metrics/compute_all_multiasset.py`` runs.

Usage
-----
    R=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 \
    PYTHONPATH="$R/src:$R/experiments" taskset -c 0-7 \
    /home/tbasseras/gpu-venv/bin/python train_multiasset.py \
        --seed 0 --device cuda:0 --steps 3000
"""

from __future__ import annotations

import argparse
import atexit
import csv
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
from fit_reference_multiasset import (  # noqa: E402
    DT,
    STATE_DIM,
    load_kernel,
    load_log_prices,
)

# The paper's spectral clip calls batched cuSOLVER `eigh` on BATCH_SIZE * 251
# symmetric 8x8 matrices per outer iteration. On 2026-08-26 it failed to converge
# 31 minutes into the lambda = 1 sweep run -- a failure mode that cannot occur at
# d = 1, where Theta is 1x1. Over a 3000-step run there are ~60x more
# decompositions than in that sweep, so this is a campaign-level risk, not a
# curiosity. The shim diagnoses the failure and retries the SAME decomposition on
# CPU LAPACK, and raises rather than retrying when Theta is non-finite. It changes
# no mathematics; see eigh_fallback.py.
eigh_fallback.install()

# ---- paper settings, byte-for-byte the committed d = 1 run ---------------- #
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
LR = 2e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 5.0
# The committed d = 1 value.  Kept as a named constant so `retuned_for_d8` in
# the config can be derived, never hand-asserted.
D1_RIDGE_LAMBDA = 1e-3
# Re-selected for d = 8 on `heston_ma_S_val_8192x252x8.npy` by
# sweep_ridge_lambda.py over {1e-3, 1, 10, 100, 1000}; see sweep/winner.json.
# The paper's 1e-3 leaves the p = 2017 column basis interpolating at B = 256 rows
# (ce_projection_r2 = 0.928, val_discrepancy = 0.0364); 1000 gives 0.0238. It is
# the ONE hyperparameter changed from the paper's value, which is why
# D1_RIDGE_LAMBDA above is kept: `retuned_for_d8` is derived from the comparison,
# never hand-asserted.
RIDGE_LAMBDA = 1000.0
CE_RIDGE = 1e-3
LOG_EVERY = 100
CHECKPOINT_STEPS = (500, 1000, 1500, 2000, 2500, 3000)
MAX_EIGH_BATCH = 32768
DTYPE = torch.float32

LOSS_FIELDS = (
    "step",
    "phase",
    "loss_total",
    "objective",
    "grad_norm",
    "control_rms",
    "elapsed_sec",
)


def build_model(
    *,
    grid: DiscreteTimeGrid,
    device: torch.device,
    seed: int,
    kernel_steps: int,
) -> DiscreteMPModel:
    """Assemble Algorithm 1's generator, control and objective at d = 8."""

    kernel = load_kernel(
        device=device,
        dtype=DTYPE,
        num_steps=kernel_steps,
        differentiable_sigma=True,
    )
    control = mc.MultivariateReferenceDriftSpecificEntropyMatrixControl(
        dt=grid.dt,
        eta=ETA,
        reference_kernel=kernel,
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
        allow_drift_correction=False,
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
    return model


def build_training(
    *, seed: int, steps: int, ridge_lambda: float | None = None
) -> DiscreteMPTrainingConfig:
    """The d = 1 ``_training(...)`` block, with only the drift-backend swap.

    ``ridge_lambda`` defaults to the module constant ``RIDGE_LAMBDA``.  It is
    exposed as a keyword only so ``sweep_ridge_lambda.py`` can re-select it on
    the VALIDATION bank without forking this file.  The 5-seed campaign never
    passes it, so it always uses the constant.
    """

    ridge = RIDGE_LAMBDA if ridge_lambda is None else float(ridge_lambda)
    return DiscreteMPTrainingConfig(
        batch_size=BATCH_SIZE,
        target_batch_size=TARGET_BATCH_SIZE,
        num_steps=int(steps),
        lr=LR,
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

    Two things happen here beyond the selection checkpoints, both for crash
    survival on a ~10 h/seed run (see MULTIASSET_GUIDELINE.md section 12.8, and
    the lambda = 1 divergence recorded in sweep/lambda_1e0.json):

    * ``losses_path`` is rewritten on **every** logged step, not once at the end.
      Held only in memory, the whole loss history dies with the process, and the
      run leaves nothing at all behind.
    * ``latest.pt`` is a rolling checkpoint written on every logged step, so at
      most ``LOG_EVERY`` steps of work is ever at risk instead of the 500 between
      selection checkpoints.

    ``CHECKPOINT_STEPS`` is the **selection grid** and is deliberately untouched:
    ``select_checkpoint_multiasset.py`` chooses among those steps, so adding to it
    would change which checkpoints compete and therefore change the method. The
    rolling file is written to a different name and is never a selection candidate.
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
        # Non-finite loss means the control has diverged, as lambda = 1 did. Report
        # the STEP INDEX -- a wall-clock crash time does not tell you where it died,
        # and the exception raised further downstream blames the eigensolver.
        if loss != loss or objective != objective:
            print(
                f"[diverged] step={step} produced a non-finite value "
                f"(loss={loss}, objective={objective}, grad={grad}). The control has "
                "blown up; the eigensolver will fail next with a misleading "
                "'ill-conditioned' message. See MULTIASSET_GUIDELINE.md section 12.8.",
                file=sys.stderr,
                flush=True,
            )

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if step in wanted:
            _save_atomic(model.checkpoint_state(), checkpoint_dir / f"step_{step:04d}.pt")
        # Rolling crash-recovery save, overwritten each time: bounded disk, and at
        # most LOG_EVERY steps of progress is ever unrecoverable.
        _save_atomic(
            {"step": step, "state": model.checkpoint_state()},
            checkpoint_dir / "latest.pt",
        )
        if losses_path is not None:
            _write_losses(losses_path, rows)

    return report


def _acquire_seed_lock(run_dir: Path, seed: int) -> Path:
    """Refuse to start if a LIVE process already owns this seed's output directory.

    Two trainers on one seed interleave writes into the same ``latest.pt`` and the
    same ``losses/seed_N_losses.csv``.  That does not produce an obvious failure --
    it produces a loadable checkpoint and a plausible loss curve that are both
    silently wrong, and nothing downstream can tell.  The collision is not
    hypothetical: ``run_campaign.sh`` schedules its waves in advance, so a seed
    started by hand can be started a second time by the driver hours later.

    The lock records a PID, and liveness is tested with ``os.kill(pid, 0)`` so a
    lock left behind by a crashed run is reclaimed instead of becoming a landmine
    that blocks every future attempt.  This runs once at startup, writes no tensor
    and touches no hyperparameter -- it cannot change a single number in the run.
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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Only ridge_lambda was re-selected, and only because the frozen package's
    # Phi^ref basis (flattened path prefix, p = 1 + (k+1)*d) goes from
    # over-determined at d = 1 to under-determined at d = 8.  See
    # sweep_ridge_lambda.py.  Every other value is the committed d = 1 value.
    retuned = ["ridge_lambda"] if RIDGE_LAMBDA != D1_RIDGE_LAMBDA else []
    payload = {
        "method": "Deep-MKV-TS",
        "dataset": "HestonMultiAsset",
        "seed": int(seed),
        "d": STATE_DIM,
        "joint_or_per_asset": "joint",
        "retuned_for_d8": retuned,
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
        "lr": LR,
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
        "ridge_lambda": RIDGE_LAMBDA,
        "ce_ridge": CE_RIDGE,
        "path_derivative_backend": "autograd",
        "drift_adjoint_backend": "autograd_replay",
        "differentiable_sigma": True,
        "max_eigh_batch": MAX_EIGH_BATCH,
        # How many times batched cuSOLVER `eigh` failed on this seed's Theta and
        # the decomposition had to be redone on CPU LAPACK. Zero is the expected
        # value; non-zero does not invalidate the run (same decomposition, other
        # backend) but it is part of the audit record and belongs in the config
        # rather than only in a log file that is not committed.
        "eigh_fallback": eigh_fallback.counters(),
        "control": "MultivariateReferenceDriftSpecificEntropyMatrixControl",
        "reference_kernel": "code/reference/reference_kernel.json",
        "paper_hyperparams": not retuned,
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
        "--checkpoint-steps", type=int, nargs="*", default=list(CHECKPOINT_STEPS)
    )
    parser.add_argument("--run-root", type=Path, default=HERE / "runs")
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

    model = build_model(
        grid=grid, device=device, seed=int(args.seed), kernel_steps=num_steps
    )
    parameters = sum(p.numel() for p in model.network.parameters())
    print(f"[model] parameters={parameters}", flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if int(args.time_only) > 0:
        training = build_training(seed=int(args.seed), steps=int(args.time_only))
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

    run_dir = Path(args.run_root) / f"seed_{int(args.seed)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _acquire_seed_lock(run_dir, int(args.seed))
    checkpoint_dir = run_dir / "training_checkpoints"
    rows: list[dict[str, float]] = []
    training = build_training(seed=int(args.seed), steps=int(args.steps))

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
            losses_path=METHOD_ROOT / "losses" / f"seed_{int(args.seed)}_losses.csv",
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
        METHOD_ROOT / "weights" / f"seed_{int(args.seed)}_model.pt",
    )
    _write_losses(METHOD_ROOT / "losses" / f"seed_{int(args.seed)}_losses.csv", rows)
    _write_config(
        METHOD_ROOT / "weights" / f"seed_{int(args.seed)}_config.json",
        seed=int(args.seed),
        steps=int(args.steps),
        num_paths=num_paths,
        num_steps=num_steps,
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
