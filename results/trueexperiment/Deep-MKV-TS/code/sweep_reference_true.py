"""Phase 1: select the frozen reference kernel's hyperparameters on TrueDataset.

This is the "do the ridge split for the reference model FIRST" step.  It sweeps
``(sigma_max, ridge_covariance, ridge_drift)`` and picks one point.  Only that
point is then frozen by ``fit_reference_true.py`` without ``--report-only``.

WHY THE SELECTION RULE IS NOT "LOWEST VALIDATION NLL"
-----------------------------------------------------
Measured on this dataset (1024 paths, 300 steps, lr 1e-2, default ridges):

    sigma_max   val NLL    on-ceiling   spectrum p99
       1.50     -0.3261      12.589%      1.500   <- pinned
       2.00     -0.6641       4.589%      2.000   <- pinned
       2.50     -0.7131       2.397%      2.500   <- pinned, and NLL-optimal
       3.50     -0.6908       0.691%      3.181
       5.00     -0.6753       0.170%      3.228
       8.00     -0.6469       0.015%      3.127

The likelihood is minimised at ``sigma_max = 2.5`` -- a ceiling that censors the
top 2.4% of the conditional volatility spectrum, with ``p99`` sitting EXACTLY on
the bound.  This is not a coincidence and not noise.  ``spectral_sigma`` clamps
the spectrum into ``[sigma_min, sigma_max]``; clamping removes variance; and a
Gaussian log-likelihood is happy to be handed less variance whenever the model
would otherwise over-predict.  So NLL actively PREFERS a censoring ceiling.
Selecting on NLL alone would freeze a reference that structurally cannot emit
the market's volatility, and every downstream vol row would pin at the clip
while the artefact reported a perfectly respectable likelihood.

The tell is that once the ceiling stops binding, ``p99`` stabilises at ~3.15-3.23
regardless of where the ceiling is put (3.181 at 3.5, 3.228 at 5.0, 3.127 at
8.0).  That plateau is the panel's actual conditional volatility reach.  Any
ceiling below it is censoring, whatever the NLL says.

Hence the two-part rule:

    ADMISSIBILITY (hard gate, checked first):
        spectrum_p99 < sigma_max * (1 - 1e-6)      ceiling not binding at p99
        clip_saturation_high <= MAX_CEILING_MASS   ceiling barely touched
        clip_saturation_low  <= MAX_FLOOR_MASS     floor barely touched
        validation_nll finite AND < baseline_nll   beats a constant covariance

    OBJECTIVE (among admissible points only):
        minimise validation_nll

This mirrors the guideline's own protocol shape -- an admissibility envelope
first, an objective second -- and it mirrors the reason: an unconstrained
goodness-of-fit objective selects a degenerate model.

``sigma_max = 2.5`` is deliberately LEFT IN the grid even though it is expected
to be rejected.  A gate that never fires is a gate nobody can audit; the
rejected row and its ``reject_reason`` are written to the CSV.

SCREEN COARSE, CONFIRM AT SCALE
-------------------------------
Stage 1 screens the full grid at ``--screen-paths`` (default 2048).  Stage 2
re-runs the top ``--confirm-top`` points at the full 6144 and re-applies the
same gate, because a saturation fraction measured on a subsample is an estimate.
A point that only wins at screen scale has not won.

PARALLELISM
-----------
Workers are heterogeneous by design: the fit is a batched ``eigh``, measured at
73 s on 8 CPU cores versus 33 s on one A100 for the same job.  GPU is 2.2x
faster per job but there are only 3 usable GPUs, whereas 64 cores buy 8 CPU
workers -- comparable throughput.  Running both together beats either alone.
Default layout is 3 GPU workers (8 cores each) + 5 CPU workers (8 cores each)
= 64 cores, which is exactly the sanctioned budget for this experiment.

GPU 0 is excluded by default: it is not ours.  ``--gpus`` overrides.

Each grid point runs as a SUBPROCESS, not an import.  A divergent point that
raises or produces a non-finite NLL then costs one row, not the sweep.

Usage
-----
    cd /home/tbasseras/benchmark && /home/tbasseras/gpu-venv/bin/python \
        results/trueexperiment/Deep-MKV-TS/code/sweep_reference_true.py

Nothing here writes ``reference/reference_kernel.json``.  Freezing is a separate,
deliberate command, printed at the end.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
METHOD_DIR = HERE.parent
LOSSES_DIR = METHOD_DIR / "losses"
FITTER = HERE / "fit_reference_true.py"
PYTHON = "/home/tbasseras/gpu-venv/bin/python"

# Starting point is the published Heston config; the grid brackets it rather
# than replacing it.  Heston froze sigma_max = 0.6, ridge_covariance = 1e-4,
# ridge_drift = 1e-3.  0.6 is not in this grid because the panel's UNCONDITIONAL
# top eigenvalue is already 1.802 -- three times that ceiling -- so 0.6 is not a
# candidate, it is a known-dead point.  The ridge defaults ARE in the grid, at
# its centre.
# 2026-08-27, second pass. The first pass ran a 27-point screen at 2048 train
# paths / 256 saturation paths and shortlisted three sigma_max = 3.5 points with
# spectrum p99 ~ 3.11, comfortably clear of the ceiling. Re-run at full scale
# (6144 / 1024) every one of them reported p99 = 3.5000 EXACTLY: the ceiling
# binds and the top percentile is censored. The screen had not sampled the tail.
# That is the guideline's section 7.2 point made concrete -- a screen result
# sitting near a bound is a statement about the screen, not about the optimum.
#
# So 2.5 and 3.5 are dropped because they were MEASURED to censor at full scale,
# not because the grid was trimmed for speed, and the remaining grid is run at
# full scale in a single stage with no screen/confirm resolution gap.
SIGMA_MAX_GRID = (5.0, 8.0)
RIDGE_COVARIANCE_GRID = (1e-5, 1e-3)
RIDGE_DRIFT_GRID = (1e-3, 1e-2)

# Admissibility thresholds.  These are choices, so they are stated once, here,
# with their justification, instead of being buried in a comparison.
#
# MAX_CEILING_MASS = 1%: the p99 test already rejects any ceiling binding at the
# 99th percentile, so 1% is the consistent companion bound -- it says "the top
# percentile may touch the ceiling, nothing below it may".  At the measured
# plateau this admits 3.5 (0.69%) and 5.0 (0.17%) and rejects 2.5 (2.40%).
#
# MAX_FLOOR_MASS = 0.1%: the floor is sigma_min = 1e-3 against a smallest
# observed eigenvalue of 0.231, so the floor should be untouched.  Any mass here
# means something has gone wrong, not that the bound needs tuning.
MAX_CEILING_MASS = 0.01
MAX_FLOOR_MASS = 0.001

# lr = 1e-2 won a scan over {5e-2, 1e-2, 3e-3, 1e-3} at this dt (val NLL -0.531,
# -0.545, -0.403, +336.29 respectively).  Heston's 1e-3 does NOT transfer: it
# leaves the kernel undertrained enough that a single validation path with a
# jump drives the NLL to +336.  Re-scanned here rather than inherited.
DEFAULT_LR = 1e-2
DEFAULT_STEPS = 300


def _grid() -> list[dict[str, float]]:
    points = []
    for sigma_max, ridge_cov, ridge_drift in itertools.product(
        SIGMA_MAX_GRID, RIDGE_COVARIANCE_GRID, RIDGE_DRIFT_GRID
    ):
        points.append(
            {
                "sigma_max": float(sigma_max),
                "ridge_covariance": float(ridge_cov),
                "ridge_drift": float(ridge_drift),
            }
        )
    return points


def _tag(point: dict[str, object]) -> str:
    return (
        f"sm{float(point['sigma_max']):g}_rc{float(point['ridge_covariance']):g}"
        f"_rd{float(point['ridge_drift']):g}"
    )


def _run_point(
    point: dict[str, float],
    *,
    worker: dict[str, object],
    num_paths: int,
    steps: int,
    lr: float,
    saturation_paths: int,
    stage_dir: Path,
    reuse_existing: bool = False,
) -> dict[str, object]:
    """Run one grid point as a subprocess and return its scorecard."""

    tag = _tag(point)
    json_out = stage_dir / f"{tag}.json"
    log_out = stage_dir / f"{tag}.log"

    if reuse_existing and json_out.exists():
        # Opt-in only, and only for points that already SUCCEEDED -- a crashed
        # point leaves no json, so it is always re-run. This exists so a fixed
        # bug in a later stage does not force a 25-minute re-screen; it is never
        # the default, because silently reusing a result computed by different
        # code is how a sweep starts reporting a mixture of two configurations.
        record: dict[str, object] = dict(point)
        record.update(json.loads(json_out.read_text()))
        record["device"] = "reused"
        record["elapsed_sec"] = 0.0
        record["returncode"] = 0
        record["failed"] = False
        record["reused_from"] = str(json_out)
        return record

    environment = dict(os.environ)
    cores = str(worker["cores"])
    environment["OMP_NUM_THREADS"] = str(worker["threads"])
    environment["MKL_NUM_THREADS"] = str(worker["threads"])
    environment["OPENBLAS_NUM_THREADS"] = str(worker["threads"])
    gpu = worker["gpu"]
    if gpu is None:
        device = "cpu"
        environment["CUDA_VISIBLE_DEVICES"] = ""
    else:
        device = "cuda:0"
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    command = [
        "taskset", "-c", cores,
        PYTHON, str(FITTER),
        "--num-paths", str(int(num_paths)),
        "--steps", str(int(steps)),
        "--log-every", str(int(steps)),
        "--lr", repr(float(lr)),
        "--sigma-max", repr(float(point["sigma_max"])),
        "--ridge-covariance", repr(float(point["ridge_covariance"])),
        "--ridge-drift", repr(float(point["ridge_drift"])),
        "--saturation-paths", str(int(saturation_paths)),
        "--device", device,
        "--report-only",
        "--json-out", str(json_out),
    ]

    started = time.time()
    with log_out.open("w") as handle:
        completed = subprocess.run(
            command, env=environment, stdout=handle, stderr=subprocess.STDOUT
        )
    elapsed = time.time() - started

    record: dict[str, object] = dict(point)
    record["device"] = device if gpu is None else f"cuda{gpu}"
    record["elapsed_sec"] = round(elapsed, 1)
    record["returncode"] = int(completed.returncode)

    if completed.returncode != 0 or not json_out.exists():
        # A point that crashed is a result: it says this corner of the grid is
        # not usable.  It is recorded, not retried and not silently dropped.
        record["failed"] = True
        record["reject_reason"] = (
            f"subprocess rc={completed.returncode}; see {log_out.name}"
        )
        return record

    record.update(json.loads(json_out.read_text()))
    record["failed"] = False
    return record


def _admissible(record: dict[str, object]) -> tuple[bool, str]:
    """Apply the hard gate.  Returns ``(ok, reason_if_not)``."""

    if record.get("failed"):
        return False, str(record.get("reject_reason", "subprocess failed"))

    sigma_max = float(record["sigma_max"])
    p99 = float(record["spectrum_p99"])
    high = float(record["clip_saturation_high"])
    low = float(record["clip_saturation_low"])
    validation_nll = float(record["validation_nll"])
    baseline_nll = float(record["baseline_nll"])

    if not (validation_nll == validation_nll and abs(validation_nll) < 1e30):
        return False, f"validation_nll not finite ({validation_nll})"
    if validation_nll >= baseline_nll:
        return False, (
            f"validation_nll {validation_nll:.4f} >= constant-covariance "
            f"baseline {baseline_nll:.4f}: memories not earning their keep"
        )
    if p99 >= sigma_max * (1.0 - 1e-6):
        return False, (
            f"ceiling binds at p99 (p99={p99:.4f} vs sigma_max={sigma_max:g}): "
            "top percentile of the volatility spectrum is censored"
        )
    if high > MAX_CEILING_MASS:
        return False, f"clip_saturation_high {high:.4%} > {MAX_CEILING_MASS:.4%}"
    if low > MAX_FLOOR_MASS:
        return False, f"clip_saturation_low {low:.4%} > {MAX_FLOOR_MASS:.4%}"
    return True, ""


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    columns: list[str] = []
    for record in records:
        for key in record:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _execute(
    points: list[dict[str, float]],
    *,
    workers: list[dict[str, object]],
    num_paths: int,
    steps: int,
    lr: float,
    saturation_paths: int,
    stage_dir: Path,
    label: str,
    reuse_existing: bool = False,
) -> list[dict[str, object]]:
    """Run every point across the worker pool.  One worker = one slot."""

    stage_dir.mkdir(parents=True, exist_ok=True)
    pending: queue.Queue = queue.Queue()
    for index, point in enumerate(points):
        pending.put((index, point))
    results: list[dict[str, object] | None] = [None] * len(points)
    lock = threading.Lock()
    counter = {"done": 0}

    def consume(worker: dict[str, object]) -> None:
        while True:
            try:
                index, point = pending.get_nowait()
            except queue.Empty:
                return
            record = _run_point(
                point,
                worker=worker,
                num_paths=num_paths,
                steps=steps,
                lr=lr,
                saturation_paths=saturation_paths,
                stage_dir=stage_dir,
                reuse_existing=reuse_existing,
            )
            results[index] = record
            with lock:
                counter["done"] += 1
                if record.get("failed"):
                    status = "FAIL"
                else:
                    status = (
                        f"val={float(record['validation_nll']):+.4f} "
                        f"ceil={100.0 * float(record['clip_saturation_high']):.3f}% "
                        f"p99={float(record['spectrum_p99']):.3f}"
                    )
                print(
                    f"[{label} {counter['done']:>3}/{len(points)}] "
                    f"{_tag(point):<28} {str(record['device']):<6} "
                    f"{float(record['elapsed_sec']):>6.1f}s  {status}",
                    flush=True,
                )

    threads = [
        threading.Thread(target=consume, args=(w,), daemon=True) for w in workers
    ]
    started = time.time()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print(f"[{label}] {len(points)} points in {time.time() - started:.0f}s", flush=True)
    return [r for r in results if r is not None]


def _build_workers(
    gpus: list[int], cpu_workers: int, threads: int
) -> list[dict[str, object]]:
    workers: list[dict[str, object]] = []
    cursor = 0
    for gpu in gpus:
        workers.append(
            {"gpu": gpu, "cores": f"{cursor}-{cursor + threads - 1}", "threads": threads}
        )
        cursor += threads
    for _ in range(int(cpu_workers)):
        workers.append(
            {"gpu": None, "cores": f"{cursor}-{cursor + threads - 1}", "threads": threads}
        )
        cursor += threads
    return workers


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-paths", type=int, default=2048)
    parser.add_argument("--confirm-paths", type=int, default=6144)
    parser.add_argument("--confirm-top", type=int, default=3)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="reuse per-point json files already present in --work-dir instead "
             "of re-running those points. Only successful points leave a json, "
             "so crashed points are always re-run.",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--screen-saturation-paths", type=int, default=256)
    parser.add_argument("--confirm-saturation-paths", type=int, default=1024)
    parser.add_argument(
        "--gpus",
        type=str,
        default="1,2,3",
        help="comma-separated; GPU 0 is excluded by default because it is not ours",
    )
    parser.add_argument("--cpu-workers", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=LOSSES_DIR)
    parser.add_argument(
        "--work-dir", type=Path, default=Path("/tmp/reference_sweep_true")
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    gpus = [int(g) for g in str(args.gpus).replace(",", " ").split() if g != ""]
    workers = _build_workers(gpus, int(args.cpu_workers), int(args.threads))
    total_cores = len(workers) * int(args.threads)
    print(
        f"[pool] {len(gpus)} GPU workers {gpus} + {args.cpu_workers} CPU workers, "
        f"{args.threads} threads each = {total_cores} cores"
    )
    if total_cores > 64:
        raise SystemExit(
            f"ABORT: pool asks for {total_cores} cores, the sanctioned budget is 64"
        )

    points = _grid()
    print(
        f"[grid] {len(points)} points = "
        f"{len(SIGMA_MAX_GRID)} sigma_max x {len(RIDGE_COVARIANCE_GRID)} ridge_cov "
        f"x {len(RIDGE_DRIFT_GRID)} ridge_drift   lr={args.lr:g} steps={args.steps}"
    )

    # ---------------------------------------------------------------- screen --
    screen = _execute(
        points,
        workers=workers,
        num_paths=int(args.screen_paths),
        steps=int(args.steps),
        lr=float(args.lr),
        saturation_paths=int(args.screen_saturation_paths),
        stage_dir=Path(args.work_dir) / "screen",
        label="screen",
        reuse_existing=bool(args.reuse_existing),
    )
    for record in screen:
        ok, reason = _admissible(record)
        record["stage"] = "screen"
        record["admissible"] = bool(ok)
        record["reject_reason"] = reason
    _write_csv(Path(args.out_dir) / "reference_sweep_screen.csv", screen)

    admissible = [r for r in screen if r["admissible"]]
    print(f"\n[gate] {len(admissible)}/{len(screen)} points admissible at screen scale")
    for record in screen:
        if not record["admissible"]:
            print(f"  reject {_tag(record):<28} {record['reject_reason']}")
    if not admissible:
        raise SystemExit(
            "ABORT: no admissible point.  Every ceiling in SIGMA_MAX_GRID still "
            "binds, or no point beats the constant-covariance baseline.  Widen "
            "SIGMA_MAX_GRID upward -- do NOT relax the gate."
        )

    admissible.sort(key=lambda r: float(r["validation_nll"]))
    shortlist = admissible[: int(args.confirm_top)]
    print(f"\n[shortlist] confirming top {len(shortlist)} at N={args.confirm_paths}")
    for rank, record in enumerate(shortlist, start=1):
        print(f"  {rank}. {_tag(record):<28} val={float(record['validation_nll']):+.4f}")

    # --------------------------------------------------------------- confirm --
    confirm = _execute(
        [
            {
                "sigma_max": float(r["sigma_max"]),
                "ridge_covariance": float(r["ridge_covariance"]),
                "ridge_drift": float(r["ridge_drift"]),
            }
            for r in shortlist
        ],
        workers=workers,
        num_paths=int(args.confirm_paths),
        steps=int(args.steps),
        lr=float(args.lr),
        saturation_paths=int(args.confirm_saturation_paths),
        stage_dir=Path(args.work_dir) / "confirm",
        label="confirm",
        reuse_existing=bool(args.reuse_existing),
    )
    for record in confirm:
        ok, reason = _admissible(record)
        record["stage"] = "confirm"
        record["admissible"] = bool(ok)
        record["reject_reason"] = reason
    _write_csv(Path(args.out_dir) / "reference_sweep_confirm.csv", confirm)

    survivors = [r for r in confirm if r["admissible"]]
    print(f"\n[gate] {len(survivors)}/{len(confirm)} survive at full scale")
    for record in confirm:
        if not record["admissible"]:
            print(f"  reject {_tag(record):<28} {record['reject_reason']}")
    if not survivors:
        raise SystemExit(
            "ABORT: every shortlisted point failed the gate at full scale.  The "
            "screen was measuring subsample noise; raise --screen-paths."
        )

    survivors.sort(key=lambda r: float(r["validation_nll"]))
    selected = survivors[0]

    selection = {
        "selected": {
            "sigma_max": float(selected["sigma_max"]),
            "ridge_covariance": float(selected["ridge_covariance"]),
            "ridge_drift": float(selected["ridge_drift"]),
            "lr": float(args.lr),
            "steps": int(args.steps),
            "validation_nll": float(selected["validation_nll"]),
            "baseline_nll": float(selected["baseline_nll"]),
            "clip_saturation_high": float(selected["clip_saturation_high"]),
            "spectrum_p99": float(selected["spectrum_p99"]),
        },
        "gate": {
            "rule": "p99 < sigma_max AND ceiling_mass <= MAX_CEILING_MASS AND "
            "floor_mass <= MAX_FLOOR_MASS AND validation_nll < baseline_nll",
            "max_ceiling_mass": MAX_CEILING_MASS,
            "max_floor_mass": MAX_FLOOR_MASS,
            "objective": "minimise validation_nll among admissible points",
            "why_not_nll_alone": "the likelihood is minimised by a censoring "
            "ceiling; clipping removes variance and a Gaussian NLL rewards that",
        },
        "stage_sizes": {
            "screen_paths": int(args.screen_paths),
            "confirm_paths": int(args.confirm_paths),
            "grid_points": len(points),
            "admissible_at_screen": len(admissible),
            "confirmed": len(survivors),
        },
        "grid": {
            "sigma_max": list(SIGMA_MAX_GRID),
            "ridge_covariance": list(RIDGE_COVARIANCE_GRID),
            "ridge_drift": list(RIDGE_DRIFT_GRID),
        },
    }
    out = Path(args.out_dir) / "reference_selection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selection, indent=2))

    print("\n" + "=" * 78)
    print("SELECTED reference-kernel hyperparameters")
    print("=" * 78)
    for key, value in selection["selected"].items():
        print(f"  {key:24s} {value}")
    print(f"\n[out] {out}")
    print("\nFreeze it with:\n")
    print(
        f"  cd /home/tbasseras/benchmark && {PYTHON} \\\n"
        f"      {FITTER} \\\n"
        f"      --sigma-max {float(selected['sigma_max']):g} "
        f"--ridge-covariance {float(selected['ridge_covariance']):g} "
        f"--ridge-drift {float(selected['ridge_drift']):g} \\\n"
        f"      --lr {args.lr:g} --steps {args.steps} --device cuda:0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
