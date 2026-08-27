#!/usr/bin/env python3
"""Instrument the control-map boundary to find WHERE Theta goes non-finite.

Why this file exists
--------------------
The stage-2 timing probe died on the very first ``model.fit`` rollout with

    Theta: shape=(256, 8, 8), dtype=torch.float32,
    non_finite_entries=16384, bad_batch_elements=[0..7] of 256

i.e. ALL 16384 entries non-finite at once, not a subset. ``eigh_fallback``
correctly refused to retry on CPU: a non-finite Theta is an exploding control,
not a cuSOLVER convergence problem.

``Theta = eta * sigma_ref^-1 + R / sqrt(dt)`` (specific_entropy_matrix.py:196).
Three candidate root causes were identified BY READING, and this script exists
because reading cannot distinguish them:

  (1) the 1/sqrt(dt) amplifier. dt = 9.512937595129376e-07 -> 1/sqrt(dt) = 1025.3,
      which is 64.6x the Heston value of 15.87 (= sqrt(1051200/252)). If R keeps
      its Heston magnitude, the noise term is 64.6x larger than it ever was.
  (2) the NaN-permeable positive-definiteness guard. Line 189 reads
      ``if torch.any(ref_eigenvalues <= 0.0)``; NaN <= 0 is False, so a NaN
      sigma_ref passes the guard, ``eta / NaN`` is NaN, and EVERY entry of Theta
      becomes NaN simultaneously -- which matches 16384/16384 exactly.
  (3) float32 overflow (the model runs in torch.float32, max ~3.4e38).

What is measured
----------------
Every call of the control map is wrapped. Per call we record, WITHOUT changing
any value that flows onward:

    x_prefix last row : min / max / non-finite count
    p  (adjoint)      : absolute max / non-finite count
    r  (noise adjoint): absolute max / non-finite count
    sigma_ref         : absolute max / non-finite count / eigenvalue range
    reference_inverse : absolute max          <- term 1 of Theta
    r / sqrt(dt)      : absolute max          <- term 2 of Theta
    Theta             : absolute max / non-finite count

The loop stops recording detail once it has captured the first bad step plus the
two steps before it, which is the only window that identifies the source: the
step where a quantity FIRST becomes non-finite names the culprit, every later
step is downstream contamination.

This script writes nothing and monkeypatches only in-process. The frozen package
under ``methods/Deep-MKV-TS/code/reference`` is not touched.

Usage
-----
    R=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 PYTHONPATH="$R/src:$R/experiments" \\
    taskset -c 0-7 /home/tbasseras/gpu-venv/bin/python diagnose_theta_nan.py
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REFERENCE = Path("/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference")
for _path in (HERE, REFERENCE / "src", REFERENCE / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid  # noqa: E402

import matrix_control_multiasset as mc  # noqa: E402
import train_true as tt  # noqa: E402
from fit_reference_true import DT  # noqa: E402

from deep_mkv_gen_path_dt.controls.specific_entropy_matrix import (  # noqa: E402
    _as_matrix,
    _symmetrise,
)

RECORDS: list[dict] = []
ROLLOUTS: list[dict] = []  # one row per rollout, for the growth curve
STOP_AFTER_BAD = 2  # keep this many steps past the first bad one, then go quiet


def _stats(name: str, value: torch.Tensor) -> dict:
    with torch.no_grad():
        finite = torch.isfinite(value)
        bad = int((~finite).sum().item())
        if bad == value.numel():
            lo = hi = float("nan")
        else:
            masked = torch.where(finite, value, torch.zeros_like(value))
            lo = float(masked.min().item())
            hi = float(masked.max().item())
    return {f"{name}_bad": bad, f"{name}_min": lo, f"{name}_max": hi}


def install(control) -> None:
    """Wrap the control map. The returned value is the untouched original."""

    cls = type(control)
    original = cls.__call__
    inv_sqrt_dt = 1.0 / math.sqrt(float(control.dt))
    eta = float(control.eta)
    state_dim = int(control.reference_kernel.state_dim)
    first_bad = {"step": None}

    def patched(self, inputs):  # noqa: ANN001
        step = int(inputs.step_index)
        quiet = (
            first_bad["step"] is not None
            and step > first_bad["step"] + STOP_AFTER_BAD
        )
        if quiet:
            return original(self, inputs)

        with torch.no_grad():
            row = {"step": step}
            row.update(_stats("x", inputs.x_prefix[:, -1, :]))
            row.update(_stats("p", inputs.expected_adjoint_next))
            row.update(_stats("r", inputs.expected_adjoint_noise_next))

            # Recompute sigma_ref exactly as __call__ does, WITHOUT the spectrum
            # check, so a NaN spectrum is observed rather than raised on.
            _, sigma_ref = self.reference_kernel.evaluate(
                inputs.x_prefix, step_index=step
            )
            row.update(_stats("sref", sigma_ref))
            sref_mat = _as_matrix(sigma_ref, name="sigma_ref", state_dim=state_dim)
            try:
                ev = torch.linalg.eigvalsh(_symmetrise(sref_mat.double()))
                row.update(_stats("sref_eig", ev))
            except Exception as exc:  # noqa: BLE001 - the failure IS the datum
                row["sref_eig_error"] = type(exc).__name__

            # The GRU head emits Z flat as (B, d*d); the multiasset subclass
            # unflattens it before the base assembly (matrix_control_multiasset
            # :176-177). Mirror that here or _as_matrix rejects the row.
            r_raw = inputs.expected_adjoint_noise_next
            if r_raw.ndim == 2 and int(r_raw.shape[-1]) == state_dim * state_dim:
                r_raw = r_raw.reshape(-1, state_dim, state_dim)
            r_mat = _symmetrise(
                _as_matrix(
                    r_raw,
                    name="expected_adjoint_noise_next",
                    state_dim=state_dim,
                )
            )
            row.update(_stats("noise_term", r_mat * inv_sqrt_dt))
            try:
                evl, evec = torch.linalg.eigh(_symmetrise(sref_mat.double()))
                ref_inv = (
                    evec @ torch.diag_embed(eta / evl) @ evec.transpose(-1, -2)
                )
                row.update(_stats("ref_inv", ref_inv))
                theta = _symmetrise(ref_inv.float() + r_mat * inv_sqrt_dt)
                row.update(_stats("theta", theta))
                # Degeneracy probe. eigh's BACKWARD divides by (lambda_i -
                # lambda_j); exactly repeated eigenvalues make that 1/0. Two
                # eigenvalues of sigma_ref both clipped to the ceiling would be
                # exactly equal, and the inverse would inherit the tie -- so the
                # gap is measured on BOTH matrices, not assumed.
                for name, mat in (("sref", sref_mat), ("theta", theta)):
                    ev = torch.linalg.eigvalsh(_symmetrise(mat.double()))
                    gaps = ev[..., 1:] - ev[..., :-1]
                    row[f"{name}_min_gap"] = float(gaps.min().item())
                    row[f"{name}_exact_ties"] = int((gaps == 0.0).sum().item())
            except Exception as exc:  # noqa: BLE001
                row["ref_inv_error"] = type(exc).__name__

            bad_here = any(
                row.get(f"{k}_bad", 0) > 0
                for k in ("x", "p", "r", "sref", "theta")
            )
            if bad_here and first_bad["step"] is None:
                first_bad["step"] = step
            RECORDS.append(row)

            # Growth curve: one row per rollout. A rollout begins at step 0, so
            # that is where a new bucket is opened. This is the measurement that
            # separates "explodes at init" from "explodes under the optimizer".
            if step == 0:
                ROLLOUTS.append(
                    {"rollout": len(ROLLOUTS), "r": 0.0, "theta": 0.0, "bad": 0}
                )
            if ROLLOUTS:
                cur = ROLLOUTS[-1]
                for key in ("r", "theta"):
                    hi, lo = row[f"{key}_max"], row[f"{key}_min"]
                    mag = max(abs(hi), abs(lo)) if hi == hi else float("inf")
                    cur[key] = max(cur[key], mag)
                cur["bad"] += sum(
                    row.get(f"{k}_bad", 0) for k in ("x", "p", "r", "sref", "theta")
                )

        return original(self, inputs)

    cls.__call__ = patched


def install_ridge_probe() -> None:
    """Report the finiteness of the ridge CE inputs vs its output.

    This is the fork in the road. If ``targets_train`` is ALREADY non-finite the
    NaN was manufactured upstream, in the adjoint labels -- which are produced by
    differentiating through the rollout, i.e. through ``eigh``. If the targets
    are finite and the OUTPUT is not, the linear solve itself is the culprit.
    ``fit_ridge`` already guards its Cholesky with ``cholesky_ex`` + an
    ``lstsq`` fallback, so the second branch would be surprising -- which is
    exactly why it has to be measured rather than assumed away.
    """

    from deep_mkv_gen_path_dt.ce.ridge import RidgeConditionalExpectation as R

    original = R.fit_predict
    calls = {"n": 0}

    def patched(self, **kwargs):  # noqa: ANN001, ANN003
        calls["n"] += 1
        pre = kwargs.get("prefixes_train")
        tgt = kwargs.get("targets_train")
        bad_pre = int((~torch.isfinite(pre)).sum().item())
        bad_tgt = int((~torch.isfinite(tgt)).sum().item())
        result = original(self, **kwargs)
        bad_out = int((~torch.isfinite(result.prediction)).sum().item())
        if (bad_pre or bad_tgt or bad_out) and calls["n"] % 25 == 0:
            print(
                f"[ridge] call {calls['n']}: NON-FINITE  "
                f"prefixes_train={bad_pre}/{pre.numel()}  "
                f"targets_train={bad_tgt}/{tgt.numel()}  "
                f"prediction={bad_out}/{result.prediction.numel()}",
                flush=True,
            )
        return result

    R.fit_predict = patched


def dump(per_step: bool = True) -> None:
    if not RECORDS:
        print("no control-map calls were recorded", flush=True)
        return
    if ROLLOUTS:
        print(
            "\ngrowth curve -- one row per rollout, max over the 127 dates\n"
            "rollout      max|r|    max|Theta|   non-finite",
            flush=True,
        )
        for row in ROLLOUTS:
            print(
                f"{row['rollout']:7d} {row['r']:11.5g} {row['theta']:12.5g} "
                f"{row['bad']:12d}",
                flush=True,
            )
    if not per_step:
        first = next(
            (r for r in ROLLOUTS if r["bad"] > 0 or not math.isfinite(r["theta"])),
            None,
        )
        if first is None:
            print("\nNO non-finite value in any rollout.", flush=True)
        else:
            print(f"\nFIRST non-finite rollout: {first['rollout']}", flush=True)
        return
    keys = ("x", "p", "r", "sref", "sref_eig", "ref_inv", "noise_term", "theta")
    print(
        "\nstep " + " ".join(f"{k:>12}|bad" for k in keys),
        flush=True,
    )
    for row in RECORDS:
        cells = []
        for k in keys:
            if f"{k}_max" not in row:
                cells.append(f"{'--':>12}|---")
                continue
            hi = row[f"{k}_max"]
            lo = row[f"{k}_min"]
            mag = max(abs(hi), abs(lo)) if hi == hi else float("nan")
            cells.append(f"{mag:12.4g}|{row[f'{k}_bad']:3d}")
        print(f"{row['step']:4d} " + " ".join(cells), flush=True)
    print(
        "\nlegend: |bad = count of non-finite entries in that tensor at that step.\n"
        "The FIRST column that shows bad > 0 names the source; everything to its "
        "right at the same step is downstream contamination.",
        flush=True,
    )
    gaps = [r for r in RECORDS if "sref_min_gap" in r and "theta_min_gap" in r]
    if gaps:
        print(
            "\neigenvalue-degeneracy probe (eigh backward divides by lambda_i - lambda_j)\n"
            f"  sigma_ref : min gap over all recorded dates = "
            f"{min(r['sref_min_gap'] for r in gaps):.6g}   "
            f"exact ties = {sum(r['sref_exact_ties'] for r in gaps)}\n"
            f"  Theta     : min gap over all recorded dates = "
            f"{min(r['theta_min_gap'] for r in gaps):.6g}   "
            f"exact ties = {sum(r['theta_exact_ties'] for r in gaps)}",
            flush=True,
        )
    first = next((r for r in RECORDS if any(r.get(f"{k}_bad", 0) for k in keys)), None)
    if first is None:
        print("NO non-finite value was observed in any recorded step.", flush=True)
    else:
        culprits = [k for k in keys if first.get(f"{k}_bad", 0) > 0]
        print(
            f"FIRST non-finite at step {first['step']} in: {', '.join(culprits)}",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ridge-lambda", type=float, default=1000.0)
    ap.add_argument("--num-paths", type=int, default=6144)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--per-step", action="store_true")
    ap.add_argument(
        "--anomaly",
        action="store_true",
        help=(
            "run under torch.autograd.set_detect_anomaly, which names the "
            "forward op whose BACKWARD first produced a non-finite value. Slow, "
            "so pair it with a small --steps."
        ),
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    target = tt.load_log_prices(num_paths=int(args.num_paths), device=device).to(
        tt.DTYPE
    )
    num_steps = int(target.shape[1]) - 1
    grid = DiscreteTimeGrid(T=num_steps * DT, num_steps=num_steps)
    print(f"[data] {tuple(target.shape)} dt={DT} 1/sqrt(dt)={1.0 / math.sqrt(DT):.2f}")

    model, sigma_max = tt.build_model(
        grid=grid, device=device, seed=int(args.seed), kernel_steps=num_steps
    )
    print(f"[model] sigma clip [{tt.SIGMA_MIN}, {sigma_max}]", flush=True)
    install(model.control_map)
    install_ridge_probe()

    training = replace(
        tt.build_training(
            seed=int(args.seed),
            steps=int(args.steps),
            ridge_lambda=float(args.ridge_lambda),
        ),
        log_every=1,
    )

    def report(_model, metrics: dict) -> None:
        """Print the per-step loss/gradient trail already computed inside fit.

        The interesting quantity is `grad_norm_unclipped`: `clip_grad_norm_`
        rescales a large FINITE gradient, but leaves a non-finite one exactly
        as it found it, so grad_clip_norm=5.0 is no protection against inf.
        """

        keys = (
            "step",
            "train_total_loss",
            "grad_norm",
            "grad_clip_scale",
            "gradient_nonfinite_fraction",
            "parameter_nonfinite_fraction",
            "parameter_update_norm",
            "noise_adjoint_prediction_target_rms_ratio",
            "noise_ce_projected_target_rms",
            "ce_projection_r2",
            "control_rms",
        )
        short = {
            "train_total_loss": "loss",
            "grad_norm": "g",
            "grad_clip_scale": "clip",
            "gradient_nonfinite_fraction": "gNaN",
            "parameter_nonfinite_fraction": "wNaN",
            "parameter_update_norm": "dW",
            "noise_adjoint_prediction_target_rms_ratio": "r_pred/tgt",
            "noise_ce_projected_target_rms": "Ztgt_rms",
            "ce_projection_r2": "ridgeR2",
            "control_rms": "ctrl",
        }
        print(
            "  ".join(
                f"{short.get(k, k)}={float(metrics.get(k, float('nan'))):.5g}"
                for k in keys
            ),
            flush=True,
        )

    if bool(args.anomaly):
        torch.autograd.set_detect_anomaly(True)
        print("[anomaly] autograd anomaly detection ON", flush=True)
    try:
        model.fit(target, training=training, log_callback=report)
        print(f"\nfit(steps={args.steps}) COMPLETED without raising.", flush=True)
    except Exception as exc:  # noqa: BLE001 - the failure IS the datum
        print(f"\nfit raised {type(exc).__name__}: {exc}", flush=True)
    finally:
        dump(per_step=bool(args.per_step))


if __name__ == "__main__":
    main()
