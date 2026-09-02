#!/usr/bin/env python3
"""Split the logged ``objective`` into its discrepancy and running-cost parts.

Why this file exists
--------------------
``losses/seed_*_losses.csv`` has one column named ``objective`` and the obvious
reading -- "distance between generated paths and the bank" -- is WRONG.  The
column is ``complete_objective_value`` (train_multiasset.py:350), and
``model.py:1875-1893`` builds that as

    objective = LAMBDA_SCALE * discrepancy  +  running_cost

with ``LAMBDA_SCALE = 50``.  The running cost is a control-effort penalty, not a
fit term (``controls/specific_entropy_matrix.py:240-262``):

    running_cost = dt * sum_t [ 0.5 * |alpha_t|^2 + entropy(sigma_t, sigma^ref_t) ]

where the packed control is ``[alpha, sigma.flatten()]``.  So a run can plateau
at a large ``objective`` while fitting the bank perfectly well, if ``alpha`` is
large.

That matters here specifically.  This arm's reference drift is
``b^ref = E_w[R_{i+1}] / dt`` (``sbts_reference.py``, equation (*)): a
kernel-weighted average log-return DIVIDED BY dt = 1/252.  Dividing a ~1e-2
quantity by ~4e-3 gives an ``alpha`` of order 1, and ``0.5|alpha|^2`` summed over
252 steps is then order 10.  The sibling arm's reference drift carries no such
1/dt factor.  If that is what is happening, the ~14 plateau this arm reports and
the ~2.4 the sibling reports are NOT measuring the same thing and the "6x worse"
comparison is meaningless.

Crucially, ``allow_drift_correction`` is FALSE in this arm's incumbent, so
``alpha`` IS the reference drift exactly -- Algorithm 1 cannot touch it.  Any
drift cost is therefore a CONSTANT FLOOR added to every logged objective, which
no amount of training can reduce.  Only the volatility part is learnable.

The already-measured zero-control discrepancy (``sweep/null_control.json``,
0.22584774 vs the sibling's 0.30009511) says the reference is the better
starting point, so the plateau cannot be blamed on the reference's fit.  This
script tests the remaining explanation: that the plateau is control-effort
accounting.

What it prints
--------------
For the UNFITTED model (``Zhat == 0``, pure reference SDE -- the same null model
``null_control_baseline.py`` scores) it reports, against the TRAIN bank so the
numbers are directly comparable to the logged ``objective`` column:

    complete_objective_value = LAMBDA_SCALE * discrepancy + running_cost
    running_cost             = drift part + volatility part
    alpha rms, sigma diagonal range, and the sigma_max clip check

Read it against ``losses/seed_*_losses.csv`` step 100.  If the drift part alone
accounts for most of the plateau, the plateau is not a fit failure.

``--arm main`` runs the identical decomposition on the SIBLING campaign
(``../../Deep-MKV-TS/code``), because "the drift cost is large" is only
meaningful relative to the run it is being compared against.  The two arms need
different builders -- this arm's ``build_model`` takes the SBTS kernel
hyperparameters from ``sweep/incumbent.json``, the sibling's takes only
``kernel_steps`` and has no incumbent file at all -- so the branch is explicit
rather than introspected.  Everything downstream of the build is shared code, so
the two numbers are produced by one implementation and cannot drift apart.

Hyperparameters come from ``sweep/incumbent.json`` by SUBSCRIPT, never ``.get``,
for the same reason as in ``null_control_baseline.py``: a defaulted field would
silently measure a different model than the promoted one.

The TEST split is never opened; both arms are decomposed against their TRAIN
bank, which is what the logged ``objective`` column is computed on.

Usage
-----
    R=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
    PYTHONPATH="$R/src:$R/experiments" taskset -c 0-3 \
    /home/tbasseras/gpu-venv/bin/python objective_decomposition.py \
        --device cuda:0 --arm sbtsref
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REFERENCE = Path("/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference")
# The sibling campaign has its own ``train_multiasset`` and ``sweep_hyperparams``
# with the SAME module names, so the arm-specific code directory is chosen at
# runtime and those modules are imported inside ``_build_arm``. Importing either
# at module scope would bind whichever directory happened to be first on the
# path and silently decompose the wrong arm.
ARM_CODE_DIRS = {
    "sbtsref": HERE,
    "main": HERE.parent.parent / "Deep-MKV-TS" / "code",
}

for _path in (REFERENCE / "src", REFERENCE / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from deep_mkv_gen_path_dt.controls.base import RunningCostInputs  # noqa: E402
from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid  # noqa: E402


def _build_arm(arm: str, device: torch.device, paths: int):
    """Return ``(model, target_log_prices, grid, tm, meta)`` for one campaign."""

    code_dir = ARM_CODE_DIRS[arm]
    if not code_dir.is_dir():
        raise FileNotFoundError(f"arm {arm!r} code directory missing: {code_dir}")
    sys.path.insert(0, str(code_dir))

    import eigh_fallback  # noqa: PLC0415

    eigh_fallback.install()

    import train_multiasset as tm  # noqa: PLC0415

    # The two arms keep S0 and the build seed in differently NAMED modules --
    # this arm renamed its sweep driver to sweep_hyperparams, the sibling still
    # calls it sweep_ridge_lambda -- so the module is selected with the arm.
    # Both are the module each arm's own null_control_baseline.py imports, which
    # is what makes the two decompositions use the same seed as their own runs.
    if arm == "sbtsref":
        from sweep_hyperparams import S0, SWEEP_SEED  # noqa: PLC0415
    else:
        from sweep_ridge_lambda import S0, SWEEP_SEED  # noqa: PLC0415

    if Path(tm.__file__).resolve().parent != code_dir:
        raise RuntimeError(
            f"imported train_multiasset from {tm.__file__}, expected {code_dir}; "
            "the wrong arm would be decomposed"
        )

    meta: dict[str, object] = {"arm": arm, "build_seed": int(SWEEP_SEED)}

    if arm == "sbtsref":
        bank_prices = tm.load_prices()
        num_steps = int(bank_prices.shape[1]) - 1
        if int(bank_prices.shape[2]) != tm.STATE_DIM:
            raise ValueError(f"expected d = {tm.STATE_DIM}, got {bank_prices.shape}")
        grid = DiscreteTimeGrid(T=num_steps * tm.DT, num_steps=num_steps)
        with open(code_dir / "sweep" / "incumbent.json", encoding="utf-8") as handle:
            incumbent = json.load(handle)
        model, _kernel = tm.build_model(
            grid=grid,
            device=device,
            seed=int(SWEEP_SEED),
            bank_prices=bank_prices,
            h=float(incumbent["h"]),
            markov_order=int(incumbent["markov_order"]),
            npi=int(incumbent["npi"]),
            weight_grad_mode=str(incumbent["weight_grad_mode"]),
            jacobian_lags=int(incumbent["jacobian_lags"]),
            allow_drift_correction=bool(incumbent["allow_drift_correction"]),
        )
        target = torch.as_tensor(
            np.ascontiguousarray(bank_prices[:paths]),
            device=device,
            dtype=model.dtype,
        ).log()
        meta["allow_drift_correction"] = bool(incumbent["allow_drift_correction"])
    elif arm == "main":
        # The sibling has no sweep/incumbent.json: its kernel is loaded from a
        # fitted checkpoint and ``build_model`` takes only ``kernel_steps``.
        target = tm.load_log_prices(num_paths=paths, device=device).to(tm.DTYPE)
        num_steps = int(target.shape[1]) - 1
        grid = DiscreteTimeGrid(T=num_steps * tm.DT, num_steps=num_steps)
        model = tm.build_model(
            grid=grid, device=device, seed=int(SWEEP_SEED), kernel_steps=num_steps
        )
        # Recorded as unknown rather than guessed: this arm exposes no such flag,
        # and writing False would assert something the code does not say.
        meta["allow_drift_correction"] = None
    else:
        raise ValueError(f"unknown arm {arm!r}")

    meta["s0"] = float(S0)
    meta["num_steps"] = num_steps
    return model, target, grid, tm, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--arm", choices=sorted(ARM_CODE_DIRS), default="sbtsref")
    # The training loop's objective is a batch statistic, not a population one.
    # Matching its batch size is what makes the printed split comparable to the
    # CSV column rather than merely similar to it.
    parser.add_argument("--paths", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()

    device = torch.device(args.device)

    model, target, grid, tm, meta = _build_arm(args.arm, device, int(args.paths))
    num_steps = int(meta["num_steps"])
    # NO fit(). The decomposition is of the reference SDE, matching the null
    # control, so the drift floor it exposes is the floor every seed started from.

    if model.running_cost is None:
        raise RuntimeError(
            "model.running_cost is None; there is no control-effort term to "
            "decompose and the premise of this script is wrong"
        )

    x0 = torch.full(
        (1, tm.STATE_DIM), math.log(float(meta["s0"])), device=device, dtype=model.dtype
    )

    with torch.no_grad():
        sampled = model.sample(num_paths=int(args.paths), x0=x0, seed=int(args.seed))
    generated, controls = sampled.paths, sampled.controls
    if not torch.isfinite(generated).all():
        raise RuntimeError("zero-control rollout contains non-finite values")

    # Packed layout is [alpha, sigma.flatten(-2)] -- specific_entropy_matrix.py:154.
    d = int(tm.STATE_DIM)
    alpha = controls[..., :d]
    sigma = controls[..., d:].unflatten(-1, (d, d))
    sigma_diag = torch.diagonal(sigma, dim1=-2, dim2=-1)

    with torch.no_grad():
        discrepancy = model.discrepancy(
            generated_paths=generated, target_paths=target, grid=grid
        )
        running = model.running_cost(
            RunningCostInputs(
                paths=generated,
                controls=controls,
                grid=grid,
                parameters=model.parameters,
            )
        )

    disc_raw = float(discrepancy.value)
    lam = float(tm.LAMBDA_SCALE)
    disc_term = lam * disc_raw
    run_total = float(running.value)
    run_drift = float(running.metrics["running_cost_drift_value"])
    run_vol = float(running.metrics["running_cost_volatility_value"])
    complete = disc_term + run_total

    print(
        f"[decomp] arm={args.arm} paths={args.paths} steps={num_steps} d={d} "
        f"LAMBDA_SCALE={lam} eta={tm.ETA} sigma_max={tm.SIGMA_MAX}\n"
        f"[decomp] objective              = {complete:.6g}\n"
        f"[decomp]   discrepancy (raw)    = {disc_raw:.6g}\n"
        f"[decomp]   discrepancy * lambda = {disc_term:.6g}"
        f"   ({100.0 * disc_term / complete:.1f}% of objective)\n"
        f"[decomp]   running cost         = {run_total:.6g}"
        f"   ({100.0 * run_total / complete:.1f}% of objective)\n"
        f"[decomp]     drift part         = {run_drift:.6g}"
        f"   ({100.0 * run_drift / complete:.1f}% of objective)  "
        f"allow_drift_correction={meta['allow_drift_correction']}\n"
        f"[decomp]     volatility part    = {run_vol:.6g}"
        f"   ({100.0 * run_vol / complete:.1f}% of objective)\n"
        f"[decomp] alpha rms = {float(alpha.pow(2).mean().sqrt()):.6g}  "
        f"|alpha|_max = {float(alpha.abs().max()):.6g}\n"
        f"[decomp] sigma diag in [{float(sigma_diag.min()):.6g}, "
        f"{float(sigma_diag.max()):.6g}]  "
        f"at_sigma_max_fraction = "
        f"{float((sigma_diag >= float(tm.SIGMA_MAX) - 1e-9).double().mean()):.4g}\n"
        f"[decomp] control_rms (all packed entries, matches the CSV column) = "
        f"{float(controls.pow(2).mean().sqrt()):.6g}",
        flush=True,
    )

    # Always written beside THIS script, never into the sibling's tree: the
    # sibling arm is read here for comparison only and its artefacts are not
    # this campaign's to author.
    out = HERE / "sweep" / f"objective_decomposition_{args.arm}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "date": date.today().isoformat(),
                "arm": args.arm,
                "arm_code_dir": str(ARM_CODE_DIRS[args.arm]),
                "paths": int(args.paths),
                "sample_seed": int(args.seed),
                "build_seed": meta["build_seed"],
                "steps": num_steps,
                "state_dim": d,
                "lambda_scale": lam,
                "eta": float(tm.ETA),
                "sigma_max": float(tm.SIGMA_MAX),
                "fitted": False,
                "complete_objective_value": complete,
                "discrepancy_raw": disc_raw,
                "discrepancy_times_lambda": disc_term,
                "running_cost_value": run_total,
                "running_cost_drift_value": run_drift,
                "running_cost_volatility_value": run_vol,
                "alpha_rms": float(alpha.pow(2).mean().sqrt()),
                "alpha_abs_max": float(alpha.abs().max()),
                "sigma_diag_min": float(sigma_diag.min()),
                "sigma_diag_max": float(sigma_diag.max()),
                "control_rms": float(controls.pow(2).mean().sqrt()),
                "allow_drift_correction": meta["allow_drift_correction"],
                "note": (
                    "Decomposition of the UNFITTED (Zhat == 0) model's objective "
                    "against the TRAIN bank, so it is comparable to the "
                    "'objective' column of losses/seed_*_losses.csv. That column "
                    "is complete_objective_value = LAMBDA_SCALE * discrepancy + "
                    "running_cost, NOT a pure goodness-of-fit number. With "
                    "allow_drift_correction false the drift part of the running "
                    "cost is a constant that Algorithm 1 cannot reduce."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[out] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
