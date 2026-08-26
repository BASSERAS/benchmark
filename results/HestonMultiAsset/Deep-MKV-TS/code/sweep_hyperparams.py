#!/usr/bin/env python3
"""Post-hoc CHECK that the d = 1 hyperparameters are near-optimal at d = 8.

What this is, and what it is not
--------------------------------
This is a VERIFICATION, not a re-selection.  The campaign in
``train_multiasset.py`` deliberately copies ``LAMBDA_SCALE``, ``KAPPA_SCALE``,
``ABS_RETURN_ACF_WEIGHT`` and ``SQUARED_RETURN_ACF_WEIGHT`` byte-for-byte from
the committed d = 1 run so that any difference in the benchmark table is
attributable to the DIMENSION and not to a re-tuned pipeline.  That design
choice explains where the numbers came from; it is not evidence they are right
at d = 8.  This script measures how far off they are.

Nothing here feeds back into the campaign automatically.  The output is a table.

Why a separate file instead of flags on ``sweep_ridge_lambda.py``
-----------------------------------------------------------------
``sweep_ridge_lambda.py`` names its output ``sweep/lambda_<ridge>.json``, keyed
on the ridge alone.  Two runs sharing a ridge but differing in ``lambda_scale``
would silently overwrite each other, and the ``sweep/`` directory is the frozen
record the README tabulates.  So this script:

  * writes to ``sweep_verify/`` -- a NEW directory; ``sweep/`` is never touched,
  * encodes every varied parameter in the filename,
  * reuses ``sweep_ridge_lambda.run_one`` UNCHANGED, so the fit, the sampling
    seed, the scoring split and the discrepancy are bit-identical to the
    original sweep.  A number produced here is directly comparable to a number
    in ``sweep/``.

How the overrides reach the model
---------------------------------
``train_multiasset.build_model`` calls ``build_heston_discrepancy`` with the
MODULE-LEVEL constants, read at call time, not captured at import.  Rebinding
``tm.LAMBDA_SCALE`` before ``run_one`` therefore reaches the discrepancy exactly
as editing the source would -- without editing the source.  ``build_training``
reads ``LAMBDA_SCALE`` the same way.  Verified by asserting the value landed in
the returned config before the fit starts.

The two different lambdas
-------------------------
``lambda_scale``  is the WEIGHT ON THE DISCREPANCY in the objective:
                  ``total_objective = lambda_scale * discrepancy.value``
                  (``model.py`` line 1374).  With ``adjoint_weight = 0`` it sets
                  the magnitude of ``Z_proxy``, i.e. how hard the control is
                  pushed away from the reference SDE.  First-order knob.
``ridge_lambda``  is the Tikhonov penalty on the Z-proxy ridge REGRESSION.  An
                  inner numerical-solver knob.  This is the one the original
                  ``sweep_ridge_lambda.py`` re-selected.

They are unrelated and are swept independently here.

Usage
-----
    R=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
    CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 \
    PYTHONPATH="$R/src:$R/experiments" taskset -c 15 \
    /home/tbasseras/gpu-venv/bin/python sweep_hyperparams.py \
        --ridge-lambda 1000 --lambda-scale 200 --steps 250 --device cuda:0

    # afterwards
    python sweep_hyperparams.py --report
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REFERENCE = Path("/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference")
for _path in (HERE, REFERENCE / "src", REFERENCE / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from path_dt_experiments.discrepancies import build_heston_discrepancy  # noqa: E402

import train_multiasset as tm  # noqa: E402
import sweep_ridge_lambda as srl  # noqa: E402
from fit_reference_multiasset import STATE_DIM  # noqa: E402

VERIFY_DIR = HERE / "sweep_verify"
S0 = 100.0

# The d = 1 values, so `--report` can mark the inherited point in the table.
# NOTE ridge_lambda = 1000 is NOT a d = 1 value -- it was already re-selected at
# d = 8 by sweep_ridge_lambda.py. It is the campaign's current setting.
D1 = {
    "ridge_lambda": 1000.0,
    "lambda_scale": 50.0,
    "kappa_scale": 100.0,
    "abs_return_acf_weight": 0.25,
    "squared_return_acf_weight": 0.125,
}


def _num(value: float) -> str:
    """Compact, filesystem-safe number: 1000 -> 1e3, 0.25 -> 0.25."""

    value = float(value)
    if value != 0.0 and (abs(value) >= 1e4 or abs(value) < 1e-2):
        return f"{value:.0e}".replace("e+0", "e").replace("e-0", "e-").replace("e+", "e")
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


def make_tag(args: argparse.Namespace) -> str:
    return (
        f"r{_num(args.ridge_lambda)}"
        f"_L{_num(args.lambda_scale)}"
        f"_K{_num(args.kappa_scale)}"
        f"_a{_num(args.abs_acf)}"
        f"_s{_num(args.sq_acf)}"
    )


def _score_on_common_metric(captured: dict, args: argparse.Namespace) -> dict:
    """Re-score the FITTED model with the frozen d = 1 discrepancy.

    This is the only number that may be compared across candidates: every
    candidate is measured by the same instrument, so a difference is a
    difference in the model rather than in the weighting of the metric.

    The model is re-sampled with ``srl.SCORE_SEED``, the same seed ``run_one``
    used, so the generated bank is bit-identical to the one behind the native
    score and the two numbers differ ONLY in the discrepancy applied.
    """

    model = captured.get("model")
    grid = captured.get("grid")
    if model is None or grid is None:
        raise SystemExit("ABORT: failed to capture the fitted model from build_model")

    device = torch.device(args.device)
    reference = build_heston_discrepancy(
        num_steps=int(grid.num_steps),
        include_acf=True,
        preset="old_fullv_w0p25",
        lambda_scale=D1["lambda_scale"],
        kappa_scale=D1["kappa_scale"],
        include_conditional_volatility=False,
        include_conditional_volatility_correlation=False,
        abs_return_acf_weight=D1["abs_return_acf_weight"],
        squared_return_acf_weight=D1["squared_return_acf_weight"],
    )

    x0 = torch.full((1, STATE_DIM), math.log(S0), device=device, dtype=model.dtype)
    with torch.no_grad():
        generated = model.sample(
            num_paths=int(args.score_paths), x0=x0, seed=srl.SCORE_SEED
        ).paths
    if not torch.isfinite(generated).all():
        return {"common_val_discrepancy": float("nan"),
                "common_valdisc_discrepancy": float("nan")}

    out: dict[str, float] = {}
    for key, filename in (
        ("common_val_discrepancy", srl.VAL_FILE),
        ("common_valdisc_discrepancy", srl.VALDISC_FILE),
    ):
        bank = srl.load_split(
            filename,
            num_paths=int(args.score_paths),
            num_steps=int(grid.num_steps),
            device=device,
            dtype=model.dtype,
        )
        with torch.no_grad():
            scored = reference(
                generated_paths=generated, target_paths=bank, grid=grid
            )
        out[key] = float(scored.value if hasattr(scored, "value") else scored)
        del bank
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(
        f"[common] d1-metric val={out['common_val_discrepancy']:.8g}  "
        f"valdisc={out['common_valdisc_discrepancy']:.8g}",
        flush=True,
    )
    return out


def run(args: argparse.Namespace) -> dict:
    tag = make_tag(args)

    # Rebind the module globals build_model/build_training read at call time.
    tm.LAMBDA_SCALE = float(args.lambda_scale)
    tm.KAPPA_SCALE = float(args.kappa_scale)
    tm.ABS_RETURN_ACF_WEIGHT = float(args.abs_acf)
    tm.SQUARED_RETURN_ACF_WEIGHT = float(args.sq_acf)

    # Redirect run_one's output. `_tag` is looked up as a module global inside
    # run_one, so rebinding it renames the file without touching run_one itself.
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    srl.SWEEP_DIR = VERIFY_DIR
    srl._tag = lambda _value, _tag=tag: _tag  # noqa: SLF001

    # Fail loudly if an override did not land, rather than silently sweeping
    # the d = 1 point four times and reporting a flat, meaningless table.
    probe = tm.build_training(
        seed=srl.SWEEP_SEED, steps=1, ridge_lambda=float(args.ridge_lambda)
    )
    if float(probe.lambda_scale) != float(args.lambda_scale):
        raise SystemExit(
            f"ABORT: lambda_scale override did not reach the training config "
            f"({probe.lambda_scale} != {args.lambda_scale})"
        )
    del probe

    # ------------------------------------------------------------------
    # WHY THE NATIVE SCORE CANNOT BE COMPARED ACROSS CANDIDATES
    # ------------------------------------------------------------------
    # `build_heston_discrepancy` computes `vol_scale = kappa_scale / lambda_scale`
    # and weights a term of the MMD by it, and the ACF weights likewise scale
    # their own terms.  So these four constants are not only training knobs --
    # they define the DISCREPANCY FUNCTION ITSELF.  `run_one` scores each
    # candidate with the discrepancy that candidate was built with, which means
    # a candidate with a smaller `vol_scale` reports a smaller number for the
    # trivial reason that it is measuring less.
    #
    # Measured, on the first four candidates: dividing each native score by its
    # own `vol_scale` collapsed a 20x spread (0.0061 .. 0.1151) to a flat
    # 0.0115 .. 0.0157.  The apparent "70% improvement" was entirely a rescaled
    # ruler; the models were indistinguishable.
    #
    # The fix is the standard one: TRAIN with the candidate weights (that is the
    # thing being tuned), but EVALUATE every candidate with ONE fixed reference
    # discrepancy -- the frozen d = 1 weights.  Only that number is comparable,
    # and only it may be used to rank.  The native score is kept and reported so
    # the trap stays visible rather than being silently dropped.
    captured: dict = {}
    _orig_build_model = tm.build_model

    def _capture(**kwargs):
        model = _orig_build_model(**kwargs)
        captured["model"] = model
        captured["grid"] = kwargs["grid"]
        return model

    tm.build_model = _capture
    print(f"[sweep] {tag}  steps={args.steps}", flush=True)
    try:
        record = srl.run_one(args)
    finally:
        tm.build_model = _orig_build_model

    common = _score_on_common_metric(captured, args)

    # run_one wrote `lambda_{tag}.json`; enrich it with what was varied.
    written = VERIFY_DIR / f"lambda_{tag}.json"
    payload = json.loads(written.read_text(encoding="utf-8"))
    payload.update(
        {
            **common,
            # Renamed on the way out so no downstream reader can mistake the
            # native score for a comparable one. `val_discrepancy` is kept only
            # because sweep_ridge_lambda wrote it; ranking must use
            # `common_val_discrepancy`.
            "native_val_discrepancy": payload.get("val_discrepancy"),
            "native_metric_note": (
                "scored with THIS candidate's own discrepancy weights; "
                "NOT comparable across candidates -- use common_val_discrepancy"
            ),
            "vol_scale": float(args.kappa_scale) / float(args.lambda_scale),
            "tag": tag,
            "lambda_scale": float(args.lambda_scale),
            "kappa_scale": float(args.kappa_scale),
            "abs_return_acf_weight": float(args.abs_acf),
            "squared_return_acf_weight": float(args.sq_acf),
            "is_d1_point": (
                float(args.lambda_scale) == D1["lambda_scale"]
                and float(args.kappa_scale) == D1["kappa_scale"]
                and float(args.abs_acf) == D1["abs_return_acf_weight"]
                and float(args.sq_acf) == D1["squared_return_acf_weight"]
            ),
        }
    )
    written.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[out] {written}", flush=True)
    return payload


def report() -> None:
    files = sorted(VERIFY_DIR.glob("lambda_*.json"))
    if not files:
        raise SystemExit(f"ABORT: no results in {VERIFY_DIR}")
    rows = [json.loads(f.read_text(encoding="utf-8")) for f in files]

    def _common(rec: dict) -> float:
        """The only comparable score. NaN if this record predates the fix."""

        value = rec.get("common_val_discrepancy")
        return float(value) if value is not None else float("nan")

    ok = [r for r in rows if not r.get("diverged") and _common(r) == _common(r)]
    stale = [r for r in rows if not r.get("diverged") and _common(r) != _common(r)]

    print("ranking on common_val_discrepancy: every candidate scored with the")
    print("FROZEN d = 1 discrepancy (L=50 K=100 a=0.25 s=0.125), so the number")
    print("measures the MODEL. 'native' is each candidate's own metric and is")
    print("NOT comparable -- it scales with vol_scale = kappa_scale/lambda_scale.\n")

    header = (
        f"{'tag':<30} {'common_val':>12} {'native':>12} {'vol_sc':>7} "
        f"{'ce_r2':>8}  note"
    )
    print(header)
    print("-" * len(header))
    best = min(ok, key=_common) if ok else None
    for r in sorted(rows, key=lambda r: _common(r) if _common(r) == _common(r)
                    else float("inf")):
        if r.get("diverged"):
            print(f"{r.get('tag', '?'):<30} {'DIVERGED':>12}")
            continue
        note = []
        if r.get("is_d1_point"):
            note.append("d=1 point")
        if best is not None and r is best:
            note.append("<-- BEST")
        if _common(r) != _common(r):
            note.append("STALE: pre-fix, native only")
        native = r.get("native_val_discrepancy", r.get("val_discrepancy"))
        cv = _common(r)
        cv_s = f"{cv:>12.6g}" if cv == cv else f"{'--':>12}"
        print(
            f"{r.get('tag', '?'):<30} {cv_s} "
            f"{(native if native is not None else float('nan')):>12.6g} "
            f"{r.get('vol_scale', float('nan')):>7.2f} "
            f"{r.get('ce_projection_r2', float('nan')):>8.3f}  {' '.join(note)}"
        )

    if stale:
        print(f"\n[warn] {len(stale)} record(s) carry only the native score and "
              f"cannot be ranked. Re-run them to get a comparable number.")

    if best is not None:
        d1 = [r for r in ok if r.get("is_d1_point")]
        if d1:
            ref = min(d1, key=_common)
            delta = 100.0 * (_common(ref) - _common(best)) / _common(ref)
            print(
                f"\nbest is {delta:.1f}% better than the d = 1 point "
                f"({best.get('tag')} vs {ref.get('tag')})"
            )
            if abs(delta) < 40.0:
                print(
                    "NOTE: checkpoint-to-checkpoint noise on this metric was "
                    "measured at ~38% (see scan_step_curve.py), so a gap this "
                    "size is NOT evidence of a better setting."
                )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ridge-lambda", type=float, default=D1["ridge_lambda"])
    p.add_argument("--lambda-scale", type=float, default=D1["lambda_scale"])
    p.add_argument("--kappa-scale", type=float, default=D1["kappa_scale"])
    p.add_argument("--abs-acf", type=float, default=D1["abs_return_acf_weight"])
    p.add_argument("--sq-acf", type=float, default=D1["squared_return_acf_weight"])
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num-paths", type=int, default=None)
    p.add_argument("--num-steps", type=int, default=None)
    p.add_argument("--score-paths", type=int, default=8192)
    p.add_argument("--report", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.report:
        report()
        return
    run(args)


if __name__ == "__main__":
    main()
