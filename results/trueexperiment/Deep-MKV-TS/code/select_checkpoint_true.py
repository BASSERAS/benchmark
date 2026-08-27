#!/usr/bin/env python3
"""Pick the reported checkpoint per seed under the section-7 rule (TrueDataset).

Why this file is NOT a copy of ``select_checkpoint_multiasset.py``
------------------------------------------------------------------
At Heston the rule is ``argmin`` of the validation discrepancy -- the same
discrepancy that trained the model.  That is legitimate there because the
Heston splits are independent draws from a known law, so a held-out
discrepancy really is out-of-sample.

Here it is forbidden.  ``truedatasetguideline.md`` section 7 is explicit:
selection on goodness of fit picks a memoriser.  The section 7.1 measurement on
this exact dataset is the proof -- the fit-best SBTS setting scored a vol error
of 6.06%, BELOW the 8.72% floor that two disjoint slices of the real data score
against each other, with a nearest-neighbour ratio of 0.197.  It had not
modelled the panel, it had copied it.

The mechanism is specific to a single-realisation dataset.  ``train``, ``val``
and ``valdisc`` are three windows of ONE crypto tape from the same era, so as
training drives the model toward ``train`` the val discrepancy keeps falling
long past the point where the model has stopped generalising and started
reproducing.  ``argmin`` over the checkpoint grid therefore reliably returns the
LAST checkpoint, which is the memoriser.  A criterion that cannot say "later is
worse" cannot perform selection.

The rule this file applies instead
----------------------------------
Two parts, admissibility then objective, both from section 7:

  admissible  <=>  vol_err <= ceiling  AND  corr_err <= ceiling
  among admissible, minimise |log NNratio|
  ties within 0.02 broken on the lower corr_err

Every ceiling is RECOMPUTED from the dataset by ``selection_true.envelope``, and
none is typed into this file.  Kurtosis is measured and reported but never
gates, per section 7.  The NNratio denominator is the ``val`` split (section
9.1), not ``test`` -- the test split is not opened by this file at all.

Cost
----
The section-7 statistics were measured at 0.8 s per candidate against the 6144
path train split, at both 512 and 2048 probe paths.  With six checkpoints per
seed and five seeds the criterion contributes about half a minute in total; the
sampling dominates.  So the fast criterion and the correct criterion are the
same criterion and no compromise is needed.

Why the full checkpoint payload and not ``network.state_dict()``
----------------------------------------------------------------
``DiscreteMPModel.checkpoint_state()`` carries the timewise target scales and
the noise-target baseline in addition to the network weights.  Those feed the
adjoint moments handed to the control map, so a bare ``state_dict`` does not
reproduce a fitted model's rollouts.  The trainer writes a provisional
final-step payload into ``weights/seed_{i}_model.pt``; this script replaces it
with the SELECTED checkpoint's payload, byte-for-byte as saved.

Usage
-----
    R=/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference
    CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 \\
    PYTHONPATH="$R/src:$R/experiments" taskset -c 0-7 \\
    /home/tbasseras/gpu-venv/bin/python select_checkpoint_true.py --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import math
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

from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid  # noqa: E402

import eigh_fallback  # noqa: E402
import selection_true as sel  # noqa: E402
import train_true as tt  # noqa: E402
from fit_reference_true import DATASET, DT, SEQ_TAG, STATE_DIM  # noqa: E402
from sweep_ridge_lambda_true import VAL_FILE, VALDISC_FILE, load_split  # noqa: E402

SELECTION_DIR = HERE / "selection"
SCORE_SEED = 20260826
S0 = 100.0


def _discrepancy(
    model,
    generated: torch.Tensor,
    *,
    grid: DiscreteTimeGrid,
    device: torch.device,
    score_paths: int,
    filename: str,
) -> float:
    """The Heston criterion, computed so it can be REPORTED and disagreed with.

    Kept because a silent divergence between the two rules is worth nothing,
    while a recorded one tells the reader exactly how much the section-7 rule
    changed the answer. It never enters ``sel.select``.
    """

    bank = load_split(
        filename,
        num_paths=int(score_paths),
        num_steps=int(grid.num_steps),
        device=device,
        dtype=model.dtype,
    )
    with torch.no_grad():
        scored = model.discrepancy(
            generated_paths=generated, target_paths=bank, grid=grid
        )
    value = float(scored.value if hasattr(scored, "value") else scored)
    del bank
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return value


def select_one(seed: int, args: argparse.Namespace, context: dict) -> dict:
    device = torch.device(args.device)
    run_dir = Path(args.run_root) / f"seed_{int(seed)}"
    ckpt_dir = run_dir / "training_checkpoints"
    checkpoints = sorted(ckpt_dir.glob("step_*.pt"))
    if not checkpoints:
        raise SystemExit(f"ABORT: no checkpoints in {ckpt_dir}")

    # The grid must match training exactly, so read its width off the train split
    # rather than assuming 127. A one-step mismatch silently rescales every
    # annualised statistic downstream.
    probe = tt.load_log_prices(num_paths=1, device="cpu")
    num_steps = int(probe.shape[1]) - 1
    del probe
    grid = DiscreteTimeGrid(T=num_steps * DT, num_steps=num_steps)

    model, sigma_max = tt.build_model(
        grid=grid, device=device, seed=int(seed), kernel_steps=num_steps
    )

    module = context["module"]
    env = context["env"]
    train_prices = context["train"]
    val_prices = context["val"]

    x0 = torch.full((1, STATE_DIM), math.log(S0), device=device, dtype=model.dtype)

    per_step: list[dict] = []
    for path in checkpoints:
        step = int(path.stem.split("_")[1])
        started = time.perf_counter()
        # weights_only=True: checkpoint_state() emits only tensors, plain dicts
        # and primitives, so there is no reason to allow arbitrary unpickling.
        model.load_checkpoint_state(
            torch.load(path, map_location=device, weights_only=True)
        )
        with torch.no_grad():
            generated = model.sample(
                num_paths=int(args.score_paths), x0=x0, seed=SCORE_SEED
            ).paths

        if not torch.isfinite(generated).all():
            # A diverged checkpoint is a RESULT. Recording it as non-finite keeps
            # it visible in the table; dropping it here would let a seed that
            # blew up at step 2000 look like a seed that was never scored there.
            bad = int((~torch.isfinite(generated)).sum())
            # The numeric keys are filled with +inf rather than omitted because
            # `selection_true.admissibility` reads them unconditionally. +inf is
            # not padding: it is the truthful statement that a diverged bank is
            # infinitely far outside every ceiling, and it makes the row fail
            # `math.isfinite(objective(...))` so it can never be selected.
            record = {
                "step": step,
                "label": f"step={step}",
                "finite": False,
                "non_finite_entries": bad,
                "min_price": 0.0,
                "vol_err_pct": float("inf"),
                "vol_ratio": float("inf"),
                "kurt_err_pct": float("inf"),
                "corr_err": float("inf"),
                "nn_ratio": float("inf"),
                "abs_log_nn_ratio": float("inf"),
                "m_probe": int(args.score_paths),
            }
            per_step.append(record)
            print(
                f"  seed={seed} step={step:5d} NON-FINITE ({bad} entries) "
                "-- excluded from selection",
                flush=True,
            )
            continue

        # `generated` is LOG prices; every section-7 statistic is defined on
        # PRICES, so the exponential happens exactly once, here.
        gen_prices = np.exp(generated.detach().to(torch.float64).cpu().numpy())
        record = sel.score_candidate(
            gen_prices,
            train=train_prices,
            val=val_prices,
            module=module,
            median_nn_heldout=env["median_nn_heldout"],
        )
        record["step"] = step
        record["label"] = f"step={step}"
        record["val_discrepancy"] = _discrepancy(
            model,
            generated,
            grid=grid,
            device=device,
            score_paths=int(args.score_paths),
            filename=VAL_FILE,
        )
        record["valdisc_discrepancy"] = _discrepancy(
            model,
            generated,
            grid=grid,
            device=device,
            score_paths=int(args.score_paths),
            filename=VALDISC_FILE,
        )
        per_step.append(record)

        reasons = sel.admissibility(record, env)
        print(
            f"  seed={seed} step={step:5d} "
            f"vol={record['vol_err_pct']:.2f}% corr={record['corr_err']:.4f} "
            f"NN={record['nn_ratio']:.4f} |logNN|={record['abs_log_nn_ratio']:.4f} "
            f"fit={record['val_discrepancy']:.6g} "
            f"{'ADMISSIBLE' if not reasons else 'REJECTED: ' + '; '.join(reasons)} "
            f"({time.perf_counter() - started:.1f}s)",
            flush=True,
        )
        del generated
        if device.type == "cuda":
            torch.cuda.empty_cache()

    best, annotated = sel.select(per_step, env)
    print(sel.format_table(annotated, env, label_key="label"), flush=True)

    if best is None:
        raise SystemExit(
            f"ABORT: seed {seed} produced no admissible checkpoint. Every saved "
            "step is outside the real-vs-real envelope this dataset measures for "
            "itself. Widening the envelope to admit one would replace a measured "
            "ceiling with a chosen one, so the honest outcomes are: retrain this "
            "seed, or report it as a failure."
        )

    selected_step = int(best["step"])
    print(
        f"  seed={seed} SELECTED step={selected_step} "
        f"|logNN|={best['abs_log_nn_ratio']:.4f} NN={best['nn_ratio']:.4f} "
        f"vol={best['vol_err_pct']:.2f}% corr={best['corr_err']:.4f}",
        flush=True,
    )

    # What the Heston rule would have said, recorded so the disagreement is
    # auditable instead of invisible.
    fitted = [r for r in per_step if r.get("finite", True) and "val_discrepancy" in r]
    fit_choice = (
        int(min(fitted, key=lambda r: r["val_discrepancy"])["step"]) if fitted else None
    )
    if fit_choice is not None and fit_choice != selected_step:
        print(
            f"  [note] the fit rule (argmin val discrepancy) would have chosen "
            f"step {fit_choice}. Section 7 forbids it; see this file's docstring.",
            flush=True,
        )

    # Promote the selected checkpoint into the reported weights slot.
    # out_root defaults to METHOD_ROOT so the published campaign is unchanged; a
    # retune passes --out-root and therefore cannot overwrite the committed
    # seed_<S>_model.pt even by accident.
    out_root = Path(getattr(args, "out_root", None) or METHOD_ROOT)
    selected_path = ckpt_dir / f"step_{selected_step:04d}.pt"
    payload = torch.load(selected_path, map_location="cpu", weights_only=True)
    weights_path = out_root / "weights" / f"seed_{int(seed)}_model.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, weights_path)

    # Record the choice inside the config the README reads.
    config_path = out_root / "weights" / f"seed_{int(seed)}_config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["selected_step"] = selected_step
        config["selection_rule"] = (
            "truedatasetguideline section 7: admissible on the measured "
            "real-vs-real vol and corr ceilings, then argmin |log NNratio|"
        )
        config["selection_file"] = VAL_FILE
        config["selection_nn_ratio"] = float(best["nn_ratio"])
        config["selection_abs_log_nn_ratio"] = float(best["abs_log_nn_ratio"])
        config["selection_vol_err_pct"] = float(best["vol_err_pct"])
        config["selection_corr_err"] = float(best["corr_err"])
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    record = {
        "seed": int(seed),
        "selected_step": selected_step,
        "selection_rule": "section-7 admissibility then argmin |log NNratio|",
        "selected": {k: v for k, v in best.items() if k != "label"},
        "fit_rule_would_have_chosen": fit_choice,
        "per_step": per_step,
        "envelope": env,
        "sigma_max": float(sigma_max),
        "selection_file": VAL_FILE,
        "secondary_file": VALDISC_FILE,
        "score_paths": int(args.score_paths),
        "score_seed": SCORE_SEED,
        "eigh_fallback": eigh_fallback.counters(),
        "date": date.today().isoformat(),
    }
    # The per-step audit trail follows the weights it describes, so a retune's
    # selection cannot be confused with the published one.
    selection_dir = (
        SELECTION_DIR if out_root == METHOD_ROOT else out_root / "selection"
    )
    selection_dir.mkdir(parents=True, exist_ok=True)
    out = selection_dir / f"seed_{int(seed)}_selection.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"  [out] {out}", flush=True)
    return record


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", type=str, default="cuda:0")
    # 2048, not 512: section 7.2 measured a 1.67pp vol-error spread across probe
    # draws at 512, which is large enough to reorder adjacent checkpoints.
    parser.add_argument("--score-paths", type=int, default=2048)
    parser.add_argument("--run-root", type=Path, default=HERE / "runs")
    parser.add_argument(
        "--out-root", type=Path, default=None,
        help="directory to hold weights/ and selection/ (default: the published "
             "Deep-MKV-TS method root). A retune MUST pass this, otherwise it "
             "overwrites the committed seed_<S>_model.pt.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Loaded ONCE and shared across seeds. The NNratio denominator must be the
    # same number for every seed, otherwise the per-seed objectives are measured
    # against different rulers and are not comparable -- which would quietly
    # break the cross-seed table this feeds.
    module = sel.load_criterion(DATASET)
    env = sel.envelope(data_dir=DATASET, seq_tag=SEQ_TAG)
    train_prices, val_prices = sel.load_splits(data_dir=DATASET, seq_tag=SEQ_TAG)
    print(sel.report_envelope(env), flush=True)
    context = {
        "module": module,
        "env": env,
        "train": train_prices,
        "val": val_prices,
    }

    records = [select_one(int(seed), args, context) for seed in args.seeds]

    print("\n seed  selected_step   NNratio   |logNN|    vol%     corr    fit-rule")
    print("-" * 74)
    for rec in records:
        sel_rec = rec["selected"]
        fit_choice = rec["fit_rule_would_have_chosen"]
        print(
            f"{rec['seed']:>5}  {rec['selected_step']:>13}   "
            f"{sel_rec['nn_ratio']:.4f}   {sel_rec['abs_log_nn_ratio']:.4f}  "
            f"{sel_rec['vol_err_pct']:>6.2f}  {sel_rec['corr_err']:.4f}    "
            f"{fit_choice if fit_choice is not None else '-'}"
        )
    steps = sorted({rec["selected_step"] for rec in records})
    print(f"\nselected steps across seeds: {steps}")


if __name__ == "__main__":
    main()
