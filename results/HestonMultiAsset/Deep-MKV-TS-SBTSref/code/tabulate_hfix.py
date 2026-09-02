"""Tabulate the corrected-adjoint bandwidth re-sweep and name its winner.

The re-sweep runs three seeds per bandwidth because the measured seed noise
floor at this learning rate is 6.91% of the mean: a single-seed gap smaller
than that is not evidence about ``h``.  So this script never reports a bare
argmin.  It reports, per bandwidth, the three seed scores, their mean, and
their spread, and then flags whether the winner's margin over the runner-up
actually clears the floor.  If it does not, the honest statement is "these two
bandwidths are indistinguishable at three seeds", and the tie is broken by the
occupancy measurement instead of by noise.

Selection metric is ``val_discrepancy`` on the validation split, which is the
same quantity the first sweep used -- hyperparameters are chosen on validation,
never on test.  ``valdisc_discrepancy`` is carried alongside as a second,
independent check file; it is reported but never selected on.

A DIVERGED arm is a result, not a missing point.  It is printed as DIVERGED and
excluded from the mean, and a bandwidth with any diverged seed can never win --
promoting a configuration that blows up on 1 seed in 3 would be indefensible.

Run:
    python tabulate_hfix.py                # report only
    python tabulate_hfix.py --promote      # also write sweep/incumbent.json
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SWEEP_DIR = HERE / "sweep"
STAGE = "hfix"

# Measured on 3 seeds at lr = 2.5e-05: scores 0.063282 / 0.059091 / 0.059553,
# mean 0.060642.  The floor is the RANGE over the mean, (max - min) / mean =
# 6.91%, NOT the standard deviation over the mean (which is 3.8%).  The two
# differ by nearly a factor of two, so the `spread%` column below reports the
# range as well -- a table that printed sd next to a range-derived threshold
# would make every bandwidth look twice as reproducible as it is.
NOISE_FLOOR_PCT = 6.91

# Occupancy is READ FROM DISK, never transcribed.  It used to be a literal dict
# whose own comment conceded the numbers were "transcribed from
# probe_occupancy.txt", and one of them was wrong by 32% (12.66 typed for a true
# 16.7) for long enough to reach a draft table.  `probe_occupancy_low.py` now
# writes `sweep/occupancy_k20.json` and reproduces every previously published
# value to the printed precision, so no human needs to retype any of them.
OCCUPANCY = SWEEP_DIR / "occupancy_k20.json"

# A bandwidth whose kernel is not actually computing a conditional expectation
# cannot be promoted on the strength of a good score, because the score is then
# measuring something other than the model.  Two ways that happens at K = 20:
#
#   * DEAD ROWS.  Every weight underflows, `b^ref` is identically 0, and the
#     arm trains against a reference drift that does not exist.
#   * COLLAPSE.  The weights concentrate on one bank path, so the "conditional
#     expectation" is a nearest-neighbour lookup of a single TRAINING path.
#     README section 3 already declares this the method's structural
#     memorisation channel, and a validation discrepancy cannot see it: a model
#     that reproduces training paths scores well on any distributional metric.
#
# These thresholds veto nothing on their own.  They decide only whether
# `--promote` may act unattended or must stop and make a human look.
MIN_ALIVE_PCT = 98.0
MIN_N_EFF = 10.0


def load_occupancy() -> dict[float, dict]:
    """Occupancy keyed by bandwidth, or an empty dict if never measured.

    Missing is not fatal: the table degrades to "?" in the occupancy columns
    and `--promote` refuses to run unattended, which is the correct response to
    "I cannot tell whether this kernel is alive".
    """

    if not OCCUPANCY.exists():
        return {}
    raw = json.loads(OCCUPANCY.read_text(encoding="utf-8"))
    return {float(k): v for k, v in raw.get("by_h", {}).items()}


def load_arms() -> list[dict]:
    files = sorted(SWEEP_DIR.glob(f"{STAGE}__*.json"))
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def summarise(arms: list[dict]) -> tuple[list[dict], list[str]]:
    """Group by bandwidth; return per-h rows and any blocking problems."""

    by_h: dict[float, list[dict]] = {}
    for rec in arms:
        by_h.setdefault(round(float(rec["h"]), 6), []).append(rec)

    rows, problems = [], []
    for h in sorted(by_h):
        recs = sorted(by_h[h], key=lambda r: int(r.get("seed", 0)))
        scores, seeds, diverged = [], [], []
        for r in recs:
            s = int(r.get("seed", 0))
            seeds.append(s)
            if r.get("diverged"):
                diverged.append(s)
                scores.append(float("nan"))
                continue
            v = float(r["val_discrepancy"])
            scores.append(v if v == v else float("nan"))
        arr = np.asarray(scores, dtype=float)
        good = arr[np.isfinite(arr)]
        rows.append({
            "h": h,
            "seeds": seeds,
            "scores": arr,
            "diverged": diverged,
            "n": int(good.size),
            "mean": float(good.mean()) if good.size else float("nan"),
            "sd": float(good.std(ddof=1)) if good.size > 1 else float("nan"),
            "check": [float(r.get("valdisc_discrepancy", float("nan"))) for r in recs],
        })
        if len(recs) != 3:
            problems.append(f"h={h}: {len(recs)} seeds present, expected 3")
    return rows, problems


def report(rows: list[dict]) -> dict | None:
    occ = load_occupancy()
    print("=" * 100)
    print("BANDWIDTH RE-SWEEP, CORRECTED ADJOINT (jacobian_lags = -1, all K = 20 lags)")
    print("=" * 100)
    print(f"{'h':>6} {'alive%':>7} {'n_eff':>9} {'seed0':>11} {'seed1':>11} {'seed2':>11}"
          f" {'mean':>11} {'spread%':>9}")
    print("-" * 100)

    for r in rows:
        cells = []
        for i in range(3):
            if i >= len(r["scores"]):
                cells.append(f"{'--':>11}")
            elif r["seeds"][i] in r["diverged"]:
                cells.append(f"{'DIVERGED':>11}")
            elif np.isfinite(r["scores"][i]):
                cells.append(f"{r['scores'][i]:>11.6f}")
            else:
                cells.append(f"{'nan':>11}")
        # Range over mean, the SAME definition as NOISE_FLOOR_PCT, so the
        # column and the threshold underneath it can be read against each other.
        good = r["scores"][np.isfinite(r["scores"])]
        spread = (
            100.0 * (good.max() - good.min()) / r["mean"]
            if good.size > 1 and r["mean"] else float("nan")
        )
        o = occ.get(r["h"])
        r["occ"] = o
        ne = o.get("n_eff_median") if o else None
        al = o.get("alive_pct") if o else None
        ne_s = f"{ne:>9.1f}" if ne is not None else f"{'?':>9}"
        al_s = f"{al:>6.1f}%" if al is not None else f"{'?':>7}"
        print(f"{r['h']:>6.2f} {al_s} {ne_s} " + " ".join(cells)
              + f" {r['mean']:>11.6f} {spread:>8.1f}%")

    print("-" * 100)
    print(f"seed noise floor = {NOISE_FLOOR_PCT:.2f}%  (a gap below this is not an h effect)")
    print("alive% / n_eff at step 120, K=20, disjoint bank/query, from "
          f"{OCCUPANCY.name}")
    print("alive% = rows whose kernel weights did not all underflow; the rest "
          "train against b^ref = 0.")
    print("n_eff  = median effective bank paths; near 1 means a 1-NN lookup of a "
          "single TRAINING path.")
    if not occ:
        print("[warn]   no occupancy file; run `python probe_occupancy_low.py` "
              "before promoting anything.")
    print()

    eligible = [r for r in rows if not r["diverged"] and r["n"] == 3
                and np.isfinite(r["mean"])]
    if not eligible:
        print("ABORT: no bandwidth has three finite seeds; nothing can be promoted.")
        return None

    ranked = sorted(eligible, key=lambda r: r["mean"])
    best, second = ranked[0], (ranked[1] if len(ranked) > 1 else None)
    print(f"[best]   h = {best['h']:.2f}   mean val_discrepancy = {best['mean']:.6f}")
    if second is not None:
        margin = 100.0 * (second["mean"] - best["mean"]) / best["mean"]
        print(f"[second] h = {second['h']:.2f}   mean = {second['mean']:.6f}"
              f"   margin = {margin:.2f}%")
        if margin < NOISE_FLOOR_PCT:
            print(f"[warn]   margin {margin:.2f}% is INSIDE the {NOISE_FLOOR_PCT:.2f}% "
                  "noise floor: these two bandwidths are not distinguishable at "
                  "three seeds.  Reporting the lower mean, but the campaign result "
                  "must not be attributed to this choice of h.")
        else:
            print(f"[ok]     margin {margin:.2f}% clears the noise floor.")
    excluded = [r["h"] for r in rows if r["diverged"]]
    if excluded:
        print(f"[note]   bandwidths excluded for divergence: {excluded}")

    # ---------------------------------------------------------------------
    # Is the winner an optimum, or just the edge of the grid?
    # ---------------------------------------------------------------------
    # A minimum attained at the smallest bandwidth tested is not evidence that
    # the objective turns around there; it is evidence that the search stopped
    # there.  This exact situation arose on the first re-sweep -- h = 0.36 won
    # by 87% and was the lowest point on the grid -- and promoting it would
    # have committed a four-hour five-seed campaign to an unexamined boundary.
    # MULTIASSET_GUIDELINE 12.3 names this failure explicitly.
    blockers: list[str] = []
    tested = sorted(r["h"] for r in eligible)
    if len(tested) > 1 and best["h"] == tested[0]:
        blockers.append(
            f"h = {best['h']:.2f} is the SMALLEST bandwidth tested "
            f"(grid starts at {tested[0]:.2f}); a boundary minimum is not an "
            "optimum until a smaller h has been shown to be worse")

    # ---------------------------------------------------------------------
    # Is the winner's kernel still a conditional expectation?
    # ---------------------------------------------------------------------
    o = best.get("occ")
    if o is None:
        blockers.append(
            f"no occupancy measurement for h = {best['h']:.2f}; run "
            "`python probe_occupancy_low.py` -- an unmeasured kernel may be "
            "returning b^ref = 0 and scoring well for the wrong reason")
    else:
        if o["alive_pct"] < MIN_ALIVE_PCT:
            dead = o["total"] - o["alive"]
            blockers.append(
                f"only {o['alive_pct']:.1f}% of query rows are alive at "
                f"h = {best['h']:.2f} ({dead}/{o['total']} train against "
                f"b^ref = 0), below the {MIN_ALIVE_PCT:.0f}% bar")
        if o["n_eff_median"] is not None and o["n_eff_median"] < MIN_N_EFF:
            blockers.append(
                f"median n_eff = {o['n_eff_median']:.2f} at h = {best['h']:.2f} "
                f"is below {MIN_N_EFF:.0f}: the reference drift is close to a "
                "nearest-neighbour lookup of one training path, which is the "
                "memorisation channel of README section 3, and the validation "
                "discrepancy cannot distinguish it from a good fit")

    if blockers:
        print()
        print("!" * 100)
        print("WINNER IS NOT PROMOTABLE UNATTENDED")
        for b in blockers:
            print(f"  - {b}")
        print("!" * 100)
        print("Re-run with --allow-degenerate once a human has read the above "
              "and decided anyway.")
    best["blockers"] = blockers
    return best


def promote(h: float) -> None:
    path = SWEEP_DIR / "incumbent.json"
    values = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    values["h"] = float(h)
    values["jacobian_lags"] = -1
    values["promoted_from"] = f"winner_{STAGE}.json"
    values["promoted_on"] = date.today().isoformat()
    values["h_note"] = (
        "Re-swept against the CORRECTED adjoint (jacobian_lags=-1, all K=20 lags). "
        "The earlier h=0.36 was selected while the adjoint carried a single lag, a "
        "gradient with 100% relative error, so that choice was not evidence about "
        "the code that now runs."
    )
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[promote] incumbent h = {h:.2f} -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", action="store_true",
                    help="write the winning h into sweep/incumbent.json")
    ap.add_argument("--allow-degenerate", action="store_true",
                    help="promote even if the winner sits on the grid boundary "
                         "or its kernel has collapsed; for a human who has read "
                         "the blockers and decided anyway")
    args = ap.parse_args()

    arms = load_arms()
    if not arms:
        raise SystemExit(f"ABORT: no {STAGE} arms in {SWEEP_DIR}")
    rows, problems = summarise(arms)
    for p in problems:
        print(f"[incomplete] {p}")
    best = report(rows)
    if best is None:
        raise SystemExit(1)
    # A blocked winner must not reach the supervisor.  run_auto_campaign.sh
    # greps WINNER_H and aborts on a non-zero exit, so withholding the line AND
    # failing is what stops a four-hour campaign from starting on a bandwidth
    # that has not been shown to be an optimum.
    if best["blockers"] and not args.allow_degenerate:
        print("WINNER_H_BLOCKED")
        if args.promote:
            raise SystemExit(1)
        raise SystemExit(0)

    if args.promote:
        promote(best["h"])
    # The supervisor greps this line to learn which bandwidth to run.
    print(f"WINNER_H={best['h']:.2f}")


if __name__ == "__main__":
    main()
