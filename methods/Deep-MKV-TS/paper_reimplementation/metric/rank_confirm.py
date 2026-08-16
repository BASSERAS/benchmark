#!/usr/bin/env python3
"""Re-rank the top hyperparameter trials on the paper's 4-seed median.

rank_hpsearch.py ranks on seed 0 alone, which cannot separate a real improvement
from seed luck. run_confirm.sh fills in seeds 1, 3 and 4 for the shortlisted
tags; this script combines them with the seed-0 run the search already produced
and ranks on the median over all four -- the same statistic the paper reports.

Every run scored here used BENCHMARK_HESTON_EVAL=heston_S_valdisc_8192x128.npy,
so this is still validation-only selection. It also prints the per-tag seed
spread, so a margin can be read against the noise it has to clear.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

from aggregate_paper_table import COLUMNS, md_cell, paper_metrics
from rank_hpsearch import ARM, TARGET, score

PR_ROOT = Path(__file__).resolve().parent.parent
SEARCH_ROOT = PR_ROOT / "runs" / "hpsearch"
CONFIRM_ROOT = PR_ROOT / "runs" / "hpsearch_confirm"
SEEDS = (0, 1, 3, 4)
# Seed 0 is the seed the search shortlisted on, so its value for a shortlisted
# tag is biased upward by the selection itself -- the tag is here partly because
# seed 0 happened to land well. Seeds 1, 3 and 4 were never used to select, so
# their median is an unbiased estimate and is what decides promotion. The
# 4-seed median is still reported, because that is the statistic the paper uses.
FRESH_SEEDS = (1, 3, 4)


def seed_metrics_path(tag: str, seed: int) -> Path:
    """Seed 0 lives in the search tree; the confirm seeds live beside it."""
    if seed == 0:
        return SEARCH_ROOT / tag / ARM / "metrics.json"
    return CONFIRM_ROOT / tag / f"seed_{seed}" / ARM / "metrics.json"


def load_tag(tag: str) -> dict[int, dict[str, float]]:
    per_seed: dict[int, dict[str, float]] = {}
    for seed in SEEDS:
        path = seed_metrics_path(tag, seed)
        if path.is_file():
            per_seed[seed] = paper_metrics(json.loads(path.read_text()))
    return per_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tags", nargs="+")
    args = parser.parse_args()

    loaded: dict[str, dict[int, dict[str, float]]] = {}
    for tag in args.tags:
        per_seed = load_tag(tag)
        if not per_seed:
            print(f"WARNING: no metrics found for {tag}, skipping")
            continue
        loaded[tag] = per_seed
    if not loaded:
        raise SystemExit("no metrics found for any requested tag")

    summary: dict[str, dict] = {}
    for tag, per_seed in loaded.items():
        seeds = sorted(per_seed)
        fresh = [s for s in seeds if s in FRESH_SEEDS]
        med = {key: median(per_seed[s][key] for s in seeds) for _, key in COLUMNS}
        # Per-seed score spread: how much of any margin is just seed noise.
        per_seed_scores = [score(per_seed[s]) for s in seeds]
        entry = {
            "seeds": seeds,
            "fresh_seeds": fresh,
            "median": med,
            "score": score(med),
            "score_min": min(per_seed_scores),
            "score_max": max(per_seed_scores),
            "per_seed": {str(s): per_seed[s] for s in seeds},
            "per_seed_score": {str(s): sc for s, sc in zip(seeds, per_seed_scores)},
        }
        if fresh:
            med_fresh = {
                key: median(per_seed[s][key] for s in fresh) for _, key in COLUMNS
            }
            entry["median_fresh"] = med_fresh
            entry["score_fresh"] = score(med_fresh)
        else:
            # Only the selection seed is present: no unbiased estimate exists yet.
            # None rather than inf so the JSON stays strictly valid.
            entry["median_fresh"] = None
            entry["score_fresh"] = None
        summary[tag] = entry

    # Rank on the fresh seeds, never on the seed the shortlist was drawn from.
    # Tags with no fresh seed yet sort last rather than winning by default.
    ranked = sorted(
        summary.items(),
        key=lambda kv: (kv[1]["score_fresh"] is None, kv[1]["score_fresh"] or 0.0),
    )

    lines = [
        "# Deep-MKV-TS shortlist confirmed on 4 seeds (scored on heston_S_valdisc)",
        "",
        "Median over seeds 0, 1, 3, 4 -- the paper's seed set and statistic. Score is",
        "the mean over the five paper columns of ours/paper; below 1.000 beats the",
        "published Deep-MKV-TS row.",
        "",
        "Ranking uses **Score(fresh)**, the median over seeds 1/3/4 only. Seed 0 chose",
        "this shortlist, so a shortlisted tag's seed-0 value is biased downward by the",
        "selection itself; seeds 1/3/4 were never used to select and are unbiased.",
        "`Seed range` is the min-max of the per-seed score -- a margin smaller than",
        "that range is seed noise, not an improvement.",
        "",
        "| | " + " | ".join(md_cell(n) for n, _ in COLUMNS) + " |",
        "|---|" + "|".join(["---:"] * len(COLUMNS)) + "|",
        "| Paper | " + " | ".join(f"{TARGET[k]:.3f}" for _, k in COLUMNS) + " |",
        "",
        "| Rank | Trial | Seeds | "
        + " | ".join(md_cell(n) for n, _ in COLUMNS)
        + " | Score(4-seed) | Score(fresh) | Seed range |",
        "|---|---|---|" + "|".join(["---:"] * (len(COLUMNS) + 3)) + "|",
    ]
    for rank, (tag, info) in enumerate(ranked, start=1):
        cells = " | ".join(f"{info['median'][key]:.4f}" for _, key in COLUMNS)
        n_seeds = len(info["seeds"])
        n_fresh = len(info["fresh_seeds"])
        fresh_cell = (
            f"{info['score_fresh']:.3f} ({n_fresh}/3)" if n_fresh else "n/a (0/3)"
        )
        lines.append(
            f"| {rank} | {tag} | {n_seeds}/4 | {cells} | {info['score']:.3f} | "
            f"{fresh_cell} | {info['score_min']:.3f}-{info['score_max']:.3f} |"
        )

    best_tag, best = ranked[0]
    beats = best["score_fresh"] is not None and best["score_fresh"] < 1.0
    lines += [
        "",
        f"**Winner: `{best_tag}` (fresh-seed score "
        f"{'n/a' if best['score_fresh'] is None else format(best['score_fresh'], '.3f')}, "
        f"4-seed {best['score']:.3f}; "
        f"{'beats' if beats else 'does NOT beat'} the paper on the unbiased seeds).**",
        "",
    ]
    if beats:
        lines.append(
            "Selected on validation only. Score it on `disc` exactly once to produce "
            "the paper-vs-ours table; do not re-enter the search after touching `disc`."
        )
    else:
        lines.append(
            "No shortlisted config beats the paper on the 4-seed median. Extend the "
            "grid along the columns still above 1.0 and re-run; `disc` stays untouched."
        )

    lines += ["", "## Per-seed scores", ""]
    for tag, info in ranked:
        cells = ", ".join(
            f"seed {s}: {info['per_seed_score'][str(s)]:.3f}" for s in info["seeds"]
        )
        lines.append(f"- `{tag}`: {cells}")
    lines.append("")

    CONFIRM_ROOT.mkdir(parents=True, exist_ok=True)
    (CONFIRM_ROOT / "RANKING_CONFIRM.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (CONFIRM_ROOT / "ranking_confirm.json").write_text(
        json.dumps(
            {
                "scored_split": "heston_S_valdisc_8192x128.npy",
                "seeds": list(SEEDS),
                "fresh_seeds": list(FRESH_SEEDS),
                "ranked_on": "score_fresh (median over seeds 1/3/4, unbiased by selection)",
                "columns": [name for name, _ in COLUMNS],
                "paper_target": TARGET,
                "winner": best_tag,
                "winner_beats_paper": beats,
                "trials": dict(ranked),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
