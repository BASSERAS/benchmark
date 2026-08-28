#!/usr/bin/env python3
"""Table D — intraday portfolio VaR/ES backtest. ONE renderer, four consumers.

WHY THIS IS A SHARED MODULE AND TABLES A/B/C ARE NOT
----------------------------------------------------
Tables A, B and C are re-implemented in each of the four renderers
(render_comparison.py and {SBTS,CSDI,Deep-MKV-TS}/code/render_readme.py).  That
duplication has already cost real work: when the baseline seed sweep landed, the
same two bootstrap rows had to be patched in three files by byte-exact
substitution, and a silent zero-hit in any one of them would have left the same
number printing two different values on two pages of the same repo.

Table D is therefore written once and imported.  If you are tempted to inline it
into a renderer "just for this method", don't -- that is precisely the drift this
module exists to prevent.

WHAT THE TABLE SAYS
-------------------
Read the alpha-quantile of the K = 256 retrieved continuations' equal-weight
portfolio return as a VaR forecast, and count breaches.  Full protocol and the
mechanism analysis live in metrics/var_backtest_multiasset.py.  Three facts
govern how the rows are ordered and how the table must be read:

  1. NOMINAL IS THE TARGET, NOT ZERO.  A 3% exception rate at alpha = 5% is
     exactly as wrong as 7% -- one is a bank holding capital against a risk that
     was not there, the other is a bank that is short of it.  Rows are therefore
     ranked by |rate - alpha|, the same principle as winner() in the other
     renderers, which ranks by |log ratio| rather than "lowest wins" (README 7.1).

  2. THE FLOOR ROW IS NOT A FLOOR HERE.  In Table C the real-training-split bank
     is a lower bound that generators should not beat.  In Table D it is the
     SECOND-WORST row (12.06% at alpha = 5%), because the failure being measured
     is a property of the retrieval step, which applies to the real data exactly
     as it applies to a generator.  Do not carry Table C's xfloor interpretation
     across; the two tables use the same bank for opposite purposes.

  3. THE LAST COLUMN IS A PRODUCT, AND BOTH FACTORS MATTER.  ens/real factors as
     (pool sd / real sd) x (conditional shrinkage).  The first factor is a
     property of the GENERATOR -- is the 8192-path pool as dispersed as the
     market?  The second is a property of the PROTOCOL -- how much does taking
     256 nearest neighbours collapse that dispersion?  CSDI and real_train_bank
     land at almost the same product (0.693 vs 0.656) by opposite routes: CSDI's
     pool is 27% too narrow and barely shrinks, the real bank's pool is correct
     and shrinks by 37%.  Reporting only the product hides which of the two is
     broken, so the column prints all three numbers.

WHY alpha = 5% ONLY
-------------------
With K = 256 the empirical alpha-quantile sits near order statistic alpha*K: the
13th of 256 at alpha = 5%, a stable estimate.  At alpha = 1% it is the ~2.6th and
the estimator's own finite-sample bias tightens VaR by a few percent.  The full
1%/5%/10% sweep is in var_backtest.json and the prose cites it -- the failure is
level-independent -- but the printed table uses the level where the estimator is
not fighting the analyst.

THE QUERIES CROSS-CHECK
-----------------------
Table D's central claim is that it re-reads the SAME K = 256 ensemble Table C
scores with CRPS.  That claim is false unless both were computed on the same
query file.  It has already been false once: the first run of this table used
dataset/TrueDataset/true_S_test_8192x128x8.npy while every CRPS artefact in the
tree uses variants/om_2022-07_N6144 with true_S_test_6144x128x8.npy.  Every
number looked plausible.  _check_queries() now compares the two strings and
render() refuses to emit anything if they disagree, so that failure mode is loud.

FALLBACK
--------
load() returns {} when var_backtest.json is absent, and render() then returns
("", "") so every caller emits no Table D at all rather than a broken one.  A
checkout without the artefact renders exactly what it rendered before this file
existed.
"""

import glob
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
VAR_JSON = os.path.join(HERE, "var_backtest.json")
CRPS_GLOB = os.path.join(HERE, "*", "losses", "crps_configs", "paper__seed_*.json")

ALPHA = 0.05
NOMINAL_PCT = ALPHA * 100.0

# Display order is computed (by |rate - alpha|), but the labels are fixed here so
# a renamed bank in the JSON shows up as a missing row rather than a raw key.
NICE = {
    "block_bootstrap": "Moving-block bootstrap *(paper baseline)*",
    "session_bootstrap": "Session bootstrap *(paper baseline)*",
    "real_train_bank": "Real train split as bank *(Table C floor)*",
    "ewma": "RiskMetrics EWMA (λ=0.94) *(desk benchmark)*",
}


def _check_queries(blob):
    """-> None if Table D and Table C read the same queries, else a reason string.

    Returns None (rather than raising) when no CRPS config is on disk: a partial
    checkout should degrade to "cannot verify, print anyway", not to a crash.
    The only hard failure is a genuine disagreement, which is the bug this guards.
    """
    mine = blob.get("queries")
    if not mine:
        return "var_backtest.json carries no `queries` field (stale artefact)"
    theirs = set()
    for f in glob.glob(CRPS_GLOB):
        try:
            with open(f) as fh:
                q = json.load(fh).get("queries")
        except (OSError, ValueError):
            continue
        if q:
            theirs.add(q)
    if not theirs:
        return None
    if theirs != {mine}:
        return (f"query mismatch: var_backtest.json used {mine!r}, "
                f"CRPS configs used {sorted(theirs)!r}")
    return None


def load(path=None):
    """var_backtest.json -> {label: {...aggregated over seeds...}}, or {}.

    Only the K = 256 test-split cells are aggregated.  Three families of cell
    share a label with their headline sibling and would silently contaminate the
    mean if they were not filtered: the K-sweep cells (16..2048), the val-split
    control cells, and (for EWMA) the k == 0 marker.  They exist to demonstrate
    the mechanism in the prose, not to be averaged into a printed row.

    `path` defaults to None rather than to VAR_JSON so the module global is read
    at CALL time. A default argument would bind the path at import time, which
    makes the absent-artefact fallback impossible to exercise from a test that
    monkeypatches VAR_JSON -- the first version of this file had exactly that
    bug, and the fallback test passed while proving nothing.
    """
    path = path or VAR_JSON
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        blob = json.load(fh)
    key = f"alpha_{ALPHA:.2f}"
    out = {}
    for r in blob.get("results", []):
        # k == 0 is the EWMA row, which does no retrieval and so has no K.
        if r.get("k") not in (256, 0):
            continue
        if r.get("split", "test") != "test":
            continue
        lv = r.get("levels", {}).get(key)
        if not lv:
            continue
        d = out.setdefault(r["label"], {"n": 0, "rate": [], "kupiec_p": [],
                                        "p_cc": [], "es_ratio": [],
                                        "ens": [], "pool": [], "shrink": []})
        d["n"] += 1
        d["rate"].append(lv["rate"] * 100.0)
        d["kupiec_p"].append(lv["kupiec_p"])
        d["p_cc"].append(lv["p_cc"])
        d["es_ratio"].append(lv["es_ratio"])
        diag = r.get("diag", {})
        if "ens_sd" in diag:
            d["ens"].append(diag["ens_sd"])
            d["pool"].append(diag["bank_sd"])
            d["shrink"].append(diag["shrinkage"])
    out["_horizon"] = blob.get("horizon", {})
    # realised_sd is the denominator of the width column.  It lives in the
    # artefact and NOT in a renderer default argument, because it is a property
    # of the query file: when the query file changed under this table the first
    # time, a hardcoded default would have kept printing the old yardstick.
    out["_realised_sd"] = blob.get("realised_sd")
    out["_queries"] = blob.get("queries")
    out["_mismatch"] = _check_queries(blob)
    return out


def _pm(vals, fmt="{:.2f}"):
    """mean ± sd when the sweep has more than one seed, bare mean otherwise.

    Same convention as the bootstrap rows of Table C: a ± is printed only when it
    is a real cross-seed spread. Single-seed rows (the floor, EWMA) print one
    number rather than a fake ±0.00, so the reader can see at a glance which rows
    carry a distribution and which are point estimates.
    """
    if not vals:
        return "-"
    m = statistics.fmean(vals)
    if len(vals) == 1:
        return fmt.format(m)
    return f"{fmt.format(m)} ± {fmt.format(statistics.stdev(vals))}"


def render():
    """-> (markdown_table, prose_note). ("", "") when the artefact is absent.

    Takes no arguments on purpose.  Every number it needs -- including the
    realised sd of the test portfolio, which is the denominator of the last
    column -- is read off var_backtest.json, so the table cannot drift away from
    the run that produced it.
    """
    data = load()
    rows = {k: v for k, v in data.items() if not k.startswith("_")}
    if not rows:
        return "", ""
    if data.get("_mismatch"):
        # Loud, not silent: printing a Table D that reads different queries than
        # Table C while claiming to read the same ensemble is worse than no table.
        return "", f"<!-- Table D suppressed: {data['_mismatch']} -->"

    hz = data.get("_horizon", {})
    mins = hz.get("minutes", 16.0)
    real_sd = data.get("_realised_sd")

    # Rank by distance from nominal. Being too CONSERVATIVE is a failure too:
    # it is capital held against a risk that was not there.
    order = sorted(rows.items(),
                   key=lambda kv: abs(statistics.fmean(kv[1]["rate"]) - NOMINAL_PCT))

    lines = [
        f"| Bank | Seeds | Exceptions @ {NOMINAL_PCT:.0f}% | Kupiec `LR_uc` *p* | "
        f"`LR_cc` *p* | ES ratio | ens sd / real sd = pool × shrink |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, d in order:
        nice = NICE.get(label, f"**{label}**")
        # The worst p across seeds, not the mean: a method that passes on four
        # seeds and is rejected at 1e-60 on the fifth has not passed.  The
        # fraction that DID pass is printed alongside, because "3/5 marginal"
        # and "0/5 at 1e-104" are different findings and a bare ✗ merges them.
        kp = min(d["kupiec_p"]) if d["kupiec_p"] else float("nan")
        cp = min(d["p_cc"]) if d["p_cc"] else float("nan")
        npass = sum(1 for p in d["kupiec_p"] if p > 0.05)
        if npass == d["n"]:
            mark = " ✅"
        elif npass:
            mark = f" *({npass}/{d['n']})*"
        else:
            mark = ""
        if d["ens"] and real_sd:
            wid = (f"**{statistics.fmean(d['ens']) / real_sd:.3f}**"
                   f" = {statistics.fmean(d['pool']) / real_sd:.3f}"
                   f" × {statistics.fmean(d['shrink']):.3f}")
        else:
            wid = "–"
        lines.append(
            f"| {nice} | {d['n']} | {_pm(d['rate'])}%{mark} | {kp:.1e} | {cp:.1e} | "
            f"{_pm(d['es_ratio'])} | {wid} |"
        )

    note = f"""**Table D — intraday portfolio VaR backtest.** Horizon **{mins:.0f} minutes**
({hz.get('bars', 32)} bars x {hz.get('bar_seconds', 30)} s), equal-weight across the 8
assets, one forecast per test query. The alpha = {NOMINAL_PCT:.0f}% VaR is the empirical
{NOMINAL_PCT:.0f}th percentile of the **same K = 256 retrieved continuations that Table C
scores with CRPS** — no retraining, no new sampling, the identical ensemble read a
different way. The two tables are checked to read the same query file before this one is
printed.

This is a **liquidation-horizon** risk figure, roughly the time needed to unwind a crypto
position. It is **not** an FRTB or Basel capital number, which is a 10-day quantity, and
it is deliberately **not** scaled to one day by sqrt(t): that scaling assumes returns are
independent across time, which is the assumption the volatility-clustering results in
Table A refute. No sqrt(t) appears anywhere in the computation.

**How to read it.** Nominal is the target, not zero — {NOMINAL_PCT:.0f}% is correct, and
both 3% and 7% are wrong, one as over-capitalisation and the other as under. Rows are
ranked by distance from nominal, the same principle by which `winner()` ranks Table A by
|log ratio| rather than "lowest wins". The *p* columns show the **worst seed**, because
passing four and being rejected at 1e-104 on the fifth is not passing; the italic fraction
records how many seeds did pass Kupiec at the 5% level.

**The finding, in one line: the moving-block bootstrap is the only bank whose exception
rate is anywhere near nominal, and it is the only bank that does not condition.** It
rebuilds paths from randomly recombined blocks, so its "nearest neighbours" carry no
information about the query. Its rate is 5.01% against a 5% target, and its per-seed
misses change sign (4.30%, 4.72%, 5.09%, 5.35%, 5.58% — three of five pass Kupiec, the
other two marginally at *p* = 0.010 and 0.040). Every bank that **does** condition misses
in the same direction on every single seed, and is rejected at *p* < 2e-19 throughout.

**`ens sd / real sd` is the whole result, and it factors.** The bold number is the sd of
the 256-member predictive ensemble over the realised sd of the test portfolio return. It
is the product of a **generator** property — is the 8192-path pool as dispersed as the
market? — and a **protocol** property — how much does taking the 256 nearest neighbours
collapse that dispersion? The two failure modes are genuinely distinct:

* The real training split's pool is *correctly* dispersed (1.043) and conditioning
  destroys 37% of it (0.629). Pure protocol failure, no generator involved.
* CSDI's pool is 27% too narrow (0.726) and conditioning costs it almost nothing (0.955),
  because its paths are already too alike for retrieval to tighten them further. Pure
  generator failure — this is the over-sharpness signature of section 7.1, priced.
* The block bootstrap is slightly over-dispersed (1.068) and shrinks by 1% (0.990),
  which is what "my neighbours are uninformative" looks like as a number.

Across the seven retrieval banks the bold column and the exception rate have Spearman
rho = 0.96 (the sole inversion is CSDI vs Deep-MKV-TS, tied on width at 0.693/0.694 but
1.6 points apart on rate — width is the dominant term, not the only one).

**That the real-data row fails is what makes this a statement about the protocol rather
than about the generators.** Queried by its own training data, at 12.06% against a 5%
target, *p* = 2.7e-104. Three controls back it:

* **Not tail noise.** The floor breaches 3.84% at alpha = 1%, 12.06% at alpha = 5% and
  19.65% at alpha = 10% — 3.8x, 2.4x and 2.0x nominal. The estimator behaves differently
  at each level; the failure does not. Full sweep in `var_backtest.json`.
* **Not the held-out era.** Re-run against the *validation* split, the same era as the
  bank: 11.17% versus 12.06% out-of-era. Regime shift explains 0.89 of a 7.06-point miss —
  13% — and the in-era run is still rejected at *p* = 2.1e-82.
* **Not too few neighbours.** Sweeping K from 16 to 2048 — 128x more neighbours — moves
  the floor only 16.24% -> 9.88%, and its retained width only 0.618 -> 0.716 — still a
  third short of the market at the largest K tested, with no sign of closing.

**`ES ratio`** is the realised mean loss given a breach, divided by the loss the model
predicted for that same breach. Expected Shortfall is not elicitable (Gneiting 2011), so
it carries **no** *p*-value by construction — cite it as a magnitude, never as a test.
Every row sits between 0.97 and 1.14, straddling 1: **conditional on a breach, the loss is
about the size the model said it would be.** The failure is in the *frequency* of
breaches, not their severity — these ensembles have roughly the right tail shape in the
wrong place.

**Nothing passes both tests.** Read the `LR_cc` column: it is rejected on every row of the
table, the block bootstrap included, and on all five of its seeds (worst 9.0e-08, best
1.8e-06). That bank gets the *frequency* right and still fails independence, because it
has no volatility dynamics at all — its VaR is near-constant across queries, so its breach
sequence simply inherits the clustering of the truth. Everything else fails frequency as
well. Correct exception rate *and* correct breach timing is achieved by no method in this
study, which is the open problem this table poses rather than solves.

The independence test is legitimate here, which is unusual — most VaR backtests on H-step
returns cannot run it, because overlapping windows induce mechanical serial dependence in
the breach sequence. This dataset's windows are disjoint (stride = seq_len:
`n_windows_available` 33 570 against 4.30 M bars at `seq_len` 128), and transitions are
counted only within contiguous 256-window blocks, so the seam between two non-adjacent
blocks never contributes one.

**Why this is not a restatement of Table C.** CRPS is a distance over the whole law and is
dominated by its centre; a forecast that nails the bulk and collapses the tail loses very
little. Coverage is a frequency in the tail alone and cannot be bought back by sharpness.
Table D also shows why a portfolio view is not a per-asset view rescaled: from train to
test the mean pairwise correlation of the 8 assets rises 0.557 -> 0.769, so although each
asset became **17.6% less** volatile, the equal-weight portfolio became only **4.1% less**
volatile. Diversification decayed enough to absorb three quarters of the per-asset drop —
a bank that calibrated per-asset vol perfectly and carried the training-era correlation
forward would still understate portfolio risk by roughly 13%."""
    return "\n".join(lines), note


HEADING = "## Table D -- intraday portfolio VaR/ES backtest"


def section():
    """-> the complete markdown block (heading + table + prose + rule), or "".

    This exists so the four consumers each write ONE line. An earlier draft had
    them assemble heading/table/note/`---` themselves, which is four copies of a
    format string -- the exact duplication this module was created to remove, and
    the thing that would drift the first time the heading was reworded.
    """
    tbl, note = render()
    if tbl:
        return f"\n{HEADING}\n\n{tbl}\n\n{note}\n\n---\n"
    # A suppression note is an HTML comment: it must reach the rendered file so a
    # broken artefact is visible in `git diff`, but it must not print as prose.
    return f"\n{note}\n" if note else ""


if __name__ == "__main__":
    t, n = render()
    print(t or "(no var_backtest.json — Table D skipped)")
    print()
    print(n)
