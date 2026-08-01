"""Render the cross-method comparison tables used by the experiment-level READMEs.

The method-level READMEs (``experiment_X/<METHOD>/README.md``) are machine-checked cell by
cell by ``check_readme_values.py``. The experiment-level comparison READMEs are not covered
by that checker, so hand-transcribing two methods' worth of numbers into a merged HTML table
would reintroduce exactly the failure mode the checker exists to prevent. This script instead
*generates* the merged tables by shelling out to the same two producers the method READMEs
use, so a comparison cell cannot disagree with the method-README cell it was copied from.

Producers invoked (unmodified, via their documented CLI flags):

* ``aggregate_pdf_metrics.py``  -- PDF section-1 metric table, and the ``--paired-dir`` variant
  that emits seedwise differences and the paired 95 % CI. Pairing is only meaningful because
  CSDI and LS4 were run on the same seeds 0-4; the producer itself refuses to pair otherwise.
* ``make_metrics_tables.py``    -- battery tables A (A1-A34) and B (curve-shape measures).

"Winner" is decided by distance to a per-metric *reference*, never by "smaller number wins":
an error/distance metric references 0 and a ``generated_*`` quantity references its own
``target_*`` row. Metrics with no defined reference get no winner, and there are two kinds:

* ``target_*`` constants -- a reading of ``test.npy`` alone, identical in both columns;
* the PDF's **raw diagnostics** -- novelty (S4), oracle posterior confidence (S3.3), and the
  ``std_ratio`` / group-mean / raw-proportion family (S5). These have a value but no direction.
  They render as ``diagnostic (<clause>)`` and are excluded from every tally, because S4 is
  explicit that they "must not be combined with fidelity metrics into a score".

For PDF metrics the winner is additionally gated on the paired 95 % CI excluding zero. Two
means far apart with a CI straddling zero are reported as a tie, which is what the producer's
own footer says to do.

Usage:
  python make_comparison_tables.py --experiment A --table pdf
  python make_comparison_tables.py --experiment B --table A
"""
import os
import re
import sys
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Method -> the model-family label the comparison header groups it under. Order fixes the
# column order of every generated table. Adding a method here is the only edit needed: the
# header, the winner logic and the pairwise section below are all written for N columns.
FAMILIES = [("CSDI", "Diffusion"), ("TimeDiT", "Diffusion Transformer"), ("LS4", "VAE")]
METHODS = [m for m, _ in FAMILIES]

# Experiment -> the glob that selects that experiment's evaluator output.
PATTERN = {"A": "*_drawdown_memory.json", "B": "*_heston_mixture.json"}

NUM = re.compile(r"^\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def run(args):
    out = subprocess.run([sys.executable] + args, cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"producer failed: {' '.join(args)}\n{out.stderr}")
    return out.stdout


def parse_md_table(text):
    """Return the pipe-table rows as lists of cells, dropping header and separator lines.

    Splitting must ignore *escaped* pipes: battery labels such as ``A2 \\|r\\| q95 Error``
    embed a literal '|' in the cell text, and a naive ``split('|')`` silently shifts every
    later column of those rows by two -- which produced a table where a metric's value was
    the string 'r\\'.
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = re.split(r"(?<!\\)\|", line)
        if cells and not cells[0].strip():
            cells = cells[1:]
        if cells and not cells[-1].strip():
            cells = cells[:-1]
        cells = [c.strip() for c in cells]
        if all(set(c) <= set("-: ") for c in cells):   # the |---|---| separator
            continue
        rows.append(cells)
    return rows[1:], rows[0]      # data rows, header row


def mean_of(cell):
    """First number of a '<mean> ± <std>' cell, or None for 'n/a' and section headers."""
    if not cell or cell.startswith("**") or cell == "n/a":
        return None
    m = NUM.match(cell)
    return float(m.group(1)) if m else None


# --------------------------------------------------------------------------------------
# PDF-declared RAW DIAGNOSTICS. These carry a value but MUST NOT carry a winner.
#
#   §5  "All displayed fidelity metrics are errors or distances, so lower is better,
#        except explicitly identified raw diagnostics such as standard-deviation ratio,
#        posterior confidence, group means, and novelty."
#   §4  novelty "does not have a universal monotone 'better' direction and must not be
#        combined with fidelity metrics into a score."
#   §3.3 "Oracle posterior entropy, maximum probability, and low-confidence fraction are
#        diagnostics, not separate winner-selection criteria."
#
# Scoring these is not a cosmetic error. In Experiment A the *only* row CSDI won was
# `generated_memory.future_rv_no_hit_mean` -- a group mean, i.e. a raw diagnostic. Counting
# it turned "LS4 wins every contested fidelity metric" into "17-1", which is a materially
# different claim about the same numbers.
DIAGNOSTIC = [
    (re.compile(r"^novelty\."),                              "§4 novelty"),
    (re.compile(r"^mixture_fidelity\.generated_"
                r"(low_confidence_fraction|mean_max_probability|mean_posterior_entropy)$"),
                                                             "§3.3 posterior confidence"),
    (re.compile(r"\.std_ratio$"),                            "§5 std-deviation ratio"),
    (re.compile(r"^generated_memory\.future_rv_(hit|no_hit)_mean$"), "§5 group mean"),
    # The fidelity quantity built from these is regime_proportion_tvd; scoring each bin
    # separately would count the same disagreement eight more times.
    (re.compile(r"^mixture_fidelity\.generated_regime_proportions\."), "§5 raw proportion"),
]


def diagnostic_of(key):
    """Return the PDF clause making ``key`` a raw diagnostic, or None if it is a fidelity metric."""
    for pat, why in DIAGNOSTIC:
        if pat.search(key):
            return why
    return None


def pdf_reference(key, means, floor):
    """The value a metric *should* take, against which both methods are measured.

    Returns None for anything that must not be scored: constants, and every raw diagnostic.
    """
    # A target row is a reading of `test.npy` alone. It is identical in both columns and is not
    # a contest. Experiment A names them `target_memory.*` (leading segment), Experiment B names
    # them `mixture_fidelity.target_*` (second segment) -- a `key.startswith("target_")` test
    # catches only the first, so all 11 of B's target rows fell through to reference 0.0, tied
    # against themselves, and were counted as 11 spurious "ties". Match any segment.
    if any(seg.startswith("target_") for seg in key.split(".")):
        return None                                   # a constant, not a contest
    if diagnostic_of(key):
        return None                                   # reported, never scored (§4/§3.3/§5)
    if key.startswith("generated_memory.") or "generated_" in key:
        target = key.replace("generated_", "target_")
        return means.get(target)                      # match the oracle's own reading
    return 0.0                                        # errors, distances, TVDs, KS, Wasserstein


def battery_reference(label):
    """Battery labels carry their own direction: '↓' means 0, '→ 1' means 1."""
    if "n/a" in label:
        return None
    if "→ 1" in label:
        return 1.0
    return 0.0


def rank(ref, values):
    """Method indices ordered by distance to ``ref``, closest first, or None if unscorable.

    Separated from ``decide`` because the PDF section-5 gate needs to know *which two* methods
    are being separated before it can look up their paired interval. With two columns that is
    trivially the only pair; with three it is the best and the runner-up, and picking the wrong
    pair would gate the winner on an interval computed for a comparison nobody is claiming.
    """
    if ref is None or any(v is None for v in values):
        return None
    return sorted(range(len(values)), key=lambda i: abs(values[i] - ref))


def decide(ref, values, tie=False):
    """Return (winner_text, [bold flag per method]). ``tie`` forces a draw regardless of distance.

    A shared closest distance is a tie: with N columns the winner must be *strictly* closer to
    the reference than every other method, otherwise the table would award a win on a
    floating-point coincidence.
    """
    order = rank(ref, values)
    if order is None:
        return "—", [False] * len(values)
    d = [abs(v - ref) for v in values]
    best = order[0]
    if tie or sum(1 for x in d if x == d[best]) > 1:
        return "<i>tie</i>", [False] * len(values)
    bold = [i == best for i in range(len(values))]
    return METHODS[best], bold


def cell(text, bold):
    return f"<b>{text}</b>" if bold else text


def unescape(text):
    """Markdown escapes a literal pipe as '\\|'; inside an HTML cell that backslash is text."""
    return text.replace("\\|", "|")


def header(extra_cols, first_col="Metric", lead_extra=None):
    fam = "".join(f"<th>{f}</th>" for _, f in FAMILIES)
    lead = f'<th rowspan="2">{first_col}</th>'
    if lead_extra:
        lead += "".join(f'<th rowspan="2">{c}</th>' for c in lead_extra)
    ext = "".join(f'<th rowspan="2">{c}</th>' for c in extra_cols)
    names = "".join(f"<th>{m}</th>" for m, _ in FAMILIES)
    return (f"<tr>{lead}{fam}<th rowspan=\"2\">Perfect</th>{ext}"
            f'<th rowspan="2">Winner</th></tr>\n'
            f"<tr>{names}</tr>")


def table_pdf(exp):
    """PDF section-1 metrics, both methods, floor, paired difference and paired CI."""
    pat = PATTERN[exp]
    per = {}
    for method, _ in FAMILIES:
        rows, _h = parse_md_table(run([
            os.path.join(HERE, "aggregate_pdf_metrics.py"),
            "--model-dir", f"experiment_{exp}/{method}",
            "--floor-dir", f"experiment_{exp}/perfect_floor",
            "--label", method, "--subdir", "pdf_metrics", "--pattern", pat,
            "--exclude-prefix", "configuration", "oracle_gate"]))
        per[method] = {r[0].strip("`"): r for r in rows}

    # PDF §5 requires the seedwise differences and the paired interval for *aligned-seed
    # model-vs-model* comparisons. With N methods there are N(N-1)/2 such comparisons and the
    # clause is about each of them, so every pair is produced -- not just the pair that happens
    # to decide the winner.
    pairs = [(METHODS[i], METHODS[j])
             for i in range(len(METHODS)) for j in range(i + 1, len(METHODS))]
    paired = {}
    for left, right in pairs:
        rows, _h = parse_md_table(run([
            os.path.join(HERE, "aggregate_pdf_metrics.py"),
            "--model-dir", f"experiment_{exp}/{left}",
            "--paired-dir", f"experiment_{exp}/{right}",
            "--label", left, "--paired-label", right,
            "--pattern", pat,
            "--exclude-prefix", "configuration", "oracle_gate", "sources"]))
        paired[(left, right)] = {r[0].strip("`"): r for r in rows}

    a_key = METHODS[0]
    means = {k: mean_of(v[1]) for k, v in per[a_key].items()}
    floor = {k: mean_of(v[3]) for k, v in per[a_key].items()}

    # With exactly two methods the single pair's three columns ride along in the main table,
    # which is how every committed comparison README already reads. With three or more, that
    # would add 3*N(N-1)/2 columns to a table that is already wide, so the pairwise blocks are
    # emitted as their own tables underneath instead. Same numbers, same producer, same clause.
    inline = len(METHODS) == 2
    extra = ["Seedwise differences", "Mean diff", "Paired 95% CI"] if inline else []

    out = ["<table>", header(extra)]
    for key, row in per[a_key].items():
        others = [per[m].get(key) for m in METHODS[1:]]
        if any(o is None for o in others):
            continue
        rows_all = [row] + others
        values = [mean_of(r[1]) for r in rows_all]
        why = diagnostic_of(key)
        ref = pdf_reference(key, means, floor)

        # Gate the winner on the paired interval of the two methods actually being separated.
        order = rank(ref, values)
        straddles = True
        pr_win = None
        if order is not None and len(order) >= 2:
            best, second = METHODS[order[0]], METHODS[order[1]]
            pr_win = paired.get((best, second)) or paired.get((second, best))
            pr_win = pr_win.get(key) if pr_win else None
            if pr_win:
                d, h = mean_of(pr_win[4]), mean_of(pr_win[5].lstrip("±"))
                # A CI whose half-width covers the mean difference straddles zero: not a real gap.
                straddles = (d is None or h is None or abs(d) <= h)

        win, bolds = decide(ref, values, tie=straddles)
        if why:
            win = f'<i>diagnostic ({why})</i>'
        cells = "".join(f"<td>{cell(r[1], b)}</td>" for r, b in zip(rows_all, bolds))
        tail = ""
        if inline:
            pr = paired[(METHODS[0], METHODS[1])].get(key)
            tail = (f"<td>{pr[3] if pr else '—'}</td><td>{pr[4] if pr else '—'}</td>"
                    f"<td>{pr[5] if pr else '—'}</td>")
        out.append(f"<tr><td><code>{key}</code></td>{cells}"
                   f"<td>{row[3]}</td>{tail}<td>{win}</td></tr>")
    out.append("</table>")

    if not inline:
        for left, right in pairs:
            out.append(f"\n<p><b>Paired, aligned-seed: {left} vs {right}</b> "
                       f"(PDF §5, seedwise differences and paired 95% CI)</p>")
            out.append("<table>")
            out.append(f"<tr><th>Metric</th><th>Seedwise differences</th>"
                       f"<th>Mean diff</th><th>Paired 95% CI</th></tr>")
            for key in per[a_key]:
                pr = paired[(left, right)].get(key)
                if not pr:
                    continue
                out.append(f"<tr><td><code>{key}</code></td><td>{pr[3]}</td>"
                           f"<td>{pr[4]}</td><td>{pr[5]}</td></tr>")
            out.append("</table>")
    return "\n".join(out)


def keyed_rows(rows, which):
    """Key every data row uniquely, preserving source order.

    Table B leaves the 'Plot' cell blank on continuation rows, so the raw pair
    ``('', '% error')`` is *not* unique -- keying on it silently made every plot group's
    sub-rows collide, and the last group's numbers won. The plot name is therefore
    forward-filled into the key while the *display* cell stays blank.
    """
    out, current = [], ""
    for r in rows:
        if which == "B":
            if r[0]:
                current = r[0]
            key, display = (current, r[1]), (r[0], r[1])
        else:
            key, display = (r[0],), (r[0],)
        out.append((key, display, r))
    return out


def md_bold(text):
    """'**x**' is markdown; inside an HTML cell it has to become '<b>x</b>'."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def table_battery(exp, which):
    """Battery table A (A1-A34) or B (curve-shape measures), both methods plus floor."""
    per = {}
    order = []
    for method, _ in FAMILIES:
        rows, _h = parse_md_table(run([
            os.path.join(HERE, "make_metrics_tables.py"),
            "--model-dir", f"experiment_{exp}/{method}",
            "--floor-dir", f"experiment_{exp}/perfect_floor",
            "--label", method, "--table", which]))
        per[method] = {}
        for key, display, r in keyed_rows(rows, which):
            if key in per[method]:
                raise SystemExit(f"duplicate row key {key} in table {which} for {method}")
            per[method][key] = r
            if method == FAMILIES[0][0]:
                order.append((key, display))

    a_key = METHODS[0]
    lead = 2 if which == "B" else 1                    # leading label columns
    ncol = lead + len(METHODS) + 2                     # + Perfect + Winner
    out = ["<table>"]
    out.append(header([], first_col="Plot", lead_extra=["Measure"]) if which == "B"
               else header([]))
    for key, display in order:
        row = per[a_key][key]
        others = [per[m].get(key) for m in METHODS[1:]]
        if any(o is None for o in others):
            continue
        rows_all = [row] + others
        is_section = row[0].startswith("**") and row[0].endswith("**") and \
            all(c == "" for c in row[1:])
        if is_section:
            out.append(f'<tr><td colspan="{ncol}"><b>{unescape(row[0].strip("*"))}</b>'
                       f"</td></tr>")
            continue
        ref = battery_reference(" ".join(x for x in key if x))
        win, bolds = decide(ref, [mean_of(r[lead]) for r in rows_all])
        lead_cells = "".join(f"<td>{md_bold(unescape(x))}</td>" for x in display)
        val_cells = "".join(f"<td>{cell(r[lead], b)}</td>" for r, b in zip(rows_all, bolds))
        out.append(f"<tr>{lead_cells}{val_cells}"
                   f"<td>{row[-1]}</td><td>{win}</td></tr>")
    out.append("</table>")
    return "\n".join(out)


def build(exp, which):
    return table_pdf(exp) if which == "pdf" else table_battery(exp, which)


# A comparison README marks each generated table with these fences, so the file can be
# refreshed from the artefacts instead of being re-transcribed when a seed is re-run.
# Every group here is *named*. A bare '(...)' sitting next to a '(?P<name>...)' is a trap:
# named groups also consume numbered slots, so in an earlier version of this pattern the
# closing fence was group 4 while the code reached for group 3 -- and group 3 was `tbl`.
# Injection therefore replaced every '<!-- END GENERATED -->' with the literal string
# 'pdf' / 'A' / 'B', destroying the fence; a second --inject run could then no longer find
# the block. Name every group and index by name only.
BLOCK = re.compile(
    r"(?P<open><!-- BEGIN GENERATED: experiment (?P<exp>[AB]) table (?P<tbl>pdf|A|B) -->\n)"
    r".*?"
    r"(?P<close>\n<!-- END GENERATED -->)",
    re.DOTALL)


def inject(path):
    """Rewrite every fenced block in ``path`` with freshly generated content."""
    text = open(path).read()
    n = 0

    def repl(m):
        nonlocal n
        n += 1
        return m.group("open") + build(m.group("exp"), m.group("tbl")) + m.group("close")

    new = BLOCK.sub(repl, text)
    if n == 0:
        raise SystemExit(f"{path}: no '<!-- BEGIN GENERATED: ... -->' block found")
    # Injection must be idempotent: the fences it consumed have to still be there afterwards,
    # or the next run silently swallows the remainder of the file.
    if len(BLOCK.findall(new)) != n:
        raise SystemExit(f"{path}: injection damaged a fence -- refusing to write")
    with open(path, "w") as fh:
        fh.write(new)
    print(f"{path}: refreshed {n} generated table(s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=("A", "B"))
    ap.add_argument("--table", choices=("pdf", "A", "B"))
    ap.add_argument("--inject", help="README whose fenced tables should be regenerated")
    a = ap.parse_args()
    if a.inject:
        inject(a.inject)
        return
    if not (a.experiment and a.table):
        raise SystemExit("need --experiment and --table, or --inject")
    print(build(a.experiment, a.table))


if __name__ == "__main__":
    main()
