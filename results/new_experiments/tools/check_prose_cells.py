"""Verify that hand-written tables in a comparison README agree with the generated blocks.

``make_comparison_tables.py --inject`` rewrites only what lies between the
``<!-- BEGIN GENERATED -->`` fences. Everything else on the page -- the verdict, the narrative,
and the small hand-written tables that quote a few rows for emphasis -- is prose, and prose
drifts. Two real defects motivated this checker:

* Experiment A kept a paragraph headed "CSDI's one win is a real one" naming a row that a fix to
  ``pdf_reference()`` had just reclassified as an unscored diagnostic;
* Experiment B's hand-written novelty table carried floor standard deviations (``± 122``,
  ``± 0.0263``, ``± 0.0685``) that appear nowhere in the artefacts; the generated block for the
  same three rows says ``± 24.5``, ``± 0.00181``, ``± 0.00338``.

Neither was caught by ``check_readme_values.py``, which validates the *method* READMEs against a
fresh regeneration and never looks at the comparison pages' free text.

The rule enforced here is narrow and mechanical: if a hand-written Markdown table quotes a metric
that also appears in a generated block on the same page, every cell it quotes must be
byte-identical to the generated one. Restating a number is allowed; restating it differently is
not.

Usage:
  python tools/check_prose_cells.py --readme experiment_A/README.md
  python tools/check_prose_cells.py --all
"""
import os
import re
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

READMES = ["README.md", "experiment_A/README.md", "experiment_B/README.md"]

# A generated data row: the metric key in <code>, then one <td> per column.
ROW = re.compile(r"<tr><td><code>(?P<key>[^<]+)</code></td>(?P<rest>.*?)</tr>")
CELL = re.compile(r"<td[^>]*>(?P<val>.*?)</td>")
MARKUP = re.compile(r"</?(?:b|i|code|sub|sup)>")

# Hand-written column header -> the generated column it must agree with. Matched by name, never
# by position: the generated header is a two-row family grouping whose layout changes whenever a
# method is added, and a positional checker would then compare the wrong pair of numbers.
HEADER_ALIASES = {
    "csdi": "CSDI",
    "timedit": "TimeDiT",
    "ls4": "LS4",
    "perfect": "Perfect",
    "perfect floor": "Perfect",
}


def strip(text):
    """A cell reduced to the characters that carry meaning, in either markup dialect.

    The generated blocks are HTML and mark the better method with ``<b>``; the hand-written
    tables are Markdown and mark it with ``**``. Both are emphasis, not data, so both are
    removed before comparing -- otherwise every winning cell in every prose table fails, which is
    the checker reporting its own bug rather than the document's.
    """
    text = MARKUP.sub("", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    return text.strip().strip("` ")


def generated_rows(text):
    """Every generated data row on the page as ``key -> {column_name: cell}``.

    A key appearing in more than one generated block on the same page (the top-level README holds
    both experiments) is dropped rather than guessed at: an ambiguous reference is not something
    a checker should resolve silently, and a wrong resolution here would report a false failure.
    """
    tables = re.findall(r"<table>.*?</table>", text, re.DOTALL)
    seen, out = {}, {}
    for table in tables:
        headers = [strip(h) for h in re.findall(r"<th[^>]*>(.*?)</th>", table, re.DOTALL)]
        named = [h for h in headers if h.lower() in HEADER_ALIASES]
        # Body order is the method columns first (they live in the second header row) and then
        # the rowspan'd 'Perfect' column, so rebuild the layout from the header text.
        methods = [h for h in named if HEADER_ALIASES[h.lower()] != "Perfect"]
        perfect = [h for h in named if HEADER_ALIASES[h.lower()] == "Perfect"]
        layout = [HEADER_ALIASES[h.lower()] for h in methods + perfect]
        # A table with no recognisable method column carries no cell a prose row could quote,
        # so it must not contribute to `seen` either. With three or more methods the pairwise
        # PDF-section-5 blocks ('Metric | Seedwise differences | Mean diff | Paired 95% CI')
        # repeat every metric key once per pair; counting them made *every* key look ambiguous
        # and silently emptied this checker -- it reported "no generated rows found" on a file
        # that had thirty of them.
        if not layout:
            continue
        for m in ROW.finditer(table):
            key = m.group("key")
            cells = [strip(c) for c in CELL.findall(m.group("rest"))]
            seen[key] = seen.get(key, 0) + 1
            out[key] = {name: cells[i] for i, name in enumerate(layout) if i < len(cells)}
    return {k: v for k, v in out.items() if seen[k] == 1}


def prose_tables(text):
    """Every hand-written Markdown pipe table, as ``(headers, [(cells, offset)])``.

    Rows inside a ``<table>`` are generated and skipped. Tables are found by Markdown's own
    grammar -- a header row, a ``|---|---|`` separator, then the consecutive pipe rows that
    follow -- and not by guessing which rows look like data.

    The earlier version decided a row was data only if one of its cells began with a backtick,
    and treated every other row as the start of a new header. That silently deleted whole
    tables. Experiment B's PRIMARY panel labels its rows 'Regime proportion TVD' rather than
    '`mixture_fidelity.regime_proportion_tvd`', so each row was reclassified as a header, the
    table never reached the comparison loop, and three stale floor standard deviations sat on
    the page through a passing run of this checker. A row that cannot be resolved to a metric
    must be *reported* as unverified -- which the caller already does, by name -- but it must
    never be invisible.
    """
    generated_spans = [m.span() for m in re.finditer(r"<table>.*?</table>", text, re.DOTALL)]

    def in_generated(pos):
        return any(a <= pos < b for a, b in generated_spans)

    rows = []
    for m in re.finditer(r"^\|(?P<row>.*)\|[ \t]*$", text, re.M):
        if in_generated(m.start()):
            continue
        cells = [c.strip() for c in m.group("row").split("|")]
        joined = "".join(cells)
        is_sep = bool(joined) and set(joined) <= set("-: ")
        rows.append((cells, m.start(), m.end(), is_sep))

    # Group into runs of physically adjacent lines: a blank line, or any prose, ends a table.
    runs, current = [], []
    for row in rows:
        if current and row[1] != current[-1][2] + 1:
            runs.append(current)
            current = []
        current.append(row)
    if current:
        runs.append(current)

    tables = []
    for run in runs:
        if len(run) < 3 or not run[1][3]:      # header, separator, and at least one body row
            continue
        headers = run[0][0]
        body = [(cells, start) for cells, start, _end, is_sep in run[2:] if not is_sep]
        if body:
            tables.append((headers, body))
    return tables


def check(readme_rel):
    path = os.path.join(ROOT, readme_rel)
    with open(path) as fh:
        text = fh.read()

    gen = generated_rows(text)
    if not gen:
        raise SystemExit(f"{readme_rel}: no generated rows found -- has --inject been run?")

    problems, compared, skipped = [], 0, []

    for headers, rows in prose_tables(text):
        if not headers:
            continue
        # Map each hand-written column index to a generated column name. The header may carry a
        # parenthetical note -- 'Novelty metric (diagnostic -- S4, not scored)' -- so strip it.
        mapping = {}
        for i, h in enumerate(headers):
            base = re.sub(r"\s*\(.*?\)\s*", " ", h).strip().lower()
            if base in HEADER_ALIASES:
                mapping[i] = HEADER_ALIASES[base]
        if not mapping:
            continue

        for cells, _pos in rows:
            name = strip(cells[0])
            # Hand-written tables quote the leaf name ('distinct_nearest_training_paths'); the
            # generated block uses the full dotted key. Resolve by suffix, and refuse to guess
            # when more than one key matches.
            matches = [k for k in gen if k == name or k.endswith("." + name)]
            if len(matches) != 1:
                # Record *what* was skipped, not just how many. A bare count is unfalsifiable:
                # "1 prose rows unmatched" reads identically whether the skipped row is a
                # provenance line ('Source', 0 matches -- the real case in both experiment
                # READMEs) or a metric whose name became ambiguous and is therefore silently
                # no longer verified. Only the second is a defect, and the count cannot
                # distinguish them.
                skipped.append(f"{name} ({len(matches)} matches)")
                continue
            key = matches[0]
            for i, col in mapping.items():
                if i >= len(cells):
                    continue
                want = gen[key].get(col)
                got = strip(cells[i])
                if want is None or not got:
                    continue
                compared += 1
                if got != want:
                    problems.append(
                        f"  FAIL {key} [{col}]: README prose says '{got}', "
                        f"generated block says '{want}'")

    print(f"=== hand-written cells vs generated blocks: {readme_rel} ===")
    print(f"  {len(gen)} unambiguous generated rows, {compared} prose cells compared, "
          f"{len(skipped)} prose rows unmatched")
    for name in skipped:
        print(f"  note unmatched, NOT verified: {name}")
    for line in problems:
        print(line)
    if problems:
        raise SystemExit(f"\n{len(problems)} prose/table disagreement(s) in {readme_rel}")
    print("PASS: every hand-written cell is byte-identical to its generated counterpart")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--readme", help="README to check, relative to results/new_experiments")
    ap.add_argument("--all", action="store_true", help="check all three comparison READMEs")
    args = ap.parse_args()
    if args.all:
        for rel in READMES:
            check(rel)
            print()
    elif args.readme:
        check(args.readme)
    else:
        ap.error("pass --readme <path> or --all")


if __name__ == "__main__":
    main()
