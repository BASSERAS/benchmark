# Contributing

Thank you for your interest in this benchmark. It provides a controlled,
reproducible testbed for generative models of financial time series, and it
grows one method at a time. Contributions that add a new method, add or refine a
metric, or fix a reproduction detail are all welcome.

This benchmark is sponsored by Murex.

## Ground rules

- **`GUIDELINE.md` is the single source of truth.** It is the absolute priority
  for folder layout, file naming, README content, and the metric protocol. Read
  it before you touch anything. If a rule here and a rule there ever disagree,
  `GUIDELINE.md` wins.
- **Numbers are never hand typed.** Every value in every comparison table is
  produced by `metrics/render_tables.py`, the deterministic renderer. If you add
  a method, you register it in the `FAMILIES` list and re render; you do not edit
  table cells by hand.
- **Reproduce before you report.** A new method must pass its paper reproduction
  gate (the ours versus paper table) and be signed off before you spend compute
  on the Heston seeds.
- **Verify against disk before you assert.** README files are contracts. Do not
  name a script, a column, or a path you have not checked on disk.

## Adding a new method

The full procedure lives in `GUIDELINE.md` (see the add a method sections). In
short:

1. **Scaffold** the method folder under `methods/<Method>/` following the exact
   tree in `GUIDELINE.md`, and commit the reference paper PDF under
   `methods/<Method>/paper_reimplementation/`.
2. **Reproduce** the paper on its own dataset first. Produce the ours versus
   paper table and STOP for sign off. Do not proceed to Heston until it matches
   or the gap is explained.
3. **Train** the five canonical Heston seeds within the hardware limits (at most
   two GPUs, sixteen cores; the machine is shared, so check `nvidia-smi` and
   `htop` first).
4. **Score** with the metrics in `metrics/` (never a private or vendored metric
   set), then run the diagnostics and the path shadowing harness.
5. **Register** the method in the `FAMILIES` list of
   `metrics/render_tables.py`, then re render the A, B, and PS tables into both
   the root `README.md` and `results/README.md`. Verify they are byte identical
   to each other and to the renderer output.
6. **Document** the method with the required READMEs (method, code, paper
   reimplementation, results, path shadowing), each carrying the exact run path.
7. **Log** the change with a dated entry in the `GUIDELINE.md` update section.

## Style

- All Markdown prose in this repository avoids typographic dashes. Use a comma
  where you would reach for an em dash, and a plain hyphen only inside compound
  words (for example `McKean-Vlasov`) or code.
- Keep tables generated, keep claims backed by files on disk, keep commits
  scoped to one logical change.
- Do not commit weights or any file above the GitHub size limit; large model
  artifacts are gitignored per method.

## Reporting issues

- For a bug, an unclear metric, or a reproduction gap, open a normal issue.
- For a security concern, follow `SECURITY.md` instead of opening a public
  issue.
- For anything else, email tbasseras@murex.com.

## Code of Conduct

By participating you agree to uphold our `CODE_OF_CONDUCT.md`.
