"""Check a method README's section-5 file-layout tree against what is actually on disk.

Guideline section 7.6 states the rule this enforces: "Every file on disk appears here and every
entry here exists on disk. Verify with ``find . -type f | sort`` against this tree -- do not
eyeball it." Eyeballing is exactly what fails: the trees use ``seed_{0..4}``, ``seed_<q>`` and
``seed_{0..4}_{disc,pred}_{gru,mlp}_loss.csv``, so a single line can stand for twenty files, and
a reader scanning for a name will not notice the twenty-first is absent.

The two failure modes this catches are both real and both have occurred:

* an artefact was produced and nobody documented it (``tables_A.md``, written by the metrics
  runner into ``code/logs/``, was missing from two of the four committed trees);
* a tree entry names a file that was never produced, or produced at a different path (the CSDI
  trees claimed ``raw_banks/seed_{0..4}_raw.npy`` while the repair tool writes
  ``raw_banks/generated_paths/seed_N/generated_paths_8192x128.npy``).

Scope, stated plainly rather than overclaimed: matching is on *basenames*, not full paths. A file
documented under the wrong parent directory passes. Full-path reconstruction is not possible
here because the trees deliberately use an inline-children convention -- ``logs/`` lists its
contents in the comment column rather than as indented child lines -- and a parser that assumed
strict indentation would report those children as missing, which is a false alarm on every
correct README. Basenames catch undocumented artefacts and phantom entries, which are the
failures that occur; wrong-parent is caught by ``check_method_layout.py`` instead.

The two directions need two different denominators, and conflating them was a bug.

*On disk but not in README* asks "did we produce something nobody documented?". Its denominator
is ``git ls-files --cached --others --exclude-standard``: tracked files plus untracked ones that
are not ignored. Using ``os.walk`` here would drag in gitignored working output -- per-seed
``.log`` files, harness ``.omc/`` scratch -- and demand the README document scratch it should not
mention. Using ``git ls-files`` alone would fail for a README written before its artefacts are
committed, which is precisely when it is most useful to run this.

*In README but not on disk* asks the opposite question -- "does this entry name a file that was
never produced?" -- and for that, git's opinion is irrelevant. Its denominator is the real
filesystem, ``os.walk``. The TimeDiT checkpoints are the case that forced the distinction: at
124.1 MiB each they are over GitHub's per-file hard cap and are gitignored, but they sit on disk
at exactly the documented path and the README's independent-retraining evidence (five distinct
MD5s per experiment) is verified against them. Reporting those as "not on disk" was false, and
the only ways to silence it under the old denominator were to delete a true tree entry or to stop
ignoring a file that cannot be pushed. A phantom is a name that exists *nowhere*, not a name git
has been told to keep out of a commit.

Usage:
  python tools/check_readme_tree.py --method experiment_A/TimeDiT
  python tools/check_readme_tree.py --all
"""
import os
import re
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # results/new_experiments
REPO = os.path.dirname(os.path.dirname(ROOT))     # the git working tree

# A token that could be a filename: word characters plus the brace/angle notation the trees use,
# ending in a dot and an extension. Deliberately greedy about punctuation so that
# "seed_{0..4}_{disc,pred}_{gru,mlp}_loss.csv" is captured as ONE token and expanded below,
# rather than shredded into fragments that match nothing.
FILENAME_TOKEN = re.compile(r"[A-Za-z0-9_<>{},.\-]*\.[A-Za-z0-9]+")

# A documented name is only *expected* on disk if it looks like a per-method artefact. The
# READMEs also name shared tools, dataset files and sibling-directory figures; none of those
# live under the method directory and none of them are defects.
ARTEFACT_NAME = re.compile(r"^(seed_\d|loss_convergence|metrics_summary|heston_diagnostics)")


def expand(token):
    """Every literal filename a tree token stands for.

    ``seed_{0..4}_losses.csv``            -> 5 names
    ``seed_<q>_metrics.json``             -> 5 names
    ``seed_0_{pca,tsne}.png``             -> 2 names
    ``seed_{0..4}_{disc,pred}_loss.csv``  -> 10 names (braces expand left to right, recursively)

    A token with no brace and no ``<q>`` expands to itself, so plain names cost nothing.
    """
    m = re.search(r"\{([^{}]*)\}", token)
    if m:
        body = m.group(1)
        if ".." in body:
            lo, hi = body.split("..")
            options = [str(i) for i in range(int(lo), int(hi) + 1)]
        else:
            options = body.split(",")
        out = []
        for opt in options:
            out.extend(expand(token[:m.start()] + opt + token[m.end():]))
        return out
    if "<q>" in token:
        return [name for q in range(5)
                for name in expand(token.replace("<q>", str(q), 1))]
    return [token]


def documented_names(text):
    """Every filename the README names, brace notation expanded.

    Scanning the whole file rather than only the fenced tree is intentional. A README that
    documents an artefact in prose ("written to ``losses/seed_0_losses_steps.csv``") has
    disclosed it, and failing that would push authors to pad the tree with duplicates. The cost
    is that this direction cannot prove the *tree* is complete, only that the README is -- which
    is the property section 7.6 is actually protecting.
    """
    names = set()
    for token in FILENAME_TOKEN.findall(text):
        names.update(expand(token))
    return names


def on_disk_names(method_rel):
    """Basenames of every file git considers part of the project, under ``method_rel``."""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", method_rel],
        cwd=REPO, capture_output=True, text=True, check=True).stdout.split("\n")
    return {p.rsplit("/", 1)[-1]: p for p in out if p.strip()}


def produced_names(method_dir):
    """Basenames of every file that physically exists under ``method_dir``.

    Used only to decide whether a documented name is a phantom. Gitignored artefacts appear here
    and correctly do not count as phantoms; they still do not appear in ``on_disk_names``, so this
    does not force the README to document scratch.
    """
    names = set()
    for _, _, files in os.walk(method_dir):
        names.update(files)
    return names


def check(method):
    """Both directions of the section-7.6 rule for one method directory."""
    method_dir = os.path.join(ROOT, method)
    readme = os.path.join(method_dir, "README.md")
    if not os.path.exists(readme):
        raise SystemExit(f"{method}: no README.md")

    rel = os.path.relpath(method_dir, REPO)
    disk = on_disk_names(rel)
    if not disk:
        raise SystemExit(f"{method}: git lists no files under {rel}")

    with open(readme) as fh:
        text = fh.read()
    documented = documented_names(text)

    produced = produced_names(method_dir)

    undocumented = sorted(name for name in disk if name not in documented)
    phantom = sorted(name for name in documented
                     if name not in produced and ARTEFACT_NAME.match(name))
    ignored = sorted(name for name in documented if name in produced and name not in disk
                     and ARTEFACT_NAME.match(name))

    print(f"--- {method}: {len(disk)} project files, {len(documented)} names in README ---")
    for name in ignored:
        print(f"  note on disk but gitignored, not pushed: {name}")
    for name in undocumented:
        print(f"  FAIL on disk but not in README: {disk[name]}")
    for name in phantom:
        print(f"  FAIL named in README but not on disk: {name}")
    if not undocumented and not phantom:
        print("  PASS")
    return len(undocumented) + len(phantom)


def discover():
    """Every method directory that has a README, across both experiments."""
    found = []
    for experiment in ("A", "B"):
        base = os.path.join(ROOT, f"experiment_{experiment}")
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            if os.path.exists(os.path.join(base, name, "README.md")):
                found.append(f"experiment_{experiment}/{name}")
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", help="e.g. experiment_A/TimeDiT")
    ap.add_argument("--all", action="store_true", help="every method directory with a README")
    args = ap.parse_args()

    if args.all:
        methods = discover()
    elif args.method:
        methods = [args.method.strip("/")]
    else:
        raise SystemExit("pass --method <experiment_X/Method> or --all")

    total = sum(check(m) for m in methods)
    if total:
        raise SystemExit(f"\n{total} section-7.6 tree problem(s) across {len(methods)} method(s)")
    print(f"\nPASS: {len(methods)} method README(s) agree with disk")


if __name__ == "__main__":
    main()
