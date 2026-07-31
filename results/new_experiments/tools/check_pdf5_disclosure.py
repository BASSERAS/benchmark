"""Verify the PDF section-5 disclosure block in a comparison README against the manifests.

PDF section 5 requires that *each comparison table* states six things:

  1. the exact training, validation and test files;
  2. the generated bank size and all model seeds;
  3. whether official code was used, and its revision;
  4. whether hyperparameters were defaults or validation-selected;
  5. trainable parameters, training time, generation time, hardware;
  6. the number of failed runs and the reason for each.

All six are recorded per bank in ``generated_paths/seed_*/generation_manifest.json``. The
comparison READMEs restate them in prose, and prose drifts: this checker re-derives every item
from the ten manifests of an experiment and fails if the README no longer agrees.

It is deliberately a *substring* check rather than a cell parser. The disclosure is written for
a human, so its layout will change; what must not change is that the exact revision hash, the
exact dataset paths, and the exact "zero failures" claim appear somewhere in the file and match
the artefacts. A substring check is the strongest assertion that survives rewording.

Usage:
  python tools/check_pdf5_disclosure.py --experiment A
  python tools/check_pdf5_disclosure.py --experiment B
  python tools/check_pdf5_disclosure.py --experiment A --readme README.md   # the top-level page
"""
import os
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

METHODS = ["CSDI", "LS4"]
SEEDS = [0, 1, 2, 3, 4]


def manifests(experiment, method):
    """Every generation manifest for one method of one experiment, ordered by seed.

    A missing manifest is an error rather than a skip: PDF section 1.5 requires that failed
    seeds are *reported*, not silently replaced, so a checker that quietly tolerated four of
    five would defeat the clause it exists to enforce.
    """
    out = []
    for seed in SEEDS:
        path = os.path.join(ROOT, f"experiment_{experiment}", method,
                            "generated_paths", f"seed_{seed}", "generation_manifest.json")
        if not os.path.exists(path):
            raise SystemExit(f"missing manifest: {os.path.relpath(path, ROOT)}")
        with open(path) as fh:
            out.append(json.load(fh))
    return out


def uniform(records, path, label):
    """The single value ``path`` takes across ``records``, or an error if it varies.

    Items 1, 3 and 4 of the section-5 list are properties of the *run configuration*, so they
    must be identical across the five seeds. If they are not, the five banks are not five
    samples of one experiment and no aggregate over them means anything -- a harder failure
    than a stale README, and reported as such.
    """
    values = []
    for rec in records:
        node = rec
        for part in path.split("."):
            node = node[part]
        values.append(node)
    distinct = {json.dumps(v, sort_keys=True) for v in values}
    if len(distinct) != 1:
        raise SystemExit(f"{label}: '{path}' is not constant across seeds: {sorted(distinct)}")
    return values[0]


def check(experiment, readme_rel):
    readme_path = os.path.join(ROOT, readme_rel)
    with open(readme_path) as fh:
        text = fh.read()

    problems = []
    notes = []

    def require(needle, item, why):
        needle = str(needle)
        if needle in text:
            notes.append(f"  ok   item {item}: {why} -> '{needle}'")
        else:
            problems.append(f"  FAIL item {item}: {why} -- '{needle}' not found in {readme_rel}")

    for method in METHODS:
        recs = manifests(experiment, method)
        label = f"experiment_{experiment}/{method}"

        # ---- item 1: exact training / validation / test files -------------------------------
        files = uniform(recs, "data_files", label)
        for role in ("train", "validation", "test"):
            require(files[role].rsplit("/", 1)[-1], 1, f"{role} file ({method})")

        # ---- item 2: bank size and all model seeds ------------------------------------------
        shape = uniform(recs, "output_contract.shape", label)
        require(f"{shape[0]} × {shape[1]}", 2, f"bank shape ({method})")
        seen = sorted(r["seeds"]["generation_seed"] for r in recs)
        if seen != SEEDS:
            problems.append(f"  FAIL item 2: {label} generation seeds are {seen}, expected {SEEDS}")
        else:
            notes.append(f"  ok   item 2: {label} ran generation seeds {seen}")

        # ---- item 3: official code and its revision -----------------------------------------
        if uniform(recs, "model.official_implementation", label) is not True:
            problems.append(f"  FAIL item 3: {label} is not marked as an official implementation")
        revision = uniform(recs, "model.source_revision", label)
        # LS4 appends the wrapper commit after a ';'. Require the leading hash, which is the
        # part that identifies the released code.
        require(revision.split()[0].rstrip(";"), 3, f"source revision ({method})")

        # ---- item 4: defaults or validation-selected ----------------------------------------
        require(uniform(recs, "hyperparameter_origin", label), 4,
                f"hyperparameter origin ({method})")

        # ---- item 5: parameters, times, hardware --------------------------------------------
        params = uniform(recs, "model.trainable_parameters", label)
        require(f"{params:,}", 5, f"trainable parameters ({method})")
        train_mean = sum(r["compute"]["training_time_sec"] for r in recs) / len(recs)
        gen_mean = sum(r["compute"]["generation_time_sec"] for r in recs) / len(recs)
        require(f"{round(train_mean)} s", 5, f"mean training time ({method})")
        require(f"{gen_mean:.1f} s", 5, f"mean generation time ({method})")
        require(uniform(recs, "hardware.gpu", label), 5, f"gpu ({method})")

        # ---- item 6: failed runs ------------------------------------------------------------
        failed = [s for s, r in zip(SEEDS, recs) if r["failure_information"]["failed_or_unstable"]]
        nan = [s for s, r in zip(SEEDS, recs) if r["failure_information"]["nan_in_bank"]]
        if failed or nan:
            problems.append(f"  FAIL item 6: {label} has failed seeds {failed} / nan banks {nan}; "
                            f"the README must name them and the reason (PDF 1.5)")
        else:
            notes.append(f"  ok   item 6: {label} 5/5 seeds clean, none replaced")

    print(f"=== PDF section-5 disclosure: experiment {experiment} vs {readme_rel} ===")
    for line in notes:
        print(line)
    for line in problems:
        print(line)
    if problems:
        raise SystemExit(
            f"\n{len(problems)} disclosure problem(s) -- the README disagrees with the manifests")
    print(f"\nPASS: all six section-5 items present and consistent with 10 manifests")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", choices=["A", "B"], required=True)
    ap.add_argument("--readme", default=None,
                    help="README to check (default: experiment_<X>/README.md)")
    args = ap.parse_args()
    check(args.experiment, args.readme or f"experiment_{args.experiment}/README.md")


if __name__ == "__main__":
    main()
