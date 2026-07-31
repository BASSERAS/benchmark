# dataset/Heston/new_experiments/

Frozen data and vendored protocol code for Experiments A and B. **Nothing here is edited by
hand**, and nothing under `protocol/` is edited at all — PDF §7 item 7 requires the supplied
evaluator scripts to be run unchanged.

```
experiment_A/   train.npy test.npy disc.npy (+ *_sigma.npy, evaluator-only) + perfect_floor/
experiment_B/   train.npy test.npy disc.npy (+ *_labels.npy, oracle_*, evaluator-only) + perfect_floor/
protocol/       ⛔ VENDORED VERBATIM — NEVER EDIT
make_perfect_floor.py
```

**The guideline moved.** `guideline_new_experiment.md` now lives next to the deliverables it
describes, at [`results/new_experiments/guideline_new_experiment.md`](../../../results/new_experiments/guideline_new_experiment.md) —
the same directory as `experiment_A/LS4/`, `experiment_B/LS4/`, `README_TEMPLATE.md` and
`tools/`, which is where someone adding a new method will be looking.

**Information firewall (guideline §4).** A generator may read `train.npy`, and `disc.npy` for
model selection. It may read nothing else in this tree. `test.npy`, `*_sigma.npy`,
`*_labels.npy`, `oracle_*`, `oracle.joblib`, `gate_report.json` and `perfect_floor/` are
evaluator-side only.
