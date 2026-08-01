"""Benchmark metric runner (A1-A34 + B curve) for a blinded-protocol experiment.

Reuses the benchmark's canonical metric code WITHOUT editing it: imports
``metrics/compute_all.py`` and overrides only its path constants and its three
data loaders, so every A/B number is produced by the untouched implementation
and stays directly comparable to the main-benchmark tables.

Two differences versus the original Heston runner:

* **A33/A34 are dropped.** Both require the Heston teacher variance ``v``, which
  neither Experiment A nor Experiment B defines. ``load_data`` returns ``v=None``
  and ``compute_all``'s existing try/except records them as ``null``.
* **The perfect floor goes through this same path.** ``--source floor`` swaps the
  generated side for an independent true-DGP draw
  (``perfect_floor/floor_seed{1000+i}.npy``) so the floor and every model are
  scored by literally the same code, as the protocol requires.

Reference split, unchanged from ``compute_all``:
  * "real" reference for A1-A17 / A20-A32 / B = ``test.npy``  (protocol seed 1)
  * A18 / A19 judge "real" class              = ``disc.npy``  (protocol seed 2)

Usage:
  CUDA_VISIBLE_DEVICES=1 python compute_metrics_experiment.py --experiment A --source model --seeds 5
"""
import os
import sys
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))                   # .../TimeDiT/code
METHOD_DIR = os.path.dirname(HERE)                                  # .../TimeDiT
EXPERIMENT_DIR = os.path.dirname(METHOD_DIR)                        # .../experiment_X
BENCH_ROOT = os.path.abspath(os.path.join(EXPERIMENT_DIR, "..", "..", ".."))
METRICS_DIR = os.path.join(BENCH_ROOT, "metrics")

ap = argparse.ArgumentParser()
ap.add_argument("--experiment", default="A", choices=("A", "B"))
ap.add_argument("--source", default="model", choices=("model", "floor"),
                help="model = this method's bank; floor = independent true-DGP draw")
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--floor-base-seed", type=int, default=1000)
args = ap.parse_args()
sys.argv = [sys.argv[0]]   # hide our flags from compute_all's import-time argparse

DATA_DIR = os.path.join(BENCH_ROOT, "dataset", "Heston", "new_experiments",
                        f"experiment_{args.experiment}")

sys.path.insert(0, METRICS_DIR)
import compute_all as C   # noqa: E402

C.DATASET_DIR = DATA_DIR
C.N_SEEDS = args.seeds
if args.source == "model":
    C.GENERATED_DIR = os.path.join(METHOD_DIR, "generated_paths")
    C.RESULTS_DIR = METHOD_DIR
else:
    C.GENERATED_DIR = os.path.join(DATA_DIR, "perfect_floor")
    C.RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "perfect_floor")
C.PLOTS_DIR = os.path.join(C.RESULTS_DIR, "plots")
os.makedirs(C.RESULTS_DIR, exist_ok=True)
os.makedirs(C.PLOTS_DIR, exist_ok=True)


def load_data():
    # v = None -> A33/A34 raise inside compute_all's try/except and land as null.
    return np.load(os.path.join(DATA_DIR, "test.npy")), None


def load_disc():
    return np.load(os.path.join(DATA_DIR, "disc.npy"))


def load_generated(seed):
    if args.source == "model":
        path = os.path.join(C.GENERATED_DIR, f"seed_{seed}",
                            "generated_paths_8192x128.npy")
        hint = f"Run train_csdi_experiment.py --seed {seed} first."
    else:
        path = os.path.join(C.GENERATED_DIR,
                            f"floor_seed{args.floor_base_seed + seed}.npy")
        hint = f"Run make_perfect_floor.py --experiment {args.experiment} first."
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}. {hint}")
    return np.load(path)


C.load_data = load_data
C.load_disc = load_disc
C.load_generated = load_generated

if __name__ == "__main__":
    C.main()
