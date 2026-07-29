"""
A/B metrics runner for the SBTS NO-PREPROCESSING baseline (§7.1 of the GUIDELINE).

Identical to ``SBTS/code/compute_metrics_logret.py`` in every computation — it
imports the untouched benchmark ``metrics/compute_all.py`` and only overrides
the path constants + the three data loaders. The ONLY difference here is that
the path roots point at ``baseline_no_preproc/`` (its own generated_paths /
results / plots) instead of the with-preproc ``SBTS/`` tree.

The baseline generator is the SBTS kernel run directly on globally-standardized
RAW prices (no log-return transform); same hyperparameters h=0.05/K=20/N_pi=50.

Scope: A1-A34 + B-curve + grid_tvd, seed 0 only. NOT path-shadowing — the
baseline is the head-to-head control for the preprocessing effect.

Usage:
  CUDA_VISIBLE_DEVICES=0 python compute_metrics_baseline.py --seeds 1
"""
import os
import sys
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))              # .../SBTS/baseline_no_preproc/code
BASE_DIR = os.path.dirname(HERE)                               # .../SBTS/baseline_no_preproc
METHOD_DIR = os.path.dirname(BASE_DIR)                         # .../SBTS
BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(METHOD_DIR))))
METRICS_DIR = os.path.join(BENCH_ROOT, "metrics")
DATA_DIR = os.path.join(BENCH_ROOT, "dataset", "Heston", "preprocessing_with_log_returns")

N = 4096

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=1, help="number of seeds (baseline: 1)")
args = ap.parse_args()
sys.argv = [sys.argv[0]]

sys.path.insert(0, METRICS_DIR)
import compute_all as C   # noqa: E402

# --- redirect all paths to the no-preproc baseline ---
C.DATASET_DIR = DATA_DIR
C.GENERATED_DIR = os.path.join(BASE_DIR, "generated_paths")
C.RESULTS_DIR = BASE_DIR
C.PLOTS_DIR = os.path.join(BASE_DIR, "plots")
C.N_SEEDS = args.seeds
os.makedirs(C.RESULTS_DIR, exist_ok=True)
os.makedirs(C.PLOTS_DIR, exist_ok=True)


def load_data():
    S = np.load(os.path.join(DATA_DIR, f"heston_S_test_{N}x128.npy"))
    v = np.load(os.path.join(DATA_DIR, f"heston_v_test_{N}x128.npy"))
    return S, v


def load_disc():
    return np.load(os.path.join(DATA_DIR, f"heston_S_disc_{N}x128.npy"))


def load_generated(seed):
    path = os.path.join(C.GENERATED_DIR, f"seed_{seed}", f"generated_paths_{N}x128.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing: {path}. Run generate_sbts_logret.py --raw "
            f"--out .../baseline_no_preproc --seed {seed} first.")
    return np.load(path)


C.load_data = load_data
C.load_disc = load_disc
C.load_generated = load_generated

if __name__ == "__main__":
    C.main()
