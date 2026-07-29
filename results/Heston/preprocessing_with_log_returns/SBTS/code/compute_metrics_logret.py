"""
Parallel metrics runner for the SBTS log-return-preprocessing experiment.

Reuses the benchmark's canonical metric code WITHOUT editing it: imports
``metrics/compute_all.py`` and overrides only its path constants and its three
data loaders so they point at the 4096-path log-return datasets and the
generated paths under results/Heston/preprocessing_with_log_returns/SBTS/.

Every A1-A34 + B curve + grid_tvd computation, and the PCA/t-SNE plotting, is
the untouched ``compute_all`` implementation, so numbers are directly comparable
to the other methods in this folder.

Usage:
  CUDA_VISIBLE_DEVICES=0 python compute_metrics_logret.py --seeds 1     # gate: seed 0 only
  CUDA_VISIBLE_DEVICES=0 python compute_metrics_logret.py --seeds 5     # all seeds
"""
import os
import sys
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))                       # .../SBTS/code
METHOD_DIR = os.path.dirname(HERE)                                      # .../SBTS
BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(METHOD_DIR))))
METRICS_DIR = os.path.join(BENCH_ROOT, "metrics")
DATA_DIR = os.path.join(BENCH_ROOT, "dataset", "Heston", "preprocessing_with_log_returns")

N = 4096  # path count for this experiment

# parse OUR args, then hide them so compute_all's import-time argparse sees nothing
ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=1, help="number of seeds (gate: 1)")
args = ap.parse_args()
sys.argv = [sys.argv[0]]

sys.path.insert(0, METRICS_DIR)
import compute_all as C   # noqa: E402  (import-time argparse now sees no flags)

# --- redirect all paths to the log-return experiment ---
C.DATASET_DIR = DATA_DIR
C.GENERATED_DIR = os.path.join(METHOD_DIR, "generated_paths")
C.RESULTS_DIR = METHOD_DIR
C.PLOTS_DIR = os.path.join(METHOD_DIR, "plots")
C.N_SEEDS = args.seeds
os.makedirs(C.RESULTS_DIR, exist_ok=True)
os.makedirs(C.PLOTS_DIR, exist_ok=True)


# --- override the three loaders to the 4096-path files ---
def load_data():
    S = np.load(os.path.join(DATA_DIR, f"heston_S_test_{N}x128.npy"))
    v = np.load(os.path.join(DATA_DIR, f"heston_v_test_{N}x128.npy"))
    return S, v


def load_disc():
    return np.load(os.path.join(DATA_DIR, f"heston_S_disc_{N}x128.npy"))


def load_generated(seed):
    path = os.path.join(C.GENERATED_DIR, f"seed_{seed}", f"generated_paths_{N}x128.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}. Run generate_sbts_logret.py --seed {seed} first.")
    return np.load(path)


C.load_data = load_data
C.load_disc = load_disc
C.load_generated = load_generated

if __name__ == "__main__":
    C.main()
