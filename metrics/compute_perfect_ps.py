"""Path-Shadowing Monte-Carlo (PS-MC) *Perfect floor* for the CRPS table.

The A- and B-metric "Perfect" columns treat a fresh, INDEPENDENT true-Heston
draw as the "generated" side and score it against the held-out TEST set. This
script does the exact same thing for the PS-MC CRPS metric: it uses the
independent-draw perfect-recovery paths (seed 1000+i, materialised under
methods/perfect_recovery/dataset/seeds/) as the *shadow database*, and the TEST
set (seed 1) as the query prefixes + true futures — identical in every knob
(PREFIX_LEN=64, K=77, horizons h32/h64) to each method's path_shadowing/run_eval.

The result is the non-zero finite-sample CRPS floor a perfect generator attains,
identical across all methods. It is written to

    methods/perfect_recovery/results/path_shadowing/summary.json

with the same schema as every method's path_shadowing/summary.json, so
render_tables.py can read it for the PS-MC "Perfect" column.

Usage (CPU/numpy only — no GPU):
    OMP_NUM_THREADS=8 /home/tbasseras/gpu-venv/bin/python metrics/compute_perfect_ps.py
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np

REPO       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# path_shadowing.py is byte-identical across methods (md5 verified); reuse LS4's.
PS_DIR     = os.path.join(REPO, "methods", "LS4", "path_shadowing")
SEEDS_DIR  = os.path.join(REPO, "methods", "perfect_recovery", "dataset", "seeds")
OUT_DIR    = os.path.join(REPO, "methods", "perfect_recovery", "results", "path_shadowing")
TEST_PATH  = os.path.join(REPO, "dataset", "Heston", "heston_S_test_8192x128.npy")

PREFIX_LEN = 64
K          = 77
HORIZONS   = {"h32": (0, 32), "h64": (0, 64)}
BATCH_N    = 512
N_SEEDS    = 5


def main() -> None:
    sys.path.insert(0, PS_DIR)
    from path_shadowing import (
        ps_mc_retrieve, uniform_weights, gaussian_weights,
        evaluate_horizon, naive_baseline,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    X_real = np.load(TEST_PATH)                     # (N, T) test set = query prefixes
    N, T   = X_real.shape
    y_fut  = X_real[:, PREFIX_LEN:]
    baseline = naive_baseline(X_real, prefix_len=PREFIX_LEN)
    print(f"Test data: {X_real.shape}  prefix={PREFIX_LEN}  K={K}", flush=True)
    print(f"Naive RW baseline: {baseline}", flush=True)

    all_results = []
    for seed in range(N_SEEDS):
        t0 = time.time()
        X_fake = np.load(os.path.join(SEEDS_DIR,
                                      f"heston_S_independent_seed{seed}.npy"))
        ensemble, distances, _, real_emb_norms = ps_mc_retrieve(
            X_real, X_fake, prefix_len=PREFIX_LEN, K=K
        )
        w_unif = uniform_weights(N, K)
        median_dist = float(np.median(distances))
        median_norm = float(np.median(real_emb_norms)) + 1e-30
        w_gauss, eta_val = gaussian_weights(
            distances, real_emb_norms=real_emb_norms,
            eta_tilde=median_dist / median_norm,
        )
        seed_res = {"seed": seed, "eta": float(eta_val)}
        for h_name, (h_start, h_end) in HORIZONS.items():
            metrics = evaluate_horizon(ensemble, y_fut, w_unif, w_gauss,
                                       h_start=h_start, h_end=h_end,
                                       batch_n=BATCH_N)
            for mk, mv in metrics.items():
                seed_res[f"{h_name}_{mk}"] = mv
        all_results.append(seed_res)
        print(f"  [seed {seed}] {time.time()-t0:.1f}s  "
              f"CRPS-h32={seed_res['h32_CRPS_uniform']:.4f}  "
              f"CRPS-h64={seed_res['h64_CRPS_uniform']:.4f}", flush=True)

    scalar_keys = [k for k in all_results[0] if k not in ("seed", "eta")]
    summary = {"baseline": baseline}
    for k in scalar_keys:
        vals = np.array([r[k] for r in all_results])
        summary[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
    summary["per_seed"] = all_results
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {OUT_DIR}/summary.json", flush=True)
    print(f"  Perfect CRPS h32 = {summary['h32_CRPS_uniform']['mean']:.4f} "
          f"+/- {summary['h32_CRPS_uniform']['std']:.4f}")
    print(f"  Perfect CRPS h64 = {summary['h64_CRPS_uniform']['mean']:.4f} "
          f"+/- {summary['h64_CRPS_uniform']['std']:.4f}")


if __name__ == "__main__":
    main()
