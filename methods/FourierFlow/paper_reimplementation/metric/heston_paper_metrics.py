"""
Fourier Flow — paper's OWN metrics (Sajjadi F-score + TSTR LSTM MAE) applied to
our Heston reimplementation.

This mirrors metric/reproduce_stock.py but swaps the dataset: instead of the
paper's Stocks windows we score the 5-seed Fourier-Flow paths generated on the
Heston benchmark dataset. The metric code is the released code, unchanged:
  - metrics.PRcurve.computeF1   (Sajjadi precision/recall support-overlap F1)
  - metrics.MAE.computeMAE      (TSTR: 2-layer LSTM/100u trained on synthetic,
                                 MAE on real next-step)

Only adaptation: the paper hardcodes MAX_STEPS=100 (Stocks T=100 -> sliced 100).
Heston paths have length 128 -> sliced 127, so we replicate computeMAE with
RNNmodel(MAX_STEPS=127). This is a data-length adaptation, NOT a metric change.

Both real and synthetic are placed on the paper's [0,1] scale using a single
global MinMax fit on the REAL Heston prices (paper normalises data to [0,1] via
MinMaxScaler before scoring; synthetic is mapped by the same scaler).

Usage:
    PYTHONPATH=<code/reference> python heston_paper_metrics.py \
        --real ../../../../dataset/heston_S_8192x128.npy \
        --gen-glob '../../../generated_paths/seed_*/generated_paths_8192x128.npy' \
        --out ../results/heston_paper_metrics.json
"""
from __future__ import absolute_import, division, print_function
import argparse, glob, json, os, time
import numpy as np
import torch

torch.distributions.Distribution.set_default_validate_args(False)

from metrics.PRcurve import computeF1
from models.sequential import RNNmodel


def minmax_fit(a):
    lo, hi = float(np.min(a)), float(np.max(a))
    return lo, hi


def minmax_apply(a, lo, hi):
    return (a - lo) / (hi - lo + 1e-7)


def compute_mae_heston(X_real, X_synth, max_steps):
    """Replica of metrics.MAE.computeMAE with MAX_STEPS set to the Heston
    sliced length (127) instead of the Stocks-hardcoded 100."""
    # train predictive LSTM on synthetic next-step (TSTR)
    model = RNNmodel(mode="LSTM", HIDDEN_UNITS=100, NUM_LAYERS=2,
                     MAX_STEPS=max_steps, INPUT_SIZE=1)
    Xs = [X_synth[k][: len(X_synth[k]) - 1] for k in range(len(X_synth))]
    Ys = [X_synth[k][1:] for k in range(len(X_synth))]
    model.fit(Xs, Ys, verbosity=False)
    # evaluate MAE on real next-step
    Xt = [X_real[k][: len(X_real[k]) - 1] for k in range(len(X_real))]
    Yt = [X_real[k][1:] for k in range(len(X_real))]
    Xp = model.predict(Xt)
    return float(np.mean(np.abs(np.array(Xp) - np.squeeze(np.array(Yt), axis=2))))


def to_seq_list(arr2d):
    # (N, L) -> list of (L, 1) float arrays
    return [arr2d[k].reshape(-1, 1).astype(np.float64) for k in range(arr2d.shape[0])]


def main(args):
    t0 = time.time()
    real = np.load(args.real)                    # (N, L) price paths
    lo, hi = minmax_fit(real)
    real_n = minmax_apply(real, lo, hi)
    L = real.shape[1]
    print(f"[data] real {real.shape} minmax=({lo:.3f},{hi:.3f}) -> [0,1]", flush=True)

    gens = sorted(glob.glob(args.gen_glob))
    assert gens, f"no generated paths matched {args.gen_glob}"
    X_real_full = to_seq_list(real_n)

    F1s, MAEs, per_seed = [], [], []
    for i, gp in enumerate(gens):
        te = time.time()
        torch.manual_seed(1000 + i)
        np.random.seed(2000 + i)
        gen = np.load(gp)                        # (N, L) original price scale
        gen_n = minmax_apply(gen, lo, hi)        # same scaler as real
        X_synth_full = to_seq_list(gen_n)

        f1 = float(computeF1(X_real_full, X_synth_full))
        mae = compute_mae_heston(X_real_full, X_synth_full, max_steps=L - 1)
        F1s.append(f1); MAEs.append(mae)
        rec = {"seed": i, "path": gp, "F_score": f1, "MAE": mae,
               "sec": round(time.time() - te, 1)}
        per_seed.append(rec)
        print(f"[seed {i}] F-score={f1:.4f} MAE={mae:.4f} ({rec['sec']}s)", flush=True)

    payload = {
        "dataset": "Heston", "seq_len": int(L), "max_steps_mae": int(L - 1),
        "scaler": {"min": lo, "max": hi, "note": "global MinMax fit on real Heston prices, applied to both"},
        "metric_source": "released FF code: metrics.PRcurve.computeF1, models.sequential.RNNmodel (LSTM/100u/2L)",
        "paper_target_stocks": {"F_score": 0.984, "MAE": 0.009},
        "per_seed": per_seed,
        "F_score_mean": float(np.mean(F1s)), "F_score_std": float(np.std(F1s)),
        "MAE_mean": float(np.mean(MAEs)), "MAE_std": float(np.std(MAEs)),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n=== Fourier Flow / Heston — paper metrics ===")
    print(f"F-score : {payload['F_score_mean']:.4f} +/- {payload['F_score_std']:.4f}")
    print(f"MAE     : {payload['MAE_mean']:.4f} +/- {payload['MAE_std']:.4f}")
    print(f"total {payload['elapsed_sec']}s -> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--real", type=str, required=True)
    p.add_argument("--gen-glob", type=str, required=True)
    p.add_argument("--out", type=str, default="../results/heston_paper_metrics.json")
    main(p.parse_args())
