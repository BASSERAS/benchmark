"""
Fourier Flow — paper reproduction on the Stocks dataset (ICLR 2021, Table 2).

Runs the *released* code verbatim (imported from ../../code/reference):
  - real_data_loading("stock", T)      -> GOOG open column, +0 prepend, length T+1
  - FourierFlow(hidden=200, n_flows=3, normalize=True, fft_size=T+1)
  - fit(epochs=1000, batch_size=500, lr=1e-3)   [full-batch; batch_size ignored by fit]
  - computeF1  (Sajjadi precision/recall support-overlap F-score)
  - computeMAE (TSTR: LSTM trained on synthetic, MAE on real next-step)

Target (paper Table 2, Fourier flow / Stocks): F-score 0.984, MAE 0.009.

Data X is loaded ONCE with a fixed numpy seed (identical dataset across all
experiments, as in the released driver). Each experiment k re-seeds torch so
model init + sampling vary, producing the 5-sample distribution for the 95% CI.

Usage:
    PYTHONPATH=<reference> python reproduce_stock.py --T 100 --n-exps 5 \
        --n-samples 10000 --epochs 1000 --out ../results/stock_repro.json
"""
from __future__ import absolute_import, division, print_function
import argparse, json, os, time
import numpy as np
import torch
import scipy.stats as st

torch.distributions.Distribution.set_default_validate_args(False)  # match torch 1.3 default

from run_experiment_2 import real_data_loading, FF_model_params, FF_train_params
from SequentialFlows import FourierFlow
from metrics.PRcurve import computeF1
from metrics.MAE import computeMAE


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2.0, n - 1)
    return float(m), float(h)


def main(args):
    t0 = time.time()
    # --- dataset: loaded ONCE, fixed seed for reproducible permutation ---
    np.random.seed(args.data_seed)
    X = real_data_loading("stock", args.T)
    Xarr = np.array(X)
    print(f"[data] stock X shape={Xarr.shape} min={Xarr.min():.4f} max={Xarr.max():.4f}", flush=True)

    mp = dict(FF_model_params["stock"])          # {hidden:200, n_flows:3, normalize:True}
    tp = dict(FF_train_params["stock"])          # {epochs:1000, batch_size:500, lr:1e-3, display_step:100}
    if args.epochs is not None:
        tp["epochs"] = args.epochs
    print(f"[cfg] model={mp} train={tp} fft_size={args.T + 1} n_samples={args.n_samples}", flush=True)

    F1s, MAEs, per_exp = [], [], []
    for k in range(args.exp_base, args.exp_base + args.n_exps):
        te = time.time()
        torch.manual_seed(1000 + k)
        np.random.seed(2000 + k)  # affects sampling / MAE LSTM batching
        model = FourierFlow(**mp, fft_size=args.T + 1)
        losses = model.fit(X, **tp)
        f1 = float(computeF1(X, model.sample(args.n_samples)))
        mae = float(computeMAE(X, model.sample(args.n_samples)))
        F1s.append(f1); MAEs.append(mae)
        rec = {"exp": k, "F1": f1, "MAE": mae,
               "loss_first": float(losses[0]), "loss_last": float(losses[-1]),
               "sec": round(time.time() - te, 1)}
        per_exp.append(rec)
        print(f"[exp {k}] F1={f1:.4f} MAE={mae:.4f} loss {losses[0]:.2f}->{losses[-1]:.2f} "
              f"({rec['sec']}s)", flush=True)
        # incremental save so progress is inspectable mid-run
        _dump(args.out, args, mp, tp, per_exp, F1s, MAEs, time.time() - t0, done=(k == args.n_exps - 1))

    f1_m, f1_h = mean_confidence_interval(F1s)
    mae_m, mae_h = mean_confidence_interval(MAEs)
    print(f"\n=== Fourier Flow / Stocks reproduction ===")
    print(f"F-score : {f1_m:.4f} +/- {f1_h:.4f}   (paper 0.984)")
    print(f"MAE     : {mae_m:.4f} +/- {mae_h:.4f}   (paper 0.009)")
    print(f"total {time.time() - t0:.1f}s")


def _dump(out, args, mp, tp, per_exp, F1s, MAEs, elapsed, done):
    payload = {
        "dataset": "stock", "T": args.T, "fft_size": args.T + 1,
        "n_samples": args.n_samples, "data_seed": args.data_seed,
        "model_params": mp, "train_params": tp,
        "paper_target": {"F_score": 0.984, "MAE": 0.009},
        "per_experiment": per_exp,
        "done": done, "elapsed_sec": round(elapsed, 1),
    }
    if len(F1s) > 1:
        payload["F_score_mean"], payload["F_score_ci95"] = mean_confidence_interval(F1s)
        payload["MAE_mean"], payload["MAE_ci95"] = mean_confidence_interval(MAEs)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--n-exps", type=int, default=5)
    p.add_argument("--exp-base", type=int, default=0)  # experiment index offset (for parallel runs)
    p.add_argument("--n-samples", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=None)  # None -> released 1000
    p.add_argument("--data-seed", type=int, default=42)
    p.add_argument("--out", type=str, default="../results/stock_repro.json")
    main(p.parse_args())
