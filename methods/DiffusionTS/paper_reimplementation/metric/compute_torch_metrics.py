"""
Paper-reimplementation metric driver (PyTorch half): Context-FID + Correlational.

Reproduces two of the four Diffusion-TS (Yuan & Qiao, ICLR 2024) paper metrics on the
Stocks (seq-len 24) reproduction, using the ORIGINAL repo metric code verbatim:
  - Context-FID   : Utils/context_fid.py        Context_FID     (TS2Vec Frechet distance)
  - Correlational : Utils/cross_correlation.py   CrossCorrelLoss (abs cross-corr error / 10)

Both metrics are computed on the [0,1]-normalized data (the same space the paper uses:
`ddpm_fake_stock.npy` and `stock_norm_truth_24_train.npy` both come out of
unnormalize_to_zero_to_one). Reported as mean +/- 95% CI (t-dist, dof=4) over N_ITER runs.

Run with the torch env:
  /home/tbasseras/gpu-venv/bin/python compute_torch_metrics.py

Writes: results/stocks_torch_metrics.json
"""
import os
import sys
import json
import numpy as np
import scipy.stats
import torch

# Make the reference repo importable (Utils/, Models/ live there)
REF = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "code", "reference"))
sys.path.insert(0, REF)

from Utils.context_fid import Context_FID
from Utils.cross_correlation import CrossCorrelLoss

N_ITER = 5
SEED = 123


def mean_ci(results):
    """mean +/- 95% CI, matching Utils/metric_utils.display_scores (t-dist, dof=5-1)."""
    mean = float(np.mean(results))
    sem = scipy.stats.sem(results)
    ci = float(sem * scipy.stats.t.ppf((1 + 0.95) / 2., 5 - 1))
    return mean, ci


def load_data():
    fake = np.load(os.path.join(REF, "OUTPUT", "stock", "ddpm_fake_stock.npy"))
    truth = np.load(os.path.join(REF, "OUTPUT", "stock", "samples", "stock_norm_truth_24_train.npy"))
    n = min(fake.shape[0], truth.shape[0])
    return truth[:n].astype(np.float32), fake[:n].astype(np.float32)


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    ori, fake = load_data()
    print(f"ori {ori.shape}  fake {fake.shape}", flush=True)

    # --- Context-FID (TS2Vec training + random permutation -> report over N_ITER) ---
    cfid = []
    for i in range(N_ITER):
        v = float(Context_FID(ori, fake[:ori.shape[0]]))
        cfid.append(v)
        print(f"[Context-FID] iter {i}: {v:.6f}", flush=True)
    cfid_mean, cfid_ci = mean_ci(cfid)

    # --- Correlational (deterministic given the two sets) ---
    corr = []
    x_real = torch.from_numpy(ori)
    for i in range(N_ITER):
        loss = CrossCorrelLoss(x_real, name="CrossCorrelLoss")
        s = float(loss.compute(torch.from_numpy(fake[:ori.shape[0]])).item())
        corr.append(s)
        print(f"[Correlational] iter {i}: {s:.6f}", flush=True)
    corr_mean, corr_ci = mean_ci(corr)

    out = {
        "context_fid": {"mean": cfid_mean, "ci95": cfid_ci, "runs": cfid},
        "correlational": {"mean": corr_mean, "ci95": corr_ci, "runs": corr},
        "n_iter": N_ITER,
        "n_samples": int(ori.shape[0]),
    }
    outdir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "stocks_torch_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
