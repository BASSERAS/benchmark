"""
"Ours — Heston" paper-metric driver (PyTorch half): Context-FID + Correlational.

Applies TWO of Diffusion-TS's OWN four headline metrics (Yuan & Qiao, ICLR 2024, Table 1)
to our already-generated 5-seed Heston pool — no retraining. This produces the **Ours — Heston**
column of the paper-comparison table in results/Heston/DiffusionTS/README.md (GUIDELINE §15.2 §6),
the exact counterpart of the paper_reimplementation Stocks reproduction (compute_torch_metrics.py).

  - Context-FID   : Utils/context_fid.py        Context_FID     (TS2Vec Frechet distance, down)
  - Correlational : Utils/cross_correlation.py   CrossCorrelLoss (abs cross-corr error / 10, down)

Both metrics run in the paper's own [0,1] space: real + fake Heston PRICES are placed on [0,1]
by a single global MinMax fit on the real Heston prices (auto_norm equivalent for 1 feature),
exactly as the released pipeline normalises before scoring. One metric value per seed; reported
as mean +/- std across the 5 seeds (the benchmark's 5-seed convention), mirroring FourierFlow's
heston_paper_metrics.py.

Run with the torch env:
  CUDA_VISIBLE_DEVICES=<free_gpu> /home/tbasseras/gpu-venv/bin/python heston_paper_metrics_torch.py

Writes/updates: results/heston_paper_metrics.json  (context_fid + correlational keys)
"""
import os
import sys
import json
import numpy as np
import torch

REF = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "code", "reference"))
sys.path.insert(0, REF)

from Utils.context_fid import Context_FID
from Utils.cross_correlation import CrossCorrelLoss

METHOD = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REAL = os.path.join(METHOD, "..", "..", "dataset", "Heston", "heston_S_8192x128.npy")
GEN = os.path.join(METHOD, "generated_paths")
RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
SEED = 123
N_SEEDS = 5


def load01():
    """Real + 5 fake seed pools, all placed on the paper's [0,1] scale via one global
    MinMax fit on the real Heston prices. Returns (real01 (N,T,1), [fake01 ...], (lo,hi))."""
    S = np.load(os.path.abspath(REAL)).astype(np.float64)  # (8192,128) price
    lo, hi = float(S.min()), float(S.max())
    real01 = ((S - lo) / (hi - lo)).astype(np.float32)[:, :, None]
    fakes = []
    for i in range(N_SEEDS):
        Xg = np.load(os.path.join(GEN, f"seed_{i}", "generated_paths_8192x128.npy")).astype(np.float64)
        f01 = ((Xg - lo) / (hi - lo)).astype(np.float32)[:, :, None]
        fakes.append(f01)
    return real01, fakes, (lo, hi)


def mean_std(vals):
    return float(np.mean(vals)), float(np.std(vals))


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    real01, fakes, (lo, hi) = load01()
    n = real01.shape[0]
    print(f"real {real01.shape} lo={lo:.4f} hi={hi:.4f}", flush=True)

    cfid, corr = [], []
    x_real = torch.from_numpy(real01)
    for i, f in enumerate(fakes):
        fi = f[:n]
        v = float(Context_FID(real01, fi))
        cfid.append(v)
        loss = CrossCorrelLoss(x_real, name="CrossCorrelLoss")
        s = float(loss.compute(torch.from_numpy(fi)).item())
        corr.append(s)
        print(f"[seed {i}] context_fid={v:.6f}  correlational={s:.6f}", flush=True)

    cfid_m, cfid_s = mean_std(cfid)
    corr_m, corr_s = mean_std(corr)

    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, "heston_paper_metrics.json")
    out = {}
    if os.path.exists(out_path):
        with open(out_path) as fh:
            out = json.load(fh)
    out["scale_minmax"] = [lo, hi]
    out["n_seeds"] = N_SEEDS
    out["n_samples"] = int(n)
    out["context_fid"] = {"mean": cfid_m, "std": cfid_s, "seeds": cfid}
    out["correlational"] = {"mean": corr_m, "std": corr_s, "seeds": corr}
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({"context_fid": out["context_fid"],
                      "correlational": out["correlational"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
