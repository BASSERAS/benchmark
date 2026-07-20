"""
3-way Heston architecture smoke-test scorer.

Ranks the three candidate Diffusion-TS configs (stocks / mujoco / etth) by how
well their 3000-step smoke samples match the real Heston paths, so we can pick
ONE architecture for the full 5-seed run.

Decisive metric: Context-FID (the paper's own headline fidelity metric, TS2Vec
Frechet distance -- lower = better). Reported alongside three cheap sanity
proxies (all lower = better): marginal-mean error, marginal-std error, lag-1
autocorrelation error.

Both real and fake paths are min-max normalised with the SAME (real) lo/hi to
[0,1] before scoring, so the comparison lives in the model's own output space.

Run:
  CUDA_VISIBLE_DEVICES=1 /home/tbasseras/gpu-venv/bin/python compare_smoke.py
"""
import os
import sys
import json
import numpy as np

REF = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "code", "reference"))
sys.path.insert(0, REF)

from Utils.context_fid import Context_FID

METHOD = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REAL = os.path.join(METHOD, "..", "..", "dataset", "Heston", "heston_S_8192x128.npy")
GEN_DIR = os.path.join(METHOD, "generated_paths", "seed_0")
ARCHES = ["stocks", "mujoco", "etth"]


def acf1(X):
    """Mean lag-1 autocorrelation across paths. X: (N, T)."""
    Xc = X - X.mean(axis=1, keepdims=True)
    num = (Xc[:, 1:] * Xc[:, :-1]).sum(axis=1)
    den = (Xc * Xc).sum(axis=1)
    den = np.where(den == 0, 1e-12, den)
    return float(np.mean(num / den))


def score(real01, fake01):
    """real01, fake01: (N, T) in [0,1]. Returns dict of errors (lower=better)."""
    n = min(real01.shape[0], fake01.shape[0])
    r = real01[:n]
    f = fake01[:n]
    r3 = r[:, :, None].astype(np.float32)
    f3 = f[:, :, None].astype(np.float32)
    cfid = float(Context_FID(r3, f3))
    return {
        "context_fid": cfid,
        "marg_mean_err": float(abs(r.mean() - f.mean())),
        "marg_std_err": float(abs(r.std() - f.std())),
        "acf1_err": float(abs(acf1(r) - acf1(f))),
    }


def main():
    S = np.load(os.path.abspath(REAL)).astype(np.float64)  # (8192,128) price
    lo, hi = float(S.min()), float(S.max())
    real01 = (S - lo) / (hi - lo)

    results = {}
    for arch in ARCHES:
        path = os.path.join(GEN_DIR, f"smoke_{arch}_generated_paths_8192x128.npy")
        if not os.path.exists(path):
            print(f"[skip] {arch}: {path} missing", flush=True)
            continue
        Xg = np.load(path).astype(np.float64)              # (N,128) price
        fake01 = (Xg - lo) / (hi - lo)                     # normalise with REAL lo/hi
        s = score(real01, fake01)
        s["gen_has_nan"] = bool(not np.isfinite(Xg).all())
        results[arch] = s
        print(f"[{arch}] {json.dumps(s)}", flush=True)

    ranked = sorted(results.items(), key=lambda kv: kv[1]["context_fid"])
    winner = ranked[0][0] if ranked else None
    out = {
        "real_minmax": [lo, hi],
        "per_arch": results,
        "ranking_by_context_fid": [k for k, _ in ranked],
        "_winner_by_context_fid": winner,
    }
    outdir = os.path.join(METHOD, "paper_reimplementation", "results")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "smoke_comparison.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
