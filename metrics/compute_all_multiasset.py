"""
compute_all_multiasset.py — A1-A34 + B curve metrics for MULTI-ASSET datasets.

Multi-asset sibling of ``compute_all.py``. That file is left untouched; every
number already committed under ``results/<1-D dataset>/`` stays reproducible by
the exact script that produced it.

Aggregation contract
--------------------
The benchmark's metric suite is written for a single series. With d assets there
are two honest ways to score, and this driver uses BOTH, per metric:

  PER-ASSET  the metric is evaluated d times on (N, T, 1) slices, one per asset,
             and the d values are AVERAGED. The averaged value is stored under
             the SAME key the 1-D driver uses, so the README tables line up
             column-for-column with the single-asset ones. The d individual
             values are kept under ``per_asset`` so nothing is lost.

  NATIVE     the metric is evaluated ONCE on the full (N, T, d) tensor, because
             it already has a genuine multivariate definition and collapsing it
             to per-asset averages would DESTROY the cross-asset information
             that is the entire point of a multi-asset dataset:

               A6-A11  MMD^2 / sliced-Wasserstein — the kernel and the random
                       projections act on the flattened (T*d) vector, so they
                       see the joint law across assets.
               A18     discriminative score — the GRU reads d input channels and
                       the MLP flattens (T-1)*d, so the classifier can exploit
                       a wrong cross-asset correlation to separate real vs fake.
                       Per-asset scoring would hide exactly that failure.
               A20     terminal covariance error — at d = 8 this is the Frobenius
                       norm of the 8x8 cross-asset covariance difference. This is
                       THE metric that tests the multi-asset structure; per-asset
                       it degenerates to 8 scalar variance errors.
               A25     terminal mean RMSE — the L2 norm of the d-vector of
                       terminal-mean errors.

             This split is the user's, verbatim: "compute them for each asset
             then do the average ... except for A6-A11, A18, A20, A25, that u
             can adapt to multi dimension".

A19 is deliberately NOT in the native list. It was not in the user's exclusion
list, and per-asset is also the semantically faithful choice: the multivariate
``_train_gru`` predicts only ``data[:, 1:, :1]``, i.e. asset 0 alone, so a native
d = 8 run would silently report a one-asset score under a multi-asset name.

B curve metrics and grid_tvd are per-asset-then-averaged (the stylised-fact
curves are univariate objects). No plots are produced.

Reference splits (same roles as the 1-D driver)
    test (seed 1)  real reference for A1-A17, A19-A34 and B
    disc (seed 2)  'real' class for the A18 classifier and the A19 TSTR test set
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

METRICS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(METRICS_DIR)
sys.path.insert(0, METRICS_DIR)

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--method", default="SBTS")
_p.add_argument("--dataset", default="HestonMultiAsset")
_p.add_argument("--gen-root", default=None,
                help="directory holding generated_paths/. Defaults to the "
                     "self-contained results/<dataset>/<method>/ tree.")
_p.add_argument("--seeds", type=int, default=5)
_p.add_argument("--dt", type=float, default=1.0 / 252.0)
_p.add_argument("--n-sub", type=int, default=1024)
_p.add_argument("--pred-steps", type=int, default=5000)
_p.add_argument("--disc-steps", type=int, default=2000)
_cli, _ = _p.parse_known_args()

METHOD, DATASET, N_SEEDS, DT = _cli.method, _cli.dataset, _cli.seeds, _cli.dt

DATASET_DIR = os.path.join(REPO_ROOT, "dataset", DATASET)
# Multi-asset results are self-contained: results/<dataset>/<method>/ holds
# code/, generated_paths/, losses/, weights/ and the README side by side, so
# inputs and outputs live in one tree instead of being split across methods/.
RESULTS_DIR = os.path.join(REPO_ROOT, "results", DATASET, METHOD)
GENERATED_DIR = os.path.join(_cli.gen_root or RESULTS_DIR, "generated_paths")
os.makedirs(RESULTS_DIR, exist_ok=True)

import torch  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

from metrics import (  # noqa: E402
    mmd2, terminal_mmd2, increment_mmd2, volatility_mmd, terminal_swd, path_swd,
    terminal_cov_error, terminal_mean_rmse, return_std_error, return_kurtosis_error,
    acf_error, teacher_sigma_metrics, logreturn_std_error, abs_return_quantile_error,
    kurtosis_ratio, sigma_mean_error, acf_lag1_abs_error, acf_lag1_sq_error,
    rv_law_loss, mean_path_rmse, vol_path_rmse, ks_logreturns, skewness_error,
    qq_rmse, tail_qq_error, rolling_vol_ks, vol_of_vol_error, terminal_ks,
    hill_tail_index_error, compute_curve_metrics, aggregate_curve_metrics,
    paths_to_points, grid_tvd, GRID_TVD_DEFAULT_BINS,
)
from discriminative_score import compute_discriminative_score  # noqa: E402
from predictive_score import compute_predictive_score  # noqa: E402

# Keys computed ONCE on the full (N, T, d) tensor — see module docstring.
NATIVE_KEYS = [
    "A6_path_mmd2", "A7_terminal_mmd2", "A8_increment_mmd2", "A9_volatility_mmd",
    "A10_terminal_swd", "A11_path_swd",
    "A18_disc_score_gru", "A18_disc_score_mlp",
    "A20_cov_error", "A25_mean_rmse",
]


def _load(name):
    path = os.path.join(DATASET_DIR, name)
    a = np.load(path)
    if a.ndim != 3:
        raise ValueError(f"{path}: expected (N, T, d), got {a.shape}")
    return a


def load_generated(seed):
    d = os.path.join(GENERATED_DIR, f"seed_{seed}")
    hits = [f for f in os.listdir(d) if f.startswith("generated_paths") and f.endswith(".npy")]
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one generated_paths*.npy in {d}, found {hits}")
    a = np.load(os.path.join(d, hits[0]))
    if a.ndim != 3:
        raise ValueError(f"{hits[0]}: expected (N, T, d), got {a.shape}")
    return a


def per_asset_metrics(real, fake, v_real, j):
    """All PER-ASSET metrics for asset j. real/fake/v_real are (N, T, d)."""
    r3 = real[:, :, j:j + 1]
    f3 = fake[:, :, j:j + 1]
    lr_r = np.diff(np.log(r3), axis=1)
    lr_f = np.diff(np.log(f3), axis=1)
    out = {
        # Fat tail A1-A5
        "A1_kurtosis_error": float(return_kurtosis_error(r3, f3)),
        "A2_abs_r_q95_error": float(abs_return_quantile_error(r3, f3, q=0.95)),
        "A3_abs_r_q99_error": float(abs_return_quantile_error(r3, f3, q=0.99)),
        "A4_tail_qq_error": float(tail_qq_error(r3, f3)),
        "A5_hill_tail_index_error": float(hill_tail_index_error(r3, f3)),
        # Distribution A12-A17  (A6-A11 are native)
        "A12_rv_law_loss": float(rv_law_loss(r3, f3)),
        "A13_mean_path_rmse": float(mean_path_rmse(r3, f3)),
        "A14_ks_logreturns": float(ks_logreturns(r3, f3)),
        "A15_skewness_error": float(skewness_error(r3, f3)),
        "A16_qq_rmse": float(qq_rmse(r3, f3)),
        "A17_terminal_ks": float(terminal_ks(r3, f3)),
        # Temporal A21-A24  (A20 is native)
        "A21_acf_abs": float(acf_error(np.abs(lr_f), np.abs(lr_r))),
        "A22_acf_sq": float(acf_error(lr_f ** 2, lr_r ** 2)),
        "A23_acf_lag1_abs_error": float(acf_lag1_abs_error(r3, f3)),
        "A24_acf_lag1_sq_error": float(acf_lag1_sq_error(r3, f3)),
        # Vol A26-A32  (A25 is native)
        "A26_std_error": float(return_std_error(r3, f3)),
        "A27_logreturn_std_error": float(logreturn_std_error(r3, f3)),
        "A28_kurtosis_ratio": float(kurtosis_ratio(r3, f3)),
        "A29_sigma_mean_error": float(sigma_mean_error(r3, f3)),
        "A30_vol_path_rmse": float(vol_path_rmse(r3, f3)),
        "A31_rolling_vol_ks": float(rolling_vol_ks(r3, f3)),
        "A32_vol_of_vol_error": float(vol_of_vol_error(r3, f3)),
    }
    # Heston spec A33-A34 — latent variance of asset j, dt from the dataset (1/252),
    # NOT the function's 1/250 default which belongs to the single-asset dataset.
    corr_ts, rmse_ts = teacher_sigma_metrics(f3, v_real[:, :, j], dt=DT)
    out["A33_sigma_corr"] = float(corr_ts)
    out["A34_sigma_rmse"] = float(rmse_ts)
    # B curve metrics + grid_tvd on the univariate (N, T) slice
    out.update({k: float(v) for k, v in
                compute_curve_metrics(real[:, :, j], fake[:, :, j]).items()})
    gtvd, _, _, _ = grid_tvd(paths_to_points(real[:, :, j]),
                             paths_to_points(fake[:, :, j]),
                             n_bins=GRID_TVD_DEFAULT_BINS)
    out["grid_tvd"] = float(gtvd)
    return out


def native_metrics(seed, real, fake, disc):
    """Metrics evaluated ONCE on the full (N, T, d) tensor."""
    N_sub = _cli.n_sub
    rng = np.random.default_rng(seed)
    R = real[rng.choice(len(real), N_sub, replace=False)]
    F = fake[rng.choice(len(fake), N_sub, replace=False)]

    out = {
        "A6_path_mmd2": float(mmd2(R, F)),
        "A7_terminal_mmd2": float(terminal_mmd2(R, F)),
        "A8_increment_mmd2": float(increment_mmd2(R, F)),
        "A9_volatility_mmd": float(volatility_mmd(R, F)),
        "A10_terminal_swd": float(terminal_swd(R, F)),
        "A11_path_swd": float(path_swd(R, F)),
        "A20_cov_error": float(terminal_cov_error(real, fake)),
        "A25_mean_rmse": float(terminal_mean_rmse(real, fake)),
    }
    for k in ("A6_path_mmd2", "A7_terminal_mmd2", "A8_increment_mmd2",
              "A9_volatility_mmd", "A10_terminal_swd", "A11_path_swd",
              "A20_cov_error", "A25_mean_rmse"):
        print(f"    {k:24s} {out[k]:.6f}", flush=True)

    lr_disc = np.diff(np.log(disc), axis=1)     # (N, T-1, d) — judge set, seed 2
    lr_fake = np.diff(np.log(fake), axis=1)

    print("    A18 discriminative (GRU + MLP, d-channel) ...", flush=True)
    d18 = compute_discriminative_score(lr_disc, lr_fake, n_steps=_cli.disc_steps,
                                       device=DEVICE)
    out["A18_disc_score_gru"] = d18["disc_score_gru"]
    out["A18_disc_score_mlp"] = d18["disc_score_mlp"]
    print(f"    A18  GRU={d18['disc_score_gru']:.6f}  MLP={d18['disc_score_mlp']:.6f}",
          flush=True)
    for arch in ("gru", "mlp"):
        with open(os.path.join(RESULTS_DIR, f"seed_{seed}_disc_{arch}_loss.csv"),
                  "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["step", "train_bce"])
            w.writeheader()
            w.writerows(d18[f"loss_history_{arch}"])
    return out


def predictive_per_asset(seed, disc, fake):
    """A19 TSTR, one univariate run per asset, then averaged."""
    d = fake.shape[2]
    lr_disc = np.diff(np.log(disc), axis=1)
    lr_fake = np.diff(np.log(fake), axis=1)
    gru, mlp, hist = [], [], {}
    for j in range(d):
        t0 = time.perf_counter()
        r = compute_predictive_score(lr_disc[:, :, j], lr_fake[:, :, j],
                                     n_steps=_cli.pred_steps, device=DEVICE)
        gru.append(r["pred_score_gru"])
        mlp.append(r["pred_score_mlp"])
        if j == 0:
            hist = {a: r[f"loss_history_{a}"] for a in ("gru", "mlp")}
        print(f"    A19 asset {j}: GRU={r['pred_score_gru']:.6f} "
              f"MLP={r['pred_score_mlp']:.6f}  ({time.perf_counter()-t0:.0f}s)", flush=True)
    # asset-0 loss curves are the ones archived, mirroring the 1-D driver's CSVs
    for arch in ("gru", "mlp"):
        with open(os.path.join(RESULTS_DIR, f"seed_{seed}_pred_{arch}_loss.csv"),
                  "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["step", "train_mae"])
            w.writeheader()
            w.writerows(hist[arch])
    return {"A19_pred_score_gru": gru, "A19_pred_score_mlp": mlp}


def compute_metrics_for_seed(seed, real, v_real, disc):
    print(f"\n=== Seed {seed} ===", flush=True)
    fake = load_generated(seed)
    if fake.shape != real.shape:
        raise ValueError(f"shape mismatch: generated {fake.shape} vs real {real.shape}")
    if fake.min() <= 0:
        raise ValueError(f"generated paths contain non-positive prices (min={fake.min()})")
    d = real.shape[2]
    t0 = time.perf_counter()

    print(f"  native (N, T, d) metrics, d={d}", flush=True)
    native = native_metrics(seed, real, fake, disc)

    print("  per-asset metrics", flush=True)
    per = {}
    for j in range(d):
        tj = time.perf_counter()
        mj = per_asset_metrics(real, fake, v_real, j)
        for k, v in mj.items():
            per.setdefault(k, []).append(v)
        print(f"    asset {j}: {len(mj)} metrics ({time.perf_counter()-tj:.0f}s)", flush=True)

    print("  A19 predictive TSTR, per asset", flush=True)
    per.update(predictive_per_asset(seed, disc, fake))

    results = {"seed": seed, "d": d, "native_keys": NATIVE_KEYS}
    results.update(native)
    for k, vals in per.items():
        results[k] = float(np.mean(vals))          # headline = average over assets
    results["per_asset"] = {k: [float(x) for x in v] for k, v in per.items()}
    results["compute_time_sec"] = round(time.perf_counter() - t0, 2)
    print(f"  Done in {results['compute_time_sec']:.1f}s", flush=True)
    return results


def main():
    real = _load("heston_ma_S_test_8192x252x8.npy")
    v_real = _load("heston_ma_v_test_8192x252x8.npy")
    disc = _load("heston_ma_S_disc_8192x252x8.npy")
    print(f"Real (test, seed 1): {real.shape}  min={real.min():.2f}  max={real.max():.2f}")
    print(f"Judge (disc, seed 2): {disc.shape}")

    all_results = []
    for seed in range(N_SEEDS):
        res = compute_metrics_for_seed(seed, real, v_real, disc)
        out = os.path.join(RESULTS_DIR, f"seed_{seed}_metrics.json")
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"  Saved {out}", flush=True)
        all_results.append(res)

    if not all_results:
        return
    skip = {"seed", "compute_time_sec", "per_asset", "native_keys", "d"}
    keys = [k for k in all_results[0] if k not in skip]
    summary = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    with open(summary, "w", newline="") as fh:
        cols = ["metric", "scope", "mean", "std"] + [f"seed_{r['seed']}" for r in all_results]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for k in keys:
            vals = [float(r[k]) for r in all_results if r.get(k) is not None]
            row = {"metric": k,
                   "scope": "native" if k in NATIVE_KEYS else "per-asset-mean",
                   "mean": round(float(np.mean(vals)), 8) if vals else None,
                   "std": round(float(np.std(vals)), 8) if vals else None}
            row.update({f"seed_{r['seed']}": r.get(k) for r in all_results})
            w.writerow(row)
    print(f"\nSaved {summary}")

    # per-asset breakdown, one row per (metric, asset)
    pa = os.path.join(RESULTS_DIR, "metrics_per_asset.csv")
    pkeys = sorted(all_results[0]["per_asset"])
    with open(pa, "w", newline="") as fh:
        cols = ["metric", "asset"] + [f"seed_{r['seed']}" for r in all_results] + ["mean"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for k in pkeys:
            for j in range(all_results[0]["d"]):
                vals = [r["per_asset"][k][j] for r in all_results]
                row = {"metric": k, "asset": j, "mean": round(float(np.mean(vals)), 8)}
                row.update({f"seed_{r['seed']}": r["per_asset"][k][j] for r in all_results})
                w.writerow(row)
    print(f"Saved {pa}")

    # B curve-shape aggregate — same schema and same combination rules as the 1-D
    # results tree, applied to the per-asset-AVERAGED B keys held at top level of
    # each seed JSON. Averaging over assets happens BEFORE this aggregation, so the
    # MSE/pct/nrmse/CVaR combination logic is untouched.
    agg = aggregate_curve_metrics(all_results)
    with open(os.path.join(RESULTS_DIR, "curve_b_aggregate.json"), "w") as fh:
        json.dump(agg, fh, indent=2)
    print(f"Saved {os.path.join(RESULTS_DIR, 'curve_b_aggregate.json')}")

    g = [float(r["grid_tvd"]) for r in all_results]
    with open(os.path.join(RESULTS_DIR, "grid_tvd_aggregate.json"), "w") as fh:
        json.dump({"name": "grid_tvd (path-cloud 2D-histogram TVD, %), mean over assets",
                   "n_bins": list(GRID_TVD_DEFAULT_BINS),
                   "mean": float(np.mean(g)), "std": float(np.std(g)),
                   "per_seed": g,
                   "per_seed_per_asset": [r["per_asset"]["grid_tvd"] for r in all_results]},
                  fh, indent=2)
    print(f"Saved {os.path.join(RESULTS_DIR, 'grid_tvd_aggregate.json')}")


if __name__ == "__main__":
    main()
