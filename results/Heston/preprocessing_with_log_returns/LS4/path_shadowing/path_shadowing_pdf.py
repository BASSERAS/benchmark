"""
Path-Shadowing — FULL PDF protocol evaluator (path_shadowing_metrics_protocol.pdf)
for LS4 + SBTS log-return preprocessing on the Heston benchmark.

This is the *paper-faithful* protocol (distinct from the CRPS-only quick eval in
path_shadowing_mc.py). It reads the PERSISTED 1,000,000-path banks produced by
gen_banks.py and evaluates, for each seed, the whole bank-size sweep taken as
NESTED PREFIXES of the same 1M bank (PDF §5.2):

    bank sizes = {4096, 16384, 65536, 262144, 1000000}
    query      = 512 held-out real paths (ps split, seed 3) — independent of bank
    split      = s = 64  (prefix = points 0..64 = 65 pts / 64 increments)
    horizon    = H = 32  (forecast points 65..96, anchored at point 64)
    neighbours = K = 256 (equally-weighted predictive ensemble)

4-BLOCK WEIGHTED, BANK-STANDARDIZED prefix embedding (PDF §3):
    recent returns  : last 32 log-returns            weight 1.0
    cumulative path : cum log-return downsampled 24  weight 0.5
    rolling vol     : windows 5/10/20 last/mean/std  weight 2.0
    dependence      : ACF of |r| and r^2 lags 1,2,5,10   weight 1.0
  standardization z~ = sqrt(w) * (z - mu_bank) / sigma_bank   (per bank size)

FORECAST QUANTITIES (return-based, so endpoint alignment is automatic — the
additive-in-log shift == multiplicative price anchoring cancels for returns):
    cumulative return over H, one-step return, horizon realized-vol (RV)

METRICS per quantity: predictive-mean RMSE, CRPS (energy score), coverage 50/90,
band width 50/90, lower/upper 90 miss-rate.
DIAGNOSTICS: terminal RMSE, prefix-distance mean/median/p95, unique-candidate
fraction, RV mean bias.
UNCERTAINTY: 2000 path-level bootstrap resamples -> 95% percentile intervals.

Nothing under methods/ is touched; banks + query are read-only inputs.

Usage:
  OMP_NUM_THREADS=16 taskset -c 0-15 \
      /home/tbasseras/gpu-venv/bin/python path_shadowing_pdf.py --seeds 0,1,2,3,4
"""
import os
import sys
import json
import time
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))                 # .../LS4/path_shadowing
LS4_DIR = os.path.dirname(HERE)
BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(LS4_DIR))))
DATA_DIR = os.path.join(BENCH_ROOT, "dataset", "Heston", "preprocessing_with_log_returns")
PS_QUERY = os.path.join(DATA_DIR, "heston_S_ps_512x128.npy")
BANK_DIR = os.path.join(HERE, "bank")
PLOT_DIR = os.path.join(HERE, "plots")

# ---- protocol constants (PDF) ----
S_IDX = 64                       # prefix split: points 0..64 known (65 pts, 64 incr)
H = 32                           # horizon in steps
K = 256                          # retrieved neighbours (uniform predictive ensemble)
BANK_SIZES = [4096, 16384, 65536, 262144, 1_000_000]
N_BOOT = 2000
SEQ_LEN = 128
S0 = 100.0

BLOCK_W = {"recent": 1.0, "cumpath": 0.5, "rollvol": 2.0, "dep": 1.0}
ACF_LAGS = [1, 2, 5, 10]
ROLL_WINDOWS = [5, 10, 20]
CUMPATH_PTS = 24                 # downsample length of the cumulative-path block
RECENT_N = 32                    # last-N log-returns block


# ----------------------------------------------------------------------------- features
def _rolling_rms(r, w):
    """Rolling RMS of returns r (N,L) with window w -> (N, L-w+1)."""
    sq = r * r
    c = np.cumsum(np.concatenate([np.zeros((sq.shape[0], 1)), sq], axis=1), axis=1)
    windowed = c[:, w:] - c[:, :-w]
    return np.sqrt(windowed / w)


def _acf(A, lag):
    """Per-path lag-`lag` autocorrelation of series A (N,L)."""
    A = A - A.mean(axis=1, keepdims=True)
    num = (A[:, lag:] * A[:, :-lag]).sum(axis=1)
    den = (A * A).sum(axis=1) + 1e-30
    return num / den


def build_features(logS):
    """
    logS : (N, 65) log-prices of the prefix (points 0..64).
    Returns feats (N, D) raw (unstandardized) and the sqrt(weight) vector (D,).
    """
    r = np.diff(logS, axis=1)                        # (N, 64) log-returns of prefix
    blocks, wparts = [], []

    # 1) recent returns : last 32
    recent = r[:, -RECENT_N:]
    blocks.append(recent)
    wparts.append(np.full(recent.shape[1], BLOCK_W["recent"]))

    # 2) cumulative path : cum log-return vs start, downsampled to 24 points
    cum = logS - logS[:, :1]                          # (N,65)
    idx = np.linspace(0, logS.shape[1] - 1, CUMPATH_PTS).round().astype(int)
    cump = cum[:, idx]
    blocks.append(cump)
    wparts.append(np.full(cump.shape[1], BLOCK_W["cumpath"]))

    # 3) rolling vol : windows 5/10/20 -> last / mean / std
    rv_feats = []
    for w in ROLL_WINDOWS:
        rr = _rolling_rms(r, w)                       # (N, 64-w+1)
        rv_feats.append(rr[:, -1:])                   # last
        rv_feats.append(rr.mean(axis=1, keepdims=True))
        rv_feats.append(rr.std(axis=1, keepdims=True))
    rv = np.concatenate(rv_feats, axis=1)             # (N, 9)
    blocks.append(rv)
    wparts.append(np.full(rv.shape[1], BLOCK_W["rollvol"]))

    # 4) dependence : ACF of |r| and r^2 at lags 1,2,5,10
    dep = []
    for series in (np.abs(r), r * r):
        for lag in ACF_LAGS:
            dep.append(_acf(series, lag)[:, None])
    dep = np.concatenate(dep, axis=1)                 # (N, 8)
    blocks.append(dep)
    wparts.append(np.full(dep.shape[1], BLOCK_W["dep"]))

    feats = np.concatenate(blocks, axis=1).astype(np.float64)
    sqrtw = np.sqrt(np.concatenate(wparts)).astype(np.float64)
    return feats, sqrtw


# ----------------------------------------------------------------------------- retrieval
def retrieve(q_std, b_std, k):
    """Brute-force KNN. Returns (idx (Nq,k), dist (Nq,k))."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean",
                          n_jobs=16)
    nn.fit(b_std)
    dist, idx = nn.kneighbors(q_std)
    return idx, dist


# ----------------------------------------------------------------------------- scoring
def _gini_mean_abs_diff(x_sorted):
    """E|X-X'| for 1-D samples via the sorted-order formula (O(K))."""
    k = x_sorted.shape[-1]
    i = np.arange(1, k + 1)
    coef = (2 * i - k - 1).astype(np.float64)
    return (2.0 / (k * k)) * (coef * x_sorted).sum(axis=-1)


def _crps_scalar(Y, y):
    """Energy-score CRPS for scalar quantity. Y (Nq,K), y (Nq,). -> (Nq,)."""
    term1 = np.abs(Y - y[:, None]).mean(axis=1)
    term2 = 0.5 * _gini_mean_abs_diff(np.sort(Y, axis=1))
    return term1 - term2


def quantity_metrics(Y, y):
    """
    Y : (Nq, K) predictive samples, y : (Nq,) realized target.
    Returns (agg dict, per-path dict-of-vectors) for bootstrap.
    """
    pred_mean = Y.mean(axis=1)
    se = (pred_mean - y) ** 2
    crps = _crps_scalar(Y, y)

    q = np.percentile(Y, [5, 25, 75, 95], axis=1)          # (4, Nq)
    lo90, lo50, hi50, hi90 = q[0], q[1], q[2], q[3]
    cov50 = ((y >= lo50) & (y <= hi50)).astype(np.float64)
    cov90 = ((y >= lo90) & (y <= hi90)).astype(np.float64)
    w50 = hi50 - lo50
    w90 = hi90 - lo90
    lower_miss = (y < lo90).astype(np.float64)
    upper_miss = (y > hi90).astype(np.float64)

    per = {"se": se, "crps": crps, "cov50": cov50, "cov90": cov90,
           "w50": w50, "w90": w90, "lower_miss90": lower_miss,
           "upper_miss90": upper_miss}
    agg = {"rmse": float(np.sqrt(se.mean())),
           "crps": float(crps.mean()),
           "coverage50": float(cov50.mean()),
           "coverage90": float(cov90.mean()),
           "width50": float(w50.mean()),
           "width90": float(w90.mean()),
           "lower_miss90": float(lower_miss.mean()),
           "upper_miss90": float(upper_miss.mean())}
    return agg, per


def bootstrap_ci(per_path_vec, agg_fn=np.mean, n_boot=N_BOOT, seed=0):
    """95% percentile CI of agg_fn over path-level resamples."""
    rng = np.random.default_rng(seed)
    n = per_path_vec.shape[0]
    stats = np.empty(n_boot)
    for b in range(n_boot):
        samp = per_path_vec[rng.integers(0, n, n)]
        stats[b] = agg_fn(samp)
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


# ----------------------------------------------------------------------------- quantities
def forecast_quantities(logpaths):
    """
    logpaths : (M, SEQ_LEN) log-prices.
    Returns dict of realized/candidate quantities anchored at S_IDX:
      cum   : cumulative log-return over horizon   logS[S_IDX+H]-logS[S_IDX]
      step  : one-step return                      logS[S_IDX+1]-logS[S_IDX]
      rv    : horizon realized-vol  sqrt(sum r^2 over the H forward steps)
    """
    anchor = logpaths[:, S_IDX]
    cum = logpaths[:, S_IDX + H] - anchor
    step = logpaths[:, S_IDX + 1] - anchor
    fwd = np.diff(logpaths[:, S_IDX:S_IDX + H + 1], axis=1)      # (M, H)
    rv = np.sqrt((fwd * fwd).sum(axis=1))
    return {"cum": cum, "step": step, "rv": rv}


# ----------------------------------------------------------------------------- per-seed
def eval_seed(seed, qlogS, q_quant, q_prefix_feat, q_sqrtw):
    bank_path = os.path.join(BANK_DIR, f"generated_bank_seed{seed}_1000000x{SEQ_LEN}.npy")
    B = np.load(bank_path, mmap_mode="r")                       # (1M,128) price
    Nq = qlogS.shape[0]
    out = {"seed": int(seed), "by_bank_size": {}}

    for bs in BANK_SIZES:
        t0 = time.time()
        Bp = np.asarray(B[:bs], dtype=np.float64)               # nested prefix
        logB = np.log(Bp)
        bfeat, _ = build_features(logB[:, :S_IDX + 1])
        mu = bfeat.mean(axis=0)
        sd = bfeat.std(axis=0) + 1e-12
        b_std = (q_sqrtw * (bfeat - mu) / sd).astype(np.float32)
        q_std = (q_sqrtw * (q_prefix_feat - mu) / sd).astype(np.float32)

        idx, dist = retrieve(q_std, b_std, K)                   # (Nq,K)
        b_quant = forecast_quantities(logB)                     # each (bs,)

        entry = {"bank_size": int(bs), "quantities": {}, "diagnostics": {}}
        for qn in ("cum", "step", "rv"):
            Y = b_quant[qn][idx]                                # (Nq,K) predictive
            agg, per = quantity_metrics(Y, q_quant[qn])
            agg["rmse_ci"] = bootstrap_ci(per["se"],
                                          lambda v: np.sqrt(v.mean()), seed=seed)
            agg["crps_ci"] = bootstrap_ci(per["crps"], np.mean, seed=seed)
            entry["quantities"][qn] = agg

        # diagnostics
        pred_cum_mean = b_quant["cum"][idx].mean(axis=1)
        term_rmse = float(np.sqrt(((pred_cum_mean - q_quant["cum"]) ** 2).mean()))
        uniq = np.unique(idx).size / float(bs)
        rv_bias = float(b_quant["rv"][idx].mean() - q_quant["rv"].mean())
        entry["diagnostics"] = {
            "terminal_rmse": term_rmse,
            "prefix_dist_mean": float(dist.mean()),
            "prefix_dist_median": float(np.median(dist)),
            "prefix_dist_p95": float(np.percentile(dist, 95)),
            "unique_candidate_frac": float(uniq),
            "rv_mean_bias": rv_bias,
            "elapsed_sec": round(time.time() - t0, 1),
        }
        out["by_bank_size"][str(bs)] = entry
        print(f"[seed {seed}] bank={bs:>8} cum.CRPS={entry['quantities']['cum']['crps']:.5f} "
              f"cum.cov90={entry['quantities']['cum']['coverage90']:.3f} "
              f"uniq={uniq:.4f} {entry['diagnostics']['elapsed_sec']}s", flush=True)
    return out


# ----------------------------------------------------------------------------- baseline
def rw_baseline(qlogS, q_quant):
    """Random-walk (zero-drift) predictive baseline: last prefix step repeated.
    Predictive samples = prefix one-step returns (bootstrap of recent innovations)."""
    r = np.diff(qlogS[:, :S_IDX + 1], axis=1)                   # (Nq, 64)
    rng = np.random.default_rng(0)
    Nq = qlogS.shape[0]
    # sample H-step continuations by resampling each path's own prefix returns
    res = {}
    K_rw = 256
    cum_s = np.empty((Nq, K_rw)); step_s = np.empty((Nq, K_rw)); rv_s = np.empty((Nq, K_rw))
    for i in range(Nq):
        draws = r[i][rng.integers(0, r.shape[1], size=(K_rw, H))]   # (K_rw,H)
        cum_s[i] = draws.sum(axis=1)
        step_s[i] = draws[:, 0]
        rv_s[i] = np.sqrt((draws * draws).sum(axis=1))
    for qn, Y in (("cum", cum_s), ("step", step_s), ("rv", rv_s)):
        agg, _ = quantity_metrics(Y, q_quant[qn])
        res[qn] = agg
    return res


# ----------------------------------------------------------------------------- plots
def make_plots(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(PLOT_DIR, exist_ok=True)
    bss = [int(b) for b in summary["by_bank_size"].keys()]

    # CRPS vs bank size (cum quantity), mean +/- std across seeds
    fig, ax = plt.subplots(figsize=(7, 5))
    for qn, c in (("cum", "C0"), ("step", "C1"), ("rv", "C2")):
        m = [summary["by_bank_size"][str(b)]["quantities"][qn]["crps_mean"] for b in bss]
        s = [summary["by_bank_size"][str(b)]["quantities"][qn]["crps_std"] for b in bss]
        ax.errorbar(bss, m, yerr=s, marker="o", label=qn, color=c, capsize=3)
    ax.set_xscale("log"); ax.set_xlabel("bank size"); ax.set_ylabel("CRPS")
    ax.set_title("LS4+logret Path-Shadowing CRPS vs bank size (5 seeds)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "pdf_crps_vs_banksize.png"), dpi=130)
    plt.close(fig)

    # coverage calibration at the largest bank
    big = str(max(bss))
    fig, ax = plt.subplots(figsize=(7, 5))
    qns = ["cum", "step", "rv"]
    x = np.arange(len(qns))
    for j, (lvl, tgt) in enumerate((("coverage50", 0.5), ("coverage90", 0.9))):
        vals = [summary["by_bank_size"][big]["quantities"][q][f"{lvl}_mean"] for q in qns]
        ax.bar(x + (j - 0.5) * 0.35, vals, width=0.35, label=lvl)
        ax.axhline(tgt, ls="--", color="k" if j == 0 else "gray", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(qns)
    ax.set_ylabel("empirical coverage"); ax.set_ylim(0, 1)
    ax.set_title(f"Coverage calibration @ bank={big} (dashed = nominal)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "pdf_coverage_calibration.png"), dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- aggregate
def aggregate(per_seed):
    """Mean +/- std across seeds per bank size / quantity / metric."""
    bss = [str(b) for b in BANK_SIZES]
    metrics = ["rmse", "crps", "coverage50", "coverage90", "width50", "width90",
               "lower_miss90", "upper_miss90"]
    diags = ["terminal_rmse", "prefix_dist_mean", "prefix_dist_median",
             "prefix_dist_p95", "unique_candidate_frac", "rv_mean_bias"]
    out = {"n_seeds": len(per_seed), "by_bank_size": {}}
    for bs in bss:
        entry = {"quantities": {}, "diagnostics": {}}
        for qn in ("cum", "step", "rv"):
            qd = {}
            for mk in metrics:
                vals = np.array([s["by_bank_size"][bs]["quantities"][qn][mk]
                                 for s in per_seed])
                qd[f"{mk}_mean"] = float(vals.mean())
                qd[f"{mk}_std"] = float(vals.std())
            entry["quantities"][qn] = qd
        for dk in diags:
            vals = np.array([s["by_bank_size"][bs]["diagnostics"][dk] for s in per_seed])
            entry["diagnostics"][f"{dk}_mean"] = float(vals.mean())
            entry["diagnostics"][f"{dk}_std"] = float(vals.std())
        out["by_bank_size"][bs] = entry
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    os.makedirs(PLOT_DIR, exist_ok=True)
    qS = np.load(PS_QUERY).astype(np.float64)                   # (512,128) price
    assert qS.shape[1] == SEQ_LEN, qS.shape
    qlogS = np.log(qS)
    q_quant = forecast_quantities(qlogS)
    q_prefix_feat, q_sqrtw = build_features(qlogS[:, :S_IDX + 1])
    print(f"[pdf] query {qS.shape} D={q_prefix_feat.shape[1]} "
          f"S_IDX={S_IDX} H={H} K={K} banks={BANK_SIZES}", flush=True)

    rw = rw_baseline(qlogS, q_quant)
    print(f"[pdf] RW baseline CRPS cum={rw['cum']['crps']:.5f} "
          f"step={rw['step']['crps']:.5f} rv={rw['rv']['crps']:.5f}", flush=True)

    per_seed = []
    for seed in seeds:
        r = eval_seed(seed, qlogS, q_quant, q_prefix_feat, q_sqrtw)
        per_seed.append(r)
        with open(os.path.join(HERE, f"pdf_results_seed{seed}.json"), "w") as f:
            json.dump(r, f, indent=2)

    summary = aggregate(per_seed)
    summary["rw_baseline"] = rw
    summary["protocol"] = {"S_IDX": S_IDX, "H": H, "K": K, "bank_sizes": BANK_SIZES,
                           "n_query": int(qS.shape[0]), "n_boot": N_BOOT,
                           "block_weights": BLOCK_W, "feature_dim": int(q_prefix_feat.shape[1])}
    with open(os.path.join(HERE, "pdf_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    make_plots(summary)
    print("[pdf] wrote pdf_summary.json + plots. DONE.", flush=True)


if __name__ == "__main__":
    main()
