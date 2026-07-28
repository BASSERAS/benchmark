"""
Path-Shadowing — FULL PDF protocol evaluator (arXiv:2308.01486) for LS4 + SBTS
log-return preprocessing on the Heston benchmark.

This is the *paper-faithful* protocol (distinct from the CRPS-only quick eval in
path_shadowing_mc.py, and from the simplified murex reference eval under
methods/LS4/path_shadowing/ — see GUIDELINE.md M7).

DESIGN — ONE SHARED 1M BANK (paper uses a single bank):
    The evaluator reads ONE persisted 1,000,000-path bank produced by
    gen_banks.py from the seed-0 generator, and evaluates the whole bank-size
    sweep taken as NESTED PREFIXES of that single 1M bank (PDF §5.2):

        bank sizes = {4096, 16384, 65536, 262144, 1000000}  (nested prefixes)
        query      = 512 held-out real paths (ps split) — independent of bank
        split      = s = 64  (prefix = points 0..64 = 65 pts / 64 increments)
        horizon    = H = 32  (forecast points 65..96, anchored at point 64)
        neighbours = K = 256 (equally-weighted predictive ensemble)

    Uncertainty is the 2000-resample PAIRED bootstrap over the 512 QUERY paths
    (a single fixed resample-index matrix, seed BOOT_SEED, reused across every
    bank size / quantity / metric so differences are directly comparable). There
    is no model-seed spread: one generator, one bank, one run.

4-BLOCK WEIGHTED, FROZEN-REFERENCE-STANDARDIZED prefix embedding (PDF §3):
    recent returns  : last 32 log-returns            block weight 1.0
    cumulative path : cum log-return downsampled 24  block weight 0.5
    rolling vol     : windows 5/10/20 last/mean/std  block weight 2.0
    dependence      : ACF of |r| and r^2 lags 1,2,5,10   block weight 1.0

  PER-BLOCK DIMENSION NORMALIZATION (GUIDELINE §9 decision B8): each feature in a
  block of dimension d gets weight w_block / d, so the block contributes total
  mass w_block regardless of how many features it has (removes the
  weight x dimensionality confound). Per-feature multiplier = sqrt(w_block / d).

  FROZEN-REFERENCE STANDARDIZATION (GUIDELINE §9 decisions A1/A2/A15/A16): the
  standardization mean/std (mu_ref, sigma_ref) are computed ONCE on the REAL
  Heston test set (heston_S_test_4096x128 — model-independent) and held FIXED
  across the entire bank-size sweep. This makes distances comparable across bank
  sizes and decouples the metric from the generator. Embedding:
        z~ = sqrt(w_block/d) * (z - mu_ref) / sigma_ref

FORECAST QUANTITIES (return-based, so endpoint alignment is automatic — the
additive-in-log shift == multiplicative price anchoring cancels for returns):
    cumulative return over H, one-step return, horizon realized-vol (RV)

METRICS per quantity: predictive-mean RMSE, CRPS (energy score), coverage 50/90,
band width 50/90, lower/upper 90 miss-rate — each with a 95% bootstrap CI.
DIAGNOSTICS: terminal RMSE, prefix-distance mean/median/p95, unique-candidate
fraction, RV mean bias.

Nothing under methods/ is touched; bank + query + reference are read-only inputs.

Usage:
  OMP_NUM_THREADS=16 taskset -c 0-15 \
      /home/tbasseras/gpu-venv/bin/python path_shadowing_pdf.py
"""
import os
import json
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))                 # .../LS4/path_shadowing
LS4_DIR = os.path.dirname(HERE)
BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(LS4_DIR))))
DATA_DIR = os.path.join(BENCH_ROOT, "dataset", "Heston", "preprocessing_with_log_returns")
PS_QUERY = os.path.join(DATA_DIR, "heston_S_ps_512x128.npy")
REF_STATS = os.path.join(DATA_DIR, "heston_S_test_4096x128.npy")  # frozen mu/sigma reference
BANK_DIR = os.path.join(HERE, "bank")
PLOT_DIR = os.path.join(HERE, "plots")

# ---- protocol constants (PDF) ----
S_IDX = 64                       # prefix split: points 0..64 known (65 pts, 64 incr)
H = 32                           # horizon in steps
K = 256                          # retrieved neighbours (uniform predictive ensemble)
BANK_SIZES = [4096, 16384, 65536, 262144, 1_000_000]
BANK_SEED = 0                    # the ONE generator seed backing the shared 1M bank
N_BOOT = 2000
BOOT_SEED = 20230814             # fixed, model-independent (arXiv:2308.01486 date)
SEQ_LEN = 128
S0 = 100.0

BLOCK_W = {"recent": 1.0, "cumpath": 0.5, "rollvol": 2.0, "dep": 1.0}
ACF_LAGS = [1, 2, 5, 10]
ROLL_WINDOWS = [5, 10, 20]
CUMPATH_PTS = 24                 # downsample length of the cumulative-path block
RECENT_N = 32                    # last-N log-returns block

# metric -> (per-path key, aggregation kind)
METRIC_MAP = {
    "rmse": ("se", "rmse"),
    "crps": ("crps", "mean"),
    "coverage50": ("cov50", "mean"),
    "coverage90": ("cov90", "mean"),
    "width50": ("w50", "mean"),
    "width90": ("w90", "mean"),
    "lower_miss90": ("lower_miss90", "mean"),
    "upper_miss90": ("upper_miss90", "mean"),
}


# ----------------------------------------------------------------------------- features
def _rolling_rms(r, w):
    """Rolling RMS of returns r (N,L) with window w -> (N, L-w+1), causal/valid only."""
    sq = r * r
    c = np.cumsum(np.concatenate([np.zeros((sq.shape[0], 1)), sq], axis=1), axis=1)
    windowed = c[:, w:] - c[:, :-w]
    return np.sqrt(windowed / w)


def _acf(A, lag):
    """Per-path biased lag-`lag` autocorrelation of series A (N,L) (prefix-only, demeaned)."""
    A = A - A.mean(axis=1, keepdims=True)
    num = (A[:, lag:] * A[:, :-lag]).sum(axis=1)
    den = (A * A).sum(axis=1) + 1e-30
    return num / den


def build_features(logS):
    """
    logS : (N, 65) log-prices of the prefix (points 0..64).
    Returns feats (N, D) raw (unstandardized) and the per-feature sqrt-weight (D,).

    Per-block dimension normalization (B8): a block of dimension d contributes
    total mass w_block, so each of its features carries weight w_block/d and the
    per-feature standardization multiplier is sqrt(w_block/d).
    """
    r = np.diff(logS, axis=1)                        # (N, 64) log-returns of prefix
    blocks, wparts = [], []

    def _add(block, wname):
        blocks.append(block)
        d = block.shape[1]
        wparts.append(np.full(d, BLOCK_W[wname] / d))

    # 1) recent returns : last 32
    _add(r[:, -RECENT_N:], "recent")

    # 2) cumulative path : cum log-return vs start, downsampled to 24 points
    cum = logS - logS[:, :1]                          # (N,65)
    idx = np.linspace(0, logS.shape[1] - 1, CUMPATH_PTS).round().astype(int)
    _add(cum[:, idx], "cumpath")

    # 3) rolling vol : windows 5/10/20 -> last / mean / std
    rv_feats = []
    for w in ROLL_WINDOWS:
        rr = _rolling_rms(r, w)                       # (N, 64-w+1)
        rv_feats.append(rr[:, -1:])                   # last
        rv_feats.append(rr.mean(axis=1, keepdims=True))
        rv_feats.append(rr.std(axis=1, keepdims=True))
    _add(np.concatenate(rv_feats, axis=1), "rollvol") # (N, 9)

    # 4) dependence : ACF of |r| and r^2 at lags 1,2,5,10
    dep = []
    for series in (np.abs(r), r * r):
        for lag in ACF_LAGS:
            dep.append(_acf(series, lag)[:, None])
    _add(np.concatenate(dep, axis=1), "dep")          # (N, 8)

    feats = np.concatenate(blocks, axis=1).astype(np.float64)
    sqrtw = np.sqrt(np.concatenate(wparts)).astype(np.float64)
    return feats, sqrtw


def frozen_reference_stats(sqrtw):
    """mu/sigma computed ONCE on the real Heston test set (model-independent),
    held fixed across the whole bank-size sweep (A1/A2/A15/A16)."""
    ref = np.load(REF_STATS).astype(np.float64)                  # (Nref,128) price
    reflog = np.log(ref)
    rfeat, _ = build_features(reflog[:, :S_IDX + 1])
    mu = rfeat.mean(axis=0)
    sd = rfeat.std(axis=0) + 1e-12
    return mu, sd


# ----------------------------------------------------------------------------- retrieval
def retrieve(q_std, b_std, k):
    """Brute-force exact KNN (lowest-index tie-break; ties are measure-zero)."""
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


def per_path_metrics(Y, y):
    """
    Y : (Nq, K) predictive samples, y : (Nq,) realized target.
    Returns dict of per-path vectors (bootstrap inputs). Quantiles via
    np.percentile linear/type-7 interpolation.
    """
    pred_mean = Y.mean(axis=1)
    se = (pred_mean - y) ** 2
    crps = _crps_scalar(Y, y)
    q = np.percentile(Y, [5, 25, 75, 95], axis=1)          # (4, Nq)
    lo90, lo50, hi50, hi90 = q[0], q[1], q[2], q[3]
    return {
        "se": se, "crps": crps,
        "cov50": ((y >= lo50) & (y <= hi50)).astype(np.float64),
        "cov90": ((y >= lo90) & (y <= hi90)).astype(np.float64),
        "w50": hi50 - lo50, "w90": hi90 - lo90,
        "lower_miss90": (y < lo90).astype(np.float64),
        "upper_miss90": (y > hi90).astype(np.float64),
    }


def _agg(per, kind):
    return float(np.sqrt(per.mean())) if kind == "rmse" else float(per.mean())


def _boot_ci(per, kind, boot_idx):
    """Paired bootstrap 95% percentile CI using a SHARED resample-index matrix."""
    samp = per[boot_idx]                                    # (N_BOOT, Nq)
    stat = np.sqrt(samp.mean(axis=1)) if kind == "rmse" else samp.mean(axis=1)
    return [float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))]


def metrics_with_ci(Y, y, boot_idx):
    per = per_path_metrics(Y, y)
    out = {}
    for mname, (pkey, kind) in METRIC_MAP.items():
        out[mname] = {"value": _agg(per[pkey], kind),
                      "ci": _boot_ci(per[pkey], kind, boot_idx)}
    return out


# ----------------------------------------------------------------------------- quantities
def forecast_quantities(logpaths):
    """
    logpaths : (M, SEQ_LEN) log-prices. Return-based quantities anchored at S_IDX:
      cum  : cumulative log-return over horizon   logS[S_IDX+H]-logS[S_IDX]
      step : one-step return                      logS[S_IDX+1]-logS[S_IDX]
      rv   : horizon realized-vol  sqrt(sum r^2 over the H forward steps)
    """
    anchor = logpaths[:, S_IDX]
    cum = logpaths[:, S_IDX + H] - anchor
    step = logpaths[:, S_IDX + 1] - anchor
    fwd = np.diff(logpaths[:, S_IDX:S_IDX + H + 1], axis=1)      # (M, H)
    rv = np.sqrt((fwd * fwd).sum(axis=1))
    return {"cum": cum, "step": step, "rv": rv}


# ----------------------------------------------------------------------------- per-bank-size
def eval_bank(qlogS, q_quant, q_prefix_feat, sqrtw, mu, sd, boot_idx):
    """Evaluate the single shared 1M bank over the nested bank-size sweep."""
    bank_path = os.path.join(BANK_DIR, f"generated_bank_seed{BANK_SEED}_1000000x{SEQ_LEN}.npy")
    B = np.load(bank_path, mmap_mode="r")                       # (1M,128) price
    q_std = (sqrtw * (q_prefix_feat - mu) / sd).astype(np.float32)
    out = {"bank_seed": int(BANK_SEED), "by_bank_size": {}}

    for bs in BANK_SIZES:
        t0 = time.time()
        Bp = np.asarray(B[:bs], dtype=np.float64)               # nested prefix
        logB = np.log(Bp)
        bfeat, _ = build_features(logB[:, :S_IDX + 1])
        b_std = (sqrtw * (bfeat - mu) / sd).astype(np.float32)  # FROZEN mu/sd

        idx, dist = retrieve(q_std, b_std, K)                   # (Nq,K)
        b_quant = forecast_quantities(logB)

        entry = {"bank_size": int(bs), "quantities": {}, "diagnostics": {}}
        for qn in ("cum", "step", "rv"):
            Y = b_quant[qn][idx]                                # (Nq,K) predictive
            entry["quantities"][qn] = metrics_with_ci(Y, q_quant[qn], boot_idx)

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
        c = entry["quantities"]["cum"]
        print(f"[pdf] bank={bs:>8} cum.CRPS={c['crps']['value']:.5f} "
              f"cum.cov90={c['coverage90']['value']:.3f} "
              f"uniq={uniq:.4f} {entry['diagnostics']['elapsed_sec']}s", flush=True)
    return out


# ----------------------------------------------------------------------------- baseline
def rw_baseline(qlogS, q_quant, boot_idx):
    """Random-walk (zero-drift) predictive baseline: resample each query path's own
    prefix one-step returns to build H-step continuations."""
    r = np.diff(qlogS[:, :S_IDX + 1], axis=1)                   # (Nq, 64)
    rng = np.random.default_rng(0)
    Nq = qlogS.shape[0]
    K_rw = K
    cum_s = np.empty((Nq, K_rw)); step_s = np.empty((Nq, K_rw)); rv_s = np.empty((Nq, K_rw))
    for i in range(Nq):
        draws = r[i][rng.integers(0, r.shape[1], size=(K_rw, H))]   # (K_rw,H)
        cum_s[i] = draws.sum(axis=1)
        step_s[i] = draws[:, 0]
        rv_s[i] = np.sqrt((draws * draws).sum(axis=1))
    res = {}
    for qn, Y in (("cum", cum_s), ("step", step_s), ("rv", rv_s)):
        res[qn] = metrics_with_ci(Y, q_quant[qn], boot_idx)
    return res


# ----------------------------------------------------------------------------- plots
def make_plots(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(PLOT_DIR, exist_ok=True)
    bss = [int(b) for b in summary["by_bank_size"].keys()]

    # CRPS vs bank size (all quantities), with 95% bootstrap CI band
    fig, ax = plt.subplots(figsize=(7, 5))
    for qn, c in (("cum", "C0"), ("step", "C1"), ("rv", "C2")):
        m = [summary["by_bank_size"][str(b)]["quantities"][qn]["crps"]["value"] for b in bss]
        lo = [summary["by_bank_size"][str(b)]["quantities"][qn]["crps"]["ci"][0] for b in bss]
        hi = [summary["by_bank_size"][str(b)]["quantities"][qn]["crps"]["ci"][1] for b in bss]
        ax.plot(bss, m, marker="o", label=qn, color=c)
        ax.fill_between(bss, lo, hi, color=c, alpha=0.2)
    ax.set_xscale("log"); ax.set_xlabel("bank size"); ax.set_ylabel("CRPS")
    ax.set_title("LS4+logret Path-Shadowing CRPS vs bank size (one 1M bank, 95% CI)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "pdf_crps_vs_banksize.png"), dpi=130)
    plt.close(fig)

    # coverage calibration at the largest bank
    big = str(max(bss))
    fig, ax = plt.subplots(figsize=(7, 5))
    qns = ["cum", "step", "rv"]
    x = np.arange(len(qns))
    for j, (lvl, tgt) in enumerate((("coverage50", 0.5), ("coverage90", 0.9))):
        vals = [summary["by_bank_size"][big]["quantities"][q][lvl]["value"] for q in qns]
        ax.bar(x + (j - 0.5) * 0.35, vals, width=0.35, label=lvl)
        ax.axhline(tgt, ls="--", color="k" if j == 0 else "gray", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(qns)
    ax.set_ylabel("empirical coverage"); ax.set_ylim(0, 1)
    ax.set_title(f"Coverage calibration @ bank={big} (dashed = nominal)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "pdf_coverage_calibration.png"), dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    qS = np.load(PS_QUERY).astype(np.float64)                   # (512,128) price
    assert qS.shape[1] == SEQ_LEN, qS.shape
    qlogS = np.log(qS)
    Nq = qlogS.shape[0]
    q_quant = forecast_quantities(qlogS)
    q_prefix_feat, sqrtw = build_features(qlogS[:, :S_IDX + 1])
    mu, sd = frozen_reference_stats(sqrtw)

    # ONE shared paired-bootstrap resample-index matrix over the query paths.
    boot_idx = np.random.default_rng(BOOT_SEED).integers(0, Nq, size=(N_BOOT, Nq))

    print(f"[pdf] query {qS.shape} D={q_prefix_feat.shape[1]} S_IDX={S_IDX} H={H} "
          f"K={K} banks={BANK_SIZES} bank_seed={BANK_SEED} boot_seed={BOOT_SEED}", flush=True)
    print(f"[pdf] frozen mu/sigma from real test set {os.path.basename(REF_STATS)}", flush=True)

    rw = rw_baseline(qlogS, q_quant, boot_idx)
    print(f"[pdf] RW baseline CRPS cum={rw['cum']['crps']['value']:.5f} "
          f"step={rw['step']['crps']['value']:.5f} rv={rw['rv']['crps']['value']:.5f}", flush=True)

    summary = eval_bank(qlogS, q_quant, q_prefix_feat, sqrtw, mu, sd, boot_idx)
    summary["rw_baseline"] = rw
    summary["protocol"] = {
        "design": "one_shared_1M_bank",
        "S_IDX": S_IDX, "H": H, "K": K, "bank_sizes": BANK_SIZES,
        "bank_seed": BANK_SEED, "n_query": int(Nq), "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED, "block_weights": BLOCK_W,
        "block_dim_normalized": True, "frozen_reference": os.path.basename(REF_STATS),
        "feature_dim": int(q_prefix_feat.shape[1]),
    }
    with open(os.path.join(HERE, "pdf_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    make_plots(summary)
    print("[pdf] wrote pdf_summary.json + plots. DONE.", flush=True)


if __name__ == "__main__":
    main()
