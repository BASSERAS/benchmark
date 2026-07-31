"""
Forecaster -> strict Path-Shadowing PDF bridge (arXiv:2308.01486).

A generator is scored in the strict PDF protocol by using its 1M generated bank
to *shadow-forecast* the 512 held-out query paths (retrieve K=256 nearest
prefixes, read their futures as the predictive ensemble). A conditional
forecaster (Chronos-2, TimesFM) instead forecasts each query DIRECTLY: given the
65-point prefix (points 0..64) it emits K predictive price continuations over the
H=32 horizon (points 65..96). Both routes end in the SAME object — a per-query
predictive ensemble of forward paths — so both are scored with the SAME return-
based quantities (cum / step / rv) and the SAME 2000-resample paired bootstrap
(BOOT_SEED=20230814). That makes a forecaster column apples-to-apples with the
generator PS columns and the Heston oracle / RW floor already in the generators'
pdf_summary.json.

This module REUSES the untouched strict scorer
(../CSDI/path_shadowing/path_shadowing_pdf.py): forecast_quantities,
metrics_with_ci, and the protocol constants S_IDX/H/K/BOOT_SEED/N_BOOT/PS_QUERY.
Only the ensemble construction differs (direct forecast vs bank retrieval).
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))                       # .../forecaster
EXP_DIR = os.path.dirname(HERE)                                         # .../preprocessing_with_log_returns
STRICT_DIR = os.path.join(EXP_DIR, "CSDI", "path_shadowing")            # untouched strict scorer
sys.path.insert(0, STRICT_DIR)
import path_shadowing_pdf as P  # noqa: E402  (forecast_quantities, metrics_with_ci, constants)


def build_forecaster_Y(ens_price, qlogS):
    """
    ens_price : (Nq, K, H) forecast PRICES for points S_IDX+1 .. S_IDX+H (i.e. 65..96).
    qlogS     : (Nq, SEQ_LEN) true query log-prices (for the anchor at S_IDX).

    Anchor every forecast member at the TRUE logS[:, S_IDX] (point 64), exactly as
    forecast_quantities anchors the truth, so the additive-in-log endpoint shift
    cancels for the return-based quantities. Returns cum/step/rv in the shapes
    per_path_metrics expects: cum/step (Nq,K,H), rv (Nq,K).
    """
    Nq, K, Hh = ens_price.shape
    assert Hh == P.H, (Hh, P.H)
    anchor = qlogS[:, P.S_IDX][:, None, None]                           # (Nq,1,1) true logS[64]
    logfut = np.log(ens_price.astype(np.float64))                      # (Nq,K,H) forecast log-prices 65..96
    logpath = np.concatenate([np.broadcast_to(anchor, (Nq, K, 1)), logfut], axis=2)  # (Nq,K,H+1)
    fwd = np.diff(logpath, axis=2)                                     # (Nq,K,H) forward increments
    cum = np.cumsum(fwd, axis=2)                                       # (Nq,K,H)
    step = fwd                                                         # (Nq,K,H)
    rv = np.sqrt((fwd * fwd).sum(axis=2))                             # (Nq,K)
    return {"cum": cum, "step": step, "rv": rv}


# RMSE aggregation -- the forecaster rows use the SAME convention as every other row.
#
# There are two places to take the square root of the per-query squared error se_q:
#   (A) root-last    sqrt(mean_q(se_q))    <- textbook RMSE
#   (B) root-inside  mean_q(sqrt(se_q))    <- what path_shadowing_pdf.py:254 does
# sqrt is concave, so Jensen gives (B) <= (A) ALWAYS. The two are not interchangeable,
# and mixing them inside one table makes the RMSE column non-comparable across rows.
#
# This bridge used to pin the forecasters to (A) via a local _metrics_frozen_rmse(), to
# keep reproducing the reproducibility report's published Chronos-2 / TimesFM cells.
# That was the only deviation left in the repo and it penalised the forecasters by a
# measured 18.0% (cum) / 5.6% (step) / 5.0% (rv) -- and _ps_winner_idx ranks forecasters
# against generators, so the Winner column was ranking those inflated cells. Removed:
# score_forecaster
# now calls P.metrics_with_ci directly, so forecasters are scored by the byte-identical
# code path as the generators, the Heston oracle and the RW floor. The report's published
# forecaster RMSE cells are therefore SUPERSEDED, not reproduced -- that is intended.


def score_forecaster(ens_price, name, out_path, extra_meta=None):
    """Score a forecaster's (Nq,K,H) price ensemble with the strict PDF metrics and
    write <name>_pdf.json. Uses the SAME boot_idx and q_quant as the generators."""
    qS = np.load(P.PS_QUERY).astype(np.float64)                       # (512,128) price
    qlogS = np.log(qS)
    Nq = qlogS.shape[0]
    assert ens_price.shape[0] == Nq, (ens_price.shape, Nq)

    q_quant = P.forecast_quantities(qlogS)                            # truth: cum/step/rv
    boot_idx = np.random.default_rng(P.BOOT_SEED).integers(0, Nq, size=(P.N_BOOT, Nq))

    Y = build_forecaster_Y(ens_price, qlogS)
    quantities = {}
    for qn in ("cum", "step", "rv"):
        quantities[qn] = P.metrics_with_ci(Y[qn], q_quant[qn], boot_idx)

    # diagnostics comparable to eval_bank's
    pred_term_mean = Y["cum"][:, :, -1].mean(axis=1)
    # This is an MAE, not an RMSE -- mean_q(|e_q|), matching eval_bank's terminal_rmse
    # bit-for-bit. Terminal cum is a scalar per query (H=1), so the root-inside form
    # mean_q(sqrt(se_q)) collapses to mean_q(|e_q|); the textbook sqrt(mean_q(se_q))
    # reported for cum/step (GUIDELINE E16b) is a DIFFERENT, larger number. The key
    # name is kept as `terminal_rmse` only because renaming it would invalidate every
    # bank JSON already on disk; the READMEs label the cell "terminal (h=H) MAE".
    term_rmse = float(np.abs(pred_term_mean - q_quant["cum"][:, -1]).mean())
    rv_bias = float(Y["rv"].mean() - q_quant["rv"].mean())

    out = {
        "model": name,
        "K": int(ens_price.shape[1]),
        "quantities": quantities,
        "diagnostics": {"terminal_rmse": term_rmse, "rv_mean_bias": rv_bias},
        "protocol": {
            "role": "conditional forecaster (direct), scored with the generator "
                    "strict PDF metrics (arXiv:2308.01486) for apples-to-apples PS",
            "S_IDX": P.S_IDX, "H": P.H, "K_ensemble": int(ens_price.shape[1]),
            "n_query": int(Nq), "n_boot": P.N_BOOT, "boot_seed": P.BOOT_SEED,
            "query": os.path.basename(P.PS_QUERY),
            "anchored_at": "true logS[:, S_IDX]",
            "forecast_quantities": "cum/step (Nq,K,H trajectory), rv (Nq,K scalar)",
        },
    }
    if extra_meta:
        out.update(extra_meta)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    c = quantities["cum"]; s = quantities["step"]; r = quantities["rv"]
    print(f"[bridge:{name}] cum.CRPS={c['crps']['value']:.5f} "
          f"step.CRPS={s['crps']['value']:.5f} rv.CRPS={r['crps']['value']:.5f} "
          f"cum.cov90={c['coverage90']['value']:.3f} term_rmse={term_rmse:.5f} "
          f"-> {os.path.basename(out_path)}", flush=True)
    return out
