"""Cross-check our runs with the authors' OWN diagnostics, not our std_ratio.

Motivation. ``std_ratio`` and the derived ``gap`` are ours, not the paper's --
the paper reports no numbers for Heston at all. Worse, a seed replicate of the
same arm moved ``gap`` by 0.27 against a base value of 0.50, so that statistic
cannot rank hyperparameters. This script therefore evaluates the generated paths
with the diagnostics the authors ship in
``code/reference/utils/plot_metrics.py``, which never touch the MLE:

  * ``my_acf``            autocorrelation of returns and of squared returns.
                          Heston returns are serially uncorrelated but their
                          squares are not (vol clustering), so this separates
                          "looks like noise" from "reproduces the dynamics".
  * ``plot_corr_matrix``  cross-channel correlation. With d=2 = (price,
                          variance) the single off-diagonal entry IS the
                          leverage effect, i.e. rho -- measured directly from
                          the paths, with no estimator in the way. This is the
                          independent check on the rho claim.

Everything is computed on log returns, matching ``reproduce_heston.py``:
``log_returns[:, 1:] = diff(log(levels), axis=1)``.

    cd metric && python check_author_metrics.py t00 t09 t11
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT.parent / "code" / "reference" / "utils"))

from plot_metrics import my_acf  # noqa: E402

DS = ROOT / "dataset" / "X_heston_paper_5000x253x2.npy"
OUT = ROOT / "results" / "sweep"
LAG = 60


def log_returns(levels):
    """Levels (M, T, 2) -> log returns (M, T-1, 2), the transform the pipeline trains on."""
    return np.diff(np.log(levels.astype(np.float64)), axis=1)


def acf_profile(R, lag=LAG):
    """Mean ACF of returns and of squared returns, averaged over paths."""
    plain = np.array([my_acf(s, lag_len=lag) for s in R])
    sq = np.array([my_acf(s ** 2, lag_len=lag) for s in R])
    return plain.mean(axis=0), sq.mean(axis=0)


def channel_corr(R):
    """Per-path corr(price return, variance return): the leverage effect."""
    return np.array([np.corrcoef(p, rowvar=False)[0, 1] for p in R])


def describe(name, R_data, R_gen):
    d0, g0 = R_data[:, :, 0], R_gen[:, :, 0]
    acf_d, acf2_d = acf_profile(d0)
    acf_g, acf2_g = acf_profile(g0)
    cd, cg = channel_corr(R_data), channel_corr(R_gen)

    # Lags 1..20 carry the clustering; lag 0 is 1 by construction, uninformative.
    return dict(
        tag=name,
        acf_r_data=float(np.abs(acf_d[1:21]).mean()),
        acf_r_gen=float(np.abs(acf_g[1:21]).mean()),
        acf_r2_data=float(acf2_d[1:21].mean()),
        acf_r2_gen=float(acf2_g[1:21].mean()),
        corr_data=float(np.nanmean(cd)), corr_gen=float(np.nanmean(cg)),
        corr_sd_data=float(np.nanstd(cd)), corr_sd_gen=float(np.nanstd(cg)),
        std_data=float(d0.std()), std_gen=float(g0.std()),
        kurt_data=float(((d0 - d0.mean()) ** 4).mean() / d0.std() ** 4),
        kurt_gen=float(((g0 - g0.mean()) ** 4).mean() / g0.std() ** 4),
    )


def main():
    tags = sys.argv[1:] or ["t00"]
    R_data = log_returns(np.load(DS))

    rows = []
    for tag in tags:
        hits = sorted(OUT.glob(f"X_sbbts_*_{tag}.npy"))
        if not hits:
            print(f"  {tag}: no generated paths on disk, skipped", flush=True)
            continue
        gen = np.load(hits[0])
        keep = np.isfinite(gen).all(axis=(1, 2)) & (gen > 0).all(axis=(1, 2))
        gen = gen[keep]
        print(f"  {tag}: {hits[0].name}  {gen.shape}  ({(~keep).sum()} dropped)", flush=True)
        rows.append(describe(tag, R_data, log_returns(gen)))

    if not rows:
        return

    r0 = rows[0]
    print(f"\ndata reference (M={len(R_data)}, N={R_data.shape[1]})")
    print(f"  mean|ACF(r)| lags1-20   {r0['acf_r_data']:.4f}   "
          f"(uncorrelated returns -> ~1/sqrt(N) = {1 / np.sqrt(R_data.shape[1]):.4f})")
    print(f"  mean ACF(r^2) lags1-20  {r0['acf_r2_data']:.4f}   (vol clustering)")
    print(f"  corr(dlogS, dlogv)      {r0['corr_data']:+.4f} +- {r0['corr_sd_data']:.4f}"
          f"   <- leverage effect: the MLE-free view of rho")
    print(f"  return std / kurtosis   {r0['std_data']:.5f} / {r0['kurt_data']:.3f}")

    print(f"\n{'tag':>6} | {'|ACF(r)|':>8} | {'ACF(r^2)':>8} | {'corr(S,v)':>19} | "
          f"{'ret std':>8} | {'kurt':>6}")
    print("-" * 78)
    for r in rows:
        print(f"{r['tag']:>6} | {r['acf_r_gen']:>8.4f} | {r['acf_r2_gen']:>8.4f} | "
              f"{r['corr_gen']:>+7.4f} +- {r['corr_sd_gen']:<8.4f} | "
              f"{r['std_gen']:>8.5f} | {r['kurt_gen']:>6.3f}")
    print(f"{'DATA':>6} | {r0['acf_r_data']:>8.4f} | {r0['acf_r2_data']:>8.4f} | "
          f"{r0['corr_data']:>+7.4f} +- {r0['corr_sd_data']:<8.4f} | "
          f"{r0['std_data']:>8.5f} | {r0['kurt_data']:>6.3f}")

    print("\nratios to data (1.000 = matched):")
    print(f"{'tag':>6} | {'ACF(r^2)':>9} | {'corr spread':>11} | {'ret std':>8} | {'kurt':>8}")
    print("-" * 56)
    for r in rows:
        print(f"{r['tag']:>6} | {r['acf_r2_gen'] / r['acf_r2_data']:>9.3f} | "
              f"{r['corr_sd_gen'] / r['corr_sd_data']:>11.3f} | "
              f"{r['std_gen'] / r['std_data']:>8.3f} | "
              f"{r['kurt_gen'] / r['kurt_data']:>8.3f}")


if __name__ == "__main__":
    main()
