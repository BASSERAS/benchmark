"""Decide whether TF32 matmuls change the TimeDiT DDPM sampler's OUTPUT DISTRIBUTION.

The honest test is not "are the two panels bit-identical" -- they cannot be, TF32
truncates the matmul mantissa and DDPM is a 1000-step recursion, so any epsilon
diverges chaotically. The question that matters for the 1M scenario bank is
whether the *distribution* moves by more than the model's own seed-to-seed noise.

So we compute the same statistic vector on five panels:
    fp32   : verify/gen_fp32_4096x128.npy   (seed 12345 sampling RNG)
    tf32   : verify/gen_tf32_4096x128.npy   (same RNG, TF32 matmuls)
    seed0  : ../generated_paths/seed_0/...  (training seed 0, fp32)
    seed1  : ../generated_paths/seed_1/...  (training seed 1, fp32)
    real   : the Heston test split

and report, per statistic:
    delta_precision = |fp32 - tf32|          <- the cost of TF32
    delta_seed      = |seed0 - seed1|        <- the noise floor we already accept
    ratio           = delta_precision / delta_seed

TF32 is declared lossless when ratio < 1 on every statistic the path-shadowing
embedding depends on (rolling vol w=2.0, ACF w=1.0, returns w=1.0, cum-path w=0.5
-- see ../../GUIDELINE.md §9.2), i.e. the precision change is smaller than a
difference we already treat as irreducible.

Usage:
  python compare_tf32.py
"""
import os
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TIMEDIT_DIR = os.path.dirname(HERE)
BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(TIMEDIT_DIR))))
DATA_DIR = os.path.join(BENCH_ROOT, "dataset", "Heston",
                        "preprocessing_with_log_returns")
VERIFY = os.path.join(HERE, "verify")

# statistics that the PS embedding actually leans on, in embedding-weight order
PS_CRITICAL = ["ret_std", "acf1_ret", "acf1_absret", "acf5_absret",
               "acf10_absret", "rvol_mean", "rvol_std"]


def stats(S):
    """Statistic vector on a (M,128) price panel."""
    R = np.log(S[:, 1:] / S[:, :-1])          # (M,127) log-returns
    flat = R.ravel()
    m, s = flat.mean(), flat.std()

    def acf(x, lag):
        a, b = x[:, :-lag], x[:, lag:]
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        num = (a * b).mean(axis=1)
        den = a.std(axis=1) * b.std(axis=1) + 1e-12
        return float((num / den).mean())

    absR = np.abs(R)
    # 10-step rolling realised vol, the w=2.0 embedding block
    w = 10
    cs = np.cumsum(R ** 2, axis=1)
    rv = np.sqrt((cs[:, w:] - cs[:, :-w]) / w)

    out = {
        "term_mean": float(S[:, -1].mean()),
        "term_std": float(S[:, -1].std()),
        "term_skew": float(((S[:, -1] - S[:, -1].mean()) ** 3).mean() / S[:, -1].std() ** 3),
        "term_kurt": float(((S[:, -1] - S[:, -1].mean()) ** 4).mean() / S[:, -1].std() ** 4),
        "ret_mean": float(m),
        "ret_std": float(s),
        "ret_skew": float(((flat - m) ** 3).mean() / s ** 3),
        "ret_kurt": float(((flat - m) ** 4).mean() / s ** 4),
        "acf1_ret": acf(R, 1),
        "acf1_absret": acf(absR, 1),
        "acf5_absret": acf(absR, 5),
        "acf10_absret": acf(absR, 10),
        "rvol_mean": float(rv.mean()),
        "rvol_std": float(rv.std()),
        "path_min": float(S.min()),
        "path_max": float(S.max()),
    }
    return out


def load(p):
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return np.load(p).astype(np.float64)


def main():
    panels = {
        "fp32": os.path.join(VERIFY, "gen_fp32_4096x128.npy"),
        "tf32": os.path.join(VERIFY, "gen_tf32_4096x128.npy"),
        "seed0": os.path.join(TIMEDIT_DIR, "generated_paths", "seed_0",
                              "generated_paths_4096x128.npy"),
        "seed1": os.path.join(TIMEDIT_DIR, "generated_paths", "seed_1",
                              "generated_paths_4096x128.npy"),
        "real": os.path.join(DATA_DIR, "heston_S_test_4096x128.npy"),
    }
    st = {k: stats(load(v)) for k, v in panels.items()}

    keys = list(st["fp32"].keys())
    rows, worst_ratio, worst_key = [], 0.0, None
    for k in keys:
        dp = abs(st["fp32"][k] - st["tf32"][k])
        ds = abs(st["seed0"][k] - st["seed1"][k])
        r = dp / ds if ds > 1e-12 else (float("inf") if dp > 1e-12 else 0.0)
        rows.append((k, st["real"][k], st["fp32"][k], st["tf32"][k], dp, ds, r))
        if k in PS_CRITICAL and r > worst_ratio:
            worst_ratio, worst_key = r, k

    hdr = f"{'stat':<14}{'real':>12}{'fp32':>12}{'tf32':>12}{'d_prec':>12}{'d_seed':>12}{'ratio':>9}"
    print(hdr)
    print("-" * len(hdr))
    for k, rl, f, t, dp, ds, r in rows:
        mark = " *" if k in PS_CRITICAL else "  "
        print(f"{k:<14}{rl:>12.5f}{f:>12.5f}{t:>12.5f}{dp:>12.3e}{ds:>12.3e}{r:>9.3f}{mark}")
    print("\n* = statistic the path-shadowing embedding depends on (GUIDELINE §9.2)")

    verdict = ("LOSSLESS: TF32-vs-fp32 shift is below the seed-to-seed noise floor "
               "on every PS-critical statistic"
               if worst_ratio < 1.0 else
               f"REJECT: '{worst_key}' moves {worst_ratio:.2f}x the seed noise under TF32")
    print(f"\nworst PS-critical ratio = {worst_ratio:.3f} ({worst_key})")
    print(f"VERDICT: {verdict}")

    out = {"stats": st,
           "table": [{"stat": k, "real": rl, "fp32": f, "tf32": t,
                      "delta_precision": dp, "delta_seed": ds, "ratio": r,
                      "ps_critical": k in PS_CRITICAL}
                     for k, rl, f, t, dp, ds, r in rows],
           "worst_ps_critical_ratio": worst_ratio,
           "worst_ps_critical_stat": worst_key,
           "verdict": verdict}
    with open(os.path.join(VERIFY, "tf32_verdict.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"-> {os.path.join(VERIFY, 'tf32_verdict.json')}")


if __name__ == "__main__":
    main()
