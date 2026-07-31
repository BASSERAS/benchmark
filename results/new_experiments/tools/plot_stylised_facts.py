"""Eight-panel stylised-facts diagnostic for the README, real vs generated.

Wraps the benchmark-standard ``metrics/plot_diagnostics.plot_diagnostics`` so every method
gets the identical figure. Two deliberate differences from the canonical CLI:

1. It takes explicit ``test.npy`` / bank paths instead of assuming the
   ``results/<Dataset>/<Method>/`` layout, which these experiments do not use.

2. It SUPPRESSES the black Heston theory curve. That curve is the closed-form single-regime
   Heston reference. Neither experiment's DGP is single-regime Heston -- Experiment A is a
   latching drawdown-memory volatility process, Experiment B is an equally-weighted 8-regime
   Heston mixture -- so the curve would be a wrong reference drawn with authority. The
   panels show Real vs Generated only.

Usage:
  python plot_stylised_facts.py --experiment A \
      --model-dir ../experiment_A/LS4 --seed 0 --label LS4
"""
import os
import sys
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(BENCH_ROOT, "metrics"))

import plot_diagnostics as pd_mod  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=("A", "B"))
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="LS4")
    a = ap.parse_args()

    test = os.path.join(BENCH_ROOT, "dataset", "Heston", "new_experiments",
                        f"experiment_{a.experiment}", "test.npy")
    bank = os.path.join(a.model_dir, "generated_paths", f"seed_{a.seed}",
                        "generated_paths_8192x128.npy")
    S_real = np.asarray(np.load(test), dtype=np.float64)
    S_gen = np.asarray(np.load(bank), dtype=np.float64)

    # Force TB=None inside plot_diagnostics: the single-regime Heston theory curve does not
    # describe either experiment's DGP. plot_diagnostics already guards every use of TB.
    pd_mod.ht.compute_theory_bundle = lambda *_a, **_k: (_ for _ in ()).throw(
        RuntimeError("theory curve suppressed: DGP is not single-regime Heston"))

    out_dir = os.path.join(a.model_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "heston_diagnostics.png")
    pd_mod.plot_diagnostics(S_real, S_gen, method=a.label, seed=a.seed, out_path=out)
    print(f"wrote {out}  real={S_real.shape} gen={S_gen.shape}")


if __name__ == "__main__":
    main()
