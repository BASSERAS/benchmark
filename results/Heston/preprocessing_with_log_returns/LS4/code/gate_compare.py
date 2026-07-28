"""
Seed-0 GATE comparison: LS4 log-return preprocessing vs original LS4.

1. Builds the new 8-panel Heston diagnostics figure for the log-return seed-0
   generated paths (imports metrics/plot_diagnostics.plot_diagnostics unchanged),
   using the 4096-path TEST set as the real reference.
2. Tabulates seed-0 A-metrics and B-curve metrics new-vs-original, side by side,
   with signed % change, so a big deviation is obvious.

Reads:
  results/Heston/LS4/seed_0_metrics.json                          (original)
  results/Heston/preprocessing_with_log_returns/LS4/seed_0_metrics.json (new)
Writes:
  results/Heston/preprocessing_with_log_returns/LS4/plots/heston_diagnostics.png
  results/Heston/preprocessing_with_log_returns/LS4/gate_seed0_compare.md
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LS4_DIR = os.path.dirname(HERE)
BENCH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(LS4_DIR))))
METRICS_DIR = os.path.join(BENCH, "metrics")
DATA_DIR = os.path.join(BENCH, "dataset", "Heston", "preprocessing_with_log_returns")
ORIG_DIR = os.path.join(BENCH, "results", "Heston", "LS4")

sys.path.insert(0, METRICS_DIR)
from plot_diagnostics import plot_diagnostics   # noqa: E402

N = 4096


def make_figure():
    S_real = np.load(os.path.join(DATA_DIR, f"heston_S_test_{N}x128.npy"))
    S_gen = np.load(os.path.join(LS4_DIR, "generated_paths", "seed_0",
                                 f"generated_paths_{N}x128.npy"))
    out = os.path.join(LS4_DIR, "plots", "heston_diagnostics.png")
    print(f"Real: {S_real.shape}  Gen: {S_gen.shape}")
    plot_diagnostics(S_real, S_gen, method="LS4 (log-ret preproc)", seed=0, out_path=out)
    return out


def load(p):
    with open(p) as f:
        return json.load(f)


def main():
    fig = make_figure()

    new = load(os.path.join(LS4_DIR, "seed_0_metrics.json"))
    orig = load(os.path.join(ORIG_DIR, "seed_0_metrics.json"))

    a_keys = sorted([k for k in new if k.startswith("A")],
                    key=lambda k: int(k[1:].split("_")[0]))
    # headline B keys: the six funct MSE + the six %err
    b_funct = [k for k in new if k.startswith("B_") and k.endswith("_funct")]
    b_pct = [k for k in new if k.startswith("B_") and k.endswith("_funct_pct")]
    b_keys = sorted(b_funct) + sorted(b_pct)

    lines = []
    lines.append("# Seed-0 GATE: LS4 log-return preproc vs original LS4\n")
    lines.append(f"New diagnostics figure: `{os.path.relpath(fig, LS4_DIR)}`\n")
    lines.append(f"Original figure: `../../LS4/plots/heston_diagnostics.png`\n")

    def tab(title, keys):
        lines.append(f"\n## {title}\n")
        lines.append("| Metric | Original s0 | New s0 | delta% |")
        lines.append("|--------|------------:|-------:|-------:|")
        for k in keys:
            o = orig.get(k)
            nv = new.get(k)
            if o is None or nv is None:
                lines.append(f"| {k} | {o} | {nv} | - |")
                continue
            d = (nv - o) / (abs(o) + 1e-12) * 100
            lines.append(f"| {k} | {o:.6g} | {nv:.6g} | {d:+.1f}% |")

    tab("A-metrics (seed 0)", a_keys)
    tab("B curve funct MSE + %err (seed 0)", b_keys)

    md = "\n".join(lines) + "\n"
    out_md = os.path.join(LS4_DIR, "gate_seed0_compare.md")
    with open(out_md, "w") as f:
        f.write(md)
    print("\n" + md)
    print(f"Saved comparison -> {out_md}")


if __name__ == "__main__":
    main()
