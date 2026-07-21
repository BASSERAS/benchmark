"""
COSCI-GAN paper reproduction driver — EEG eye-state Table 4 (correlation-matrix MAE).

Produces the single committed, traceable JSON that backs every number in
  ../README.md  ("4. Results — ours vs paper")
namely  ../results/eeg_table4.json.

It computes the catch22 cross-channel correlation-MAE metric (Seyfi et al.,
NeurIPS 2022, Table 4) for:

  * ours     — our re-trained COSCI-GAN generated pool
               (../results/ours_eeg_generated_label{L}.npy, produced by ../train_eeg.py)
  * GroupGAN — the authors' OWN released COSCI-GAN samples  (metric self-validation)
  * timeGAN  — the authors' OWN released TimeGAN samples     (metric self-validation)
  * FF        — the authors' OWN released Fourier-Flow samples (metric self-validation)

The three authors' methods are scored ONLY to prove that our metric port
reproduces the paper's published Table-4 numbers exactly (0.111 / 0.257 / 0.146),
decoupling "is the metric right" from "is our training right". The authors' npy
files are NOT redistributed in this repo; pass their directory with --authors_dir
(they live in the upstream repo at
  Experiments/Correlation_Analysis/EEG_Dataset/generated_datasets/).
When --authors_dir is absent, only the `ours` entry is (re)computed and the
authors' entries already present in the JSON are preserved.

The metric functions are imported verbatim from eeg_corr_mae.py (same folder),
which is the faithful port validated against the paper notebook.

Usage (exact run path in ../README.md §5):
  /home/tbasseras/gpu-venv/bin/python eeg_table4.py \
      --real ../dataset/EEG_Eye_State_ZeroOne_chop_5best_1.csv \
      --ours ../results/ours_eeg_generated_label1.npy \
      --authors_dir /tmp/COSCI-GAN-src/Experiments/Correlation_Analysis/EEG_Dataset/generated_datasets \
      --label 1 --out ../results/eeg_table4.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eeg_corr_mae import build_channel_frames, corr_mae, load_real  # noqa: E402

# paper Table 4 published values (Seyfi et al., NeurIPS 2022), EEG eye-state
PAPER = {"ours": 0.111, "GroupGAN": 0.111, "timeGAN": 0.257, "FF": 0.146}
# upstream file prefixes (GroupGAN = COSCI-GAN's old name)
AUTHOR_PREFIX = {"GroupGAN": "GroupGAN", "timeGAN": "timeGAN", "FF": "FF"}


def score(real_data, gen_data, n_channels, tag):
    df_real = build_channel_frames(real_data, "real")
    df_gen = build_channel_frames(gen_data, tag)
    mean, std, maes = corr_mae(df_real, df_gen, n_channels, "real", tag)
    return {"mean": mean, "std": std, "per_pair": [float(m) for m in maes]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--ours", required=True, help="our re-trained gen .npy (N,100,5)")
    ap.add_argument("--authors_dir", default="", help="dir with the authors' released npy (optional)")
    ap.add_argument("--label", type=int, default=1)
    ap.add_argument("--n_channels", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    real_data = load_real(a.real, a.n_channels)          # (N, 100, 5)

    # preserve any existing entries (so a --ours-only rerun keeps authors' numbers)
    out = {"metric": "catch22 cross-channel correlation-MAE (paper Table 4)",
           "dataset": f"EEG_Eye_State_ZeroOne_chop_5best_{a.label}",
           "n_pairs": a.n_channels * (a.n_channels + 1) // 2,
           "paper": PAPER, "methods": {}}
    if os.path.exists(a.out):
        try:
            out = json.load(open(a.out))
            out.setdefault("methods", {})
        except Exception:
            pass

    # our re-trained COSCI-GAN
    ours = np.load(a.ours)
    print(f"[ours] gen{ours.shape}", flush=True)
    out["methods"]["ours"] = score(real_data, ours, a.n_channels, "ours")
    print(f"[ours] corr-MAE mean={out['methods']['ours']['mean']:.4f} "
          f"std={out['methods']['ours']['std']:.4f}", flush=True)

    # authors' released samples (metric self-validation)
    if a.authors_dir:
        for name, prefix in AUTHOR_PREFIX.items():
            f = os.path.join(a.authors_dir,
                             f"{prefix}_EEG_Eye_State_ZeroOne_chop_5best_{a.label}.npy")
            if not os.path.exists(f):
                print(f"[{name}] MISSING {f} — skipped", flush=True)
                continue
            g = np.load(f)
            if g.ndim == 2:                              # (N,500) -> (N,100,5)
                g = g.reshape(g.shape[0], -1, a.n_channels)
            print(f"[{name}] gen{g.shape}", flush=True)
            out["methods"][name] = score(real_data, g, a.n_channels, name)
            print(f"[{name}] corr-MAE mean={out['methods'][name]['mean']:.4f} "
                  f"std={out['methods'][name]['std']:.4f}  (paper {PAPER[name]})", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
