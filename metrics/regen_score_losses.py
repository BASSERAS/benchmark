"""Regenerate the A18 discriminative + A19 predictive loss-curve CSVs and plots
for one method, TRAINING ON THE DISC SPLIT (dataset seed 2) — matching the
current `compute_all.py` data flow (lines 141, 184, 196) exactly.

Why this exists
---------------
The per-seed loss CSVs shipped with each method were produced before the
"test-set-everywhere" split was finalised, so it was not verifiable that the
classifier/forecaster loss curves were trained against the seed-2 disc split.
This script re-derives ONLY the four loss CSVs per metric-seed:

    results/Heston/<M>/seed_{i}_disc_{gru,mlp}_loss.csv   (cols: step,train_bce)
    results/Heston/<M>/seed_{i}_pred_{gru,mlp}_loss.csv   (cols: step,train_mae)

and then regenerates the two figures via plot_score_losses.py:

    results/Heston/<M>/plots/disc_classifier_loss.png
    results/Heston/<M>/plots/pred_score_loss.png

It does NOT touch any seed_*_metrics.json — the validated A18/A19 SCORES are
left untouched; only the illustrative loss curves are regenerated so the plots
provably show the loss for training on the disc split (dataset seed 2).

The "real" class = seed-2 disc draw (heston_S_disc); the "fake" class = the
method's generated paths for that metric-seed. Both are converted to log-returns
exactly as compute_all.py does.

Usage:
    /home/tbasseras/gpu-venv/bin/python metrics/regen_score_losses.py --method LS4
    # optionally pin a GPU: CUDA_VISIBLE_DEVICES=1 ... --device cuda
"""
from __future__ import annotations
import argparse, csv, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from discriminative_score import compute_discriminative_score   # noqa: E402
from predictive_score import compute_predictive_score           # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--dataset", default="Heston")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--disc-steps", type=int, default=2000)
    ap.add_argument("--pred-steps", type=int, default=5000)
    args = ap.parse_args()

    dataset_dir  = os.path.join(REPO, "dataset", args.dataset)
    gen_dir      = os.path.join(REPO, "methods", args.method, "generated_paths")
    results_dir  = os.path.join(REPO, "results", args.dataset, args.method)
    os.makedirs(results_dir, exist_ok=True)

    # real class for A18/A19 = the seed-2 disc split (identical to compute_all.py)
    S_disc = np.load(os.path.join(dataset_dir, "heston_S_disc_8192x128.npy"))  # (8192,128)
    lr_disc = np.diff(np.log(S_disc), axis=1)                                  # (8192,127)

    for seed in range(5):
        fake_path = os.path.join(gen_dir, f"seed_{seed}",
                                 "generated_paths_8192x128.npy")
        fake = np.load(fake_path)                        # (8192,128)
        lr_fake = np.diff(np.log(fake), axis=1)          # (8192,127)

        # ── A18 discriminative (GRU + MLP), trained real=disc(seed2) vs fake ──
        d18 = compute_discriminative_score(lr_disc, lr_fake,
                                           n_steps=args.disc_steps,
                                           device=args.device)
        for arch in ("gru", "mlp"):
            with open(os.path.join(results_dir,
                                   f"seed_{seed}_disc_{arch}_loss.csv"),
                      "w", newline="") as lf:
                w = csv.DictWriter(lf, fieldnames=["step", "train_bce"])
                w.writeheader(); w.writerows(d18[f"loss_history_{arch}"])

        # ── A19 predictive TSTR (GRU + MLP), same disc(seed2) real set ────────
        d19 = compute_predictive_score(lr_disc, lr_fake,
                                       n_steps=args.pred_steps,
                                       device=args.device)
        for arch in ("gru", "mlp"):
            with open(os.path.join(results_dir,
                                   f"seed_{seed}_pred_{arch}_loss.csv"),
                      "w", newline="") as lf:
                w = csv.DictWriter(lf, fieldnames=["step", "train_mae"])
                w.writeheader(); w.writerows(d19[f"loss_history_{arch}"])

        print(f"[{args.method}] seed {seed}: "
              f"disc gru={d18['disc_score_gru']:.4f} mlp={d18['disc_score_mlp']:.4f} | "
              f"pred gru={d19['pred_score_gru']:.4f} mlp={d19['pred_score_mlp']:.4f}",
              flush=True)

    # ── regenerate the two figures from the fresh CSVs ────────────────────────
    import subprocess
    subprocess.run([sys.executable,
                    os.path.join(HERE, "plot_score_losses.py"),
                    "--method", args.method, "--dataset", args.dataset],
                   check=True)
    print(f"[{args.method}] done.", flush=True)


if __name__ == "__main__":
    main()
