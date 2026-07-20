"""
Paper-reimplementation metric driver (TensorFlow half): Discriminative + Predictive.

Reproduces two of the four Diffusion-TS (Yuan & Qiao, ICLR 2024) paper metrics on the
Stocks (seq-len 24) reproduction, using the ORIGINAL repo metric code verbatim:
  - Discriminative : Utils/discriminative_metric.py  discriminative_score_metrics
                     (Yoon 2019 post-hoc GRU classifier, |acc - 0.5|)
  - Predictive     : Utils/predictive_metric.py       predictive_score_metrics
                     (Yoon 2019 TSTR one-step-ahead MAE)

Both metrics run on the [0,1]-normalized data. Reported as mean +/- 95% CI (t-dist, dof=4)
over N_ITER independent classifier/predictor trainings.

Run with the TensorFlow env:
  /home/tbasseras/dts-tf-venv/bin/python compute_tf_metrics.py

Writes: results/stocks_tf_metrics.json
"""
import os
import sys
import json
import numpy as np
import scipy.stats

# Make the reference repo importable (Utils/ lives there)
REF = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "code", "reference"))
sys.path.insert(0, REF)

from Utils.discriminative_metric import discriminative_score_metrics
from Utils.predictive_metric import predictive_score_metrics

N_ITER = 5
SEED = 123


def mean_ci(results):
    mean = float(np.mean(results))
    sem = scipy.stats.sem(results)
    ci = float(sem * scipy.stats.t.ppf((1 + 0.95) / 2., 5 - 1))
    return mean, ci


def load_data():
    fake = np.load(os.path.join(REF, "OUTPUT", "stock", "ddpm_fake_stock.npy"))
    truth = np.load(os.path.join(REF, "OUTPUT", "stock", "samples", "stock_norm_truth_24_train.npy"))
    n = min(fake.shape[0], truth.shape[0])
    return truth[:n].astype(np.float32), fake[:n].astype(np.float32)


def main():
    np.random.seed(SEED)
    ori, fake = load_data()
    print(f"ori {ori.shape}  fake {fake.shape}", flush=True)

    # --- Discriminative ---
    disc = []
    for i in range(N_ITER):
        score, _, _ = discriminative_score_metrics(ori, fake[:ori.shape[0]])
        disc.append(float(score))
        print(f"[Discriminative] iter {i}: {score:.6f}", flush=True)
    disc_mean, disc_ci = mean_ci(disc)

    # --- Predictive ---
    pred = []
    for i in range(N_ITER):
        score = float(predictive_score_metrics(ori, fake[:ori.shape[0]]))
        pred.append(score)
        print(f"[Predictive] iter {i}: {score:.6f}", flush=True)
    pred_mean, pred_ci = mean_ci(pred)

    out = {
        "discriminative": {"mean": disc_mean, "ci95": disc_ci, "runs": disc},
        "predictive": {"mean": pred_mean, "ci95": pred_ci, "runs": pred},
        "n_iter": N_ITER,
        "n_samples": int(ori.shape[0]),
    }
    outdir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "stocks_tf_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
