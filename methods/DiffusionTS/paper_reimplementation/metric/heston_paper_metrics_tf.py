"""
"Ours — Heston" paper-metric driver (TensorFlow half): Discriminative + Predictive.

Applies the other TWO of Diffusion-TS's OWN four headline metrics (Yuan & Qiao, ICLR 2024,
Table 1) to our already-generated 5-seed Heston pool — no retraining. This fills the remaining
two rows of the **Ours — Heston** column of the paper-comparison table in
results/Heston/DiffusionTS/README.md (GUIDELINE §15.2 §6), counterpart of compute_tf_metrics.py.

  - Discriminative : Utils/discriminative_metric.py  discriminative_score_metrics (|acc-0.5|, down)
  - Predictive     : Utils/predictive_metric.py       predictive_score_metrics     (TSTR MAE, down)

Both metrics run in the paper's own [0,1] space: real + fake Heston PRICES are placed on [0,1] by a
single global MinMax fit on the real Heston prices (auto_norm equivalent for 1 feature). One metric
value per seed (a fresh GRU judge/predictor per call); reported as mean +/- std across the 5 seeds.

Run with the TF env (needs legacy keras):
  TF_USE_LEGACY_KERAS=1 /home/tbasseras/dts-tf-venv/bin/python heston_paper_metrics_tf.py

Writes/updates: results/heston_paper_metrics.json  (discriminative + predictive keys)
"""
import os
import sys
import json
import numpy as np

REF = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "code", "reference"))
sys.path.insert(0, REF)

from Utils.discriminative_metric import discriminative_score_metrics
from Utils.predictive_metric import predictive_score_metrics

METHOD = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REAL = os.path.join(METHOD, "..", "..", "dataset", "Heston", "heston_S_8192x128.npy")
GEN = os.path.join(METHOD, "generated_paths")
RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
SEED = 123
N_SEEDS = 5


def load01():
    S = np.load(os.path.abspath(REAL)).astype(np.float64)  # (8192,128) price
    lo, hi = float(S.min()), float(S.max())
    real01 = ((S - lo) / (hi - lo)).astype(np.float32)[:, :, None]
    fakes = []
    for i in range(N_SEEDS):
        Xg = np.load(os.path.join(GEN, f"seed_{i}", "generated_paths_8192x128.npy")).astype(np.float64)
        f01 = ((Xg - lo) / (hi - lo)).astype(np.float32)[:, :, None]
        fakes.append(f01)
    return real01, fakes, (lo, hi)


def mean_std(vals):
    return float(np.mean(vals)), float(np.std(vals))


def main():
    np.random.seed(SEED)
    real01, fakes, (lo, hi) = load01()
    n = real01.shape[0]
    print(f"real {real01.shape} lo={lo:.4f} hi={hi:.4f}", flush=True)

    disc, pred = [], []
    for i, f in enumerate(fakes):
        fi = f[:n]
        d, _, _ = discriminative_score_metrics(real01, fi)
        disc.append(float(d))
        p = float(predictive_score_metrics(real01, fi))
        pred.append(p)
        print(f"[seed {i}] discriminative={d:.6f}  predictive={p:.6f}", flush=True)

    disc_m, disc_s = mean_std(disc)
    pred_m, pred_s = mean_std(pred)

    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, "heston_paper_metrics.json")
    out = {}
    if os.path.exists(out_path):
        with open(out_path) as fh:
            out = json.load(fh)
    out["scale_minmax"] = [lo, hi]
    out["n_seeds"] = N_SEEDS
    out["n_samples"] = int(n)
    out["discriminative"] = {"mean": disc_m, "std": disc_s, "seeds": disc}
    out["predictive"] = {"mean": pred_m, "std": pred_s, "seeds": pred}
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({"discriminative": out["discriminative"],
                      "predictive": out["predictive"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
