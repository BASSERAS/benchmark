"""
TimeVAE — paper reproduction (scoring half).  Runs in the legacy-keras TF env
(``/home/tbasseras/dts-tf-venv``), CPU only.

Scores the arrays produced by ``reproduce_paper.py`` with the **original TimeGAN
metric code** (Yoon et al., NeurIPS 2019) — verbatim from
``methods/TimeGAN/code/reference/metrics/{discriminative_metrics,predictive_metrics}.py``.
These are exactly the two metrics the TimeVAE paper uses for Table 1 / Table 2:

  - discriminative_score = |accuracy - 0.5| of a post-hoc GRU judge (lower = better)
  - predictive_score     = TSTR one-step-ahead MAE of a post-hoc GRU (lower = better)

Both are computed on the [0,1] scaled arrays (the space TimeVAE generates in).  For
each seed the real set is scored against that seed's generated set; the metric's own
internal randomness (train/test split, GRU init) is averaged by ``--metric-iter``
repeats, then we report mean +/- std across seeds — the benchmark's 5-seed convention.

Usage:
  TF_USE_LEGACY_KERAS=1 /home/tbasseras/dts-tf-venv/bin/python score_paper.py \
      --dataset sine --artifacts ../results/artifacts --metric-iter 3 \
      --out ../results/sine_paper_metrics.json
"""
import argparse, glob, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# original TimeGAN metric code + its utils
TG_REF = os.path.abspath(os.path.join(HERE, "..", "..", "..", "TimeGAN", "code", "reference"))
sys.path.insert(0, TG_REF)                       # utils.py
sys.path.insert(0, os.path.join(TG_REF, "metrics"))

# The TimeGAN reference metrics use the TF1 optimizer API (tf.train.AdamOptimizer).
# Their in-module compat shim patches many v1 names onto `tf` but not `tf.train`,
# which TF 2.21 removed.  Restore it here (same fix as timegan.py:47) — `tf` is a
# singleton, so this propagates into discriminative_metrics.py / predictive_metrics.py.
import tensorflow as tf                             # noqa: E402
import tensorflow.compat.v1 as _tf1                 # noqa: E402
if not hasattr(tf.train, "AdamOptimizer"):
    tf.train = _tf1.train
# predictive_metrics.py calls tf.losses.absolute_difference (a v1 API).  Under
# TF_USE_LEGACY_KERAS=1, tf.losses resolves to tf_keras losses which lacks it, and
# the in-module shim only patches tf.losses when absent — so force the v1 namespace.
tf.losses = _tf1.losses

from discriminative_metrics import discriminative_score_metrics  # noqa: E402
from predictive_metrics import predictive_score_metrics          # noqa: E402

# The metric modules' in-module shim replaces tf.contrib.layers.fully_connected
# with a Keras Dense built inside the TF1 graph.  Under disable_v2_behavior those
# Dense variables are NOT collected into the discriminator/predictor var-list, so
# the output head never trains -> the judge returns a degenerate accuracy of
# exactly 0 or 1 and the score is pinned at 0.5 for ANY input (even real-vs-real).
# fully_connected is looked up at call time inside the metric, so overriding it
# here with the functional tf1.layers.dense (which registers vars in the graph)
# repairs both metrics.  Verified: disc(real_half_A, real_half_B) ~= 0.
def _fc_dense(inputs, num_outputs, activation_fn=None, **kw):  # noqa: E306
    return _tf1.layers.dense(inputs, num_outputs, activation=activation_fn)
tf.contrib.layers.fully_connected = _fc_dense


# Paper targets — filled ONLY with values verified against the PDF.
# Table 1 (discriminative) and Table 2 (predictive), 100%-train TimeVAE rows,
# both confirmed by reading arXiv:2111.08095v3 (pages 7 and 9).
# NOTE on predictive: "lower = better" means "close to the real-data floor"
# (the paper's "Original" LSTM row), not "close to 0".  For sine the Original
# floor is 0.213 and TimeVAE matches it exactly (0.213 +/- 0.000).
PAPER_TARGET = {
    "sine":   {"discriminative": "0.021 +/- 0.040", "predictive": "0.213 +/- 0.000"},
    "stockv": {"discriminative": "0.009 +/- 0.009", "predictive": "0.019 +/- 0.001"},
    "energy": {"discriminative": "0.498 +/- 0.006", "predictive": "0.268 +/- 0.004"},
    "air":    {"discriminative": "0.381 +/- 0.037", "predictive": "0.013 +/- 0.002"},
}
# Real-data predictive floor ("Original" LSTM, Table 2, 100% train) — for context.
PAPER_ORIGINAL_PREDICTIVE = {
    "sine": "0.213 +/- 0.000", "stockv": "0.019 +/- 0.004",
    "energy": "0.229 +/- 0.002", "air": "0.004 +/- 0.000",
}


def _stats(v):
    v = np.asarray(v, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=0)), "runs": v.tolist()}


def main(a):
    t0 = time.time()
    real = np.load(os.path.join(a.artifacts, f"{a.dataset}_real_scaled.npy")).astype(np.float64)
    gen_files = sorted(glob.glob(os.path.join(a.artifacts, f"{a.dataset}_gen_seed*.npy")))
    if not gen_files:
        raise SystemExit(f"no generated files for {a.dataset} in {a.artifacts}")
    print(f"[data] real {real.shape}  gens {len(gen_files)}", flush=True)

    per_seed = []
    disc_all, pred_all = [], []
    for gf in gen_files:
        seed = int(os.path.basename(gf).split("seed")[1].split(".")[0])
        gen = np.load(gf).astype(np.float64)
        n = min(real.shape[0], gen.shape[0])
        r, g = real[:n], gen[:n]

        d_runs, p_runs = [], []
        for it in range(a.metric_iter):
            d = float(discriminative_score_metrics(r, g))
            p = float(predictive_score_metrics(r, g))
            d_runs.append(d); p_runs.append(p)
            print(f"[{a.dataset} seed {seed}] iter {it}: disc={d:.4f} pred={p:.4f}", flush=True)
        d_m, p_m = float(np.mean(d_runs)), float(np.mean(p_runs))
        disc_all.append(d_m); pred_all.append(p_m)
        per_seed.append({"seed": seed, "n": n,
                         "discriminative": d_m, "predictive": p_m,
                         "disc_runs": d_runs, "pred_runs": p_runs})
        print(f"[{a.dataset} seed {seed}] disc={d_m:.4f} pred={p_m:.4f}", flush=True)

    out = {
        "dataset": a.dataset,
        "metric_source": "TimeGAN (Yoon et al. 2019) discriminative_metrics.py / predictive_metrics.py",
        "metric_iter": a.metric_iter,
        "paper_target": PAPER_TARGET.get(a.dataset, {}),
        "discriminative": _stats(disc_all),
        "predictive": _stats(pred_all),
        "per_seed": per_seed,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== TimeVAE / %s reproduction ===" % a.dataset)
    print(f"discriminative : {out['discriminative']['mean']:.4f} +/- {out['discriminative']['std']:.4f}"
          f"   (paper {out['paper_target'].get('discriminative','?')})")
    print(f"predictive     : {out['predictive']['mean']:.4f} +/- {out['predictive']['std']:.4f}"
          f"   (paper {out['paper_target'].get('predictive','?')})")
    print(f"total {time.time() - t0:.1f}s -> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sine")
    p.add_argument("--artifacts", default=os.path.join(HERE, "..", "results", "artifacts"))
    p.add_argument("--metric-iter", type=int, default=3, dest="metric_iter")
    p.add_argument("--out", default=os.path.join(HERE, "..", "results", "sine_paper_metrics.json"))
    main(p.parse_args())
