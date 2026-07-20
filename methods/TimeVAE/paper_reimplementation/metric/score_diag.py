"""
Score diagnostic arrays (recon / prior) with the verbatim TimeGAN discriminative
metric — same judge score_paper.py uses.  Legacy-keras TF env, CPU.

Usage:
  TF_USE_LEGACY_KERAS=1 /home/tbasseras/dts-tf-venv/bin/python score_diag.py \
      --iter 3 --tags base_wt3 base_wt8 seas_wt3
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TG_REF = os.path.abspath(os.path.join(HERE, "..", "..", "..", "TimeGAN", "code", "reference"))
sys.path.insert(0, TG_REF)
sys.path.insert(0, os.path.join(TG_REF, "metrics"))

import tensorflow as tf                          # noqa: E402
import tensorflow.compat.v1 as _tf1              # noqa: E402
if not hasattr(tf.train, "AdamOptimizer"):
    tf.train = _tf1.train
tf.losses = _tf1.losses
from discriminative_metrics import discriminative_score_metrics  # noqa: E402


def _fc_dense(inputs, num_outputs, activation_fn=None, **kw):
    return _tf1.layers.dense(inputs, num_outputs, activation=activation_fn)
tf.contrib.layers.fully_connected = _fc_dense


def score(real, arr, it):
    n = min(real.shape[0], arr.shape[0])
    runs = [float(discriminative_score_metrics(real[:n], arr[:n])) for _ in range(it)]
    return float(np.mean(runs)), runs


def main(a):
    art = os.path.join(HERE, "..", "results", "artifacts")
    diag = os.path.join(HERE, "..", "results", "diag")
    real = np.load(os.path.join(art, "sine_real_scaled.npy")).astype(np.float64)
    print(f"real {real.shape} std={real.std():.4f}\n")
    for tag in a.tags:
        for kind in ("recon", "prior"):
            f = os.path.join(diag, f"sine_diag_{tag}_{kind}.npy")
            if not os.path.exists(f):
                print(f"  [skip] {tag}/{kind}: missing"); continue
            arr = np.load(f).astype(np.float64)
            m, runs = score(real, arr, a.iter)
            print(f"  {tag:12s} {kind:5s}  disc={m:.4f}  std={arr.std():.4f}  "
                  f"runs={[round(r,3) for r in runs]}", flush=True)
        print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--iter", type=int, default=3)
    p.add_argument("--tags", nargs="+", default=["base_wt3"])
    main(p.parse_args())
