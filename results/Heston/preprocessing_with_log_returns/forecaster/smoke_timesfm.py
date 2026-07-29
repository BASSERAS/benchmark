"""
TimesFM-1.0-200m SMOKE TEST — confirm we are loading/running the exact same model
that produced the published main-benchmark number before trusting the local 4096
fine-tune's path-shadowing scores.

Mirror of smoke_chronos.py. It reuses the UNTOUCHED harness in
methods/TimesFM/path_shadowing/run_forecaster_ref.py (model_cfg, load_tfm,
forecast_ensemble, score_ensemble) to forecast the 8192 main TEST paths with the
seed-0 fine-tuned checkpoint, and checks that the price-space h32 CRPS reproduces
the known value 3.136352 (results/Heston/TimesFM/forecaster_summary.json ->
finetune.per_seed[0].h32_CRPS).

Match => model + weights + pipeline are correct; the forecaster row is trustworthy.

Usage:
  CUDA_VISIBLE_DEVICES=0 /home/tbasseras/timesfm-v1-venv/bin/python smoke_timesfm.py
"""
import os
import sys
import json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
TFM_PS = os.path.join(BENCH_ROOT, "methods", "TimesFM", "path_shadowing")
sys.path.insert(0, TFM_PS)
import run_forecaster_ref as R  # noqa: E402  (untouched harness)

MODEL_ID = "google/timesfm-1.0-200m-pytorch"
KNOWN_H32 = 3.1363522135848143   # results/Heston/TimesFM finetune per_seed[0]
TOL = 5e-3


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_layers, use_pos = R.model_cfg(MODEL_ID)
    X = np.load(R.TEST_DATA).astype(np.float64)           # (8192,128) main test set
    prefixes = X[:, :R.PREFIX_LEN].astype(np.float32)
    y_fut = X[:, R.PREFIX_LEN:]
    print(f"[smoke] test {X.shape} prefix={R.PREFIX_LEN} horizon={R.HORIZON} "
          f"K={R.K} dev={device}", flush=True)

    sd_path = os.path.join(R.WEIGHTS_DIR, "seed_0_model.pt")
    tfm = R.load_tfm(device, MODEL_ID, num_layers, use_pos, state_dict_path=sd_path)
    rng = np.random.default_rng(0)                        # same RNG seed as harness seed 0
    ens = R.forecast_ensemble(tfm, prefixes, R.HORIZON, R.K, rng, batch_size=R.FC_BATCH)
    res, _ = R.score_ensemble(ens, y_fut)

    got = float(res["h32_CRPS"])
    match = abs(got - KNOWN_H32) < TOL
    out = {"h32_CRPS": got, "h64_CRPS": float(res["h64_CRPS"]),
           "known_h32_CRPS": KNOWN_H32, "abs_diff": abs(got - KNOWN_H32),
           "tol": TOL, "match": bool(match)}
    with open(os.path.join(HERE, "smoke_timesfm.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[smoke] h32_CRPS got={got:.6f} known={KNOWN_H32:.6f} diff={out['abs_diff']:.2e} "
          f"-> {'MATCH' if match else 'MISMATCH'}", flush=True)
    if not match:
        sys.exit(1)


if __name__ == "__main__":
    main()
