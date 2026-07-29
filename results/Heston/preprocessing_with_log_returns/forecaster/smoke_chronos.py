"""
Chronos-2 SMOKE TEST — confirm we are loading/running the exact same model that
produced the published main-benchmark number before spending GPU on the 4096
fine-tune.

It reuses the UNTOUCHED harness in methods/Chronos2/path_shadowing/run_forecaster_ref.py
(load_pipeline, forecast_ensemble, score_ensemble) to forecast the 8192 main
TEST paths with the seed-0 fine-tuned checkpoint, and checks that the price-space
h32 CRPS reproduces the known value 2.760116 (results/Heston/Chronos2/
forecaster_summary.json -> finetune.per_seed[0].h32_CRPS).

Match => the model + weights + pipeline are correct; proceed to the real 4096 run.

Usage:
  CUDA_VISIBLE_DEVICES=0 /home/tbasseras/gpu-venv/bin/python smoke_chronos.py
"""
import os
import sys
import json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
CHRONOS_PS = os.path.join(BENCH_ROOT, "methods", "Chronos2", "path_shadowing")
sys.path.insert(0, CHRONOS_PS)
import run_forecaster_ref as R  # noqa: E402  (untouched harness)

KNOWN_H32 = 2.760115673991436   # results/Heston/Chronos2 finetune seed 0
TOL = 5e-3


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = np.load(R.TEST_DATA).astype(np.float64)           # (8192,128) main test set
    prefixes = X[:, :R.PREFIX_LEN]
    y_fut = X[:, R.PREFIX_LEN:]
    print(f"[smoke] test {X.shape} prefix={R.PREFIX_LEN} horizon={R.HORIZON} K={R.K} dev={device}", flush=True)

    sd_path = os.path.join(R.WEIGHTS_DIR, "seed_0_model.pt")
    pipe = R.load_pipeline(device, state_dict_path=sd_path)
    rng = np.random.default_rng(0)                        # same RNG seed as harness seed 0
    ens = R.forecast_ensemble(pipe, prefixes, R.HORIZON, R.K, rng, batch_size=R.FC_BATCH)
    res, _ = R.score_ensemble(ens, y_fut)

    got = float(res["h32_CRPS"])
    match = abs(got - KNOWN_H32) < TOL
    out = {"h32_CRPS": got, "h64_CRPS": float(res["h64_CRPS"]),
           "known_h32_CRPS": KNOWN_H32, "abs_diff": abs(got - KNOWN_H32),
           "tol": TOL, "match": bool(match)}
    with open(os.path.join(HERE, "smoke_chronos.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[smoke] h32_CRPS got={got:.6f} known={KNOWN_H32:.6f} diff={out['abs_diff']:.2e} "
          f"-> {'MATCH' if match else 'MISMATCH'}", flush=True)
    if not match:
        sys.exit(1)


if __name__ == "__main__":
    main()
