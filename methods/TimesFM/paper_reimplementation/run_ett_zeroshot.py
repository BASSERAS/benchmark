#!/usr/bin/env python
"""Zero-shot ETT reproduction for TimesFM (paper Table 5, arXiv:2310.10688v4).

Reproduces the "MAE for ETT datasets for prediction horizons 96 and 192"
zero-shot table. Protocol (paper Table 5 caption + Informer/GFQW23):
  * MAE reported on the LAST test window of the original test split.
  * Per-channel z-normalisation fitted on the TRAIN range only (StandardScaler),
    matching timesfm/data_loader.py::TimeSeriesdata._normalize_data.
  * Point forecast = median = quantile_forecast[:, :, 5] (index 5 = q0.5),
    matching experiments/long_horizon_benchmarks/run_eval.py::get_forecasts.

The windowing / normalisation here is a pure-numpy re-implementation of the
official TimeSeriesdata loader (the only TF usage upstream is a thin
tf.data.Dataset.from_generator wrapper around the pure-numpy test_val_gen).

Usage:
  python run_ett_zeroshot.py --model google/timesfm-1.0-200m-pytorch --context 512
  python run_ett_zeroshot.py --model google/timesfm-2.0-500m-pytorch --context 512
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import timesfm

# Official Informer splits (train/val/test upper boundaries), from
# experiments/long_horizon_benchmarks/run_eval.py::DATA_DICT
DATA_DICT = {
    "etth1": {"boundaries": [8640, 11520, 14400], "csv": "ETTh1.csv"},
    "etth2": {"boundaries": [8640, 11520, 14400], "csv": "ETTh2.csv"},
    "ettm1": {"boundaries": [34560, 46080, 57600], "csv": "ETTm1.csv"},
    "ettm2": {"boundaries": [34560, 46080, 57600], "csv": "ETTm2.csv"},
}
HORIZONS = [96, 192]
DATETIME_COL = "date"


def load_model(model_path, context_len, horizon_len):
    if "2.0-500m" in model_path:
        num_layers, pos_emb = 50, False
    else:  # 1.0-200m
        num_layers, pos_emb = 20, True
    return timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="gpu",
            per_core_batch_size=32,
            horizon_len=horizon_len,
            num_layers=num_layers,
            context_len=context_len,
            use_positional_embedding=pos_emb,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=model_path),
    )


def eval_dataset(model, data_dir, dataset, pred_len, hist_len):
    cfg = DATA_DICT[dataset]
    b = cfg["boundaries"]
    df = pd.read_csv(os.path.join(data_dir, cfg["csv"]))
    df.fillna(0, inplace=True)
    ts_cols = [c for c in df.columns if c != DATETIME_COL]
    # (num_ts, T), truncate to test upper boundary (as in TimeSeriesdata.__init__)
    data_mat = df[ts_cols].to_numpy().transpose()[:, : b[2]]
    # normalise with TRAIN stats only
    scaler = StandardScaler().fit(data_mat[:, : b[0]].transpose())
    data_mat = scaler.transform(data_mat.transpose()).transpose()
    # LAST test window of the original test split: actuals = final pred_len points
    end = b[2]
    ctx = data_mat[:, end - pred_len - hist_len : end - pred_len]  # (num_ts, hist_len)
    act = data_mat[:, end - pred_len : end]                        # (num_ts, pred_len)
    _, quant = model.forecast(list(ctx), freq=[0] * ctx.shape[0])
    median = quant[:, :pred_len, 5]                                # index 5 = q0.5
    mae = float(np.mean(np.abs(median - act)))
    return mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/timesfm-1.0-200m-pytorch")
    ap.add_argument("--data-dir", default="/tmp/ETT-small")
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    model = load_model(args.model, args.context, max(HORIZONS))
    load_s = time.time() - t0
    print(f"loaded {args.model} in {load_s:.1f}s (context={args.context})", flush=True)

    results = {}
    for dataset in DATA_DICT:
        for h in HORIZONS:
            mae = eval_dataset(model, args.data_dir, dataset, h, args.context)
            results[f"{dataset}_h{h}"] = round(mae, 4)
            print(f"  {dataset} h{h}: MAE={mae:.4f}", flush=True)
    results["avg"] = round(float(np.mean(list(results.values()))), 4)
    results["_model"] = args.model
    results["_context"] = args.context
    print("AVG MAE:", results["avg"], flush=True)

    out = args.out or f"results/ett_zeroshot_{args.model.split('/')[-1]}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
