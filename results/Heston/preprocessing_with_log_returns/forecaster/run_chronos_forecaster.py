"""
Chronos-2 forecaster @ this experiment's scale — fine-tune on the 4096 TRAIN
paths, forecast the 512 path-shadowing queries (H=32, K=256), and score with the
strict PDF metrics (via pdf_bridge) so the row is directly comparable to the LS4 /
CSDI generator PS columns and the Heston-oracle / RW floor.

Fine-tune recipe is byte-identical to the main-benchmark Chronos-2 finetune
(methods/Chronos2/old/code/train_heston.py): pipeline.fit(prediction_length=16,
finetune_mode="full", lr=1e-4, num_steps=1000, batch_size=256). Only the TRAIN set
(4096 vs 8192) and the forecast target (512 ps queries, H=32) change. The smoke
test already confirmed the load/forecast/score pipeline reproduces the published
main-benchmark h32_CRPS to 1e-6, so the model + weights path are correct.

Usage:
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-7 \
      /home/tbasseras/gpu-venv/bin/python run_chronos_forecaster.py
"""
import os
import sys
import time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(HERE)
BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(EXP_DIR)))
DATA_DIR = os.path.join(BENCH_ROOT, "dataset", "Heston", "preprocessing_with_log_returns")
TRAIN_DATA = os.path.join(DATA_DIR, "heston_S_4096x128.npy")

# untouched Chronos-2 harness: BaseChronosPipeline + forecast_ensemble
REF_SRC = os.path.join(BENCH_ROOT, "methods", "Chronos2", "code", "reference", "src")
CHRONOS_PS = os.path.join(BENCH_ROOT, "methods", "Chronos2", "path_shadowing")
sys.path.insert(0, REF_SRC)
sys.path.insert(0, CHRONOS_PS)
from chronos import BaseChronosPipeline           # noqa: E402
import run_forecaster_ref as R                    # noqa: E402  (forecast_ensemble)

sys.path.insert(0, HERE)
import pdf_bridge as B                            # noqa: E402  (score_forecaster, P)

MODEL_ID = "amazon/chronos-2"
SEED = 0
K = 256                     # ensemble members == generator retrieval K (comparable)
FT_PRED_LEN = 16            # main-benchmark finetune pred_len
FT_STEPS = 1000
FT_LR = 1e-4
FT_BATCH = 256


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED); np.random.seed(SEED)
    S_IDX, H = B.P.S_IDX, B.P.H

    S_train = np.load(TRAIN_DATA).astype(np.float64)              # (4096,128) price
    qS = np.load(B.P.PS_QUERY).astype(np.float64)                # (512,128) price
    prefixes = qS[:, :S_IDX + 1]                                 # (512,65) points 0..64
    print(f"[chronos] train{S_train.shape} query{qS.shape} prefix={prefixes.shape[1]} "
          f"H={H} K={K} dev={device}", flush=True)

    t0 = time.time()
    pipe = BaseChronosPipeline.from_pretrained(MODEL_ID, device_map=device,
                                               torch_dtype=torch.float32)
    print(f"[chronos] loaded {MODEL_ID} in {time.time()-t0:.1f}s "
          f"quantiles={len(pipe.quantiles)}", flush=True)

    t1 = time.time()
    inputs = [torch.tensor(S_train[i], dtype=torch.float32) for i in range(S_train.shape[0])]
    print(f"[chronos] finetune {len(inputs)} series pred_len={FT_PRED_LEN} "
          f"steps={FT_STEPS} lr={FT_LR} batch={FT_BATCH}", flush=True)
    pipe = pipe.fit(inputs, prediction_length=FT_PRED_LEN, finetune_mode="full",
                    learning_rate=FT_LR, num_steps=FT_STEPS, batch_size=FT_BATCH,
                    remove_printer_callback=False,
                    logging_steps=max(1, FT_STEPS // 10),
                    output_dir=os.path.join("/tmp", f"c2fc_{SEED}_{int(time.time())}"))
    ft_time = time.time() - t1
    print(f"[chronos] finetune done in {ft_time:.1f}s", flush=True)

    wdir = os.path.join(HERE, "weights"); os.makedirs(wdir, exist_ok=True)
    torch.save({"model": pipe.model.state_dict(), "seed": SEED, "n_train": S_train.shape[0],
                "ft_pred_len": FT_PRED_LEN, "ft_steps": FT_STEPS},
               os.path.join(wdir, "chronos2_seed0_4096.pt"))

    t2 = time.time()
    rng = np.random.default_rng(SEED)
    ens = R.forecast_ensemble(pipe, prefixes, H, K, rng, batch_size=512)   # (512,K,H) price
    print(f"[chronos] forecast {ens.shape} in {time.time()-t2:.1f}s "
          f"nan={not np.isfinite(ens).all()}", flush=True)

    B.score_forecaster(
        ens, "chronos2", os.path.join(HERE, "chronos2_pdf.json"),
        extra_meta={"model_id": MODEL_ID, "n_train": int(S_train.shape[0]),
                    "ft_pred_len": FT_PRED_LEN, "ft_steps": FT_STEPS,
                    "ft_lr": FT_LR, "ft_batch": FT_BATCH,
                    "ft_time_sec": round(ft_time, 1), "seed": SEED})
    print("[chronos] DONE.", flush=True)


if __name__ == "__main__":
    main()
