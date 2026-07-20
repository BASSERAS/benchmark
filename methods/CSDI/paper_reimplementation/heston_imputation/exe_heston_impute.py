"""CSDI Heston imputation-CRPS driver — mirrors exe_physio.py verbatim in structure.

Applies CSDI's OWN paper metric (quantile CRPS, utils.calc_quantile_CRPS) to the
benchmark's 8192×128 Heston paths, by training the authors' *conditional*
CSDI_Physio (target_dim=1, is_unconditional=0, target_strategy='random') on a
random-missing imputation task and scoring held-out points. This is the
results-README §6 "Ours — Heston" column: the exact paper metric, transferred.

Reuses the authors' code UNCHANGED from ../../code/reference/:
  CSDI_Physio (main_model.py), train + evaluate (utils.py), base.yaml.
Only the dataloader (dataset_heston.py, this folder) is new — it swaps the
PhysioNet .txt reader for the Heston .npy reader; batch format is identical.

CRPS convention matches PhysioNet exactly: evaluate(..., scaler=1, mean_scaler=0),
so CRPS is measured on the z-scored scale and normalised by sum|target| — the
same number type as the paper's Table 2.

Usage (one missing ratio per run; parallelise ratios across GPUs):
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-7 \
      python exe_heston_impute.py --testmissingratio 0.1 --nsample 100 --seed 1
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.abspath(os.path.join(HERE, "..", "..", "code", "reference"))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
sys.path.insert(0, REF)          # reuse authors' code verbatim
sys.path.insert(0, HERE)         # our Heston dataloader

from main_model import CSDI_Physio      # noqa: E402  (authors', unchanged)
from utils import train, evaluate       # noqa: E402  (authors', unchanged)
from dataset_heston import get_dataloader  # noqa: E402  (this folder)

parser = argparse.ArgumentParser(description="CSDI-Heston-imputation")
parser.add_argument("--config", type=str, default=os.path.join(REF, "config", "base.yaml"))
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument("--nfold", type=int, default=0)
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--epochs", type=int, default=0, help="0 = base.yaml default (200)")
args = parser.parse_args()
print(args)

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# conditional imputation, random self-supervised mask — the PhysioNet headline setting
config["model"]["is_unconditional"] = 0
config["model"]["target_strategy"] = "random"
config["model"]["test_missing_ratio"] = args.testmissingratio
if args.epochs > 0:
    config["train"]["epochs"] = args.epochs
print(json.dumps(config, indent=4))

np.random.seed(args.seed)
torch.manual_seed(args.seed)

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = os.path.join(HERE, "save", f"heston_r{args.testmissingratio}_seed{args.seed}_{current_time}")
os.makedirs(foldername, exist_ok=True)
with open(os.path.join(foldername, "config.json"), "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader = get_dataloader(
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
)

model = CSDI_Physio(config, args.device, target_dim=1).to(args.device)

train(model, config["train"], train_loader, valid_loader=valid_loader, foldername=foldername)

# evaluate: scaler=1, mean_scaler=0  → CRPS on z-scored scale, exactly like exe_physio.py
evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)

# read back the authors' result pickle [rmse, mae, CRPS] and mirror to a JSON cell
import pickle  # noqa: E402  # reads our own just-written result file (self-generated, trusted)
with open(os.path.join(foldername, f"result_nsample{args.nsample}.pk"), "rb") as f:
    rmse, mae, crps = pickle.load(f)

os.makedirs(RESULTS, exist_ok=True)
out = os.path.join(RESULTS, f"heston_impute_crps_r{args.testmissingratio}.json")
with open(out, "w") as f:
    json.dump(
        {
            "task": "heston_imputation",
            "missing_ratio": args.testmissingratio,
            "seed": args.seed,
            "nsample": args.nsample,
            "epochs": config["train"]["epochs"],
            "rmse": float(rmse),
            "mae": float(mae),
            "CRPS": float(crps),
            "model_folder": os.path.relpath(foldername, RESULTS),
        },
        f,
        indent=2,
    )
print(f"\nWrote {out}  (CRPS={crps:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f})")
