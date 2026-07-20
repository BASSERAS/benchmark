"""Eval-only CSDI Heston imputation — loads an ALREADY-TRAINED model.pth and
scores held-out points, skipping the ~20-min retrain.

The full driver (exe_heston_impute.py) always trains fresh. When training has
already completed (model.pth on disk) but the evaluation was interrupted before
writing result_nsample{N}.pk, this script resumes just the evaluation:

  * rebuilds the SAME config overrides (is_unconditional=0, target_strategy=
    'random', test_missing_ratio),
  * rebuilds the dataloaders with the SAME seed (deterministic 70/10/20 split
    and deterministic test gt_mask),
  * loads the saved state_dict into CSDI_Physio,
  * runs the authors' evaluate(..., scaler=1, mean_scaler=0) -> writes
    result_nsample{N}.pk into --modelfolder,
  * mirrors [rmse, mae, CRPS] to ../results/heston_impute_crps_r{ratio}.json.

CRPS convention identical to exe_heston_impute.py / exe_physio.py.

Usage:
    CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 taskset -c 8-15 \
      python eval_only_heston_impute.py \
        --testmissingratio 0.1 --nsample 100 --seed 1 \
        --modelfolder heston_r0.1_seed1_20260720_074000
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.abspath(os.path.join(HERE, "..", "..", "code", "reference"))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
sys.path.insert(0, REF)
sys.path.insert(0, HERE)

from main_model import CSDI_Physio      # noqa: E402  (authors', unchanged)
from utils import evaluate              # noqa: E402  (authors', unchanged)
from dataset_heston import get_dataloader  # noqa: E402  (this folder)

parser = argparse.ArgumentParser(description="CSDI-Heston-imputation-EVAL-ONLY")
parser.add_argument("--config", type=str, default=os.path.join(REF, "config", "base.yaml"))
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument("--nfold", type=int, default=0)
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--modelfolder", type=str, required=True,
                    help="folder under save/ containing the trained model.pth")
args = parser.parse_args()
print(args)

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = 0
config["model"]["target_strategy"] = "random"
config["model"]["test_missing_ratio"] = args.testmissingratio

np.random.seed(args.seed)
torch.manual_seed(args.seed)

foldername = os.path.join(HERE, "save", args.modelfolder)
model_path = os.path.join(foldername, "model.pth")
assert os.path.isfile(model_path), f"no trained model at {model_path}"

train_loader, valid_loader, test_loader = get_dataloader(
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
)

model = CSDI_Physio(config, args.device, target_dim=1).to(args.device)
model.load_state_dict(torch.load(model_path, map_location=args.device))
print(f"loaded trained model from {model_path}")

evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)

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
