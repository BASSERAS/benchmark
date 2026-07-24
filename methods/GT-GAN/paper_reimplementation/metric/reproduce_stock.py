#!/usr/bin/env python
"""Reproduce the GT-GAN paper (Jeon et al., NeurIPS'22, "GT-GAN: General Purpose
Time Series Generation with Generative Adversarial Networks") Stocks result
using the *official* GT-GAN code, verbatim from the authors' eval path.

Paper Table 1 (Stocks), GT-GAN row:
    Discriminative score = 0.010 +/- 0.008
    Predictive   score  = 0.017 +/- 0.000

--------------------------------------------------------------------------------
IMPORTANT — this driver documents the EXACT reference reproduction path; it is
NOT re-run on this box. Reason: GT-GAN's evaluation is TensorFlow 1.x
(metrics/{discriminative,predictive}_metrics.py) glued to a Torch NeuralCDE +
CNF generator stack. TF1 cannot bind the CUDA-13 driver on this machine (same
caveat as TimeGAN). The committed 5-seed Stocks numbers in ../README.md were
obtained with the benchmark's shared PyTorch TSTR port (the "one identical
metric" used across TimeGAN / SBTS / GT-GAN); this script mirrors the reference
protocol so the paper pipeline stays auditable next to it.

Pipeline (identical to GTGAN_stocks.py main() eval-only else-branch):
  1. Load datasets/stock_data.csv via TimeDataset_regular(seq_len=24) -> (24, 7)
     windows (last channel = normalised time index), input_size = 6.
  2. Rebuild the 'gtgan' modules: FinalTanh CDE field -> NeuralCDE embedder,
     Multi_Layer_ODENetwork recovery (r_layer=2, last_activation tanh), CNF
     generator (build_model_tabular_nonlinear + run_latent_ctfp_model5_3).
  3. Load the trained stock_model/stock/{generator,recovery}.pt weights.
  4. Generate: sample z, times; h_hat = run_model(...); x_hat = recovery(h_hat).
  5. Score real vs recovered with the reference TF1 metrics, looped
     max_steps_metric=10 times -> mean/std of |acc-0.5| and TSTR MAE.

Run (requires the tf1 env with the GT-GAN torch stack; NOT gpu-venv):
    cd metric
    CUDA_VISIBLE_DEVICES=0 python reproduce_stock.py
"""
import os
import sys
import json

import numpy as np
import torch

# --- wire up the official reference package exactly like GTGAN_stocks.py -------
REF = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "code", "reference")
)
sys.path.insert(0, REF)
sys.path.insert(0, os.path.join(REF, "metrics"))

# reference symbols (same import lines as GTGAN_stocks.py) ----------------------
import controldiffeq                                                    # noqa: E402
from time_dataset import TimeDataset_regular, batch_generator           # noqa: E402
from ctfp_tools import (                                                # noqa: E402
    run_latent_ctfp_model5_3 as run_model,
    parse_arguments,
)
from train_misc import (                                               # noqa: E402
    set_cnf_options,
    create_regularization_fns,
    build_model_tabular_nonlinear,
)
# model classes live in GTGAN_stocks.py itself
from GTGAN_stocks import (                                             # noqa: E402
    FinalTanh,
    NeuralCDE,
    Multi_Layer_ODENetwork,
    Net,
)
from metrics.discriminative_metrics import discriminative_score_metrics  # noqa: E402
from metrics.predictive_metrics import predictive_score_metrics2         # noqa: E402

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
os.makedirs(OUT, exist_ok=True)

# paper Table 1 (Stocks) and our committed 5-seed aggregate --------------------
PAPER = {"discriminative": "0.010 +/- 0.008", "predictive": "0.017 +/- 0.000"}
OURS_COMMITTED = {"discriminative": "0.026 +/- 0.012", "predictive": "0.018 +/- 0.003"}


def build_gtgan_modules(args, input_size, hidden_size, num_layers, x_hidden, device):
    """Recreate the 'gtgan' embedder / recovery / generator exactly as
    GTGAN_stocks.py main() does for args.model1 == 'gtgan'."""
    ode_func = FinalTanh(input_size, hidden_size, hidden_size, num_layers)
    embedder = NeuralCDE(
        func=ode_func, input_channels=input_size,
        hidden_channels=hidden_size, output_channels=hidden_size,
    ).to(device)
    recovery = Multi_Layer_ODENetwork(
        input_size=hidden_size, hidden_size=hidden_size, output_size=input_size,
        gru_input_size=hidden_size, x_hidden=x_hidden,
        num_layer=args.r_layer, last_activation=args.last_activation_r, delta_t=0.5,
    ).to(device)
    regularization_fns, _ = create_regularization_fns(args)
    generator = build_model_tabular_nonlinear(
        args, args.effective_shape, regularization_fns=regularization_fns,
    ).to(device)
    set_cnf_options(args, generator)
    return embedder, recovery, generator


def main():
    parser = parse_arguments()
    # eval-only defaults matching the reproduction command / GTGAN_stocks.py
    args = parser.parse_args([])
    args.model1 = "gtgan"
    args.data = "stock"
    args.seq_len = 24
    args.batch_size = 128
    args.train = False
    args.r_layer = 2
    args.d_layer = 1
    args.last_activation_r = "tanh"
    args.last_activation_d = "identity"
    args.max_steps_metric = 10

    seed = 7777
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} visible={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

    input_size = 6
    hidden_size = 24
    num_layers = 3
    x_hidden = 48
    args.effective_shape = input_size

    # 1) real Stock windows (official csv) -------------------------------------
    data_path = os.path.join(REF, "datasets", "stock_data.csv")
    dataset = TimeDataset_regular(data_path, args.seq_len)
    dataset_size = len(dataset)
    print(f"loaded {dataset_size} Stocks windows of shape {tuple(dataset[0].shape)}", flush=True)

    embedder, recovery, generator = build_gtgan_modules(
        args, input_size, hidden_size, num_layers, x_hidden, device
    )

    # 2) load the trained GT-GAN Stocks weights --------------------------------
    ckpt = os.path.join(REF, "stock_model", "stock")
    generator.load_state_dict(torch.load(os.path.join(ckpt, "generator.pt")))
    recovery.load_state_dict(torch.load(os.path.join(ckpt, "recovery.pt")))
    generator.eval()
    recovery.eval()

    # 3) generate (eval-only path) ---------------------------------------------
    with torch.no_grad():
        x = batch_generator(dataset, dataset_size).to(device)
        obs = x[:, :, -1]
        x = x[:, :, :-1]
        z = torch.randn(dataset_size, x.size(1), args.effective_shape).to(device)
        time = torch.FloatTensor(list(range(args.seq_len))).to(device)
        final_index = (torch.ones(dataset_size) * (args.seq_len - 1)).to(device)
        train_coeffs = controldiffeq.natural_cubic_spline_coeffs(time, x)
        _ = embedder(time, train_coeffs, final_index)  # kept for parity w/ reference
        times = time.unsqueeze(0).unsqueeze(2).repeat(obs.shape[0], 1, 1)
        h_hat = run_model(args, generator, z, times, device, z=True)
        x_hat = recovery(h_hat, obs)

    real_list = [x[i].cpu().numpy() for i in range(dataset_size)]
    fake_list = [x_hat[i].cpu().numpy() for i in range(dataset_size)]

    # 4) reference TF1 scoring, looped max_steps_metric times ------------------
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()

    disc = [discriminative_score_metrics(real_list, fake_list)
            for _ in range(args.max_steps_metric)]
    pred = [predictive_score_metrics2(real_list, fake_list)
            for _ in range(args.max_steps_metric)]
    disc = np.asarray(disc)
    pred = np.asarray(pred)

    result = {
        "dataset": "Stocks",
        "source": "official GT-GAN code (reference/), GTGAN_stocks.py eval-only path",
        "checkpoint": "stock_model/stock/{generator,recovery}.pt",
        "protocol": {
            "max_steps_metric": args.max_steps_metric,
            "disc": "TF1 GRU 1-layer hidden=int(dim/2), |acc-0.5|",
            "pred": "TF1 GRU 1-layer hidden=int(dim/2), TSTR MAE (predictive_score_metrics2)",
        },
        "discriminative_score": {"mean": float(disc.mean()), "std": float(disc.std()),
                                 "runs": disc.tolist()},
        "predictive_score": {"mean": float(pred.mean()), "std": float(pred.std()),
                             "runs": pred.tolist()},
        "ours_committed_5seed_pytorch_port": OURS_COMMITTED,
        "paper_reference": PAPER,
        "seed": seed,
    }
    with open(os.path.join(OUT, "gtgan_stock_scores.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("\n==== GT-GAN Stocks reproduction (reference path) ====")
    print(f"  disc  ours = {disc.mean():.3f} +/- {disc.std():.3f}   paper = {PAPER['discriminative']}")
    print(f"  pred  ours = {pred.mean():.3f} +/- {pred.std():.3f}   paper = {PAPER['predictive']}")
    print("saved -> results/gtgan_stock_scores.json")


if __name__ == "__main__":
    main()
