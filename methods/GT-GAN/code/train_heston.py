"""
GT-GAN (Jeon, Kim, Song, Cho, Park -- "GT-GAN: General Purpose Time Series
Generation with Generative Adversarial Networks", NeurIPS 2022, arXiv:2210.02040)
-- training + generation on the Heston target.

Trains one GT-GAN model per seed on the 8192x128 Heston price paths in
  ../../../dataset/Heston/heston_S_8192x128.npy
then generates 8192 length-128 paths (prior noise -> CNF generator -> ODE
recovery) and writes them back in price scale.

Reuses the released GT-GAN "gtgan" architecture verbatim (``code/reference/``):
  embedder   = NeuralCDE(FinalTanh)                       [Neural CDE encoder]
  recovery   = Multi_Layer_ODENetwork(r_layer=2, tanh)    [ODE decoder]
  generator  = build_model_tabular_nonlinear (CNF)        [continuous normalizing flow]
  discriminator = Multi_Layer_ODENetwork(d_layer=1, identity)
with the SAME paper Stocks hyperparameters that reproduced Table 1
(predictive exact, discriminative order-of-magnitude): hidden_size=24,
num_layers=3, x_hidden=48, batch_size=128, first_epoch=10000 (embedding
pretrain), max_steps=8500 (joint adversarial), delta_t=0.5 euler recovery/disc,
adaptive (sym12async) CNF generator.  All CTFP/CNF regularization coefficients
are kept at the released ``parse_arguments`` defaults (reconstruction=0.01,
kinetic_energy=0.05, jacobian_norm2=0.01, directional_penalty=0.01,
atol=1e-3, rtol=1e-2).

Only two things change from the paper Stocks setup, both dictated by the data:
  * FEATURE dim 6 -> 1   (Heston price is univariate)
  * seq_len     24 -> 128
The latent/effective_shape stays 24 (= input_size = hidden_size): the CNF
generator operates in the 24-dim embedding space, NOT the raw feature space,
so it is unchanged by the feature-dim switch.  The hardcoded ``range(24)`` /
``final_index=23`` time grid of the reference train loop is replaced by
``range(seq_len)`` / ``final_index=seq_len-1``.

Normalisation chain (global min-max to [0,1], exactly the paper ``normalize``
transform applied globally, invertible):
  S(price) --(X-min)/(max-min+1e-7)--> [0,1]  (model trains here)
  x_hat --x_hat*(max-min+1e-7)+min--> price
Denormalized paths are clipped to >= 1e-6 (GUIDELINE floor).

Usage:
  CUDA_VISIBLE_DEVICES=0 python train_heston.py --seed 0
  CUDA_VISIBLE_DEVICES=0 python train_heston.py --seed 0 --frac 0.05 \
      --first_epoch 300 --max_steps 200 --gen_num 512 --tag smoke
"""
import os
import sys
import csv
import json
import time
import random
import argparse
from itertools import chain

import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F

METHOD_CODE = os.path.dirname(os.path.abspath(__file__))          # methods/GT-GAN/code
METHOD_DIR = os.path.dirname(METHOD_CODE)                          # methods/GT-GAN
BENCH_ROOT = os.path.dirname(os.path.dirname(METHOD_DIR))          # benchmark/
REFERENCE = os.path.join(METHOD_CODE, "reference")                # methods/GT-GAN/code/reference
sys.path.insert(0, REFERENCE)

import controldiffeq                                              # noqa: E402
from ctfp_tools import (                                          # noqa: E402
    parse_arguments,
    log_jaco,
    inversoft_jaco,
    compute_ll,
)
from lib.utils import sample_standard_gaussian                    # noqa: E402
from train_misc import (                                          # noqa: E402
    set_cnf_options,
    create_regularization_fns,
    build_model_tabular_nonlinear,
)
from GTGAN_stocks import (                                        # noqa: E402
    FinalTanh,
    NeuralCDE,
    Multi_Layer_ODENetwork,
)

DEFAULT_DATA = os.path.join(BENCH_ROOT, "dataset", "Heston", "heston_S_8192x128.npy")


# --------------------------------------------------------------------------- #
#  loss functions (verbatim from GTGAN_stocks.train, gamma-free gtgan subset)  #
# --------------------------------------------------------------------------- #
def _loss_e_t0(x_tilde, x):
    return F.mse_loss(x_tilde, x)


def _loss_e_0(loss_e_t0):
    return torch.sqrt(loss_e_t0) * 10


def _loss_g_u(y_fake):
    return F.binary_cross_entropy_with_logits(y_fake, torch.ones_like(y_fake))


def _loss_g_v(x_hat, x):
    loss_g_v1 = torch.mean(
        torch.abs(torch.sqrt(torch.var(x_hat, 0) + 1e-6)
                  - torch.sqrt(torch.var(x, 0) + 1e-6)))
    loss_g_v2 = torch.mean(torch.abs(torch.mean(x_hat, 0) - torch.mean(x, 0)))
    return loss_g_v1 + loss_g_v2


def _loss_g3(loss_g_u, loss_g_v):
    return loss_g_u + 100 * loss_g_v


def _loss_d2(y_real, y_fake):
    loss_d_real = F.binary_cross_entropy_with_logits(y_real, torch.ones_like(y_real))
    loss_d_fake = F.binary_cross_entropy_with_logits(y_fake, torch.zeros_like(y_fake))
    return loss_d_real + loss_d_fake


def run_ctfp(args, aug_model, values, times, device, z=True):
    """Faithful copy of reference ``ctfp_tools.run_latent_ctfp_model5_3`` with the
    seq_len / effective_shape conflation fixed so it is correct when
    ``seq_len != effective_shape`` (Heston: 128 vs 24).

    The reference reads ``values.shape[1]`` as the latent feature dimension in two
    places; in the paper's Stocks/Energy setups ``seq_len == hidden_size ==
    effective_shape == 24`` so this silently worked.  Here we substitute
    ``args.effective_shape`` in exactly those two spots:
      * z=True  : latent noise dim  (was ``values.shape[1]``)
      * z=False : ``transform_values`` reshape  (was ``transform_values.shape[1]``)
    Both substitutions are no-ops when seq_len == effective_shape == 24, so this is
    byte-identical to the reference on the reproduced paper case.
    """
    eff = args.effective_shape
    if z:
        mu = torch.zeros(1, values.shape[0], eff).to(device)
        stdvs = torch.ones(1, values.shape[0], eff).to(device)
        latent = sample_standard_gaussian(mu, stdvs)
        latent_sequence = latent.view(-1, latent.shape[2]).unsqueeze(1)
        max_length = times.shape[1]
        latent_sequence = latent_sequence.repeat(1, max_length, 1)
        aux = torch.cat([latent_sequence, times], dim=2)
        aux = aux.view(-1, aux.shape[2])
        aux, _, _ = aug_model(aux, torch.zeros(aux.shape[0], 1).to(aux), reverse=True)
        aux = aux[:, :eff]
        aux = aux.view(values.shape[0], -1, eff)
        if args.activation == "exp":
            aux, _ = log_jaco(aux, reverse=True)
        elif args.activation == "softplus":
            aux, _ = inversoft_jaco(aux, reverse=True)
        elif args.activation == "identity":
            pass
        else:
            raise NotImplementedError
        return aux
    else:
        num_iwae = args.num_iwae_samples
        stdvs = torch.ones(1, values.shape[0], values.shape[1]).to(device)
        vars = torch.ones_like(stdvs).squeeze(0)
        masks = torch.ones_like(stdvs).squeeze(0)
        time_to_cat = times.repeat(num_iwae, 1, 1)
        values = values.repeat(num_iwae, 1, 1)
        aux = torch.cat([torch.zeros_like(values), time_to_cat], dim=2)
        aux = aux.view(-1, aux.shape[2])
        aux, _, _ = aug_model(aux, torch.zeros(aux.shape[0], 1).to(aux), reverse=True)
        aux = aux[:, eff:]
        if args.activation == "exp":
            transform_values, transform_logdet = log_jaco(values)
        elif args.activation == "softplus":
            transform_values, transform_logdet = inversoft_jaco(values)
        elif args.activation == "identity":
            transform_values = values
            transform_logdet = torch.sum(torch.zeros_like(values), dim=2)
        else:
            raise NotImplementedError
        aug_values = transform_values.view(-1, eff)               # FIX: eff (was shape[1]=seq_len)
        aug_values = torch.cat([aug_values, aux], dim=1)
        if args.kinetic_energy is None:
            base_values, flow_logdet, _ = aug_model(
                aug_values, torch.zeros(aug_values.shape[0], 1).to(aug_values))
        else:
            base_values, flow_logdet, reg_states = aug_model(
                aug_values, torch.zeros(aug_values.shape[0], 1).to(aug_values))
            reg_states = tuple(torch.mean(rs) for rs in reg_states)
        base_values = base_values[:, :eff]
        base_values = base_values.view(values.shape[0], -1, eff)
        flow_logdet = flow_logdet.sum(-1).view(1 * base_values.shape[0], -1)
        transform_logdet = transform_logdet.view(1 * base_values.shape[0], -1)
        if len(vars.shape) == 2:
            vars_unsqueed = vars.unsqueeze(-1)
        else:
            vars_unsqueed = vars
        ll = compute_ll(flow_logdet + transform_logdet, base_values,
                        vars_unsqueed.repeat(1, 1, 1), masks.repeat(1, 1))
        ll = ll.view(num_iwae, int(base_values.shape[0] / num_iwae))
        weights = ll
        loss = -torch.logsumexp(weights, 0) + np.log(num_iwae)
        loss = torch.sum(loss) / (int(base_values.shape[0] / num_iwae) * base_values.shape[1])
        loss_training = -torch.sum(F.softmax(weights, 0).detach() * weights) / (
            int(base_values.shape[0] / num_iwae) * base_values.shape[1])
        if args.kinetic_energy is None:
            return loss, loss_training
        else:
            return loss, loss_training, reg_states[0]


def batch_generator_np(dataset, batch_size):
    """Stack `batch_size` random samples -> (batch, seq, feat+1) float tensor.

    Local copy of time_dataset.batch_generator that draws from a numpy stack;
    dataset is an ndarray (N, seq, feat+1).
    """
    idx = np.random.permutation(len(dataset))[:batch_size]
    return torch.from_numpy(dataset[idx]).float()


def main():
    # --- build the released CTFP/CNF parser, then add GT-GAN + our args ------
    parser = parse_arguments()                     # returns the parser (input_size=24, dims, reg coeffs, solver...)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=8500)     # joint adversarial steps (paper Stocks)
    parser.add_argument("--first_epoch", type=int, default=10000)  # embedding pretrain steps (paper)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--d_layer", type=int, default=1)
    parser.add_argument("--r_layer", type=int, default=2)
    parser.add_argument("--last_activation_r", type=str, default="tanh")
    parser.add_argument("--last_activation_d", type=str, default="identity")
    parser.add_argument("--log_time", type=int, default=2)         # supervised-gen update cadence
    # --- our I/O + smoke controls ---
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", type=str, default=DEFAULT_DATA)
    parser.add_argument("--gen_num", type=int, default=8192)
    parser.add_argument("--gen_batch", type=int, default=1024)
    parser.add_argument("--frac", type=float, default=1.0,
                        help="fraction of training paths to use (smoke test)")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--tag", type=str, default="",
                        help="run tag (e.g. 'smoke'); prefixes output names, skips canonical weights")
    args = parser.parse_args()
    args.effective_shape = args.input_size         # CNF operates in the 24-dim latent (= input_size = hidden)

    # --- determinism ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    tagp = (args.tag + "_") if args.tag else ""

    # --- data: global min-max normalize to [0,1] (paper `normalize`) ---------
    S = np.load(os.path.abspath(args.data)).astype(np.float64)     # (8192, 128) price
    if args.frac < 1.0:
        n = int(round(S.shape[0] * args.frac))
        S = S[:n]
    N, seq_len = S.shape
    assert seq_len == args.seq_len, f"data seq_len {seq_len} != --seq_len {args.seq_len}"
    dmin = float(S.min())
    dmax = float(S.max())
    denom = (dmax - dmin) + 1e-7
    Sn = (S - dmin) / denom                                        # (N, T) in [0,1]

    # dataset[i] = (seq_len, feat+1): [normalized price, observation-time index]
    tcol = np.arange(seq_len, dtype=np.float64)[None, :, None].repeat(N, 0)   # (N, T, 1)
    dataset = np.concatenate([Sn[:, :, None], tcol], axis=2)       # (N, T, 2)

    print(f"=== GT-GAN Heston  seed={args.seed}  CUDA_VISIBLE_DEVICES={cvd}  device={dev_name} ===",
          flush=True)
    print(f"[data] S{S.shape} price[min={dmin:.2f},max={dmax:.2f}]  "
          f"dataset{dataset.shape}  first_epoch={args.first_epoch} max_steps={args.max_steps} "
          f"batch={args.batch_size} eff_shape={args.effective_shape}", flush=True)

    # --- models (gtgan mode, feat_dim=1) -------------------------------------
    input_size = 1                 # Heston is univariate
    hidden_size = 24
    num_layers = 3
    x_hidden = 48

    ode_func = FinalTanh(input_size, hidden_size, hidden_size, num_layers)
    embedder = NeuralCDE(func=ode_func, input_channels=input_size,
                         hidden_channels=hidden_size, output_channels=hidden_size).to(device)
    recovery = Multi_Layer_ODENetwork(
        input_size=hidden_size, hidden_size=hidden_size, output_size=input_size,
        gru_input_size=hidden_size, x_hidden=x_hidden, num_layer=args.r_layer,
        last_activation=args.last_activation_r, delta_t=0.5).to(device)
    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    generator = build_model_tabular_nonlinear(
        args, args.effective_shape, regularization_fns=regularization_fns).to(device)
    set_cnf_options(args, generator)
    discriminator = Multi_Layer_ODENetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=1,
        gru_input_size=hidden_size, x_hidden=x_hidden,
        last_activation=args.last_activation_d, num_layer=args.d_layer, delta_t=0.5).to(device)

    g_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    d_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    total_params = sum(p.numel() for m in (embedder, recovery, generator, discriminator)
                       for p in m.parameters())
    print(f"[model] generator={g_params} discriminator={d_params} total={total_params}", flush=True)

    optimizer_er = optim.Adam(chain(embedder.parameters(), recovery.parameters()))
    optimizer_gs = optim.Adam(generator.parameters())
    optimizer_d = optim.Adam(discriminator.parameters())

    time_grid = torch.FloatTensor(list(range(seq_len))).to(device)
    hist = []   # loss log rows

    t0 = time.time()

    # ------------------- Phase 1: embedding network pretrain ------------------
    embedder.train(); recovery.train()
    print("Start Embedding Network Training", flush=True)
    for step in range(1, args.first_epoch + 1):
        x = batch_generator_np(dataset, args.batch_size).to(device)
        obs = x[:, :, -1]
        x = x[:, :, :-1]
        final_index = (torch.ones(x.size(0)) * (seq_len - 1)).to(device)
        train_coeffs = controldiffeq.natural_cubic_spline_coeffs(time_grid, x)
        h = embedder(time_grid, train_coeffs, final_index)
        x_tilde = recovery(h, obs)
        loss_e_t0 = _loss_e_t0(x_tilde, x)
        loss_e_0 = _loss_e_0(loss_e_t0)
        optimizer_er.zero_grad()
        loss_e_0.backward()
        optimizer_er.step()
        if step % args.log_every == 0 or step == 1:
            e_rmse = float(np.sqrt(loss_e_t0.item()))
            hist.append({"step": step, "phase": "embed", "loss_e": e_rmse,
                         "loss_d": "", "loss_g_u": "", "loss_g_v": "", "loss_s": ""})
            print(f"[embed {step:5d}/{args.first_epoch}] loss_e={e_rmse:.4f}", flush=True)
    print("Finish Embedding Network Training", flush=True)

    # ------------------- Phase 2: joint adversarial training ------------------
    print("Start Joint Training", flush=True)
    loss_d = torch.tensor(0.0)
    loss_s = torch.tensor(0.0)
    for step in range(1, args.max_steps + 1):
        # --- inner: discriminator + recovery (x2) ---
        for _ in range(2):
            generator.train(); recovery.train()
            x = batch_generator_np(dataset, args.batch_size).to(device)
            obs = x[:, :, -1]
            x = x[:, :, :-1]
            z = torch.randn(x.size(0), args.effective_shape).to(device)
            final_index = (torch.ones(x.size(0)) * (seq_len - 1)).to(device)
            train_coeffs = controldiffeq.natural_cubic_spline_coeffs(time_grid, x)
            h = embedder(time_grid, train_coeffs, final_index)
            times = time_grid.unsqueeze(0).unsqueeze(2).repeat(obs.shape[0], 1, 1)
            h_hat = run_ctfp(args, generator, z, times, device, z=True)
            x_real = recovery(h, obs)
            x_fake = recovery(h_hat, obs)
            y_fake = discriminator(x_fake, obs)
            y_real = discriminator(x_real, obs)
            loss_d = _loss_d2(y_real, y_fake)
            if loss_d.item() > 0.15:
                optimizer_d.zero_grad()
                loss_d.backward()
                optimizer_d.step()
            # recovery update
            h = embedder(time_grid, train_coeffs, final_index)
            x_tilde = recovery(h, obs)
            loss_e_t0 = _loss_e_t0(x_tilde, x)
            loss_e = _loss_e_0(loss_e_t0)
            optimizer_er.zero_grad()
            loss_e.backward()
            optimizer_er.step()

        # --- supervised generator update every log_time steps ---
        if step % args.log_time == 0:
            x = batch_generator_np(dataset, args.batch_size).to(device)
            obs = x[:, :, -1]
            x = x[:, :, :-1]
            final_index = (torch.ones(x.size(0)) * (seq_len - 1)).to(device)
            train_coeffs = controldiffeq.natural_cubic_spline_coeffs(time_grid, x)
            h = embedder(time_grid, train_coeffs, final_index)
            times = time_grid.unsqueeze(0).unsqueeze(2).repeat(obs.shape[0], 1, 1)
            if args.kinetic_energy is None:
                loss_s, _loss = run_ctfp(args, generator, h, times, device, z=False)
                optimizer_gs.zero_grad()
                loss_s.backward()
            else:
                loss_s, _loss, reg_state = run_ctfp(args, generator, h, times, device, z=False)
                optimizer_gs.zero_grad()
                (loss_s + reg_state).backward()
            optimizer_gs.step()

        # --- adversarial generator update ---
        x = batch_generator_np(dataset, args.batch_size).to(device)
        obs = x[:, :, -1]
        x = x[:, :, :-1]
        z = torch.randn(x.size(0), args.effective_shape).to(device)
        final_index = (torch.ones(x.size(0)) * (seq_len - 1)).to(device)
        train_coeffs = controldiffeq.natural_cubic_spline_coeffs(time_grid, x)
        h = embedder(time_grid, train_coeffs, final_index)
        times = time_grid.unsqueeze(0).unsqueeze(2).repeat(obs.shape[0], 1, 1)
        h_hat = run_ctfp(args, generator, z, times, device, z=True)
        x_hat = recovery(h_hat, obs)
        y_fake = discriminator(x_hat, obs)
        loss_g_u = _loss_g_u(y_fake)
        loss_g_v = _loss_g_v(x_hat, x)
        loss_g = _loss_g3(loss_g_u, loss_g_v)
        optimizer_gs.zero_grad()
        loss_g.backward()
        optimizer_gs.step()

        if step % args.log_every == 0 or step == 1:
            row = {"step": step, "phase": "joint",
                   "loss_e": float(np.sqrt(loss_e_t0.item())),
                   "loss_d": float(loss_d.item()),
                   "loss_g_u": float(loss_g_u.item()),
                   "loss_g_v": float(loss_g_v.item()),
                   "loss_s": float(loss_s.item())}
            hist.append(row)
            print(f"[joint {step:5d}/{args.max_steps}] loss_d={row['loss_d']:.4f} "
                  f"loss_g_u={row['loss_g_u']:.4f} loss_g_v={row['loss_g_v']:.4f} "
                  f"loss_s={row['loss_s']:.4f} loss_e={row['loss_e']:.4f}", flush=True)
    print("Finish Joint Training", flush=True)
    train_time = time.time() - t0

    # ------------------- generation (prior noise -> CNF -> recovery) ----------
    gen_n = int(round(args.gen_num * args.frac)) if args.frac < 1.0 else args.gen_num
    generator.eval(); recovery.eval()
    g0 = time.time()
    chunks = []
    with torch.no_grad():
        done = 0
        while done < gen_n:
            b = min(args.gen_batch, gen_n - done)
            z = torch.randn(b, args.effective_shape).to(device)
            obs = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(0).repeat(b, 1)
            times = time_grid.unsqueeze(0).unsqueeze(2).repeat(b, 1, 1)
            h_hat = run_ctfp(args, generator, z, times, device, z=True)
            x_hat = recovery(h_hat, obs)                            # (b, T, 1) in [0,1]
            chunks.append(x_hat[:, :, 0].detach().cpu().numpy())
            done += b
    gen_n_norm = np.concatenate(chunks, axis=0).astype(np.float64)  # (gen_n, T)
    Xg = gen_n_norm * denom + dmin                                 # invert to price scale
    Xg = np.clip(Xg, 1e-6, None)
    gen_time = time.time() - g0
    gen_has_nan = bool(not np.isfinite(Xg).all())

    # ------------------- output dirs ------------------------------------------
    weights_dir = os.path.join(METHOD_DIR, "weights")
    losses_dir = os.path.join(METHOD_DIR, "losses")
    gen_dir = os.path.join(METHOD_DIR, "generated_paths", f"seed_{args.seed}")
    for d in (weights_dir, losses_dir, gen_dir):
        os.makedirs(d, exist_ok=True)

    # --- generated paths ---
    out_npy = os.path.join(gen_dir, f"{tagp}generated_paths_8192x128.npy")
    np.save(out_npy, Xg.astype(np.float64))

    # --- loss curve ---
    loss_name = f"{tagp}seed_{args.seed}_losses.csv" if args.tag else f"seed_{args.seed}_losses.csv"
    with open(os.path.join(losses_dir, loss_name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "phase", "loss_e", "loss_d",
                                          "loss_g_u", "loss_g_v", "loss_s"])
        w.writeheader()
        w.writerows(hist)

    # --- weights + config (only for canonical runs) ---
    if not args.tag:
        torch.save({"embedder": embedder.state_dict(),
                    "recovery": recovery.state_dict(),
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "data_min": dmin, "data_max": dmax, "seed": args.seed},
                   os.path.join(weights_dir, f"seed_{args.seed}_model.pt"))
        cfg = {"method": "GT-GAN", "variant": "GT-GAN (released gtgan mode, paper Stocks hyperparams)",
               "seed": args.seed, "feat_dim": input_size, "seq_len": seq_len,
               "hidden_size": hidden_size, "num_layers": num_layers, "x_hidden": x_hidden,
               "effective_shape": args.effective_shape, "batch_size": args.batch_size,
               "first_epoch": args.first_epoch, "max_steps": args.max_steps,
               "r_layer": args.r_layer, "d_layer": args.d_layer,
               "last_activation_r": args.last_activation_r, "last_activation_d": args.last_activation_d,
               "solver": args.solver, "atol": args.atol, "rtol": args.rtol,
               "dims": args.dims, "reconstruction": args.reconstruction,
               "kinetic_energy": args.kinetic_energy, "jacobian_norm2": args.jacobian_norm2,
               "directional_penalty": args.directional_penalty,
               "scaler": "global_minmax_0_1", "data_min": dmin, "data_max": dmax,
               "params": int(total_params)}
        with open(os.path.join(weights_dir, f"seed_{args.seed}_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    # --- metadata (GUIDELINE §4.3 schema) ---
    meta = {"method": "GT-GAN", "seed": args.seed, "shape": list(Xg.shape),
            "min_val": float(Xg.min()), "max_val": float(Xg.max()),
            "generated_mean": float(Xg.mean()), "generated_std": float(Xg.std()),
            "real_mean": float(S.mean()), "real_std": float(S.std()),
            "gen_sec": round(gen_time, 1), "train_time_sec": round(train_time, 1),
            "gpu": "A100-SXM4-80GB", "date": time.strftime("%Y-%m-%d"),
            "params": int(total_params), "first_epoch": args.first_epoch,
            "max_steps": args.max_steps, "gen_has_nan": gen_has_nan}
    meta_name = f"{tagp}metadata.json" if args.tag else "metadata.json"
    with open(os.path.join(gen_dir, meta_name), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] seed={args.seed} gen={Xg.shape} price=[{Xg.min():.2f},{Xg.max():.2f}] "
          f"nan={gen_has_nan} train={train_time:.1f}s gen={gen_time:.1f}s", flush=True)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
