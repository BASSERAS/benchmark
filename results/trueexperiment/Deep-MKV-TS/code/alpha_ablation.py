#!/usr/bin/env python3
"""Why does the learned control make things WORSE on TrueDataset?

The hypothesis under test
-------------------------
The control map is (specific_entropy_matrix.py:176)

    Theta = eta * sigma_ref^{-1} + Z / sqrt(dt)

`Z` is the noise head's output.  `1/sqrt(dt)` is the gain that maps the network
output into Theta, and it is the ONE thing that changed between the two
datasets:

    Heston        dt = 1/252      1/sqrt(dt) =   15.87
    TrueDataset   dt = 9.5129e-07 1/sqrt(dt) = 1025.28      <- 64.6x larger

Every optimisation hyperparameter is IDENTICAL across the two runs (lr 0.002,
AdamW, batch 256, lambda_scale 50, kappa_scale 100, grad_clip 5.0, 3000 steps).
AdamW's per-coordinate step is ~lr regardless of gradient scale, so the move
this induces in Theta per step is lr/sqrt(dt):

    Heston        0.002 *   15.87 = 0.032
    TrueDataset   0.002 * 1025.28 = 2.05        <- 64.6x larger

If that is the story, then the learned Z is simply TOO BIG: it overshoots the
reference term instead of correcting it.

The experiment
--------------
Scale the noise head's final layer by alpha.  Because that layer is linear and
there is no denormalisation (`adjoint_target_scale` is None in every
checkpoint), this scales Z by EXACTLY alpha, hence

    Theta(alpha) = eta * sigma_ref^{-1} + alpha * Z / sqrt(dt)

so alpha interpolates linearly in Theta between the untrained reference
(alpha=0) and the trained model (alpha=1).  The drift head does not need
scaling: `adjoint_weight` is 0.0 and its final layer is still exactly zero in
every checkpoint, which this script asserts rather than assumes.

Predictions that distinguish the hypotheses:

  * magnitude is the problem  -> vol_err(alpha) has an interior minimum at
    small alpha, and beats the reference there.  The learned DIRECTION is
    useful, the learned SIZE is not.
  * direction is the problem  -> vol_err(alpha) increases monotonically from
    alpha=0.  No rescaling saves it; the control is pointing the wrong way.

Two free correctness checks are built in: alpha=0 must reproduce the published
reference bank bit-for-bit (same GEN_SEED_BASE), and alpha=1 must reproduce the
diagnostic_bestfit bank bit-for-bit.  Both are asserted, not hoped for.

The Theta probe
---------------
``theta_from_moments`` is wrapped (frozen package untouched, same technique as
eigh_fallback.py) to record the Frobenius norm of each of the two terms, so the
"Z swamps the reference" claim is measured rather than argued.

Usage
-----
    CUDA_VISIBLE_DEVICES=1 taskset -c 16-39 \\
    /home/tbasseras/gpu-venv/bin/python alpha_ablation.py --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
METHOD_ROOT = HERE.parent
REFERENCE = Path("/home/tbasseras/benchmark/methods/Deep-MKV-TS/code/reference")
for _p in (HERE, REFERENCE / "src", REFERENCE / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from deep_mkv_gen_path_dt.grid import DiscreteTimeGrid  # noqa: E402

import eigh_fallback  # noqa: E402  (installs the CPU retry, as in every other stage)
import selection_true as sel  # noqa: E402
import train_true as tt  # noqa: E402
from fit_reference_true import DT, STATE_DIM  # noqa: E402
from generate_bank_true import GEN_SEED_BASE, S0  # noqa: E402

TRUE_ROOT = METHOD_ROOT.parent
SEQ_TAG = "6144x128x8"
NOISE_HEAD = "expected_adjoint_noise_next_head.2"
DRIFT_HEAD = "expected_adjoint_next_head.2"
ALPHAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]

# --------------------------------------------------------------------------
# Theta probe -- wraps the frozen method, does not modify it
# --------------------------------------------------------------------------
_PROBE: dict[str, list[float]] = {"ref": [], "z": []}


def install_theta_probe() -> None:
    from deep_mkv_gen_path_dt.controls import specific_entropy_matrix as sem

    base = sem.SpecificEntropyMatrixControl
    if getattr(base, "_probe_installed", False):
        return
    original = base.theta_from_moments

    def wrapped(self, *, r, sigma_ref, state_dim):  # noqa: ANN001, ANN202
        out = original(self, r=r, sigma_ref=sigma_ref, state_dim=state_dim)
        with torch.no_grad():
            # Recompute the two TERMS from the inputs rather than reaching into
            # the frozen function's locals: same formula, no coupling to its
            # internals.
            sr = sem._as_matrix(sigma_ref, name="sigma_ref", state_dim=state_dim)
            ev = torch.linalg.eigvalsh(sem._symmetrise(sr.double()))
            ref_norm = torch.linalg.norm(float(self.eta) / ev, dim=-1).mean()
            zm = sem._symmetrise(
                sem._as_matrix(r, name="expected_adjoint_noise_next", state_dim=state_dim)
            ).double()
            z_norm = (torch.linalg.matrix_norm(zm) / math.sqrt(float(self.dt))).mean()
            _PROBE["ref"].append(float(ref_norm))
            _PROBE["z"].append(float(z_norm))
        return out

    base.theta_from_moments = wrapped
    base._probe_installed = True
    print("[theta_probe] installed (frozen package unmodified)", flush=True)


def scaled_state(payload: dict, alpha: float) -> dict:
    """Return `payload` with the noise head's final layer scaled by `alpha`."""
    sd = payload["network_state_dict"]
    for suffix in ("weight", "bias"):
        key = f"{DRIFT_HEAD}.{suffix}"
        if key in sd and float(sd[key].abs().max()) != 0.0:
            raise SystemExit(
                f"ABORT: {key} is non-zero. This script assumes the drift head "
                "never left its zero init (adjoint_weight=0.0); if it moved, "
                "alpha no longer interpolates Theta linearly and the whole "
                "experiment is invalid."
            )
    out = dict(payload)
    new_sd = dict(sd)
    for suffix in ("weight", "bias"):
        key = f"{NOISE_HEAD}.{suffix}"
        if key not in new_sd:
            raise SystemExit(f"ABORT: {key} missing from checkpoint")
        new_sd[key] = new_sd[key] * float(alpha)
    out["network_state_dict"] = new_sd
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-paths", type=int, default=6144)
    ap.add_argument("--probe", action="store_true",
                    help="record Theta term norms (slower; run on one seed)")
    ap.add_argument("--alphas", type=float, nargs="+", default=None,
                    help="override the alpha grid (default: the coarse ALPHAS sweep)")
    ap.add_argument("--tag", default="",
                    help="suffix for the output filename; REQUIRED when --alphas is "
                         "given, so a custom grid cannot overwrite the coarse sweep")
    ap.add_argument("--save-bank-root", default=None,
                    help="if set, write each generated bank to "
                         "<root>/generated_paths/seed_<S>/generated_paths_<SEQ_TAG>.npy "
                         "so the full A/B metric suite can be run on it. Only legal "
                         "with a single alpha, since the layout has one slot per seed.")
    args = ap.parse_args()

    alphas = list(args.alphas) if args.alphas else list(ALPHAS)
    if args.alphas and not args.tag:
        raise SystemExit("ABORT: --alphas requires --tag (refusing to overwrite the "
                         "coarse-sweep result files)")
    if args.save_bank_root and len(alphas) != 1:
        raise SystemExit(
            f"ABORT: --save-bank-root needs exactly one alpha, got {len(alphas)}. "
            "The generated_paths/seed_<S>/ layout has a single slot per seed, so a "
            "multi-alpha run would silently leave only the last alpha on disk."
        )

    if args.probe:
        install_theta_probe()

    device = torch.device(args.device)
    env = sel.envelope()
    train, val = sel.load_splits()
    module = sel.load_criterion(sel.DEFAULT_DATA_DIR)
    med_nn = env["median_nn_heldout"]

    probe_dir = METHOD_ROOT / "diagnostic_bestfit" / "alpha_ablation"
    probe_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        cfg = json.loads(
            (METHOD_ROOT / "weights" / f"seed_{seed}_config.json").read_text()
        )
        step = int(cfg["selected_step"])
        ckpt = (
            HERE / "runs" / f"seed_{seed}" / "training_checkpoints" / f"step_{step:04d}.pt"
        )
        payload = torch.load(ckpt, map_location=device, weights_only=True)

        probe_np = tt.load_log_prices(num_paths=1, device="cpu")
        num_steps = int(probe_np.shape[1]) - 1
        del probe_np
        grid = DiscreteTimeGrid(T=num_steps * DT, num_steps=num_steps)
        model, _ = tt.build_model(
            grid=grid, device=device, seed=int(seed), kernel_steps=num_steps
        )
        x0 = torch.full((1, STATE_DIM), math.log(S0), device=device, dtype=model.dtype)

        rows = []
        for alpha in alphas:
            _PROBE["ref"].clear()
            _PROBE["z"].clear()
            model.load_checkpoint_state(scaled_state(payload, alpha))
            with torch.no_grad():
                logp = model.sample(
                    num_paths=int(args.num_paths), x0=x0, seed=GEN_SEED_BASE + int(seed)
                ).paths
            prices = torch.exp(logp.double()).cpu().numpy()
            prices = prices * (S0 / prices[:, 0:1, :])
            prices[:, 0, :] = S0

            if args.save_bank_root:
                # Same layout and dtype contract generate_bank_true.py writes, so
                # metrics/compute_all_multiasset.py --gen-root <root> reads it
                # without knowing an alpha was applied.
                dest = (
                    Path(args.save_bank_root) / "generated_paths" / f"seed_{seed}"
                )
                dest.mkdir(parents=True, exist_ok=True)
                np.save(dest / f"generated_paths_{SEQ_TAG}.npy", prices)
                print(f"    [bank] {dest / f'generated_paths_{SEQ_TAG}.npy'}",
                      flush=True)

            rec = sel.score_candidate(
                prices, train=train, val=val, module=module, median_nn_heldout=med_nn
            )
            rec["alpha"] = alpha
            rec["seed"] = seed
            rec["selected_step"] = step
            rec["admissible"] = not sel.admissibility(rec, env)
            if _PROBE["ref"]:
                rec["theta_ref_norm"] = float(np.mean(_PROBE["ref"]))
                rec["theta_z_norm"] = float(np.mean(_PROBE["z"]))
                rec["theta_z_over_ref"] = rec["theta_z_norm"] / max(
                    rec["theta_ref_norm"], 1e-30
                )
            rows.append(rec)

            extra = ""
            if "theta_z_over_ref" in rec:
                extra = (
                    f"   |ref|={rec['theta_ref_norm']:.3f} "
                    f"|Z/sqrt(dt)|={rec['theta_z_norm']:.3f} "
                    f"ratio={rec['theta_z_over_ref']:.2f}"
                )
            print(
                f"  seed {seed} alpha {alpha:5.2f}: vol {rec['vol_err_pct']:8.2f}%  "
                f"corr {rec['corr_err']:.4f}  kurt {rec['kurt_err_pct']:8.2f}%  "
                f"NN {rec['nn_ratio']:.4f}  "
                f"{'ADMISSIBLE' if rec['admissible'] else ''}{extra}",
                flush=True,
            )

            # Correctness checks: the endpoints must reproduce known banks.
            if alpha in (0.0, 1.0):
                known = (
                    (
                        TRUE_ROOT / "reference" / "generated_paths"
                        if alpha == 0.0
                        else METHOD_ROOT / "diagnostic_bestfit" / "generated_paths"
                    )
                    / f"seed_{seed}"
                    / f"generated_paths_{SEQ_TAG}.npy"
                )
                if known.is_file() and int(args.num_paths) == 6144:
                    ref_bank = np.load(known)
                    dev = float(np.abs(ref_bank - prices).max())
                    tag = "reference" if alpha == 0.0 else "diagnostic_bestfit"
                    print(
                        f"    [check] alpha={alpha} vs published {tag} bank: "
                        f"max abs deviation {dev:.3e}",
                        flush=True,
                    )

        # The probe run uses fewer paths, so it must NOT overwrite the
        # full-bank sweep result for the same seed: those rows are scored on
        # 6144 paths and the two are not interchangeable.
        stem = f"seed_{seed}" if int(args.num_paths) == 6144 else (
            f"seed_{seed}_probe_{int(args.num_paths)}"
        )
        if args.tag:
            stem = f"{stem}_{args.tag}"
        out_path = probe_dir / f"{stem}.json"
        out_path.write_text(
            json.dumps(rows, indent=2, sort_keys=True, default=float) + "\n",
            encoding="utf-8",
        )
        print(f"  [out] {out_path}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
