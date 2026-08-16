#!/usr/bin/env python3
"""Export the Deep-MKV-TS training runs into the layout GUIDELINE.md 4.3/4.4 expects.

Deep-MKV-TS is trained by the upstream driver
`code/reference/experiments/scripts/run_matched_control_synthetic_validation.py`,
which writes a rich run tree (checkpoint evaluations, optimiser trajectory,
per-step diagnostics). The benchmark harness reads a much smaller, flat layout.
This script is the adapter between the two; it *only* copies and reshapes, it
never recomputes a metric and never touches the model.

  runs/seed_{i}/volatility_only_online_mp/
    checkpoint_evaluations/step_2500/validation_bank.npy  -> generated_paths/seed_{i}/generated_paths_8192x128.npy
    training_checkpoints/step_2500.pt                     -> weights/seed_{i}_model.pt
    run_manifest.json                                     -> weights/seed_{i}_config.json
    source_history.jsonl                                  -> losses/seed_{i}_losses.csv

Why step 2500 and not the final step 3000: the paper reports K = 2500. The run
trains 3000 steps and checkpoints every 500 so the trajectory is inspectable,
but the reported model is step 2500. There is no LR scheduler anywhere in the
codebase, so training 3000 and reading step 2500 is bitwise identical to having
stopped at 2500 -- the extra 500 steps cannot reach backwards.

Usage:
    /home/tbasseras/gpu-venv/bin/python methods/Deep-MKV-TS/code/export_benchmark_artifacts.py
    # add --seeds 0 1 3 4 to export a subset, --step 3000 to export another checkpoint
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
from pathlib import Path

import numpy as np
import torch

METHOD = "Deep-MKV-TS"
METHOD_ROOT = Path(__file__).resolve().parent.parent
RUNS = METHOD_ROOT / "paper_reimplementation" / "runs"
ARM = "volatility_only_online_mp"
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_STEP = 2500

# Columns written to losses/seed_{i}_losses.csv. `step, phase, loss_total` are
# the three GUIDELINE 4.4 requires; the rest are the Deep-MKV-TS-specific terms
# that actually explain the total, so the CSV is readable without the run tree.
#   loss_total   -- the quantity the optimiser descends (adjoint noise consistency)
#   discrepancy  -- the MMD discrepancy between generated and target path law
#   objective    -- discrepancy + eta * running cost (the full control objective)
#   running_cost -- specific-entropy cost of the volatility correction
LOSS_FIELDS = [
    ("loss_total", "train_total_loss"),
    ("discrepancy_objective", "discrepancy_objective_value"),
    ("complete_objective", "complete_objective_value"),
    ("running_cost", "running_cost_value"),
    ("grad_norm", "grad_norm"),
    ("mmd_observed_path", "path_law_observed_path_mmd_path_law_observed_path_value"),
    ("mmd_increments", "path_law_increments_mmd_path_law_increments_value"),
    ("mmd_terminal", "path_law_terminal_mmd_path_law_terminal_value"),
    ("mmd_global_rv", "volatility_law_global_rv_mmd_volatility_law_global_rv_value"),
    (
        "mmd_abs_return_acf",
        "volatility_law_abs_return_acf_mmd_volatility_law_abs_return_acf_value",
    ),
    (
        "mmd_squared_return_acf",
        "volatility_law_squared_return_acf_mmd_volatility_law_squared_return_acf_value",
    ),
]


def arm_dir(seed: int) -> Path:
    return RUNS / f"seed_{seed}" / ARM


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def export_paths(seed: int, step: int) -> dict:
    """Copy one checkpoint's sample bank out as the seed's generated draw.

    The bank is stored float32 to keep the run tree small; the benchmark
    contract is float64 in the original price scale, so it is widened here.
    GUIDELINE 4.3 asks for a clip at 1e-6; this model is a geometric SDE and
    cannot emit a non-positive price, so a non-positive value would be a real
    bug. It is therefore asserted rather than silently clipped away.
    """
    src = arm_dir(seed) / "checkpoint_evaluations" / f"step_{step:04d}" / "validation_bank.npy"
    bank = np.load(src)
    if bank.shape != (8192, 128):
        raise ValueError(f"seed {seed}: expected (8192, 128), got {bank.shape} from {src}")
    paths = bank.astype(np.float64)
    n_nonpositive = int((paths <= 0).sum())
    if n_nonpositive:
        raise ValueError(
            f"seed {seed}: {n_nonpositive} non-positive prices in {src}; a geometric "
            "SDE cannot produce these, so this is a real bug, not something to clip away"
        )

    out_dir = METHOD_ROOT / "generated_paths" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "generated_paths_8192x128.npy", paths)
    return {
        "shape": list(paths.shape),
        "min_val": float(paths.min()),
        "max_val": float(paths.max()),
        "source": str(src.relative_to(METHOD_ROOT)),
    }


def _network_config(checkpoint_path: Path) -> dict:
    """Architecture facts read from the checkpoint itself.

    The checkpoint is authoritative: it is the artefact that produced the
    reported numbers, whereas the manifest records what was *requested*.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    arch = ckpt["architecture"]
    return {
        "hidden_dim": arch.get("hidden_dim"),
        "num_layers": arch.get("num_layers"),
        "state_dim": arch.get("state_dim"),
        "noise_dim": arch.get("noise_dim"),
        "adjoint_input_mode": arch.get("adjoint_input_mode"),
        "num_parameters": int(
            sum(v.numel() for v in ckpt["network_state_dict"].values())
        ),
    }


def export_weights(seed: int, step: int, manifest: dict) -> None:
    src = arm_dir(seed) / "training_checkpoints" / f"step_{step:04d}.pt"
    out_dir = METHOD_ROOT / "weights"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out_dir / f"seed_{seed}_model.pt")

    ref = manifest.get("reference_summary", {})
    weights = manifest.get("objective_weights", {})
    config = {
        "method": METHOD,
        "seed": seed,
        "dataset": "Heston",
        "n_train": 8192,
        "seq_len": 128,
        "reported_step": step,
        "trained_steps": 3000,
        "paper_hyperparams": True,
        # Control / objective, paper Section 3 and Table 6.
        "lambda_scale_w_path": weights.get("lambda_scale"),
        "kappa_scale_w_vol": weights.get("kappa_scale"),
        "abs_return_acf_weight": weights.get("abs_return_acf_weight"),
        "squared_return_acf_weight": weights.get("squared_return_acf_weight"),
        "joint_volatility_weight": weights.get("joint_volatility_weight"),
        "eta_running_cost": manifest.get("running_cost", {}).get("eta"),
        "running_cost": manifest.get("running_cost", {}).get("name"),
        # Optimiser.
        "lr": 2e-3,
        "grad_clip_norm": 5.0,
        "bank_size": 8192,
        "sample_batch_size": 2048,
        "solver": manifest.get("solver"),
        # Frozen interpretable reference, paper Section 2.1.
        "reference_kind": manifest.get("reference_kind"),
        "reference_sigma_min": ref.get("reference_sigma_min"),
        "reference_sigma_max": ref.get("reference_sigma_max"),
        "guyon_trend_half_life_fast_steps": ref.get("guyon_trend_half_life_fast_steps"),
        "guyon_trend_half_life_slow_steps": ref.get("guyon_trend_half_life_slow_steps"),
        "guyon_activity_half_life_fast_steps": ref.get("guyon_activity_half_life_fast_steps"),
        "guyon_activity_half_life_slow_steps": ref.get("guyon_activity_half_life_slow_steps"),
        "guyon_calibration_nll": ref.get("guyon_calibration_nll"),
        "guyon_validation_nll": ref.get("guyon_validation_nll"),
        # Correction network. Read off the checkpoint, not the manifest: the
        # manifest carries transformer-shaped keys that are null for this arm
        # (the correction network is a GRU), and a null is worse than absent.
        "network": manifest.get("adjoint_network"),
        **_network_config(src),
        "physical_drift": manifest.get("physical_drift"),
        "selection_scope": manifest.get("selection_scope"),
    }
    (out_dir / f"seed_{seed}_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def export_losses(seed: int) -> int:
    """Flatten source_history.jsonl into the CSV the benchmark reads.

    The history has 334 diagnostic keys per row; only the handful in LOSS_FIELDS
    describe the objective. A missing key is written as an empty cell rather
    than 0.0, because 0.0 is a legitimate value for several of these terms and
    would be indistinguishable from "not logged".
    """
    src = arm_dir(seed) / "source_history.jsonl"
    rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    out_dir = METHOD_ROOT / "losses"
    out_dir.mkdir(parents=True, exist_ok=True)

    header = ["step", "phase"] + [name for name, _ in LOSS_FIELDS]
    with (out_dir / f"seed_{seed}_losses.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in rows:
            cells = [int(row["step"]), "source"]
            cells += ["" if row.get(key) is None else row[key] for _, key in LOSS_FIELDS]
            writer.writerow(cells)
    return len(rows)


def plot_convergence(seeds: list[int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("loss_total", "Total training loss"),
        ("discrepancy_objective", "MMD discrepancy"),
        ("complete_objective", "Complete objective (discrepancy + eta x cost)"),
        ("grad_norm", "Gradient norm"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(1600 / 150, 900 / 150), dpi=150)
    for ax, (col, title) in zip(axes.ravel(), panels):
        for seed in seeds:
            path = METHOD_ROOT / "losses" / f"seed_{seed}_losses.csv"
            if not path.is_file():
                continue
            with path.open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            xs = [int(r["step"]) for r in rows if r[col] != ""]
            ys = [float(r[col]) for r in rows if r[col] != ""]
            ax.plot(xs, ys, label=f"Seed {seed}", linewidth=1.2)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        # These span several orders of magnitude across the run; on a linear
        # axis every seed renders as a flat line at 0 after a few hundred steps.
        if col in {"loss_total", "discrepancy_objective"}:
            ax.set_yscale("log")
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle(f"{METHOD} training loss convergence (Heston, {len(seeds)} seeds)", fontsize=11)
    fig.tight_layout()
    fig.savefig(METHOD_ROOT / "losses" / "loss_convergence.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    parser.add_argument(
        "--gpu", default="A100-SXM4-80GB", help="recorded in metadata.json, not queried"
    )
    args = parser.parse_args()

    today = dt.date.today().isoformat()
    exported: list[int] = []
    for seed in args.seeds:
        arm = arm_dir(seed)
        if not (RUNS / f"seed_{seed}" / "COMPLETE.json").is_file():
            print(f"SKIP seed {seed}: no COMPLETE.json, run is unfinished or absent")
            continue
        manifest = _load_json(arm / "run_manifest.json")
        traj = _load_json(arm / "checkpoint_trajectory_manifest.json")

        info = export_paths(seed, args.step)
        export_weights(seed, args.step, manifest)
        n_rows = export_losses(seed)

        wall = float(manifest.get("resources", {}).get("wall_seconds", float("nan")))
        bank_seed = next(
            (e["bank_seed"] for e in traj["evaluations"] if e["step"] == args.step), None
        )
        meta = {
            "method": METHOD,
            "seed": seed,
            "shape": info["shape"],
            "min_val": info["min_val"],
            "max_val": info["max_val"],
            # The bank is drawn inside the training run's checkpoint evaluation,
            # so sampling is not separately timed. Reporting the total wall time
            # under train_time_sec and leaving gen_time_sec null is honest;
            # inventing a number here would be worse than a null.
            "gen_time_sec": None,
            "train_time_sec": wall,
            "gpu": args.gpu,
            "date": today,
            "reported_checkpoint_step": args.step,
            "bank_seed": bank_seed,
            "source": info["source"],
        }
        (METHOD_ROOT / "generated_paths" / f"seed_{seed}" / "metadata.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        exported.append(seed)
        print(
            f"seed {seed}: paths [{info['min_val']:.2f}, {info['max_val']:.2f}], "
            f"{n_rows} loss rows, {wall / 60:.1f} min train, bank_seed={bank_seed}"
        )

    if not exported:
        raise SystemExit("nothing exported; no seed had a COMPLETE.json")
    plot_convergence(exported)
    print(f"exported seeds {exported} -> generated_paths/, weights/, losses/")


if __name__ == "__main__":
    main()
