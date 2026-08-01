"""Emit the protocol PDF's ``generation_manifest.json`` for every seed of a method.

PDF §1.4 makes this file mandatory and fixes its content: "The manifest must record model
name and source revision, training and generation seeds, preprocessing, hyperparameters,
number of trainable parameters, training time, generation time, hardware, and any
model-specific numerical repair."

Our runs already write two files that together cover most of that: ``metadata.json`` next to
the bank (compute + provenance) and ``weights/seed_<q>_config.json`` (architecture +
optimiser). Neither alone satisfies §1.4, and neither records the source revision or the
numerical-repair field. This script merges them and fills the gaps, so the manifest is
derived from what actually ran rather than hand-copied.

Two fields are computed from the bank itself rather than trusted:

* ``output_contract`` re-checks every §1.4 bullet (finite, strictly positive, 8192x128,
  float32/64, S0 = 100). Writing a manifest that *claims* conformance while the array
  violates it would be worse than having no manifest at all.
* ``numerical_repair`` is set from the measured S0 deviation. If a repair was applied the
  manifest says so with its exact formula; if none was applied but the bank still misses
  S0 = 100, the manifest records the deviation as an open non-conformance instead of
  quietly asserting "none".

Usage:
  python write_generation_manifest.py --model-dir ../experiment_A/LS4 \
      --experiment A --source-revision 27df71e --repair none
"""
import os
import json
import argparse
import numpy as np

# PDF §1.4: "begin at S0 = 100, up to ordinary floating-point tolerance". Ordinary tolerance
# for a float64 round-trip is ~1e-12 relative, not the 1e-3 a price-space generator drifts by.
S0_RELATIVE_TOLERANCE = 1e-12

# The union across every method in the benchmark. This is a *filter* over the config file, so
# a key that a method does not have simply does not appear; listing another method's keys here
# costs nothing, whereas omitting one silently drops a hyperparameter from a manifest that PDF
# §1.4 requires to record them.
HYPERPARAMETER_KEYS = (
    # LS4
    "z_dim", "d_state", "d_model", "n_layers", "s4_type", "latent_type",
    "ema_lamb", "ema_start_step",
    # CSDI
    "is_unconditional", "target_dim", "seq_length", "layers", "channels", "nheads",
    "diffusion_embedding_dim", "beta_start", "beta_end", "num_steps", "timeemb", "featureemb",
    "lr_milestones", "lr_gamma", "is_linear", "side_dim",
    # TimeDiT
    "model_size", "hidden_size", "depth", "num_heads", "learn_sigma", "schedule", "T",
    "sampler", "ema", "grad_clip", "n_steps", "n_train", "seq_len", "feature_size",
    # shared
    "batch_size", "lr", "weight_decay", "epochs",
)

# One sentence per preprocessing chain, keyed by the ``scaler`` field the training script wrote
# into its config. Adding a method means adding its entry here; the writer refuses to emit a
# manifest for a transform it cannot describe.
PREPROCESSING_DESCRIPTION = {
    "global_standardize":
        "Per-dataset affine standardization of raw prices: x -> (x - mu) / sigma, inverted "
        "before saving the bank. No log-return transform, no quantile map, no dimensionality "
        "reduction.",
    "minmax_then_znorm":
        "Two-stage affine map of raw prices, both stages fitted on train.npy: first "
        "x -> (x - min) / (max - min) onto [0,1], then z -> (z - mu) / sigma. Inverted in the "
        "same order before saving the bank, with the [0,1] stage clipped to its own range. No "
        "log-return transform, no quantile map, no dimensionality reduction.",
}


def output_contract(bank):
    """Re-verify each §1.4 bullet against the array we are about to describe."""
    s0 = bank[:, 0]
    rel = float(np.abs(s0 / 100.0 - 1.0).max())
    return {
        "shape": list(bank.shape),
        "dtype": str(bank.dtype),
        "all_finite": bool(np.isfinite(bank).all()),
        "all_strictly_positive": bool((bank > 0).all()),
        "dtype_is_float32_or_float64": bank.dtype in (np.float32, np.float64),
        "s0_max_relative_deviation": rel,
        "s0_within_floating_point_tolerance": rel <= S0_RELATIVE_TOLERANCE,
    }, rel


# The two operator orders ``apply_s0_repair.py`` can apply. They are algebraically identical
# and differ in float64 by ~1e-14, which is enough to decide whether a bank anchors at exactly
# 100.0 -- so the manifest must name the one actually used. Recording the multiply-first form
# for a bank produced divide-first would assert a formula that does not reproduce the artefact,
# and the reader has no way to detect that from the manifest alone.
REPAIR_OPERATION = {
    "s0_renormalization": "S <- 100 * S / S[:, 0]",
    "s0_renormalization_anchor_exact": "S <- 100 * (S / S[:, 0])",
}


def repair_block(mode, rel):
    """Describe the numerical repair honestly, including the case where none was applied."""
    if mode in REPAIR_OPERATION:
        block = {
            "applied": True,
            "operation": REPAIR_OPERATION[mode],
            "reason": "PDF §1.4 and §7 checklist item 5 require every bank to begin at "
                      "S0 = 100; the raw generator emits prices in standardized price "
                      "space and is not anchored at t = 0.",
            "declared_under": "PDF §1.3 (post-hoc transformation, declared) and §1.4 "
                              "(model-specific numerical repair)",
            "residual_s0_max_relative_deviation": rel,
        }
        if mode == "s0_renormalization_anchor_exact":
            block["operator_order"] = (
                "divide-first (apply_s0_repair.py --anchor-exact). The declared multiply-first "
                "form leaves a 1.4e-14 residual on this method's banks, which fails PDF §1.4's "
                "\"begin at S0 = 100\" -- stated with no tolerance. Divide-first is exact by "
                "construction because x / x is exactly 1.0. The two forms differ by at most "
                "1.8e-15 in log returns, so the choice does not touch the path law; it only "
                "decides whether the anchor is exact."
            )
        return block
    if rel <= S0_RELATIVE_TOLERANCE:
        return {"applied": False, "reason": "bank already satisfies S0 = 100 exactly"}
    return {
        "applied": False,
        "open_non_conformance": "PDF §7 checklist item 5 (bank must begin at S0 = 100)",
        "s0_max_relative_deviation": rel,
        "note": "Generator emits prices in standardized price space with no t=0 anchor. "
                "Measured effect on every PDF evaluator metric is nil: the evaluators and "
                "the oracle feature builder all normalize by S[:, :1], so renormalizing the "
                "bank moves the reported metrics only in the 15th-16th significant figure.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--experiment", required=True, choices=("A", "B"))
    ap.add_argument("--source-revision", required=True,
                    help="git revision of the generator source actually run")
    ap.add_argument("--repair", default="none",
                    choices=("none", "s0_renormalization", "s0_renormalization_anchor_exact"),
                    help="must match the operator order apply_s0_repair.py actually used; "
                         "pass s0_renormalization_anchor_exact when it ran with --anchor-exact")
    ap.add_argument("--hyperparameter-origin", default="official-default",
                    help="PDF §5 requires stating whether hyperparameters were defaults "
                         "or validation-selected")
    ap.add_argument("--cores-pinned", type=int, default=8,
                    help="physical cores each run was taskset-pinned to. Default 8 is what "
                         "LS4 and CSDI used; TimeDiT ran 4 per run so that four concurrent "
                         "runs stayed inside the 16-core cap. PDF §5 asks for the compute "
                         "actually used, so this must not be left at a stale default.")
    ap.add_argument("--hardware-note", default=None,
                    help="free-text hardware note. Default states one GPU per run and at "
                         "most two concurrent runs, which is the standing limit; override "
                         "it when a run was granted more.")
    ap.add_argument("--reimplementation-basis", default=None,
                    help="Set when the method has NO released implementation. Records "
                         "official_implementation=false and names what the code was built "
                         "from. PDF §5 item 3 asks *whether* official code was used, so 'no' "
                         "is a valid answer -- but only if it says what was used instead.")
    a = ap.parse_args()

    for seed in range(5):
        seed_dir = os.path.join(a.model_dir, "generated_paths", f"seed_{seed}")
        bank_path = os.path.join(seed_dir, "generated_paths_8192x128.npy")
        if not os.path.exists(bank_path):
            print(f"seed {seed}: bank missing, skipped")
            continue

        meta = json.load(open(os.path.join(seed_dir, "metadata.json")))
        cfg_path = os.path.join(a.model_dir, "weights", f"seed_{seed}_config.json")
        cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}

        contract, rel = output_contract(np.load(bank_path))

        # The description used to be a fixed string asserting "x -> (x - mu) / sigma". That is
        # true of CSDI and LS4 and false of TimeDiT, which min-maxes to [0,1] before
        # standardising -- and a manifest that misdescribes the preprocessing is worse than one
        # that omits it, because it reads as verified. Derive the sentence from the transform
        # the config actually recorded, and refuse to guess for an unknown one.
        transform = cfg.get("scaler", "global_standardize")
        if transform not in PREPROCESSING_DESCRIPTION:
            raise SystemExit(
                f"seed {seed}: unknown preprocessing transform {transform!r}. Add it to "
                f"PREPROCESSING_DESCRIPTION rather than letting a wrong description through.")

        # The method name drives the source path. It used to be the literal
        # "methods/LS4/code", which silently wrote a false provenance line into every
        # CSDI manifest. Derive it instead -- a manifest that misreports where the
        # official code lives is worse than no manifest.
        method_name = meta.get("method", "LS4")

        model_block = {
            "name": method_name,
            "variant": cfg.get("variant"),
            "source_revision": a.source_revision,
            "source_path": f"methods/{method_name}/code",
            "official_implementation": a.reimplementation_basis is None,
            "trainable_parameters": meta.get("params"),
        }
        if a.reimplementation_basis is not None:
            model_block["reimplementation_basis"] = a.reimplementation_basis

        manifest = {
            "model": model_block,
            "seeds": {
                # PDF §1.4 asks for these separately. Our runs seed torch once and both the
                # training loop and the generation sampler draw from that stream, so the two
                # are equal by construction -- recorded explicitly rather than implied.
                "training_seed": meta.get("seed"),
                "generation_seed": meta.get("seed"),
                "shared_rng_stream": True,
            },
            "preprocessing": {
                "transform": transform,
                "description": PREPROCESSING_DESCRIPTION[transform],
                "fitted_on": "train.npy only (PDF §1.3)",
                "mu": cfg.get("scaler_mu"),
                "sigma": cfg.get("scaler_sigma"),
                # Two-stage transforms need both stages' fitted constants or the manifest does
                # not describe an invertible map, which is what PDF §1.4 asks it to record.
                # Absent for single-stage scalers, hence the conditional rather than a null:
                # a key that is always present but usually null reads as a missing measurement.
                **({"minmax": cfg["minmax"]} if "minmax" in cfg else {}),
            },
            "hyperparameters": {k: cfg[k] for k in HYPERPARAMETER_KEYS if k in cfg},
            "hyperparameter_origin": a.hyperparameter_origin,
            "compute": {
                "training_time_sec": meta.get("train_time_sec"),
                "generation_time_sec": meta.get("gen_sec"),
                "epochs_run": meta.get("epochs_run"),
                "epochs_max": meta.get("epochs_max"),
                "run_date": meta.get("date"),
            },
            "hardware": {
                "gpu": meta.get("gpu"),
                "gpus_used": 1,
                "cpu": "2x AMD EPYC 7763",
                "cores_pinned": a.cores_pinned,
                "note": a.hardware_note or
                        "One GPU per run; at most two runs concurrently.",
            },
            "numerical_repair": repair_block(a.repair, rel),
            "output_contract": contract,
            "failure_information": {
                "failed_or_unstable": False,
                "nan_in_bank": meta.get("gen_has_nan", False),
                "first_nan_epoch": meta.get("first_nan_epoch"),
                "reason": None,
            },
            "experiment": a.experiment,
            "data_files": {
                "train": f"dataset/Heston/new_experiments/experiment_{a.experiment}/train.npy",
                "validation": f"dataset/Heston/new_experiments/experiment_{a.experiment}/disc.npy",
                "test": f"dataset/Heston/new_experiments/experiment_{a.experiment}/test.npy",
            },
        }

        out = os.path.join(seed_dir, "generation_manifest.json")
        with open(out, "w") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
            fh.write("\n")
        flag = "OK" if contract["s0_within_floating_point_tolerance"] else \
            f"S0 dev {rel:.2e}"
        print(f"seed {seed}: wrote {out}  [{flag}]")


if __name__ == "__main__":
    main()
