"""
TimeVQVAE — Heston training driver.

Reuses the paper-era reference harness (commit b9650e9d) VERBATIM: the same
VQ-VAE (stage1) and MaskGIT prior (stage2) model/experiment classes, the same
STFT LF/HF tokenization, the same AdamW + CosineAnnealingLR schedule, the same
iterative-decoding sampler. NO model, loss, or optimizer code is changed.

What is Heston-specific here (and only here):
  * data source     : dataset/Heston/heston_S_8192x128.npy  (8192, 128) price paths
  * preprocessing   : GLOBAL z-normalisation by the TRAIN mean/std — this is exactly
                      the paper's own `DatasetImporterUCR(data_scaling=True)` scaling
                      (mean/var over the whole train matrix). Stored so generated
                      samples are inverted back to the original PRICE scale.
  * no FCN metrics  : the paper's FID/IS use a UCR-pretrained FCN that does not exist
                      for Heston, so stage2's FCN evaluation block is not run here.
                      Generation itself uses the reference `unconditional_sample`.

Parallel-safety: the reference `get_root_dir()` is the module directory, so
`save_model` (stage1 ckpts) and `MaskGIT.load` (stage2) both read/write
`<workdir>/saved_models`. Each seed therefore runs inside its OWN copy of the
reference tree (``--workdir``); we ``os.chdir`` into it before importing.

Outputs (mirror the TimeVAE layout, written to absolute paths under methods/TimeVQVAE/):
  generated_paths/seed_{i}/generated_paths_8192x128.npy   (float64, PRICE scale)
  generated_paths/seed_{i}/metadata.json
  losses/seed_{i}_stage1_losses.csv , losses/seed_{i}_stage2_losses.csv
  weights/seed_{i}_model.pt  (all stage1 modules + maskgit state_dicts)
  weights/seed_{i}_config.json
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------- #
#  paths (resolved before we chdir into the per-seed workdir)                    #
# ----------------------------------------------------------------------------- #
THIS_FILE = Path(__file__).resolve()
CODE_DIR = THIS_FILE.parent                       # methods/TimeVQVAE/code
METHOD_DIR = CODE_DIR.parent                       # methods/TimeVQVAE
BENCH_ROOT = METHOD_DIR.parent.parent              # benchmark/
DEFAULT_DATA = BENCH_ROOT / "dataset" / "Heston" / "heston_S_8192x128.npy"
DEFAULT_REF = CODE_DIR / "reference"


def parse_args():
    p = argparse.ArgumentParser(description="TimeVQVAE Heston training driver")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    p.add_argument("--workdir", type=str, required=True,
                   help="per-seed copy of the reference tree (isolates saved_models/)")
    p.add_argument("--config", type=str, default=None,
                   help="path to config.yaml (defaults to <workdir>/configs/config.yaml)")
    p.add_argument("--stage1_epochs", type=int, default=2000)
    p.add_argument("--stage2_epochs", type=int, default=10000)
    p.add_argument("--gen_num", type=int, default=8192)
    p.add_argument("--frac", type=float, default=1.0,
                   help="fraction of the 8192 paths to train on (smoke tests)")
    p.add_argument("--tag", type=str, default="",
                   help="filename prefix for outputs (e.g. 'smoke_')")
    return p.parse_args()


class HestonImporter:
    """Drop-in stand-in for DatasetImporterUCR: exposes X_train/X_test/Y_train/Y_test.

    UCRDataset only touches these four attributes, so the entire reference data
    pipeline (build_data_pipeline -> UCRDataset -> DataLoader) works unchanged.
    A single (dummy) class label 0 is assigned to every path — TimeVQVAE is trained
    unconditionally (p_unconditional handles the null class internally).
    """

    def __init__(self, X, mean, std):
        # X: (N, L) already z-normalised
        self.X_train = X.astype(np.float32)
        self.X_test = X[: min(256, len(X))].astype(np.float32)  # unused in training
        self.Y_train = np.zeros((len(self.X_train), 1), dtype=np.int64)
        self.Y_test = np.zeros((len(self.X_test), 1), dtype=np.int64)
        self.mean = float(mean)
        self.std = float(std)


def build_callback(pl, stage):
    class _LossCSVLogger(pl.Callback):
        def __init__(self, stage):
            super().__init__()
            self.stage = stage
            self.rows = []
            self._accum = {}
            self._count = 0

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if not isinstance(outputs, dict):
                return
            for k, v in outputs.items():
                try:
                    val = float(v.detach().cpu()) if hasattr(v, "detach") else float(v)
                except (TypeError, ValueError):
                    continue
                self._accum[k] = self._accum.get(k, 0.0) + val
            self._count += 1

        def on_train_epoch_end(self, trainer, pl_module):
            if self._count == 0:
                return
            row = {"stage": self.stage, "epoch": int(trainer.current_epoch)}
            for k, v in self._accum.items():
                row[k] = v / self._count
            self.rows.append(row)
            self._accum = {}
            self._count = 0

    return _LossCSVLogger(stage)


def write_loss_csv(path, rows):
    if not rows:
        return
    keys = ["stage", "epoch"]
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    args = parse_args()
    workdir = Path(args.workdir).resolve()
    config_path = Path(args.config).resolve() if args.config else workdir / "configs" / "config.yaml"

    # --- isolate: run inside the per-seed reference copy so get_root_dir()==workdir
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_SILENT", "true")
    sys.path.insert(0, str(workdir))
    os.chdir(workdir)

    import torch
    import pytorch_lightning as pl
    import wandb

    # The reference ExpBase logs every step/epoch via `wandb.log`. The paper's
    # stage scripts satisfy that by building a WandbLogger (which calls wandb.init).
    # We disable the logger, so init wandb in DISABLED mode -> every wandb.log is a
    # silent no-op. No model/loss code is touched.
    wandb.init(project="TimeVQVAE-Heston", mode="disabled")

    from utils import load_yaml_param_settings, save_model  # noqa: E402
    from preprocessing.data_pipeline import build_data_pipeline  # noqa: E402
    from experiments.exp_vq_vae import ExpVQVAE  # noqa: E402
    from experiments.exp_maskgit import ExpMaskGIT  # noqa: E402
    from generators.sample import unconditional_sample  # noqa: E402

    pl.seed_everything(args.seed, workers=True)

    # --- config: force Heston preset (in_channels=1, single class) --------------
    config = load_yaml_param_settings(str(config_path))
    config["dataset"]["dataset_name"] = "Heston"
    config["dataset"]["in_channels"] = 1
    config["dataset"]["data_scaling"] = True
    config["trainer_params"]["gpus"] = [0]  # CUDA_VISIBLE_DEVICES selects the physical GPU
    config["trainer_params"]["max_epochs"]["stage1"] = args.stage1_epochs
    config["trainer_params"]["max_epochs"]["stage2"] = args.stage2_epochs

    # --- data: load prices, subsample for smoke tests, global z-norm ------------
    S = np.load(args.data)  # (8192, 128) price
    assert S.ndim == 2 and S.shape[1] == 128, f"unexpected shape {S.shape}"
    if args.frac < 1.0:
        n = max(config["dataset"]["batch_sizes"]["stage1"], int(round(len(S) * args.frac)))
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(S), size=min(n, len(S)), replace=False)
        S = S[idx]
    mean = float(np.mean(S))
    std = float(np.std(S))
    Xn = (S - mean) / std  # global z-norm (paper's data_scaling)

    importer = HestonImporter(Xn, mean, std)
    batch1 = config["dataset"]["batch_sizes"]["stage1"]
    batch2 = config["dataset"]["batch_sizes"]["stage2"]
    train_loader1 = build_data_pipeline(batch1, importer, config, "train")
    train_loader2 = build_data_pipeline(batch2, importer, config, "train")
    input_length = train_loader1.dataset.X.shape[-1]
    n_classes = 1

    print(f"[seed {args.seed}] train paths={len(Xn)} L={input_length} "
          f"mean={mean:.4f} std={std:.4f} "
          f"stage1_ep={args.stage1_epochs} stage2_ep={args.stage2_epochs}", flush=True)

    t0 = time.time()

    # ========================= STAGE 1 — VQ-VAE ================================ #
    cb1 = build_callback(pl, "stage1")
    exp1 = ExpVQVAE(input_length, config, len(train_loader1.dataset))
    trainer1 = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[cb1],
        max_epochs=args.stage1_epochs,
        devices=config["trainer_params"]["gpus"],
        accelerator="gpu",
    )
    trainer1.fit(exp1, train_dataloaders=train_loader1)
    # persist stage1 modules into <workdir>/saved_models (MaskGIT will load them)
    save_model({
        "encoder_l": exp1.encoder_l, "decoder_l": exp1.decoder_l, "vq_model_l": exp1.vq_model_l,
        "encoder_h": exp1.encoder_h, "decoder_h": exp1.decoder_h, "vq_model_h": exp1.vq_model_h,
    }, id="Heston")
    stage1_last = cb1.rows[-1].get("loss") if cb1.rows else None
    print(f"[seed {args.seed}] stage1 done in {time.time()-t0:.1f}s last_loss={stage1_last}", flush=True)

    # ========================= STAGE 2 — MaskGIT prior ======================== #
    cb2 = build_callback(pl, "stage2")
    exp2 = ExpMaskGIT(input_length, config, len(train_loader2.dataset), n_classes)
    trainer2 = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[cb2],
        max_epochs=args.stage2_epochs,
        devices=config["trainer_params"]["gpus"],
        accelerator="gpu",
    )
    trainer2.fit(exp2, train_dataloaders=train_loader2)
    save_model({"maskgit": exp2.maskgit}, id="Heston")
    stage2_last = cb2.rows[-1].get("loss") if cb2.rows else None
    print(f"[seed {args.seed}] stage2 done last_loss={stage2_last}", flush=True)

    # ========================= GENERATION ===================================== #
    device = torch.device("cuda", 0)
    maskgit = exp2.maskgit.to(device).eval()
    gen_t0 = time.time()
    _, _, x_new = unconditional_sample(maskgit, args.gen_num, device, batch_size=256)  # (N,1,L) normalised
    x_new = x_new.squeeze(1).cpu().numpy()  # (N, L)
    gen_sec = time.time() - gen_t0

    # invert global z-norm -> price scale
    gen_price = (x_new * std + mean).astype(np.float64)
    gen_has_nan = bool(np.isnan(gen_price).any())

    # ========================= SAVE OUTPUTS =================================== #
    tag = args.tag
    gen_dir = METHOD_DIR / "generated_paths" / f"seed_{args.seed}"
    loss_dir = METHOD_DIR / "losses"
    w_dir = METHOD_DIR / "weights"
    for d in (gen_dir, loss_dir, w_dir):
        d.mkdir(parents=True, exist_ok=True)

    np.save(gen_dir / f"{tag}generated_paths_8192x128.npy", gen_price)
    write_loss_csv(loss_dir / f"{tag}seed_{args.seed}_stage1_losses.csv", cb1.rows)
    write_loss_csv(loss_dir / f"{tag}seed_{args.seed}_stage2_losses.csv", cb2.rows)

    torch.save({
        "encoder_l": exp1.encoder_l.state_dict(), "decoder_l": exp1.decoder_l.state_dict(),
        "vq_model_l": exp1.vq_model_l.state_dict(), "encoder_h": exp1.encoder_h.state_dict(),
        "decoder_h": exp1.decoder_h.state_dict(), "vq_model_h": exp1.vq_model_h.state_dict(),
        "maskgit": exp2.maskgit.state_dict(),
    }, w_dir / f"{tag}seed_{args.seed}_model.pt")

    config_out = {
        "method": "TimeVQVAE",
        "seed": args.seed,
        "in_channels": 1, "input_length": input_length,
        "n_fft": config["VQ-VAE"]["n_fft"],
        "encoder_dim": config["encoder"]["dim"],
        "n_resnet_blocks": config["encoder"]["n_resnet_blocks"],
        "downsampled_width": config["encoder"]["downsampled_width"],
        "codebook_sizes": config["VQ-VAE"]["codebook_sizes"],
        "codebook_dim": config["VQ-VAE"]["codebook_dim"],
        "ema_decay": config["VQ-VAE"]["decay"],
        "perceptual_loss_weight": config["VQ-VAE"]["perceptual_loss_weight"],
        "maskgit": config["MaskGIT"],
        "guidance_scale": config["class_guidance"]["guidance_scale"],
        "LR": config["exp_params"]["LR"], "weight_decay": config["exp_params"]["weight_decay"],
        "batch_sizes": config["dataset"]["batch_sizes"],
        "stage1_epochs": args.stage1_epochs, "stage2_epochs": args.stage2_epochs,
        "preprocessing": "global z-norm by train mean/std (paper data_scaling)",
    }
    with open(w_dir / f"{tag}seed_{args.seed}_config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    metadata = {
        "method": "TimeVQVAE",
        "seed": args.seed,
        "shape": list(gen_price.shape),
        "scale": "price (original)",
        "znorm_mean": mean, "znorm_std": std,
        "n_train_paths": int(len(Xn)),
        "generated_mean": float(np.mean(gen_price)), "generated_std": float(np.std(gen_price)),
        "real_mean": float(np.mean(S)), "real_std": float(np.std(S)),
        "gen_sec": gen_sec, "train_time_sec": time.time() - t0,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
        "date": datetime.now().isoformat(timespec="seconds"),
        "stage1_epochs": args.stage1_epochs, "stage2_epochs": args.stage2_epochs,
        "stage1_last_loss": stage1_last, "stage2_last_loss": stage2_last,
        "gen_has_nan": gen_has_nan,
    }
    with open(gen_dir / f"{tag}metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[seed {args.seed}] SAVED gen shape={gen_price.shape} "
          f"gen_mean={metadata['generated_mean']:.3f} real_mean={metadata['real_mean']:.3f} "
          f"gen_has_nan={gen_has_nan} total={time.time()-t0:.1f}s", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
