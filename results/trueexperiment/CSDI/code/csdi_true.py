"""CSDI (Tashiro et al., NeurIPS 2021) on TrueDataset -- shared model + data library.

This is the library half of the port. `train_true.py` fits a model and writes the
A/B bank; `generate_bank_true.py` reloads a checkpoint and draws an arbitrarily
sized bank (used for the 8 192-path conditional-CRPS pool of guideline section 8).
Both need the same subclass and the same normalisation chain, so both live here
rather than being duplicated -- a second copy of `rescale_to_s0` that drifts from
the first is exactly the class of bug guideline section 13.9 is about.

WHAT IS THE AUTHORS' AND WHAT IS OURS
-------------------------------------
Everything that defines the method is imported unmodified from `reference/`, a
byte-identical copy of `methods/CSDI/code/reference/` (verified by md5 at copy
time; digests in code/README.md). `main_model.CSDI_base` is the authors'
diffusion model, `diff_models.diff_CSDI` the 2-D (time x feature) Transformer
denoiser, and `calc_loss` / `calc_loss_valid` / `impute` / `get_side_info` are
theirs verbatim. Ours is only the data plumbing: which array is loaded, how it is
standardised, and the thin subclass that pins `cond_mask = 0`.

CSDI IS NATIVELY MULTIVARIATE -- THIS IS ONE JOINT MODEL, NOT EIGHT
--------------------------------------------------------------------
Guideline section 2, question 1. `target_dim` is a constructor argument that flows
into the feature embedding and into `forward_feature`, a Transformer that attends
ACROSS assets at every residual block. Setting `target_dim=8` therefore gives one
model over all eight assets, and A20 (the cross-asset covariance row) is a
question this architecture can actually answer. That matters here: the 28
realised cross-asset correlations on this build average 0.609, so a per-asset
ensemble would be discarding most of the dependence structure.

NORMALISATION -- PRICES, PER CHANNEL
-------------------------------------
    S (price) --(-mean_a)/std_a--> standardized   [model trains and samples here]
    sample    --*std_a + mean_a--> price

Per **channel**, following CSDI's own PhysioNet convention and the d = 8 Heston
sibling. Decided 2026-08-26 with the repo owner over the log-return alternative;
the reasoning and the cost are in code/README.md under "Input space". Two facts
about this dataset that the reader needs at the point of use:

  * `mean_a` is ~100.006 for every asset and `std_a` ranges 0.373 (BTC) to 0.853
    (DOGE). That is a **30x smaller dynamic range** than the Heston sibling saw
    (std 11.14-18.55), because a 64-minute crypto window moves far less than a
    one-year Heston path. The z-score absorbs it -- that is what it is for -- but
    the absolute reconstruction error in price space is scaled by `std_a`, so a
    given error in standardized space costs 30x less here.
  * every real path is anchored at S[:, 0, :] == 100 exactly, so after
    standardisation every path starts at the *same* value, (100 - mean_a)/std_a
    ~ -0.016. That first marginal is a point mass and the model must spend
    capacity learning it. `rescale_to_s0` below re-imposes the anchor exactly on
    the way out, which is why the metadata carries `s0_rescaled: true`.

A per-channel affine map is a shift-and-scale of each asset independently, so it
leaves every cross-asset **correlation** invariant. A20 is unaffected by this
choice.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "reference"))

from main_model import CSDI_base  # noqa: E402  (path set above)

S0 = 100.0
GPU_NAME = "A100-SXM4-80GB"

# Verbatim from the released reference/config/base.yaml -- the config that
# reproduced the paper's Table 2, and byte-identical to what the d = 1 run in
# methods/CSDI and the d = 8 Heston run both used. `epochs` and `batch_size` are
# overridable on the CLI; any override is recorded in weights/seed_{i}_config.json
# under "retuned_for_truedata" and flips "paper_hyperparams" to false.
#
# epochs=200 is kept VERBATIM rather than rescaled to match the Heston sibling's
# gradient-step count. That sibling ran N=8192 / batch 16 = 512 steps/epoch;
# this build gives 6144 / 16 = 384, so 200 epochs here is 76 800 steps against
# 102 400 there -- 25 % less optimisation for the same nominal hyperparameter.
# Decided 2026-08-26 with the repo owner: keeping the number the authors
# published is worth more than equalising a budget across two datasets that
# differ in a dozen other ways, and it keeps `paper_hyperparams: true` honest.
# The validation curve in losses/seed_{i}_losses.csv is the evidence for whether
# 200 was enough; read it before trusting the A-table.
BASE_CONFIG = {
    "train": {"epochs": 200, "batch_size": 16, "lr": 1.0e-3},
    "diffusion": {
        "layers": 4, "channels": 64, "nheads": 8, "diffusion_embedding_dim": 128,
        "beta_start": 0.0001, "beta_end": 0.5, "num_steps": 50,
        "schedule": "quad", "is_linear": False,
    },
    "model": {
        "is_unconditional": 1,        # unconditional generation variant, paper Sec 4.1
        "timeemb": 128, "featureemb": 16,
        "target_strategy": "random",  # unused: forward() forces cond_mask = 0
    },
}


class CSDI_TrueData(CSDI_base):
    """Unconditional CSDI over K jointly-modelled assets, sequence length L.

    Why unconditional generation is `is_unconditional=1` plus `cond_mask == 0`
    -----------------------------------------------------------------------
    The paper (Sec 4.1 / Appendix C) states the `is_unconditional=1` variant "can
    also be used for data generation". In that mode `CSDI_base.set_input_to_diffmodel`
    feeds the network ONLY the noisy sequence -- `cond_mask` never gates the network
    input, it only selects which points enter the loss via
    `target_mask = observed_mask - cond_mask`. So with `observed_mask = 1` and
    `cond_mask = 0` everywhere:

      * training  -> target_mask == 1 everywhere -> every timestep is a denoising
                     target, i.e. the plain DDPM objective
                     E_t || eps - eps_theta(x_t, t) ||^2 ;
      * sampling  -> `impute` with cond_mask == 0 collapses to pure ancestral
                     sampling, no conditioning term.

    Training and generation therefore see the identical input distribution. The
    architecture, diffusion process and hyperparameters are the paper's
    `is_unconditional` variant, unchanged. `get_side_info`, `calc_loss`,
    `calc_loss_valid` and `impute` are the parent's code, untouched.
    """

    def __init__(self, config, device, target_dim=8):
        super().__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()   # (B, L, K)
        B, L, K = observed_data.shape
        observed_mask = torch.ones(B, L, K, device=self.device)
        gt_mask = torch.zeros(B, L, K, device=self.device)
        observed_tp = torch.arange(L, device=self.device).float().unsqueeze(0).expand(B, -1)

        observed_data = observed_data.permute(0, 2, 1)                   # (B, K, L)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        cut_length = torch.zeros(B, device=self.device).long()
        return observed_data, observed_mask, observed_tp, gt_mask, observed_mask, cut_length

    def forward(self, batch, is_train=1):
        observed_data, observed_mask, observed_tp, _gt, _fpm, _cl = self.process_data(batch)
        cond_mask = torch.zeros_like(observed_mask)
        side_info = self.get_side_info(observed_tp, cond_mask)
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    @torch.no_grad()
    def generate(self, n_paths, seq_len, gen_batch=128):
        """Draw `n_paths` unconditional samples -> (n_paths, seq_len, K), standardized.

        Returns all K channels. The d = 1 script indexed `samples[:, 0, 0, :]`,
        which silently keeps asset 0 only -- do not copy that line here.
        """
        self.eval()
        out, done = [], 0
        while done < n_paths:
            B = min(gen_batch, n_paths - done)
            cond_mask = torch.zeros(B, self.target_dim, seq_len, device=self.device)
            observed_tp = torch.arange(seq_len, device=self.device).float().unsqueeze(0).expand(B, -1)
            side_info = self.get_side_info(observed_tp, cond_mask)
            dummy = torch.zeros(B, self.target_dim, seq_len, device=self.device)  # shapes the sampler only
            samples = self.impute(dummy, cond_mask, side_info, n_samples=1)       # (B, 1, K, L)
            out.append(samples[:, 0].permute(0, 2, 1).cpu().numpy())              # (B, L, K)
            done += B
        return np.concatenate(out, axis=0)[:n_paths]


def load_split(data_dir, seq_tag, split=""):
    """Load one TrueDataset split as (N, T, d) float64 prices.

    `split` is "" (train), "val", "test", "disc" or "valdisc". The filename
    convention is `true_S{_split}_{seq_tag}.npy` -- note the tag is a CLI
    argument, not a constant: guideline section 13.1 records that hardcoding
    `8192x128x8` sends the loader silently to the wrong build.
    """
    suffix = f"_{split}" if split else ""
    path = os.path.join(data_dir, f"true_S{suffix}_{seq_tag}.npy")
    if not os.path.exists(path):
        raise SystemExit(f"ABORT: split not found: {path}")
    return np.load(path).astype(np.float64)


def zscore_stats(S_train):
    """Per-channel mean/std over (path, time). Fitted on TRAIN ONLY.

    Guideline section 3.5: val and test are standardised with the training
    statistics, never with their own. Refitting on val would leak the validation
    era's level into the model's input scale; refitting on test would void the
    run outright.
    """
    return S_train.mean(axis=(0, 1)), S_train.std(axis=(0, 1))


def rescale_to_s0(Xg):
    """Put every path on S[:, 0, :] == 100.0 exactly, per guideline section 4.

    A per-(path, asset) constant multiplier is exactly a shift of the log-price
    level, so every log-return is bit-identical before and after and the whole
    A-table except the level rows is unaffected by construction. Overwriting the
    first row alone would instead dump the entire correction into the first
    increment. The final assignment removes the float residue of the division.

    Returns (Xg, n_nonpositive_at_t0, n_nonpositive_total). Clipping non-positive
    prices to 1e-6 IS a real distortion -- unlike the multiplier it does not
    preserve log-returns -- so it is counted and reported rather than hidden.
    `n_s0` is broken out separately because a non-positive *first* price is far
    worse than an interior one: the multiplier is 100/S[:, 0, :], so a first
    price clipped to 1e-6 rescales that entire path by 1e8.
    """
    n_s0 = int((Xg[:, 0:1, :] <= 0).sum())
    n_total = int((Xg <= 0).sum())
    Xg = np.where(Xg <= 0.0, 1e-6, Xg)
    Xg = Xg * (S0 / Xg[:, 0:1, :])
    Xg[:, 0, :] = S0
    return Xg, n_s0, n_total
