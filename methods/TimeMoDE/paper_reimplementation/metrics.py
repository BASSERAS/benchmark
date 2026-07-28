"""TimeMoDE paper metrics: Context-FID, Discriminative, Predictive.

TimeMoDE (Appendix B.4) reports the same three generation metrics as the
scarce-time-series benchmark it builds on (Yao et al., 2025):

  - Context-FID (c-FID) : Frechet distance between TS2Vec embeddings of real
                          vs generated windows (lower = better).
  - Discriminative      : |accuracy - 0.5| of a post-hoc sequence classifier
                          (lower = better; 0 = indistinguishable).
  - Predictive          : train-on-synthetic / test-on-real one-step MAE
                          (lower = better).

To stay comparable with the other methods already in this benchmark, we REUSE
the benchmark's existing, validated implementations rather than reimplement them:

  - c-FID    : DiffusionTS reference repo Utils.context_fid.Context_FID
               (the original TS2Vec-based implementation).
  - disc/pred: TimeDiT paper_reimplementation yoon_metrics (GRU classifier /
               predictor).  NOTE: the TimeMoDE paper text says "LSTM"; the
               canonical TimeGAN benchmark code (and this repo) uses a 2-layer
               GRU.  We keep the repo standard for cross-method comparability
               and flag the discrepancy here.

Arrays are (N, L, C) float32 in [0, 1].
"""
import os
import sys

# DiffusionTS reference repo (Utils/, Models/) — same wiring as
# DiffusionTS/paper_reimplementation/metric/compute_torch_metrics.py
_HERE = os.path.dirname(os.path.abspath(__file__))
_DIFFTS_REF = os.path.abspath(os.path.join(
    _HERE, "..", "..", "DiffusionTS", "code", "reference"))
_TIMEDIT = os.path.abspath(os.path.join(
    _HERE, "..", "..", "TimeDiT", "paper_reimplementation"))
for p in (_DIFFTS_REF, _TIMEDIT):
    if p not in sys.path:
        sys.path.insert(0, p)

from Utils.context_fid import Context_FID  # noqa: E402
from yoon_metrics import discriminative_score, predictive_score  # noqa: E402


def context_fid(real, fake):
    """TS2Vec Context-FID (real trains the encoder)."""
    n = min(len(real), len(fake))
    return float(Context_FID(real[:n], fake[:n]))


__all__ = ["context_fid", "discriminative_score", "predictive_score"]
