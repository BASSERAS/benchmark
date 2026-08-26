# d = 8 latent-size selection — criterion, fixed before the sweep was run

LS4's released `solar_weekly` preset has `z_dim = 5`. Every other architectural
constant (`d_state = 64`, `d_model = 64`, `n_layers = 4`, `backbone = autoreg`,
`s4_type = s4`, `latent_type = split`, `sigma = 0.1`) and the whole optimisation
recipe (`AdamW(lr = 1e-3, wd = 0)` + `ReduceLROnPlateau(patience = 20, factor = 0.5)`
+ `EMA(lamb = 0.99, start_step = 200)`, batch 128) is **carried over unchanged**
from the committed d = 1 run.

`z_dim` is the one hyperparameter that cannot be carried across dimension without
an argument. The released value was tuned for a **1-channel** series; here the
decoder must reconstruct **8 correlated channels** whose joint law is exactly the
object under test (metric A20 scores the cross-asset covariance directly). A
5-dimensional latent is not self-evidently sufficient for that, and a
"5 per asset → 40" rescaling is not self-evidently necessary either. Neither is
assumed; both are measured.

## Split discipline

`z_dim` is scored on the **validation split**
(`dataset/HestonMultiAsset/heston_ma_S_val_8192x252x8.npy`), **never** on the
test split. Every reported metric is computed against the test split, so tuning
on it would leak. The validation prices are standardised with the **training**
scaler `(mu_j, sigma_j)`, not with their own — using validation statistics would
itself be a form of leakage.

## Sweep design (deliberately cheap)

Fixed before any config was run:

- **candidates:** `z_dim ∈ {5, 16, 32, 40}` — the released value, two intermediate
  sizes, and the "5 per asset" rescaling;
- **budget:** 20 epochs on a **25 % subset** (`--frac 0.25`, i.e. 2048 train /
  2048 validation paths), seed 0 only;
- **everything else identical across the four runs**, including the seed, so the
  only varying quantity is `z_dim`.

This is a screening test, not a convergence study. It is sized to separate
"clearly too small" from "adequate", and it is honest about not being able to
separate two adjacent adequate sizes. The measured cost is ≈ 1 min per config.

## Criterion

**Primary objective:** lowest **mean validation ELBO over the last 5 epochs**
(epochs 15–19). The mean over 5 epochs rather than the single final epoch, and
rather than the minimum, because a single-epoch minimum on a 2048-path subset is
dominated by noise; averaging the tail is the cheapest available variance
reduction, and taking the minimum would bias the comparison toward whichever run
happened to fluctuate low.

**Hard constraint** (a violation disqualifies whatever the ELBO says): the loss
must be finite at every logged epoch — `first_nan_epoch` must be `null`.

**Tie-break** (validation ELBO within 1 % of the winner): **prefer the smaller
`z_dim`**, on the usual parsimony grounds and because it stays closer to the
released preset. This is stated now so that a near-tie cannot later be resolved
in favour of whichever value happens to score better on the test metrics.

## Known limits of this criterion — stated, not hidden

1. **ELBO is not the reported metric.** The benchmark scores distributional
   distances (A6–A11), a discriminative score (A18) and covariance error (A20).
   A lower ELBO on a 2048-path subset does not mechanically imply better
   A-metrics. Selecting on the actual metrics would, however, require scoring
   against a split and would risk exactly the leakage the split discipline above
   forbids, so the ELBO is used as the honest proxy and this gap is disclosed.
2. **20 epochs is not convergence.** The final runs get 100 epochs — the same
   budget as the committed d = 1 run. A latent size that is best at 20 epochs is
   not guaranteed to be best at 100. The screening is accepted as-is because the
   alternative, a full 100-epoch sweep, buys a marginally better hyperparameter
   at several times the cost of the experiment it is meant to configure.
3. **One seed.** There is no seed-to-seed error bar on the selection itself.

## Result

| `z_dim` | val ELBO, mean ep 15–19 | val ELBO, final ep | train ELBO, final ep | params | finite |
|--------:|------------------------:|-------------------:|---------------------:|-------:|:------:|
| **40**  | **−5.3692**             | −2.1053            | −4.7250              | 2 205 832 | yes |
| 32      | −2.9061                 | −3.8584            | −0.8543              | 2 190 832 | yes |
| 16      | −1.4742                 | −2.9408            | −3.0958              | 2 163 904 | yes |
| 5 (released) | 71.6199            | 66.9943            | 66.9252              | 2 147 767 | yes |

**Selected: `z_dim = 40`.** It wins outright — nothing else falls within the 1 %
band, so the parsimony tie-break never fires.

The interesting row is the released `z_dim = 5`: at **71.62** it is not merely
worse, it is on the wrong side of zero while all three larger latents are
comfortably negative, and its training ELBO (66.93) is essentially equal to its
validation ELBO (66.99) — the model is not overfitting, it is *under-capacity*.
A 5-dimensional latent cannot carry 8 correlated channels. This is the measured
justification for treating `z_dim` as the one non-transferable hyperparameter,
and it is the reason `retuned_for_d8` lists it.

The ordering 40 < 32 < 16 also has no interior optimum inside the tested range,
so the honest reading is "bigger was still better at 20 epochs"; `z_dim = 40`
is the largest candidate and was chosen on that basis, not because a maximum was
located. Larger latents were not tested, and that boundary is disclosed rather
than presented as a converged optimum.

Full numbers: `zdim_selection.json`. Per-epoch curves: `zdim{N}_seed_0_losses.csv`.
