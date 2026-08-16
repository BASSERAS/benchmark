# Deep-MKV-TS, paper reimplementation on Heston

Medians over the paper's four training seeds (0, 1, 3, 4), 8192 paths per split,
evaluated against `dataset/Heston/heston_S_disc_8192x128.npy`. Lower is better.

Tolerance: within 25% of the paper value or 0.005 absolute, whichever is wider.

The Reference row is informational, not a reproduction target: the published row came from the `local_gaussian` reference, while this reimplementation uses the Guyon-Lekeufack reference that paper Section 2.1 describes. Only the Deep-MKV-TS row is scored against the paper.

| Row | Metric | Ours (median, 4 seeds) | Paper | Delta | Verdict |
|-----|--------|-----------------------:|------:|------:|:-------:|
| Reference | SWD | 0.0381 | 0.060 | -0.0219 | context (better) |
| Reference | RV W1 | 0.0566 | 0.089 | -0.0324 | context (better) |
| Reference | \|r\| ACF | 0.0484 | 0.068 | -0.0196 | context (better) |
| Reference | Early-future | 0.1019 | 0.214 | -0.1121 | context (better) |
| Reference | MDD W1 | 0.0287 | 0.059 | -0.0303 | context (better) |
| Deep-MKV-TS | SWD | 0.0737 | 0.062 | +0.0117 | match |
| Deep-MKV-TS | RV W1 | 0.0128 | 0.014 | -0.0012 | match |
| Deep-MKV-TS | \|r\| ACF | 0.0132 | 0.016 | -0.0028 | match |
| Deep-MKV-TS | Early-future | 0.0178 | 0.018 | -0.0002 | match |
| Deep-MKV-TS | MDD W1 | 0.0232 | 0.022 | +0.0012 | match |

## Per-seed values

### Reference

| Seed | SWD | RV W1 | \|r\| ACF | Early-future | MDD W1 |
|------|------:|------:|------:|------:|------:|
| 0 | 0.0368 | 0.0595 | 0.0512 | 0.1202 | 0.0344 |
| 1 | 0.0328 | 0.0574 | 0.0492 | 0.1123 | 0.0276 |
| 3 | 0.0395 | 0.0549 | 0.0476 | 0.0883 | 0.0222 |
| 4 | 0.0555 | 0.0559 | 0.0475 | 0.0915 | 0.0298 |

### Deep-MKV-TS

| Seed | SWD | RV W1 | \|r\| ACF | Early-future | MDD W1 |
|------|------:|------:|------:|------:|------:|
| 0 | 0.0528 | 0.0210 | 0.0230 | 0.0180 | 0.0103 |
| 1 | 0.0655 | 0.0127 | 0.0128 | 0.0108 | 0.0218 |
| 3 | 0.0818 | 0.0129 | 0.0064 | 0.0413 | 0.0423 |
| 4 | 0.0839 | 0.0112 | 0.0135 | 0.0177 | 0.0246 |

**Overall: 5/5 Deep-MKV-TS metrics within tolerance (rows scored: Deep-MKV-TS).**

