# Heston Dataset — log-return preprocessing variant, 4096 paths, seq_len = 128

## Why this folder exists

This is a **variant of the main [Heston dataset](../README.md)** built for the
`preprocessing_with_log_returns` experiment. Two things change versus the main benchmark:

1. **Sample count.** The train / test / disc splits are **4096 paths** each (not 8192).
2. **A path-shadowing query split** of **512** strictly-independent paths is added.

Everything else — the SDE, the parameters, the Euler-Maruyama full-truncation scheme, the RNG
convention — is **byte-identical** to the main benchmark. The generator here simply *imports* the
untouched parent `generate_heston.generate_heston` and calls it with different sample counts, so the
paths are the same SDE draws truncated to 4096, not a reparameterised model.

The point of the experiment is to feed each generator **log-returns with the volatility scaled out**
(the SBTS preprocessing) instead of standardized prices, then map the generated returns back to price
space. This folder holds the real data for that pipeline.

## Parameters (unchanged from the main benchmark)

| Symbol | Meaning | Value |
|--------|---------|-------|
| $\mu$ | Drift | 0.05 |
| $\kappa$ | Mean-reversion speed | 2.0 |
| $\theta$ | Long-run variance | 0.04 |
| $\xi$ | Vol-of-vol | 0.3 |
| $\rho$ | Spot-vol correlation | −0.7 |
| $S_0$ | Initial price | 100.0 |
| $v_0$ | Initial variance | 0.04 |
| $dt$ | Time step | 1/250 (daily) |
| $T$ | Sequence length | 128 |

## Splits

| Split | Seed | N | Role |
|-------|:----:|:----:|------|
| **train** | 0 | 4096 | The only data a generator ever sees. LS4 is fit on this split (after log-return preprocessing). |
| **test** | 1 | 4096 | Held-out real reference. Every A/B metric scores generated paths against this split. |
| **disc** | 2 | 4096 | Third independent split, used **only** by the A18/A19 discriminative/predictive classifiers. |
| **ps** | 3 | 512 | Fresh path-shadowing query paths. Strictly independent of every other split **and** of the 1M generated bank, per the path-shadowing protocol. |

Seeds 0/1/2 match the main benchmark's train/test/disc convention exactly. The path-shadowing query
split uses a **new** seed (3), so it never overlaps train/test/disc or the generated bank.

## Files

All arrays are **float64**; price (`S`) and variance (`v`) are saved for every split (the Heston-spec
metrics A33/A34 need variance).

| File | Split | Field | Shape |
|------|-------|-------|-------|
| `heston_S_4096x128.npy` | train (seed 0) | $S_t$ | (4096, 128) |
| `heston_v_4096x128.npy` | train (seed 0) | $v_t$ | (4096, 128) |
| `heston_S_test_4096x128.npy` | test (seed 1) | $S_t$ | (4096, 128) |
| `heston_v_test_4096x128.npy` | test (seed 1) | $v_t$ | (4096, 128) |
| `heston_S_disc_4096x128.npy` | disc (seed 2) | $S_t$ | (4096, 128) |
| `heston_v_disc_4096x128.npy` | disc (seed 2) | $v_t$ | (4096, 128) |
| `heston_S_ps_512x128.npy` | ps (seed 3) | $S_t$ | (512, 128) |
| `heston_v_ps_512x128.npy` | ps (seed 3) | $v_t$ | (512, 128) |
| `generate_datasets.py` | — | — | Generator (imports the untouched parent `generate_heston`) |

## The SBTS log-return preprocessing

Each generator is trained on volatility-scaled log-returns, exactly as SBTS does it:

$$
R = \log\!\frac{S_{:,1:}}{S_{:,:-1}} \quad (4096, 127), \qquad
\sigma = \operatorname{std}(R)\ \text{(pooled over all paths} \times \text{timesteps)}, \qquad
\tilde R = R \,\frac{\sqrt{dt}}{\sigma}.
$$

A dummy zero column is prepended to get $\tilde X \in \mathbb{R}^{4096 \times 128}$. Generation is
inverted by $R_{\text{gen}} = \tilde R_{\text{gen}}\, \sigma/\sqrt{dt}$, then
$S_{\text{gen}}[:,0]=100$, $S_{\text{gen}}[:,t{+}1]=S_{\text{gen}}[:,t]\exp(R_{\text{gen}}[:,t])$.

**Estimated on the train split (seed 0, 4096×128):**

| Quantity | Value |
|----------|-------|
| $dt$ | 0.004000 (= 1/250) |
| $\sqrt{dt}$ | 0.063246 |
| $\sigma = \operatorname{std}(R)$ | **0.01263163** |
| $\operatorname{std}(\tilde R)$ (check) | 0.063246 (= $\sqrt{dt}$) |

$\sigma$ is the single pooled standard deviation of all $4096 \times 127$ raw log-returns — the same
estimator SBTS uses (`sigma = float(R.std())`).

## Reproduce

```bash
cd dataset/Heston/preprocessing_with_log_returns
python generate_datasets.py
```

This writes all eight `.npy` files and prints the SBTS $\sigma$. Sanity check: every `heston_S*.npy`
has `S[:, 0].mean() == 100` and `S[:, 0].std() == 0` (all paths start at $S_0 = 100$).
