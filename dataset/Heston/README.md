# Heston Dataset — seq_len = 128

## Quick description

8 192 synthetic price paths generated from the **Heston stochastic volatility model**
(Heston 1993) using an Euler–Maruyama full-truncation scheme.
Each path has **128 time steps** representing roughly half a year of daily prices
(dt = 1/250).

## Stochastic differential equations

$$
dS_t = \mu S_t \, dt + \sqrt{v_t} \, S_t \, dW_t^S
$$
$$
dv_t = \kappa (\theta - v_t^+) \, dt + \xi \sqrt{v_t^+} \, dW_t^v
$$
$$
\mathbb{E}[dW_t^S \, dW_t^v] = \rho \, dt
$$

where $v_t^+ = \max(v_t, 0)$ is the full-truncation fix to keep variance non-negative.

## Parameters

| Symbol | Meaning | Value |
|--------|---------|-------|
| $\mu$ | Drift | 0.05 |
| $\kappa$ | Mean-reversion speed | 2.0 |
| $\theta$ | Long-run variance | 0.04 (≈ 20 % vol) |
| $\xi$ | Vol-of-vol | 0.3 |
| $\rho$ | Spot–vol correlation | −0.7 |
| $S_0$ | Initial price | 100.0 |
| $v_0$ | Initial variance | 0.04 |
| $dt$ | Time step | 1/250 (daily) |
| $T$ | Sequence length | 128 |
| $N$ | Number of paths | 8 192 |
| Seed | RNG seed | 0 |

The Feller condition $2\kappa\theta \geq \xi^2$ gives $2 \times 2.0 \times 0.04 = 0.16 \geq 0.09$: satisfied.

## Files

| File | Shape | dtype | Description |
|------|-------|-------|-------------|
| `heston_S_8192x128.npy` | (8192, 128) | float64 | Price paths $S_t$ |
| `heston_v_8192x128.npy` | (8192, 128) | float64 | Variance paths $v_t$ |

## Reproduce

```bash
cd dataset/Heston
python generate_heston.py
```

Expected output: `heston_S_8192x128.npy` with `S[:, 0].mean() ≈ 100`.
