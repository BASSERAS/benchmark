# Drawdown-constrained position sizing

This experiment asks whether a scenario generator can support better futures
risk decisions, rather than whether it can produce trading alpha.

For every held-out ES session, the first 65 price points are observed. The
runner finds the 256 scenario-bank paths with the most similar prefixes, using
the same return, path-shape, volatility, and autocorrelation features as the
paper's path-shadowing evaluation. Their next 32 increments form a conditional
scenario fan.

The portfolio holds a constant long ES futures notional over that future
window. For each scenario, the runner calculates the largest peak-to-trough
decline in futures P&L. It then chooses the largest leverage for which the
predicted 90th percentile drawdown is no greater than 1% of capital, subject to
a five-times leverage cap. The rule is applied to the actual held-out
continuation.

The main economic outputs are the realized risk-limit violation rate, violation
severity, and average permitted leverage. A model is useful if it supports more
exposure without materially increasing violations, or reduces violations
without materially reducing exposure. This is an internal risk-budgeting
experiment; it is neither derivative pricing nor an estimate of exchange
margin requirements.

The first run is validation-only. The protected test and crisis-event periods
remain unavailable until the model and decision rule are frozen.

## Prepared command

Replace the MP bank path if the final ES experiment produces a newer bank.

```bash
source /home/samer/venvs/mfc/bin/activate
python experiments/scripts/run_drawdown_risk_budgeting.py \
  --run-dir runs/es_drawdown_risk_budgeting_validation_v1 \
  --protocol-manifest experiments/protocols/drawdown_risk_budgeting_validation_v1.json \
  --bank reference=runs/final_nested_volatility_only_validation_20260804/real/ES/seed_0/reference/validation_bank.npy \
  --bank nested_mp=runs/final_nested_volatility_only_validation_20260804/real/ES/seed_0/volatility_only_nested_mp/validation_8192/validation_bank.npy \
  --bank sbts=runs/real_data_protocol/SBTS/ES/seed_0/primary_2026_holdout/generated_bank.npy \
  --include-historical-neighbor \
  --primary-method nested_mp
```

The runner writes one directory per method, path-level outcomes, bootstrap
intervals, paired MP-minus-comparator intervals, a safety-factor frontier,
`report.json`, and `SUMMARY.md`.

The reported frontier holds the 1% monetary budget fixed and varies a
predeclared conservatism factor on predicted drawdown. This is the appropriate
leverage--violation curve: merely changing the monetary budget would rescale
both leverage and drawdown without changing violations until the leverage cap
binds.
