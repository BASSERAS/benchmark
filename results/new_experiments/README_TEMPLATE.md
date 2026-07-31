<!--
=============================================================================
 README TEMPLATE — results/new_experiments/experiment_<X>/<Method>/README.md
=============================================================================
 Copy this file, fill every <PLACEHOLDER>, delete every HTML comment, and the
 result satisfies guideline_new_experiment.md §7 by construction.

 THE ONE RULE THAT MATTERS (§7.0): every table below is EMITTED BY A TOOL and
 pasted verbatim. You never type a number into a table. If a cell looks ugly
 -- `0.67399 ± 0`, a CI of `0` -- you leave it ugly. Hand-editing machine
 output is a silent data-loss event: in this very benchmark, annotating the
 `target_*` rows as "(target) / --" would have erased the one target row that
 is genuinely not constant (Experiment B, `target_low_confidence_fraction`,
 std 8.63e-05).

 Before you write a word of this file:
   python ../../tools/check_method_layout.py --root . --experiment <X>
 must print PASS. (§6.6)

 After you finish: re-run every `*Reproduce:*` command below and require an
 EMPTY DIFF against what is in the file. That check is what catches a
 hand-edited cell. (§8 item 17)
=============================================================================
-->

# <Method> — Experiment <X>: <Delayed Drawdown Memory | Balanced Heston Parameter Mixture>

<!-- 3-6 sentences, no numbers you have not read from a JSON this session:
     what the experiment tests, what this method is, and the headline verdict
     stated as a ratio to the perfect floor. State the verdict up front; do
     not make the reader find it in section 4. -->

<PARAGRAPH: what the experiment asks of a generator, in one sentence.>
<PARAGRAPH: what <Method> is, its parameter count, and the frozen config.>
<PARAGRAPH: the verdict — "<Method> <passes|fails> <what>, at <N>x the
perfect floor on <metric>.">

---

## 1. PDF metrics — protocol evaluator

<!-- §7.2. THIS SECTION COMES FIRST AND STANDS ALONE. It is the protocol's own
     evaluator, run unchanged (PDF §7 item 7). Nothing from the A1-A34 battery
     belongs here. -->

<!-- Block 1: provenance. Which evaluator script, which commit, which input. -->
Scored by `protocol/experiments/scripts/evaluate_<drawdown_memory|heston_parameter_mixture>.py`,
**run unchanged** (PDF §7 item 7), on the 5 model banks against `test.npy`.
<!-- NOTE the asymmetry: the SCRIPT for B is evaluate_heston_parameter_mixture.py
     but the OUTPUT FILE it writes is seed_N_heston_mixture.json. They do not
     match, and guessing either from the other is how §7.10 E4 happens. -->

The perfect-floor column is the same evaluator on 5 true-DGP draws
(seeds 1000-1004), which is what makes the ratio meaningful.

<!-- Block 2: THE TABLE. Emitted, never typed. Paste the tool's stdout here. -->
<PASTE THE FULL AGGREGATOR TABLE HERE — every row the tool prints, including
rows whose std is 0 and whose CI is 0.>

<!-- Block 3: the caption. Explain any zero IN PROSE. Never in a cell. -->
<CAPTION: if some rows have std 0, say WHY — e.g. "these are properties of the
frozen test set and the frozen oracle, recomputed identically by every seed, so
a zero here means 'never resampled', not 'we measured a variance and it came out
small'." Then verify the claim rather than asserting it:>

```python
# Confirms which target_* rows are genuinely constant. Run it; do not assume.
rows = [flatten(json.load(open(f))) for f in sorted(glob.glob("pdf_metrics/seed_*.json"))]
for k in [k for k in rows[0] if "target" in k]:
    vals = [r[k] for r in rows]
    print("IDENTICAL" if len(set(map(repr, vals))) == 1 else "** VARIES", k, vals[0])
```

### 1.1 PRIMARY panel — the four metrics the protocol designates

<!-- Experiment A: early_hit_rate_error, future_rv_hit_gap_error,
                   early_history_incremental_r2_error, future_rv_wasserstein
     Experiment B: regime proportion TVD, W-bar_param (DERIVED, not emitted by
                   the evaluator -- say so), RV Wasserstein, leverage RMSE.
     There is NO aggregate score in either experiment. Do not invent one. -->

<PASTE THE FOUR-ROW PRIMARY TABLE.>

<PROSE: each primary metric as a ratio to the floor. This is the paragraph a
reader who reads nothing else will read.>

### 1.2 Headline — raw diagnostics behind the primary panel

<!-- PDF §5 permits raw diagnostics (std ratio, posterior confidence, group
     means, novelty) to be reported WITHOUT the mean +- std treatment. Use that
     exemption here, and say you are using it. -->

<PASTE / PROSE.>

### 1.3 Validation vs test — evidence that `test.npy` stayed blind

<!-- The same evaluator against `disc.npy`. If the two sides agree, the model
     did not overfit the test set; that agreement IS the firewall evidence.
     ALSO: declare here any post-hoc transformation applied to the banks
     (PDF §1.3) -- e.g. an S0 renormalisation -- with the invariant you
     asserted, not assumed. -->

<TABLE + PROSE.>

### 1.4 Protocol reporting block (PDF §5)

<!-- §7.2 block 5. The required rows, all of them: -->

| Item | Value |
|---|---|
| Model seeds | `0, 1, 2, 3, 4` |
| **Independent retraining** | <5 distinct checkpoint MD5s; per-seed config; 5 distinct 100-epoch loss curves; all 10 A+B checkpoints mutually distinct> |
| **Trained on this experiment's own data** | <`train.npy` MD5 …; fitted scaler mu/sigma, which DIFFER from the other experiment's — the one check that does not trust the config file> |
| Failed / unstable runs | <PDF §1.5: "Failed or unstable seeds must be reported and must not be silently replaced." If a seed's final-epoch loss spiked, say so, say whether generation uses EMA weights, and give that seed's rank on the primary metric.> |
| Evaluator | <script, unchanged> |
| Post-hoc transformations | <PDF §1.3 — or "none"> |
| Output contract | <re-MEASURED from the arrays on disk: finite, > 0, (8192,128), dtype, S0 == 100.0 — not asserted> |
| CI convention | 95 %, t(0.975, 4) = 2.776, half-width = t·s/√5 |

<!-- The two experiments take DIFFERENT --exclude-prefix lists. Copy the right
     one; do not merge them. B has an oracle_gate block that A does not. -->

*Reproduce (A):* `python ../../tools/aggregate_pdf_metrics.py --model-dir . --floor-dir ../perfect_floor --pattern '*_drawdown_memory.json' --label <Method> --exclude-prefix configuration sources`

*Reproduce (B):* `python ../../tools/aggregate_pdf_metrics.py --model-dir . --floor-dir ../perfect_floor --pattern '*_heston_mixture.json' --label <Method> --exclude-prefix oracle_gate sources configuration`

---

## 2. Metrics A1–A34 + B — benchmark standard battery, mean ± std across 5 seeds

<!-- §7.3. A33/A34 are DROPPED in both experiments (Theo's ruling) — they land
     as null via the adapter's `v = None`. Say so; do not leave the reader
     wondering why the battery has holes. -->

### 2.1 A1–A34

<PASTE THE EMITTED TABLE.>

### 2.2 B — curve-shape metrics

<PASTE THE EMITTED TABLE.>

<!-- CROSS-SUITE DISAGREEMENT. If §1 and §2 disagree, that disagreement is a
     RESULT, not an inconsistency to smooth over. State it explicitly and name
     the generalised lesson. The one found here:

       a GRU discriminator (A18/A19) sitting at 1.5x floor and the predictive
       score at 1.0x floor, against 15-79x for the distributional metrics,
       means every individual path is plausible and only the POPULATION is
       wrong -> **A18/A19 are blind to mode collapse.**

     Do not write "essentially floor-level" when the measurement says 1.5x.
     Quote the ratio. -->

*Reproduce:*
`python ../../tools/make_metrics_tables.py --model-dir . --floor-dir ../perfect_floor --label <Method> --table A`
(and `--table B`).

---

## 3. Stylised Facts Diagnostic (real vs <Method>, seed 0)

<!-- §7.4 + §7.7. Seed 0 by convention. Every figure carries a prose caption
     quoting a number that has been cross-checked against the evaluator JSON. -->

![stylised facts diagnostic](plots/heston_diagnostics.png)

<CAPTION quoting at least one number verified against a JSON this session.>

---

## 4. Losses

<!-- §7.5. If generation uses weights other than the last checkpoint (EMA,
     best-validation, ...), SAY SO HERE. It changes how a final-epoch loss
     spike should be read. Do not hide an anomalous seed behind a
     final-loss-only table. -->

![loss convergence](losses/loss_convergence.png)

| Seed | Final loss | Min loss | Epoch of min |
|---|---|---|---|
<PASTE — five rows, read from losses/seed_N_losses.csv, not from memory.>

---

## 5. File layout

<!-- §7.6. Every file on disk appears here and every entry here exists on disk.
     Verify with `find . -type f | sort` against this tree — do not eyeball it. -->

```
experiment_<X>/<Method>/
├── README.md
├── code/            train_<method>_experiment.py, compute_metrics_experiment.py, logs/
├── generated_paths/ seed_{0..4}/{generated_paths_8192x128.npy, metadata.json, generation_manifest.json}
├── weights/         seed_N_model.pt, seed_N_config.json
├── losses/          seed_N_losses.csv, loss_convergence.png
├── pdf_metrics/            seed_N_<stem>.json   (vs test.npy)
├── pdf_metrics_validation/ seed_N_<stem>.json   (vs disc.npy)
├── metrics_summary.csv
├── seed_{0..4}_metrics.json
└── plots/
```

<!-- FIGURES THAT DO NOT EXIST: if a planned figure could not be produced, do
     NOT leave a dash, a placeholder, or an empty image link. Delete the row
     and state in prose why the figure is absent. (§7.7) -->

---

## Provenance

Protocol: `synthetic_benchmark_protocol_drawdown_heston_mixture.pdf`. Method,
gate, hardware rules and the full PDF conformance cross-check are in
[`guideline_new_experiment.md`](../../guideline_new_experiment.md).
