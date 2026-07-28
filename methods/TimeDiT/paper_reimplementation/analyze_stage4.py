"""Analyze the STAGE-4 broad HP search and pick a winner under the user's rule.

RULE (verbatim intent):
  "no ema worked better so u keep this config and try optimize parameters ...
   if ur method will allow to beat the no ema u keep it but else u do a no ema
   ... until reaching paper results" (paper Sine disc target = 0.0086)

So no-EMA is the BASE.  EMA is only "kept" if, at a MATCHED (lr, weight_decay,
batch) operating point AND at the same fixed budget, its disc_mean is lower
than the no-EMA twin by more than the twin's disc_std (i.e. a real, not noise,
improvement).  Otherwise we report the best no-EMA recipe as the winner.

Reads hpo_stage4_shard0.jsonl + hpo_stage4_shard1.jsonl (24 rows total).
Prints: full ranking, the no-EMA leader, the EMA head-to-head verdict, and the
recommended recipe to carry into the full-data multi-seed GATE.
"""
import glob
import json

PAPER_DISC = 0.0086


def load_rows():
    rows = []
    for f in sorted(glob.glob("hpo_stage4_shard*.jsonl")):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def key(r):
    return (r["lr"], r["weight_decay"], r["batch"])


def main():
    rows = load_rows()
    if not rows:
        print("no stage-4 rows yet")
        return
    rows = [r for r in rows if r.get("disc_mean") == r.get("disc_mean")]  # drop NaN
    rows.sort(key=lambda r: r["disc_mean"])

    print(f"=== STAGE-4 broad HP search: {len(rows)} finished trials "
          f"(paper Sine disc target {PAPER_DISC}) ===\n")
    print(f"{'rank':>4} {'disc_mean':>10} {'disc_std':>9} {'pred':>7} "
          f"{'lr':>7} {'ema':>6} {'wd':>7} {'batch':>6}")
    for i, r in enumerate(rows):
        print(f"{i+1:>4} {r['disc_mean']:>10.4f} {r.get('disc_std',0):>9.4f} "
              f"{r['pred_mean']:>7.4f} {r['lr']:>7.0e} {r['ema']:>6} "
              f"{r['weight_decay']:>7.0e} {r['batch']:>6}")

    no_ema = [r for r in rows if r["ema"] == 0.0]
    ema = {key(r): r for r in rows if r["ema"] > 0.0}
    base = min(no_ema, key=lambda r: r["disc_mean"]) if no_ema else None

    print("\n--- no-EMA vs EMA head-to-head (matched lr/wd/batch) ---")
    ema_genuine_wins = []
    for r in no_ema:
        e = ema.get(key(r))
        if e is None:
            continue
        margin = r["disc_mean"] - e["disc_mean"]           # >0 => EMA lower/better
        genuine = margin > r.get("disc_std", 0.0)          # beats no-EMA's own noise
        tag = "EMA GENUINELY BETTER" if genuine else ("ema lower (within noise)"
              if margin > 0 else "no-EMA better")
        print(f"  lr={r['lr']:.0e} wd={r['weight_decay']:.0e} b={r['batch']:>3}: "
              f"noEMA={r['disc_mean']:.4f}+-{r.get('disc_std',0):.4f}  "
              f"EMA={e['disc_mean']:.4f}  d={margin:+.4f}  -> {tag}")
        if genuine:
            ema_genuine_wins.append((e, r))

    print("\n--- VERDICT ---")
    if base is not None:
        print(f"best no-EMA : disc={base['disc_mean']:.4f}+-{base.get('disc_std',0):.4f} "
              f"pred={base['pred_mean']:.4f}  lr={base['lr']:.0e} "
              f"wd={base['weight_decay']:.0e} batch={base['batch']}")
    if ema_genuine_wins:
        best_ema = min(ema_genuine_wins, key=lambda t: t[0]["disc_mean"])[0]
        print(f"EMA genuinely beats its no-EMA twin in {len(ema_genuine_wins)} cell(s); "
              f"best such EMA: disc={best_ema['disc_mean']:.4f} lr={best_ema['lr']:.0e} "
              f"wd={best_ema['weight_decay']:.0e} batch={best_ema['batch']}")
        winner = min(rows, key=lambda r: r["disc_mean"])
    else:
        print("EMA does NOT genuinely beat no-EMA anywhere -> keep no-EMA (per rule).")
        winner = base

    print(f"\nRECOMMENDED for full-data multi-seed GATE:")
    print(f"  lr={winner['lr']:.0e}  ema={winner['ema']}  "
          f"weight_decay={winner['weight_decay']:.0e}  batch={winner['batch']}  "
          f"steps={winner.get('steps')}")
    print(f"  subset-4000 disc={winner['disc_mean']:.4f}+-{winner.get('disc_std',0):.4f} "
          f"(target {PAPER_DISC}); overlaps target: "
          f"{winner['disc_mean']-winner.get('disc_std',0) <= PAPER_DISC}")


if __name__ == "__main__":
    main()
