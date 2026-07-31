#!/usr/bin/env python
"""Rebuild the experiment README's path-shadowing block from build_cross_tables.py.

The block is the four collapsed <details> sweep tables plus the expanded 1M
headline table and its win-count comment. Run this after ANY change to
build_cross_tables.py (new method, new metric row, new RMSE routing) --
GUIDELINE 10.2: never hand-type a cell.

    /home/tbasseras/gpu-venv/bin/python inject_readme_ps.py

The block bounds are DETECTED, not hardcoded: the first `<details>` after the
"nested prefixes of the same 1M bank" paragraph, through the
`<!-- PS strict win-counts` comment. An earlier throwaway revision of this
script pinned literal line numbers, which is exactly the kind of thing that
mangles the file the first time a row is added above the block.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
README = os.path.join(HERE, "README.md")

out = subprocess.run([PY, "build_cross_tables.py", "--which", "PS", "--all-bank-sizes"],
                     cwd=HERE, capture_output=True, text=True, check=True).stdout

blocks, cur = [], None
for ln in out.splitlines():
    if ln.startswith("<!-- ===== PS STRICT-PDF"):
        cur = {"label": ln.split("bank ")[1].split(" =====")[0].strip(),
               "table": [], "wins": None, "in": False}
        blocks.append(cur)
    elif ln.startswith("<table>"):
        cur["in"] = True
        cur["table"].append(ln)
    elif ln.startswith("</table>"):
        cur["table"].append(ln)
        cur["in"] = False
    elif cur and cur["in"]:
        cur["table"].append(ln)
    elif ln.startswith("<!-- PS strict win-counts"):
        cur["wins"] = ln

assert len(blocks) == 5, f"expected 5 banks, got {len(blocks)}"


def summary(wins):
    """'TimeDiT (raw)=14, CSDI=3, LS4=1' -> 'TimeDiT (raw) 14 · CSDI 3 · LS4 1'"""
    tail = wins.split("): ", 1)[1].rsplit("-->", 1)[0].strip()
    return " · ".join(p.strip().replace("=", " ") for p in tail.split(","))


new = []
for b in blocks[:-1]:
    new += ["<details>",
            f'<summary><b>Bank {b["label"]}</b> — winners: {summary(b["wins"])}</summary>',
            ""] + b["table"] + ["", "</details>", ""]
last = blocks[-1]
new += ["#### Bank 1 000 000 — headline", ""] + last["table"] + ["", last["wins"]]

lines = open(README, encoding="utf-8").read().splitlines()
start = next(i for i, l in enumerate(lines)
             if l.startswith("<details>")
             and any("nested prefixes of the same 1M bank" in x
                     for x in lines[max(0, i - 15):i]))
end = next(i for i, l in enumerate(lines) if l.startswith("<!-- PS strict win-counts"))
assert start < end, (start, end)
open(README, "w", encoding="utf-8").write(
    "\n".join(lines[:start] + new + lines[end + 1:]) + "\n")

print(f"replaced README lines {start + 1}-{end + 1}: {end - start + 1} -> {len(new)}")
for b in blocks:
    print(" ", b["label"], "|", summary(b["wins"]))
