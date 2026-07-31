#!/usr/bin/env python
"""Compact PDF: the five path-shadowing comparison tables, nothing else.

The full audit document (`ps_rmse_report.tex`, ~47 pages with every figure and
the whole leaf-diff control) stays where it is -- it is the trail. This is the
thing you actually hand someone: one page of RMSE/MAE definitions, then the five
nested-prefix bank tables.

Every cell comes from `build_report.ps_table`, which is the same
`build_cross_tables` row/winner machinery the READMEs use, so this file adds no
numbers of its own (GUIDELINE 10.2).

    /home/tbasseras/gpu-venv/bin/python build_ps_tables_pdf.py
    cd report && pdflatex -interaction=nonstopmode ps_tables.tex
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_report as R  # noqa: E402

OUT = os.path.join(HERE, "ps_tables.tex")

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.0cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{float}
% 10 columns (metric + 4 generators + 2 forecasters + oracle + RW + winner)
% overflow A4 portrait by ~170pt at \small. Landscape absorbs it without
% shrinking the type; the definitions page stays portrait.
\usepackage{pdflscape}
\setlength{\parskip}{0.5em}
\setlength{\emergencystretch}{3em}
\sloppy
\pagestyle{empty}
\title{\textbf{Strict path shadowing --- cross-method comparison}\\[0.3em]
\large Five nested-prefix bank sizes, one 1\,000\,000-path bank}
\author{\texttt{results/Heston/preprocessing\_with\_log\_returns}\\
BASSERAS/benchmark}
\date{31 July 2026}
\begin{document}
\maketitle
\thispagestyle{empty}
"""

INTRO = r"""
\section*{How RMSE and MAE are aggregated}

Fix a query path $q\in\{1,\dots,m\}$ ($m=512$) and a horizon offset
$u\in\{1,\dots,H\}$ ($H=32$). The forecast is the mean over the $K=256$ retrieved
continuations, so the per-cell squared error and its horizon average are
%
\[
se_{q,u}=\Bigl(\tfrac{1}{K}\textstyle\sum_{k=1}^{K}\hat Y_{q,u,k}-y_{q,u}\Bigr)^{2},
\qquad
se_{q}=\frac{1}{H}\sum_{u=1}^{H}se_{q,u}.
\]

There are two ways to finish, and they are \emph{not} equal. Taking the root last
gives the textbook RMSE; taking it inside first gives the mean of per-path
root-mean-square errors:
%
\[
\underbrace{\text{RMSE}=\sqrt{\frac{1}{m}\sum_{q=1}^{m}se_{q}}}_{\text{root last --- reported below}}
\qquad\qquad
\underbrace{\frac{1}{m}\sum_{q=1}^{m}\sqrt{se_{q}}}_{\text{root inside}} .
\]

Because $\sqrt{\cdot}$ is concave, Jensen's inequality makes the root-inside form
\emph{always} the smaller of the two; it is a real statistic, but it is not an
RMSE. \textbf{The \texttt{cum} and \texttt{step} rows below report the textbook
(root-last) form.}

\paragraph{Why \texttt{rv} gets two rows.} Horizon realised volatility is a
\emph{scalar} per query ($H=1$), so the inner mean collapses and the two
aggregations above reduce to two different standard norms of the same
512-vector of signed errors $e_q$:
%
\[
\text{MAE}=\frac{1}{m}\sum_{q=1}^{m}\lvert e_q\rvert,
\qquad\qquad
\text{RMSE}=\sqrt{\frac{1}{m}\sum_{q=1}^{m}e_q^{2}} .
\]
%
Both are listed. For CSDI at the 1\,000\,000-path bank they are
\textbf{@CSDI_MAE@} and \textbf{@CSDI_RMSE@} --- a ratio of
@CSDI_RATIO@, against @STEP_RATIO@ for \texttt{step}. That ratio is a
direct read-out of how much of the error sits in the tail: it equals 1 when
every query errs by the same amount and grows with error dispersion. rv is the
quantity where the generators separate, and it is also the quantity whose error
distribution is the most skewed --- which is exactly why quoting one norm and
labelling it the other is not a cosmetic issue.

\paragraph{Reading the tables.} Lower is better everywhere except coverage,
whose target is the nominal level. \emph{Oracle} is the Heston ground-truth
ceiling (retrieval from the true data-generating process, size-matched to the
bank); \emph{RW} is the random-walk floor. The Winner column ranks the
generators and forecasters only --- never the oracle or the floor --- and the
two \texttt{width} rows are diagnostic, so they name no winner. That leaves
\textbf{18 ranked rows} per table ($6$ ranked metrics $\times$ $3$ quantities).

$^{\dagger}$ The rv root-last RMSE row is bolded and its winner named in
\emph{italics}, but it is \textbf{not counted} in the 18-row total: ranking two
norms of one error vector would double-count rv against \texttt{cum} and
\texttt{step}.
\clearpage
"""


def main():
    csdi_mae = R.rmse_of("CSDI", "rv", R.RMSE_INSIDE)
    csdi_rmse = R.rmse_of("CSDI", "rv", R.RMSE_LAST)
    step_ratio = (R.rmse_of("CSDI", "step", R.RMSE_LAST)
                  / R.rmse_of("CSDI", "step", R.RMSE_INSIDE))

    # Token replacement, not %-formatting: INTRO is LaTeX and its bare `%`
    # comment lines would be eaten as format specifiers.
    intro = INTRO
    for tok, val in (("@CSDI_MAE@", R.RT.fmt(csdi_mae)),
                     ("@CSDI_RMSE@", R.RT.fmt(csdi_rmse)),
                     ("@CSDI_RATIO@", f"{csdi_rmse / csdi_mae:.2f}"),
                     ("@STEP_RATIO@", f"{step_ratio:.2f}")):
        assert tok in intro, tok
        intro = intro.replace(tok, val)
    L = [PREAMBLE, intro]

    for bs in R.BANK_SIZES:
        tbl, wins = R.ps_table(bs)
        ranked = sum(wins.values())
        order = sorted(wins.items(), key=lambda kv: (-kv[1], kv[0]))
        summary = ", ".join(f"{n} {c}" for n, c in order if c)
        L += [r"\begin{landscape}",
              r"\section*{Bank size $N = %s$}" % f"{bs:,}".replace(",", r"\,"),
              r"\noindent\textbf{Winners (of %d ranked rows):} %s."
              % (ranked, summary),
              r"\begin{center}\small", tbl, r"\end{center}",
              r"\end{landscape}"]

    L.append(r"\end{document}")
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"wrote {OUT} ({len(R.BANK_SIZES)} tables)")


if __name__ == "__main__":
    main()
