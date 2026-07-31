#!/usr/bin/env python3
"""build_report.py — emit `ps_rmse_report.tex` for the RMSE-convention report.

Answers Theo's question: *why does the CSDI rv RMSE in the reproducibility PDF
differ from the one in the README?*  Ships the explanation, the five
cross-method PS tables, and every figure in the experiment folder.

GUIDELINE 10.2 ("never hand-type a cell") is enforced structurally: every table
number and every Winner cell is pulled through `build_cross_tables` — the SAME
functions the README tables are rendered with (`PS_MODELS`, `PS_QUANT`,
`PS_METRICS`, `_ps_value`, `_ps_winner_idx`, `render_tables.fmt`). The LaTeX
tables therefore cannot drift from the README: they share one code path and one
set of `pdf_summary.json` inputs.

The only literals in this file are (a) prose, and (b) the *historical*
pre-99977a3 numbers quoted from PDF Table 8, which no longer exist in any JSON
on disk and are marked as such in HISTORICAL_OLD_RMSE below.

Usage:
    /home/tbasseras/gpu-venv/bin/python report/build_report.py
    cd report && pdflatex -interaction=nonstopmode ps_rmse_report.tex
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)

import build_cross_tables as B  # noqa: E402  (path must be set first)

RT = B.RT                       # canonical renderer, already imported by B

# ---------------------------------------------------------------------------
# Historical values. Produced by the PRE-99977a3 aggregation; not recoverable
# from any pdf_summary.json currently on disk, so they are quoted rather than
# computed. Source: PDF Table 8 (§4.3) == README before commit 99977a3,
# cross-checked against `git show 99977a3`.
# ---------------------------------------------------------------------------
HISTORICAL_OLD_RMSE = {  # quantity -> {method: root-of-mean RMSE @1M bank}
    "cum":  {"CSDI": 0.04925, "LS4": 0.04894, "SBTS": 0.05523},
    "step": {"CSDI": 0.01245, "LS4": 0.01245, "SBTS": 0.01421},
    "rv":   {"CSDI": 0.01772, "LS4": 0.01819, "SBTS": 0.01972},
}

# ---------------------------------------------------------------------------
# The forecaster RMSE cells as they stood BEFORE the 2026-07-31 convention fix,
# i.e. what forecaster/pdf_bridge.py::_metrics_frozen_rmse produced under (A).
# Quoted, not computed: the fix overwrote chronos2_pdf.json / timesfm_pdf.json,
# and the pre-fix values are recoverable only from git history
# (`git show HEAD~1:.../forecaster/chronos2_pdf.json`). The post-fix values in
# the same table ARE read live from those JSONs, so the ratio column cannot
# drift away from the shipped numbers.
# ---------------------------------------------------------------------------
FORECASTER_PRE_FIX = {  # method -> {quantity: (A) root-last RMSE}
    "Chronos-2": {"cum": 0.049206019623857795,
                  "step": 0.01295227380817077,
                  "rv": 0.23074659175504625},
    "TimesFM":   {"cum": 0.05093751948245248,
                  "step": 0.013315990532177139,
                  "rv": 0.2864952717060515},
}

# LaTeX-ification of the display labels used by build_cross_tables.
LATEX_LABEL = {
    "RMSE ↓": r"RMSE $\downarrow$",
    "CRPS ↓": r"CRPS $\downarrow$",
    "coverage₅₀ (→0.50)": r"coverage$_{50}$ ($\to$0.50)",
    "coverage₉₀ (→0.90)": r"coverage$_{90}$ ($\to$0.90)",
    "width₅₀ (diag)": r"width$_{50}$ (diag)",
    "width₉₀ (diag)": r"width$_{90}$ (diag)",
    "lower miss₉₀ (→0.05)": r"lower miss$_{90}$ ($\to$0.05)",
    "upper miss₉₀ (→0.05)": r"upper miss$_{90}$ ($\to$0.05)",
    "cum (M×H trajectory)": r"cum ($M\times H$ trajectory)",
    "step (M×H)": r"step ($M\times H$)",
    "rv (M scalar)": r"rv ($M$ scalar)",
}


def tex_escape(s):
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"),
                 ("%", r"\%"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def lbl(s):
    return LATEX_LABEL.get(s, tex_escape(s))


def ps_table(bank_size):
    """One PS table at `bank_size`, numbers/winners straight from build_cross_tables."""
    gen_path = os.path.join(EXP, B.PS_MODELS[0][1])
    full = json.load(open(gen_path))
    oracle_q = full["heston_oracle"]["by_bank_size"][str(bank_size)]["quantities"]
    rw_q = full["rw_baseline"]

    names = [n for n, _, _ in B.PS_MODELS]
    paths_kinds = [(os.path.join(EXP, p), k) for _, p, k in B.PS_MODELS]
    ncol = 1 + len(names) + 3

    rows = [r"\begin{tabular}{l" + "r" * (len(names) + 2) + "l}", r"\toprule",
            "Quantity / metric & " + " & ".join(tex_escape(n) for n in names)
            + r" & Oracle & RW & Winner \\", r"\midrule"]

    wins = {n: 0 for n in names}
    for qi, (qn, qlabel) in enumerate(B.PS_QUANT):
        if qi:
            rows.append(r"\addlinespace")
        rows.append(r"\multicolumn{%d}{l}{\textbf{%s}} \\" % (ncol, lbl(qlabel)))
        for metric, mlabel, rule in B.PS_METRICS:
            vals = [B._ps_value(p, k, qn, metric, bank_size) for p, k in paths_kinds]
            wi = B._ps_winner_idx(vals, rule)
            if wi is not None:
                wins[names[wi]] += 1
            cells = []
            for i, v in enumerate(vals):
                t = RT.fmt(v)
                cells.append(r"\textbf{%s}" % t if (wi is not None and i == wi) else t)
            ov = RT.fmt(oracle_q[qn][metric]["value"])
            rv = RT.fmt(rw_q[qn][metric]["value"])
            win = r"\textbf{%s}" % tex_escape(names[wi]) if wi is not None else "---"
            rows.append(f"{lbl(mlabel)} & " + " & ".join(cells)
                        + f" & {ov} & {rv} & {win} " + r"\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows), wins


def fig(path, caption, width=r"\linewidth"):
    """One figure; `path` is relative to the report/ dir."""
    return "\n".join([r"\begin{figure}[H]", r"\centering",
                      r"\includegraphics[width=%s]{%s}" % (width, path),
                      r"\caption{%s}" % caption, r"\end{figure}"])


def collect_figures():
    """Every PNG in the experiment folder, grouped: (section, [(relpath, caption)], per_row)."""
    methods = ["CSDI", "TimeDiT", "LS4", "SBTS"]
    groups = []

    diag = []
    for m in methods:
        for arm, tag in (("plots", "log-return preprocessing"),
                         ("baseline_no_preproc/plots", "raw price, no preprocessing")):
            p = f"{m}/{arm}/heston_diagnostics.png"
            if os.path.exists(os.path.join(EXP, p)):
                diag.append((p, f"{m} --- {tag}. Eight-panel Heston stylised-fact "
                                f"diagnostics (seed 0)."))
    groups.append(("Stylised-fact diagnostics (8-panel)", diag, 1))

    ps = []
    for m in methods:
        base = ("TimeDiT/baseline_no_preproc/path_shadowing/plots" if m == "TimeDiT"
                else f"{m}/path_shadowing/plots")
        for f, cap in (("pdf_crps_vs_banksize.png",
                        "CRPS vs bank size (log $x$); solid = generator, dashed = "
                        "size-matched Heston oracle, shaded = 95\\% bootstrap band"),
                       ("pdf_coverage_calibration.png",
                        "Coverage calibration at the 1M bank; dashed lines = 0.50 / "
                        "0.90 nominal targets"),
                       ("crps_per_step.png", "CRPS per horizon step")):
            p = f"{base}/{f}"
            if os.path.exists(os.path.join(EXP, p)):
                tag = "TimeDiT (raw)" if m == "TimeDiT" else m
                ps.append((p, f"{tag} --- {cap}."))
    groups.append(("Path-shadowing convergence and calibration", ps, 1))

    tr = []
    for m in methods:
        for f, cap in (("losses/loss_convergence.png", "training loss convergence"),
                       ("plots/disc_classifier_loss.png",
                        "discriminative-score classifier loss (A18)"),
                       ("plots/pred_score_loss.png", "predictive-score loss (A19)")):
            p = f"{m}/{f}"
            if os.path.exists(os.path.join(EXP, p)):
                tr.append((p, f"{m} --- {cap}."))
    groups.append(("Training and scoring diagnostics", tr, 2))

    emb = []
    for m in methods:
        for arm, tag in (("plots", "log-ret"), ("baseline_no_preproc/plots", "raw")):
            d = os.path.join(EXP, m, arm)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.endswith(("_pca.png", "_tsne.png")):
                    kind = "PCA" if f.endswith("_pca.png") else "t-SNE"
                    seed = f.split("_")[1]
                    emb.append((f"{m}/{arm}/{f}",
                                f"{m} ({tag}) --- {kind}, seed {seed}."))
    groups.append(("Embedding visualisations (PCA / t-SNE)", emb, 2))
    return groups


BANK_SIZES = [4096, 16384, 65536, 262144, 1000000]

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{float}
\usepackage{pdflscape}
\usepackage[colorlinks=true,linkcolor=blue,urlcolor=blue]{hyperref}
\usepackage{caption}
\captionsetup{font=small}
\setlength{\parskip}{0.5em}
% Long \texttt{} identifiers (np.sqrt(samp).mean(axis=1), _metrics_frozen_rmse)
% cannot hyphenate and were overflowing the right margin; give TeX slack.
\setlength{\emergencystretch}{4em}
\sloppy
\title{\textbf{The Path-Shadowing RMSE Discrepancy}\\[0.3em]
\large Why the reproducibility report and the repository README disagree,\\
and what the number actually measures}
\author{\texttt{results/Heston/preprocessing\_with\_log\_returns}\\
BASSERAS/benchmark}
\date{31 July 2026}
\begin{document}
\maketitle
"""


def main():
    L = [PREAMBLE, r"\tableofcontents", r"\clearpage"]

    csdi_old = HISTORICAL_OLD_RMSE["rv"]["CSDI"]
    csdi_new = B._ps_value(os.path.join(EXP, B.PS_MODELS[0][1]), "gen",
                           "rv", "rmse", 1000000)
    L.append(r"""
\section{The short answer}

Nothing was recomputed and no bank changed. \textbf{The two documents apply a
different final aggregation step to the same per-path errors}, and the CSDI
horizon-realised-volatility (rv) row is where the gap is widest.

\begin{center}
\begin{tabular}{llr}
\toprule
Source & Aggregation & CSDI rv RMSE @1M \\
\midrule
Report Table 8 (\S4.3, ``originally published'') & $\sqrt{\overline{se}}$ (root last) & %s \\
Report Table 5 (\S3.5, ``corrected'') & $\overline{\sqrt{se}}$ (root inside) & %s \\
Repository README (current) & $\overline{\sqrt{se}}$ (root inside) & %s \\
\bottomrule
\end{tabular}
\end{center}

So the discrepancy you spotted is \textbf{internal to the reproducibility
report}: it prints both conventions, in Table 8 and Table 5, without ever
showing them side by side. The README agrees with Table 5 to every printed
digit; it disagrees with Table 8 by a factor of %.3f. Only the RMSE rows differ
--- CRPS, coverage, width and miss-rate cells are identical between Table 8 and
Table 5, because those metrics are plain arithmetic means and are untouched by
the aggregation choice.
""" % (RT.fmt(csdi_old), RT.fmt(csdi_new), RT.fmt(csdi_new), csdi_old / csdi_new))

    L.append(r"""
\section{How the RMSE is computed}

\subsection{From ensemble to per-path error}

Fix a held-out query path $q \in \{1,\dots,m\}$, $m = 512$, and a horizon step
$u \in \{1,\dots,H\}$. Path shadowing retrieves $K = 256$ nearest neighbours
from the bank and forms an ensemble forecast $Y_{q,u,k}$. The squared error of
the \emph{ensemble mean} against the realised value $y_{q,u}$ is
\[
se_{q,u} \;=\; \Bigl(\tfrac{1}{K}\textstyle\sum_{k=1}^{K} Y_{q,u,k} \;-\; y_{q,u}\Bigr)^{2},
\]
and each path is reduced to a single scalar by averaging over the horizon,
\[
se_q \;=\; \frac{1}{H}\sum_{u=1}^{H} se_{q,u}.
\]
Everything above is common to both documents. \textbf{The disagreement is
entirely in the last step}, which turns the length-$m$ vector $(se_q)$ into one
reported number.

\subsection{The two aggregations}

\[
\textbf{(A) root-last:}\quad R_A=\sqrt{\frac{1}{m}\sum_{q=1}^{m} se_q}
\qquad\qquad
\textbf{(B) root-inside:}\quad R_B=\frac{1}{m}\sum_{q=1}^{m}\sqrt{se_q}
\]

(A) is the textbook root-mean-square error: one square root, taken last. (B) is
the \emph{mean per-path root error} --- a square root per query path, then an
ordinary average.

Because $x \mapsto \sqrt{x}$ is \textbf{concave}, Jensen's inequality gives
\[
R_B \;=\; \mathbb{E}\bigl[\sqrt{se}\bigr] \;\le\; \sqrt{\mathbb{E}[se]} \;=\; R_A ,
\]
with equality if and only if every $se_q$ is identical. \textbf{(B) is therefore
always the smaller number}, and the gap widens as the spread of per-path errors
grows. This is why the effect is largest on rv: a scalar quantity with no
horizon averaging to smooth its error distribution.

\subsection{What the code actually does}

The shipped scorer (\texttt{path\_shadowing\_pdf.py}, \texttt{\_agg}) is
convention (B):

\begin{quote}\footnotesize\ttfamily
return float(np.sqrt(per).mean()) \\
\hspace*{2em} if kind == "rmse" else float(per.mean())
\end{quote}

\noindent
\texttt{np.sqrt(per).mean()} roots \emph{then} averages. The paired bootstrap
uses the matching form, \texttt{np.sqrt(samp).mean(axis=1)}, so the confidence
intervals are consistent with the point estimate.

\subsection{A naming caveat worth stating plainly}

Under convention (B) the reported quantity \textbf{is not a root-mean-square
error}, even though every table calls it ``RMSE''. For the scalar quantity rv
the horizon collapses ($H=1$), so $se_q = e_q^2$ and
\[
R_B=\frac{1}{m}\sum_q \sqrt{e_q^{2}}=\frac{1}{m}\sum_q |e_q| \;=\; \textbf{MAE exactly.}
\]
The rv ``RMSE'' row is a mean absolute error wearing the wrong label. For cum
and step it is a horizon-RMS averaged across paths --- a hybrid with no standard
name. This is inherited from the report, not invented here, but it should be
said out loud in any write-up that quotes the number.
""")

    rows = []
    for q in ("cum", "step", "rv"):
        for m_ in ("CSDI", "LS4", "SBTS"):
            o = HISTORICAL_OLD_RMSE[q][m_]
            idx = [n for n, _, _ in B.PS_MODELS].index(m_)
            p, k = B.PS_MODELS[idx][1], B.PS_MODELS[idx][2]
            n_ = B._ps_value(os.path.join(EXP, p), k, q, "rmse", 1000000)
            rows.append(r"%s & %s & %s & %s & %.3f \\" %
                        (q, m_, RT.fmt(o), RT.fmt(n_), o / n_))
    L.append(r"""
\section{Why the difference exists}

The repository originally used (A). Commit \texttt{99977a3}, ``Align strict
path-shadowing RMSE with reproducibility report Table 5'', switched
\texttt{\_agg}, \texttt{\_boot\_ci} and the \texttt{terminal\_rmse} diagnostic
in all five scorers to (B) and re-ran the CSDI / LS4 / SBTS pipelines. Only the
RMSE fields moved; every other metric reproduced bit-identically, and the
Heston-oracle block stayed bit-identical across all three summaries.

The measured inflation $R_A/R_B$ at the 1M bank:

\begin{center}
\begin{tabular}{llrrr}
\toprule
Quantity & Method & (A) root-last & (B) root-inside & ratio $R_A/R_B$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{center}

The ratio is near-constant \emph{within} a quantity and very different
\emph{between} quantities ($\approx 1.18$ cum, $\approx 1.06$ step,
$\approx 1.28$ rv). That is the signature predicted above: the gap is set by the
dispersion of the per-path error distribution of the quantity, and across these
three \emph{generators} it barely moves.

One caveat, since it turned out to matter in \S5. That stability holds across
models of the \emph{same kind}; it is not a law. Re-running the two conditional
forecasters under (B) gave a cum ratio of $1.180$ and a step ratio of $1.056$,
both in line with the generators --- but an rv ratio of only $1.050$, against the
generators' $1.28$. Retrieval and direct forecasting produce differently
dispersed per-path rv errors, and the ratio follows the dispersion. So the ratio
may be used to \emph{anticipate} the size of a correction, never to substitute
for recomputing it.
""")

    L.append(r"""
\section{Three defects in the reproducibility report}

These are stated so the report can be corrected; none of them changes a number
in the tables of Section~\ref{sec:tables}.

\paragraph{1. \S2.7 describes the opposite of the shipped code.} The prose says
``the per-path squared errors are averaged over the evaluation set first, and a
\emph{single} square root is taken last'', and the accompanying listing prints
\texttt{np.sqrt(np.mean(per))} with the comment \texttt{\# per holds se\_i, NOT
sqrt(se\_i)}. The shipped code is \texttt{np.sqrt(per).mean()} --- the reverse.
The bootstrap listing is transposed in the same way. \S2.7 therefore documents
convention (A) while every table in the report except Table 8 is convention (B).

\paragraph{2. \S2.7 states Jensen's inequality backwards.} It claims the
mean-of-roots form ``is $\ge$ the above whenever the $se_i$ are not all equal,
and would overstate RMSE''. Since $\sqrt{\cdot}$ is concave the inequality runs
the other way: mean-of-roots \emph{under}states. The data confirms the direction
--- every ratio in the table above exceeds 1.

\paragraph{3. The Table 8 caption swaps the two labels.} It says Table 8's
column ``was produced with an earlier aggregation convention (mean of per-path
root errors)'' and that Table 5 ``follows the $\sqrt{\mathrm{mean}_i(se_i)}$
definition''. Both halves are inverted: Table 8 is the root-last form and
Table 5 is the mean-of-roots form. Table 8's numbers are uniformly \emph{larger},
which by Jensen is only possible if Table 8 is the root-last one. The caption on
Table 5 (``RMSE fixed to $\mathrm{mean}_i(\sqrt{se_i})$'') is the correct one,
and it contradicts both \S2.7 and the Table 8 caption.

\paragraph{Consequence.} The repository was aligned against a \emph{table
caption} that contradicts the report's own methods section. The alignment is
self-consistent and reproducible, but the report should be corrected so a reader
is not led to convention (A) by \S2.7.
""")

    L.append(r"""
\section{A mixed-convention defect in the tables --- found and fixed}

\textbf{Until 2026-07-31 the RMSE rows of the five tables below mixed both
conventions.} The four generators, the Heston oracle and the RW floor used (B).
The two conditional forecasters did not: \texttt{forecaster/pdf\_bridge.py}
pinned them to (A) through a local \texttt{\_metrics\_frozen\_rmse()}, so that a
re-run would reproduce the reproducibility report's published Chronos-2 /
TimesFM cells rather than silently drift away from them.

Because $R_B \le R_A$ always, this \textbf{systematically inflated the
forecasters' RMSE relative to every other column}. The defect was not cosmetic.
\texttt{build\_cross\_tables.py::\_ps\_winner\_idx} ranks \emph{all} model
columns, forecasters included, so the Winner column was silently comparing
inflated cells against un-inflated ones --- while the README footnote next to it
told the reader those cells were not comparable. A table cannot both rank a
number and disclaim it.

\textbf{The fix.} \texttt{\_metrics\_frozen\_rmse()} was deleted;
\texttt{score\_forecaster} now calls the shared \texttt{P.metrics\_with\_ci},
and the \texttt{terminal\_rmse} diagnostic was moved to
\texttt{np.abs(...).mean()} to match \texttt{eval\_bank} line 326 exactly. Both
forecasters were then re-run from their seed-0 recipes. Every non-RMSE leaf of
\texttt{chronos2\_pdf.json} and \texttt{timesfm\_pdf.json} reproduced
bit-identically (83 and 85 leaves respectively; CRPS, coverages, widths and miss
rates all unchanged to the last digit), which confirms the aggregation was the
only thing that moved.
""")

    # Built from FORECASTER_PRE_FIX (historical) + the live JSONs, so the shipped
    # "after" column is by construction the same number the tables in §6 print.
    L.append(r"""
\begin{center}\small
\begin{tabular}{llrrr}
\hline
Method & Quantity & before (A) & after (B) & ratio \\
\hline
""")
    for meth in ("Chronos-2", "TimesFM"):
        path = dict((m, p) for m, p, _ in B.PS_MODELS)[meth]
        live = B._fc_quantities(os.path.join(EXP, path))
        for qn in ("cum", "step", "rv"):
            a = FORECASTER_PRE_FIX[meth][qn]
            b = live[qn]["rmse"]["value"]
            L.append(f"{tex_escape(meth)} & {qn} & {a:.5f} & {b:.5f} & "
                     f"{a / b:.3f} \\\\\n")
    L.append(r"""\hline
\end{tabular}
\end{center}
""")

    L.append(r"""
\textbf{What it changes.} Chronos-2's cum RMSE falls from $0.04921$ to
$0.04168$, against CSDI's $0.04144$: a row that read as a $19\%$ deficit is a
$0.6\%$ one. The estimate offered before the re-run --- ``roughly $0.0416$'' from
the generator cum ratio --- was accurate for cum and step, but the rv ratio came
out at $1.050$ rather than the generators' $1.28$ (see the caveat in \S3), so
only the recomputation settles it.

\textbf{What it does not change.} Win counts were recomputed at all five bank
sizes and are identical: TimeDiT (raw) 14 at every bank, with CSDI 2 / SBTS 2 at
4\,096, CSDI 3 / SBTS 1 at 16\,384, and CSDI 3 / LS4 1 at the top three. CSDI
remains the row minimum for cum and step RMSE, so \textbf{no Winner cell moved}.
The tables in \S6 are the post-fix numbers.

\textbf{The cost.} The reproducibility report's published Chronos-2 / TimesFM
RMSE cells are now \emph{superseded, not reproduced} --- a re-run will no longer
match them, by design. That is the correct trade: one internally consistent
table is worth more than fidelity to a column the report itself never
recomputed. This is recorded in GUIDELINE E16b.
""")

    L.append(r"\clearpage" + "\n" + r"\section{The five cross-method tables}"
             + "\n" + r"\label{sec:tables}" + r"""

Each table is one nested-prefix bank size of the \emph{same} 1M bank
(GUIDELINE \S9.1). Generator and oracle columns move with bank size; the
forecaster columns and the RW floor have no bank-size axis and repeat unchanged.
Winner excludes the oracle and RW floors. Width rows are diagnostics and carry
no winner: a degenerate zero-width interval would ``win'' while being completely
uncalibrated. \textbf{Bold} = best in row. TimeDiT's column is labelled
\textbf{TimeDiT (raw)} because its bank comes from the no-preprocessing
checkpoint; the protocol, $K=256$, query set, embedding, oracle and floor are
identical to the other generators.

\textbf{Every number below is generated by the same code path as the README
tables} (\texttt{build\_cross\_tables.py}), reading the same
\texttt{pdf\_summary.json} files, so the two cannot drift apart.
""")
    for bs in BANK_SIZES:
        tab, wins = ps_table(bs)
        order = sorted(wins.items(), key=lambda x: -x[1])
        won = " $\\cdot$ ".join(f"{tex_escape(k)} {v}" for k, v in order if v)
        bs_lab = f"{int(bs):,}".replace(",", "\\,")
        L.append(r"\begin{landscape}" + "\n"
                 + r"\subsection{Bank size $N = %s$}" % bs_lab + "\n"
                 + r"\begin{center}\footnotesize" + "\n" + tab + "\n"
                 + r"\end{center}" + "\n"
                 + r"\noindent\textbf{Winners (of 18 ranked rows = 6 ranked metrics "
                   r"$\times$ 3 quantities):} " + won + "."
                 + "\n" + r"\end{landscape}")

    L.append(r"\clearpage" + "\n" + r"\section{Figures}" + r"""

Every figure in the experiment folder, grouped. The eight-panel diagnostics are,
row by row: real vs generated path bundles; log-return density and QQ vs real;
ACF of $|r|$ and ACF of $r^2$; 5-day rolling-volatility density and
terminal-price tail survival (log-log). Blue = real test set, red/orange =
generated, dashed = theoretical or empirical reference.

\textbf{Known caption artefact:} the path-bundle panel titles read ``8192
total''; the true count is 4096. The shared plotter
\texttt{metrics/plot\_diagnostics.py} hardcodes 8192 (lines 151/160) from the
main benchmark. The plotted data is the correct 4096-path test set --- only the
string is stale, it affects all four methods identically, and it changes no
number in any table.
""")
    for title, items, per_row in collect_figures():
        if not items:
            continue
        L.append(r"\subsection{%s}" % title)
        w = r"\linewidth" if per_row == 1 else r"0.48\linewidth"
        for p, cap in items:
            L.append(fig("../" + p, cap, w))

    L.append(r"\end{document}")

    out = os.path.join(HERE, "ps_rmse_report.tex")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")

    nfig = sum(len(i) for _, i, _ in collect_figures())
    print(f"wrote {out}")
    print(f"  tables : {len(BANK_SIZES)}")
    print(f"  figures: {nfig}")
    for bs in BANK_SIZES:
        _, wins = ps_table(bs)
        print(f"  bank {bs:>9}: " +
              ", ".join(f"{k}={v}" for k, v in sorted(wins.items(), key=lambda x: -x[1]) if v))


if __name__ == "__main__":
    main()
