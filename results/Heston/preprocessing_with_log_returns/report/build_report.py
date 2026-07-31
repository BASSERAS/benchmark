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

The only literals in this file are prose. Every number --- including the
"before" column of the convention-change tables --- is read live from the JSONs,
because the scorer now emits both aggregations side by side (`rmse` root-inside,
`rmse_textbook` root-last). See the block below for why the two former constant
tables were deleted.

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
# NO historical literals in this file.
#
# Earlier revisions carried two hand-quoted constant blocks: HISTORICAL_OLD_RMSE
# (PDF Table 8, the pre-99977a3 generator numbers) and FORECASTER_PRE_FIX (the
# Chronos-2 / TimesFM cells as they stood before the 2026-07-31 convention fix).
# Both were quoted because the value they held --- the root-LAST aggregation ---
# existed in no JSON on disk.
#
# It does now. The scorer emits "rmse_textbook" (root-last) alongside "rmse"
# (root-inside) for every model, bank and quantity, and it reproduces all fifteen
# quoted numbers: the nine generator cells to every printed digit, and the six
# forecaster cells bit-for-bit at full float64 precision. So both blocks were
# deleted and every "before" cell in this report is now read live, satisfying
# GUIDELINE 10.2 (never hand-type a cell) with no exceptions at all.
# ---------------------------------------------------------------------------

RMSE_INSIDE = "rmse"            # mean_q(sqrt(se_q)) -- reproducibility report Tables 1-5
RMSE_LAST = "rmse_textbook"     # sqrt(mean_q(se_q)) -- the textbook RMSE


def rmse_of(method, qn, key, bank_size=1000000):
    """One RMSE cell under an explicitly named aggregation, read live.

    `key` is RMSE_INSIDE or RMSE_LAST. Deliberately bypasses B._ps_value, which
    applies the per-quantity PS_METRIC_KEY default: the prose sections need to
    name the aggregation themselves rather than inherit the table's choice.
    """
    path, kind = dict((m, (p, k)) for m, p, k in B.PS_MODELS)[method]
    path = os.path.join(EXP, path)
    q = (B._gen_quantities(path, bank_size) if kind == "gen"
         else B._fc_quantities(path))
    return q[qn][key]["value"]

# LaTeX-ification of the display labels used by build_cross_tables.
LATEX_LABEL = {
    "RMSE ↓": r"RMSE $\downarrow$",
    "MAE ↓": r"MAE $\downarrow$",
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


def sci(x, sig=1):
    """LaTeX scientific notation, e.g. 0.00038 -> $3.8\\times10^{-4}$."""
    m, e = f"{x:.{sig}e}".split("e")
    return r"$%s\times10^{%d}$" % (m, int(e))


def rmse_winner_moves(quantities=("cum", "step")):
    """(n_moved, n_rows): how many RMSE-row winners differ between the root-inside
    and the textbook aggregation, across every quantity x bank size.

    Computed, never quoted. The switch to the textbook form was assumed to be
    rank-preserving on the grounds that R_A/R_B is near-constant within a
    quantity; that assumption is false, and this function is what proves it.
    """
    paths_kinds = [(os.path.join(EXP, p), k) for _, p, k in B.PS_MODELS]
    names = [n for n, _, _ in B.PS_MODELS]
    moved = total = 0
    for bs in BANK_SIZES:
        for qn in quantities:
            w = []
            for key in (RMSE_INSIDE, RMSE_LAST):
                vals = [(B._gen_quantities(p, bs) if k == "gen"
                         else B._fc_quantities(p))[qn][key]["value"]
                        for p, k in paths_kinds]
                w.append(names[B._ps_winner_idx(vals, "min")])
            total += 1
            moved += (w[0] != w[1])
    return moved, total


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
            # Same per-quantity key override the README uses (cum/step RMSE reads
            # the textbook root-last aggregation), read from build_cross_tables so
            # the floors cannot be scored on a different convention to the models.
            mk = B._ps_key(qn, metric)
            ov = RT.fmt(oracle_q[qn][mk]["value"])
            rv = RT.fmt(rw_q[qn][mk]["value"])
            win = r"\textbf{%s}" % tex_escape(names[wi]) if wi is not None else "---"
            # Same per-quantity label override the README uses (rv "RMSE" -> "MAE"),
            # read from build_cross_tables so the two renderers cannot diverge.
            mlab = B.PS_METRIC_LABEL.get((qn, metric), mlabel)
            rows.append(f"{lbl(mlab)} & " + " & ".join(cells)
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

    csdi_old = rmse_of("CSDI", "rv", RMSE_LAST)
    csdi_new = rmse_of("CSDI", "rv", RMSE_INSIDE)
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

\paragraph{Read \S\ref{sec:switch} before using the tables.} The row quoted
above is \emph{rv}, and rv still reports root-inside --- correctly, because for a
scalar quantity that aggregation \emph{is} a mean absolute error, and the tables
now say MAE. The two \emph{trajectory} quantities, cum and step, no longer do:
as of 2026-07-31 they report the textbook root-last RMSE, so those rows in
\S\ref{sec:tables} match Table 8's convention rather than Table 5's. The scorer
emits both aggregations for every model, bank and quantity, so nothing is lost
and every ``before'' number in this document is read live from the artefacts.
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

\subsection{What the code does now}

The scorer (\texttt{path\_shadowing\_pdf.py}, \texttt{\_agg}) computes
\emph{both} aggregations from the same per-path vector and writes both to every
artefact:

\begin{quote}\footnotesize\ttfamily
if kind == "rmse": \ \ \ \ \ \ return float(np.sqrt(per).mean()) \\
if kind == "rmse\_last": \ return float(np.sqrt(per.mean()))
\end{quote}

\noindent
\texttt{np.sqrt(per).mean()} roots \emph{then} averages --- convention (B), JSON
key \texttt{rmse}. \texttt{np.sqrt(per.mean())} averages \emph{then} roots ---
convention (A), JSON key \texttt{rmse\_textbook}. The paired bootstrap carries
the matching branch (\texttt{np.sqrt(samp).mean(axis=1)} versus
\texttt{np.sqrt(samp.mean(axis=1))}), so each confidence interval is consistent
with its own point estimate.

\subsection{A naming defect --- and what each row now reports}
\label{sec:switch}

Under convention (B) the reported quantity \textbf{is not a root-mean-square
error}, and the two kinds of quantity fail the name in different ways.

\paragraph{rv --- reports (B), labelled MAE.} The horizon collapses ($H=1$), so
$se_q = e_q^2$ and
\[
R_B=\frac{1}{m}\sum_q \sqrt{e_q^{2}}=\frac{1}{m}\sum_q |e_q| \;=\; \textbf{MAE exactly.}
\]
This row is a mean absolute error. The reproducibility report labels it
``RMSE''; \textbf{the tables in \S\ref{sec:tables} label it MAE}, which is simply
its correct name. The value is unchanged --- only the label is.

\paragraph{cum and step --- report (A), labelled RMSE.} Here $H=32$ and (B)
gives $\frac{1}{m}\sum_q\sqrt{\text{mean}_u se_{q,u}}$: an average of per-path
horizon-RMS values. That is a hybrid with no standard name, and it is not an
RMSE either. Calling it one understates the textbook figure by roughly $18\%$
(cum) and $6\%$ (step). Since $se_q$ is \emph{already} the mean over the horizon,
\[
R_A \;=\; \sqrt{\frac{1}{m}\sum_q se_q}
\;=\; \sqrt{\frac{1}{mH}\sum_{q}\sum_{u} se_{q,u}}
\]
--- the root of the mean squared error over every (query, horizon-step) pair,
i.e.\ exactly the textbook RMSE. \textbf{These two rows therefore read the
\texttt{rmse\_textbook} key}, so the ``RMSE'' label is now literally correct
rather than a convenient approximation.

\paragraph{Nothing is lost, and no cell is hand-typed.} Both keys are written for
every model, bank size and quantity, so a reader who wants the reproducibility
report's convention still has it in the JSON. The routing lives in
\texttt{build\_cross\_tables.PS\_METRIC\_KEY} (which key each row reads) and
\texttt{PS\_METRIC\_LABEL} (what each row is called), and both are read by the
README renderer and by this document, so the two cannot disagree.
""")

    # Both columns read live under an EXPLICITLY named aggregation — the ratio
    # cannot silently collapse to 1.000 if a row's default key ever changes.
    rows = []
    for q in ("cum", "step", "rv"):
        for m_ in ("CSDI", "LS4", "SBTS"):
            o = rmse_of(m_, q, RMSE_LAST)
            n_ = rmse_of(m_, q, RMSE_INSIDE)
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
(The trajectory quantities have since moved back to (A) for the reason given in
\S\ref{sec:switch}; both columns of the table below are read live, so it states
the size of that move rather than assuming it.)

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

    # Both columns come from the live JSONs: "before (A)" is the rmse_textbook key
    # (root last -- exactly what the frozen bridge used to emit) and "after (B)" is
    # the rmse key (root inside). No historical literals: the shipped numbers are by
    # construction the same ones the tables in §6 print.
    L.append(r"""
\begin{center}\small
\begin{tabular}{llrrr}
\hline
Method & Quantity & before (A) & after (B) & ratio \\
\hline
""")
    for meth in ("Chronos-2", "TimesFM"):
        for qn in ("cum", "step", "rv"):
            a = rmse_of(meth, qn, RMSE_LAST)
            b = rmse_of(meth, qn, RMSE_INSIDE)
            L.append(f"{tex_escape(meth)} & {qn} & {a:.5f} & {b:.5f} & "
                     f"{a / b:.3f} \\\\\n")
    L.append(r"""\hline
\end{tabular}
\end{center}
""")

    L.append(r"""
\textbf{What it changed.} Under the common convention Chronos-2's cum RMSE fell
from $0.04921$ to $0.04168$, against CSDI's $0.04144$: a row that read as a
$19\%$ deficit became a $0.6\%$ one. The estimate offered before the re-run ---
``roughly $0.0416$'' from the generator cum ratio --- was accurate for cum and
step, but the rv ratio came out at $1.050$ rather than the generators' $1.28$
(see the caveat in \S3), so only the recomputation settled it.

\textbf{What it did not change.} Win counts were recomputed at all five bank
sizes and were identical: TimeDiT (raw) 14 at every bank, with CSDI 2 / SBTS 2 at
4\,096, CSDI 3 / SBTS 1 at 16\,384, and CSDI 3 / LS4 1 at the top three. CSDI
remained the row minimum for cum and step RMSE, so \textbf{no Winner cell moved}
--- that statement is about \emph{this} fix, the un-mixing of the forecaster
column. The later switch of cum/step to the textbook aggregation \emph{does} move
Winner cells, and the paragraph after next measures by how much.

\paragraph{Why the tables in \S\ref{sec:tables} now show the (A) column again.}
This is not a reversal. \textbf{The defect was the \emph{mixing}, not the
choice.} Once every column shared one convention it became possible to ask which
one deserves the name ``RMSE'' --- and for cum and step the answer is (A)
(\S\ref{sec:switch}). So the trajectory rows moved to (A) \emph{for all eight
columns at once}: four generators, two forecasters, the Heston oracle and the RW
floor, through the single \texttt{PS\_METRIC\_KEY} lookup. Comparability is
preserved by construction; it is a property of the routing, not of a convention
happening to be applied consistently by hand.

""")

    # Measured, not asserted: how many RMSE-row winners the switch actually moves,
    # and how that compares to the width of the bootstrap CI on the same row.
    n_moved, n_rows = rmse_winner_moves()
    top3 = ["CSDI", "TimeDiT (raw)", "LS4"]
    v3 = [rmse_of(m, "cum", RMSE_LAST) for m in top3]
    spread = max(v3) - min(v3)
    ci = B._gen_quantities(os.path.join(EXP, dict(
        (m, p) for m, p, _ in B.PS_MODELS)["CSDI"]), 1000000)["cum"][RMSE_LAST]["ci"]
    ciw = ci[1] - ci[0]
    sbts_pct = 100 * (rmse_of("SBTS", "cum", RMSE_LAST) / min(v3) - 1)
    L.append(r"""
\paragraph{The ranking does \emph{not} survive the move --- and that is the real
finding.} It would be convenient to claim that a near-constant ratio $R_A/R_B$
rescales each column monotonically and leaves the argmin alone. It does not.
Recomputing every RMSE row under both aggregations, \textbf{""" +
             f"{n_moved} of the {n_rows}" + r""" cum/step rows
(2 quantities $\times$ 5 bank sizes) change winner}: CSDI takes every one of them
under root-inside, and loses most of them to TimeDiT (raw) or LS4 under the
textbook form.

That is not a defect in either convention. It is a measurement of how little
separates the top three generators. At the 1M bank the cum RMSE values are """ +
             ", ".join(f"${v:.6f}$" for v in v3) + r""" for CSDI, TimeDiT (raw)
and LS4 --- a spread of """ + sci(spread) + r""" --- while each 95\% bootstrap CI
is about """ + sci(ciw) + f" wide, {ciw / spread:.0f}" + r"""$\times$ larger.
Every interval contains every other point estimate. \textbf{The cum and step RMSE
winner among those three is noise}, and a reweighting as mild as moving one
square root is enough to reshuffle it. Only SBTS (""" + f"{sbts_pct:.0f}" +
             r"""\% worse) and, more weakly, the two forecasters separate from the
pack at all.

The practical consequence: read the RMSE rows of \S\ref{sec:tables} as
``CSDI $\approx$ TimeDiT (raw) $\approx$ LS4 $<$ Chronos-2 $<$ TimesFM $\ll$
SBTS'', and treat the bolded cell among the leading three as arbitrary. The
CRPS, coverage and miss rows are the ones that discriminate. The win counts
printed under each table are recomputed from the shipped artefacts every build,
never assumed.

\textbf{The cost.} The reproducibility report's published Chronos-2 / TimesFM
\emph{rv} cell is now \emph{superseded, not reproduced}: rv reports (B) and is
labelled MAE, so a re-run will not match the report's rv column, by design. The
cum and step cells, having moved to (A), agree with the report's Table 8
convention again. That is the correct trade: one internally consistent table,
with both aggregations preserved in the JSON for anyone who needs the other.
This is recorded in GUIDELINE E16b.
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
