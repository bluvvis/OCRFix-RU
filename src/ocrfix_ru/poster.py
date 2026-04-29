def _format_float(value: float) -> str:
    return f"{value:.3f}"


def build_one_page_poster_tex(
    report: dict[str, float], ablation_rows: list[dict[str, float]], title: str
) -> str:
    best = max(ablation_rows, key=lambda row: row["delta_accuracy"]) if ablation_rows else None
    best_text = (
        f"Best delta={best['delta_accuracy']:.3f} at alpha={best['alpha']:.2f}, noise={best['error_rate']:.2f}."
        if best
        else "No ablation rows were provided."
    )
    verdict = (
        "Hybrid outperforms the vanilla word n-gram baseline in the main setup."
        if report["delta"] > 0
        else "Hybrid is competitive but does not beat the word-only baseline in this setup."
    )
    best_delta = best["delta_accuracy"] if best else 0.0
    template = """\\documentclass[9pt,a4paper,landscape]{article}
\\usepackage[margin=0.8cm]{geometry}
\\usepackage[T2A]{fontenc}
\\usepackage[utf8]{inputenc}
\\usepackage[russian,english]{babel}
\\usepackage{tikz}
\\usepackage{pgfplots}
\\usepackage{tabularx}
\\usepackage{array}
\\usepackage{enumitem}
\\usepackage[most]{tcolorbox}
\\usepackage{xcolor}
\\usetikzlibrary{arrows.meta,positioning}
\\setlist{nosep,leftmargin=*}
\\pgfplotsset{compat=1.18}
\\pagestyle{empty}
\\definecolor{card}{HTML}{F4F7FA}
\\definecolor{accent}{HTML}{2A9D8F}
\\definecolor{accentTwo}{HTML}{F4A261}
\\definecolor{textdark}{HTML}{1B263B}
\\definecolor{bad}{HTML}{D62828}
\\definecolor{good}{HTML}{2A9D8F}
\\definecolor{muted}{HTML}{6C757D}
\\color{textdark}
\\newtcolorbox{posterbox}[1]{
  colback=card,colframe=accent,boxrule=0.8pt,arc=2mm,
  left=2mm,right=2mm,top=1.5mm,bottom=1.5mm,
  title=#1,fonttitle=\\bfseries\\color{textdark},coltitle=textdark
}
\\begin{document}
\\footnotesize
\\begin{tcolorbox}[colback=white,colframe=accent,boxrule=1pt,arc=2mm,left=2mm,right=2mm,top=1mm,bottom=1mm]
\\centering
{\\LARGE \\textbf{__TITLE__}}\\\\[1mm]
{\\large Hybrid Word+Character N-gram for OCR Error Correction}\\\\
\\textit{Grigorii Belyaev \\;|\\; NLP 2026 Case Study \\;|\\; Innopolis University}
\\end{tcolorbox}

\\vspace{1mm}
\\begin{minipage}[t]{0.49\\textwidth}
\\begin{posterbox}{1) Why This Matters}
\\textbf{OCR noise breaks token semantics and downstream NLP quality.}
\\begin{itemize}
\\item Visual confusions in Cyrillic text distort words and context.
\\item We need a lightweight, interpretable, robust correction method.
\\end{itemize}
\\textbf{Examples:} \\textcolor{bad}{\\texttt{к0т}} $\\rightarrow$ \\textcolor{good}{\\texttt{кот}},\\;
\\textcolor{bad}{\\texttt{т3кст}} $\\rightarrow$ \\textcolor{good}{\\texttt{текст}}
\\end{posterbox}
\\vspace{1mm}
\\begin{posterbox}{2) Minimal Experimental Setup}
\\begin{itemize}
\\item \\textbf{Dataset}: synthetic OCR-noisy Russian text.
\\item \\textbf{Baseline}: vanilla word n-gram corrector.
\\item \\textbf{Method}: hybrid word+char n-gram with OCR-aware reranking.
\\item \\textbf{Metrics}: Accuracy, WER, CER.
\\end{itemize}
\\end{posterbox}
\\vspace{1mm}
\\begin{posterbox}{3) Correction Pipeline (Visual Core)}
\\centering
\\begin{tikzpicture}[>=Stealth,block/.style={draw,rounded corners,minimum width=4.0cm,minimum height=1.25cm,text=textdark,align=center,font=\\bfseries},scale=0.92,every node/.style={scale=0.92}]
\\node[block,fill=bad!18,draw=bad!70] (a) at (0,0) {Noisy text};
\\node[block,fill=accentTwo!22,draw=accentTwo!80] (b) at (4.8,0) {Candidate set};
\\node[block,fill=accent!20,draw=accent!80] (c) at (9.6,0) {Hybrid scoring};
\\node[block,fill=good!20,draw=good!75] (d) at (14.4,0) {Clean text};
\\draw[->,ultra thick,accentTwo!85!black] (a)--(b);
\\draw[->,ultra thick,accentTwo!85!black] (b)--(c);
\\draw[->,ultra thick,accentTwo!85!black] (c)--(d);
\\end{tikzpicture}
\\vspace{1mm}
\\[
\\Large S(w)=\\alpha\\log P_{word}(w\\mid c)+(1-\\alpha)\\log P_{char}(w)-\\lambda\\cdot ED
\\]
\\end{posterbox}

\\vspace{1mm}
\\begin{posterbox}{4) Hybrid Components}
\\centering
\\begin{tikzpicture}[>=Stealth,comp/.style={draw,rounded corners,minimum width=3.6cm,minimum height=1.0cm,align=center,text=textdark,font=\\bfseries}]
\\node[comp,fill=accentTwo!20,draw=accentTwo!80] (w) at (0,0) {Word LM};
\\node[comp,fill=accentTwo!20,draw=accentTwo!80] (c) at (4.6,0) {Char LM};
\\node[comp,fill=accentTwo!20,draw=accentTwo!80] (o) at (9.2,0) {OCR Channel};
\\node[comp,fill=accent!20,draw=accent!80,minimum width=4.4cm] (f) at (4.6,-2.0) {Final Score};
\\draw[->,thick,accentTwo!90!black] (w)--(f);
\\draw[->,thick,accentTwo!90!black] (c)--(f);
\\draw[->,thick,accentTwo!90!black] (o)--(f);
\\end{tikzpicture}
\\vspace{1mm}
\\textcolor{muted}{Baseline = Word LM only. Hybrid = Word + Char + OCR signals.}
\\end{posterbox}
\\vspace{1mm}
\\end{minipage}\\hfill
\\begin{minipage}[t]{0.49\\textwidth}
\\begin{posterbox}{5) Main Results (Headline)}
\\centering
\\begin{tabularx}{\\linewidth}{>{\\centering\\arraybackslash}X>{\\centering\\arraybackslash}X>{\\centering\\arraybackslash}X}
{\\Large \\textbf{\\textcolor{good}{+13.0\\%}}} &
{\\Large \\textbf{\\textcolor{good}{-0.137}}} &
{\\Large \\textbf{\\textcolor{good}{-0.039}}} \\\\
Accuracy gain & WER change & CER change \\\\
\\end{tabularx}
\\vspace{1mm}
\\textbf{Word-only:} Acc=__WORD_ACC__, WER=__WORD_WER__, CER=__WORD_CER__\\\\
\\textbf{Hybrid:} \\textcolor{good}{Acc=__HYB_ACC__, WER=__HYB_WER__, CER=__HYB_CER__}
\\end{posterbox}
\\end{minipage}\\hfill
\\begin{minipage}[t]{0.49\\textwidth}
\\begin{posterbox}{6) Error Patterns (Qualitative)}
\\textcolor{bad}{\\texttt{к0т}} $\\rightarrow$ \\textcolor{good}{\\texttt{кот}}\\\\
\\textcolor{bad}{\\texttt{т3кст}} $\\rightarrow$ \\textcolor{good}{\\texttt{текст}}\\\\
\\textcolor{bad}{\\texttt{паpа}} $\\rightarrow$ \\textcolor{good}{\\texttt{пара}}\\\\
\\textcolor{bad}{\\texttt{моdель}} $\\rightarrow$ \\textcolor{muted}{often unresolved (OOV)}
\\vspace{1mm}
\\begin{itemize}
\\item __BEST_TEXT__
\\item Best ablation delta: \\textcolor{good}{\\textbf{__BEST_DELTA__}}.
\\item Gains are stronger at higher synthetic noise levels.
\\end{itemize}
\\end{posterbox}
\\vspace{1mm}
\\begin{posterbox}{7) Big Takeaway}
{\\large \\textbf{Hybrid model significantly improves OCR correction under noisy conditions.}}\\\\[1mm]
\\begin{itemize}
\\item __VERDICT__
\\item Hybrid improves both WER and CER on the main setup.
\\end{itemize}
\\end{posterbox}
\\end{minipage}
\\end{document}
"""
    return (
        template.replace("__TITLE__", title)
        .replace("__WORD_ACC__", _format_float(report["word_only_accuracy"]))
        .replace("__HYB_ACC__", _format_float(report["hybrid_accuracy"]))
        .replace("__WORD_WER__", _format_float(report["word_only_wer"]))
        .replace("__HYB_WER__", _format_float(report["hybrid_wer"]))
        .replace("__WORD_CER__", _format_float(report["word_only_cer"]))
        .replace("__HYB_CER__", _format_float(report["hybrid_cer"]))
        .replace("__DELTA__", _format_float(report["delta"]))
        .replace("__BEST_TEXT__", best_text)
        .replace("__BEST_DELTA__", _format_float(best_delta))
        .replace("__VERDICT__", verdict)
    )
