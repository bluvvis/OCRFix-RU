from pathlib import Path
import shutil
import subprocess
import sys

# Ensure local src package is used when running script directly.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocrfix_ru.dataset import download_text_corpus, load_corpus_sentences
from ocrfix_ru.evaluate import evaluate_models_on_synthetic_ocr, run_ablation_study, save_report_json
from ocrfix_ru.poster import build_one_page_poster_tex

DEFAULT_CORPUS_URL = "https://www.gutenberg.org/cache/epub/2554/pg2554.txt"


def main() -> None:
    corpus_path = Path("data/raw/russian_corpus.txt")
    if not corpus_path.exists():
        download_text_corpus(DEFAULT_CORPUS_URL, corpus_path)

    sentences = load_corpus_sentences(corpus_path, min_words=3, max_sentences=1800)
    # Main benchmark is a controlled synthetic OCR setup used in tests.
    # This keeps the headline result comparable and stable for the case study.
    report = evaluate_models_on_synthetic_ocr(
        seed=42,
        sample_size=80,
        error_rate=0.30,
        alpha=0.90,
    )
    rows = run_ablation_study(
        seeds=[7, 11, 19],
        error_rates=[0.1, 0.2, 0.3],
        alphas=[0.0, 0.3, 0.7, 1.0],
        sample_size=24,
        base_sentences=sentences,
    )
    save_report_json(Path("artifacts/report.json"), report, rows)
    tex = build_one_page_poster_tex(report=report, ablation_rows=rows, title="OCRFix-RU")
    poster_path = Path("poster/poster.tex")
    poster_path.parent.mkdir(parents=True, exist_ok=True)
    poster_path.write_text(tex, encoding="utf-8")
    print("Built artifacts/report.json and poster/poster.tex")

    pdflatex = shutil.which("pdflatex")
    if pdflatex:
        # Run twice to resolve references and stabilize layout.
        for _ in range(2):
            subprocess.run(
                [pdflatex, "-interaction=nonstopmode", "poster.tex"],
                cwd=poster_path.parent,
                check=False,
                capture_output=True,
                text=True,
            )
        pdf_path = poster_path.with_suffix(".pdf")
        if pdf_path.exists():
            print("Built poster/poster.pdf")
        else:
            print("pdflatex is available but poster.pdf was not created")
    else:
        print("pdflatex not found; poster PDF not generated (TeX source is ready)")


if __name__ == "__main__":
    main()
