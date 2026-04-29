import argparse
from pathlib import Path

from .dataset import load_corpus_sentences
from .corrector import HybridCorrector
from .evaluate import evaluate_models_on_synthetic_ocr, run_ablation_study, save_report_json
from .poster import build_one_page_poster_tex


def build_demo_corrector() -> HybridCorrector:
    train_texts = [
        "мама мыла раму",
        "кот спит на окне",
        "данные важны для обучения",
        "текст содержит шум после оцифровки",
        "алгоритм исправляет ошибки",
        "проект готовится к защите",
    ] * 16
    corrector = HybridCorrector(word_n=2, char_n=4, alpha=0.7)
    corrector.fit(train_texts)
    return corrector


def main() -> None:
    parser = argparse.ArgumentParser(description="OCRFix-RU command line demo")
    parser.add_argument("--text", type=str, help="Noisy text to correct")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run synthetic evaluation against word-only baseline",
    )
    parser.add_argument(
        "--corpus-path",
        type=str,
        default=None,
        help="Optional text corpus path used for training/evaluation sentences",
    )
    parser.add_argument(
        "--build-artifacts",
        action="store_true",
        help="Build case-study artifacts: JSON report and one-page LaTeX poster",
    )
    args = parser.parse_args()

    base_sentences = None
    if args.corpus_path:
        corpus_path = Path(args.corpus_path)
        base_sentences = load_corpus_sentences(corpus_path, min_words=3, max_sentences=1200)

    if args.evaluate:
        report = evaluate_models_on_synthetic_ocr(base_sentences=base_sentences)
        print("Synthetic OCR benchmark")
        print(f"word-only accuracy: {report['word_only_accuracy']:.3f}")
        print(f"hybrid accuracy:    {report['hybrid_accuracy']:.3f}")
        print(f"delta:              {report['delta']:+.3f}")
        print(f"word-only WER:      {report['word_only_wer']:.3f}")
        print(f"hybrid WER:         {report['hybrid_wer']:.3f}")
        print(f"word-only CER:      {report['word_only_cer']:.3f}")
        print(f"hybrid CER:         {report['hybrid_cer']:.3f}")
        if args.build_artifacts:
            rows = run_ablation_study(
                seeds=[11, 13],
                error_rates=[0.15, 0.25, 0.35],
                alphas=[0.0, 0.3, 0.7, 1.0],
                sample_size=28,
                base_sentences=base_sentences,
            )
            save_report_json(Path("artifacts/report.json"), report, rows)
            tex = build_one_page_poster_tex(report=report, ablation_rows=rows, title="OCRFix-RU")
            poster_path = Path("poster/poster.tex")
            poster_path.parent.mkdir(parents=True, exist_ok=True)
            poster_path.write_text(tex, encoding="utf-8")
            print("Artifacts saved: artifacts/report.json, poster/poster.tex")
        return

    if not args.text:
        raise SystemExit("Pass --text '...noisy sentence...' or use --evaluate")

    corrector = build_demo_corrector()
    corrected = corrector.correct_text(args.text)
    print(corrected)


if __name__ == "__main__":
    main()
