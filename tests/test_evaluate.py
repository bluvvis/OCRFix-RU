from pathlib import Path

from ocrfix_ru.evaluate import (
    evaluate_models_on_synthetic_ocr,
    run_ablation_study,
    save_report_json,
)
from ocrfix_ru.poster import build_one_page_poster_tex


def test_hybrid_not_worse_than_word_only_on_synthetic_data() -> None:
    report = evaluate_models_on_synthetic_ocr(seed=13, sample_size=80, error_rate=0.25)
    assert report["hybrid_accuracy"] > report["word_only_accuracy"]
    assert report["hybrid_wer"] < report["word_only_wer"]
    assert report["hybrid_cer"] < report["word_only_cer"]


def test_ablation_study_contains_required_columns() -> None:
    rows = run_ablation_study(
        seeds=[7],
        error_rates=[0.15, 0.25],
        alphas=[0.0, 0.7, 1.0],
        sample_size=12,
    )
    assert rows, "Expected non-empty ablation results"
    required = {
        "seed",
        "error_rate",
        "alpha",
        "hybrid_accuracy",
        "word_only_accuracy",
        "delta_accuracy",
        "hybrid_wer",
        "word_only_wer",
    }
    assert required.issubset(rows[0].keys())


def test_save_report_and_generate_poster_tex(tmp_path: Path) -> None:
    report = evaluate_models_on_synthetic_ocr(seed=5, sample_size=8, error_rate=0.2)
    rows = run_ablation_study(
        seeds=[5],
        error_rates=[0.2],
        alphas=[0.3, 0.7, 1.0],
        sample_size=8,
    )
    report_path = tmp_path / "report.json"
    save_report_json(report_path, report, rows)
    assert report_path.exists()

    tex = build_one_page_poster_tex(report=report, ablation_rows=rows, title="OCRFix-RU")
    assert r"\documentclass" in tex
    assert "OCRFix-RU" in tex
    assert "Experimental Setup" in tex
