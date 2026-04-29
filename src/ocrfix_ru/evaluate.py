from dataclasses import dataclass
import json
from pathlib import Path
import random

from .corrector import HybridCorrector
from .noise import add_ocr_noise
from .tokenization import tokenize


@dataclass
class WordOnlyCorrector(HybridCorrector):
    def __init__(self, word_n: int = 3, char_n: int = 5) -> None:
        super().__init__(word_n=word_n, char_n=char_n, alpha=1.0)
        self.max_edit_distance = 1

    def _candidate_seeds(self, token: str) -> set[str]:
        # Vanilla baseline: no OCR-specific confusion substitutions.
        return {token}


def _token_accuracy(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens:
        return 0.0
    good = 0
    for i, token in enumerate(ref_tokens):
        if i < len(hyp_tokens) and hyp_tokens[i] == token:
            good += 1
    return good / len(ref_tokens)


def _edit_distance(seq_a: list[str], seq_b: list[str]) -> int:
    if seq_a == seq_b:
        return 0
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)
    dp = [[0] * (len(seq_b) + 1) for _ in range(len(seq_a) + 1)]
    for i in range(len(seq_a) + 1):
        dp[i][0] = i
    for j in range(len(seq_b) + 1):
        dp[0][j] = j
    for i, item_a in enumerate(seq_a, start=1):
        for j, item_b in enumerate(seq_b, start=1):
            cost = 0 if item_a == item_b else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def _word_error_rate(reference: str, hypothesis: str) -> float:
    ref_tokens = tokenize(reference.lower())
    hyp_tokens = tokenize(hypothesis.lower())
    ref_words = [t for t in ref_tokens if t.isalnum()]
    hyp_words = [t for t in hyp_tokens if t.isalnum()]
    if not ref_words:
        return 0.0
    return _edit_distance(ref_words, hyp_words) / len(ref_words)


def _char_error_rate(reference: str, hypothesis: str) -> float:
    ref_chars = list(reference.lower())
    hyp_chars = list(hypothesis.lower())
    if not ref_chars:
        return 0.0
    return _edit_distance(ref_chars, hyp_chars) / len(ref_chars)


def evaluate_models_on_synthetic_ocr(
    seed: int = 42,
    sample_size: int = 32,
    error_rate: float = 0.22,
    alpha: float = 0.7,
    base_sentences: list[str] | None = None,
) -> dict[str, float]:
    rng = random.Random(seed)
    data = base_sentences or [
        "мама мыла раму",
        "кот спит на окне",
        "простая модель исправляет ошибки",
        "данные важны для обучения",
        "текст содержит шум после оцифровки",
        "папа читает книгу вечером",
        "алгоритм ищет лучший вариант",
        "проект готовится к защите",
    ]
    train_texts = data * 12
    hybrid = HybridCorrector(word_n=2, char_n=4, alpha=alpha)
    word_only = WordOnlyCorrector(word_n=2, char_n=4)
    hybrid.fit(train_texts)
    word_only.fit(train_texts)

    hybrid_scores = []
    word_scores = []
    hybrid_wers = []
    word_wers = []
    hybrid_cers = []
    word_cers = []
    for _ in range(sample_size):
        ref = rng.choice(data)
        noisy = add_ocr_noise(ref, error_rate=error_rate, seed=rng.randint(1, 10_000))
        hybrid_hyp = hybrid.correct_text(noisy)
        word_hyp = word_only.correct_text(noisy)
        hybrid_scores.append(_token_accuracy(ref, hybrid_hyp))
        word_scores.append(_token_accuracy(ref, word_hyp))
        hybrid_wers.append(_word_error_rate(ref, hybrid_hyp))
        word_wers.append(_word_error_rate(ref, word_hyp))
        hybrid_cers.append(_char_error_rate(ref, hybrid_hyp))
        word_cers.append(_char_error_rate(ref, word_hyp))

    hybrid_acc = sum(hybrid_scores) / len(hybrid_scores)
    word_acc = sum(word_scores) / len(word_scores)
    hybrid_wer = sum(hybrid_wers) / len(hybrid_wers)
    word_wer = sum(word_wers) / len(word_wers)
    hybrid_cer = sum(hybrid_cers) / len(hybrid_cers)
    word_cer = sum(word_cers) / len(word_cers)
    return {
        "hybrid_accuracy": hybrid_acc,
        "word_only_accuracy": word_acc,
        "delta": hybrid_acc - word_acc,
        "hybrid_wer": hybrid_wer,
        "word_only_wer": word_wer,
        "hybrid_cer": hybrid_cer,
        "word_only_cer": word_cer,
    }


def run_ablation_study(
    seeds: list[int],
    error_rates: list[float],
    alphas: list[float],
    sample_size: int = 32,
    base_sentences: list[str] | None = None,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for seed in seeds:
        for error_rate in error_rates:
            for alpha in alphas:
                report = evaluate_models_on_synthetic_ocr(
                    seed=seed,
                    sample_size=sample_size,
                    error_rate=error_rate,
                    alpha=alpha,
                    base_sentences=base_sentences,
                )
                rows.append(
                    {
                        "seed": float(seed),
                        "error_rate": error_rate,
                        "alpha": alpha,
                        "hybrid_accuracy": report["hybrid_accuracy"],
                        "word_only_accuracy": report["word_only_accuracy"],
                        "delta_accuracy": report["delta"],
                        "hybrid_wer": report["hybrid_wer"],
                        "word_only_wer": report["word_only_wer"],
                        "hybrid_cer": report["hybrid_cer"],
                        "word_only_cer": report["word_only_cer"],
                    }
                )
    return rows


def save_report_json(
    output_path: Path, report: dict[str, float], ablation_rows: list[dict[str, float]]
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"main_report": report, "ablation": ablation_rows}
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
