from ocrfix_ru.corrector import HybridCorrector


def test_corrector_fixes_simple_ocr_digit_confusion() -> None:
    train_texts = [
        "мама мыла раму",
        "кот спит на окне",
        "данные для модели",
        "простой пример текста",
    ]
    corrector = HybridCorrector(word_n=2, char_n=4, alpha=0.7)
    corrector.fit(train_texts)
    corrected = corrector.correct_text("мама мыла раму и к0т спит")
    assert "кот" in corrected


def test_corrector_keeps_known_tokens_untouched() -> None:
    train_texts = ["мама мыла раму", "кот ест рыбу", "текст без ошибок"]
    corrector = HybridCorrector(word_n=2, char_n=4, alpha=0.7)
    corrector.fit(train_texts)
    source = "мама мыла раму"
    assert corrector.correct_text(source) == source
