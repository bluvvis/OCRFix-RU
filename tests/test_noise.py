from ocrfix_ru.noise import add_ocr_noise


def test_add_ocr_noise_changes_text_with_nonzero_rate() -> None:
    source = "мама мыла раму"
    noisy = add_ocr_noise(source, error_rate=0.35, seed=7)
    assert noisy != source
    assert len(noisy) >= len(source) - 4


def test_add_ocr_noise_deterministic_with_seed() -> None:
    source = "данные для теста"
    first = add_ocr_noise(source, error_rate=0.25, seed=11)
    second = add_ocr_noise(source, error_rate=0.25, seed=11)
    assert first == second
