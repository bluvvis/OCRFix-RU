from ocrfix_ru.lm import CharNGramLM, WordNGramLM


def test_word_ngram_prefers_seen_phrase() -> None:
    lm = WordNGramLM(n=2)
    corpus = [
        "мама мыла раму",
        "мама любит кофе",
        "папа мыл окно",
    ]
    lm.fit(corpus)
    seen_score = lm.score_sentence("мама мыла раму")
    unseen_score = lm.score_sentence("мама раму мыла")
    assert seen_score > unseen_score


def test_char_ngram_prefers_plausible_word() -> None:
    lm = CharNGramLM(n=4)
    lm.fit(["коррекция ошибок", "ошибки в тексте", "текстовый анализ"])
    assert lm.score_word("ошибки") > lm.score_word("ошбик1")
