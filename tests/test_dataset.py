from pathlib import Path

from ocrfix_ru.dataset import load_corpus_sentences


def test_load_corpus_sentences_from_local_text(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "мама мыла раму.\nкот спит на окне!\nданные важны для обучения.\n",
        encoding="utf-8",
    )
    sentences = load_corpus_sentences(corpus, min_words=2, max_sentences=10)
    assert len(sentences) >= 3
    assert any("кот спит" in sent for sent in sentences)
