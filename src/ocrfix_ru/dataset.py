from pathlib import Path
import re
from urllib.request import urlopen


def load_corpus_sentences(
    path: Path, min_words: int = 3, max_sentences: int = 2500
) -> list[str]:
    text = path.read_text(encoding="utf-8")
    raw_sentences = re.split(r"[.!?]+", text)
    out: list[str] = []
    for sent in raw_sentences:
        cleaned = " ".join(sent.strip().split())
        if not cleaned:
            continue
        if len(cleaned.split()) < min_words:
            continue
        out.append(cleaned.lower())
        if len(out) >= max_sentences:
            break
    return out


def download_text_corpus(url: str, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=30) as response:  # nosec: URL is controlled by user code
        content = response.read().decode("utf-8", errors="ignore")
    target_path.write_text(content, encoding="utf-8")
    return target_path
