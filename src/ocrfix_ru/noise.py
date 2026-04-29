import random

OCR_SUBSTITUTIONS = {
    "о": "0",
    "О": "0",
    "а": "@",
    "е": "3",
    "л": "1",
    "з": "3",
    "с": "c",
    "р": "p",
    "к": "k",
}


def add_ocr_noise(text: str, error_rate: float = 0.1, seed: int | None = None) -> str:
    if error_rate <= 0:
        return text

    rng = random.Random(seed)
    chars = list(text)
    out: list[str] = []
    for ch in chars:
        if ch.isspace():
            out.append(ch)
            continue
        if rng.random() < error_rate:
            action = rng.choice(("replace", "drop", "keep"))
            if action == "replace" and ch in OCR_SUBSTITUTIONS:
                out.append(OCR_SUBSTITUTIONS[ch])
            elif action == "drop":
                continue
            else:
                out.append(ch)
        else:
            out.append(ch)
    return "".join(out)
