import re

TOKEN_PATTERN = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+|[^\w\s]", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def is_word_token(token: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-zА-Яа-яЁё0-9]+", token))
