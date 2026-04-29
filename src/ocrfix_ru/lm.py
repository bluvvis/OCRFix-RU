import math
from collections import Counter, defaultdict

from .tokenization import tokenize


class WordNGramLM:
    def __init__(self, n: int = 3) -> None:
        self.n = n
        self.ngram_counts: Counter[tuple[str, ...]] = Counter()
        self.context_counts: Counter[tuple[str, ...]] = Counter()
        self.vocab: set[str] = set()
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        pad = ["<s>"] * (self.n - 1)
        for text in texts:
            tokens = [t.lower() for t in tokenize(text) if t.isalnum()]
            self.vocab.update(tokens)
            seq = pad + tokens + ["</s>"]
            for i in range(self.n - 1, len(seq)):
                ngram = tuple(seq[i - self.n + 1 : i + 1])
                context = ngram[:-1]
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1
        self._fitted = True

    def score_sentence(self, text: str) -> float:
        if not self._fitted:
            raise RuntimeError("WordNGramLM is not fitted")
        tokens = [t.lower() for t in tokenize(text) if t.isalnum()]
        pad = ["<s>"] * (self.n - 1)
        seq = pad + tokens + ["</s>"]
        vocab_size = max(len(self.vocab), 1)
        score = 0.0
        for i in range(self.n - 1, len(seq)):
            ngram = tuple(seq[i - self.n + 1 : i + 1])
            context = ngram[:-1]
            numerator = self.ngram_counts[ngram] + 1
            denominator = self.context_counts[context] + vocab_size
            score += math.log(numerator / denominator)
        return score

    def next_word_logprob(self, context_tokens: list[str], candidate: str) -> float:
        if not self._fitted:
            raise RuntimeError("WordNGramLM is not fitted")
        normalized_context = [t.lower() for t in context_tokens][-self.n + 1 :]
        while len(normalized_context) < self.n - 1:
            normalized_context.insert(0, "<s>")
        ngram = tuple(normalized_context + [candidate.lower()])
        context = tuple(normalized_context)
        vocab_size = max(len(self.vocab), 1)
        numerator = self.ngram_counts[ngram] + 1
        denominator = self.context_counts[context] + vocab_size
        return math.log(numerator / denominator)


class CharNGramLM:
    def __init__(self, n: int = 5) -> None:
        self.n = n
        self.ngram_counts: Counter[tuple[str, ...]] = Counter()
        self.context_counts: Counter[tuple[str, ...]] = Counter()
        self.charset: set[str] = set()
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        for text in texts:
            for raw_word in tokenize(text):
                if not raw_word.isalnum():
                    continue
                word = raw_word.lower()
                self.charset.update(word)
                seq = ["^"] * (self.n - 1) + list(word) + ["$"]
                for i in range(self.n - 1, len(seq)):
                    ngram = tuple(seq[i - self.n + 1 : i + 1])
                    context = ngram[:-1]
                    self.ngram_counts[ngram] += 1
                    self.context_counts[context] += 1
        self._fitted = True

    def score_word(self, word: str) -> float:
        if not self._fitted:
            raise RuntimeError("CharNGramLM is not fitted")
        normalized = word.lower()
        seq = ["^"] * (self.n - 1) + list(normalized) + ["$"]
        charset_size = max(len(self.charset), 1)
        score = 0.0
        for i in range(self.n - 1, len(seq)):
            ngram = tuple(seq[i - self.n + 1 : i + 1])
            context = ngram[:-1]
            numerator = self.ngram_counts[ngram] + 1
            denominator = self.context_counts[context] + charset_size
            score += math.log(numerator / denominator)
        return score


def build_word_inventory(texts: list[str]) -> dict[str, int]:
    freq: defaultdict[str, int] = defaultdict(int)
    for text in texts:
        for token in tokenize(text):
            if token.isalnum():
                freq[token.lower()] += 1
    return dict(freq)
