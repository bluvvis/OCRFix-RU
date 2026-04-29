from dataclasses import dataclass

from .lm import CharNGramLM, WordNGramLM, build_word_inventory
from .tokenization import is_word_token, tokenize

AMBIGUOUS_MAP = {
    "0": "о",
    "1": "л",
    "3": "е",
    "@": "а",
    "c": "с",
    "p": "р",
    "k": "к",
}
REVERSE_AMBIGUOUS_MAP = {v: k for k, v in AMBIGUOUS_MAP.items()}


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i, ca in enumerate(a, start=1):
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def _join_tokens(tokens: list[str]) -> str:
    text = " ".join(tokens)
    for mark in [".", ",", "!", "?", ";", ":"]:
        text = text.replace(f" {mark}", mark)
    return text


@dataclass
class HybridCorrector:
    word_n: int = 3
    char_n: int = 5
    alpha: float = 0.7
    max_edit_distance: int = 2

    def __post_init__(self) -> None:
        self.word_lm = WordNGramLM(n=self.word_n)
        self.char_lm = CharNGramLM(n=self.char_n)
        self.word_freq: dict[str, int] = {}

    def fit(self, texts: list[str]) -> None:
        self.word_lm.fit(texts)
        self.char_lm.fit(texts)
        self.word_freq = build_word_inventory(texts)

    def _candidate_seeds(self, token: str) -> set[str]:
        seeds = {token}
        built = list(token)
        for i, ch in enumerate(built):
            if ch in AMBIGUOUS_MAP:
                changed = built.copy()
                changed[i] = AMBIGUOUS_MAP[ch]
                seeds.add("".join(changed))
        return seeds

    def _generate_candidates(self, token: str) -> list[str]:
        if token in self.word_freq:
            return [token]
        seeds = self._candidate_seeds(token)
        out = set(seeds)
        for vocab_word in self.word_freq:
            for seed in seeds:
                if abs(len(vocab_word) - len(seed)) > self.max_edit_distance:
                    continue
                if _edit_distance(vocab_word, seed) <= self.max_edit_distance:
                    out.add(vocab_word)
        if not out:
            out.add(token)
        return list(out)

    def _is_suspicious(self, token: str) -> bool:
        if token in self.word_freq:
            return False
        return any(ch in AMBIGUOUS_MAP for ch in token) or len(token) > 3

    def _ocr_channel_score(self, noisy_word: str, candidate: str) -> float:
        # Rewards candidates that explain common OCR confusions.
        score = 0.0
        for noisy_ch, cand_ch in zip(noisy_word, candidate):
            if noisy_ch == cand_ch:
                score += 0.25
            elif noisy_ch in AMBIGUOUS_MAP and AMBIGUOUS_MAP[noisy_ch] == cand_ch:
                score += 0.9
            elif cand_ch in REVERSE_AMBIGUOUS_MAP and REVERSE_AMBIGUOUS_MAP[cand_ch] == noisy_ch:
                score += 0.6
            else:
                score -= 0.2
        length_gap = abs(len(noisy_word) - len(candidate))
        score -= 0.35 * length_gap
        return score

    def correct_text(self, text: str) -> str:
        tokens = tokenize(text)
        corrected: list[str] = []
        for token in tokens:
            low = token.lower()
            if not is_word_token(token):
                corrected.append(token)
                continue
            if not self._is_suspicious(low):
                corrected.append(low)
                continue
            candidates = self._generate_candidates(low)
            context = [t for t in corrected if t.isalnum()]
            best_word = low
            best_score = float("-inf")
            for cand in candidates:
                word_score = self.word_lm.next_word_logprob(context, cand)
                char_score = self.char_lm.score_word(cand)
                ocr_score = self._ocr_channel_score(low, cand)
                penalty = _edit_distance(cand, low) * 0.4
                score = (
                    self.alpha * word_score
                    + (1 - self.alpha) * char_score
                    + 0.35 * ocr_score
                    - penalty
                )
                if score > best_score:
                    best_score = score
                    best_word = cand
            corrected.append(best_word)
        return _join_tokens(corrected)
