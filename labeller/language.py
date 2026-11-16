"""Language detection and filtering. Local and free, so it can run before any Gemini call.

How it decides:
- Unicode script settles Chinese, Japanese, Korean and other single-language scripts. A statistical
  detector gets Chinese and Japanese wrong, so it is not used for them.
- py3langid separates languages that share a script (Latin, Cyrillic, Arabic, Devanagari). It is run
  on lowercased letters only, over windows of text, and restricted to a candidate set. Without that
  restriction it labels noisy OCR text of English forms as Nigerian Pidgin or Luxembourgish.
- A document is mixed when a second language holds at least `mixed_min_share` of the letters.
- Too little text, or low confidence, gives "unknown". What happens then is `unknown_policy`.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

from .config import LanguageConfig

LATIN_LANGS = ["en", "es", "fr", "de", "it", "pt", "nl", "sv", "da", "no", "fi", "pl", "cs", "ro", "hu", "tr",
               "id", "vi", "ca", "hr", "sk", "sl", "lt", "lv", "et", "ms", "tl", "sw", "af", "eu", "gl", "is",
               "sq", "uz"]
SCRIPT_GROUPS: dict[str, list[str]] = {
    "LATIN": LATIN_LANGS,
    "CYRILLIC": ["ru", "uk", "bg", "sr", "mk", "be", "kk"],
    "ARABIC": ["ar", "fa", "ur"],
    "DEVANAGARI": ["hi", "mr", "ne"],
}
# Scripts that map to one language.
SCRIPT_TO_LANG = {
    "BENGALI": "bn", "THAI": "th", "GREEK": "el", "HEBREW": "he", "HANGUL": "ko", "GEORGIAN": "ka",
    "ARMENIAN": "hy", "TAMIL": "ta", "TELUGU": "te", "GUJARATI": "gu", "KANNADA": "kn", "MALAYALAM": "ml",
    "GURMUKHI": "pa", "SINHALA": "si", "KHMER": "km", "LAO": "lo", "MYANMAR": "my", "ETHIOPIC": "am",
}
CJK_SCRIPTS = ("HAN", "KANA", "HANGUL")
SCRIPT_CONFIDENCE = 0.95
WINDOW_LETTERS = 150          # statistical detection runs on windows of about this many letters
WINDOW_FLOOR = 0.30           # a window the detector is less sure of than this is left unassigned
JAPANESE_KANA_SHARE = 0.10    # kana at this share of kana+han marks the text as Japanese

_MARKUP = re.compile(r"\[handwritten:\s*|\[signature\]|\[x\]|\[ \]|\]", re.IGNORECASE)
_WORD = re.compile(r"[^\W\d_]+")
_UNKNOWN_CODES = {"", "und", "unk", "unknown", "xx", "n/a", "none", "null", "mul", "mis", "zxx"}


@dataclass(frozen=True)
class LanguageResult:
    primary: str | None                       # ISO 639-1 code, or None when unknown
    confidence: float = 0.0
    shares: dict[str, float] = field(default_factory=dict)   # language -> share of the decided letters
    mixed: bool = False
    method: str = "local"                     # local | stage1 (Gemini's own language code)

    def to_dict(self) -> dict:
        return {"primary": self.primary, "confidence": round(self.confidence, 3),
                "shares": {k: round(v, 3) for k, v in sorted(self.shares.items(), key=lambda kv: -kv[1])},
                "mixed": self.mixed, "method": self.method}

    @classmethod
    def from_dict(cls, data: dict) -> "LanguageResult":
        return cls(primary=data.get("primary"), confidence=float(data.get("confidence", 0.0)),
                   shares=dict(data.get("shares") or {}), mixed=bool(data.get("mixed", False)),
                   method=str(data.get("method", "local")))


UNKNOWN = LanguageResult(primary=None)


def normalize_code(code: object) -> str | None:
    """Gemini's language answer as a bare ISO code ('en-US' -> 'en'), or None if it is not usable."""
    if not isinstance(code, str):
        return None
    base = re.split(r"[-_]", code.strip().lower())[0]
    if base in _UNKNOWN_CODES or not (2 <= len(base) <= 3 and base.isalpha()):
        return None
    return base


def from_code(code: object, method: str = "stage1") -> LanguageResult:
    normalized = normalize_code(code)
    if normalized is None:
        return LanguageResult(primary=None, method=method)
    return LanguageResult(primary=normalized, confidence=0.9, shares={normalized: 1.0}, method=method)


@lru_cache(maxsize=None)
def _script(char: str) -> str:
    first = unicodedata.name(char, "").split(" ", 1)[0]
    if first == "CJK":
        return "HAN"
    if first in ("HIRAGANA", "KATAKANA"):
        return "KANA"
    return first or "UNKNOWN"


@lru_cache(maxsize=None)
def _identifier(candidates: tuple[str, ...]):
    from py3langid.langid import MODEL_DIR, MODEL_FILE, LanguageIdentifier

    identifier = LanguageIdentifier.from_modelpath(MODEL_DIR / MODEL_FILE, norm_probs=True)
    known = {str(name) for name in identifier.nb_classes}
    identifier.set_languages([code for code in candidates if code in known])
    return identifier


def _script_runs(text: str):
    """Yields (script, letters) for runs of letters of one script. Digits and punctuation are dropped."""
    for word in _WORD.findall(text):
        start = 0
        for index in range(1, len(word) + 1):
            if index == len(word) or _script(word[index]) != _script(word[start]):
                yield _script(word[start]), word[start:index]
                start = index


def _windows(words: list[str]) -> list[str]:
    windows, current, size = [], [], 0
    for word in words:
        current.append(word)
        size += len(word) + 1
        if size >= WINDOW_LETTERS:
            windows.append(" ".join(current))
            current, size = [], 0
    if current:
        tail = " ".join(current)
        if windows and len(tail) < 40:
            windows[-1] += " " + tail
        else:
            windows.append(tail)
    return windows


def detect_language(text: str, config: LanguageConfig) -> LanguageResult:
    """Detects the language of a document's text. Returns UNKNOWN when there is too little to go on."""
    text = unicodedata.normalize("NFKC", _MARKUP.sub(" ", text)).lower()
    letters: dict[str, int] = defaultdict(int)
    words: dict[str, list[str]] = defaultdict(list)
    for script, run in _script_runs(text):
        if script == "LATIN" and len(run) < 2:
            continue  # single stray letters are noise in OCR text
        letters[script] += len(run)
        if script in SCRIPT_GROUPS:
            words[script].append(run)

    # Enough text? CJK and Hangul carry about three times the information per letter.
    weighted = sum(count * (3 if script in CJK_SCRIPTS else 1) for script, count in letters.items())
    if weighted < config.min_chars:
        return UNKNOWN

    weight: dict[str, float] = defaultdict(float)   # language -> letters assigned
    confidence_sum: dict[str, float] = defaultdict(float)

    def assign(language: str, count: float, confidence: float) -> None:
        if count <= 0:
            return
        weight[language] += count
        confidence_sum[language] += count * confidence

    han, kana = letters.get("HAN", 0), letters.get("KANA", 0)
    if han or kana:
        if kana >= JAPANESE_KANA_SHARE * (han + kana):
            assign("ja", han + kana, SCRIPT_CONFIDENCE)
        else:
            assign("zh", han, SCRIPT_CONFIDENCE)
            assign("ja", kana, SCRIPT_CONFIDENCE)
    for script, language in SCRIPT_TO_LANG.items():
        if letters.get(script):
            assign(language, letters[script], SCRIPT_CONFIDENCE)

    for script, candidates in SCRIPT_GROUPS.items():
        script_words = words.get(script)
        if not script_words:
            continue
        # Always consider the allowed languages of this script alongside the usual candidates.
        extra = [code for code in config.allowed if code in candidates]
        identifier = _identifier(tuple(dict.fromkeys(candidates + extra)))
        for window in _windows(script_words):
            language, probability = identifier.classify(window)
            if probability >= WINDOW_FLOOR:
                assign(language, len(window.replace(" ", "")), float(probability))

    total = sum(weight.values())
    if total <= 0:
        return UNKNOWN
    shares = {language: count / total for language, count in weight.items()}
    primary = max(shares, key=shares.get)
    confidence = confidence_sum[primary] / weight[primary]
    if confidence < config.min_confidence:
        return LanguageResult(primary=None, confidence=confidence, shares=shares)
    mixed = any(language != primary and share >= config.mixed_min_share for language, share in shares.items())
    return LanguageResult(primary=primary, confidence=confidence, shares=shares, mixed=mixed)


def language_reason(result: LanguageResult, config: LanguageConfig) -> str | None:
    """The exclusion reason for this result under the policy, or None when the document passes."""
    if not config.enabled:
        return None
    if result.primary is None:
        return "language_unknown" if config.unknown_policy == "reject" else None
    present = sorted(code for code, share in result.shares.items() if share >= config.mixed_min_share)
    if result.mixed:
        if config.mixed_language_policy == "reject_mixed":
            return f"language_mixed:{'+'.join(present)}"
        if config.mixed_language_policy == "accept_if_all_allowed":
            disallowed = [code for code in present if code not in config.allowed]
            if disallowed:
                return f"language_mixed:{'+'.join(present)}"
    if result.primary not in config.allowed:
        return f"language:{result.primary}"
    return None
