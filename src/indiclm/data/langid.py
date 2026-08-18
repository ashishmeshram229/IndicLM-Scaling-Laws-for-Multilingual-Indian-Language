"""Language identification for Indian-language text.

Design note: script != language. We first detect the dominant Unicode
script (Devanagari, Bengali, Tamil, ...), then disambiguate languages that
share a script (Hindi vs Marathi both use Devanagari) using small
function-word wordlists, since a full statistical LID model is out of
scope for this milestone. Code-mixing is detected structurally: if a
meaningful fraction of tokens are Latin-script while the rest are native
script, or if romanized text contains Indic-language function words, we
flag `is_code_mixed=True` rather than forcing a single label.

This is an honest, inspectable heuristic — not a production LID model
(e.g. fastText lid.176 or IndicLID). Swapping in a model-based backend
later only requires implementing `LanguageIdentifier`.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from indiclm.data.text_utils import split_words

# Unicode block ranges (start, end) -> script name
SCRIPT_RANGES: dict[str, tuple[int, int]] = {
    "devanagari": (0x0900, 0x097F),
    "bengali": (0x0980, 0x09FF),
    "gurmukhi": (0x0A00, 0x0A7F),
    "gujarati": (0x0A80, 0x0AFF),
    "tamil": (0x0B80, 0x0BFF),
    "telugu": (0x0C00, 0x0C7F),
    "kannada": (0x0C80, 0x0CFF),
    "malayalam": (0x0D00, 0x0D7F),
    "latin": (0x0041, 0x024F),
}

SCRIPT_TO_LANGUAGES: dict[str, list[str]] = {
    "devanagari": ["hin", "mar"],
    "bengali": ["ben"],
    "gurmukhi": ["pan"],
    "gujarati": ["guj"],
    "tamil": ["tam"],
    "telugu": ["tel"],
    "kannada": ["kan"],
    "malayalam": ["mal"],
    "latin": ["eng"],
}

# Small, hand-curated function-word wordlists used only to disambiguate
# languages that share a script (currently: Hindi vs Marathi, both
# Devanagari). Not exhaustive; documented as a heuristic, not ground truth.
DISAMBIGUATION_WORDLISTS: dict[str, dict[str, set[str]]] = {
    "devanagari": {
        "mar": {"आहे", "आणि", "मध्ये", "यांनी", "साठी", "होते", "झाला", "केले", "तो"},
        "hin": {"है", "और", "में", "के", "था", "किया", "हुआ", "को", "से", "कि"},
    }
}

# Common romanized Indic function words, used to flag likely code-mixed
# Latin-script text (e.g. Hinglish) rather than mislabel it as English.
ROMANIZED_INDIC_MARKERS: set[str] = {
    "hai", "nahi", "nahin", "kyun", "kyunki", "aur", "bahut", "accha", "kaise",
    "mujhe", "tumhe", "kal", "aaj", "mera", "tera", "hum", "aap", "karo",
    "karenge", "gaya", "gayi", "raha", "rahi", "jaana", "khaana", "ghar",
    "aahe", "tyacha", "amchya", "tyala", "mala", "tula", "kaay", "kasa",
    "iruku", "irundhu", "romba", "semma", "vanga", "sapadu", "pannunga",
}


class LanguageIdentifier(Protocol):
    def identify(self, text: str) -> "LangIdResult": ...


@dataclass
class LangIdResult:
    language: str
    confidence: float
    script: str
    is_code_mixed: bool


def _dominant_script(text: str) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter()
    for ch in text:
        if ch.isspace() or unicodedata.category(ch).startswith("P"):
            continue
        cp = ord(ch)
        for script, (lo, hi) in SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[script] += 1
                break
        else:
            counts["other"] += 1
    if not counts:
        return "unknown", counts
    return counts.most_common(1)[0][0], counts


class RuleBasedLanguageIdentifier:
    """Script-range + function-word heuristic language identifier."""

    def __init__(self, code_mixed_threshold: float = 0.15) -> None:
        self.code_mixed_threshold = code_mixed_threshold

    def identify(self, text: str) -> LangIdResult:
        script, counts = _dominant_script(text)
        total = sum(counts.values())
        if total == 0 or script == "unknown":
            return LangIdResult("unknown", 0.0, "unknown", False)

        dominant_frac = counts[script] / total
        latin_frac = counts.get("latin", 0) / total

        candidates = SCRIPT_TO_LANGUAGES.get(script, ["unknown"])
        language = candidates[0]
        confidence = dominant_frac

        if script in DISAMBIGUATION_WORDLISTS:
            words = split_words(text)
            wordlists = DISAMBIGUATION_WORDLISTS[script]
            scores = {lang: sum(1 for w in words if w in wl) for lang, wl in wordlists.items()}
            if any(scores.values()):
                language = max(scores, key=lambda k: scores[k])
                confidence = min(0.99, dominant_frac)
            else:
                # Ambiguous within the script family: default to the more
                # widely spoken language but lower confidence honestly.
                language = candidates[0]
                confidence = dominant_frac * 0.6

        is_code_mixed = False
        if script != "latin" and 0 < latin_frac:
            # Native-script-dominant text with a meaningful Latin fraction.
            is_code_mixed = latin_frac >= self.code_mixed_threshold
        elif script == "latin":
            words = [w.lower() for w in split_words(text) if re.fullmatch(r"[a-zA-Z']+", w)]
            if words:
                romanized_hits = sum(1 for w in words if w in ROMANIZED_INDIC_MARKERS)
                romanized_frac = romanized_hits / len(words)
                if romanized_frac >= self.code_mixed_threshold:
                    is_code_mixed = True
                    language = "eng"  # dominant script is Latin; flagged as mixed
                    confidence = max(0.3, 1.0 - romanized_frac)

        return LangIdResult(
            language=language,
            confidence=round(confidence, 3),
            script=script,
            is_code_mixed=is_code_mixed,
        )
