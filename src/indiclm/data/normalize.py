"""Text normalization: Unicode normalization and whitespace cleanup.

Kept deliberately simple and lossless with respect to script content — we
do not transliterate or alter Indic characters, only normalize Unicode
composition and collapse whitespace anomalies.
"""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"[ \t ​]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()
