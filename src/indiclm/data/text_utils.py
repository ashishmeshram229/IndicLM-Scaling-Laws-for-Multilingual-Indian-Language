"""Shared word-tokenization helper for scripts with combining marks.

Design note: Python's `re` `\\w` excludes Unicode combining marks (category
Mn/Mc), so naively splitting Devanagari/Tamil/etc. text on `\\w+` breaks
every consonant+vowel-sign cluster into fragments (e.g. "सूर्य" ->
["स", "र", "य"]), which silently corrupts language-ID wordlist matching,
quality unique-word-ratio, and n-gram overlap for every Indic script.

Instead we tokenize by splitting on whitespace and a fixed set of ASCII
and Indic punctuation, keeping every other run of characters (base
consonants + attached matras/virama) together as one "word". This is an
approximation, not a linguistic tokenizer, but it is script-agnostic and
does not fragment combining-mark clusters.
"""

from __future__ import annotations

import re

_PUNCTUATION = r"""!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~।॥…—–""" + '"' + '"'
_WORD_SPLIT_RE = re.compile(rf"[^\s{re.escape(_PUNCTUATION)}]+", re.UNICODE)


def split_words(text: str) -> list[str]:
    return _WORD_SPLIT_RE.findall(text)
