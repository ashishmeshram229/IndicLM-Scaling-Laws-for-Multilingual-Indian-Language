"""Unit tests for language identification, quality scoring, and the
combining-mark-safe word tokenizer they both depend on."""

from __future__ import annotations

from indiclm.data.langid import RuleBasedLanguageIdentifier
from indiclm.data.quality import RuleBasedQualityScorer
from indiclm.data.schema import Document
from indiclm.data.text_utils import split_words


def test_split_words_does_not_fragment_devanagari_clusters() -> None:
    words = split_words("सूर्य दररोज उगवतो")
    assert "सूर्य" in words
    assert "दररोज" in words


def test_langid_detects_devanagari_script() -> None:
    lid = RuleBasedLanguageIdentifier()
    result = lid.identify("सूर्य हर दिन पूर्व दिशा में उगता है।")
    assert result.script == "devanagari"
    assert result.language in ("hin", "mar")
    assert result.confidence > 0


def test_langid_detects_english() -> None:
    lid = RuleBasedLanguageIdentifier()
    result = lid.identify("The sun rises in the east every day.")
    assert result.language == "eng"
    assert result.script == "latin"
    assert not result.is_code_mixed


def test_langid_flags_hinglish_as_code_mixed() -> None:
    lid = RuleBasedLanguageIdentifier()
    result = lid.identify("Aaj mausam bahut accha hai, let's go for a walk.")
    assert result.is_code_mixed


def test_quality_scorer_penalizes_repeated_character_spam() -> None:
    scorer = RuleBasedQualityScorer()
    spammy = Document(text="!" * 50, source="test")
    spammy.language = "eng"
    _score, reasons = scorer.score(spammy)
    assert "high_repeated_char_ratio" in reasons


def test_quality_scorer_accepts_reasonable_sentence() -> None:
    scorer = RuleBasedQualityScorer()
    doc = Document(text="The library near our house has thousands of books.", source="test")
    doc.language = "eng"
    score, _reasons = scorer.score(doc)
    assert score >= 0.8
