import pytest

from src.h2_asr_wer import normalize_words, normalized_word_error


def test_normalize_words_is_case_and_punctuation_stable() -> None:
    assert normalize_words(" Hello—WORLD!  It's 2026. ") == ("hello", "world", "it's", "2026")
    assert normalize_words("we’re  READY___now") == ("we're", "ready", "now")


def test_normalized_word_error_counts_insertions_deletions_and_substitutions() -> None:
    result = normalized_word_error("one two three", "one too three extra")
    assert result.reference_words == ("one", "two", "three")
    assert result.hypothesis_words == ("one", "too", "three", "extra")
    assert result.errors == 2
    assert result.word_error_rate == pytest.approx(2.0 / 3.0)
    assert normalized_word_error("one two", "").word_error_rate == 1.0


def test_normalized_word_error_rejects_empty_original_transcript() -> None:
    with pytest.raises(ValueError, match="undefined"):
        normalized_word_error("?! ...", "anything")
