"""Unit tests for the pure text/JSON helpers in synthesis.py."""

import synthesis


def test_safe_json_extract_plain():
    assert synthesis._safe_json_extract('{"answer": "hi"}') == {"answer": "hi"}


def test_safe_json_extract_embedded_in_noise():
    text = 'Here you go:\n{"answer": "hi", "confidence": 0.5}\nthanks'
    assert synthesis._safe_json_extract(text) == {"answer": "hi", "confidence": 0.5}


def test_safe_json_extract_none_when_no_json():
    assert synthesis._safe_json_extract("no json here") is None
    assert synthesis._safe_json_extract("{not valid json}") is None


def test_extract_sentences_filters_short_and_lowercase():
    text = "short. This is a full sentence that is long enough. also lowercase start here."
    sents = synthesis._extract_sentences(text, max_sentences=5)
    assert sents == ["This is a full sentence that is long enough."]


def test_extract_sentences_respects_max():
    text = "This is sentence number one here. This is sentence number two here. This is three here."
    assert len(synthesis._extract_sentences(text, max_sentences=2)) == 2


def test_normalize_sentence_key_prefix_and_strip():
    key = synthesis._normalize_sentence_key("Hello, WORLD! This is: a test.")
    assert key == "hello world this is a test"


def test_looks_technical_detects_json_markers():
    assert synthesis._looks_technical('{"id": 1}') is True
    assert synthesis._looks_technical("provenance offsets") is True
    assert synthesis._looks_technical("A perfectly normal sentence about courage.") is False


def test_clean_answer_text_drops_structured_lines():
    raw = 'Real answer line.\n{"provenance": []}\n[array line]\nMore real text.'
    cleaned = synthesis._clean_answer_text(raw)
    assert "provenance" not in cleaned
    assert "Real answer line." in cleaned
    assert "More real text." in cleaned


def test_is_natural_answer():
    assert (
        synthesis._is_natural_answer(
            "Seneca teaches that fear is often worse than the thing feared, so we should reason through it."
        )
        is True
    )
    assert synthesis._is_natural_answer("too short") is False
    assert (
        synthesis._is_natural_answer('{"answer": "this is json and not natural at all really"}')
        is False
    )


def test_clamp_natural_answer_dedupes_and_bounds():
    text = (
        "Fear is a projection of the mind about the future. "
        "Fear is a projection of the mind about the future. "
        "Reason lets us examine whether the fear is justified today."
    )
    out = synthesis._clamp_natural_answer(text, min_sentences=1, max_sentences=4)
    # Near-duplicate collapsed to a single occurrence.
    assert out.count("Fear is a projection of the mind about the future.") == 1
    assert "Reason lets us examine" in out
