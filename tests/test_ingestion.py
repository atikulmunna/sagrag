"""Unit tests for the pure chunking/flattening helpers in ingestion.py."""

import ingestion


def test_chunk_text_offsets_and_count():
    chunks = ingestion.chunk_text("a b c d e f", chunk_words=3, overlap=1)
    # step = chunk_words - overlap = 2 -> starts at word 0, 2, 4 => 3 chunks.
    assert len(chunks) == 3
    first_chunk, start, end = chunks[0]
    assert first_chunk == "a b c"
    assert (start, end) == (0, 5)


def test_flatten_json_nested():
    lines = ingestion._flatten_json({"a": {"b": 1}, "c": [2, 3]})
    assert "a.b: 1" in lines
    assert "c.0: 2" in lines
    assert "c.1: 3" in lines


def test_flatten_csv_pairs_header():
    lines = ingestion._flatten_csv("h1,h2\n1,2\n3,4")
    assert lines == ["h1: 1 | h2: 2", "h1: 3 | h2: 4"]


def test_flatten_csv_empty():
    assert ingestion._flatten_csv("") == []
