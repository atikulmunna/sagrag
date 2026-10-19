"""Unit tests for the pure chunking/flattening helpers in ingestion.py."""

import json
import random

import ingestion


def test_chunk_text_offsets_and_count():
    chunks = ingestion.chunk_text("a b c d e f", chunk_words=3, overlap=1)
    # step = chunk_words - overlap = 2 -> starts at word 0, 2, 4 => 3 chunks.
    assert len(chunks) == 3
    first_chunk, start, end = chunks[0]
    assert first_chunk == "a b c"
    assert (start, end) == (0, 5)


def _random_text(rng, n_words):
    seps = [" ", "  ", "\n", "\n\n", "\t", " \n "]
    vocab = [f"w{i}" for i in range(37)]  # repeated tokens, like real prose
    parts = []
    for _ in range(n_words):
        parts.append(rng.choice(vocab))
        parts.append(rng.choice(seps))
    return rng.choice(["", "  ", "\n"]) + "".join(parts)


def test_chunk_offsets_slice_back_to_the_chunk():
    # Regression: offsets drifted (whitespace was normalized and each search
    # started after the previous chunk's end despite the overlap), so cited
    # spans pointed at the wrong text.
    rng = random.Random(7)
    for n_words in (1, 49, 200, 201, 523, 1000):
        text = _random_text(rng, n_words)
        chunks = ingestion.chunk_text(text, chunk_words=200, overlap=50)
        for chunk, start, end in chunks:
            assert text[start:end] == chunk
        # Every word is covered by some chunk.
        covered = set()
        for _, start, end in chunks:
            covered.update(text[start:end].split())
        assert covered == set(text.split())


def test_chunk_text_keeps_newlines_for_structured_lines():
    text = "Name: Seneca\nRole: Philosopher\nEra: Roman"
    ((chunk, _, _),) = ingestion.chunk_text(text)
    assert chunk.splitlines() == ["Name: Seneca", "Role: Philosopher", "Era: Roman"]


def test_chunk_text_has_no_redundant_tail_chunk():
    text = " ".join(f"w{i}" for i in range(500))
    starts = [
        chunk.split()[0] for chunk, _, _ in ingestion.chunk_text(text, chunk_words=200, overlap=50)
    ]
    # 0-199, 150-349, 300-499; a fourth 450-499 window would be fully contained.
    assert starts == ["w0", "w150", "w300"]


def test_chunk_text_empty_and_validates_overlap():
    assert ingestion.chunk_text("") == []
    try:
        ingestion.chunk_text("a b", chunk_words=2, overlap=2)
    except ValueError:
        pass
    else:
        raise AssertionError("overlap >= chunk_words must be rejected")


def test_point_ids_are_unique_per_namespace():
    a = ingestion.point_id("docs_t_acme", "letters.txt", 0)
    b = ingestion.point_id("docs_t_globex", "letters.txt", 0)
    assert a != b
    assert a == ingestion.point_id("docs_t_acme", "letters.txt", 0)


def test_read_document_offset_basis(tmp_path):
    txt = tmp_path / "a.txt"
    txt.write_text("plain text", encoding="utf-8")
    js = tmp_path / "b.json"
    js.write_text(json.dumps({"k": 1}), encoding="utf-8")
    assert ingestion._read_document(txt) == ("plain text", "txt", "raw")
    assert ingestion._read_document(js) == ("k: 1", "json", "flattened")


def test_merge_author_index_unions_runs(tmp_store, monkeypatch):
    monkeypatch.setattr(
        ingestion.settings, "domain_keywords", {"stoicism": ["seneca", "epictetus"]}
    )
    assert ingestion._merge_author_index(["seneca_letters.txt"]) is True
    assert ingestion._merge_author_index(["epictetus_enchiridion.txt"]) is True
    data = json.loads((tmp_store / "author_index.json").read_text(encoding="utf-8"))
    assert data == {"seneca": ["seneca_letters.txt"], "epictetus": ["epictetus_enchiridion.txt"]}


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
