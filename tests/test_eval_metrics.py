"""Unit tests for the pure scoring functions in tools/eval_metrics.py.

The LLM-judge path (llm_judge / _ollama_json) needs a live Ollama and is not
exercised here; these cover the deterministic retrieval + lexical metrics and
the ground-truth loader that the report is built from.
"""

import json

import eval_metrics


# --- lexical_overlap --------------------------------------------------------


def test_lexical_overlap_identical_is_one():
    text = "Seneca teaches that fear is largely imagined and reason dissolves it."
    assert eval_metrics.lexical_overlap(text, text) == 1.0


def test_lexical_overlap_disjoint_is_zero():
    assert (
        eval_metrics.lexical_overlap("apples oranges bananas", "quantum relativity physics") == 0.0
    )


def test_lexical_overlap_partial_between_zero_and_one():
    score = eval_metrics.lexical_overlap(
        "fear is imagined and reason dissolves it",
        "reason dissolves fear which is only imagined worry",
    )
    assert 0.0 < score < 1.0


def test_lexical_overlap_empty_is_zero():
    assert eval_metrics.lexical_overlap("", "something") == 0.0
    assert eval_metrics.lexical_overlap("something", "") == 0.0


# --- recall@k / hit_rate ----------------------------------------------------


def test_recall_at_k_normalizes_paths_and_case():
    retrieved = ["docs/stoicism/Seneca_Fear.txt", "other.txt", "third.txt"]
    expected = ["seneca_fear.txt", "seneca_letters.txt"]
    # 1 of 2 expected present in top-3 → 0.5
    assert eval_metrics.recall_at_k(retrieved, expected, 3) == 0.5


def test_recall_at_k_respects_cutoff():
    retrieved = ["a.txt", "b.txt", "seneca_fear.txt"]
    expected = ["seneca_fear.txt"]
    assert eval_metrics.recall_at_k(retrieved, expected, 2) == 0.0
    assert eval_metrics.recall_at_k(retrieved, expected, 3) == 1.0


def test_recall_at_k_empty_expected_is_zero():
    assert eval_metrics.recall_at_k(["a.txt"], [], 5) == 0.0


def test_hit_rate_is_binary():
    assert eval_metrics.hit_rate_at_k(["x.txt", "seneca_fear.txt"], ["seneca_fear.txt"], 5) == 1.0
    assert eval_metrics.hit_rate_at_k(["x.txt", "y.txt"], ["seneca_fear.txt"], 5) == 0.0


# --- ground truth + scoring -------------------------------------------------


def test_load_ground_truth(tmp_path):
    p = tmp_path / "gt.jsonl"
    p.write_text(
        json.dumps({"query": "q1", "expected_sources": ["a.txt"], "reference_answer": "ref"})
        + "\n"
        + "\n"  # blank line ignored
        + json.dumps({"query": "q2", "expected_sources": []})
        + "\n",
        encoding="utf-8",
    )
    gt = eval_metrics.load_ground_truth(str(p))
    assert set(gt) == {"q1", "q2"}
    assert gt["q1"]["expected_sources"] == ["a.txt"]
    assert gt["q1"]["reference_answer"] == "ref"
    assert gt["q2"]["reference_answer"] == ""


def test_score_run_aggregates_and_counts_errors():
    gt = {
        "q1": {"expected_sources": ["seneca_fear.txt"], "reference_answer": "fear is imagined"},
    }
    rows = [
        {
            "base_url": "A",
            "query": "q1",
            "answer": "fear is imagined",
            "sources": ["seneca_fear.txt", "x.txt"],
            "confidence": 0.8,
        },
        {"base_url": "A", "query": "q1", "error": "boom"},
    ]
    agg = eval_metrics.score_run(rows, gt, k=5, judge=False)
    a = agg["A"]
    assert a["n"] == 2
    assert a["errors"] == 1
    assert a["recall@5"] == 1.0
    assert a["hit_rate@5"] == 1.0
    assert a["lexical_overlap"] == 1.0
    # judge disabled → no faithfulness/relevance
    assert a["faithfulness"] is None
    assert a["confidence"] == 0.8


def test_render_markdown_contains_table():
    agg = {
        "A": {
            "n": 1,
            "errors": 0,
            "recall@5": 1.0,
            "hit_rate@5": 1.0,
            "lexical_overlap": 0.5,
            "faithfulness": None,
            "relevance": None,
            "confidence": 0.8,
        }
    }
    md = eval_metrics.render_markdown(agg, 5)
    assert "recall@5" in md
    assert "| A |" in md
    assert "n/a" in md  # None rendered as n/a
