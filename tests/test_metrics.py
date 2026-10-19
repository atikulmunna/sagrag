"""Unit tests for the in-process metrics recorder and Prometheus rendering."""

import metrics


def test_synthesis_outcome_counter(reset_metrics):
    metrics.record_synthesis("success", 120)
    metrics.record_synthesis("success", 300)
    metrics.record_synthesis("timeout", 5000)
    out = metrics.render_prometheus()
    assert 'sag_rag_synthesis_total{outcome="success"} 2' in out
    assert 'sag_rag_synthesis_total{outcome="timeout"} 1' in out
    # Latency histogram emits a +Inf bucket per outcome.
    assert 'sag_rag_synthesis_latency_ms_bucket{outcome="success",le="+Inf"} 2' in out


def test_histograms_have_sum_and_count(reset_metrics):
    metrics.record_synthesis("success", 120)
    metrics.record_synthesis("success", 300)
    out = metrics.render_prometheus()
    assert "# TYPE sag_rag_synthesis_latency_ms histogram" in out
    assert 'sag_rag_synthesis_latency_ms_sum{outcome="success"} 420' in out
    assert 'sag_rag_synthesis_latency_ms_count{outcome="success"} 2' in out
    # Buckets are cumulative.
    assert 'sag_rag_synthesis_latency_ms_bucket{outcome="success",le="250"} 1' in out
    assert 'sag_rag_synthesis_latency_ms_bucket{outcome="success",le="500"} 2' in out


def test_retrieval_failure_counter(reset_metrics):
    metrics.record_retrieval_failure("no_domain")
    metrics.record_retrieval_failure("no_domain")
    metrics.record_retrieval_failure("author_gap")
    out = metrics.render_prometheus()
    assert 'sag_rag_retrieval_failures_total{tag="no_domain"} 2' in out
    assert 'sag_rag_retrieval_failures_total{tag="author_gap"} 1' in out


def test_request_counter(reset_metrics):
    metrics.record_request("POST", "/v1/query", 200, 42)
    out = metrics.render_prometheus()
    assert 'sag_rag_requests_total{method="POST",path="/v1/query",status="200"} 1' in out
    assert 'sag_rag_request_latency_ms_count{method="POST",path="/v1/query",status="200"} 1' in out


def test_label_values_are_escaped(reset_metrics):
    metrics.record_request("GET", 'a"b\\c\nd', 404, 1)
    out = metrics.render_prometheus()
    assert 'path="a\\"b\\\\c\\nd"' in out


def test_author_label_cardinality_is_capped(reset_metrics, monkeypatch):
    monkeypatch.setattr(metrics, "_MAX_AUTHOR_LABELS", 3)
    for i in range(10):
        metrics.record_author_query([f"author{i}"])
    out = metrics.render_prometheus()
    assert out.count("sag_rag_author_queries_total{") == 4  # 3 distinct + "other"
    assert 'sag_rag_author_queries_total{author="other"} 7' in out


def test_hallucination_and_coverage_clamped(reset_metrics):
    metrics.record_hallucination_risk(1.5)  # clamped to 1.0
    metrics.record_evidence_coverage(-0.2)  # clamped to 0.0
    out = metrics.render_prometheus()
    assert 'sag_rag_hallucination_risk_bucket{le="0.8"} 0' in out
    assert 'sag_rag_hallucination_risk_bucket{le="1"} 1' in out
    assert 'sag_rag_evidence_coverage_bucket{le="0.25"} 1' in out
