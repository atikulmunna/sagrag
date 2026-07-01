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


def test_hallucination_and_coverage_clamped(reset_metrics):
    metrics.record_hallucination_risk(1.5)  # clamped to 1.0
    metrics.record_evidence_coverage(-0.2)  # clamped to 0.0
    out = metrics.render_prometheus()
    assert "sag_rag_hallucination_risk_bucket" in out
    assert "sag_rag_evidence_coverage_bucket" in out
