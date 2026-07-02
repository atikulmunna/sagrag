"""Shared pytest fixtures.

Tests import app modules by bare name (e.g. `import synthesis`); `app/` is put
on sys.path via [tool.pytest.ini_options] pythonpath in pyproject.toml.
"""

import pytest


@pytest.fixture
def reset_metrics():
    """Clear the module-level metric dicts so counter assertions are isolated."""
    import metrics

    dict_names = [
        "_request_count",
        "_request_latency_ms",
        "_request_latency_counts",
        "_author_gap_count",
        "_author_query_count",
        "_retrieval_failure_count",
        "_hallucination_risk_counts",
        "_evidence_coverage_counts",
        "_synthesis_outcome_count",
        "_synthesis_latency_ms_total",
        "_synthesis_latency_ms_counts",
    ]
    for name in dict_names:
        getattr(metrics, name).clear()
    yield
    for name in dict_names:
        getattr(metrics, name).clear()


class FakeLLM:
    """Stand-in for the shared llm client's async `completion` method.

    Configure `responses` (list, consumed per call) or `raises` (exception to
    throw). Records prompts for assertions.
    """

    def __init__(self, responses=None, raises=None, stream_chunks=None):
        self.responses = list(responses or [])
        self.raises = raises
        self.stream_chunks = list(stream_chunks or [])
        self.calls = []

    async def completion(self, prompt, max_tokens=512, **kwargs):
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens, **kwargs})
        if self.raises is not None:
            raise self.raises
        if not self.responses:
            return ""
        # Repeat the last response for any extra calls (e.g. naturalizer retry).
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)

    async def completion_stream(self, prompt, max_tokens=512, **kwargs):
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens, "stream": True, **kwargs})
        if self.raises is not None:
            raise self.raises
        for chunk in self.stream_chunks:
            yield chunk
