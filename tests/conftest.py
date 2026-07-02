"""Shared pytest fixtures.

Tests import app modules by bare name (e.g. `import synthesis`); `app/` is put
on sys.path via [tool.pytest.ini_options] pythonpath in pyproject.toml.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def _domain_packs_dir(monkeypatch):
    """Point domain-pack loading at the shipped data/domain_packs so unit tests
    reflect real default behavior (in the container /data/domain_packs is
    mounted; on the host it's the repo dir)."""
    import domain_packs
    from config import settings

    packs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "domain_packs")
    monkeypatch.setattr(settings, "domain_packs_path", packs_dir)
    domain_packs._PACKS_CACHE = None
    domain_packs._PACKS_MTIME = None
    yield
    domain_packs._PACKS_CACHE = None
    domain_packs._PACKS_MTIME = None


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
