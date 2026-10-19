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

    metrics.reset()
    yield
    metrics.reset()


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


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    """Point every on-disk path (SQLite, exports, author index) at a temp dir."""
    import store
    from config import settings

    monkeypatch.setattr(settings, "audit_db_path", str(tmp_path / "audit.db"))
    monkeypatch.setattr(settings, "feedback_db_path", str(tmp_path / "feedback.db"))
    monkeypatch.setattr(settings, "audit_export_path", str(tmp_path / "export" / "audit.jsonl"))
    monkeypatch.setattr(settings, "learning_export_path", str(tmp_path / "export" / "train.jsonl"))
    monkeypatch.setattr(settings, "author_index_path", str(tmp_path / "author_index.json"))
    store._INITIALIZED.clear()
    yield tmp_path
    store._INITIALIZED.clear()


@pytest.fixture
def api_client(monkeypatch, tmp_store, reset_metrics):
    """TestClient over the real app with external services switched off.

    Lifespan is not run (no `with`), so nothing tries to reach Neo4j/Redis.
    """
    from fastapi.testclient import TestClient

    import agents
    import main
    import redis_client
    from config import settings

    for name, value in {
        "redis_url": "",
        "redis_cache_enabled": False,
        "redis_rate_limit_enabled": False,
        "rate_limit_per_minute": 0,
        "auth_enabled": False,
        "tenant_isolation": False,
        "graph_enabled": False,
        "api_keys": "",
        "api_key_map": {},
        "trusted_proxy_hops": 0,
        "domain_keywords": {},
        "domain_aliases": {},
        "domain_fallbacks": [],
        "domain_index_map": {},
    }.items():
        monkeypatch.setattr(settings, name, value)
    monkeypatch.setattr(redis_client, "_client", None)
    monkeypatch.setattr(agents, "_list_domain_indices", lambda tenant=None: [])
    main._rate_state.clear()
    yield TestClient(main.app)
    main._rate_state.clear()
