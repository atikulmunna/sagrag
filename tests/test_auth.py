"""API-key auth: key -> tenant resolution, exemptions, and the tenant-binding
behavior of api._resolve_tenant. The middleware itself is a thin wrapper over
these pure functions (main imports heavy deps, so it is not exercised here)."""

import types

import pytest

import security
import api

_MISSING = object()


@pytest.fixture(autouse=True)
def _clear_key_config(monkeypatch):
    """Start each test from an empty key config; tests opt in explicitly."""
    monkeypatch.setattr(security.settings, "api_keys", "")
    monkeypatch.setattr(security.settings, "api_key_map", {})


def _req(tenant=_MISSING):
    """Build a fake request with an optional request.state.tenant."""
    state = types.SimpleNamespace()
    if tenant is not _MISSING:
        state.tenant = tenant
    return types.SimpleNamespace(state=state)


def test_authenticate_from_comma_list(monkeypatch):
    monkeypatch.setattr(security.settings, "api_keys", "k1, k2 ,k3")
    # tenant defaults to the key itself
    assert security.authenticate("k1") == "k1"
    assert security.authenticate("k2") == "k2"
    assert security.authenticate("k3") == "k3"


def test_authenticate_from_key_map(monkeypatch):
    monkeypatch.setattr(security.settings, "api_key_map", {"secret": "acme"})
    assert security.authenticate("secret") == "acme"


def test_key_map_overrides_list_tenant_default(monkeypatch):
    monkeypatch.setattr(security.settings, "api_keys", "shared")
    monkeypatch.setattr(security.settings, "api_key_map", {"shared": "globex"})
    # explicit map tenant wins over the "tenant == key" default
    assert security.authenticate("shared") == "globex"


def test_authenticate_rejects_unknown_and_empty(monkeypatch):
    monkeypatch.setattr(security.settings, "api_keys", "k1")
    assert security.authenticate("nope") is None
    assert security.authenticate("") is None
    assert security.authenticate(None) is None


def test_blank_map_entries_ignored(monkeypatch):
    monkeypatch.setattr(security.settings, "api_key_map", {"  ": "x", "good": "  "})
    # blank key dropped; blank tenant falls back to the key
    assert security.authenticate("good") == "good"
    assert security.authenticate("  ") is None


def test_is_exempt():
    assert security.is_exempt("/health")
    assert security.is_exempt("/metrics")
    assert security.is_exempt("/v1/metrics")
    assert security.is_exempt("/docs")
    assert not security.is_exempt("/v1/query")
    assert not security.is_exempt("/ingest")


def test_resolve_tenant_prefers_authenticated_key():
    # authenticated tenant is authoritative; body tenant is ignored
    req = _req(tenant="acme")
    assert api._resolve_tenant(req, {"tenant": "attacker"}) == "acme"


def test_resolve_tenant_falls_back_to_body_when_unauthenticated():
    req = _req()  # no request.state.tenant
    assert api._resolve_tenant(req, {"tenant": "acme"}) == "acme"
    assert api._resolve_tenant(req, {}) is None
    assert api._resolve_tenant(req, None) is None
