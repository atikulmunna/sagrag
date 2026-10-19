# app/security.py
"""API-key authentication and key -> tenant resolution.

Auth is opt-in (settings.auth_enabled). When enabled, every request outside
EXEMPT_PATHS must carry a valid `x-api-key` header; the key determines the
tenant, which is enforced server-side (a body-supplied tenant is ignored for
authenticated requests). When disabled, behavior is unchanged and no key is
required, so local/dev flows keep working.
"""
import hashlib
import hmac

import namespaces
from config import settings

API_KEY_HEADER = "x-api-key"

# Paths that never require a key: liveness/readiness, metrics (so Prometheus
# can scrape), and the self-documenting OpenAPI endpoints.
EXEMPT_PATHS = frozenset(
    {
        "/health",
        "/ready",
        "/metrics",
        "/v1/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
)


def default_tenant_for_key(key: str) -> str:
    """Tenant id for a key without an explicit mapping.

    Derived from a hash so the secret itself never appears in storage names,
    audit rows, logs, or metrics.
    """
    return "key-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def _key_tenant_map() -> dict[str, str]:
    """Build {api_key: tenant}. Sources, merged in order:
    - api_key_map: explicit key -> tenant mapping.
    - api_keys: comma-separated keys; each gets a hash-derived tenant.
    Parsed per call (the key set is tiny) so config/env changes take effect.
    """
    mapping: dict[str, str] = {}
    raw_map = settings.api_key_map or {}
    if isinstance(raw_map, dict):
        for k, v in raw_map.items():
            key = str(k).strip()
            if not key:
                continue
            tenant = str(v).strip() if v is not None else ""
            mapping[key] = namespaces.tenant_slug(tenant) if tenant else default_tenant_for_key(key)
    raw_keys = settings.api_keys or ""
    if isinstance(raw_keys, str):
        for k in raw_keys.split(","):
            key = k.strip()
            if key:
                mapping.setdefault(key, default_tenant_for_key(key))
    return mapping


def authenticate(api_key):
    """Return the tenant bound to a valid key, or None for a missing/unknown key.

    Compares against every configured key in constant time.
    """
    if not api_key:
        return None
    supplied = str(api_key).encode("utf-8")
    tenant = None
    for key, mapped in _key_tenant_map().items():
        if hmac.compare_digest(key.encode("utf-8"), supplied):
            tenant = mapped
    return tenant


def is_exempt(path: str) -> bool:
    """True if `path` is reachable without an api key even when auth is on."""
    return path in EXEMPT_PATHS
