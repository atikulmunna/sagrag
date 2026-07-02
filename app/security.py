# app/security.py
"""API-key authentication and key -> tenant resolution.

Auth is opt-in (settings.auth_enabled). When enabled, every request outside
EXEMPT_PATHS must carry a valid `x-api-key` header; the key determines the
tenant, which is enforced server-side (a body-supplied tenant is ignored for
authenticated requests). When disabled, behavior is unchanged and no key is
required, so local/dev flows keep working.
"""
from config import settings

API_KEY_HEADER = "x-api-key"

# Paths that never require a key: liveness + metrics (so Prometheus can scrape)
# and the self-documenting OpenAPI endpoints.
EXEMPT_PATHS = frozenset(
    {
        "/health",
        "/metrics",
        "/v1/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
)


def _key_tenant_map() -> dict[str, str]:
    """Build {api_key: tenant}. Sources, merged in order:
    - api_key_map: explicit key -> tenant mapping.
    - api_keys: comma-separated keys; each key's tenant defaults to the key.
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
            mapping[key] = tenant or key
    raw_keys = settings.api_keys or ""
    if isinstance(raw_keys, str):
        for k in raw_keys.split(","):
            key = k.strip()
            if key:
                mapping.setdefault(key, key)
    return mapping


def authenticate(api_key):
    """Return the tenant bound to a valid key, or None for a missing/unknown key."""
    if not api_key:
        return None
    return _key_tenant_map().get(api_key)


def is_exempt(path: str) -> bool:
    """True if `path` is reachable without an api key even when auth is on."""
    return path in EXEMPT_PATHS
