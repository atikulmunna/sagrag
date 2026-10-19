# app/namespaces.py
"""Storage namespace naming for Qdrant collections and Elasticsearch indices.

Two layouts, selected by whether a tenant is supplied:

- shared (tenant isolation off): ``docs`` and ``docs_{domain}``. This is the
  original layout, so single-tenant deployments keep their data.
- isolated (tenant isolation on): ``docs_t_{tenant}`` and
  ``docs_t_{tenant}__d_{domain}``.

Names are slugs with no consecutive separators, so the ``__d_`` marker is
unambiguous, and domains may not start with ``t_``, so a domain can never be
mistaken for (or used to reach) a tenant namespace. When a tenant is set the
search targets never fall back to the shared namespace: isolation fails closed.
"""
import hashlib
import re

BASE_COLLECTION = "docs"
BASE_INDEX = "docs_index"
MAX_SLUG_LEN = 48

_SLUG_RE = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")
_NON_SLUG_RE = re.compile(r"[^a-z0-9_-]+")
_SEP_RUN_RE = re.compile(r"[-_]{2,}")


class InvalidName(ValueError):
    """A tenant or domain name that cannot be used in a storage name."""


def is_slug(value) -> bool:
    return isinstance(value, str) and len(value) <= MAX_SLUG_LEN and bool(_SLUG_RE.match(value))


def validate_domain(value) -> str:
    """Return the normalized domain slug or raise InvalidName."""
    d = str(value or "").strip().lower()
    if not is_slug(d) or d.startswith("t_"):
        raise InvalidName(f"invalid domain name: {value!r}")
    return d


def slugify(value) -> str | None:
    """Best-effort slug for folder-derived names ("Stoic Texts" -> "stoic-texts")."""
    s = _NON_SLUG_RE.sub("-", str(value or "").strip().lower())
    s = _SEP_RUN_RE.sub(lambda m: m.group(0)[0], s).strip("-_")
    s = s[:MAX_SLUG_LEN].strip("-_")
    if not s or s.startswith("t_"):
        return None
    return s if is_slug(s) else None


def tenant_slug(value) -> str:
    """Map any tenant identifier to a storage-safe slug.

    Slug-shaped ids are kept as-is; anything else (emails, mixed case, long
    strings) is hashed so it can never inject separators into a storage name.
    """
    raw = str(value or "").strip()
    lowered = raw.lower()
    if lowered == raw and is_slug(lowered):
        return lowered
    return "h-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _suffix(tenant: str | None, domain: str | None) -> str | None:
    if tenant:
        return f"t_{tenant}__d_{domain}" if domain else f"t_{tenant}"
    return domain


def target_names(tenant: str | None, domain: str | None) -> tuple[str, str]:
    """(qdrant collection, es index) that ingestion writes to."""
    t = tenant_slug(tenant) if tenant else None
    d = validate_domain(domain) if domain else None
    suffix = _suffix(t, d)
    if suffix is None:
        return BASE_COLLECTION, BASE_INDEX
    return f"{BASE_COLLECTION}_{suffix}", f"{BASE_INDEX}_{suffix}"


def search_targets(tenant: str | None, domain: str | None) -> list[tuple[str, str]]:
    """Ordered (collection, index) candidates for a search.

    Shared layout: the domain namespace, then the base namespace as a fallback
    for legacy data. Isolated layout: only the tenant's own namespace.
    """
    primary = target_names(tenant, domain)
    if tenant or not domain:
        return [primary]
    return [primary, (BASE_COLLECTION, BASE_INDEX)]


def owns_name(name: str, tenant: str | None, base: str) -> bool:
    """True if a collection/index name belongs to this tenant's layout."""
    if not isinstance(name, str):
        return False
    if tenant:
        own = f"{base}_t_{tenant_slug(tenant)}"
        return name == own or name.startswith(own + "__d_")
    if name == base:
        return True
    prefix = base + "_"
    if not name.startswith(prefix):
        return False
    suffix = name[len(prefix):]
    return not suffix.startswith("t_") and is_slug(suffix)


def domains_from_index_names(names, tenant: str | None) -> list[str]:
    """Domains present in a tenant's layout, parsed from ES index names."""
    out = []
    for name in names or []:
        if not owns_name(name, tenant, BASE_INDEX) or name == BASE_INDEX:
            continue
        if tenant:
            marker = name.find("__d_")
            if marker < 0:
                continue
            domain = name[marker + len("__d_"):]
        else:
            domain = name[len(BASE_INDEX) + 1:]
        if is_slug(domain) and domain not in out:
            out.append(domain)
    return out
