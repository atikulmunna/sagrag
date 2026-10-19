# app/ingestion.py
from pathlib import Path
import csv
import json
import logging
import re
import uuid

import embeddings
import namespaces
from config import settings

_LOG = logging.getLogger(__name__)
_WORD_RE = re.compile(r"\S+")
SUPPORTED_SUFFIXES = (".txt", ".csv", ".json")


def chunk_text(text, chunk_words=200, overlap=50):
    """Split text into overlapping word windows.

    Returns ``(chunk, start, end)`` tuples where ``chunk == text[start:end]``
    exactly, so provenance offsets always point at the cited characters (and
    newlines inside a chunk are preserved for the structured agent).
    """
    if overlap >= chunk_words:
        raise ValueError("overlap must be smaller than chunk_words")
    spans = [m.span() for m in _WORD_RE.finditer(text or "")]
    chunks = []
    step = chunk_words - overlap
    for i in range(0, len(spans), step):
        window = spans[i:i + chunk_words]
        start, end = window[0][0], window[-1][1]
        chunks.append((text[start:end], start, end))
        if i + chunk_words >= len(spans):
            break
    return chunks


def _flatten_json(value, prefix="", lines=None, max_lines=500):
    if lines is None:
        lines = []
    if len(lines) >= max_lines:
        return lines
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten_json(v, f"{prefix}{k}.", lines, max_lines=max_lines)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _flatten_json(v, f"{prefix}{i}.", lines, max_lines=max_lines)
    else:
        lines.append(f"{prefix[:-1]}: {value}")
    return lines


def _flatten_csv(text: str, max_lines=500):
    lines = []
    reader = csv.reader(text.splitlines())
    rows = list(reader)
    if not rows:
        return lines
    header = rows[0]
    for row in rows[1:]:
        if len(lines) >= max_lines:
            break
        pairs = []
        for i, cell in enumerate(row):
            key = header[i] if i < len(header) else f"col_{i}"
            pairs.append(f"{key}: {cell}")
        lines.append(" | ".join(pairs))
    return lines


def _target_names(tenant: str | None, domain: str | None):
    """(qdrant collection, es index) for a tenant/domain. See namespaces.py."""
    return namespaces.target_names(tenant, domain)


def point_id(collection: str, source: str, j: int) -> str:
    """Deterministic chunk id, unique per namespace (no cross-tenant collisions)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{collection}:{source}:{j}"))


def _batch_upsert_qdrant(qdrant, collection, points, batch_size, errors=None, dim=None):
    """Upsert in batches; failures are appended to ``errors`` (never raised)."""
    written = 0
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        try:
            qdrant.upsert(collection_name=collection, points=batch)
            written += len(batch)
        except Exception as e:
            msg = str(e)
            if dim and "Collection" in msg and "doesn't exist" in msg:
                try:
                    from qdrant_client.http import models as qdrant_models
                    qdrant.create_collection(
                        collection_name=collection,
                        vectors_config=qdrant_models.VectorParams(
                            size=dim, distance=qdrant_models.Distance.COSINE
                        ),
                    )
                    qdrant.upsert(collection_name=collection, points=batch)
                    written += len(batch)
                    continue
                except Exception as e2:
                    e = e2
            _LOG.warning("qdrant_upsert_failed collection=%s: %s", collection, e)
            if errors is not None:
                errors.append({"stage": "qdrant_upsert", "collection": collection, "error": str(e)[:300]})
    return written


def _batch_index_es(es, actions, errors=None):
    """Bulk index; per-document failures are reported, not swallowed."""
    if not actions:
        return 0
    try:
        from elasticsearch import helpers
        ok, failures = helpers.bulk(es, actions, raise_on_error=False)
        if failures and errors is not None:
            errors.append({"stage": "es_bulk", "failed_docs": len(failures), "error": str(failures[0])[:300]})
        return int(ok)
    except Exception as e:
        _LOG.warning("es_bulk_failed: %s", e)
        if errors is not None:
            errors.append({"stage": "es_bulk", "error": str(e)[:300]})
        return 0


def _qdrant_source_filter(source):
    from qdrant_client.http import models as qdrant_models
    return qdrant_models.FilterSelector(
        filter=qdrant_models.Filter(
            must=[qdrant_models.FieldCondition(
                key="source", match=qdrant_models.MatchValue(value=source)
            )]
        )
    )


def _delete_source_from_stores(qdrant, es, source, collection, index, graph_enabled, tenant=None):
    """Delete a source's existing points/docs/graph nodes before re-adding."""
    try:
        qdrant.delete(collection_name=collection, points_selector=_qdrant_source_filter(source))
    except Exception as e:
        _LOG.info("qdrant delete-by-source skipped for %s: %s", source, e)
    try:
        es.delete_by_query(
            index=index,
            query={"term": {"source.keyword": source}},
            ignore_unavailable=True,
            conflicts="proceed",
        )
    except Exception as e:
        _LOG.info("es delete-by-source skipped for %s: %s", source, e)
    if graph_enabled:
        try:
            from graph import delete_source_from_graph
            delete_source_from_graph(source, tenant=tenant)
        except Exception as e:
            _LOG.warning("graph delete-by-source failed for %s: %s", source, e)


def delete_source(source: str, tenant: str | None = None):
    """Remove a source from every namespace the caller owns (and the graph).

    With tenant isolation on, only ``docs_t_{tenant}*`` is touched; otherwise
    only the shared layout (``docs`` / ``docs_{domain}``). A tenant can never
    delete another tenant's copy of a file.
    """
    from qdrant_client import QdrantClient
    from elasticsearch import Elasticsearch
    qdrant = QdrantClient(url=settings.qdrant_url)
    es = Elasticsearch(settings.elastic_url)
    deleted = {"source": source, "qdrant_collections": [], "es_indices": [], "graph": False, "errors": []}
    try:
        for c in qdrant.get_collections().collections:
            name = getattr(c, "name", None)
            if not namespaces.owns_name(name, tenant, namespaces.BASE_COLLECTION):
                continue
            try:
                qdrant.delete(collection_name=name, points_selector=_qdrant_source_filter(source))
                deleted["qdrant_collections"].append(name)
            except Exception as e:
                deleted["errors"].append({"stage": "qdrant_delete", "collection": name, "error": str(e)[:300]})
    except Exception as e:
        deleted["errors"].append({"stage": "qdrant_list", "error": str(e)[:300]})
    try:
        names = list(es.indices.get(index=f"{namespaces.BASE_INDEX}*", ignore_unavailable=True).keys())
        own = [n for n in names if namespaces.owns_name(n, tenant, namespaces.BASE_INDEX)]
        if own:
            es.delete_by_query(
                index=",".join(own),
                query={"term": {"source.keyword": source}},
                ignore_unavailable=True,
                allow_no_indices=True,
                conflicts="proceed",
            )
        deleted["es_indices"] = own
    except Exception as e:
        deleted["errors"].append({"stage": "es_delete", "error": str(e)[:300]})
    if settings.graph_enabled:
        try:
            from graph import delete_source_from_graph
            delete_source_from_graph(source, tenant=tenant)
            deleted["graph"] = True
        except Exception as e:
            deleted["errors"].append({"stage": "graph_delete", "error": str(e)[:300]})
    return deleted


def _read_document(p: Path):
    """Return (text, source_type, offset_basis) for a supported file."""
    source_type = p.suffix.lower().lstrip(".")
    raw_text = p.read_text(encoding="utf-8")
    # Offsets index into the text that was chunked: the raw file for .txt, the
    # flattened "key: value" rendering for structured formats.
    if source_type == "json":
        try:
            return "\n".join(_flatten_json(json.loads(raw_text))), source_type, "flattened"
        except ValueError:
            return raw_text, source_type, "raw"
    if source_type == "csv":
        try:
            return "\n".join(_flatten_csv(raw_text)), source_type, "flattened"
        except csv.Error:
            return raw_text, source_type, "raw"
    return raw_text, source_type, "raw"


def _merge_author_index(sources):
    """Union this run's author -> source hits into the on-disk index.

    Merging (instead of overwriting) keeps entries from earlier ingests and
    other tenants; the index is only a ranking hint, never an access check.
    """
    author_terms = set()
    for keywords in (settings.domain_keywords or {}).values():
        for kw in keywords:
            kw_l = str(kw).strip().lower()
            if kw_l:
                author_terms.add(kw_l)
    if not author_terms or not sources:
        return False
    path = Path(settings.author_index_path)
    index = {}
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(existing, dict):
            index = {str(k): set(v) for k, v in existing.items() if isinstance(v, list)}
    except (OSError, ValueError):
        index = {}
    changed = False
    for src in sources:
        src_l = src.lower()
        for term in author_terms:
            if term in src_l and src not in index.get(term, set()):
                index.setdefault(term, set()).add(src)
                changed = True
    if not changed:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: sorted(v) for k, v in index.items()}
        path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
        return True
    except OSError as e:
        _LOG.warning("author index write failed: %s", e)
        return False


def ingest_folder(folder_path: str = "/data/docs", tenant: str | None = None):
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
    from elasticsearch import Elasticsearch
    qdrant = QdrantClient(url=settings.qdrant_url)
    es = Elasticsearch(settings.elastic_url)
    graph_enabled = settings.graph_enabled
    dim = embeddings.dimension()

    created = set()
    errors: list[dict] = []

    def _ensure_collection(name):
        if name in created:
            return
        try:
            if not qdrant.collection_exists(collection_name=name):
                qdrant.create_collection(
                    collection_name=name,
                    vectors_config=qdrant_models.VectorParams(size=dim, distance=qdrant_models.Distance.COSINE),
                )
            created.add(name)
        except Exception as e:
            errors.append({"stage": "qdrant_create", "collection": name, "error": str(e)[:300]})

    def _ensure_index(name):
        try:
            if not es.indices.exists(index=name):
                es.indices.create(index=name)
        except Exception as e:
            errors.append({"stage": "es_create", "index": name, "error": str(e)[:300]})

    idx = 0
    sources = []
    batch_size = max(1, int(settings.ingest_batch_size))
    root = Path(folder_path)
    for p in sorted(root.glob("**/*")):
        if not p.is_file() or p.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        domain = None
        if p.parent != root:
            domain = namespaces.slugify(p.parent.name)
            if domain is None:
                errors.append({"stage": "domain", "source": p.name, "error": f"folder name {p.parent.name!r} is not a usable domain"})
                continue
        source = str(p.name)
        try:
            collection, index = _target_names(tenant, domain)
            text, source_type, offset_basis = _read_document(p)
            chunks = chunk_text(text)
            if not chunks:
                continue
            vectors = embeddings.encode([c[0] for c in chunks])
        except Exception as e:
            errors.append({"stage": "read_embed", "source": source, "error": str(e)[:300]})
            continue
        _ensure_collection(collection)
        _ensure_index(index)

        # Re-ingest replaces the source: drop its existing points/docs/graph
        # first so the store stops being append-only.
        if settings.reingest_replaces_source:
            _delete_source_from_stores(qdrant, es, source, collection, index, graph_enabled, tenant=tenant)

        mtime = p.stat().st_mtime
        points = []
        es_actions = []
        for j, (chunk, start, end) in enumerate(chunks):
            pid = point_id(collection, source, j)
            payload = {
                "text": chunk,
                "source": source,
                "timestamp": mtime,
                "source_type": source_type,
                "domain": domain,
                "offset_start": start,
                "offset_end": end,
                "offset_basis": offset_basis,
            }
            points.append({"id": pid, "vector": vectors[j], "payload": payload})
            es_actions.append({"_index": index, "_id": pid, "_source": dict(payload)})
            if graph_enabled:
                try:
                    from utils import extract_entities_and_relations
                    from graph import add_chunk_entities_claims
                    entities, relations = extract_entities_and_relations(chunk)
                    if entities:
                        add_chunk_entities_claims(pid, chunk, entities, relations=relations, source=source, tenant=tenant)
                except Exception as e:
                    errors.append({"stage": "graph", "source": source, "chunk": j, "error": str(e)[:300]})
            idx += 1

        # Batched writes: one upsert per batch, one bulk request per file.
        _batch_upsert_qdrant(qdrant, collection, points, batch_size, errors=errors, dim=dim)
        _batch_index_es(es, es_actions, errors=errors)
        sources.append(source)
    author_index_written = _merge_author_index(sources)
    _LOG.info("ingested %d chunks from %d files (%d errors)", idx, len(sources), len(errors))
    return {
        "ingested_chunks": idx,
        "sources": sources,
        "author_index_written": author_index_written,
        "errors": errors[:50],
        "error_count": len(errors),
    }
