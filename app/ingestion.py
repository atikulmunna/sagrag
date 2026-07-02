# app/ingestion.py
from pathlib import Path
import csv
import json
import uuid
from config import settings

MODEL_NAME = settings.embedding_model_name
EMB_DIM = 384  # all-MiniLM-L6-v2 → 384

def chunk_text(text, chunk_words=200, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    pos = 0
    while i < len(words):
        slice_words = words[i:i+chunk_words]
        chunk = " ".join(slice_words)
        start = text.find(slice_words[0], pos) if slice_words else pos
        if start < 0:
            start = pos
        end = start + len(chunk)
        chunks.append((chunk, start, end))
        pos = end
        i += chunk_words - overlap
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
    """Derive the (qdrant collection, es index) names for a tenant/domain.

    Note: a domain resets the names off the base (dropping the tenant suffix) —
    preserved from the original behavior.
    """
    base_collection = "docs"
    base_index = "docs_index"
    collection = base_collection
    index = base_index
    if tenant:
        collection = f"{collection}_{tenant}"
        index = f"{index}_{tenant}"
    if domain:
        collection = f"{base_collection}_{domain}"
        index = f"{base_index}_{domain}"
    return collection, index


def _batch_upsert_qdrant(qdrant, collection, points, batch_size):
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        try:
            qdrant.upsert(collection_name=collection, points=batch)
        except Exception as e:
            msg = str(e)
            if "Collection" in msg and "doesn't exist" in msg:
                try:
                    from qdrant_client.http import models as qdrant_models
                    qdrant.create_collection(
                        collection_name=collection,
                        vectors_config=qdrant_models.VectorParams(
                            size=EMB_DIM, distance=qdrant_models.Distance.COSINE
                        ),
                    )
                    qdrant.upsert(collection_name=collection, points=batch)
                except Exception as e2:
                    print(f"qdrant batch upsert failed for {collection}: {e2}")
            else:
                print(f"qdrant batch upsert failed for {collection}: {e}")


def _batch_index_es(es, actions):
    if not actions:
        return
    try:
        from elasticsearch import helpers
        helpers.bulk(es, actions, raise_on_error=False)
    except Exception as e:
        print(f"es bulk index failed: {e}")


def _qdrant_source_filter(source):
    from qdrant_client.http import models as qdrant_models
    return qdrant_models.FilterSelector(
        filter=qdrant_models.Filter(
            must=[qdrant_models.FieldCondition(
                key="source", match=qdrant_models.MatchValue(value=source)
            )]
        )
    )


def _delete_source_from_stores(qdrant, es, source, collection, index, graph_enabled):
    """Delete a source's existing points/docs/graph nodes before re-adding."""
    try:
        qdrant.delete(collection_name=collection, points_selector=_qdrant_source_filter(source))
    except Exception as e:
        print(f"qdrant delete-by-source failed for {source}: {e}")
    try:
        es.delete_by_query(
            index=index,
            query={"term": {"source.keyword": source}},
            ignore_unavailable=True,
            conflicts="proceed",
        )
    except Exception as e:
        print(f"es delete-by-source failed for {source}: {e}")
    if graph_enabled:
        try:
            from graph import delete_source_from_graph
            delete_source_from_graph(source)
        except Exception as e:
            print(f"graph delete-by-source failed for {source}: {e}")


def delete_source(source: str, tenant: str | None = None):
    """Remove a source from every store (all docs* collections/indices + graph).

    Used by the DELETE endpoint so the corpus supports removals, not just adds.
    """
    from qdrant_client import QdrantClient
    from elasticsearch import Elasticsearch
    qdrant = QdrantClient(url=settings.qdrant_url)
    es = Elasticsearch(settings.elastic_url)
    deleted = {"source": source, "qdrant_collections": [], "es": False, "graph": False}
    try:
        for c in qdrant.get_collections().collections:
            name = getattr(c, "name", None)
            if not name or not name.startswith("docs"):
                continue
            try:
                qdrant.delete(collection_name=name, points_selector=_qdrant_source_filter(source))
                deleted["qdrant_collections"].append(name)
            except Exception as e:
                print(f"qdrant delete failed for {name}: {e}")
    except Exception as e:
        print(f"qdrant list collections failed: {e}")
    try:
        es.delete_by_query(
            index="docs_index*",
            query={"term": {"source.keyword": source}},
            ignore_unavailable=True,
            allow_no_indices=True,
            conflicts="proceed",
        )
        deleted["es"] = True
    except Exception as e:
        print(f"es delete-by-source failed for {source}: {e}")
    if settings.graph_enabled:
        try:
            from graph import delete_source_from_graph
            delete_source_from_graph(source)
            deleted["graph"] = True
        except Exception as e:
            print(f"graph delete-by-source failed for {source}: {e}")
    return deleted


def ingest_folder(folder_path: str = "/data/docs", tenant: str | None = None):
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    from elasticsearch import Elasticsearch
    model = SentenceTransformer(MODEL_NAME)
    qdrant = QdrantClient(url=settings.qdrant_url)
    es = Elasticsearch(settings.elastic_url)
    graph_enabled = settings.graph_enabled

    created = set()

    def _ensure_collection(name):
        if name in created:
            return
        try:
            # Use explicit Distance enum to avoid invalid distance strings.
            from qdrant_client.http import models as qdrant_models
            qdrant.create_collection(
                collection_name=name,
                vectors_config=qdrant_models.VectorParams(
                    size=EMB_DIM,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
            created.add(name)
            return
        except Exception as e:
            # Creation can fail if it already exists or if config is invalid.
            print(f"qdrant create_collection failed for {name}: {e}")
        # If create failed, check whether the collection exists before caching.
        try:
            qdrant.get_collection(collection_name=name)
            created.add(name)
        except Exception:
            pass

    def _ensure_index(name):
        try:
            if not es.indices.exists(index=name):
                es.indices.create(index=name)
        except Exception:
            pass

    idx = 0
    sources = []
    batch_size = max(1, int(settings.ingest_batch_size))
    root = Path(folder_path)
    for p in root.glob("**/*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in [".txt", ".csv", ".json"]:
            continue
        domain = None
        try:
            if p.parent != root:
                domain = p.parent.name.lower()
        except Exception:
            domain = None
        collection, index = _target_names(tenant, domain)
        _ensure_collection(collection)
        _ensure_index(index)
        mtime = p.stat().st_mtime
        source_type = p.suffix.lower().lstrip(".")
        raw_text = p.read_text(encoding="utf-8")
        text = raw_text
        if source_type == "json":
            try:
                data = json.loads(raw_text)
                text = "\n".join(_flatten_json(data))
            except Exception:
                text = raw_text
        if source_type == "csv":
            try:
                text = "\n".join(_flatten_csv(raw_text))
            except Exception:
                text = raw_text
        chunks = chunk_text(text)
        if not chunks:
            continue
        embeddings = model.encode([c[0] for c in chunks])

        # Re-ingest replaces the source: drop its existing points/docs/graph
        # first so the store stops being append-only.
        if settings.reingest_replaces_source:
            _delete_source_from_stores(qdrant, es, str(p.name), collection, index, graph_enabled)

        points = []
        es_actions = []
        for j, (chunk, start, end) in enumerate(chunks):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{p.name}:{j}"))
            payload = {
                "text": chunk,
                "source": str(p.name),
                "timestamp": mtime,
                "source_type": source_type,
                "domain": domain,
                "offset_start": start,
                "offset_end": end,
            }
            points.append({"id": point_id, "vector": embeddings[j].tolist(), "payload": payload})
            es_actions.append({"_index": index, "_id": point_id, "_source": dict(payload)})
            if graph_enabled:
                try:
                    from utils import extract_entities, extract_relations
                    from graph import add_chunk_entities_claims
                    entities = extract_entities(chunk)
                    relations = extract_relations(chunk)
                    if entities:
                        add_chunk_entities_claims(point_id, chunk, entities, relations=relations, source=str(p.name))
                except Exception as e:
                    print(f"graph ingest failed for {point_id}: {e}")
            idx += 1

        # Batched writes: one upsert per batch, one bulk request per file.
        _batch_upsert_qdrant(qdrant, collection, points, batch_size)
        _batch_index_es(es, es_actions)
        sources.append(str(p.name))
    author_index = {}
    author_terms = set()
    for keywords in (settings.domain_keywords or {}).values():
        for kw in keywords:
            kw_l = str(kw).strip().lower()
            if kw_l:
                author_terms.add(kw_l)
    if author_terms and sources:
        for src in sources:
            src_l = src.lower()
            for term in author_terms:
                if term in src_l:
                    author_index.setdefault(term, set()).add(src)
        # write index to disk for fast author filtering
        path = Path(settings.author_index_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {k: sorted(list(v)) for k, v in author_index.items()}
            path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
        except Exception as e:
            print(f"author index write failed: {e}")
    print(f"Ingested {idx} chunks from {len(sources)} files.")
    return {"ingested_chunks": idx, "sources": sources, "author_index_written": bool(author_index)}
