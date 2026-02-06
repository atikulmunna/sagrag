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

def ingest_folder(folder_path: str = "/data/docs", tenant: str | None = None):
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    from elasticsearch import Elasticsearch
    model = SentenceTransformer(MODEL_NAME)
    qdrant = QdrantClient(url=settings.qdrant_url)
    es = Elasticsearch(settings.elastic_url)
    graph_enabled = settings.graph_enabled

    base_collection = "docs"
    base_index = "docs_index"
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
        collection = base_collection
        index = base_index
        if tenant:
            collection = f"{collection}_{tenant}"
            index = f"{index}_{tenant}"
        if domain:
            collection = f"{base_collection}_{domain}"
            index = f"{base_index}_{domain}"
        _ensure_collection(collection)
        _ensure_index(index)
        mtime = p.stat().st_mtime
        source_type = p.suffix.lower().lstrip(".")
        raw_text = p.read_text(encoding="utf-8")
        text = raw_text
        if source_type == "json":
            try:
                data = json.loads(raw_text)
                lines = _flatten_json(data)
                text = "\n".join(lines)
            except Exception:
                text = raw_text
        if source_type == "csv":
            try:
                lines = _flatten_csv(raw_text)
                text = "\n".join(lines)
            except Exception:
                text = raw_text
        chunks = chunk_text(text)
        embeddings = model.encode([c[0] for c in chunks])
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
            point = {"id": point_id, "vector": embeddings[j].tolist(), "payload": payload}
            try:
                qdrant.upsert(collection_name=collection, points=[point])
            except Exception as e:
                # If collection is missing, create and retry once.
                msg = str(e)
                if "Collection" in msg and "doesn't exist" in msg:
                    try:
                        _ensure_collection(collection)
                        qdrant.upsert(collection_name=collection, points=[point])
                    except Exception as e2:
                        print(f"qdrant upsert failed for {point['id']}: {e2}")
                else:
                    print(f"qdrant upsert failed for {point['id']}: {e}")
            try:
                es.index(
                    index=index,
                    id=point["id"],
                    document={
                        "text": chunk,
                        "source": str(p.name),
                        "timestamp": mtime,
                        "source_type": source_type,
                        "domain": domain,
                        "offset_start": start,
                        "offset_end": end,
                    },
                )
            except Exception as e:
                print(f"es index failed for {point['id']}: {e}")
            if graph_enabled:
                try:
                    from utils import extract_entities, extract_relations
                    from graph import add_chunk_entities_claims
                    entities = extract_entities(chunk)
                    relations = extract_relations(chunk)
                    if entities:
                        add_chunk_entities_claims(point["id"], chunk, entities, relations=relations)
                except Exception as e:
                    print(f"graph ingest failed for {point['id']}: {e}")
            idx += 1
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
