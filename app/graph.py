# app/graph.py
"""Neo4j knowledge graph: chunks, claims, entities, relations, contradictions.

Every Chunk, Claim and RELATES edge carries a ``tenant`` key ("" in the shared
layout) and every read filters on it, so graph signals never cross tenants.
Nodes written before tenant keys existed have no ``tenant`` property and are
treated as shared ("") via ``coalesce``.
"""
import logging
import threading

import namespaces
from config import settings

_LOG = logging.getLogger(__name__)
_DRIVER = None
_LOCK = threading.Lock()

SCHEMA_STATEMENTS = (
    "CREATE CONSTRAINT sagrag_chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT sagrag_claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT sagrag_entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
    "CREATE INDEX sagrag_chunk_source IF NOT EXISTS FOR (c:Chunk) ON (c.source)",
    "CREATE INDEX sagrag_claim_created IF NOT EXISTS FOR (c:Claim) ON (c.created_at)",
)


def get_driver():
    global _DRIVER
    if _DRIVER is None:
        with _LOCK:
            if _DRIVER is None:
                from neo4j import GraphDatabase
                _DRIVER = GraphDatabase.driver(
                    settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
                )
    return _DRIVER


def close_driver():
    global _DRIVER
    if _DRIVER is not None:
        try:
            _DRIVER.close()
        finally:
            _DRIVER = None


def ensure_schema():
    """Create uniqueness constraints (MERGE without them scans the whole label)."""
    with get_driver().session() as session:
        for stmt in SCHEMA_STATEMENTS:
            try:
                session.run(stmt).consume()
            except Exception as e:
                # e.g. duplicate legacy Entity names block the constraint.
                _LOG.warning("neo4j schema statement failed (%s): %s", stmt, e)


def tenant_key(tenant) -> str:
    return namespaces.tenant_slug(tenant) if tenant else ""


def _split_sentences(text: str, limit: int):
    # Simple splitter to avoid heavy dependencies in ingestion.
    parts = []
    for chunk in text.replace("\n", " ").split("."):
        s = chunk.strip()
        if s:
            parts.append(s)
        if len(parts) >= limit:
            break
    return parts


def _extract_relations_fallback(text: str, entities: list):
    relations = []
    if not entities:
        return relations
    lowered = text.lower()
    predicates = [" is ", " are ", " has ", " have ", " causes ", " leads to ", " supports "]
    for pred in predicates:
        if pred in lowered:
            for i in range(len(entities)):
                for j in range(len(entities)):
                    if i == j:
                        continue
                    a = entities[i]
                    b = entities[j]
                    if a.lower() in lowered and b.lower() in lowered:
                        relations.append((a, pred.strip(), b))
            break
    return relations


def add_chunk_entities_claims(
    chunk_id: str,
    text: str,
    entities: list,
    relations: list | None = None,
    source: str | None = None,
    tenant: str | None = None,
):
    tkey = tenant_key(tenant)
    claims = _split_sentences(text, settings.graph_max_claims)
    if relations is None:
        relations = _extract_relations_fallback(text, entities)
    claim_rows = []
    for idx, claim in enumerate(claims):
        claim_lower = claim.lower()
        claim_rows.append({
            "id": f"{chunk_id}::claim::{idx}",
            "text": claim,
            "entities": [e for e in entities if e.lower() in claim_lower],
        })
    rel_rows = [{"a": a, "p": p, "b": b} for a, p, b in relations]

    def _work(tx):
        tx.run(
            "MERGE (c:Chunk {id:$id}) SET c.text=$text, c.source=$source, c.tenant=$tenant",
            id=chunk_id, text=text, source=source, tenant=tkey,
        )
        if entities:
            tx.run("UNWIND $names AS name MERGE (e:Entity {name:name})", names=entities)
            tx.run(
                """
                MATCH (c:Chunk {id:$cid})
                UNWIND $names AS name
                MATCH (e:Entity {name:name})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                cid=chunk_id, names=entities,
            )
        if claim_rows:
            tx.run(
                """
                MATCH (c:Chunk {id:$cid})
                UNWIND $claims AS claim
                MERGE (cl:Claim {id: claim.id})
                SET cl.text = claim.text, cl.tenant = $tenant, cl.created_at = timestamp()
                MERGE (c)-[:SUPPORTS]->(cl)
                WITH cl, claim
                UNWIND claim.entities AS name
                MATCH (e:Entity {name:name})
                MERGE (cl)-[:MENTIONS]->(e)
                """,
                cid=chunk_id, claims=claim_rows, tenant=tkey,
            )
        if rel_rows:
            tx.run(
                """
                UNWIND $rels AS rel
                MATCH (a:Entity {name: rel.a}), (b:Entity {name: rel.b})
                MERGE (a)-[:RELATES {predicate: rel.p, tenant: $tenant}]->(b)
                """,
                rels=rel_rows, tenant=tkey,
            )

    with get_driver().session() as session:
        session.execute_write(_work)
        _add_contradictions(session, [(c["id"], c["text"]) for c in claim_rows], entities, tkey)


def _negate(text: str):
    negators = [" not ", " no ", " never ", " cannot ", " can't ", " won't ", " isn't ", " aren't ", " wasn't ", " weren't "]
    lowered = f" {text.lower()} "
    return any(n in lowered for n in negators)


def _normalize(text: str):
    return " ".join(text.lower().split())


def _token_set(text: str):
    tokens = []
    for t in _normalize(text).replace(",", " ").replace(";", " ").replace(":", " ").split():
        if t.isalpha() and len(t) > 2:
            tokens.append(t)
    return set(tokens)


def _rough_match(a: str, b: str):
    # Allow paraphrases: require token overlap, not exact match.
    ta = _token_set(a)
    tb = _token_set(b)
    if not ta or not tb:
        return False
    overlap = len(ta.intersection(tb))
    min_size = min(len(ta), len(tb))
    return overlap >= max(3, int(settings.graph_contradiction_overlap * min_size))


def _extract_numbers(text: str):
    nums = []
    token = ""
    for ch in text:
        if ch.isdigit() or ch == ".":
            token += ch
        elif token:
            try:
                nums.append(float(token))
            except ValueError:
                pass
            token = ""
    if token:
        try:
            nums.append(float(token))
        except ValueError:
            pass
    return nums


def _entity_overlap(text_a: str, text_b: str, entities: list):
    la = text_a.lower()
    lb = text_b.lower()
    return any(ent.lower() in la and ent.lower() in lb for ent in entities or [])


def find_contradictions(claim_meta, recent, entities, nli=None, threshold: float = 0.5):
    """Pairs (new_id, old_id) whose claims look contradictory.

    Candidates need heavily overlapping wording plus a shared entity (or
    differing numbers). With ``nli`` (a function returning entailment and
    contradiction probabilities per (premise, hypothesis) pair), a candidate
    is a contradiction when either direction is judged contradictory; without
    it, opposite negation is the fallback heuristic.
    """
    candidates = []
    for new_id, new_text in claim_meta:
        new_norm = _normalize(new_text)
        new_nums = _extract_numbers(new_text)
        for r in recent:
            old_text = r.get("text") or ""
            old_norm = _normalize(old_text)
            if not old_norm or not _rough_match(new_norm, old_norm):
                continue
            # Require entity overlap or numeric mismatch to reduce false positives
            if not _entity_overlap(new_text, old_text, entities=entities):
                old_nums = _extract_numbers(old_text)
                if not new_nums or not old_nums or new_nums == old_nums:
                    continue
            candidates.append((new_id, new_text, r.get("id"), old_text))
    if not candidates:
        return []
    if nli is not None:
        try:
            pairs = [(old, new) for _, new, _, old in candidates] + [(new, old) for _, new, _, old in candidates]
            probs = nli(pairs)
            n = len(candidates)
            return [
                (c[0], c[2])
                for i, c in enumerate(candidates)
                if max(probs[i]["contradiction"], probs[i + n]["contradiction"]) >= threshold
            ]
        except Exception as e:
            _LOG.warning("nli contradiction check failed, using negation heuristic: %s", e)
    return [(new_id, old_id) for new_id, new_text, old_id, old_text in candidates if _negate(old_text) != _negate(new_text)]


def _add_contradictions(session, claim_meta, entities, tkey: str):
    if not claim_meta:
        return
    # Compare against this tenant's most recent claims only.
    query = """
    MATCH (cl:Claim)
    WHERE coalesce(cl.tenant, '') = $tenant AND NOT cl.id IN $ids
    RETURN cl.id AS id, cl.text AS text
    ORDER BY cl.created_at DESC
    LIMIT $limit
    """
    ids = [c[0] for c in claim_meta]
    recent = session.execute_read(
        lambda tx: tx.run(query, tenant=tkey, ids=ids, limit=settings.graph_contradiction_window).data()
    )
    nli = None
    if settings.graph_nli_contradictions:
        import verification
        nli = verification.nli_probabilities
    pairs = find_contradictions(claim_meta, recent, entities, nli=nli, threshold=settings.verify_threshold)
    if not pairs:
        return
    rows = [{"a": a, "b": b} for a, b in pairs]
    session.execute_write(lambda tx: tx.run(
        """
        UNWIND $rows AS row
        MATCH (a:Claim {id: row.a}), (b:Claim {id: row.b})
        MERGE (a)-[:CONTRADICTS]->(b)
        """,
        rows=rows,
    ).consume())


def delete_source_from_graph(source: str, tenant: str | None = None):
    """Delete one tenant's Chunk nodes for a source and their Claims. Shared
    Entity nodes are left in place (harmless orphans)."""
    q = """
    MATCH (c:Chunk {source:$source})
    WHERE coalesce(c.tenant, '') = $tenant
    OPTIONAL MATCH (c)-[:SUPPORTS]->(cl:Claim)
    DETACH DELETE c, cl
    """
    with get_driver().session() as session:
        session.execute_write(lambda tx: tx.run(q, source=source, tenant=tenant_key(tenant)).consume())


def support_density_for_entities(entity_names: list, tenant: str | None = None):
    q = """
    MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
    WHERE e.name IN $names AND coalesce(c.tenant, '') = $tenant
    RETURN e.name AS name, count(distinct c) AS chunk_count
    """
    with get_driver().session() as session:
        return session.execute_read(
            lambda tx: [
                {"name": r["name"], "chunk_count": r["chunk_count"]}
                for r in tx.run(q, names=entity_names, tenant=tenant_key(tenant))
            ]
        )


def subgraph_for_chunks(chunk_ids: list, max_entities: int = 50, tenant: str | None = None):
    if not chunk_ids:
        return {"chunks": [], "entities": []}
    q = """
    MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
    WHERE c.id IN $chunk_ids AND coalesce(c.tenant, '') = $tenant
    RETURN c.id AS chunk_id, e.name AS entity
    LIMIT $limit
    """
    with get_driver().session() as session:
        pairs = session.execute_read(
            lambda tx: [
                {"chunk_id": r["chunk_id"], "entity": r["entity"]}
                for r in tx.run(q, chunk_ids=chunk_ids, limit=max_entities, tenant=tenant_key(tenant))
            ]
        )
    entities = sorted({p["entity"] for p in pairs})
    return {"chunks": chunk_ids, "entities": entities, "pairs": pairs}


def summarize_reasoning(claims, contra, entities, chunk_scores, paths, relations):
    """Pure post-processing of the reasoner's query rows (unit-testable)."""
    contra_map = {r["id"]: r["contradict_count"] for r in contra}
    for c in claims:
        c["contradict_count"] = int(contra_map.get(c["id"], 0))
    relation_strength = {}
    relation_predicates = {}
    for r in relations:
        key = f"{r.get('src')}|{r.get('predicate')}|{r.get('dst')}"
        relation_strength[key] = relation_strength.get(key, 0) + int(r.get("rel_count", 0))
        sp_key = f"{r.get('src')}|{r.get('dst')}"
        relation_predicates.setdefault(sp_key, set()).add(r.get("predicate"))
    strong_relations = [
        {"relation": k, "count": v}
        for k, v in sorted(relation_strength.items(), key=lambda x: x[1], reverse=True)
        if v >= 2
    ]
    conflict_pairs = [
        {"pair": k, "predicates": sorted(preds)}
        for k, preds in relation_predicates.items()
        if len(preds) > 1
    ]
    evidence_scores = []
    for row in chunk_scores:
        support = int(row.get("support_count", 0))
        contradict = int(row.get("contradict_count", 0))
        entities_count = int(row.get("entity_count", 0))
        score = support + (0.1 * entities_count) - (0.5 * contradict)
        evidence_scores.append({
            "chunk_id": row.get("chunk_id"),
            "support_count": support,
            "contradict_count": contradict,
            "entity_count": entities_count,
            "score": round(score, 3),
        })
    return {
        "claims": claims,
        "entity_density": entities,
        "relations": relations,
        "relation_strength": strong_relations,
        "relation_conflicts": conflict_pairs,
        "evidence_scores": evidence_scores,
        "paths": paths,
    }


def graph_reasoner(chunk_ids: list, max_claims: int = 10, max_entities: int = 10, tenant: str | None = None):
    if not chunk_ids:
        return {"claims": [], "entity_density": []}
    scope = "c.id IN $chunk_ids AND coalesce(c.tenant, '') = $tenant"
    q_claims = f"""
    MATCH (c:Chunk)-[:SUPPORTS]->(cl:Claim)
    WHERE {scope}
    RETURN cl.id AS id, cl.text AS text, count(distinct c) AS support_count
    ORDER BY support_count DESC
    LIMIT $limit
    """
    q_contra = f"""
    MATCH (c:Chunk)-[:SUPPORTS]->(cl:Claim)
    WHERE {scope}
    OPTIONAL MATCH (cl)-[:CONTRADICTS]->(other:Claim)
    RETURN cl.id AS id, count(distinct other) AS contradict_count
    """
    q_entities = f"""
    MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
    WHERE {scope}
    RETURN e.name AS name, count(distinct c) AS chunk_count
    ORDER BY chunk_count DESC
    LIMIT $limit
    """
    q_chunk_scores = f"""
    MATCH (c:Chunk)
    WHERE {scope}
    OPTIONAL MATCH (c)-[:SUPPORTS]->(cl:Claim)
    OPTIONAL MATCH (cl)-[:CONTRADICTS]->(other:Claim)
    OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
    RETURN c.id AS chunk_id,
           count(distinct cl) AS support_count,
           count(distinct other) AS contradict_count,
           count(distinct e) AS entity_count
    """
    q_paths = f"""
    MATCH (c:Chunk)-[:SUPPORTS]->(cl:Claim)-[:MENTIONS]->(e:Entity)
    WHERE {scope}
    MATCH (e)-[r:RELATES]->(e2:Entity)
    WHERE coalesce(r.tenant, '') = $tenant
    RETURN c.id AS chunk_id, e.name AS src, e2.name AS dst, count(*) AS path_count
    ORDER BY path_count DESC
    LIMIT $limit
    """
    q_relations = f"""
    MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)-[r:RELATES]->(e2:Entity)
    WHERE {scope} AND coalesce(r.tenant, '') = $tenant
    RETURN e.name AS src, r.predicate AS predicate, e2.name AS dst, count(r) AS rel_count
    ORDER BY rel_count DESC
    LIMIT $limit
    """
    tkey = tenant_key(tenant)

    def _read(tx):
        params = {"chunk_ids": chunk_ids, "tenant": tkey}
        return summarize_reasoning(
            tx.run(q_claims, limit=max_claims, **params).data(),
            tx.run(q_contra, **params).data(),
            tx.run(q_entities, limit=max_entities, **params).data(),
            tx.run(q_chunk_scores, **params).data(),
            tx.run(q_paths, limit=max_entities, **params).data(),
            tx.run(q_relations, limit=max_entities, **params).data(),
        )

    with get_driver().session() as session:
        return session.execute_read(_read)
