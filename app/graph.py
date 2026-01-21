# app/graph.py
from neo4j import GraphDatabase
from config import settings

driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))

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

def add_chunk_and_entities(chunk_id: str, text: str, entities: list):
    def _work(tx):
        tx.run("MERGE (c:Chunk {id:$id}) SET c.text=$text", id=chunk_id, text=text)
        for ent in entities:
            tx.run("MERGE (e:Entity {name:$name})", name=ent)
            tx.run("""
                MATCH (c:Chunk {id:$cid}), (e:Entity {name:$name})
                MERGE (c)-[:MENTIONS]->(e)
            """, cid=chunk_id, name=ent)
    with driver.session() as session:
        session.execute_write(lambda tx: _work(tx))

def add_chunk_entities_claims(chunk_id: str, text: str, entities: list, relations: list | None = None):
    claims = _split_sentences(text, settings.graph_max_claims)
    claim_meta = []
    if relations is None:
        relations = _extract_relations_fallback(text, entities)
    def _work(tx):
        tx.run("MERGE (c:Chunk {id:$id}) SET c.text=$text", id=chunk_id, text=text)
        for ent in entities:
            tx.run("MERGE (e:Entity {name:$name})", name=ent)
            tx.run("""
                MATCH (c:Chunk {id:$cid}), (e:Entity {name:$name})
                MERGE (c)-[:MENTIONS]->(e)
            """, cid=chunk_id, name=ent)
        for idx, claim in enumerate(claims):
            claim_id = f"{chunk_id}::claim::{idx}"
            tx.run("MERGE (cl:Claim {id:$id}) SET cl.text=$text", id=claim_id, text=claim)
            tx.run("""
                MATCH (c:Chunk {id:$cid}), (cl:Claim {id:$clid})
                MERGE (c)-[:SUPPORTS]->(cl)
            """, cid=chunk_id, clid=claim_id)
            claim_meta.append((claim_id, claim))
            for ent in entities:
                if ent.lower() in claim.lower():
                    tx.run("""
                        MATCH (cl:Claim {id:$clid}), (e:Entity {name:$name})
                        MERGE (cl)-[:MENTIONS]->(e)
                    """, clid=claim_id, name=ent)
        for a, pred, b in relations:
            tx.run("""
                MATCH (a:Entity {name:$a}), (b:Entity {name:$b})
                MERGE (a)-[:RELATES {predicate:$pred}]->(b)
            """, a=a, b=b, pred=pred)
    with driver.session() as session:
        session.execute_write(lambda tx: _work(tx))
        _add_contradictions(session, claim_meta, entities)

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
            except Exception:
                pass
            token = ""
    if token:
        try:
            nums.append(float(token))
        except Exception:
            pass
    return nums

def _entity_overlap(text_a: str, text_b: str, entities: list):
    if not entities:
        return False
    la = text_a.lower()
    lb = text_b.lower()
    hits = 0
    for ent in entities:
        ent_l = ent.lower()
        if ent_l in la and ent_l in lb:
            hits += 1
        if hits >= 1:
            return True
    return False

def _add_contradictions(session, claim_meta, entities):
    if not claim_meta:
        return
    # best-effort: match recent claims with same core text but opposing negation
    claim_ids = [c[0] for c in claim_meta]
    query = """
    MATCH (cl:Claim)
    WHERE NOT cl.id IN $ids
    RETURN cl.id AS id, cl.text AS text
    ORDER BY cl.id DESC
    LIMIT $limit
    """
    recent = session.execute_read(lambda tx: tx.run(query, ids=claim_ids, limit=settings.graph_contradiction_window).data())
    def _relate(tx, a_id, b_id):
        tx.run("""
            MATCH (a:Claim {id:$a}), (b:Claim {id:$b})
            MERGE (a)-[:CONTRADICTS]->(b)
        """, a=a_id, b=b_id)
    for new_id, new_text in claim_meta:
        new_norm = _normalize(new_text)
        new_neg = _negate(new_text)
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
            if _negate(old_text) != new_neg:
                print(f"[graph] CONTRADICTS {new_id} -> {r.get('id')}")
                session.execute_write(lambda tx: _relate(tx, new_id, r.get("id")))

def support_density_for_entities(entity_names: list):
    q = """
    MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
    WHERE e.name IN $names
    RETURN e.name AS name, count(distinct c) AS chunk_count
    """
    def _read(tx):
        res = tx.run(q, names=entity_names)
        return [{ "name": r["name"], "chunk_count": r["chunk_count"] } for r in res]
    with driver.session() as session:
        return session.execute_read(lambda tx: _read(tx))

def subgraph_for_chunks(chunk_ids: list, max_entities: int = 50):
    if not chunk_ids:
        return {"chunks": [], "entities": []}
    q = """
    MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
    WHERE c.id IN $chunk_ids
    RETURN c.id AS chunk_id, e.name AS entity
    LIMIT $limit
    """
    def _read(tx):
        res = tx.run(q, chunk_ids=chunk_ids, limit=max_entities)
        return [{ "chunk_id": r["chunk_id"], "entity": r["entity"] } for r in res]
    with driver.session() as session:
        pairs = session.execute_read(lambda tx: _read(tx))
    entities = sorted({p["entity"] for p in pairs})
    return {"chunks": chunk_ids, "entities": entities, "pairs": pairs}

def graph_reasoner(chunk_ids: list, max_claims: int = 10, max_entities: int = 10):
    if not chunk_ids:
        return {"claims": [], "entity_density": []}
    q_claims = """
    MATCH (c:Chunk)-[:SUPPORTS]->(cl:Claim)
    WHERE c.id IN $chunk_ids
    RETURN cl.id AS id, cl.text AS text, count(distinct c) AS support_count
    ORDER BY support_count DESC
    LIMIT $limit
    """
    q_contra = """
    MATCH (c:Chunk)-[:SUPPORTS]->(cl:Claim)
    WHERE c.id IN $chunk_ids
    OPTIONAL MATCH (cl)-[:CONTRADICTS]->(other:Claim)
    RETURN cl.id AS id, count(distinct other) AS contradict_count
    """
    q_entities = """
    MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
    WHERE c.id IN $chunk_ids
    RETURN e.name AS name, count(distinct c) AS chunk_count
    ORDER BY chunk_count DESC
    LIMIT $limit
    """
    q_chunk_scores = """
    MATCH (c:Chunk)
    WHERE c.id IN $chunk_ids
    OPTIONAL MATCH (c)-[:SUPPORTS]->(cl:Claim)
    OPTIONAL MATCH (cl)-[:CONTRADICTS]->(other:Claim)
    OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
    RETURN c.id AS chunk_id,
           count(distinct cl) AS support_count,
           count(distinct other) AS contradict_count,
           count(distinct e) AS entity_count
    """
    q_paths = """
    MATCH (c:Chunk)-[:SUPPORTS]->(cl:Claim)-[:MENTIONS]->(e:Entity)
    WHERE c.id IN $chunk_ids
    MATCH (e)-[:RELATES]->(e2:Entity)
    RETURN c.id AS chunk_id, e.name AS src, e2.name AS dst, count(*) AS path_count
    ORDER BY path_count DESC
    LIMIT $limit
    """
    q_relations = """
    MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)-[r:RELATES]->(e2:Entity)
    WHERE c.id IN $chunk_ids
    RETURN e.name AS src, r.predicate AS predicate, e2.name AS dst, count(r) AS rel_count
    ORDER BY rel_count DESC
    LIMIT $limit
    """
    def _read(tx):
        claims = tx.run(q_claims, chunk_ids=chunk_ids, limit=max_claims).data()
        contra = tx.run(q_contra, chunk_ids=chunk_ids).data()
        entities = tx.run(q_entities, chunk_ids=chunk_ids, limit=max_entities).data()
        chunk_scores = tx.run(q_chunk_scores, chunk_ids=chunk_ids).data()
        paths = tx.run(q_paths, chunk_ids=chunk_ids, limit=max_entities).data()
        relations = tx.run(q_relations, chunk_ids=chunk_ids, limit=max_entities).data()
        contra_map = {r["id"]: r["contradict_count"] for r in contra}
        for c in claims:
            c["contradict_count"] = int(contra_map.get(c["id"], 0))
        relation_strength = {}
        relation_predicates = {}
        for r in relations:
            key = f"{r.get('src')}|{r.get('predicate')}|{r.get('dst')}"
            relation_strength[key] = relation_strength.get(key, 0) + int(r.get("rel_count", 0))
            sp_key = f"{r.get('src')}|{r.get('dst')}"
            preds = relation_predicates.get(sp_key, set())
            preds.add(r.get("predicate"))
            relation_predicates[sp_key] = preds
        strong_relations = [
            {"relation": k, "count": v}
            for k, v in sorted(relation_strength.items(), key=lambda x: x[1], reverse=True)
            if v >= 2
        ]
        conflict_pairs = []
        for k, preds in relation_predicates.items():
            if len(preds) > 1:
                conflict_pairs.append({"pair": k, "predicates": sorted(list(preds))})
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
    with driver.session() as session:
        return session.execute_read(lambda tx: _read(tx))
