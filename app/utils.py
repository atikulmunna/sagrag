import threading

_NLP = None
_LOCK = threading.Lock()


def _nlp():
    global _NLP
    if _NLP is None:
        with _LOCK:
            if _NLP is None:
                import spacy
                _NLP = spacy.load("en_core_web_sm")
    return _NLP


def _entities(doc):
    return list({ent.text for ent in doc.ents})


def _relations(doc):
    relations = []
    for sent in doc.sents:
        root = None
        for token in sent:
            if token.dep_ == "ROOT":
                root = token
                break
        if root is None:
            continue
        subj = None
        obj = None
        for child in root.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                subj = child.text
            if child.dep_ in ("dobj", "attr", "pobj", "dative"):
                obj = child.text
        if subj and obj:
            relations.append((subj, root.lemma_, obj))
    return relations


def extract_entities(text):
    return _entities(_nlp()(text))


def extract_relations(text):
    return _relations(_nlp()(text))


def extract_entities_and_relations(text):
    """Entities and (subject, verb, object) triples from one spaCy parse."""
    doc = _nlp()(text)
    return _entities(doc), _relations(doc)
