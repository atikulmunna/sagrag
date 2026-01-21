import spacy
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    return list({ent.text for ent in doc.ents})

def extract_relations(text):
    doc = nlp(text)
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
