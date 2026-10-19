# app/verification.py
"""Claim-level verification of a synthesized answer.

Each answer sentence is checked against the passages the answer cites, using a
local NLI cross-encoder (premise = cited passage, hypothesis = sentence). A
sentence is ``supported`` when some cited passage entails it, ``contradicted``
when one contradicts it more strongly than any entails it, and ``unsupported``
otherwise. This replaces the self-reported confidence as the basis for the
hallucination-risk score whenever it runs.

Verification is opt-in (``VERIFY_ENABLED``) and best-effort: if the model
cannot load or times out, the answer is returned unverified.
"""
import asyncio
import logging
import re
import threading

from config import settings

_LOG = logging.getLogger(__name__)
_MODEL = None
_LOCK = threading.Lock()
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_PREMISE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+")
_ATTRIBUTION_RE = re.compile(
    r"^(?:according to [^,]{1,60},\s*"
    r"|(?:(?:the\s+)?[A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*){0,3}|he|she|they|it)\s+"
    r"(?:also\s+)?(?:says|said|argues|argued|teaches|taught|writes|wrote|holds|held|believes|believed|"
    r"claims|claimed|notes|noted|observes|observed|explains|explained|maintains|maintained|"
    r"suggests|suggested|advises|advised|insists|insisted|reminds us|counsels|counselled)"
    r"\s+(?:that\s+)?)",
    re.IGNORECASE,
)
MIN_SENTENCE_CHARS = 12
MAX_UNITS_PER_PASSAGE = 12
_WORD_RE = re.compile(r"[a-z][a-z']+")
_STOP = frozenset(
    "the and that this with from have has had are was were been being not but for nor yet you your our "
    "ours thy thee thou its his her their they them she him who whom which what when where why how all any "
    "some such than then there here into onto upon about over under more most very also only just even "
    "will would shall should can could may might must does did doing done one ones".split()
)


def split_sentences(text: str) -> list[str]:
    """Split an answer into sentences; very short fragments join the previous one."""
    parts = [p.strip() for p in _SENTENCE_RE.split((text or "").strip()) if p and p.strip()]
    out: list[str] = []
    for p in parts:
        if out and len(p) < MIN_SENTENCE_CHARS:
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    return out


def strip_attribution(sentence: str) -> str:
    """Drop a leading "X says (that)" / "According to X," frame.

    The cited passage rarely names its own author, so the frame would make a
    faithful sentence look unsupported; the claim itself is what gets checked.
    """
    stripped = _ATTRIBUTION_RE.sub("", sentence.strip(), count=1)
    if len(stripped) < MIN_SENTENCE_CHARS:
        return sentence.strip()
    return stripped[0].upper() + stripped[1:]


def premise_units(passage: str) -> list[str]:
    """Sentences of a passage plus adjacent pairs.

    Small NLI models judge long premises poorly; checking each claim against
    sentence-sized units (and pairs, for claims spanning two sentences) and
    taking the best match is far more reliable.
    """
    sents = [p.strip() for p in _PREMISE_SPLIT_RE.split((passage or "").strip()) if len(p.strip()) >= MIN_SENTENCE_CHARS]
    if not sents:
        return [passage.strip()] if passage and passage.strip() else []
    units = list(sents)
    units += [f"{a} {b}" for a, b in zip(sents, sents[1:])]
    return units[:MAX_UNITS_PER_PASSAGE]


def _content_words(text: str) -> set[str]:
    """Lower-cased content words with a light plural/verb-suffix stem."""
    out = set()
    for w in _WORD_RE.findall(text.lower()):
        if len(w) < 3 or w in _STOP:
            continue
        for suffix in ("ing", "ed", "es", "s"):
            if w.endswith(suffix) and len(w) - len(suffix) >= 4:
                w = w[: -len(suffix)]
                break
        out.add(w)
    return out


def related(hypothesis: str, premise: str) -> bool:
    """True when the two texts share enough content words to contradict.

    Small NLI models often answer "contradiction" for unrelated pairs; a claim
    can only be contradicted by a passage sentence that talks about the same
    things.
    """
    h = _content_words(hypothesis)
    return len(h & _content_words(premise)) >= min(2, len(h))


def is_question(text: str) -> bool:
    """True for a premise unit that ends as a question (closing quotes ignored)."""
    return text.rstrip().rstrip("\"'’”)").endswith("?")


def can_contradict(hypothesis: str, premise: str) -> bool:
    """Whether a premise unit is allowed to contradict a claim.

    It must talk about the same things (``related``) and must not be a
    question: philosophical texts argue with rhetorical questions ("what is
    baser than to chase after pleasure in mourning?") that assert the opposite
    of their literal form, which small NLI models read as contradiction.
    """
    return related(hypothesis, premise) and not is_question(premise)


def source_author(source) -> str | None:
    """Author named by a source file ("seneca_moral_letters.txt" -> "Seneca").

    Only configured author names (domain packs) count, so an unrelated word in
    a file name is never mistaken for an author.
    """
    import domain_packs

    stem = str(source or "").lower().rsplit(".", 1)[0]
    tokens = [t for t in re.split(r"[^a-z]+", stem) if t]
    known = domain_packs.authors()
    for token in tokens:
        if token in known:
            return token.capitalize()
    return None


def get_model():
    global _MODEL
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                from sentence_transformers import CrossEncoder
                _MODEL = CrossEncoder(settings.verify_model_name, revision=settings.verify_model_revision)
    return _MODEL


def _label_index(model) -> dict[str, int]:
    """Map NLI labels to output columns using the model's own config."""
    config = getattr(getattr(model, "model", None), "config", None) or getattr(model, "config", None)
    id2label = getattr(config, "id2label", None) or {}
    index = {str(v).lower(): int(k) for k, v in id2label.items()}
    if {"entailment", "contradiction"} <= set(index):
        return index
    # cross-encoder/nli-* default ordering.
    return {"contradiction": 0, "entailment": 1, "neutral": 2}


def warm_up():
    """Load the model and run one prediction so the first query is not slow."""
    nli_probabilities([("Warm-up premise sentence.", "Warm-up hypothesis sentence.")])


def nli_probabilities(pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
    """[{entailment, contradiction}] per (premise, hypothesis) pair. Blocking."""
    if not pairs:
        return []
    model = get_model()
    labels = _label_index(model)
    scores = model.predict(pairs, apply_softmax=True)
    return [
        {"entailment": float(row[labels["entailment"]]), "contradiction": float(row[labels["contradiction"]])}
        for row in scores
    ]


def classify(sentences, passages, probabilities, threshold: float) -> list[dict]:
    """Per-sentence status from a full sentences x units NLI matrix (row-major).

    ``passages`` is a list of (id, premise text) units, several per passage.
    """
    n = len(passages)
    return [
        classify_one(sentence, passages, probabilities[i * n:(i + 1) * n], threshold)
        for i, sentence in enumerate(sentences)
    ]


def classify_one(sentence, units, probabilities, threshold: float) -> dict:
    """Pure decision step for one sentence over its candidate premise units."""
    best_entail, support_id, support_text, worst_contra = 0.0, None, None, 0.0
    for (pid, text), p in zip(units, probabilities):
        if p["entailment"] > best_entail:
            best_entail, support_id, support_text = p["entailment"], pid, text
        if p["contradiction"] > worst_contra and can_contradict(sentence, text):
            worst_contra = p["contradiction"]
    if best_entail >= threshold:
        status = "supported"
    elif worst_contra >= threshold and worst_contra > best_entail:
        status = "contradicted"
    else:
        status = "unsupported"
    return {
        "text": sentence,
        "status": status,
        "support_id": support_id if status == "supported" else None,
        "support_text": support_text if status == "supported" else None,
        "entailment": round(best_entail, 3),
        "contradiction": round(worst_contra, 3),
    }


def verify_sync(answer: str, evidence: list[dict], provenance: list[dict]) -> dict | None:
    """Verify ``answer`` against the passages its provenance cites."""
    sentences = split_sentences(answer)
    by_id = {e.get("id"): e for e in evidence}
    passages = []
    for p in provenance or []:
        e = by_id.get(p.get("id"))
        if e and e.get("text") and all(e.get("id") != pid for pid, _, _ in passages):
            passages.append((e.get("id"), e["text"], source_author(e.get("source"))))
    if not sentences:
        return None
    if not passages:
        claims = [
            {"text": s, "status": "unsupported", "support_id": None, "support_text": None, "entailment": 0.0, "contradiction": 0.0}
            for s in sentences
        ]
    else:
        units = [(pid, unit, author) for pid, text, author in passages for unit in premise_units(text)]
        unit_words = [_content_words(unit) for _, unit, _ in units]
        hypotheses = [strip_attribution(s) for s in sentences]
        # Only premise units sharing a content word with the claim go to the
        # model: the rest can neither entail nor contradict it, and skipping
        # them cuts the NLI work several-fold.
        candidates = []
        for hyp in hypotheses:
            words = _content_words(hyp)
            candidates.append([u for u, uw in zip(units, unit_words) if words & uw])
        # The model sees who wrote each passage ("Seneca writes: ..."), so a
        # claim such as "Seneca criticizes ..." can be checked against his text.
        pairs = [
            (f"{author} writes: {unit}" if author else unit, hyp)
            for hyp, cands in zip(hypotheses, candidates)
            for _, unit, author in cands
        ]
        probabilities = nli_probabilities(pairs)
        claims, pos = [], 0
        for hyp, original, cands in zip(hypotheses, sentences, candidates):
            plain = [(pid, unit) for pid, unit, _ in cands]
            claim = classify_one(hyp, plain, probabilities[pos:pos + len(cands)], settings.verify_threshold)
            pos += len(cands)
            claim["text"] = original  # report the sentence as written
            claims.append(claim)
    supported = sum(1 for c in claims if c["status"] == "supported")
    return {"claims": claims, "supported_ratio": supported / len(claims)}


async def verify(answer: str, evidence: list[dict], provenance: list[dict]) -> dict | None:
    """Async, time-boxed verification; None when disabled or unavailable."""
    if not settings.verify_enabled or not answer:
        return None
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(verify_sync, answer, evidence, provenance),
            timeout=settings.verify_timeout_s,
        )
    except Exception as e:
        _LOG.warning("verification_failed: %r", e)
        return None
