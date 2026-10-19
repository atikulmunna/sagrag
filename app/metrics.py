# app/metrics.py
"""In-process Prometheus metrics (text exposition format).

Counters and histograms are kept per process. Histograms emit cumulative
``_bucket`` series plus ``_sum`` and ``_count``; label values are escaped.
Request paths are recorded as route templates (see main.py), so label
cardinality stays bounded.
"""
import threading

_lock = threading.Lock()
_LE_INF = 'le="+Inf"'


def _escape(value) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels(names, values, extra=None) -> str:
    pairs = [f'{n}="{_escape(v)}"' for n, v in zip(names, values)]
    if extra:
        pairs.append(extra)
    return "{" + ",".join(pairs) + "}" if pairs else ""


def _fmt(v) -> str:
    return str(int(v)) if float(v).is_integer() else repr(float(v))


class _Counter:
    def __init__(self, name, help_text, label_names):
        self.name, self.help, self.label_names = name, help_text, tuple(label_names)
        self.values: dict = {}

    def inc(self, labels: tuple, amount=1):
        with _lock:
            self.values[labels] = self.values.get(labels, 0) + amount

    def render(self, lines):
        lines.append(f"# HELP {self.name} {self.help}")
        lines.append(f"# TYPE {self.name} counter")
        with _lock:
            for labels, v in self.values.items():
                lines.append(f"{self.name}{_labels(self.label_names, labels)} {_fmt(v)}")

    def reset(self):
        with _lock:
            self.values.clear()


class _Histogram:
    def __init__(self, name, help_text, buckets, label_names=()):
        self.name, self.help, self.label_names = name, help_text, tuple(label_names)
        self.buckets = list(buckets)
        self.series: dict = {}  # labels -> [bucket counts..., +Inf count, sum]

    def observe(self, labels: tuple, value: float):
        with _lock:
            s = self.series.setdefault(labels, [0] * (len(self.buckets) + 1) + [0.0])
            for i, b in enumerate(self.buckets):
                if value <= b:
                    s[i] += 1
            s[len(self.buckets)] += 1
            s[-1] += value

    def render(self, lines):
        lines.append(f"# HELP {self.name} {self.help}")
        lines.append(f"# TYPE {self.name} histogram")
        with _lock:
            for labels, s in self.series.items():
                for i, b in enumerate(self.buckets):
                    le = f'le="{_fmt(b)}"'
                    lines.append(f"{self.name}_bucket{_labels(self.label_names, labels, le)} {s[i]}")
                count = s[len(self.buckets)]
                lines.append(f"{self.name}_bucket{_labels(self.label_names, labels, _LE_INF)} {count}")
                lines.append(f"{self.name}_sum{_labels(self.label_names, labels)} {_fmt(s[-1])}")
                lines.append(f"{self.name}_count{_labels(self.label_names, labels)} {count}")

    def reset(self):
        with _lock:
            self.series.clear()


_REQUESTS = _Counter("sag_rag_requests_total", "Total HTTP requests", ("method", "path", "status"))
_REQUEST_LATENCY = _Histogram(
    "sag_rag_request_latency_ms", "Request latency in ms",
    [50, 100, 250, 500, 1000, 2000, 5000, 10000, 30000], ("method", "path", "status"),
)
_AUTHOR_QUERIES = _Counter("sag_rag_author_queries_total", "Queries with explicit author terms", ("author",))
_AUTHOR_GAPS = _Counter("sag_rag_author_gap_total", "Author queries with no keyword-matching passages", ("author",))
_RETRIEVAL_FAILURES = _Counter("sag_rag_retrieval_failures_total", "Retrieval failure tags", ("tag",))
_HALLUCINATION_RISK = _Histogram("sag_rag_hallucination_risk", "Hallucination risk per answer", [0.2, 0.4, 0.6, 0.8, 1.0])
_EVIDENCE_COVERAGE = _Histogram("sag_rag_evidence_coverage", "Evidence coverage ratio per answer", [0.25, 0.5, 0.75, 1.0])
_SYNTHESIS = _Counter("sag_rag_synthesis_total", "Synthesis outcomes", ("outcome",))
_CLAIMS = _Counter("sag_rag_claims_total", "Verified answer sentences by status", ("status",))
_SYNTHESIS_LATENCY = _Histogram(
    "sag_rag_synthesis_latency_ms", "Synthesis latency in ms by outcome",
    [100, 250, 500, 1000, 2000, 5000, 10000, 20000], ("outcome",),
)
_ALL = (
    _REQUESTS, _REQUEST_LATENCY, _AUTHOR_QUERIES, _AUTHOR_GAPS, _RETRIEVAL_FAILURES,
    _HALLUCINATION_RISK, _EVIDENCE_COVERAGE, _SYNTHESIS, _SYNTHESIS_LATENCY, _CLAIMS,
)

# Fallback cap so an unexpected label source can never grow memory unbounded.
_MAX_AUTHOR_LABELS = 200


def reset():
    """Clear every metric (tests)."""
    for m in _ALL:
        m.reset()


def _clamp01(value):
    try:
        r = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, r))


def record_request(method: str, path: str, status: int, latency_ms: int):
    labels = (method, path, str(status))
    _REQUESTS.inc(labels)
    _REQUEST_LATENCY.observe(labels, float(latency_ms))


def _author_key(author_terms):
    if not author_terms:
        return None
    key = ",".join(sorted({str(t).lower() for t in author_terms if str(t)}))
    return key or None


def _inc_author(counter: _Counter, author_terms):
    key = _author_key(author_terms)
    if key is None:
        return
    if (key,) not in counter.values and len(counter.values) >= _MAX_AUTHOR_LABELS:
        key = "other"
    counter.inc((key,))


def record_author_query(author_terms):
    _inc_author(_AUTHOR_QUERIES, author_terms)


def record_author_gap(author_terms):
    _inc_author(_AUTHOR_GAPS, author_terms)


def record_retrieval_failure(tag: str):
    key = str(tag or "").strip().lower()
    if key:
        _RETRIEVAL_FAILURES.inc((key,))


def record_hallucination_risk(risk: float):
    r = _clamp01(risk)
    if r is not None:
        _HALLUCINATION_RISK.observe((), r)


def record_evidence_coverage(ratio: float):
    r = _clamp01(ratio)
    if r is not None:
        _EVIDENCE_COVERAGE.observe((), r)


def record_claims(claims):
    """Count verified sentences by status (supported/unsupported/contradicted)."""
    for claim in claims or []:
        status = str(claim.get("status") or "unknown")
        if status in ("supported", "unsupported", "contradicted"):
            _CLAIMS.inc((status,))


def record_synthesis(outcome: str, latency_ms: float):
    key = str(outcome or "").strip().lower() or "unknown"
    try:
        ms = max(0.0, float(latency_ms))
    except (TypeError, ValueError):
        ms = 0.0
    _SYNTHESIS.inc((key,))
    _SYNTHESIS_LATENCY.observe((key,), ms)


def render_prometheus():
    lines: list[str] = []
    for m in _ALL:
        m.render(lines)
    return "\n".join(lines) + "\n"
