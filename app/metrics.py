import threading

_lock = threading.Lock()
_request_count = {}
_request_latency_ms = {}
_request_latency_buckets = [50, 100, 250, 500, 1000, 2000, 5000]
_request_latency_counts = {}
_author_gap_count = {}
_author_query_count = {}
_retrieval_failure_count = {}
_hallucination_risk_buckets = [0.2, 0.4, 0.6, 0.8, 1.0]
_hallucination_risk_counts = {}

def _key(method: str, path: str, status: int):
    return f"{method}|{path}|{status}"

def record_request(method: str, path: str, status: int, latency_ms: int):
    key = _key(method, path, status)
    with _lock:
        _request_count[key] = _request_count.get(key, 0) + 1
        _request_latency_ms[key] = _request_latency_ms.get(key, 0) + int(latency_ms)
        for b in _request_latency_buckets:
            if latency_ms <= b:
                bkey = f"{key}|{b}"
                _request_latency_counts[bkey] = _request_latency_counts.get(bkey, 0) + 1
        bkey = f"{key}|+Inf"
        _request_latency_counts[bkey] = _request_latency_counts.get(bkey, 0) + 1

def record_author_query(author_terms):
    if not author_terms:
        return
    key = ",".join(sorted({str(t).lower() for t in author_terms if str(t)}))
    if not key:
        return
    with _lock:
        _author_query_count[key] = _author_query_count.get(key, 0) + 1

def record_author_gap(author_terms):
    if not author_terms:
        return
    key = ",".join(sorted({str(t).lower() for t in author_terms if str(t)}))
    if not key:
        return
    with _lock:
        _author_gap_count[key] = _author_gap_count.get(key, 0) + 1

def record_retrieval_failure(tag: str):
    if not tag:
        return
    key = str(tag).strip().lower()
    if not key:
        return
    with _lock:
        _retrieval_failure_count[key] = _retrieval_failure_count.get(key, 0) + 1

def record_hallucination_risk(risk: float):
    try:
        r = float(risk)
    except Exception:
        return
    if r < 0:
        r = 0.0
    if r > 1:
        r = 1.0
    with _lock:
        for b in _hallucination_risk_buckets:
            if r <= b:
                key = f"{b:.1f}"
                _hallucination_risk_counts[key] = _hallucination_risk_counts.get(key, 0) + 1
        _hallucination_risk_counts["+Inf"] = _hallucination_risk_counts.get("+Inf", 0) + 1

def render_prometheus():
    lines = []
    lines.append("# HELP sag_rag_requests_total Total HTTP requests")
    lines.append("# TYPE sag_rag_requests_total counter")
    with _lock:
        for key, count in _request_count.items():
            method, path, status = key.split("|", 2)
            lines.append(
                f'sag_rag_requests_total{{method="{method}",path="{path}",status="{status}"}} {count}'
            )
    lines.append("# HELP sag_rag_request_latency_ms_total Total request latency in ms")
    lines.append("# TYPE sag_rag_request_latency_ms_total counter")
    with _lock:
        for key, total_ms in _request_latency_ms.items():
            method, path, status = key.split("|", 2)
            lines.append(
                f'sag_rag_request_latency_ms_total{{method="{method}",path="{path}",status="{status}"}} {total_ms}'
            )
    lines.append("# HELP sag_rag_request_latency_ms_bucket Request latency histogram buckets")
    lines.append("# TYPE sag_rag_request_latency_ms_bucket histogram")
    with _lock:
        for key, count in _request_latency_counts.items():
            method, path, status, bucket = key.split("|", 3)
            lines.append(
                f'sag_rag_request_latency_ms_bucket{{method="{method}",path="{path}",status="{status}",le="{bucket}"}} {count}'
            )
    lines.append("# HELP sag_rag_author_queries_total Queries with explicit author terms")
    lines.append("# TYPE sag_rag_author_queries_total counter")
    with _lock:
        for key, count in _author_query_count.items():
            lines.append(f'sag_rag_author_queries_total{{author="{key}"}} {count}')
    lines.append("# HELP sag_rag_author_gap_total Author queries with no keyword-matching passages")
    lines.append("# TYPE sag_rag_author_gap_total counter")
    with _lock:
        for key, count in _author_gap_count.items():
            lines.append(f'sag_rag_author_gap_total{{author="{key}"}} {count}')
    lines.append("# HELP sag_rag_retrieval_failures_total Retrieval failure tags")
    lines.append("# TYPE sag_rag_retrieval_failures_total counter")
    with _lock:
        for key, count in _retrieval_failure_count.items():
            lines.append(f'sag_rag_retrieval_failures_total{{tag="{key}"}} {count}')
    lines.append("# HELP sag_rag_hallucination_risk_bucket Hallucination risk buckets")
    lines.append("# TYPE sag_rag_hallucination_risk_bucket histogram")
    with _lock:
        for key, count in _hallucination_risk_counts.items():
            lines.append(f'sag_rag_hallucination_risk_bucket{{le="{key}"}} {count}')
    return "\n".join(lines) + "\n"
