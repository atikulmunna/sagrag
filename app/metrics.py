import threading

_lock = threading.Lock()
_request_count = {}
_request_latency_ms = {}
_request_latency_buckets = [50, 100, 250, 500, 1000, 2000, 5000]
_request_latency_counts = {}

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
    return "\n".join(lines) + "\n"
