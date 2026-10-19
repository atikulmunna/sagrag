# app/jsonutil.py
"""Lenient JSON extraction for LLM output."""
import json

_DECODER = json.JSONDecoder()


def safe_json_extract(text):
    """Return the first JSON object found in ``text``, or None.

    Tries the whole string first, then decodes from each ``{`` so prose before
    or after the object (or a second object) does not break parsing.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except ValueError:
        pass
    idx = text.find("{")
    while idx >= 0:
        try:
            value, _ = _DECODER.raw_decode(text, idx)
            if isinstance(value, dict):
                return value
        except ValueError:
            pass
        idx = text.find("{", idx + 1)
    return None


def as_unit_float(value, default=None):
    """Coerce a model-reported score to a float in [0, 1], or ``default``."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if f != f:  # NaN
        return default
    return max(0.0, min(1.0, f))
