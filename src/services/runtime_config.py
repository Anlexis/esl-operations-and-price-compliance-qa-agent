"""AgentCore Platform v1.0"""

# RET-C2-131 — runtime configuration loader.
#
# config/agent.yaml is the static manifest: identity, entry point, trust level
# and the compile-time `requires` gates. It holds NO runtime values. Everything
# tunable at runtime — max_retry, timeout_s and the `retrieval` block — lives in
# config/config.yaml and is read from here.
#
# One loader, so there is exactly one answer to "where does this value come
# from". A reader pointed at the manifest instead would not fail: it would find
# no such key, fall back to its own default, and every declared value in
# config/config.yaml would be dead while the tests stayed green.
#
# Every numeric read here goes through a finite + bounded parser. `float()`
# accepts "NaN" and "Infinity", and every comparison against NaN is False — a
# NaN score_threshold would silently drop every grounding passage and the agent
# would answer "not grounded" for every question with no error anywhere.

from pathlib import Path
from typing import Any, Dict, Optional

# src/services/runtime_config.py -> parents[2] = repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "config" / "config.yaml"

# Bounds for the declared retrieval tuning. A value outside these bounds is a
# deployment mistake, not a runtime condition: the loader falls back to the
# documented default rather than letting an out-of-range value change grounding.
_TOP_K_MIN, _TOP_K_MAX = 1, 50
_THRESHOLD_MIN, _THRESHOLD_MAX = 0.0, 1.0
_MAX_RETRY_MIN, _MAX_RETRY_MAX = 0, 9
_TIMEOUT_MIN, _TIMEOUT_MAX = 1, 600

# Last-resort values, used only when config/config.yaml cannot be read at all
# (an exotic deployment layout). They mirror the shipped file, so an unreadable
# config degrades to the documented behaviour rather than to an empty mapping.
FALLBACK_RETRIEVAL: Dict[str, Any] = {
    "top_k": 5,
    "score_threshold": 0.5,
    "hybrid_search": True,
}
FALLBACK_AGENT: Dict[str, Any] = {
    "max_retry": 3,
    "timeout_s": 30,
}


def _finite_in_range(value: Any, low: float, high: float) -> Optional[float]:
    """Return value as a finite float inside [low, high], else None.

    Rejects bools (``isinstance(True, int)`` is True in Python), non-numeric
    strings, NaN and +/-Infinity. Fails CLOSED: the caller substitutes the
    documented default rather than proceeding with an unusable number.
    """
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    # NaN fails both comparisons; Infinity fails one. Neither can pass.
    if not (low <= number <= high):
        return None
    return number


def load_runtime_config() -> Dict[str, Any]:
    """Read config/config.yaml. Returns {} only if it is genuinely unreadable."""
    try:
        import yaml

        loaded = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def retrieval_config() -> Dict[str, Any]:
    """The declared `retrieval` block, bounds-checked — never an empty mapping."""
    declared = load_runtime_config().get("retrieval")
    if not isinstance(declared, dict):
        return dict(FALLBACK_RETRIEVAL)

    resolved = dict(FALLBACK_RETRIEVAL)
    top_k = _finite_in_range(declared.get("top_k"), _TOP_K_MIN, _TOP_K_MAX)
    if top_k is not None:
        resolved["top_k"] = int(top_k)
    threshold = _finite_in_range(declared.get("score_threshold"), _THRESHOLD_MIN, _THRESHOLD_MAX)
    if threshold is not None:
        resolved["score_threshold"] = threshold
    hybrid = declared.get("hybrid_search")
    if isinstance(hybrid, bool):
        resolved["hybrid_search"] = hybrid
    return resolved


def agent_config() -> Dict[str, Any]:
    """The graph-level runtime block, plus the retrieval tuning.

    This is what the entry point passes as ``Graph(config=...)``, so the values
    declared in config/config.yaml actually reach the framework backbone
    instead of it falling back to its own built-in defaults. ``max_retry`` is
    read by the backbone's retry route; ``timeout_s`` is the declared per-call
    budget carried on the graph config for the hosting runtime.
    """
    declared = load_runtime_config()
    merged: Dict[str, Any] = dict(FALLBACK_AGENT)

    max_retry = _finite_in_range(declared.get("max_retry"), _MAX_RETRY_MIN, _MAX_RETRY_MAX)
    if max_retry is not None:
        merged["max_retry"] = int(max_retry)
    timeout_s = _finite_in_range(declared.get("timeout_s"), _TIMEOUT_MIN, _TIMEOUT_MAX)
    if timeout_s is not None:
        merged["timeout_s"] = int(timeout_s)

    merged["retrieval"] = retrieval_config()
    return merged
