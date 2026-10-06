"""AgentCore Platform v1.0"""

# State must be a flat TypedDict — never a Pydantic model. Graph checkpoints are
# serialised with msgpack, and Pydantic objects corrupt silently there. Extend
# the framework's AgentState with agent-specific fields only; never put
# credentials, secrets or model objects in it.
#
# RET-C2-131 — ESL Operations & Price Compliance Q&A Agent
# Two layers share this schema: the outer backbone and the inner
# question-answering workflow. The fields below cover both.
#
# Every dict/list-valued field is stored as a JSON-serialised Optional[str],
# using the to_json() / from_json() helpers below at every producer and
# consumer. Typing such a field as a bare dict or list causes msgpack
# serialisation failures at checkpoint time.

import json
from typing import Any, NotRequired, Optional

from framework.schemas.agent_state import AgentState


def to_json(value: Any) -> Optional[str]:
    """Serialize a value to a JSON string for State storage."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Optional[str], default: Any = None) -> Any:
    """Deserialize a JSON string from State storage."""
    if value is None:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


class State(AgentState):
    """Flat TypedDict for RET-C2-131.

    All shared fields (user_input, status, session_id, node_history,
    error_log, hitl_*, ...) are inherited from AgentState.

    formatted_output is NOT re-declared here — it is inherited.
    """

    # ------------------------------------------------------------------
    # Outer layer — written by the entry node
    # ------------------------------------------------------------------

    # Validated and normalised JSON string of the staff query payload.
    # Shape: {"query": str, "store_id": str|None}
    validated_input: NotRequired[Optional[str]]

    # JSON-serialised channel/request metadata.
    # Shape: {"source": str, "channel": str, "store_id": str}
    enriched_context: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Inner layer — the question-answering workflow
    # ------------------------------------------------------------------

    # JSON-serialised validated query dict.
    # Shape: {query (str), store_id (str|None), query_type (str),
    #   price_discrepancy (bool), hybrid (bool)}
    query_data: NotRequired[Optional[str]]

    # True when the staff query concerns a price mismatch, which raises the
    # 景品表示法 compliance reminder in the answer.
    price_discrepancy: NotRequired[Optional[bool]]

    # Retrieval tuning, seeded from config/config.yaml by the inner graph. This
    # is the channel the ranking step actually reads: the framework threads no
    # runtime config into a node's execute(). Absent → documented defaults.
    top_k: NotRequired[Optional[int]]
    score_threshold: NotRequired[Optional[float]]

    # JSON-serialised list of retrieved candidate passages.
    # Each item: {doc_id, text, score, source}
    retrieved_docs: NotRequired[Optional[str]]

    # JSON-serialised list of reranked and thresholded passages (length ≤ top_k).
    filtered_docs: NotRequired[Optional[str]]

    # JSON-serialised grounded answer.
    # Shape: {answer (str), citations (list[str]), grounded (bool)}
    generated_answer: NotRequired[Optional[str]]

    # 景品表示法 (Act against Unjustifiable Premiums and Misleading
    # Representations) compliance reminder text, or "" when not applicable.
    compliance_reminder: NotRequired[Optional[str]]

    # Rendered staff-facing response text, before the output boundary runs.
    esl_response: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Outer layer — written by the output boundary
    # ------------------------------------------------------------------

    # The caller-facing answer, set only after the output boundary approved it.
    # formatted_output (inherited) is set by the same node. Both are cleared,
    # present and empty, when the boundary withholds — an omitted key would
    # leave the previous value in state, because partial deltas are merged.
    result: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Tracing — framework-managed; do NOT write these from node code
    # ------------------------------------------------------------------

    trace_id: NotRequired[Optional[str]]
    correlation_id: NotRequired[Optional[str]]
    # node_history is inherited from AgentState
