"""AgentCore Platform v1.0"""

# RET-C2-131 — InputValidateNode
# Inner pipeline step 1: domain-level classification of the staff query.
#
# Distinct from the entry node, which enforces trust and validates the caller's
# data: this node applies business rules — query classification, ESL/pricing
# intent detection, and price-discrepancy flagging, which drives the 景品表示法
# compliance reminder downstream.
#
# Inner step — ANONYMOUS trust. The entry node already required
# VERIFIED_EXTERNAL, and inner steps must accept ANONYMOUS so the caller's
# InvocationContext passes through the subgraph boundary without being refused
# a second time.
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import json
import logging
from typing import Any, ClassVar, Dict, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import to_json
from src.services.input_guard import normalise_identifier

logger = logging.getLogger(__name__)

# Query-type keyword signals (lowercased match on the query text).
_QUERY_TYPE_SIGNALS = {
    "price_discrepancy": (
        "price discrepancy",
        "price mismatch",
        "wrong price",
        "price error",
        "価格不一致",
        "価格の相違",
        "値札",
        "表示価格",
        "景品表示",
    ),
    "battery_alert": (
        "battery",
        "low battery",
        "battery alert",
        "replace battery",
        "電池",
        "バッテリー",
    ),
    "esl_update": (
        "update",
        "refresh",
        "sync",
        "push price",
        "label update",
        "更新",
        "反映",
        "同期",
    ),
}

# Keywords that raise the price-discrepancy / 景品表示法 compliance flag.
_PRICE_DISCREPANCY_SIGNALS = (
    "price discrepancy",
    "price mismatch",
    "wrong price",
    "price error",
    "different price",
    "not match",
    "mismatch",
    "discrepan",
    "価格不一致",
    "価格の相違",
    "値札",
    "表示価格",
    "景品表示",
    "二重価格",
)


def _classify_query_type(query_lc: str) -> str:
    """Return the coarse query_type for the (lowercased) query text."""
    for qtype, signals in _QUERY_TYPE_SIGNALS.items():
        if any(sig in query_lc for sig in signals):
            return qtype
    return "general"


def _detect_price_discrepancy(query_lc: str) -> bool:
    """True when the query concerns a price mismatch / 景品表示法 topic."""
    return any(sig in query_lc for sig in _PRICE_DISCREPANCY_SIGNALS)


def _needs_hybrid(query: str) -> bool:
    """Enable hybrid search when the query carries codes / part numbers / IDs."""
    return any(ch.isdigit() for ch in query) or "-" in query


class InputValidateNode(FunctionNode):
    """Classify the validated staff query.

    Applies business rules beyond the entry node's structural validation:
    query classification, price-discrepancy detection, and the hybrid-search
    signal for part-number / code queries.

    Inner pipeline step — ANONYMOUS trust (see module comment).

    Input state keys:
        validated_input: str  — normalised JSON string from the entry node.
                                Falls back to user_input inside the subgraph,
                                where the entry node's output arrives as the
                                subgraph's own input.

    Output state keys (partial dict):
        query_data:        str   — JSON-serialised validated query dict
        price_discrepancy: bool  — 景品表示法 compliance flag
        status:            str
        error_log:         list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        raw = state.get("validated_input") or state.get("user_input", "")

        # ── Parse (normalised JSON from the entry node, or plain text) ────────
        # The parse failure reason is a fixed label: an exception message quotes
        # offsets into caller data and belongs nowhere near the error path.
        query = ""
        store_id: Optional[str] = None
        if isinstance(raw, str) and raw.strip().startswith("{"):
            try:
                payload: Dict[str, Any] = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                logger.error("InputValidateNode: validated input is not valid JSON")
                emit_trace_event(
                    "input_validate_failed",
                    {"reason": "malformed_payload"},
                    state,
                )
                return {
                    "status": AgentStatus.ERROR.value,
                    "error_log": ["InputValidateNode: malformed_payload in validated_input"],
                }
            query = str(payload.get("query", "")).strip()
            # Already normalised upstream; re-checked so a bypassed entry node
            # cannot put free text into a field that renders into the response.
            store_id = normalise_identifier(payload.get("store_id"))
        else:
            query = str(raw).strip()

        if not query:
            emit_trace_event(
                "input_validate_failed",
                {"reason": "empty_query"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["InputValidateNode: query is empty"],
            }

        # ── Classify + flag ───────────────────────────────────────────────────
        query_lc = query.lower()
        query_type = _classify_query_type(query_lc)
        price_discrepancy = _detect_price_discrepancy(query_lc) or query_type == "price_discrepancy"
        hybrid = _needs_hybrid(query)

        query_data: Dict[str, Any] = {
            "query": query,
            "store_id": store_id,
            "query_type": query_type,
            "price_discrepancy": price_discrepancy,
            "hybrid": hybrid,
        }

        logger.info(
            "InputValidateNode: query_type=%s price_discrepancy=%s hybrid=%s store_id=%s",
            query_type,
            price_discrepancy,
            hybrid,
            store_id or "unknown",
        )
        emit_trace_event(
            "input_validate_complete",
            {
                "query_type": query_type,
                "price_discrepancy": price_discrepancy,
                "hybrid": hybrid,
            },
            state,
        )

        return {
            "query_data": to_json(query_data),
            "price_discrepancy": price_discrepancy,
            "status": AgentStatus.SUCCESS.value,
        }
