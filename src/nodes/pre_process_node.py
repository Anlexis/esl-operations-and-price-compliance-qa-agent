"""AgentCore Platform v1.0"""

# RET-C2-131 — PreProcessNode
# The node that owns the caller contract: trust enforcement plus validation of
# everything the caller supplied.
#
# Responsibilities:
#   - Require VERIFIED_EXTERNAL trust (declared, enforced before execute() runs)
#   - Reject empty or oversized input early, fail-closed
#   - Accept either a plain natural-language query string or a JSON object with
#     a "query" field and an optional "store_id"
#   - Screen the caller's text for chat-template control markers and directive
#     overrides in its own right, rather than relying on the framework alone —
#     a deployment can put a different front door in front of this agent, and
#     `<<SYS>>` is a marker the framework's screen does not score
#   - Screen the request context channel for credential-shaped values, and lock
#     every caller string that can render into the response to an inert
#     identifier shape
#   - Emit an audit event for every validation decision
#
# Rejections name the FIELD and a closed-set reason. The rejected value is never
# echoed: repeating it puts caller-controlled text into the error path, and a
# credential-shaped value would then be scanned and re-raised by the framework.
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
from src.services.input_guard import (
    REASON_CREDENTIAL,
    REASON_INVALID_IDENTIFIER,
    normalise_identifier,
    screen_credentials,
    screen_payload,
    screen_text,
    validate_context,
)

logger = logging.getLogger(__name__)

# Upper bound on a single staff query (defensive input-size limit).
_MAX_QUERY_CHARS = 4000


def _coerce_query(user_input: str) -> Dict[str, Any]:
    """Coerce the raw user_input into a {query, store_id} dict.

    Accepts a JSON object with a "query" field, or a plain query string.
    The raw store_id is carried through untouched here; it is validated against
    the inert identifier shape by the caller of this helper.
    """
    stripped = user_input.strip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
            if isinstance(payload, dict) and payload.get("query"):
                return {
                    "query": str(payload.get("query", "")).strip(),
                    "store_id": payload.get("store_id"),
                    "parsed": payload,
                }
        except (json.JSONDecodeError, ValueError):
            pass  # fall through: treat as a plain query string
    return {"query": stripped, "store_id": None, "parsed": None}


class PreProcessNode(FunctionNode):
    """Validate the caller-supplied staff query before the domain workflow runs.

    This is the only node that requires VERIFIED_EXTERNAL trust, so an
    unauthenticated caller is refused here and the inner pipeline never sees
    untrusted input directly.

    Input state keys:
        user_input:    str   — caller-supplied query (plain string or JSON)
        input_context: dict  — request context channel (channel, store_id)

    Output state keys (partial dict):
        validated_input:  str        — normalised JSON string {query, store_id}
        enriched_context: str        — JSON-serialised channel metadata
        status:           str        — AgentStatus.SUCCESS or ERROR
        error_log:        list[str]  — set only on ERROR
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def _refuse(self, state: AgentState, reason: str, field: str) -> Dict[str, Any]:
        """Fail closed with a field-naming, value-free reason."""
        logger.warning("PreProcessNode: refused %s on %s", reason, field)
        emit_trace_event(
            "pre_process_validation_failed",
            {"reason": reason, "field": field},
            state,
        )
        return {
            "status": AgentStatus.ERROR.value,
            "error_log": [f"PreProcessNode: {reason} in {field}"],
        }

    def execute(self, state: AgentState) -> Dict[str, Any]:
        user_input = state.get("user_input", "")
        raw_context = state.get("input_context", {})

        # ── Emptiness check ───────────────────────────────────────────────────
        if not user_input or not isinstance(user_input, str) or not user_input.strip():
            return self._refuse(state, "empty_input", "input")

        # ── Size guard ────────────────────────────────────────────────────────
        if len(user_input) > _MAX_QUERY_CHARS:
            return self._refuse(state, "value_too_long", "input")

        # ── Request context channel ───────────────────────────────────────────
        # Unsupported keys are dropped; supported keys must be inert and must
        # not look like a credential.
        accepted_context, context_refusals = validate_context(raw_context)
        if context_refusals:
            reason, field = context_refusals[0]
            return self._refuse(state, reason, f"input_context.{field}")

        # ── Caller text screen ────────────────────────────────────────────────
        coerced = _coerce_query(user_input)
        if not coerced["query"]:
            return self._refuse(state, "empty_input", "input.query")

        if coerced["parsed"] is not None:
            # A JSON request: screen every value AND every key, depth-first.
            found = screen_payload(coerced["parsed"], "input")
            if found:
                return self._refuse(state, found[0], found[1])
        else:
            text_reason = screen_text(coerced["query"])
            if text_reason:
                return self._refuse(state, text_reason, "input")
            if screen_credentials(coerced["query"]):
                return self._refuse(state, REASON_CREDENTIAL, "input")

        # ── Caller identifiers ────────────────────────────────────────────────
        # store_id renders into the response, so it is locked to the inert
        # identifier shape. The context channel wins when both are supplied.
        store_id: Optional[str] = None
        raw_store_id = coerced["store_id"]
        if raw_store_id is not None:
            store_id = normalise_identifier(raw_store_id)
            if store_id is None:
                return self._refuse(state, REASON_INVALID_IDENTIFIER, "input.store_id")
        if accepted_context.get("store_id"):
            store_id = accepted_context["store_id"]

        normalised_json = json.dumps({"query": coerced["query"], "store_id": store_id}, ensure_ascii=False)

        logger.info(
            "PreProcessNode: validated query store_id=%s query_chars=%d",
            store_id or "unknown",
            len(coerced["query"]),
        )
        emit_trace_event(
            "pre_process_validated",
            {"store_id": store_id or "unknown", "query_chars": len(coerced["query"])},
            state,
        )

        return {
            "validated_input": normalised_json,
            "enriched_context": to_json(
                {
                    "source": "ESLOperationsPriceComplianceQAAgent",
                    "channel": accepted_context.get("channel", "unknown"),
                    "store_id": store_id or "unknown",
                }
            ),
            "status": AgentStatus.SUCCESS.value,
        }
