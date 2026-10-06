"""AgentCore Platform v1.0"""

# RET-C2-131 — OutputFormatNode (inner pipeline step 5, last inside the domain
# workflow). Renders the staff-facing response from the assembled answer, its
# citations, and the 景品表示法 compliance reminder. The string it writes to
# esl_response is what the output boundary gates before anything reaches the
# caller.
#
# The document shape itself lives in src/services/response_format.py, shared
# with the output boundary so the renderer and the check cannot drift apart.
#
# Inner step — ANONYMOUS trust.
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import from_json
from src.services.response_format import assemble_response

logger = logging.getLogger(__name__)


class OutputFormatNode(FunctionNode):
    """Render the staff-facing response document.

    Inner pipeline step — ANONYMOUS trust (see module comment).

    Input state keys:
        generated_answer:    str  — JSON: {answer, citations, grounded}
        compliance_reminder: str  — 景品表示法 reminder text, or ""
        query_data:          str  — JSON: carries the validated store identifier

    Output state keys (partial dict):
        esl_response: str
        status:       str
        error_log:    list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        generated: Dict[str, Any] = from_json(state.get("generated_answer"), {})
        compliance_reminder: str = state.get("compliance_reminder") or ""
        query_data: Dict[str, Any] = from_json(state.get("query_data"), {})
        # Already normalised to the inert identifier shape upstream; "" when the
        # caller supplied none.
        store_id = str(query_data.get("store_id") or "")

        if not generated or not str(generated.get("answer", "")).strip():
            logger.error("OutputFormatNode: generated_answer missing/empty in state")
            emit_trace_event(
                "output_format_failed",
                {"reason": "missing_generated_answer"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["OutputFormatNode: generated_answer missing/empty in state"],
            }

        answer = str(generated.get("answer", ""))
        citations: List[str] = [str(c) for c in generated.get("citations", [])]
        grounded = bool(generated.get("grounded", False))

        response = assemble_response(answer, citations, grounded, compliance_reminder, store_id)

        logger.info(
            "OutputFormatNode: response_chars=%d citations=%d grounded=%s reminder=%s",
            len(response),
            len(citations),
            grounded,
            bool(compliance_reminder.strip()),
        )
        emit_trace_event(
            "output_format_complete",
            {
                "response_length": len(response),
                "citation_count": len(citations),
                "grounded": grounded,
                "has_compliance_reminder": bool(compliance_reminder.strip()),
            },
            state,
        )

        return {
            "esl_response": response,
            "status": AgentStatus.SUCCESS.value,
        }
