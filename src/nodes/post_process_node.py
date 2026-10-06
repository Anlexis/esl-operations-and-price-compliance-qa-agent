"""AgentCore Platform v1.0"""

# RET-C2-131 — PostProcessNode: the output boundary.
#
# Nothing reaches the caller without passing through here. Two independent
# invariants are enforced, each with its own audit event:
#
#   Credentials  — no credential-shaped string may ship. The scan delegates to
#                  the framework's own detector and adds one pattern of its own
#                  (an inline `secret: value` assignment, which the framework
#                  does not score). A template screen NARROWER than the
#                  framework's is a bypass, not a shortcut: the value passes
#                  here, the framework raises deeper in the pipeline, and the
#                  containment this node performed is discarded along with the
#                  rest of the node's delta.
#
#   Grounding    — every line of procedure that ships must be a passage from the
#                  knowledge base, verbatim, cited by an identifier that exists.
#                  This is the promise the response makes with its "Sources:"
#                  and "Grounded:" lines, and it is re-derived here from the
#                  corpus rather than trusted from the assembling step. It is
#                  what catches content that the caller's own text, or a drifted
#                  assembler, tried to pass off as cited guidance.
#
# On a violation the node returns ERROR **and clears every output-bearing
# field**. Returning ERROR alone is not containment: the framework resolves the
# caller-facing answer as `formatted_output or result`, with no status check, so
# an un-cleared result ships inside the error envelope. The replacement notice
# is deliberately truthy — an empty string is falsy and re-opens that fallback —
# and carries nothing read back out of the state being cleared.
#
# Violation reasons are closed-set labels. Never the matched text: the framework
# scans every value this node returns, so quoting a credential into the reason
# would make the gate itself the thing that raises.
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from framework.security.credential_detector import detect_credentials_in_value
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import from_json
from src.services.knowledge_base import (
    COMPLIANCE_REMINDER,
    NO_GROUNDING_ANSWER,
    PASSAGE_TEXT,
)
from src.services.response_format import GROUNDED_YES, NO_SOURCES_LINE

logger = logging.getLogger(__name__)

# Violation labels. Closed set — these strings, and nothing derived from the
# content, are all that leaves this node.
VIOLATION_CREDENTIAL = "credential_pattern"
VIOLATION_UNKNOWN_CITATION = "unknown_citation"
VIOLATION_GROUNDING_FLAG = "grounding_flag_mismatch"
VIOLATION_UNSOURCED_CONTENT = "unsourced_content"

# The one credential shape the framework's detector does not score. Everything
# else is delegated, so this template can never be the narrower screen.
_ASSIGNMENT_RE = re.compile(
    r"(?:password|passwd|secret|api_key|token|access_key|private_key)\s*[:=]\s*\S{8,}",
    re.IGNORECASE,
)

# A numbered procedural line inside the assembled answer: "3. [DOC-ID] text".
_BODY_LINE_RE = re.compile(r"^\d+\.\s*\[([^\]]*)\]\s*(.*)$")

# Every state field that can carry answer text or a payload. The error path
# clears all of them, and test_output_boundary's inventory guard fails if a new
# field is added to State without a decision being recorded here.
OUTPUT_BEARING_FIELDS: Tuple[str, ...] = (
    "result",
    "esl_response",
    "generated_answer",
    "compliance_reminder",
)

# Withheld-answer notice. Truthy on purpose, and content-free: it is built from
# this constant alone, never from the state being cleared.
WITHHELD_NOTICE = (
    "[ANSWER WITHHELD: the response did not pass the output checks and has not "
    "been released. Contact the store systems team with the trace identifier "
    "shown by your client.]"
)


def _security_gate_output(content: str) -> Optional[str]:
    """Return a violation label when the output carries a credential, else None.

    The framework's detector decides first; the local assignment pattern only
    widens the result. Keeping the union in one function means the runtime gate
    and the agent-class marker share exactly one definition of "credential".
    """
    if not isinstance(content, str) or not content:
        return None
    if detect_credentials_in_value(content):
        return VIOLATION_CREDENTIAL
    if _ASSIGNMENT_RE.search(content):
        return VIOLATION_CREDENTIAL
    return None


def _grounding_violation(response: str, generated_answer: Any) -> Optional[str]:
    """Return a violation label when the response is not grounded as it claims.

    Checks, in order: every citation names a real passage; the grounded flag
    agrees with the citations; every numbered line of procedure is a cited
    passage verbatim; the ungrounded answer is the fixed abstention text; and
    the rendered document's own "Sources:"/"Grounded:" lines agree with all of
    the above.
    """
    generated: Dict[str, Any] = from_json(generated_answer, {}) or {}
    if not isinstance(generated, dict):
        return VIOLATION_UNSOURCED_CONTENT

    answer = str(generated.get("answer", ""))
    citations = [str(c) for c in generated.get("citations", []) or []]
    grounded = bool(generated.get("grounded", False))

    for doc_id in citations:
        if doc_id not in PASSAGE_TEXT:
            return VIOLATION_UNKNOWN_CITATION

    if grounded != bool(citations):
        return VIOLATION_GROUNDING_FLAG

    if grounded:
        # First line is the echoed question; the rest must be cited passages.
        lines = answer.split("\n")
        body = [line for line in lines[1:] if line.strip()]
        if not body:
            return VIOLATION_UNSOURCED_CONTENT
        for line in body:
            match = _BODY_LINE_RE.match(line.strip())
            if not match:
                return VIOLATION_UNSOURCED_CONTENT
            doc_id, text = match.group(1).strip(), match.group(2).strip()
            if doc_id not in citations:
                return VIOLATION_UNKNOWN_CITATION
            if text != PASSAGE_TEXT[doc_id].strip():
                return VIOLATION_UNSOURCED_CONTENT
    elif answer.strip() != NO_GROUNDING_ANSWER.strip():
        return VIOLATION_UNSOURCED_CONTENT

    # The rendered document must not claim more than the answer supports.
    if (GROUNDED_YES in response) != grounded:
        return VIOLATION_GROUNDING_FLAG
    if grounded and NO_SOURCES_LINE in response:
        return VIOLATION_GROUNDING_FLAG
    if not grounded and NO_SOURCES_LINE not in response:
        return VIOLATION_GROUNDING_FLAG

    # The compliance reminder is a template-owned constant or absent; anything
    # else in that slot would be unsourced regulatory text.
    reminder_start = response.find("Compliance reminder")
    if reminder_start != -1 and COMPLIANCE_REMINDER not in response:
        return VIOLATION_UNSOURCED_CONTENT

    return None


def _cleared_output_state() -> Dict[str, Any]:
    """Every output-bearing field, present and empty.

    Present matters as much as empty. LangGraph merges partial deltas, so a key
    simply omitted keeps its previous value in state — a gate that "cleared" a
    field by leaving it out would clear nothing at all, while a test asserting
    `not result.get(field)` would still pass.
    """
    return {field: "" for field in OUTPUT_BEARING_FIELDS}


class PostProcessNode(FunctionNode):
    """The output boundary: enforce the output invariants, or withhold.

    Declared ANONYMOUS — trust was enforced once, at the entry node.

    Input state keys:
        esl_response:      str  — rendered response from the domain workflow
        generated_answer:  str  — JSON: {answer, citations, grounded}
        price_discrepancy: bool — 景品表示法 compliance flag

    Output state keys (partial dict):
        formatted_output: str
        result:           str
        status:           str
        error_log:        list[str]  (only on ERROR)
        plus every output-bearing field, cleared, on a violation
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def _withhold(self, state: AgentState, violation: str) -> Dict[str, Any]:
        logger.error("PostProcessNode: output withheld — %s", violation)
        emit_trace_event(
            "post_process_output_withheld",
            {"violation": violation},
            state,
        )
        error_log: List[str] = [f"PostProcessNode: output withheld — {violation}"]
        return {
            **_cleared_output_state(),
            "formatted_output": WITHHELD_NOTICE,
            "status": AgentStatus.ERROR.value,
            "error_log": error_log,
        }

    def execute(self, state: AgentState) -> Dict[str, Any]:
        esl_response: str = state.get("esl_response") or ""
        price_discrepancy: bool = bool(state.get("price_discrepancy"))

        # ── Fallback for an empty answer ──────────────────────────────────────
        if not esl_response.strip():
            logger.warning("PostProcessNode: esl_response is empty — using fallback message")
            return self._withhold(state, VIOLATION_UNSOURCED_CONTENT)

        # ── Credential invariant ──────────────────────────────────────────────
        violation = _security_gate_output(esl_response)
        if violation:
            return self._withhold(state, violation)

        # ── Grounding invariant ───────────────────────────────────────────────
        violation = _grounding_violation(esl_response, state.get("generated_answer"))
        if violation:
            return self._withhold(state, violation)

        logger.info(
            "PostProcessNode: output checks passed — length=%d price_discrepancy=%s",
            len(esl_response),
            price_discrepancy,
        )
        emit_trace_event(
            "post_process_complete",
            {
                "output_length": len(esl_response),
                "price_discrepancy": price_discrepancy,
            },
            state,
        )

        return {
            "formatted_output": esl_response,
            "result": esl_response,
            "status": AgentStatus.SUCCESS.value,
        }
