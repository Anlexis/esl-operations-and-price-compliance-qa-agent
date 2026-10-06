"""AgentCore Platform v1.0"""

# RET-C2-131 — GenerateAnswerNode
# Inner pipeline step 4: assemble the grounded operational answer.
#
# The answer is assembled deterministically and strictly from the passages the
# filter kept: every procedural line is a verbatim passage, and when nothing was
# retrieved the node says so rather than producing text of its own. No language
# model is called, so nothing can be invented. The output boundary re-checks
# this independently — see src/nodes/post_process_node.py.
#
# When the query was flagged as a price discrepancy, the 景品表示法 compliance
# reminder is attached to the answer.
#
# The staff question is echoed back as a header line so the reader can see what
# was asked. It is caller text, so it is collapsed to a single line and length
# capped before it is rendered: an un-normalised echo can carry newlines, and a
# newline inside the answer block is indistinguishable from a numbered
# procedural step attributed to a real document.
#
# Inner step — ANONYMOUS trust.
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
import re
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import from_json, to_json
from src.services.knowledge_base import COMPLIANCE_REMINDER, NO_GROUNDING_ANSWER

logger = logging.getLogger(__name__)

# Longest echoed question. Anything past this is truncated with an ellipsis —
# the reader gets the gist, the renderer gets a bounded string.
_MAX_ECHO_CHARS = 200

_WHITESPACE_RUN = re.compile(r"\s+")

# Square brackets are the visual signature of a citation in this document, so
# they are softened to parentheses inside the echo. A question rarely needs
# them, and text that keeps them there reads as a source attribution.
_ECHO_SUBSTITUTIONS = {"[": "(", "]": ")", '"': "'"}


def _echo_query(query: str) -> str:
    """Render the caller's question as one bounded, quoted, single-line string.

    Three things happen here, and each closes the same hole from a different
    side. Whitespace collapses, so the question cannot become extra lines inside
    the answer block. Citation brackets soften, so it cannot wear the costume of
    a sourced step. And the result is quoted, so a reader can see where the
    caller's words end and the guidance begins.
    """
    collapsed = _WHITESPACE_RUN.sub(" ", query).strip()
    for old, new in _ECHO_SUBSTITUTIONS.items():
        collapsed = collapsed.replace(old, new)
    if len(collapsed) > _MAX_ECHO_CHARS:
        collapsed = collapsed[: _MAX_ECHO_CHARS - 1].rstrip() + "…"
    return f'"{collapsed}"'


def _synthesize_answer(query: str, passages: List[Dict[str, Any]]) -> str:
    """Compose the grounded, step-by-step answer from the retrieved passages.

    Each numbered line is the cited passage verbatim — nothing is paraphrased,
    summarised or added, so the text can be checked against the corpus later.
    """
    lead = f"Guidance for: {_echo_query(query)}\n"
    body_lines: List[str] = []
    for i, p in enumerate(passages, start=1):
        body_lines.append(f"{i}. [{p.get('doc_id', 'N/A')}] {p.get('text', '').strip()}")
    return lead + "\n".join(body_lines)


class GenerateAnswerNode(FunctionNode):
    """Assemble the grounded ESL operations answer.

    Inner pipeline step — ANONYMOUS trust (see module comment).

    Input state keys:
        filtered_docs:     str   — JSON-serialised grounding passages
        query_data:        str   — JSON-serialised validated query dict
        price_discrepancy: bool  — 景品表示法 compliance flag

    Output state keys (partial dict):
        generated_answer:    str   — JSON: {answer, citations, grounded}
        compliance_reminder: str   — 景品表示法 reminder text, or "" if not applicable
        price_discrepancy:   bool  — passthrough of the compliance flag
        status:              str
        error_log:           list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        query_data: Dict[str, Any] = from_json(state.get("query_data"), {})
        passages: List[Dict[str, Any]] = from_json(state.get("filtered_docs"), [])
        price_discrepancy = bool(state.get("price_discrepancy", query_data.get("price_discrepancy", False)))

        query = str(query_data.get("query", "")).strip()

        # Grounded assembly — no text of our own when grounding is absent.
        if passages:
            answer_text = _synthesize_answer(query, passages)
            citations = [p.get("doc_id") for p in passages if p.get("doc_id")]
            grounded = True
        else:
            answer_text = NO_GROUNDING_ANSWER
            citations = []
            grounded = False

        compliance_reminder = COMPLIANCE_REMINDER if price_discrepancy else ""

        generated_answer: Dict[str, Any] = {
            "answer": answer_text,
            "citations": citations,
            "grounded": grounded,
        }

        logger.info(
            "GenerateAnswerNode: grounded=%s citations=%d price_discrepancy=%s",
            grounded,
            len(citations),
            price_discrepancy,
        )
        emit_trace_event(
            "generate_answer_complete",
            {
                "grounded": grounded,
                "citation_count": len(citations),
                "price_discrepancy": price_discrepancy,
            },
            state,
        )

        return {
            "generated_answer": to_json(generated_answer),
            "compliance_reminder": compliance_reminder,
            "price_discrepancy": price_discrepancy,
            "status": AgentStatus.SUCCESS.value,
        }
