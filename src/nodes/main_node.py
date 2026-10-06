"""AgentCore Platform v1.0"""

# RET-C2-131 — MainNode: superseded, retained for import compatibility.
#
# The `main` slot of ESLOperationsPriceComplianceQAAgent is filled by
# ESLQAGraphNode (src/graph/graph.py), which delegates the whole workflow to
# DomainWorkflowGraph. This module is not imported by the graph or by any other
# template code; it remains only so that code referencing this path keeps
# importing. Do not build on it.

from typing import Any, ClassVar, Dict

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

_SUPERSEDED_MSG = (
    "MainNode is superseded by the DomainWorkflowGraph pipeline "
    "(ESLQAGraphNode -> DomainWorkflowGraph). "
    "It is retained for import compatibility only."
)


class MainNode(FunctionNode):
    """Superseded by the pipeline steps in DomainWorkflowGraph."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        # Every execute() emits a domain audit event, this one included.
        emit_trace_event(
            "main_node_superseded_called",
            {
                "warning": "MainNode is superseded and not part of the pipeline",
                "replacement": "ESLQAGraphNode + DomainWorkflowGraph",
            },
            state,
        )
        return {
            "status": AgentStatus.SUCCESS,
            "result": _SUPERSEDED_MSG,
        }
