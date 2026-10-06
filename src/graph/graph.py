"""AgentCore Platform v1.0"""

# RET-C2-131 — outer graph (two-layer nested architecture).
#
# Architecture:
#
#   Outer backbone (fixed — do NOT override add_edges()):
#     START -> initialize -> pre_process -> main -> {route} -> post_process
#              -> finalize -> END
#                                        \ (RETRY, up to max_retry)
#                                         -> pre_process
#
#   The `main` slot is a GraphNode subclass (ESLQAGraphNode) that delegates the
#   whole question-answering workflow to DomainWorkflowGraph (the inner graph).
#   Domain complexity lives entirely inside that inner graph; the backbone is
#   never modified.
#
# Directory layout:
#   src/graph/graph.py                 <- outer graph (this file)
#   src/graph/domain_workflow_graph.py <- inner graph (multi-step topology)
#
# Runtime configuration comes from config/config.yaml through
# src/services/runtime_config.py — never from config/agent.yaml, which is the
# static manifest and holds no runtime values. A reader pointed at the manifest
# would not fail; it would find no such key, fall back to its own default, and
# every declared value would be dead while the tests stayed green.
#
# get_output() is intentionally NOT overridden. The caller-facing deliverable is
# the assembled staff-guidance text, including any inline 景品表示法 compliance
# reminder; the output boundary writes it to formatted_output and result, and
# the framework's own envelope already surfaces it. There is no structured
# payload beyond that text, so an override would only add a second place for the
# error path to leak from.

from typing import Any, ClassVar, Dict, Optional

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from src.nodes.post_process_node import PostProcessNode, _security_gate_output
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State
from src.services.runtime_config import retrieval_config


class ESLQAGraphNode(GraphNode):
    """The `main` slot of ESLOperationsPriceComplianceQAAgent.

    Wraps DomainWorkflowGraph, and is called by the backbone after the entry
    node and before the output boundary.

    Contracts:
      get_subgraph()  — instantiate and return DomainWorkflowGraph
      extract_input() — pull the validated query out of outer state
      merge_output()  — map sub_result fields into the outer state delta
      error_strategy  — "propagate": re-raise inner errors (fail fast)
    """

    error_strategy: ClassVar[str] = "propagate"
    propagate_hitl: ClassVar[bool] = False

    def get_subgraph(self) -> Any:
        """Instantiate and return the inner domain workflow graph.

        Imported inside the method to avoid a circular import at module load.
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(config=self._parent_config())

    def extract_input(self, state: AgentState) -> str:
        """Return the string passed into the inner graph's invoke().

        The entry node validates and normalises the raw request and writes the
        result to validated_input. Prefer that; fall back to the raw input only
        where validated_input is absent, which happens when a node is exercised
        directly rather than through the graph.
        """
        return str(state.get("validated_input", state.get("user_input", "")))

    def merge_output(self, state: AgentState, sub_result: Dict[str, Any]) -> Dict[str, Any]:
        """Map the inner graph's output into the outer state delta.

        Returns ONLY changed keys — never the full state. Designed together
        with DomainWorkflowGraph.get_output(): the keys emitted there are
        exactly the keys read here, and the output boundary reads esl_response
        and generated_answer from the result to enforce the output invariants.
        """
        return {
            "esl_response": sub_result.get("esl_response"),
            "generated_answer": sub_result.get("generated_answer"),
            "compliance_reminder": sub_result.get("compliance_reminder"),
            "price_discrepancy": sub_result.get("price_discrepancy", False),
            "status": sub_result.get("status"),
        }

    def _parent_config(self) -> Dict[str, Any]:
        """Forward the declared runtime tuning to the inner graph.

        The framework does not thread a graph's config into node.execute(), so
        the inner graph seeds these values into its own initial state (see
        DomainWorkflowGraph._extra_initial_state) and the ranking step reads
        them from there. That is the only path on which a declared value
        actually reaches a node on a real invocation.
        """
        retrieval = retrieval_config()
        return {"configurable": dict(retrieval)}


class ESLOperationsPriceComplianceQAAgent(AgentBaseGraph):
    """Outer graph for RET-C2-131.

    Domain logic is fully encapsulated in ESLQAGraphNode (the `main` slot),
    which delegates to DomainWorkflowGraph.

    Backbone (fixed):
        START -> initialize -> pre_process -> main -> post_process
                 -> finalize -> END

    register_nodes() is the ONLY override:
      - super().register_nodes() fills initialize and finalize
      - pre_process:  PreProcessNode   (VERIFIED_EXTERNAL — the caller contract)
      - main:         ESLQAGraphNode   (delegates to DomainWorkflowGraph)
      - post_process: PostProcessNode  (the output boundary)

    add_edges() is NOT overridden — backbone wiring belongs to the framework.
    The class name must match the `class:` field in config/agent.yaml exactly.
    """

    @property
    def name(self) -> str:
        """Agent identifier used by the registry."""
        return "ESLOperationsPriceComplianceQAAgent"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Fill all 5 backbone slots.

        super().register_nodes() MUST be called first — it injects the
        framework's default initialize node (session identifiers, trust level)
        and finalize node (response metadata, total time).
        """
        super().register_nodes()  # fills: initialize, finalize

        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = ESLQAGraphNode()
        self._nodes["post_process"] = PostProcessNode()

    # add_edges() is NOT overridden — backbone wiring belongs to the framework.

    @staticmethod
    def _security_gate_output(content: str) -> Optional[str]:
        """The output credential check, exposed on the agent class.

        Delegates to the implementation in post_process_node.py so the runtime
        check (called from inside the boundary node, which cannot reach the
        agent instance) and this marker share exactly one definition. Returns
        the violation label, or None when the output is clean.
        """
        return _security_gate_output(content)


# Alias kept for callers that import the entry point by a generic name.
# The class name matches the `class:` field in config/agent.yaml.
Graph = ESLOperationsPriceComplianceQAAgent
