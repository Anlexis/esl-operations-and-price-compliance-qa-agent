"""AgentCore Platform v1.0"""

# RET-C2-131 — DomainWorkflowGraph (the inner graph).
#
# It encapsulates the ESL operations and price-compliance question-answering
# pipeline:
#
#   START
#     -> input_validate   (InputValidateNode)
#     -> retrieve         (RetrieveNode)
#     -> rerank_filter    (RerankFilterNode)
#     -> generate_answer  (GenerateAnswerNode)
#     -> output_format    (OutputFormatNode)
#     -> END
#
# Called by ESLQAGraphNode.get_subgraph() in graph.py. get_output() shapes the
# dict consumed by merge_output() there.
#
# Structure rules this file keeps:
#   - inherits BaseGraph, for a fully custom topology with no forced backbone
#   - register_nodes() does NOT call super() (it is abstract on BaseGraph)
#   - does NOT register initialize / finalize; those are the outer backbone's
#   - every step declares required_trust_level = TrustLevel.ANONYMOUS
#   - get_output() is designed together with ESLQAGraphNode.merge_output()
#   - every step is constructed with no arguments

from typing import Any, Dict

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import State


class DomainWorkflowGraph(BaseGraph):
    """The inner question-answering workflow for RET-C2-131.

    Inherits BaseGraph directly for a fully custom topology.
    Called by ESLQAGraphNode.get_subgraph() in graph.py.

    Pipeline (linear):
        START
          -> input_validate   (InputValidateNode)
          -> retrieve         (RetrieveNode)
          -> rerank_filter    (RerankFilterNode)
          -> generate_answer  (GenerateAnswerNode)
          -> output_format    (OutputFormatNode)
          -> END

    Every step is a FunctionNode subclass declaring ANONYMOUS trust.
    initialize / finalize are outer backbone concerns — not registered here.
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "ret_c2_131_esl_operations_price_compliance_workflow"

    @property
    def state_schema(self) -> type:
        return State

    # ── Config validation ─────────────────────────────────────────────────────

    def _validate_config(self) -> None:
        """No mandatory config for v1 deterministic RAG inner graph."""
        pass

    # ── Runtime config seeding ────────────────────────────────────────────────

    def _extra_initial_state(self) -> Dict[str, Any]:
        """Seed the declared retrieval tuning into the inner initial state.

        The framework does not thread a graph's ``self.config`` into
        node.execute() — a node signature that accepted one would be
        permanently empty. The outer GraphNode forwards the `retrieval` block
        from config/config.yaml via _parent_config(); here it is copied into the
        inner state, which is the channel the ranking step actually reads on a
        real invocation. Values the loader could not accept were never
        forwarded, so the step falls back to its documented defaults.
        """
        configurable = (self.config or {}).get("configurable", {}) or {}
        seeded: Dict[str, Any] = {}
        raw_top_k = configurable.get("top_k")
        if raw_top_k is not None:
            try:
                seeded["top_k"] = int(raw_top_k)
            except (TypeError, ValueError):
                pass
        raw_threshold = configurable.get("score_threshold")
        if raw_threshold is not None:
            try:
                seeded["score_threshold"] = float(raw_threshold)
            except (TypeError, ValueError):
                pass
        return seeded

    # ── Node registration ─────────────────────────────────────────────────────

    def register_nodes(self) -> None:
        """Register all 5 pipeline steps.

        No super() call — BaseGraph.register_nodes() is abstract.
        Do NOT register initialize or finalize; those belong to the outer
        backbone in graph.py. Every key registered here is referenced in
        add_edges(), and every step is constructed with no arguments.
        """
        self._nodes["input_validate"] = InputValidateNode()
        self._nodes["retrieve"] = RetrieveNode()
        self._nodes["rerank_filter"] = RerankFilterNode()
        self._nodes["generate_answer"] = GenerateAnswerNode()
        self._nodes["output_format"] = OutputFormatNode()

    # ── Edge wiring ───────────────────────────────────────────────────────────

    def add_edges(self) -> None:
        """Wire the linear question-answering topology.

            input_validate -> retrieve -> rerank_filter -> generate_answer
            -> output_format -> END.

        There is no conditional branching, so add_conditional_edges() is not
        used and route() below is never called at runtime.
        """
        self._sg.add_edge(START, "input_validate")
        self._sg.add_edge("input_validate", "retrieve")
        self._sg.add_edge("retrieve", "rerank_filter")
        self._sg.add_edge("rerank_filter", "generate_answer")
        self._sg.add_edge("generate_answer", "output_format")
        self._sg.add_edge("output_format", END)

    # ── Routing ───────────────────────────────────────────────────────────────

    def route(self, state: State) -> str:
        """Conditional routing — required by the BaseGraph contract.

        The topology is linear, so add_conditional_edges() is not used and this
        method is never called at runtime. It is annotated with this graph's own
        State rather than the framework base: the graph engine reads a path
        callable's annotation as its input schema and projects away every field
        the annotation does not declare, so a base-class annotation here would
        silently hide the very fields a future branch would route on.

        Returns END on error so an unexpected call cannot re-enter a step.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return END
        return "output_format"

    # ── Output shape ──────────────────────────────────────────────────────────

    def get_output(self, state: AgentState) -> Dict[str, Any]:
        """Shape the dict returned to the outer graph.

        Received by ESLQAGraphNode.merge_output() in graph.py. The two are
        designed together, so the keys emitted here are exactly the keys read
        there: esl_response, generated_answer, compliance_reminder,
        price_discrepancy and status.

        The rendered response is deliberately surfaced under esl_response only.
        It is not also placed in `result`: the framework resolves the
        caller-facing answer as `formatted_output or result`, so a `result`
        written before the output boundary has run would be reachable from an
        error envelope that the boundary never approved.
        """
        return {
            "esl_response": state.get("esl_response"),
            "generated_answer": state.get("generated_answer"),
            "compliance_reminder": state.get("compliance_reminder"),
            "price_discrepancy": state.get("price_discrepancy", False),
            "status": state.get("status"),
            "node_history": state.get("node_history", []),
            "correlation_id": state.get("correlation_id"),
        }
