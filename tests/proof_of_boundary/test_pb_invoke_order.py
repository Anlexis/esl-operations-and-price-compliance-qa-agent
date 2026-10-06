# PB-6: invoke execution order.
# Verifies that BaseNode.__call__() enforces: trust gate -> node_start ->
# input gate -> execute() -> output gate -> node_complete, for every concrete
# node under src/nodes/.
#
# Also verifies the full backbone invoke order for the outer graph:
#   InitializeNode -> PreProcessNode (pre_process) -> ESLQAGraphNode (main)
#   -> PostProcessNode (post_process) -> FinalizeNode
#
# The invoke uses VERIFIED_EXTERNAL caller trust — the real external path. That
# clears the entry node's trust requirement and still passes through the inner
# ANONYMOUS steps. An INTERNAL context would not represent a real caller, so it
# is deliberately not used.

import importlib
import inspect
import json
import pkgutil
from pathlib import Path

import pytest

# ── Template-specific constants ───────────────────────────────────────────────

# Class name of the node in the `main` backbone slot.
_MAIN_SLOT_NODE = "ESLQAGraphNode"

# A grounded price-compliance query for the backbone invoke test. A plain
# natural-language string (the entry node accepts either a plain query or a
# JSON {query, store_id} object), chosen so the pipeline returns grounded
# passages AND raises the 景品表示法 compliance reminder:
#   - "price discrepancy" / "different price" -> price_discrepancy = True
#   - "price" / "shelf" / "pos" / "label"     -> grounds on PRICE-GOV-003
#   - the SKU digits                          -> hybrid search signalled
#
# deploy/invoke_payload.json["input"] MUST equal this exact string, so the
# deployment's first-invoke check and this test exercise the identical payload.
# test_invoke_payload_matches_pb6 below asserts that equality.
_VALID_PAYLOAD = (
    "There is a price discrepancy: the shelf ESL label shows a different price "
    "than the POS for SKU 4901234567894. How should staff handle this and stay "
    "compliant?"
)

# ─────────────────────────────────────────────────────────────────────────────


def _discover_node_classes() -> list[type]:
    """Import every module under src/nodes/ and collect concrete BaseNode subclasses."""
    from framework.nodes.base_node import BaseNode

    try:
        pkg = importlib.import_module("src.nodes")
    except ImportError:
        return []

    discovered = []
    for _, modname, _ in pkgutil.walk_packages(pkg.__path__, prefix="src.nodes."):
        module = importlib.import_module(modname)
        for attr in vars(module).values():
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseNode)
                and attr is not BaseNode
                and attr.__module__ == modname
                and not inspect.isabstract(attr)
            ):
                discovered.append(attr)
    return discovered


def _patch_domain_emit(monkeypatch):
    """Patch emit_trace_event in every node module (avoids audit-backend calls)."""
    for mod_suffix in (
        "pre_process_node",
        "input_validate_node",
        "retrieve_node",
        "rerank_filter_node",
        "generate_answer_node",
        "output_format_node",
        "post_process_node",
        "main_node",
    ):
        try:
            monkeypatch.setattr(
                f"src.nodes.{mod_suffix}.emit_trace_event",
                lambda *a, **k: None,
            )
        except AttributeError:
            pass  # module not yet imported / no emit symbol; fine


class TestInvokeOrder:
    """__call__ must run trust gate -> node_start -> input gate -> execute()
    -> output gate -> node_complete."""

    def test_call_order_for_every_node(self, monkeypatch):
        node_classes = _discover_node_classes()
        if not node_classes:
            pytest.skip("no concrete BaseNode subclasses found under src/nodes/")

        import framework.nodes.base_node as base_node_module

        failures: list[str] = []
        for node_cls in node_classes:
            order: list[str] = []
            monkeypatch.setattr(
                base_node_module,
                "emit_trace_event",
                lambda event_type, _payload, _state, _o=order: _o.append(f"event:{event_type}"),
            )

            for method_name, label in (
                ("_security_gate_input", "security_gate_input"),
                ("execute", "execute"),
                ("_security_gate_output", "security_gate_output"),
            ):
                original = getattr(node_cls, method_name)

                def spy(self, arg, _o=order, _label=label, _orig=original):
                    _o.append(_label)
                    return _orig(self, arg)

                monkeypatch.setattr(node_cls, method_name, spy)

            instance = node_cls()
            # Caller trust equals the node's required level so the trust gate
            # always passes here; the denial branch is asserted separately in
            # TestTrustGate below.
            state = {
                "caller_trust_level": node_cls.required_trust_level.value,
                "correlation_id": "pb6-invoke-order-test",
            }
            instance(state)

            expected = [
                "event:node_start",
                "security_gate_input",
                "execute",
                "security_gate_output",
                "event:node_complete",
            ]
            if order != expected:
                failures.append(
                    f"{node_cls.__name__}: invoke order violation.\n" f"expected: {expected}\nactual:   {order}"
                )

        assert not failures, "\n\n".join(failures)


class TestTrustGate:
    """The trust gate in BaseNode.__call__ runs BEFORE execute() and denies a
    caller whose trust is below the node's required_trust_level."""

    def test_entry_node_denies_anonymous_caller(self, monkeypatch):
        """PreProcessNode (requires VERIFIED_EXTERNAL) must refuse an ANONYMOUS caller."""
        _patch_domain_emit(monkeypatch)
        from framework.schemas.agent_status import AgentStatus
        from framework.schemas.invocation_context import TrustLevel
        from src.nodes.pre_process_node import PreProcessNode

        node = PreProcessNode()
        result = node(
            {
                "caller_trust_level": TrustLevel.ANONYMOUS.value,
                "user_input": _VALID_PAYLOAD,
                "correlation_id": "pb6-trust-denial",
            }
        )
        assert result["status"] == AgentStatus.ERROR.value
        assert any(
            "trust gate" in e.lower() for e in result.get("error_log", [])
        ), f"expected a trust-gate denial, got error_log={result.get('error_log')}"

    def test_entry_node_admits_verified_external_caller(self, monkeypatch):
        """The same node admits a VERIFIED_EXTERNAL caller and runs execute() to SUCCESS."""
        _patch_domain_emit(monkeypatch)
        from framework.schemas.agent_status import AgentStatus
        from framework.schemas.invocation_context import TrustLevel
        from src.nodes.pre_process_node import PreProcessNode

        node = PreProcessNode()
        result = node(
            {
                "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
                "user_input": _VALID_PAYLOAD,
                "input_context": {},
                "correlation_id": "pb6-trust-admit",
            }
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["validated_input"] is not None


class TestBackboneInvokeOrder:
    """A full Graph().invoke() runs the 5-node backbone in order.

    Backbone order: InitializeNode -> PreProcessNode (pre_process) ->
                    ESLQAGraphNode (main) ->
                    PostProcessNode (post_process) -> FinalizeNode

    Uses VERIFIED_EXTERNAL caller trust — the real external path. An INTERNAL
    context would not represent a real caller, so it is not used.
    """

    def _invoke(self, monkeypatch):
        _patch_domain_emit(monkeypatch)
        from framework.schemas.invocation_context import InvocationContext, TrustLevel
        from src.graph.graph import Graph

        agent = Graph()
        agent.compile()
        ctx = InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)
        return agent.invoke(_VALID_PAYLOAD, ctx=ctx)

    def test_backbone_invoke_succeeds_and_returns_output(self, monkeypatch):
        from framework.schemas.agent_status import AgentStatus

        result = self._invoke(monkeypatch)
        assert result.get("status") == AgentStatus.SUCCESS.value, (
            f"Expected status={AgentStatus.SUCCESS.value!r}, got: {result.get('status')!r}\n"
            f"error_log: {result.get('error_log')}"
        )
        assert result.get("output") is not None, "output must be set after a successful invoke"
        # The assembled staff-guidance response (grounded) must be in the surfaced output.
        assert "ESL OPERATIONS" in result["output"]
        assert "Grounded: YES" in result["output"]

    def test_backbone_node_history_matches_expected_order(self, monkeypatch):
        result = self._invoke(monkeypatch)
        history = result.get("node_history", [])
        assert history == [
            "InitializeNode",
            "PreProcessNode",
            "ESLQAGraphNode",
            "PostProcessNode",
            "FinalizeNode",
        ], f"unexpected backbone node_history: {history}"

    def test_main_slot_is_esl_qa_graph_node(self):
        """The `main` backbone slot must be ESLQAGraphNode, a GraphNode."""
        from framework.nodes.graph_node import GraphNode
        from src.graph.graph import ESLOperationsPriceComplianceQAAgent, ESLQAGraphNode

        agent = ESLOperationsPriceComplianceQAAgent()
        agent.compile()
        main_node = agent._nodes.get("main")
        assert main_node is not None, "main slot must be registered"
        assert isinstance(
            main_node, ESLQAGraphNode
        ), f"main slot must be ESLQAGraphNode, got {type(main_node).__name__}"
        assert isinstance(main_node, GraphNode), "main slot node must subclass GraphNode"
        assert main_node.__class__.__name__ == _MAIN_SLOT_NODE

    def test_invoke_payload_matches_pb6(self):
        """deploy/invoke_payload.json["input"] MUST equal _VALID_PAYLOAD.

        The deployment's first-invoke check posts invoke_payload.json as the
        request body, so it must exercise the same payload this file asserts
        yields SUCCESS.
        """
        repo_root = Path(__file__).resolve().parents[2]
        payload_file = repo_root / "deploy" / "invoke_payload.json"
        assert payload_file.exists(), "deploy/invoke_payload.json is required to deploy"
        body = json.loads(payload_file.read_text())
        assert (
            body.get("input") == _VALID_PAYLOAD
        ), "deploy/invoke_payload.json['input'] must equal the PB-6 _VALID_PAYLOAD"
        # The forwarded payload must be a non-empty staff query string; the
        # entry node accepts a plain query, so no JSON envelope is required.
        assert (
            isinstance(body["input"], str) and body["input"].strip()
        ), "invoke_payload input must be a non-empty query string"
