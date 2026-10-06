# RET-C2-131 — Unit tests: the retained MainNode stub and the trust gate.
#
# Node invocations here go through __call__ (`node(state)`) rather than
# execute() directly, so the framework's trust gate is actually exercised: it
# runs inside __call__ before execute(), and calling execute() straight past it
# would leave the negative-authorisation case untested.

import pytest

from src.nodes.main_node import MainNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel


# Positive-control query for the admit test: all lowercase, no '@', no digit
# groups, no two consecutive title-case words — so the framework's input gate
# has no personal data to mask and validated_input stays intact.
_CLEAN_QUERY = "how should staff handle a shelf price that does not match the register " "and stay compliant"


class TestMainNode:
    """The retained stub — not part of the pipeline, but it must stay working."""

    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        # Silence the audit backend at the node-module level (same pattern as
        # tests/unit/test_nodes.py).
        monkeypatch.setattr("src.nodes.main_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        self.node = MainNode()

    def test_success_path(self):
        """Processes valid input and returns SUCCESS.

        Invoked via __call__ (MainNode is ANONYMOUS, so an ANONYMOUS caller
        clears the trust gate), so the full wrapper path is exercised.
        """
        state = {
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
            "validated_input": "test input",
            "node_history": [],
            "error_log": [],
        }
        result = self.node(state)
        assert result["status"] == AgentStatus.SUCCESS
        assert result["result"] is not None

    def test_empty_input(self):
        """Handles empty input gracefully."""
        state = {
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
            "validated_input": "",
            "node_history": [],
            "error_log": [],
        }
        result = self.node(state)
        assert result["status"] == AgentStatus.SUCCESS

    def test_execute_method_signature(self):
        """The node contract is execute(state) — nothing else.

        Overriding a differently-named entry point would leave the framework
        calling a method that does not exist on the concrete class.
        """
        import inspect

        assert hasattr(MainNode, "execute"), "MainNode must implement execute()"

        sig = inspect.signature(MainNode.execute)
        params = list(sig.parameters.keys())
        assert len(params) >= 2, f"execute() must accept (self, state), got params: {params}"
        assert params[1] == "state", f"Second parameter must be 'state', got '{params[1]}'"

        assert (
            "_invoke_impl" not in MainNode.__dict__
        ), "_invoke_impl() must not be defined in MainNode — use execute() instead"


class TestTrustGate:
    """The framework's trust gate runs inside __call__ BEFORE execute() and
    denies a caller whose trust is below the node's required_trust_level.

    PreProcessNode is the only node requiring VERIFIED_EXTERNAL — everything
    else is ANONYMOUS — so it is the correct rejection target. On denial the
    gate RETURNS an error dict rather than raising, so the assertions are on the
    returned dict.
    """

    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.pre_process_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.pre_process_node import PreProcessNode

        self.node = PreProcessNode()

    def test_gate_rejects_untrusted_caller_before_execute(self):
        """An ANONYMOUS caller is denied inside __call__; execute() is skipped.

        Invoked via __call__ (node(state)), NOT node.execute(state), so the gate
        is actually exercised. PreProcessNode requires VERIFIED_EXTERNAL, so the
        caller is rejected with an ERROR dict and execute() never runs —
        therefore no validated_input is produced.
        """
        result = self.node(
            {
                "caller_trust_level": TrustLevel.ANONYMOUS.value,
                "user_input": _CLEAN_QUERY,
                "input_context": {},
                "correlation_id": "trust-denial",
            }
        )
        assert result["status"] == AgentStatus.ERROR.value
        assert any(
            "trust gate" in e.lower() for e in result.get("error_log", [])
        ), f"expected a trust-gate denial, got error_log={result.get('error_log')}"
        # execute() was bypassed by the gate -> no validated_input written.
        assert "validated_input" not in result

    def test_gate_admits_trusted_caller(self):
        """A VERIFIED_EXTERNAL caller clears the gate; execute() runs to SUCCESS."""
        result = self.node(
            {
                "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
                "user_input": _CLEAN_QUERY,
                "input_context": {},
                "correlation_id": "trust-admit",
            }
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["validated_input"] is not None
