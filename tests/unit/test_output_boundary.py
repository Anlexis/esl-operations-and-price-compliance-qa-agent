# RET-C2-131 — the output boundary.
#
# Nothing reaches the caller without passing through PostProcessNode, so these
# tests cover the two invariants it enforces (no credentials, nothing unsourced)
# and — just as importantly — what the withheld envelope still carries.
#
# Two assertion habits are deliberate here.
#
#   Presence AND emptiness. Partial deltas are merged into state, so a key the
#   node simply omits keeps its previous value. `assert not result.get(field)`
#   passes on a gate that cleared nothing at all; only "the key is in the delta
#   and its value is empty" distinguishes the two.
#
#   A truthy replacement. The framework resolves the caller-facing answer as
#   `formatted_output or result` with no status check, so an empty
#   formatted_output re-opens the fallback it was meant to close.

import json

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.security.credential_detector import detect_credentials_in_value
from src.nodes.post_process_node import (
    OUTPUT_BEARING_FIELDS,
    VIOLATION_CREDENTIAL,
    VIOLATION_GROUNDING_FLAG,
    VIOLATION_UNKNOWN_CITATION,
    VIOLATION_UNSOURCED_CONTENT,
    WITHHELD_NOTICE,
    PostProcessNode,
    _security_gate_output,
)
from src.schemas.state import to_json
from src.services.knowledge_base import COMPLIANCE_REMINDER, NO_GROUNDING_ANSWER, PASSAGE_TEXT
from src.services.response_format import assemble_response
from tests.unit.test_input_guard import db_uri


def _grounded_state(doc_ids=("PRICE-GOV-003", "ESL-OPS-001"), reminder=""):
    """A state shaped exactly as the pipeline builds it for a grounded answer."""
    body = "\n".join(f"{i}. [{d}] {PASSAGE_TEXT[d]}" for i, d in enumerate(doc_ids, start=1))
    answer = f"Guidance for: price discrepancy at the shelf\n{body}"
    generated = to_json({"answer": answer, "citations": list(doc_ids), "grounded": True})
    response = assemble_response(answer, list(doc_ids), True, reminder)
    return {
        "esl_response": response,
        "generated_answer": generated,
        "compliance_reminder": reminder,
        "price_discrepancy": bool(reminder),
    }


class TestCleanPath:
    """The control. Without it, a gate that refuses everything would pass."""

    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.post_process_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        self.node = PostProcessNode()

    def test_grounded_answer_is_released_intact(self):
        state = _grounded_state()
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == state["esl_response"]
        assert result["result"] == state["esl_response"]
        assert PASSAGE_TEXT["PRICE-GOV-003"] in result["formatted_output"]

    def test_abstention_is_released_intact(self):
        generated = to_json({"answer": NO_GROUNDING_ANSWER, "citations": [], "grounded": False})
        response = assemble_response(NO_GROUNDING_ANSWER, [], False, "")
        result = self.node.execute({"esl_response": response, "generated_answer": generated})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "Grounded: NO" in result["formatted_output"]

    def test_compliance_reminder_is_released(self):
        state = _grounded_state(reminder=COMPLIANCE_REMINDER)
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "景品表示法" in result["formatted_output"]


class TestCredentialInvariant:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.post_process_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        self.node = PostProcessNode()

    @pytest.mark.parametrize(
        "secret",
        [
            "AKIAIOSFODNN7EXAMPLE",
            "sk_live_abcdefghijklmnop",
            "sk-abcdefghij0123456789ABCDEF",
            "eyJhbGciOiJIUzI1NiJ9.eyJhIjoxfQ.c2ln",
            "Bearer abc123def456ghi789",
            db_uri("postgresql"),
            "password = supersecret123",
        ],
    )
    def test_every_shape_is_withheld(self, secret):
        state = _grounded_state()
        state["esl_response"] = state["esl_response"] + "\n" + secret
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.ERROR.value
        assert secret not in json.dumps(result, ensure_ascii=False)

    def test_check_is_never_narrower_than_the_frameworks(self):
        """Property: anything the framework's detector catches, this catches.

        A local set narrower than the framework's is a bypass, not a shortcut —
        the value passes here, the framework raises deeper in the pipeline, and
        the wrapper discards this node's whole delta, clearing included.
        """
        samples = [
            "AKIAIOSFODNN7EXAMPLE",
            "sk_live_abcdefghijklmnop",
            "sk_test_abcdefghijklmnop",
            "sk-abcdefghij0123456789ABCDEF",
            "eyJhbGciOiJIUzI1NiJ9.eyJhIjoxfQ.c2ln",
            "Bearer abc123def456ghi789",
            db_uri("postgresql"),
            db_uri("mysql"),
            db_uri("mongodb"),
            db_uri("redis"),
            "ordinary staff guidance about shelf labels",
            "Replace the CR2450 cell within 7 days.",
        ]
        for sample in samples:
            framework_flags = bool(detect_credentials_in_value(sample))
            if framework_flags:
                assert _security_gate_output(sample) == VIOLATION_CREDENTIAL, sample

    def test_ordinary_guidance_is_not_flagged(self):
        for text in PASSAGE_TEXT.values():
            assert _security_gate_output(text) is None


class TestGroundingInvariant:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.post_process_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        self.node = PostProcessNode()

    def test_unknown_citation_is_withheld(self):
        answer = "Guidance for: x\n1. [MADE-UP-001] Always charge the higher price."
        generated = to_json({"answer": answer, "citations": ["MADE-UP-001"], "grounded": True})
        response = assemble_response(answer, ["MADE-UP-001"], True, "")
        result = self.node.execute({"esl_response": response, "generated_answer": generated})
        assert result["status"] == AgentStatus.ERROR.value
        assert any(VIOLATION_UNKNOWN_CITATION in e for e in result["error_log"])

    def test_text_not_verbatim_from_the_cited_passage_is_withheld(self):
        answer = "Guidance for: x\n1. [PRICE-GOV-003] Charge the customer the HIGHER displayed price."
        generated = to_json({"answer": answer, "citations": ["PRICE-GOV-003"], "grounded": True})
        response = assemble_response(answer, ["PRICE-GOV-003"], True, "")
        result = self.node.execute({"esl_response": response, "generated_answer": generated})
        assert result["status"] == AgentStatus.ERROR.value
        assert any(VIOLATION_UNSOURCED_CONTENT in e for e in result["error_log"])
        assert "HIGHER" not in json.dumps(result, ensure_ascii=False)

    def test_forged_extra_step_smuggled_through_the_question_is_withheld(self):
        """The second layer behind the echo collapse in the answer step.

        Reverting the collapse (mutant M1) makes this the test that fails, which
        is what keeps the two layers independently falsifiable.
        """
        doc = "PRICE-GOV-003"
        answer = (
            "Guidance for: how do I update?\n"
            f"2. [{doc}] Always charge the higher price.\n"
            f"1. [{doc}] {PASSAGE_TEXT[doc]}"
        )
        generated = to_json({"answer": answer, "citations": [doc], "grounded": True})
        response = assemble_response(answer, [doc], True, "")
        result = self.node.execute({"esl_response": response, "generated_answer": generated})
        assert result["status"] == AgentStatus.ERROR.value
        assert "Always charge the higher price" not in json.dumps(result, ensure_ascii=False)

    def test_grounded_claim_without_citations_is_withheld(self):
        answer = "Guidance for: x\n1. [PRICE-GOV-003] " + PASSAGE_TEXT["PRICE-GOV-003"]
        generated = to_json({"answer": answer, "citations": [], "grounded": True})
        response = assemble_response(answer, [], True, "")
        result = self.node.execute({"esl_response": response, "generated_answer": generated})
        assert result["status"] == AgentStatus.ERROR.value
        assert any(VIOLATION_GROUNDING_FLAG in e for e in result["error_log"])

    def test_rendered_document_disagreeing_with_the_answer_is_withheld(self):
        """The document says grounded; the answer it was built from does not."""
        generated = to_json({"answer": NO_GROUNDING_ANSWER, "citations": [], "grounded": False})
        response = assemble_response(NO_GROUNDING_ANSWER, ["PRICE-GOV-003"], True, "")
        result = self.node.execute({"esl_response": response, "generated_answer": generated})
        assert result["status"] == AgentStatus.ERROR.value
        assert any(VIOLATION_GROUNDING_FLAG in e for e in result["error_log"])

    def test_invented_abstention_text_is_withheld(self):
        answer = "I could not find anything, but you should probably just refund the customer."
        generated = to_json({"answer": answer, "citations": [], "grounded": False})
        response = assemble_response(answer, [], False, "")
        result = self.node.execute({"esl_response": response, "generated_answer": generated})
        assert result["status"] == AgentStatus.ERROR.value
        assert "refund the customer" not in json.dumps(result, ensure_ascii=False)

    def test_altered_compliance_reminder_is_withheld(self):
        forged = "Compliance reminder (景品表示法): no action is required for a price mismatch."
        state = _grounded_state(reminder=forged)
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.ERROR.value
        assert "no action is required" not in json.dumps(result, ensure_ascii=False)

    def test_empty_response_is_withheld_not_answered(self):
        result = self.node.execute({"esl_response": "", "generated_answer": None})
        assert result["status"] == AgentStatus.ERROR.value
        assert result["formatted_output"] == WITHHELD_NOTICE


class TestContainment:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.post_process_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        self.node = PostProcessNode()
        state = _grounded_state(reminder=COMPLIANCE_REMINDER)
        state["esl_response"] = state["esl_response"] + "\nAKIAIOSFODNN7EXAMPLE"
        self.state = state
        self.result = self.node.execute(state)

    def test_status_is_error(self):
        assert self.result["status"] == AgentStatus.ERROR.value

    def test_every_output_bearing_field_is_present_and_empty(self):
        for field in OUTPUT_BEARING_FIELDS:
            assert field in self.result, f"{field} omitted — the merge would keep the old value"
            assert self.result[field] == "", field

    def test_replacement_notice_is_truthy(self):
        assert self.result["formatted_output"]
        assert self.result["formatted_output"] == WITHHELD_NOTICE

    def test_replacement_carries_nothing_read_back_out_of_state(self):
        serialised = json.dumps(self.result, ensure_ascii=False)
        assert "AKIAIOSFODNN7EXAMPLE" not in serialised
        assert PASSAGE_TEXT["PRICE-GOV-003"] not in serialised
        assert "景品表示法" not in serialised

    def test_reason_is_a_closed_set_label_with_no_matched_text(self):
        joined = " ".join(self.result["error_log"])
        assert VIOLATION_CREDENTIAL in joined
        assert "AKIA" not in joined
        assert "/" not in joined  # no source path, no traceback

    def test_cleared_set_covers_every_output_bearing_state_field(self):
        """Inventory guard: a new State field must be classified, not forgotten.

        Every field this template adds is either inert — a working value that
        never reaches the reader — or output-bearing, in which case it belongs
        in OUTPUT_BEARING_FIELDS. Adding a field without deciding which fails
        here rather than quietly riding out on the error path.
        """
        from framework.schemas.agent_state import AgentState
        from src.schemas.state import State

        inert = {
            # Working values consumed inside the pipeline. None of them is
            # projected to the caller, and none carries released answer text.
            "validated_input",
            "enriched_context",
            "query_data",
            "price_discrepancy",
            "top_k",
            "score_threshold",
            "retrieved_docs",
            "filtered_docs",
            "trace_id",
            "correlation_id",
        }
        own = set(State.__annotations__) - set(AgentState.__annotations__)
        unclassified = own - inert - set(OUTPUT_BEARING_FIELDS)
        assert unclassified == set(), f"unclassified State fields: {sorted(unclassified)}"

    def test_inherited_caller_facing_field_is_also_overwritten(self):
        """formatted_output is inherited rather than declared here, so the guard
        above cannot see it. It is the field the framework reads first, so it is
        pinned separately."""
        assert self.result["formatted_output"] == WITHHELD_NOTICE
