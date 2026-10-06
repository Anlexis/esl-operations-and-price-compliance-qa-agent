# RET-C2-131 — runtime configuration.
#
# The failure these tests exist to catch is silent: a reader pointed at the
# wrong file finds no key, falls back to its own default, and every declared
# value is dead while the suite stays green. So they assert the SHIPPED file is
# what gets read, and that a value outside its bounds is refused rather than
# used.

import pytest
import yaml

from src.services import runtime_config
from src.services.runtime_config import (
    _CONFIG_PATH,
    _finite_in_range,
    agent_config,
    load_runtime_config,
    retrieval_config,
)


class TestFiniteInRange:
    @pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", float("nan"), float("inf"), float("-inf")])
    def test_non_finite_is_rejected(self, value):
        assert _finite_in_range(value, 0.0, 1.0) is None

    @pytest.mark.parametrize("value", [True, False, None, "", "abc", [], {}])
    def test_non_numeric_and_bools_are_rejected(self, value):
        assert _finite_in_range(value, 0.0, 1.0) is None

    @pytest.mark.parametrize("value", [-0.1, 1.1, 1e30, -1e30])
    def test_out_of_range_is_rejected(self, value):
        assert _finite_in_range(value, 0.0, 1.0) is None

    @pytest.mark.parametrize("value", [0, 1, 0.5, "0.75"])
    def test_in_range_is_accepted(self, value):
        assert _finite_in_range(value, 0.0, 1.0) == float(value)


class TestShippedConfig:
    def test_the_shipped_file_is_the_file_that_is_read(self):
        assert _CONFIG_PATH.name == "config.yaml"
        assert _CONFIG_PATH.parent.name == "config"
        assert _CONFIG_PATH.exists()
        assert load_runtime_config() == yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))

    def test_declared_retrieval_values_are_the_ones_used(self):
        declared = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))["retrieval"]
        resolved = retrieval_config()
        assert resolved["top_k"] == declared["top_k"]
        assert resolved["score_threshold"] == declared["score_threshold"]
        assert resolved["hybrid_search"] == declared["hybrid_search"]

    def test_graph_config_carries_the_declared_backbone_values(self):
        declared = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
        resolved = agent_config()
        assert resolved["max_retry"] == declared["max_retry"]
        assert resolved["timeout_s"] == declared["timeout_s"]
        assert resolved["retrieval"] == retrieval_config()

    def test_manifest_holds_no_runtime_values(self):
        """The manifest is identity and entry point only.

        A reader that went looking for runtime values there would find nothing
        and silently use its own defaults, which is exactly the failure the
        split is meant to make impossible.
        """
        manifest_path = _CONFIG_PATH.with_name("agent.yaml")
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        assert "agent" not in manifest
        for key in ("max_retry", "timeout_s", "retrieval", "llm"):
            assert key not in manifest


class TestBadConfigDegradesToDocumentedDefaults:
    def _with_config(self, monkeypatch, mapping):
        monkeypatch.setattr(runtime_config, "load_runtime_config", lambda: mapping)

    def test_non_finite_threshold_falls_back(self, monkeypatch):
        self._with_config(monkeypatch, {"retrieval": {"score_threshold": "NaN"}})
        assert retrieval_config()["score_threshold"] == 0.5

    def test_out_of_range_top_k_falls_back(self, monkeypatch):
        self._with_config(monkeypatch, {"retrieval": {"top_k": 0}})
        assert retrieval_config()["top_k"] == 5

    def test_non_boolean_hybrid_flag_falls_back(self, monkeypatch):
        self._with_config(monkeypatch, {"retrieval": {"hybrid_search": "yes"}})
        assert retrieval_config()["hybrid_search"] is True

    def test_unreadable_config_still_yields_documented_values(self, monkeypatch):
        self._with_config(monkeypatch, {})
        assert retrieval_config() == {"top_k": 5, "score_threshold": 0.5, "hybrid_search": True}
        assert agent_config()["max_retry"] == 3

    def test_out_of_range_max_retry_falls_back(self, monkeypatch):
        # The framework refuses a max_retry at or above its own ceiling at
        # compile time, so an out-of-range value must never reach the graph.
        self._with_config(monkeypatch, {"max_retry": 99})
        assert agent_config()["max_retry"] == 3

    def test_compiles_with_the_shipped_config(self):
        from src.graph.graph import ESLOperationsPriceComplianceQAAgent

        agent = ESLOperationsPriceComplianceQAAgent(config=agent_config())
        agent.compile()
        assert agent.config["max_retry"] == 3
