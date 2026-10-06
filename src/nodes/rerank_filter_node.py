"""AgentCore Platform v1.0"""

# RET-C2-131 — RerankFilterNode
# Inner pipeline step 3: rerank the retrieved passages and apply the score
# threshold and top_k cutoff, so only the most relevant grounding passages
# reach the answer assembler.
#
# The shipped implementation is a deterministic score-sort plus threshold/top_k
# filter over the retrieval scores. A cross-encoder reranker slots in behind the
# same filter contract.
#
# Runtime tuning (top_k, score_threshold) is declared in config/config.yaml and
# arrives in the inner state, seeded by DomainWorkflowGraph. There is no
# `config` parameter on execute(): the framework calls execute(state) with one
# argument, so a config parameter would be permanently empty and every declared
# value would be silently replaced by this module's defaults.
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

from src.schemas.state import from_json, to_json
from src.services.runtime_config import FALLBACK_RETRIEVAL

logger = logging.getLogger(__name__)

# Used only when the declared block never reached state (an inner graph built
# without config). They mirror the `retrieval` block in config/config.yaml.
_DEFAULT_TOP_K: int = int(FALLBACK_RETRIEVAL["top_k"])
_DEFAULT_SCORE_THRESHOLD: float = float(FALLBACK_RETRIEVAL["score_threshold"])


class RerankFilterNode(FunctionNode):
    """Rerank and threshold/top_k filter the retrieved passages.

    Inner pipeline step — ANONYMOUS trust (see module comment).

    Input state keys:
        retrieved_docs:  str   — JSON-serialised candidate list
        top_k:           int   — declared cap, seeded from config/config.yaml
        score_threshold: float — declared threshold, seeded from config/config.yaml

    Output state keys (partial dict):
        filtered_docs: str  — JSON-serialised surviving passages
        status:        str
        error_log:     list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        retrieved: List[Dict[str, Any]] = from_json(state.get("retrieved_docs"), [])

        # The declared tuning reaches this node through state because the
        # framework threads no runtime config into execute(). Anything the
        # loader could not accept never got seeded, so the default applies.
        raw_top_k = state.get("top_k", _DEFAULT_TOP_K)
        raw_threshold = state.get("score_threshold", _DEFAULT_SCORE_THRESHOLD)
        try:
            top_k = int(raw_top_k)
        except (TypeError, ValueError):
            top_k = _DEFAULT_TOP_K
        try:
            score_threshold = float(raw_threshold)
        except (TypeError, ValueError):
            score_threshold = _DEFAULT_SCORE_THRESHOLD

        if not isinstance(retrieved, list) or not retrieved:
            logger.warning("RerankFilterNode: no retrieved_docs to filter")
            emit_trace_event(
                "rerank_filter_empty",
                {"reason": "no_candidates"},
                state,
            )
            # Not an error — the answer assembler handles the no-grounding case.
            return {
                "filtered_docs": to_json([]),
                "status": AgentStatus.SUCCESS.value,
            }

        # Sort by score desc (fresh list), keep those >= threshold, cap at top_k.
        ranked = sorted(retrieved, key=lambda d: d.get("score", 0.0), reverse=True)
        filtered = [d for d in ranked if float(d.get("score", 0.0)) >= score_threshold][:top_k]

        logger.info(
            "RerankFilterNode: candidates=%d kept=%d (top_k=%d threshold=%.2f)",
            len(retrieved),
            len(filtered),
            top_k,
            score_threshold,
        )
        emit_trace_event(
            "rerank_filter_complete",
            {
                "candidate_count": len(retrieved),
                "kept_count": len(filtered),
                "top_k": top_k,
                "score_threshold": score_threshold,
            },
            state,
        )

        return {
            "filtered_docs": to_json(filtered),
            "status": AgentStatus.SUCCESS.value,
        }
