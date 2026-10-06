"""AgentCore Platform v1.0"""

# RET-C2-131 — RetrieveNode
# Inner pipeline step 2: retrieve candidate passages from the ESL operations
# manual and the pricing-governance SOP for the validated staff query.
#
# The shipped retrieval is deterministic keyword overlap over the in-module
# corpus in src/services/knowledge_base.py, so the template runs end to end with
# no vector database attached. Swap in a real hybrid/vector retriever behind the
# same passage contract (doc_id / text / score / source) and nothing downstream
# changes.
#
# Inner step — ANONYMOUS trust: the caller's trust was enforced once, at the
# entry node that owns the caller contract.
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import from_json, to_json
from src.services.knowledge_base import KB_PASSAGES

logger = logging.getLogger(__name__)

# The corpus this node scores against. Kept as a module attribute so a test can
# assert the node never mutates it.
_KB: List[Dict[str, Any]] = KB_PASSAGES


def _score(query_lc: str, entry: Dict[str, Any], hybrid: bool) -> float:
    """Deterministic keyword-overlap score in [0, 1], with an optional hybrid boost."""
    keywords = entry["keywords"]
    hits = sum(1 for kw in keywords if kw in query_lc)
    base = hits / (len(keywords) ** 0.5) if keywords else 0.0
    # Hybrid boost: exact code/doc_id token match (part-numbers, IDs).
    if hybrid and entry["doc_id"].lower() in query_lc:
        base += 0.5
    return float(round(min(base, 1.0), 4))


class RetrieveNode(FunctionNode):
    """Retrieve candidate ESL/pricing passages for the staff query.

    Inner pipeline step — ANONYMOUS trust (see module comment).

    Input state keys:
        query_data: str  — JSON-serialised validated query dict

    Output state keys (partial dict):
        retrieved_docs: str  — JSON-serialised list of {doc_id, text, score, source}
        status:         str
        error_log:      list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        query_data: Dict[str, Any] = from_json(state.get("query_data"), {})

        query = str(query_data.get("query", "")).strip()
        if not query:
            logger.error("RetrieveNode: query missing in query_data")
            emit_trace_event(
                "retrieve_failed",
                {"reason": "missing_query"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["RetrieveNode: query missing in query_data"],
            }

        query_lc = query.lower()
        hybrid = bool(query_data.get("hybrid", False))

        # Build a fresh scored list (never mutate the module-global corpus).
        retrieved: List[Dict[str, Any]] = []
        for entry in _KB:
            score = _score(query_lc, entry, hybrid)
            if score > 0.0:
                retrieved.append(
                    {
                        "doc_id": entry["doc_id"],
                        "text": entry["text"],
                        "score": score,
                        "source": entry["source"],
                    }
                )

        retrieved.sort(key=lambda d: d["score"], reverse=True)

        logger.info(
            "RetrieveNode: query_type=%s hybrid=%s candidates=%d",
            query_data.get("query_type", "general"),
            hybrid,
            len(retrieved),
        )
        emit_trace_event(
            "retrieve_complete",
            {
                "candidate_count": len(retrieved),
                "hybrid": hybrid,
                "top_doc_id": retrieved[0]["doc_id"] if retrieved else None,
            },
            state,
        )

        return {
            "retrieved_docs": to_json(retrieved),
            "status": AgentStatus.SUCCESS.value,
        }
