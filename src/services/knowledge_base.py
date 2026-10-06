"""AgentCore Platform v1.0"""

# RET-C2-131 — ESL operations + pricing-governance knowledge base.
#
# The shipped corpus is a small, deterministic in-module set so the template
# runs and can be tested end to end with no vector database attached. Replace
# KB_PASSAGES with your own store's procedures — the passage contract
# (doc_id / source / keywords / text) is what the rest of the pipeline depends
# on, and a real vector index can be wired behind the same contract.
#
# It lives in its own module because two independent consumers read it:
#   - RetrieveNode          — scores passages against the staff query;
#   - the output boundary   — re-checks that every line of guidance actually
#                             shipped is verbatim from a cited passage.
# A gate that read the retrieval node's own working copy could only confirm the
# node agreed with itself.

from typing import Any, Dict, List, Tuple

KB_PASSAGES: List[Dict[str, Any]] = [
    {
        "doc_id": "ESL-OPS-001",
        "source": "ESL Operations Manual §2",
        "keywords": ("update", "refresh", "sync", "push", "price", "label", "更新", "反映"),
        "text": (
            "ESL price update procedure: (1) Confirm the new price is approved in the "
            "pricing master. (2) From the store controller, select the affected SKU and "
            "push the update. (3) Verify the label shows the new price and the sync LED is "
            "green within 60 seconds. (4) If the label does not refresh, re-scan the label "
            "barcode and re-push."
        ),
    },
    {
        "doc_id": "ESL-OPS-014",
        "source": "ESL Operations Manual §5",
        "keywords": ("battery", "low", "alert", "replace", "電池", "バッテリー"),
        "text": (
            "Battery alert protocol: A blinking amber icon indicates low battery. Replace "
            "the CR2450 cell within 7 days. After replacement, trigger a re-pair from the "
            "store controller and confirm the label re-syncs the current price."
        ),
    },
    {
        "doc_id": "PRICE-GOV-003",
        "source": "Pricing Governance SOP §3",
        "keywords": ("price", "discrepancy", "mismatch", "wrong", "shelf", "pos", "価格", "不一致", "値札", "表示価格"),
        "text": (
            "Price-discrepancy resolution: When the shelf (ESL) price differs from the POS "
            "price, the customer is charged the LOWER displayed price for that transaction. "
            "Immediately correct the ESL, log the incident with SKU, store_id and timestamp, "
            "and escalate to the store manager if the gap exceeds 10%."
        ),
    },
    {
        "doc_id": "PRICE-GOV-009",
        "source": "Pricing Governance SOP §7 (景品表示法)",
        "keywords": ("compliance", "景品表示", "二重価格", "表示", "regulation", "price", "discrepancy"),
        "text": (
            "景品表示法 (Act against Unjustifiable Premiums and Misleading Representations) "
            "compliance: a displayed price that misleads customers (including an uncorrected "
            "ESL/POS mismatch or an unsubstantiated struck-through 'was' price) may constitute "
            "an unfair representation. Correct mispriced labels without delay and retain the "
            "correction record for audit."
        ),
    },
    {
        "doc_id": "ESL-OPS-021",
        "source": "ESL Operations Manual §8",
        "keywords": ("offline", "disconnect", "network", "gateway", "troubleshoot", "オフライン"),
        "text": (
            "Offline label troubleshooting: If a label shows the offline icon, check the "
            "aisle gateway power and the controller network status. Labels buffer the last "
            "known price while offline; prices are not guaranteed current until reconnected."
        ),
    },
]

# doc_id -> passage text, for the output boundary's verbatim check.
PASSAGE_TEXT: Dict[str, str] = {entry["doc_id"]: entry["text"] for entry in KB_PASSAGES}

# The fixed sentences the pipeline may emit that are NOT drawn from a passage:
# the abstention answer and the compliance reminder. They are listed here so the
# output boundary can tell "template-owned constant" from "unsourced claim".
NO_GROUNDING_ANSWER = (
    "No matching ESL operations procedure or pricing SOP was found in the knowledge "
    "base for this query. Please rephrase with the specific task (e.g. label update, "
    "battery replacement, price discrepancy) or escalate to the store systems team."
)

COMPLIANCE_REMINDER = (
    "Compliance reminder (景品表示法): a shelf/POS price mismatch left uncorrected "
    "may be an unfair or misleading representation. Charge the lower displayed price "
    "for the current transaction, correct the ESL without delay, and record the "
    "correction (SKU, store_id, timestamp) for audit."
)


def known_doc_ids() -> Tuple[str, ...]:
    """Every doc_id the knowledge base can legitimately cite."""
    return tuple(PASSAGE_TEXT)
