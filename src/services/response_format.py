"""AgentCore Platform v1.0"""

# RET-C2-131 — the shape of the staff-facing response document.
#
# The renderer and the output boundary must agree on this shape exactly: the
# boundary re-reads the document's own "Sources:" and "Grounded:" lines and
# compares them against the corpus. Two copies of these constants would drift,
# and a drifted boundary check passes on anything.

from typing import List

SEPARATOR = "=" * 68
SUBSEP = "-" * 68

TITLE = "ESL OPERATIONS & PRICE COMPLIANCE — STAFF GUIDANCE"
NO_SOURCES_LINE = "Sources: (none — answer not grounded in the knowledge base)"
SOURCES_PREFIX = "Sources: "
GROUNDED_YES = "Grounded: YES"
GROUNDED_NO = "Grounded: NO"


def assemble_response(
    answer: str,
    citations: List[str],
    grounded: bool,
    compliance_reminder: str,
    store_id: str = "",
) -> str:
    """Render the staff-facing response document."""
    lines = [SEPARATOR, TITLE, SEPARATOR, ""]
    if store_id:
        lines += [f"Store: {store_id}", ""]
    lines += [answer.strip(), ""]

    if compliance_reminder.strip():
        lines += [SUBSEP, compliance_reminder.strip(), ""]

    lines.append(SUBSEP)
    if citations:
        lines.append(SOURCES_PREFIX + ", ".join(citations))
    else:
        lines.append(NO_SOURCES_LINE)
    lines.append(GROUNDED_YES if grounded else GROUNDED_NO)
    lines.append(SEPARATOR)

    return "\n".join(lines)
