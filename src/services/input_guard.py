"""AgentCore Platform v1.0"""

# RET-C2-131 — caller-input screening shared by the HTTP entry point and the
# node that owns the caller contract.
#
# Two rules drive everything in this module.
#
# 1. The template owns its guarantees. The framework screens the request text
#    for prompt injection, but a deployment can put this agent behind a
#    different front door, so the node that accepts caller data screens it too
#    and refuses in its own right. Refusals are asserted behaviourally (error
#    status, nothing carried forward), never by matching a message.
#
# 2. Screen the class, not the example. Chat-template control markers are a
#    family: `<|...|>`, `[INST]`, `<<SYS>>`, `<s>`, `<system>`. The framework
#    scores `<|im_start|>` and `[INST]` as high confidence but returns nothing
#    for `<<SYS>>`, so a directive wrapped in that marker reaches the answer
#    path and is echoed back to the reader. This module closes the family.
#
# Every string is screened twice: raw, which catches a marker before any tag
# stripping could remove it, and markup-stripped, which catches a directive
# spliced across inline tags (`ig<b>nore</b> all previous instructions`).
# Structured payloads are walked depth-first including KEYS, because a JSON
# `\u` escape is decoded before this code sees it and a hostile field NAME is
# caller data exactly like a value.
#
# Refusal reasons come from a CLOSED SET of labels. A reason is a location and a
# category, never the matched text — quoting the offending value back would put
# it in the error path we are trying to keep clean.

import re
from typing import Any, Dict, List, Optional, Tuple

from framework.security.credential_detector import detect_credentials_in_value
from framework.security.injection_detector import detect_injection

# Closed set of refusal reasons. Callers surface these verbatim; nothing else.
REASON_CONTROL_TOKEN = "control_token"
REASON_INJECTION = "injection_directive"
REASON_CREDENTIAL = "credential_shape"
REASON_UNSUPPORTED_FIELD = "unsupported_field"
REASON_INVALID_IDENTIFIER = "invalid_identifier"
REASON_TOO_LONG = "value_too_long"

# Chat-template control markers. `<\|...\|>` and `[INST]`/`[SYS]` overlap with the
# framework's own high-confidence set on purpose — this template must not be the
# narrower screen. `<<SYS>>`, `<</SYS>>`, `<s>` and `</s>` are the members the
# framework does not score, and are the reason this list exists.
_CONTROL_TOKEN_RE = re.compile(
    r"<\|[^|>]{0,64}\|>"  # <|im_start|>, <|endoftext|>, ...
    r"|<{2}\s*/?\s*SYS\s*>{2}"  # <<SYS>>, <</SYS>>
    r"|\[/?\s*(?:INST|SYS)\s*\]"  # [INST], [/INST], [SYS]
    r"|</?\s*s\s*>"  # <s>, </s>
    r"|</?\s*(?:system|user|assistant)\s*>",  # <system>, </assistant>, ...
    re.IGNORECASE,
)

# Inline markup, removed for the second pass so a directive split across tags is
# re-assembled before it is screened.
_MARKUP_RE = re.compile(r"<[^<>]{0,200}>")

# Every caller string that renders into the response is locked to this inert
# identifier shape: no whitespace, no markup characters, no directive-capable
# punctuation. Hyphen and underscore are included because store codes use them.
_IDENTIFIER_RE = re.compile(r"^[a-z0-9_-]{1,32}$")

# Field names are caller data too — a name is echoed only when it is itself inert.
_FIELD_NAME_RE = re.compile(r"^[a-z0-9_]{1,32}$")

# The only context keys this agent accepts. Anything else is dropped before the
# graph is invoked: an undeclared key is not merely ignored by a validator, it
# travels on the context channel and is returned verbatim by the framework's
# first node, where the output gate scans it and fails the run opaquely.
SUPPORTED_CONTEXT_KEYS: Tuple[str, ...] = ("channel", "store_id")

_MAX_CONTEXT_VALUE_CHARS = 64


def _high_confidence_injection(text: str) -> bool:
    """True when the framework scores a high-confidence injection finding."""
    return any(f.get("confidence") == "high" for f in detect_injection(text))


def screen_text(value: str) -> Optional[str]:
    """Return a closed-set refusal reason for a caller-supplied string, or None.

    Screened raw and markup-stripped; the union of both passes decides.
    """
    if not isinstance(value, str) or not value:
        return None
    candidates = [value]
    # Removed, not replaced with a space: the point of the second pass is to
    # re-assemble a word a tag was inserted into ("ig<b>nore</b>" -> "ignore").
    stripped = _MARKUP_RE.sub("", value)
    if stripped != value:
        candidates.append(stripped)
    for candidate in candidates:
        if _CONTROL_TOKEN_RE.search(candidate):
            return REASON_CONTROL_TOKEN
        if _high_confidence_injection(candidate):
            return REASON_INJECTION
    return None


def screen_credentials(value: Any) -> bool:
    """True when the framework's own credential detector matches this value.

    Delegating rather than keeping a local pattern set is deliberate: a screen
    narrower than the framework's lets a value through that the framework then
    raises on deeper in the pipeline, where the refusal is opaque and any
    clearing this template did is discarded with the rest of the node's delta.
    """
    return bool(detect_credentials_in_value(value))


def safe_field_label(name: Any, index: int) -> str:
    """A field name safe to put in an error message.

    The name is caller data: echo it only when it is an inert identifier that
    trips no credential pattern of its own. Otherwise report it positionally.
    """
    if isinstance(name, str) and _FIELD_NAME_RE.match(name) and not screen_credentials(name):
        return name
    return f"field #{index}"


def screen_payload(payload: Any, path: str = "") -> Optional[Tuple[str, str]]:
    """Depth-first screen of a parsed payload, keys included.

    Returns ``(reason, location)`` for the first refusal, or None. ``location``
    is a dotted path built only from inert names, so it can be shown to the
    caller.
    """
    if isinstance(payload, dict):
        for index, (key, value) in enumerate(payload.items(), start=1):
            label = safe_field_label(key, index)
            here = f"{path}.{label}" if path else label
            if isinstance(key, str):
                reason = screen_text(key)
                if reason:
                    return reason, here
            found = screen_payload(value, here)
            if found:
                return found
        return None
    if isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            found = screen_payload(item, f"{path}[{index}]")
            if found:
                return found
        return None
    if isinstance(payload, str):
        reason = screen_text(payload)
        if reason:
            return reason, path or "value"
        if screen_credentials(payload):
            return REASON_CREDENTIAL, path or "value"
    return None


def normalise_identifier(raw: Any) -> Optional[str]:
    """Lower-case and validate a caller identifier against the inert shape.

    Returns the normalised value, or None when it does not fit. Callers must
    treat None as a refusal — never as "use the raw value".
    """
    if not isinstance(raw, str):
        return None
    candidate = raw.strip().lower()
    if not _IDENTIFIER_RE.match(candidate):
        return None
    return candidate


def validate_context(
    context: Any,
) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    """Reduce a caller context mapping to the supported, inert subset.

    Returns ``(accepted, refusals)``. Unknown keys are DROPPED rather than
    carried through: an ignored key is still present on the context channel,
    and the framework returns that channel verbatim from its first node.
    ``refusals`` lists ``(reason, location)`` pairs for supported keys whose
    value cannot be accepted — the request cannot succeed with them, so the
    caller gets a reason it can act on instead of an opaque failure later.
    """
    accepted: Dict[str, str] = {}
    refusals: List[Tuple[str, str]] = []
    if not isinstance(context, dict):
        return accepted, refusals

    for index, (key, value) in enumerate(context.items(), start=1):
        label = safe_field_label(key, index)
        if key not in SUPPORTED_CONTEXT_KEYS:
            # Dropped, not refused: an unsupported key is never forwarded, so it
            # cannot reach the channel the framework echoes back.
            continue
        if not isinstance(value, str):
            refusals.append((REASON_INVALID_IDENTIFIER, label))
            continue
        if len(value) > _MAX_CONTEXT_VALUE_CHARS:
            refusals.append((REASON_TOO_LONG, label))
            continue
        if screen_credentials(value):
            refusals.append((REASON_CREDENTIAL, label))
            continue
        reason = screen_text(value)
        if reason:
            refusals.append((reason, label))
            continue
        normalised = normalise_identifier(value)
        if normalised is None:
            refusals.append((REASON_INVALID_IDENTIFIER, label))
            continue
        accepted[key] = normalised
    return accepted, refusals
