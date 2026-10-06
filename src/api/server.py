"""AgentCore Platform v1.0"""

# Standalone HTTP entry point for the agent.
# Entry points are adapters only — no business logic here. When the agent runs
# behind the platform gateway instead, the gateway calls agent.invoke() directly
# and this module is not in the path, which is why every guarantee below is also
# enforced by the node that owns the caller contract.

import os
import secrets
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from shared.secrets import factory as secrets_factory
from src.graph.graph import ESLOperationsPriceComplianceQAAgent
from src.services.input_guard import validate_context
from src.services.runtime_config import agent_config

app = FastAPI(title="Agent")

# The declared runtime values reach the backbone here. Constructing the graph
# with no config would leave every value in config/config.yaml unread, with the
# framework quietly substituting its own built-in defaults.
agent = ESLOperationsPriceComplianceQAAgent(config=agent_config())
agent.compile()
# Replace namespace/agent_name to match the agent's manifest values.
agent.provision_secrets(secrets_factory(namespace="ret-c2-131", agent_name="ESLOperationsPriceComplianceQAAgent"))


class InvokeRequest(BaseModel):
    input: str
    session_id: str = ""
    # Optional request context. Supported keys: `channel`, `store_id`.
    # Anything else is dropped before the graph runs — see the note below.
    input_context: Optional[Dict[str, Any]] = None


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request) -> Dict[str, Any]:
    trust = getattr(request.state, "trust_level", TrustLevel.ANONYMOUS)
    # Caller authentication for the standalone deployment: when
    # INVOKE_AUTH_TOKEN is set on the server environment, a caller that no
    # upstream middleware vouched for (still ANONYMOUS) must present it as a
    # Bearer token, and then runs as VERIFIED_EXTERNAL. Trust established by
    # middleware is never demoted. This adapter is the entry-point auth
    # boundary — a deployment-level caller credential rather than an agent
    # secret, so the secrets provider does not apply here: no invocation
    # context exists yet at this point.
    expected = os.environ.get("INVOKE_AUTH_TOKEN")
    if expected and trust is TrustLevel.ANONYMOUS:
        supplied = request.headers.get("authorization", "")
        # Compare bytes: compare_digest raises TypeError on non-ASCII str input
        # (headers decode as latin-1), which would 500 instead of the generic 401.
        if not secrets.compare_digest(supplied.encode(), f"Bearer {expected}".encode()):
            # Generic body on purpose — do not leak whether the token was absent,
            # malformed, or wrong.
            raise HTTPException(status_code=401, detail="Token is invalid or expired.")
        trust = TrustLevel.VERIFIED_EXTERNAL

    # Reduce the request context to the supported, inert subset BEFORE the graph
    # is invoked. Two things make this the right place for it:
    #   - an unsupported key is not merely ignored downstream. It stays on the
    #     context channel, the framework's first node returns that channel
    #     verbatim inside its own result, and the framework's output scan then
    #     fails the whole run with an error the caller cannot act on;
    #   - a credential-shaped value fails the same way. The request cannot
    #     succeed either way, so refuse it here with the field named.
    # 400 rather than 422: the validation layer owns 422 and answers there with
    # a list of error objects, so reusing it makes client handling ambiguous.
    # The field name is echoed only when it is itself inert; otherwise the
    # refusal is positional. The value is never echoed.
    accepted_context, refusals = validate_context(req.input_context)
    if refusals:
        reason, field = refusals[0]
        raise HTTPException(
            status_code=400,
            detail=f"Rejected input_context.{field}: {reason}",
        )

    with bound_secrets(agent._secrets_provider):
        ctx = InvocationContext(
            session_id=req.session_id or str(uuid4()),
            caller_trust_level=trust,
            caller_id=getattr(request.state, "caller_id", ""),
        )
        envelope: Dict[str, Any] = agent.invoke(req.input, ctx=ctx, input_context=accepted_context)
        return envelope


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "agent": "ESLOperationsPriceComplianceQAAgent"}
