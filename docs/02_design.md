# Template Design Specification — RET-C2-131 ESL Operations & Price Compliance Q&A Agent

## Position in the framework architecture

| Aspect | Value |
|---|---|
| Agent class | `ESLOperationsPriceComplianceQAAgent` (alias `Graph`) |
| L1 Base (framework base class) | `AgentBaseGraph` — direct framework inheritance |
| Inner graph base class | `BaseGraph` — `DomainWorkflowGraph` |
| Pattern | Cat 2, two-layer nested: a fixed outer 5-slot backbone with a `GraphNode` in the `main` slot wrapping an inner `BaseGraph` workflow |

**Three-layer separation**

- State: flat TypedDict composition (never a Pydantic model — checkpoints are msgpack, which
  corrupts Pydantic objects silently); dict/list fields are JSON-serialised strings
- Node: framework inheritance, overriding `execute(self, state: dict) -> dict` only
- Graph: composition — `register_nodes()` for substitution, the inner workflow via `GraphNode`

## Domain context

A question-answering assistant for retail store staff operating electronic shelf labels (ESL).
Staff ask about label update procedures, battery-alert protocols, price-discrepancy resolution,
and 景品表示法 (Act against Unjustifiable Premiums and Misleading Representations) compliance. The
agent retrieves the relevant procedure from the ESL operations manual and the pricing-governance
SOP, and returns grounded step-by-step guidance with citations — plus a compliance reminder when a
price discrepancy is in scope.

**Input**: a natural-language staff query, plain or as `{"query": ..., "store_id": ...}`, with an
optional request context carrying `channel` and `store_id`.
**Output**: a step-by-step procedure grounded in the knowledge base, with citations and, when
applicable, the 景品表示法 compliance reminder.

## Architecture overview

### Outer backbone (fixed 5-node pipeline)

```
START → initialize → pre_process → main(GraphNode) → post_process → finalize → END
                                         ↓ (retry, up to max_retry)
                                       pre_process
```

### Inner workflow (linear 5-step pipeline)

```
START → input_validate → retrieve → rerank_filter → generate_answer
          → output_format → END
```

### Node configuration

| Node | Class | File | Trust | Responsibility | Input Keys | Output Keys |
|------|-------|------|-------|---------------|------------|-------------|
| initialize | InitializeNode | framework | — | session init | — | session_id, schema_version |
| pre_process | PreProcessNode | src/nodes/pre_process_node.py | VERIFIED_EXTERNAL | trust + caller-data validation and screening | user_input, input_context | validated_input, enriched_context |
| main | ESLQAGraphNode | src/graph/graph.py | — | delegates to DomainWorkflowGraph | validated_input | esl_response, generated_answer, compliance_reminder, price_discrepancy |
| post_process | PostProcessNode | src/nodes/post_process_node.py | ANONYMOUS | output boundary: credential and grounding invariants | esl_response, generated_answer | formatted_output, result |
| finalize | FinalizeNode | framework | — | response metadata | — | response_metadata, total_time_ms |
| input_validate (inner) | InputValidateNode | src/nodes/input_validate_node.py | ANONYMOUS | query classification + price-discrepancy flag | validated_input | query_data, price_discrepancy |
| retrieve (inner) | RetrieveNode | src/nodes/retrieve_node.py | ANONYMOUS | knowledge-base retrieval (deterministic; hybrid for codes) | query_data | retrieved_docs |
| rerank_filter (inner) | RerankFilterNode | src/nodes/rerank_filter_node.py | ANONYMOUS | score sort + score_threshold + top_k cutoff | retrieved_docs, top_k, score_threshold | filtered_docs |
| generate_answer (inner) | GenerateAnswerNode | src/nodes/generate_answer_node.py | ANONYMOUS | grounded answer + citations | filtered_docs, query_data | generated_answer, compliance_reminder |
| output_format (inner) | OutputFormatNode | src/nodes/output_format_node.py | ANONYMOUS | render the staff-facing response | generated_answer, compliance_reminder, query_data | esl_response |

### Data flow

```
user_input (staff query string / JSON) + input_context {channel, store_id}
    │
    ▼ PreProcessNode (VERIFIED_EXTERNAL)
validated_input (normalised JSON {query, store_id})
enriched_context (JSON string)
    │
    ▼ ESLQAGraphNode → DomainWorkflowGraph
    │   InputValidateNode  → query_data (JSON string), price_discrepancy (bool)
    │   RetrieveNode       → retrieved_docs (JSON list)
    │   RerankFilterNode   → filtered_docs (JSON list)
    │   GenerateAnswerNode → generated_answer (JSON), compliance_reminder (str)
    │   OutputFormatNode   → esl_response (str)
    ▼ merge_output
esl_response, generated_answer, compliance_reminder, price_discrepancy → outer state
    │
    ▼ PostProcessNode (ANONYMOUS) — the output boundary
formatted_output, result   (or, on a violation, a withheld notice + cleared fields)
```

`result` is written only by the output boundary. The framework resolves the caller-facing answer
as `formatted_output or result`, so anything placed in `result` earlier would be reachable from an
error envelope the boundary never approved.

### State definition

| Field | Type | Purpose | Producer |
|-------|------|---------|----------|
| validated_input | NotRequired[Optional[str]] | Normalised query JSON string | PreProcessNode |
| enriched_context | NotRequired[Optional[str]] | JSON: {source, channel, store_id} | PreProcessNode |
| query_data | NotRequired[Optional[str]] | JSON: {query, store_id, query_type, price_discrepancy, hybrid} | InputValidateNode |
| price_discrepancy | NotRequired[Optional[bool]] | True when the query concerns a price mismatch | InputValidateNode |
| top_k / score_threshold | NotRequired[Optional[int/float]] | Declared retrieval tuning, seeded into the inner state | DomainWorkflowGraph |
| retrieved_docs | NotRequired[Optional[str]] | JSON list of {doc_id, text, score, source} | RetrieveNode |
| filtered_docs | NotRequired[Optional[str]] | JSON list (reranked + thresholded, ≤ top_k) | RerankFilterNode |
| generated_answer | NotRequired[Optional[str]] | JSON: {answer, citations, grounded} | GenerateAnswerNode |
| compliance_reminder | NotRequired[Optional[str]] | 景品表示法 reminder text or "" | GenerateAnswerNode |
| esl_response | NotRequired[Optional[str]] | Rendered response, before the output boundary | OutputFormatNode |
| result | NotRequired[Optional[str]] | The approved caller-facing answer | PostProcessNode |

All dict/list-valued fields use JSON-serialised `Optional[str]`, via the `to_json()` / `from_json()`
helpers in `src/schemas/state.py` at every producer and consumer boundary.

**Prohibited**: re-declaring `formatted_output` (it is inherited), credentials in State, Pydantic
models.

### Configuration

Two files, with no overlap between them.

`config/agent.yaml` is the static manifest — identity, entry point, trust level, and the
compile-time `requires` gates. It holds no runtime values.

`config/config.yaml` holds everything tunable, and is read by `src/services/runtime_config.py`:

```yaml
max_retry: 3
timeout_s: 30
retrieval:
  top_k: 5
  score_threshold: 0.5
  hybrid_search: true   # staff queries carry SKU codes and part numbers
```

Every value has a reader. `max_retry` reaches the backbone's retry route because the entry point
constructs the graph as `Graph(config=agent_config())`; `timeout_s` is carried on the same graph
config as the declared per-call budget for the hosting runtime; the `retrieval` block reaches the
ranking step through the inner graph's initial state, because the framework threads no runtime
config into a node's `execute()`. Every number is bounds-checked on load: a non-finite or
out-of-range value falls back to the documented default rather than silently changing what the
agent grounds on.

The manifest declares `generation_mode: deterministic`. No language model is called: the answer is
assembled from the retrieved passages verbatim. `prompts/esl_operations.j2` is the starting point
for a fork that wants a model in the answer step; nothing in `src/` loads it, and its header says
so.

## Security configuration

| Concern | Where | How |
|---------|-------|-----|
| Trust | PreProcessNode `required_trust_level = VERIFIED_EXTERNAL` | The entry node is the only one requiring elevated trust; the framework's wrapper enforces it before `execute()` runs |
| Caller input | PreProcessNode + `src/services/input_guard.py` | Chat-template control markers (`<\|...\|>`, `[INST]`, `<<SYS>>`, `<s>`, `<system>`) and directive overrides, screened raw AND markup-stripped, depth-first over the parsed payload including keys; credential shapes via the framework's own detector; size caps; caller strings that render into the response locked to `[a-z0-9_-]{1,32}` |
| Request context | the entry point + PreProcessNode | Only `channel` and `store_id` are accepted; every other key is DROPPED before the graph runs, because an undeclared key stays on the context channel and the framework returns that channel verbatim from its first node |
| Output | PostProcessNode `_security_gate_output()` + `_grounding_violation()` | Two invariants, each with its own audit event; on violation, ERROR plus every output-bearing field cleared and a truthy withheld notice |
| Audit | `emit_trace_event()` in every `execute()` | A domain event on every reachable path |
| Credentials | — | None in State; secrets only through the invocation context |

### The output invariant this agent enforces

The response makes two promises, in its own `Sources:` and `Grounded:` lines. The boundary
re-derives both from the corpus rather than trusting the step that assembled them:

1. every citation names a passage that exists;
2. `Grounded: YES` appears only when there is at least one citation;
3. every numbered line of procedure is that cited passage **verbatim**;
4. an ungrounded answer is the fixed abstention text, not improvised prose;
5. the compliance reminder is the template's own constant or absent.

The alternative invariant used by templates that render money — rounding every monetary aggregate
to a fixed grid — does not apply here: this agent renders no monetary values. It renders procedure
text, and the grounding invariant is what "the output means what it says" amounts to for that.

A violation returns ERROR, clears every output-bearing field, and replaces the answer with a
truthy notice. Returning ERROR alone would not contain anything: the framework resolves the
caller-facing answer as `formatted_output or result` with no status check, and an empty
`formatted_output` re-opens that fallback rather than closing it.

### Known framework behaviour worth knowing about

The framework masks apparent personal names in the request text before any template code runs.
Title-case product names are indistinguishable from person names to that detector, so a question
about a branded product reaches the pipeline with the brand replaced by a mask, and the echoed
question in the response shows the mask. Retrieval is unaffected in practice — the knowledge base
is procedural rather than product-specific, so the domain words that drive retrieval survive — but
a fork that indexes per-product content should expect this. It is framework behaviour, not
something this template can switch off.

## Framework utilisation

### Shared components used

- [x] InvocationContext — session identity, caller trust, request context
- [x] `detect_credentials_in_value()` — the framework's own credential detector, used by both the
      caller-input screen and the output boundary, so neither can be narrower than the framework's
- [x] `detect_injection()` — the framework's injection screen, widened by this template to cover
      the control-marker family it does not score
- [x] `emit_trace_event()` — at least one domain event per `execute()`
- [x] `to_json()` / `from_json()` in `src/schemas/state.py` — the serialisation contract

### Composition pattern

- **Pattern**: two-layer nested — a GraphNode wrapping an inner BaseGraph
- **Outer graph**: `ESLOperationsPriceComplianceQAAgent(AgentBaseGraph)` — fixed 5-node backbone
- **Inner graph**: `DomainWorkflowGraph(BaseGraph)` — 5-step linear pipeline
- **Error propagation**: propagate — an inner failure raises, and the backbone routes to finalize

## Import isolation

- [x] The template imports no platform SDK internals
- [x] Import targets: `framework/` and `shared/` only

## Design decision record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| Base class | AgentBaseGraph | AutonomousBaseGraph | AgentBaseGraph | A fixed sequential pipeline; no reasoning loop is required |
| Composition | flat | nested GraphNode | nested | Five sequential retrieval steps belong behind one backbone slot |
| State dict fields | bare dict | JSON-serialised str | JSON-serialised str | Checkpoint serialisation safety |
| Answer generation | language model | deterministic assembly from passages | deterministic | Nothing can be invented, and the grounding invariant becomes checkable rather than aspirational |
| Output invariant | monetary precision grid | grounding + citation | grounding + citation | This agent renders no monetary values; the grid would be a check on something that never appears |
| Output credential check | local pattern set | delegate to the framework detector | delegate | A narrower local set is a bypass: the value passes the template check, the framework raises deeper in the pipeline, and the wrapper discards the containment the template performed |
| Caller `store_id` | free text | inert identifier, surfaced in the response | inert identifier | It renders into the response, so free text there is caller-controlled output injection — and a field that is validated and then dropped is worse than one that is not accepted at all |
| Unsupported context keys | ignore | drop before invoke | drop | Ignoring is not stripping: an ignored key still travels on the context channel and is returned verbatim by the framework's first node |
