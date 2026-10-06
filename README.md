# ESL Operations & Price Compliance Q&A Agent

AI agent for answering electronic shelf label operations and price compliance questions, built with Agentic Star.

> **Category**: Cat 2 (domain-specific pipeline)
> **Industry**: Retail
> **Template ID**: RET-C2-131

## Overview

A question-answering agent for retail store staff who work with electronic shelf labels
(ESL). A staff member asks a question in plain language — how to push a price update, what
to do about a low-battery alert, what to do when the shelf price and the till price
disagree — and the agent answers with the matching procedure, quoted from the store's
operations manual and pricing rules and cited by document identifier.

Two properties define it. Every line of procedure it returns is a knowledge-base passage
verbatim: nothing is paraphrased or invented, and when nothing relevant is found it says so
rather than improvising. And when the question concerns a price discrepancy, the answer
carries a compliance reminder about Japan's 景品表示法 (Act against Unjustifiable Premiums
and Misleading Representations), which governs how a misleading displayed price must be
handled. The output boundary re-checks both properties against the corpus before anything
is released, and withholds the answer if either fails.

The shipped knowledge base is a small, deterministic sample so the agent runs end to end
out of the box. Replace it with your own procedures — that is the main thing a fork
changes.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent
fails at graph compile / start-up preflight rather than starting in a partially
working state. This is intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Project Structure

```
src/          agent implementation (nodes, services, schemas)
tests/        unit, integration and boundary tests
config/       agent.yaml (identity and entry point) + config.yaml (runtime values)
prompts/      starting point for a fork that wants a language model in the answer step
docs/         design and test specification
```

See `docs/` for the design specification and test specification.

## Customising

1. Replace the passages in `src/services/knowledge_base.py` with your own operations
   manual and pricing rules. The passage contract (`doc_id` / `source` / `keywords` /
   `text`) is what the rest of the pipeline depends on, so a real vector index can be
   wired in behind it without touching anything downstream.
2. Adjust `config/config.yaml` for your own retrieval tuning. Values are bounds-checked
   on load; an out-of-range entry falls back to the documented default.
3. Review `src/nodes/` for the pipeline steps, and `src/nodes/post_process_node.py` for
   the output boundary — if you change what the response may contain, change the checks
   there too.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
