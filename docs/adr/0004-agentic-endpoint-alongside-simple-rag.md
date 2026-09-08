# LangGraph agentic pipeline alongside, not instead of, the simple RAG endpoints

## Context and Problem Statement

Week 7 introduced an agentic pipeline (Guardrail -> Retrieve -> Grade -> Rewrite -> Generate, `src/services/agents/`) with adaptive, multi-step retrieval. The existing `/api/v1/ask` and `/api/v1/stream` endpoints already served simple, single-pass queries well.

## Considered Options

- Replace `/ask`/`/stream` with the agentic pipeline
- Add the agentic pipeline as a new, separate `/api/v1/agentic-ask` endpoint

## Decision Outcome

Chosen option: "Add the agentic pipeline as a new, separate `/api/v1/agentic-ask` endpoint", because it preserves a fast, predictable single-pass path for simple queries while offering adaptive retrieval (guardrails, relevance grading, query rewriting) only where it earns its extra latency and cost.

### Consequences

- Good, because callers (API clients, Telegram bot) choose which mode they need per request instead of always paying agentic latency
- Bad, because the two pipelines duplicate some retrieval/generation logic that must be kept in sync
