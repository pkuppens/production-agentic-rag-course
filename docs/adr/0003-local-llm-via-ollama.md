# Local LLM (Ollama) instead of a cloud LLM API

## Context and Problem Statement

Generation needs an LLM backend. This is a learner-focused course, so cost and data locality matter as much as raw model quality: the course should stay runnable at ~$0, and paper content/queries ideally shouldn't have to leave the machine.

## Considered Options

- A hosted LLM API (OpenAI, Anthropic, etc.)
- A local Ollama server (`llama3.2`, configured via `OLLAMA_HOST`/`OLLAMA_MODEL`)

## Decision Outcome

Chosen option: "A local Ollama server", because it keeps the course free to run and guarantees paper content and queries never leave the machine, at the cost of slower and lower-quality generation than a hosted frontier model.

### Consequences

- Good, because there's no per-request API cost or external data exposure
- Bad, because generation quality/speed is capped by local hardware - Week 5 mitigates this with an 80% prompt-size reduction for a 6x speedup rather than switching providers
