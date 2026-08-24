# Cache is a performance layer, not a dependency

## Context and Problem Statement

`/ask` responses can be cached in Redis to avoid recomputation on repeated queries (Week 6 measured 150-400x speedup on repeated queries). Redis is not otherwise part of the RAG pipeline, so a Redis outage shouldn't be able to take the whole API down.

## Considered Options

- Treat Redis as a required dependency (fail requests/startup if Redis is unreachable)
- Treat Redis as an optional performance layer (fail open: degrade to "no cache" on any Redis error)

## Decision Outcome

Chosen option: "Treat Redis as an optional performance layer (fail open)", because caching here is a performance optimization, not a source of truth that other components depend on - trading away guaranteed cache hits for availability is an acceptable trade-off.

### Consequences

- Good, because a Redis outage degrades performance rather than causing an outage
- Bad, because it requires discipline at every call site to catch-and-continue rather than let Redis errors propagate

**Known gap (found during #2, not fixed there):** this fail-open behavior only covers per-request cache operations. `make_cache_client`/`make_redis_client` (`src/services/cache/factory.py`) re-raise on connection failure, and `src/main.py`'s `lifespan()` doesn't catch that - so the whole API currently fails to start if Redis is unreachable at boot, contradicting the "cache is not a dependency" decision above. Tracked in #7 (catch-and-log in `lifespan()` the way OpenSearch connectivity is already handled; possibly a circuit-breaker on `CacheClient` itself).
