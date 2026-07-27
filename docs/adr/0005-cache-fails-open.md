# Cache is a performance layer, not a dependency

Redis caching (`src/services/cache/`) for `/ask` responses is designed to fail open: if Redis is unavailable, the RAG endpoints must continue serving live results rather than erroring out (`make_cache_client` builds a client that degrades gracefully). This trades away guaranteed cache hits for availability — acceptable because caching here is a performance optimization (Week 6 measured 150–400x speedup on repeated queries), not a source of truth that other components depend on.
