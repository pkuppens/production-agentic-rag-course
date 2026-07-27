# Keyword search (BM25) before vector search

Retrieval was built keyword-first: OpenSearch BM25 scoring shipped in Week 3, with vector/hybrid search layered on top only in Week 4. This is a deliberate deviation from vector-first RAG tutorials — BM25 gives fast, interpretable, exact-match results without an embedding model, and mirrors how production search systems (Elasticsearch, Algolia) actually evolve. Vector search is treated as an enhancement on top of keyword search, not a replacement for it — the system should always be able to fall back to plain BM25.
