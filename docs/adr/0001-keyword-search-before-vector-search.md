# Keyword search (BM25) before vector search

## Context and Problem Statement

Retrieval needed a starting point before any embedding model existed in the system. Most RAG tutorials build vector search first; this course instead ships OpenSearch BM25 scoring in Week 3, with vector/hybrid search layered on top only in Week 4.

## Considered Options

- Vector search first, add BM25 later
- BM25 first, add vector search as an enhancement

## Decision Outcome

Chosen option: "BM25 first, add vector search as an enhancement", because it gives fast, interpretable, exact-match results without needing an embedding model, and mirrors how production search systems (Elasticsearch, Algolia) actually evolve.

### Consequences

- Good, because the system can always fall back to plain BM25 if vector/hybrid search is degraded or unavailable
- Bad, because vector search is treated as strictly additive rather than a first-class design constraint from day one
