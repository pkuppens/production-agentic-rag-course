# One hybrid OpenSearch index instead of separate BM25/vector stores

## Context and Problem Statement

Once vector search joined BM25 (ADR-0001), Chunks needed to live somewhere queryable by both. A separate vector database alongside OpenSearch is a common pattern, but it means syncing two stores and reconciling drift between them.

## Considered Options

- A dedicated vector database (e.g. Chroma, Qdrant) alongside OpenSearch for BM25
- A single OpenSearch index carrying both BM25 text fields and dense vectors

## Decision Outcome

Chosen option: "A single OpenSearch index carrying both BM25 text fields and dense vectors", because it keeps keyword and hybrid search reading from one consistent dataset, queried through an RRF pipeline, and avoids the operational cost of syncing two stores.

### Consequences

- Good, because there is exactly one dataset per Chunk - no cross-store consistency to maintain
- Bad, because both retrieval modes are now coupled to OpenSearch's hybrid-query feature set
