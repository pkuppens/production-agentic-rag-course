# arXiv Paper Curator

A self-hosted RAG system that curates arXiv CS.AI papers and answers research questions over them, built incrementally as a 7-week course in production RAG engineering (see `docs/agents/issue-tracker.md` for the weekly plan).

## Language

**Paper**:
An arXiv publication ingested via the arXiv API, parsed with Docling, and stored as metadata + full text in Postgres.
_Avoid_: Document, article

**Chunk**:
A section-aware, overlapping slice of a Paper's parsed text, sized for embedding and indexing (`src/services/indexing/text_chunker.py`).
_Avoid_: Segment, passage

**Hybrid Index**:
The single OpenSearch index (`{index_name}-{chunk_index_suffix}`) holding both BM25 text fields and dense vectors for every Chunk. There is exactly one — BM25-only and vector-only queries both run against it.
_Avoid_: vector store, search index

**RRF (Reciprocal Rank Fusion)**:
The OpenSearch pipeline (`rrf_pipeline_name`) that merges a BM25 result list and a vector result list into one ranked hybrid result.

**Guardrail**:
The agent node that judges whether an incoming query is in-domain (arXiv CS.AI research) before any retrieval happens; out-of-domain queries are routed to the out-of-scope node instead of the retriever.

**Document Grading**:
The agent node that scores retrieved Chunks for relevance to the query, deciding whether to proceed to generation or trigger a Query Rewrite.

**Query Rewrite**:
The agent node that reformulates a query when Document Grading finds the retrieved Chunks insufficient, triggering another retrieval attempt.

**Agentic RAG**:
The LangGraph state machine (Guardrail → Retrieve → Grade → Rewrite → Generate) exposed via `/api/v1/agentic-ask`, distinct from the simpler single-pass `/ask` and `/stream` endpoints. See ADR-0004.
_Avoid_: "the agent" alone (ambiguous with the Telegram bot)

**Week**:
A course milestone (Week 1–7) that adds one architectural layer to the system; tagged in git as `weekN.0` and developed on its own branch. See `docs/agents/issue-tracker.md`.
