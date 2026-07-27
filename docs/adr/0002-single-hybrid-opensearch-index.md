# One hybrid OpenSearch index instead of separate BM25/vector stores

All Chunks are indexed once into a single OpenSearch index (`{index_name}-{chunk_index_suffix}`, see `OpenSearchSettings`) that carries both BM25 text fields and dense vectors, queried through an RRF pipeline rather than maintaining a separate vector database alongside OpenSearch. This keeps keyword and hybrid search reading from one consistent dataset and avoids the operational cost of syncing two stores, at the cost of coupling both retrieval modes to OpenSearch's hybrid-query feature set.
