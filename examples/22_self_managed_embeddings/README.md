# 22 · Self-managed (precomputed) embeddings

Query a Databricks Vector Search index built with **precomputed vectors** —
a Delta Sync index with `embedding_vector_columns`, or a Direct Access index —
through the `ai_search` tool.

A managed-embeddings index embeds the query server-side, so the retriever needs
nothing extra. A **self-managed** index does not, so the `vector_store` declares
two additional fields:

| Field | Meaning |
|-------|---------|
| `text_column` | Column returned as the document content the LLM reads. **Not** the embedding/vector column — that lives in the index and is detected automatically. |
| `embedding_model` | Endpoint used to embed the incoming query at runtime. **Must** be the same model + dimension that produced the stored vectors. |

The embedding mode is auto-detected from the index metadata: set these two
fields for a self-managed index; omit both for a managed-embeddings index.

See [`self_managed_embeddings.yaml`](./self_managed_embeddings.yaml).
