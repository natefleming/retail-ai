# Lakebase Reference

End-to-end reference for using Databricks Lakebase Postgres with dao-ai:
`DatabaseModel` connection helpers, `LakebaseVectorStoreModel` retriever
configuration, table provisioning, and embedding backfill.

## Model overview

| Model | Role | Auth |
|---|---|---|
| `DatabaseModel` (`resources.databases.<name>`) | Postgres connection (Lakebase project + branch OR host/port/user/password). | Own `IsDatabricksResource` — `client_id`/`client_secret`/`workspace_host` for Lakebase; `user`/`password` for standard Postgres. |
| `LakebaseVectorStoreModel` (`resources.vector_stores.<name>` with `type: lakebase_search`) | Retriever-schema wrapper over a `DatabaseModel`. Describes table + columns + embedding endpoint. | Delegates to the nested `database` — no auth fields of its own. |
| `LakebaseRetrieverModel` (`retrievers.<name>` with `type: lakebase_search`) | Runtime retriever. Composes a vector store with `search_parameters` (ANN / BM25 / HYBRID) + optional `rerank` + optional `instructed` pipeline. | Inherited via `vector_store.database`. |

## `DatabaseModel` — connection primitives

The five helpers below all wrap `PostgresPoolManager.get_pool(db)` so
notebook / setup code doesn't need to import the pool manager directly.
Every helper opens a transaction, commits on success, rolls back on any
exception raised inside the call (or inside the `with` block for
`connect()`).

### `execute_update(statements, parameters=None) -> None`

Write operations (DDL, INSERT / UPDATE / DELETE). Accepts either a single
SQL string or a sequence of strings — the whole sequence runs in one
transaction. Any result set is dropped.

```python
database.execute_update("CREATE TABLE t (id int, name text);")
database.execute_update(
    ["CREATE EXTENSION IF NOT EXISTS lakebase_vector CASCADE;",
     "CREATE INDEX ON kb USING lakebase_ann (embedding vector_cosine_ops);"]
)
database.execute_update("INSERT INTO t VALUES (%s, %s)", parameters=(1, "a"))
```

Parameter binding is single-statement-only — `execute_many` handles bulk
parameterized writes.

### `execute_query(query, parameters=None) -> list[dict[str, Any]]`

Read operations (SELECT, RETURNING, SHOW). Returns rows as `list[dict]`
keyed by column name (Lakebase pools use `row_factory=dict_row`). Returns
`[]` when the query produces no rows.

```python
row_count: int = database.execute_query("SELECT COUNT(*) AS n FROM kb")[0]["n"]
rows: list[dict] = database.execute_query(
    "SELECT id, passage FROM kb WHERE category = %s", parameters=("auth",),
)
```

### `execute_many(query, param_seq) -> None`

Bulk parameterized writes via `cursor.executemany` — one round-trip for
many rows.

```python
database.execute_many(
    "UPDATE kb SET embedding = %s::vector WHERE id = %s",
    [(vec, row_id) for row_id, vec in zip(ids, vectors)],
)
```

### `connect() -> Iterator[Cursor]`

Escape hatch for multi-statement transactions, streaming
`fetchmany` loops, or any workflow psycopg handles natively. Auto-commits
on success, rolls back on any exception raised inside the `with` block.

```python
with database.connect() as cur:
    cur.execute("SELECT id, passage FROM kb WHERE embedding IS NULL")
    rows = cur.fetchall()
    if rows:
        vectors = embedder.embed_documents([r["passage"] for r in rows])
        for row, vec in zip(rows, vectors):
            cur.execute(
                "UPDATE kb SET embedding = %s::vector WHERE id = %s",
                (vec, row["id"]),
            )
```

For the vector-backfill case specifically, `dao_ai.lakebase.backfill_embeddings`
(below) is a one-liner.

## `LakebaseVectorStoreModel.provision(dimension, ...)`

Idempotently creates the Postgres extensions, table, and indexes for a
Lakebase retriever. Every statement uses `IF NOT EXISTS` — safe to
re-run.

```python
retriever.vector_store.provision(
    dimension=1024,                                  # matches your embedding endpoint
    metadata_column_types={"priority": "int"},       # non-string metadata columns
    id_column_type="text",                           # or "bigint", "uuid", etc.
)
```

What lands in Postgres:

| Object | Emitted when | Statement |
|---|---|---|
| `lakebase_vector` extension | Always | `CREATE EXTENSION IF NOT EXISTS lakebase_vector CASCADE` |
| `lakebase_text` extension | `tsvector_column` set | `CREATE EXTENSION IF NOT EXISTS lakebase_text` |
| Table | Always | `CREATE TABLE IF NOT EXISTS {schema}.{table} (id {type} PK, content text NOT NULL, embedding vector({dimension}), <metadata cols typed>, <tsvector_column tsvector GENERATED>)` |
| ANN index | Always | `CREATE INDEX IF NOT EXISTS {table}_{embedding_column}_ann USING lakebase_ann ({embedding_column} {vector_cosine_ops \| vector_l2_ops \| vector_ip_ops})` — operator chosen from `distance_metric` |
| BM25 index | `tsvector_column` set | `CREATE INDEX IF NOT EXISTS {table}_{tsvector_column}_bm25 USING lakebase_bm25 ({tsvector_column})` |

`metadata_column_types` maps metadata column names to Postgres types —
default is `text` for unlisted columns. Use standard type names:
`int`, `bigint`, `numeric`, `timestamp`, `boolean`, etc.

`dimension` has no default. The caller knows the embedding endpoint
(`1024` for `databricks-gte-large-en`, `1536` for `text-embedding-3-small`
via foundation-model API, etc.).

## `dao_ai.lakebase.backfill_embeddings(vector_store, embedder=None, *, batch_size=128)`

Encodes + populates the embedding column for every row where it's NULL.
Reads via `execute_query`, encodes in chunks, writes back via
`execute_many`. Idempotent — safe after every seed or incremental insert.

```python
from dao_ai.lakebase import backfill_embeddings

n_updated: int = backfill_embeddings(vector_store)
```

`embedder` defaults to `vector_store.embedding_model.as_embeddings_model()`
— same endpoint the retriever uses at query time, so write-side and
read-side embeddings stay consistent. Pass an explicit embedder to
override (any callable with `embed_documents(list[str]) -> list[list[float]]`).

`batch_size` tunes how many rows per `embed_documents` call. Tune down
for large-dimension models under memory pressure; up for small models to
reduce round-trips.

## End-to-end example

Setup a KB assistant against Lakebase from Python:

```python
from pathlib import Path

from dao_ai.config import AppConfig, LakebaseRetrieverModel, LakebaseVectorStoreModel
from dao_ai.lakebase import backfill_embeddings

config: AppConfig = AppConfig.from_file("kb_assistant.yaml", params=params)
retriever: LakebaseRetrieverModel = config.retrievers["kb_retriever"]
vector_store: LakebaseVectorStoreModel = retriever.vector_store

# 1. Extensions + table + indexes
vector_store.provision(dimension=1024, metadata_column_types={"priority": "int"})

# 2. Seed rows (INSERT ... ON CONFLICT DO NOTHING for idempotency)
vector_store.database.execute_update(Path("data/kb_articles.sql").read_text())

# 3. Embed the passages
backfill_embeddings(vector_store)

# 4. Query
tool = config.tools["kb_search"].function.as_tools()[0]
docs = tool.invoke({"query": "How do I reset my password?"})
```

## Deploy-path notes

- **App bundle** (`dao-ai agent generate` + `databricks bundle deploy`) —
  Lakebase entries under `resources.vector_stores` emit no
  `vector-search-index` resource (the App SP authenticates via
  `database.client_id` / `client_secret` at runtime).
- **Model Serving** (`config.deploy_agent(target=ServingMode.MODEL_SERVING)`) —
  Lakebase entries delegate their `as_resources()` call to the nested
  `DatabaseModel`. For autoscaling Lakebase projects the delegate returns
  `[]` (MLflow doesn't have a matching resource type yet — tracked at
  [mlflow/mlflow#22452](https://github.com/mlflow/mlflow/issues/22452)),
  and auth flows via the OAuth credentials on the runtime DB connection.

## Reference examples

| Example | Demonstrates |
|---|---|
| [`examples/20_lakebase_search/ann_only.yaml`](../examples/20_lakebase_search/ann_only.yaml) | Minimal ANN — no tsvector column required. |
| [`examples/20_lakebase_search/bm25_only.yaml`](../examples/20_lakebase_search/bm25_only.yaml) | BM25 only — exact-term lookup (SKUs, error codes). |
| [`examples/20_lakebase_search/hybrid_rrf.yaml`](../examples/20_lakebase_search/hybrid_rrf.yaml) | ANN + BM25 fused via RRF. |
| [`examples/20_lakebase_search/reranked.yaml`](../examples/20_lakebase_search/reranked.yaml) | HYBRID + FlashRank cross-encoder rerank. |
| [`examples/20_lakebase_search/filters_and_traces.yaml`](../examples/20_lakebase_search/filters_and_traces.yaml) | Static filters + MLflow `trace_location`. |
| [`examples/20_lakebase_search/dynamic_schema.yaml`](../examples/20_lakebase_search/dynamic_schema.yaml) | Hand-declared `ColumnInfo` for per-column filter narrowing. |
| [`examples/20_lakebase_search/instructed.yaml`](../examples/20_lakebase_search/instructed.yaml) | Full instructed retrieval pipeline (decompose → parallel → RRF → LLM rerank → verify). |
