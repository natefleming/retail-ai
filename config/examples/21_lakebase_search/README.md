# 21. Lakebase Search

**Retrieval over Databricks Lakebase Postgres — ANN, BM25, and hybrid RRF**

Query embeddings and lexical text living in Lakebase Postgres directly from an
agent, using the `lakebase_vector` and `lakebase_text` extensions. Sibling of
`ai_search`; same YAML shape, different backend.

## When to use this vs. AI Search

| | `type: ai_search` | `type: lakebase_search` |
|---|---|---|
| Backend | Databricks-managed Vector Search index | Postgres in a Lakebase project |
| Data location | Delta table (source) synced to the index | Rows in Lakebase Postgres |
| Auto-embedding | Yes (Delta Sync) | No — client-side embed via Model Serving |
| Hybrid | Native | Client-side RRF |
| Best fit | Static or Delta-native document corpora | Data already in Postgres (OLTP joins, transactional writes) |

Pick `lakebase_search` when your source of truth is already in Lakebase and
you'd like retrieval to be a straight SQL query alongside your other Postgres
data (users, orders, sessions, etc.).

## Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#1565c0'}}}%%
flowchart TB
    Q["📝 User Query"]

    subgraph Agent["🤖 Agent"]
        LLM["LLM (Claude / GPT / ...)"]
    end

    subgraph Tool["🔧 lakebase_search Tool"]
        Emb["Embed via<br/>databricks-gte-large-en<br/>(only for ANN / HYBRID)"]
        Retriever["LakebaseRetriever"]
    end

    subgraph Lakebase["🗄️ Lakebase Postgres"]
        ANN["<b>lakebase_ann</b> index<br/><i>embedding VECTOR(1024)</i>"]
        BM25["<b>lakebase_bm25</b> index<br/><i>passage_tsv TSVECTOR</i>"]
    end

    RRF["Reciprocal Rank Fusion<br/>(HYBRID only)"]
    Docs["Ranked Documents"]

    Q --> LLM
    LLM --> Emb
    Emb --> Retriever
    Retriever -->|"<=> distance"| ANN
    Retriever -->|"<@> BM25 score"| BM25
    ANN --> RRF
    BM25 --> RRF
    RRF --> Docs
    Docs --> LLM

    style Tool fill:#e3f2fd,stroke:#1565c0
    style Lakebase fill:#e8f5e9,stroke:#2e7d32
```

## Examples

| File | Mode | Use case |
|------|------|----------|
| [`ann_only.yaml`](./ann_only.yaml) | ANN | Semantic similarity search — minimal config, no BM25 setup needed |
| [`hybrid_rrf.yaml`](./hybrid_rrf.yaml) | HYBRID | Dense + lexical fused via RRF — highest recall |
| [`bm25_only.yaml`](./bm25_only.yaml) | BM25 | Exact-term lookup (SKUs, error codes) — no embedding calls |
| [`filters_and_traces.yaml`](./filters_and_traces.yaml) | HYBRID | Static filters (equality / IN / comparison / LIKE) + `trace_location` to UC OTEL tables |
| [`dynamic_schema.yaml`](./dynamic_schema.yaml) | HYBRID | Hand-declared `ColumnInfo` with per-column operator narrowing + LLM-facing descriptions (Mode A discovery) |
| [`reranked.yaml`](./reranked.yaml) | HYBRID / ANN | FlashRank cross-encoder reranking (`rerank: true` shorthand + full `RerankParametersModel` form) |

## Prerequisites

1. **Enable Lakebase Search on the project** (workspace UI → Compute →
   Postgres → your project → Settings → Enable Lakebase Search).
   This is a one-time, project-level, **irreversible** toggle that restarts
   all compute in the project.
2. **Install the extensions** in the target database:
   ```sql
   CREATE EXTENSION IF NOT EXISTS lakebase_vector CASCADE;
   CREATE EXTENSION IF NOT EXISTS lakebase_text;    -- only if BM25 / HYBRID
   ```
3. **Create the table** with a vector column matching your embedding
   endpoint's output dimension (1024 for `databricks-gte-large-en`):
   ```sql
   CREATE TABLE kb_articles (
     id           TEXT PRIMARY KEY,
     category     TEXT,
     source_url   TEXT,
     passage      TEXT NOT NULL,
     embedding    VECTOR(1024),
     passage_tsv  TSVECTOR GENERATED ALWAYS AS
       (to_tsvector('english', passage)) STORED     -- only if BM25 / HYBRID
   );

   CREATE INDEX ON kb_articles USING lakebase_ann  (embedding vector_cosine_ops);
   CREATE INDEX ON kb_articles USING lakebase_bm25 (passage_tsv);   -- only if BM25 / HYBRID
   ```
4. **Populate embeddings client-side.** Lakebase has no in-database
   embedding function; the caller must embed and INSERT the vector:
   ```python
   from databricks_langchain import DatabricksEmbeddings

   emb = DatabricksEmbeddings(endpoint="databricks-gte-large-en")
   vec = emb.embed_documents([passage])[0]   # list[float] of length 1024
   cur.execute(
       "INSERT INTO kb_articles (id, category, passage, embedding) "
       "VALUES (%s, %s, %s, %s::vector)",
       (id, category, passage, vec),
   )
   # passage_tsv is generated automatically by Postgres; no client work.
   ```

## Configuration surface

### `LakebaseVectorStoreModel` — bare table reference

| Field | Required | Default | Notes |
|---|---|---|---|
| `database` | ✓ | — | Reference to a `DatabaseModel` (Lakebase or standard Postgres) |
| `schema_name` | | `"public"` | Postgres schema |
| `table` | ✓ | — | Table name |
| `id_column` | | `"id"` | Primary-key column; populated on `Document.id` and `metadata` |
| `content_column` | ✓ | — | Text column → `Document.page_content` |
| `embedding_column` | ✓ | — | `VECTOR(N)` column |
| `tsvector_column` | | `None` | `TSVECTOR` column; **required** for BM25 / HYBRID |
| `metadata_columns` | | `[]` | Extra columns surfaced on `Document.metadata`; also the filter allowlist |
| `embedding_model` | ✓ | — | `InferenceEndpointModel` — bare string is auto-coerced |
| `bm25_index_name` | | auto | `<schema>.<table>_<tsvector_column>_bm25` if omitted |
| `distance_metric` | | `"cosine"` | `cosine` / `l2` / `ip` → `<=>` / `<->` / `<#>` |
| `tsv_language` | | `"english"` | Passed to `to_tsvector(<lang>, ...)` |

### `LakebaseRetrieverModel` — full retriever

Wraps a `LakebaseVectorStoreModel` with `SearchParametersModel`
(`num_results`, `filters`, `query_type`). Enforces that
`query_type in {BM25, HYBRID}` requires `tsvector_column`.

### Filter operators (aligned with `ai_search`)

Filter keys use the same suffix convention as `ai_search`: a column name
optionally followed by a space and an operator token. Two shapes:

1. **LLM-facing runtime** (`list[FilterItem]`) — what the LLM emits:
   ```json
   [{"key": "category", "value": "auth"},
    {"key": "priority >=", "value": 2},
    {"key": "status", "value": ["published", "reviewed"]}]
   ```
2. **Static config** (`search_parameters.filters`, `dict[str, Any]`) —
   what you write in YAML:
   ```yaml
   filters:
     category: auth
     "priority >=": 2
     "status": [published, reviewed]
   ```

Both forms are merged at runtime before hitting Postgres.

| Suffix | SQL emitted (scalar) | SQL emitted (list value) |
|---|---|---|
| *(none — equality)* | `col = %s` | `col = ANY(%s)` (IN) |
| ` NOT` | `col <> %s` | `NOT (col = ANY(%s))` (NOT IN) |
| ` <` ` <=` ` >` ` >=` | `col <op> %s` | *error — list value not supported* |
| ` LIKE` | `col ILIKE %s` | *error* |
| ` NOT LIKE` | `NOT (col ILIKE %s)` | *error* |

Notes:
- **`LIKE` maps to Postgres `ILIKE`** (case-insensitive). Agent-friendly
  default — LLMs reason about pattern match without caring about case, and
  Vector Search's tokenized LIKE is likewise case-insensitive.
- Column names come from `metadata_columns` on the vector store (or from
  `retriever.columns` when hand-declared with `ColumnInfo`). Unknown keys
  fail Pydantic validation at tool-call time; the LLM never sees them as
  legal options.
- Column names are wrapped in `psycopg.sql.Identifier`; values are always
  bound as `%s` parameters — no string interpolation ever.
- `IS NULL` / `IS NOT NULL` are intentionally not exposed — agents don't
  reason well about null. Use a static `search_parameters.filters` entry
  with a specific sentinel value if you need it.

### Dynamic schema

When the retriever can enumerate its columns (Mode A hand-declared /
Mode B declared + Postgres enrichment / Mode C empty + Postgres discovery),
the tool's `args_schema` narrows `filters[].key` to a `Literal[...]` of the
actual `column × suffix` cross-product. The LLM sees the exact enum in its
function-calling schema — no more guessing column names. Mirrors
ai_search's `_build_filter_item_model`.

Discovery calls Postgres exactly once per tool build:

```sql
SELECT column_name, data_type,
       col_description(...::regclass, ordinal_position) AS comment
FROM information_schema.columns
WHERE table_schema = %s AND table_name = %s
ORDER BY ordinal_position;
```

Columns used by the retriever internally (id / content / embedding /
tsvector) are stripped so they don't appear on the LLM-facing filter enum.
See [`dynamic_schema.yaml`](./dynamic_schema.yaml) for a hand-declared
example that locks per-column operators via `ColumnInfo.operators`.

## MLflow tracing

All three retrieval helpers are wrapped with
`@mlflow.trace(span_type=SpanType.RETRIEVER)`:

```
[TOOL      ] lakebase_search
  [RETRIEVER ] LakebaseRetriever                    (LangChain BaseRetriever auto-span)
    [RETRIEVER ] lakebase_hybrid_search             (@mlflow.trace)
      [RETRIEVER ] lakebase_ann_search              (@mlflow.trace)
      [RETRIEVER ] lakebase_bm25_search             (@mlflow.trace)
```

This matches the span shape `ai_search` produces, so existing trace-review
tooling (the `analyze-mlflow-trace` and `retrieving-mlflow-traces` skills,
MLflow UI) works unmodified. See `filters_and_traces.yaml` for a
`trace_location` config that ships spans to a UC OTEL Delta table set.

## What's not yet supported (roadmap)

- **Rerank + instructed retrieval** — planned Stage 2. Today the retriever
  matches `ai_search`'s field shape for `search_parameters` but does not yet
  honor `rerank` or `instructed`.
- **Auto-provisioning** — planned Stage 3. Today the caller is responsible
  for `CREATE TABLE`, indexes, and embedding backfill; there is no
  `source_table` / auto-sync equivalent to `AiSearchIndexModel`.
- **Per-column typed filter kwargs on the tool schema** — planned Stage 4.
  Today the tool exposes a single `filters: dict` argument; the LLM must
  know the column names. Static filters set in `search_parameters.filters`
  work with equal fidelity to `ai_search`.

## Try it

```bash
# Validate one of the configs — swap in your own project / schema / table names first.
dao-ai validate -c config/examples/21_lakebase_search/hybrid_rrf.yaml

# Deploy as a Databricks App bundle.
dao-ai generate-bundle -c config/examples/21_lakebase_search/hybrid_rrf.yaml \
  -o /tmp/lakebase-agent --force --development -p <profile>
cd /tmp/lakebase-agent
databricks bundle deploy --target dev -p <profile>
databricks bundle run dao-ai-lakebase-hybrid-example --target dev -p <profile>
```

## Reranking

FlashRank cross-encoder reranking, same shape ai_search uses. The retriever
fetches `num_results` docs (ANN / BM25 / HYBRID); FlashRank then re-scores
each `(query, doc)` pair and returns the top `rerank.top_n` in descending
score order. Every returned `Document.metadata` carries a `reranker_score`
field.

```yaml
retrievers:
  kb_reranked:
    vector_store: *products_vector_store
    search_parameters:
      num_results: 20                   # cast a wide net
      query_type: HYBRID
    rerank:                             # narrow to top-5 after cross-encoder
      model: ms-marco-MiniLM-L-12-v2
      top_n: 5
```

**`rerank: true` shorthand** — equivalent to
`rerank: { model: ms-marco-MiniLM-L-12-v2 }` with no `top_n`, matching
`ai_search`'s default coercion.

**No serving endpoint required.** FlashRank is a local ONNX cross-encoder;
weights download once to `~/.dao_ai/cache/flashrank/` (or
`/tmp/dao_ai/cache/flashrank` under Databricks Apps / Model Serving).

**Works with all three query types.** ANN / BM25 / HYBRID all produce a
`list[Document]`; the same rerank pass runs on any of them. Verified live
against `retail-consumer-goods`.

See [`reranked.yaml`](./reranked.yaml) for a full config.
