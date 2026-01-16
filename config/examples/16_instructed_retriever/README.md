# 16. Instructed Retriever

**Improve retrieval quality with query decomposition and constraint-aware filtering**

Instructed Retriever extends traditional RAG by carrying system specifications through query decomposition, retrieval, and reranking stages. It automatically translates natural language constraints into executable metadata filters.

## Examples

| File | Description | Use Case |
|------|-------------|----------|
| `instructed_retriever.yaml` | Full instructed retrieval with RRF merging | Complex queries with metadata constraints |

## What You'll Learn

- **Query Decomposition** - Break complex queries into focused subqueries
- **Metadata Reasoning** - Auto-translate constraints to filters ("last month" → timestamp filter)
- **RRF Merging** - Combine results from multiple queries using Reciprocal Rank Fusion
- **Constraint Following** - Enforce recency, exclusions, and other user instructions

## Quick Start

```bash
dao-ai chat -c config/examples/16_instructed_retriever/instructed_retriever.yaml
```

Try queries like:
- "Find Milwaukee power tools under $200 from the last 6 months"
- "Show me cordless drills excluding DeWalt"
- "Recent paint products in the exterior category"

## Why Instructed Retriever?

### The Problem
Standard vector search retrieves based on semantic similarity alone:
- Ignores metadata constraints in natural language
- Can't handle relative time references ("last month")
- Misses relevant documents when query has multiple intents

### The Solution
Instructed Retriever adds an LLM-powered decomposition stage:
1. **Decompose**: Break query into subqueries with explicit filters
2. **Search**: Execute subqueries in parallel
3. **Merge**: Combine results using RRF (rank-based, not score-based)
4. **Rerank**: Apply final reranking for precision

### The Benefit
- **35-50% recall improvement** over naive RAG
- **Automatic filter generation** from natural language
- **Parallel execution** for low latency
- **Graceful fallback** if decomposition fails

## Configuration Pattern

```yaml
retrievers:
  instructed_retriever:
    vector_store: *products_vector_store
    search_parameters:
      num_results: 50
      query_type: HYBRID
    instructed:
      enabled: true
      decomposition_model: *fast_llm  # Smaller model for low latency
      schema_description: |
        Products table columns:
        - brand_name (STRING): Brand/manufacturer name
        - category (STRING): Product category
        - price (DOUBLE): Price in USD
        - updated_at (TIMESTAMP): Last update timestamp
        
        Valid filter operators:
        - Equality: {"column": "value"}
        - Comparison: {"column >": value}
        - Exclusion: {"column NOT": "value"}
      constraints:
        - "Prefer recently updated products"
      max_subqueries: 3
      rrf_k: 60
      examples:
        - query: "cheap Milwaukee drills"
          filters: {"price <": 100, "brand_name": "Milwaukee"}
    rerank:
      top_n: 10
```

## Key Configuration Fields

### `decomposition_model`
Use a smaller, faster model (GPT-3.5, Llama 3 8B) for decomposition to keep latency low while the main agent uses a larger model for synthesis.

### `schema_description`
Critical for filter translation. Must include:
- Exact column names and types
- Valid filter syntax for Databricks Vector Search
- Example values when helpful

### `examples`
Few-shot examples teach the LLM your metadata "dialect":
```yaml
examples:
  - query: "cheap Milwaukee drills"
    filters: {"price <": 100, "brand_name": "Milwaukee"}
  - query: "exterior paint from last month"
    filters: {"category": "Exterior Paint", "updated_at >": "2025-12-01"}
```

### `rrf_k`
RRF constant (default: 60). Lower values weight top ranks more heavily.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│  Query Decomposition (LLM)      │
│  "cheap Milwaukee drills"       │
│           ↓                     │
│  [SubQuery1, SubQuery2, ...]    │
│  + metadata filters             │
└─────────────────────────────────┘
    │
    ▼ (parallel execution)
┌─────────────────────────────────┐
│  Vector Search × N              │
│  Each with its own filters      │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  RRF Merge                      │
│  Rank-based fusion              │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Rerank (FlashRank)             │
│  Final precision pass           │
└─────────────────────────────────┘
    │
    ▼
  Results
```

## Performance Tuning

### Decomposition Model
- Use a smaller model (GPT-3.5, Llama 3 8B) for speed
- Larger models improve filter accuracy but add latency

### `max_subqueries`
- 2-3 for simple queries
- 3-5 for complex multi-intent queries
- More subqueries = broader coverage but higher latency

### `rrf_k`
- Default 60 works well for most cases
- Lower (30-40) for denser vector spaces
- Higher (80-100) for sparser spaces

## Fallback Behavior

If decomposition fails (LLM error, parsing error), the system automatically falls back to standard single-query search. This ensures robustness in production.

## Next Steps

- **03_reranking/** - Combine with FlashRank for maximum precision
- **15_complete_applications/** - See instructed retrieval in production apps
