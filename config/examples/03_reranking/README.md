# 03. Reranking

**Improve search result relevance with semantic and instruction-aware reranking**

Improve search quality by reranking initial results using a cross-encoder or LLM-based reranker.

## Architecture Overview

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#1565c0'}}}%%
flowchart TB
    subgraph Query["📝 User Query"]
        Q["Best cordless drill for DIY projects"]
    end

    subgraph Stage1["🔍 Stage 1: Initial Retrieval"]
        VS["Vector Search<br/>━━━━━━━━━━━━━━━━<br/>Fast embedding lookup<br/>Top 100 candidates"]
        Results1["📋 100 Results<br/><i>Broad recall</i>"]
    end

    subgraph Stage2["🎯 Stage 2: Reranking"]
        Reranker["Cross-Encoder Reranker<br/>━━━━━━━━━━━━━━━━<br/>Score each (query, doc) pair<br/>Semantic similarity"]
        Results2["📋 Top 10 Results<br/><i>Precise ranking</i>"]
    end

    subgraph Response["📤 Final Response"]
        Best["🥇 Best matches"]
    end

    Q --> VS
    VS --> Results1
    Results1 --> Reranker
    Reranker --> Results2
    Results2 --> Best

    style Stage1 fill:#e3f2fd,stroke:#1565c0
    style Stage2 fill:#e8f5e9,stroke:#2e7d32
```

## Examples

| File | Description | Use Case |
|------|-------------|----------|
| [`vector_search_with_reranking.yaml`](./vector_search_with_reranking.yaml) | Vector search + FlashRank reranking | High-quality semantic search with minimal latency |
| [`instruction_aware_reranking.yaml`](./instruction_aware_reranking.yaml) | FlashRank + LLM instruction-aware reranking | Constraint-aware reranking for price/brand/category |

## Why Reranking?

```mermaid
%%{init: {'theme': 'base'}}%%
graph TB
    subgraph Problem["❌ Without Reranking"]
        P1["Vector search is fast but imprecise"]
        P2["Embeddings miss nuance"]
        P3["Good recall, poor precision"]
    end
    
    subgraph Solution["✅ With Reranking"]
        S1["Cross-encoder sees query + doc together"]
        S2["Captures semantic nuance"]
        S3["Good recall AND precision"]
    end

    style Problem fill:#ffebee,stroke:#c62828
    style Solution fill:#e8f5e9,stroke:#2e7d32
```

**Key Topics:**
- **Reranking Basics** - How reranking improves search quality
- **FlashRank Integration** - Fast, efficient cross-encoder models
- **Instruction-Aware Reranking** - LLM-based constraint prioritization
- **Performance Trade-offs** - Balance between quality and latency
- **Configuration Patterns** - Setting up reranking pipelines

## Two-Stage Process

```mermaid
%%{init: {'theme': 'base'}}%%
sequenceDiagram
    autonumber
    participant 👤 as User
    participant 🔍 as Vector Search
    participant 🎯 as Reranker
    participant 🤖 as Agent

    👤->>🔍: "Best drill for DIY"
    🔍->>🔍: Embed query
    🔍->>🔍: Find top 100 by cosine similarity
    🔍-->>🎯: 100 candidates
    
    loop For each candidate
        🎯->>🎯: Score (query, document) pair
    end
    
    🎯->>🎯: Sort by relevance score
    🎯-->>🤖: Top 10 most relevant
    🤖-->>👤: The best drill for DIY is...
```

## Configuration

```yaml
resources:
  vector_stores:
    products_store: &products_store
      catalog_name: retail_consumer_goods
      schema_name: hardware_store
      index_name: products_vs_index
      columns:
        - product_name
        - description
        - category

  rerankers:
    product_reranker: &product_reranker
      type: cross_encoder           # or 'llm' for LLM-based
      model: databricks-gte-large-en
      top_k: 10                      # Final results count

tools:
  search_tool: &search_tool
    name: search_products
    function:
      type: factory
      name: dao_ai.tools.create_vector_search_tool
      args:
        vector_store: *products_store
        reranker: *product_reranker   # ← Add reranking
        num_results: 100              # ← Initial retrieval count
```

## Reranker Types

```mermaid
%%{init: {'theme': 'base'}}%%
graph TB
    subgraph Types["🎯 Reranker Types"]
        subgraph CrossEncoder["🔀 Cross-Encoder"]
            CE1["<b>type: cross_encoder</b>"]
            CE2["• Fast (batch processing)"]
            CE3["• Purpose-built for ranking"]
            CE4["• Lower cost"]
        end
        
        subgraph LLM["🧠 LLM-Based"]
            LLM1["<b>type: llm</b>"]
            LLM2["• More nuanced"]
            LLM3["• Can follow instructions"]
            LLM4["• Higher cost"]
        end
    end

    style CrossEncoder fill:#e3f2fd,stroke:#1565c0
    style LLM fill:#e8f5e9,stroke:#2e7d32
```

### Cross-Encoder Configuration

```yaml
rerankers:
  cross_encoder_reranker: &cross_encoder_reranker
    type: cross_encoder
    model: databricks-gte-large-en
    top_k: 10
```

### LLM-Based Configuration

```yaml
rerankers:
  llm_reranker: &llm_reranker
    type: llm
    model: *default_llm
    top_k: 10
    prompt: |
      Rate the relevance of this document to the query.
      Query: {query}
      Document: {document}
      Score (0-10):
```

## Performance Trade-offs

```mermaid
%%{init: {'theme': 'base'}}%%
graph LR
    subgraph Tradeoffs["⚖️ Performance Trade-offs"]
        subgraph Fast["⚡ Faster"]
            F1["Smaller num_results"]
            F2["Lower top_k"]
            F3["Cross-encoder model"]
        end
        
        subgraph Quality["🎯 Higher Quality"]
            Q1["Larger num_results"]
            Q2["Higher top_k"]
            Q3["LLM reranker"]
        end
    end

    style Fast fill:#e3f2fd,stroke:#1565c0
    style Quality fill:#e8f5e9,stroke:#2e7d32
```

| Setting | Trade-off |
|---------|-----------|
| `num_results: 50` | Faster, might miss relevant docs |
| `num_results: 200` | Slower, better recall |
| `top_k: 5` | Focused results |
| `top_k: 20` | More comprehensive |

## Quick Start

```bash
# Run with reranking
dao-ai chat -c config/examples/03_reranking/vector_search_with_reranking.yaml

# Compare results
> Search for cordless drills
# Notice: Results are more relevant to intent
```

Ask questions about your data - results will be semantically reranked for better relevance.

## Detailed Reranking Guide

### The Problem
Standard vector search retrieves the top-k results based on vector similarity, but:
- Vector similarity doesn't always match semantic relevance
- Initial retrieval may miss nuanced query intent
- Results may not be ordered optimally for the specific question

### The Solution
Reranking adds a second-stage model that:
1. Takes the initial top-k results (e.g., top 100)
2. Reorders them using a more sophisticated model
3. Returns the top-n most relevant results (e.g., top 5)

### The Benefit
- **Better Relevance**: 20-40% improvement in result quality
- **Minimal Latency**: Only reranks a small set of candidates
- **Cost Effective**: Reranking is faster than re-querying with better embeddings

## Reranking Models

### FlashRank (Recommended)
- **Speed**: Very fast (~10ms per query)
- **Quality**: Excellent for most use cases
- **Size**: Lightweight (~100MB)
- **Local**: Runs locally, no API calls

```yaml
reranker:
  model: flashrank
  top_n: 5
  num_candidates: 20
```

### Custom Rerankers
You can implement custom reranking logic:

```yaml
reranker:
  model: custom
  function: my_package.custom_reranker
  top_n: 5
```

## When to Use Reranking

### ✅ Use Reranking When:
- **Precision matters**: Need the absolute best results
- **Complex queries**: Multi-faceted or nuanced questions
- **Diverse corpus**: Documents vary significantly in style/format
- **User-facing search**: End-user applications where quality is critical

### ❌ Skip Reranking When:
- **Simple lookups**: Exact match queries
- **Small result sets**: Retrieving < 10 results anyway
- **Latency critical**: Need sub-10ms response times
- **Limited compute**: Resource-constrained environments

## Architecture Patterns

### Pattern 1: Two-Stage Retrieval (Recommended)

```
Query → Vector Search (top 100) → Rerank (top 5) → Agent
```

- Fast initial retrieval
- High-quality final results
- Best balance of speed and accuracy

### Pattern 2: FlashRank + Instruction-Aware

```
Query → Vector Search (top 50) → FlashRank (top 20) → Instruction-Aware LLM (top 10) → Agent
```

- FlashRank for semantic relevance (~10ms)
- LLM for constraint prioritization (~100ms)
- Best for queries with explicit constraints (price, brand, category)

## Instruction-Aware Reranking

Instruction-aware reranking adds an LLM stage after FlashRank to consider user constraints. It is configured under `instructed.rerank` because it depends on the schema context provided by the `instructed` block:

```yaml
rerank:
  model: ms-marco-MiniLM-L-12-v2
  top_n: 20                          # FlashRank outputs 20 candidates
instructed:
  schema_description: "..."
  rerank:
    model: *fast_llm                 # Use small model for speed
    instructions: |
      Prioritize results matching price and brand constraints.
    top_n: 10                        # Final count after instruction reranking
```

### When to Use Instruction-Aware Reranking
- Queries with explicit price constraints ("under $100")
- Brand preferences ("Milwaukee", "not DeWalt")
- Category requirements ("power tools")
- When FlashRank alone misses nuanced user intent

### Latency Comparison

| Configuration | Latency | Use Case |
|--------------|---------|----------|
| FlashRank only | ~110ms | General queries |
| FlashRank + Instruction-Aware | ~210ms | Constrained queries |

## Best Practices

```mermaid
%%{init: {'theme': 'base'}}%%
graph TB
    subgraph Best["✅ Best Practices"]
        BP1["📊 Set num_results 5-10x larger than top_k"]
        BP2["⚡ Use cross-encoder for speed"]
        BP3["🧠 Use LLM for complex queries"]
        BP4["📈 Monitor latency vs quality"]
    end

    style Best fill:#e8f5e9,stroke:#2e7d32
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Slow reranking | Reduce num_results, use cross-encoder |
| Poor results | Increase num_results, try LLM reranker |
| Missing relevant docs | Increase num_results in initial retrieval |

## Next Steps

- **04_genie/** - Add caching for repeated queries
- **02_mcp/** - Use MCP for vector search
- **10_agent_integrations/** - Combine with other tools

## Related Documentation

- [Reranking Configuration](../../../docs/key-capabilities.md#reranking)
- [Vector Search](../../../docs/configuration-reference.md#vector-stores)
