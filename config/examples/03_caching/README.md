# 03. Caching

**Improve performance and reduce costs through intelligent caching**

Caching strategies that dramatically improve response times and reduce API/LLM costs.

## Examples

| File | Description | Cost Reduction |
|------|-------------|----------------|
| `genie_lru_cache.yaml` | LRU (Least Recently Used) in-memory cache | 50-70% for repeated queries |
| `genie_semantic_cache.yaml` | Two-tier semantic caching with embeddings | 60-80% for similar queries |

## What You'll Learn

- **LRU caching** - Fast, in-memory caching for exact matches
- **Semantic caching** - Embedding-based similarity for query variations
- **Two-tier strategy** - Combine exact + semantic for optimal results
- **Cache configuration** - Size limits, TTL, similarity thresholds

## Caching Strategies

### LRU Cache (Level 1)
```
User Query → Hash → Cache Lookup → Hit? Return : Execute + Cache
```
- **Exact match** only
- **In-memory** (fast but per-instance)
- **Best for**: Identical repeated queries

### Semantic Cache (Level 2)
```
User Query → Embedding → Similarity Search → Hit? Return : Execute + Cache
```
- **Similar queries** matched (e.g., "weather today" ≈ "what's the weather")
- **PostgreSQL/Lakebase** (shared across instances)
- **Context-aware** (considers conversation history)
- **Best for**: Natural language variations

### Two-Tier (L1 + L2)
Combines both for maximum hit rate: exact match first, then semantic similarity.

## Quick Start

### Test LRU cache
```bash
dao-ai chat -c config/examples/03_caching/genie_lru_cache.yaml
```

Try asking the same question twice - second response will be instant!

### Test semantic cache
```bash
dao-ai chat -c config/examples/03_caching/genie_semantic_cache.yaml
```

Try variations: "top customers" vs "best customers" vs "highest spending customers"

## Prerequisites

### For LRU Cache
- ✅ No additional requirements (in-memory only)

### For Semantic Cache
- ✅ PostgreSQL or Databricks Lakebase
- ✅ Embedding model endpoint
- ✅ Databricks warehouse for SQL execution

## Configuration

### LRU Cache Parameters
```yaml
lru_cache_parameters:
  max_size: 100           # Max cached items
  ttl: 3600               # Time-to-live (seconds)
```

### Semantic Cache Parameters
```yaml
semantic_cache_parameters:
  database: *postgres_db
  similarity_threshold: 0.85    # 0.8-0.95 recommended
  embedding_model: *embed_model
  max_results: 5               # Top K similar queries
```

## Performance Tuning

**LRU Cache:**
- Increase `max_size` for more cached queries (but uses more memory)
- Adjust `ttl` based on data freshness requirements

**Semantic Cache:**
- **Lower threshold** (0.8): More cache hits, less precision
- **Higher threshold** (0.95): Fewer hits, more precision
- Monitor hit rates and adjust

## Cache Hit Rates

Typical improvements:
- 📊 **LRU only**: 50-70% hit rate for repeated queries
- 📊 **Semantic only**: 40-60% hit rate including variations
- 📊 **Two-tier**: 70-85% combined hit rate

## Next Steps

👉 **04_memory/** - Add persistent conversation state  
👉 **05_quality_control/** - Ensure cached responses meet quality standards

## Related Documentation

- [Genie Caching Architecture](../../../docs/key-capabilities.md#genie-caching)
- [Performance Optimization Guide](../../../docs/faq.md)

