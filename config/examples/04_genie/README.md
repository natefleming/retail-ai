# 04. Genie

**Natural language to SQL with Databricks Genie**

Query your data using natural language. This directory shows basic Genie integration and performance optimization with caching strategies.

## Examples

| File | Description | Use Case |
|------|-------------|----------|
| `genie_basic.yaml` | Simple Genie integration | Natural language to SQL |
| `genie_lru_cache.yaml` | Genie with LRU caching | Fast in-memory caching |
| `genie_semantic_cache.yaml` | Genie with semantic caching | Semantic similarity matching |

## What You'll Learn

- **Genie Basics** - Connect to Genie spaces and query data
- **LRU Caching** - Simple, fast in-memory caching
- **Semantic Caching** - Match similar queries with embeddings
- **Performance Optimization** - Reduce costs by 50-90%

## Quick Start

### Basic Genie
```bash
dao-ai chat -c config/examples/04_genie/genie_basic.yaml
```

Ask: "What are the top selling products this month?"

### With Caching
```bash
# LRU cache - exact matches
dao-ai chat -c config/examples/04_genie/genie_lru_cache.yaml

# Semantic cache - similar questions
dao-ai chat -c config/examples/04_genie/genie_semantic_cache.yaml
```

## Configuration Pattern

```yaml
tools:
  genie_tool:
    name: genie
    function:
      type: factory
      name: dao_ai.tools.create_genie_tool
      args:
        genie_room:
          space_id: your_space_id
    cache:
      type: lru  # or semantic
      max_size: 100
      ttl: 3600
```

## Caching Strategies

### LRU Cache
- **Best for**: Exact query repetition
- **Hit rate**: 10-30%
- **Setup**: Simple, no dependencies

### Semantic Cache
- **Best for**: Natural language variations
- **Hit rate**: 40-70%
- **Setup**: Requires vector search index

## Performance

Example with 1000 queries/day:

| Strategy | API Calls | Cost Savings | Latency |
|----------|-----------|--------------|---------|
| No cache | 1000 | 0% | ~5s |
| LRU (20% hits) | 800 | 20% | ~4s |
| Semantic (60% hits) | 400 | 60% | ~2s |

## Prerequisites

### For Genie
- ✅ Databricks Genie space created
- ✅ Space ID from Genie settings
- ✅ Query permissions

### For Semantic Cache (additional)
- ✅ Vector Search index configured
- ✅ Embedding model endpoint

## Best Practices

1. **Start simple**: Begin with basic Genie, add caching later
2. **Monitor metrics**: Track hit rate, latency, and costs
3. **Tune TTL**: Balance freshness vs performance
4. **Choose cache type**: LRU for repetition, semantic for variations

## Next Steps

👉 **05_memory/** - Add persistent conversation memory  
👉 **11_complete_applications/** - See Genie in production

## Related Documentation

- [Genie Integration](../../../docs/key-capabilities.md#genie)
- [Caching Strategies](../../../docs/key-capabilities.md#caching)
