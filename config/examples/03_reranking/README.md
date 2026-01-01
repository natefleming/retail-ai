# 03. Reranking

**Improve search result relevance with semantic reranking**

Reranking refines search results by using advanced models to reorder results based on semantic similarity to the query, dramatically improving result quality without changing the initial retrieval.

## Examples

| File | Description | Use Case |
|------|-------------|----------|
| `vector_search_with_reranking.yaml` | Vector search + FlashRank reranking | High-quality semantic search with minimal latency |

## What You'll Learn

- **Reranking Basics** - How reranking improves search quality
- **FlashRank Integration** - Fast, efficient reranking models
- **Performance Trade-offs** - Balance between quality and latency
- **Configuration Patterns** - Setting up reranking pipelines

## Quick Start

```bash
dao-ai chat -c config/examples/03_reranking/vector_search_with_reranking.yaml
```

Ask questions about your data - results will be semantically reranked for better relevance.

## Why Reranking?

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

## Configuration Pattern

```yaml
tools:
  search_tool:
    name: search_documents
    function:
      type: vector_search
      vector_search_index: catalog.schema.index_name
      columns: [content, title, url]
      reranker:
        model: flashrank
        top_n: 5              # Return top 5 after reranking
        num_candidates: 20    # Rerank from top 20 retrieved
```

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

## Performance Tuning

### Top-N Selection
- **top_n: 3-5** - Best for focused questions
- **top_n: 10-15** - Better for broad topics
- **top_n: 20+** - When you need comprehensive coverage

### Candidate Pool Size
- **num_candidates: 20** - Fast, good for simple queries
- **num_candidates: 50** - Balanced quality/speed
- **num_candidates: 100** - Best quality, slower

### Rule of Thumb
Set `num_candidates` to 3-5x your `top_n` value.

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

### Pattern 2: Multi-Pass Reranking
```
Query → Vector Search (top 200) → Coarse Rerank (top 50) → Fine Rerank (top 5) → Agent
```
- Maximum quality
- Higher latency
- Use for critical applications

### Pattern 3: Hybrid Retrieval + Reranking
```
Query → [Vector Search + Keyword Search] → Merge → Rerank (top 5) → Agent
```
- Best of both retrieval methods
- Reranking unifies the results
- Excellent for diverse queries

## Best Practices

### 1. Choose Appropriate Candidate Pool
```yaml
# Good: 4x ratio
top_n: 5
num_candidates: 20

# Better: 5x ratio for complex queries
top_n: 10
num_candidates: 50
```

### 2. Cache Reranked Results
Reranking is deterministic - cache results for identical queries:

```yaml
tools:
  search_tool:
    function:
      type: vector_search
      reranker:
        model: flashrank
        top_n: 5
        num_candidates: 20
    cache:
      type: semantic
      ttl: 3600  # Cache for 1 hour
```

### 3. Monitor Performance
Track these metrics:
- **Retrieval time**: Initial vector search
- **Reranking time**: Reranker inference
- **Total latency**: End-to-end time
- **Result relevance**: User feedback/clicks

### 4. A/B Test Configuration
Test different `top_n` and `num_candidates` values:

```python
# Baseline
top_n=5, num_candidates=20

# Test variants
top_n=5, num_candidates=50   # More candidates
top_n=10, num_candidates=50  # More results
```

## Prerequisites

### For FlashRank
- ✅ `flashrank` Python package installed
- ✅ First run downloads model (~100MB)
- ✅ Sufficient RAM (models use ~200-500MB)

### For Vector Search
- ✅ Databricks Vector Search index created
- ✅ Index populated with embeddings
- ✅ Proper permissions to query index

## Troubleshooting

**"FlashRank model not found"**
- Run once to download the model
- Check internet connectivity
- Verify disk space for model cache

**"Reranking too slow"**
- Reduce `num_candidates` (e.g., 50 → 20)
- Reduce `top_n` (e.g., 10 → 5)
- Consider caching reranked results

**"Results not improving"**
- Increase `num_candidates` to give reranker more options
- Verify initial retrieval is working well
- Check if reranker model is appropriate for your domain

**"Out of memory errors"**
- Reduce `num_candidates`
- Use smaller reranker model
- Increase available RAM

## Performance Benchmarks

Typical latencies for FlashRank:

| Candidates | Top-N | Reranking Time | Total Time* |
|------------|-------|----------------|-------------|
| 20 | 5 | ~5ms | ~105ms |
| 50 | 10 | ~12ms | ~112ms |
| 100 | 20 | ~25ms | ~125ms |

*Total time includes vector search (~100ms) + reranking

## Advanced Topics

### Custom Reranking Functions

Implement your own reranking logic:

```python
from typing import List, Dict, Any

def custom_reranker(
    query: str,
    documents: List[Dict[str, Any]],
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Custom reranking function.
    
    Args:
        query: User's search query
        documents: Retrieved documents with 'content' field
        top_n: Number of documents to return
    
    Returns:
        Reranked documents (top_n most relevant)
    """
    # Your reranking logic here
    scores = compute_relevance_scores(query, documents)
    ranked_docs = sort_by_scores(documents, scores)
    return ranked_docs[:top_n]
```

### Combining Multiple Rerankers

Use ensemble reranking for maximum quality:

```yaml
reranker:
  type: ensemble
  models:
    - flashrank
    - custom_domain_reranker
  weights: [0.7, 0.3]
  top_n: 5
  num_candidates: 50
```

## Next Steps

👉 **04_genie/** - Cache reranked results for performance  
👉 **02_mcp/** - Combine with other tool integrations  
👉 **11_complete_applications/** - See reranking in production

## Related Documentation

- [Vector Search](../02_mcp/README.md#vector-search)
- [Caching Strategies](../04_genie/README.md)
- [Performance Optimization](../../../docs/key-capabilities.md#performance)

## Example Output

### Without Reranking
```
Query: "How do I reset my password?"

Top Results:
1. General FAQ page (moderate relevance)
2. Account settings overview (low relevance)  
3. Password reset guide (high relevance) ← Should be #1!
4. Security best practices (moderate relevance)
5. Login troubleshooting (moderate relevance)
```

### With Reranking
```
Query: "How do I reset my password?"

Top Results (Reranked):
1. Password reset guide (high relevance) ✓
2. Login troubleshooting (high relevance) ✓
3. Account settings overview (moderate relevance)
4. Security best practices (moderate relevance)
5. General FAQ page (low relevance)
```

The reranker correctly identifies and promotes the most relevant document to the top!
