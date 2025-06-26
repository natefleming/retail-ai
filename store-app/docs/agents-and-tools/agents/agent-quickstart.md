# Agent Quickstart Guide

This guide provides a quick start for building your own retail AI agents and optimizing their performance.

## Building Your Own Agent

### Quick Start Guide

1. **Define Agent Purpose** - Determine the specific retail domain and capabilities
2. **Select Appropriate Tools** - Choose tools based on performance and functionality needs
3. **Configure Agent** - Set up prompts, guardrails, and tool orchestration
4. **Add Guardrails** - Implement safety, quality, and business rule checks

### Example Agent Structure

```python
def my_retail_agent(model_config: ModelConfig) -> AgentCallable:
    # Extract configuration
    model = model_config.get("agents").get("my_agent").get("model").get("name")
    
    @mlflow.trace()
    def agent(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
        # Initialize LLM and tools
        llm = ChatDatabricks(model=model, temperature=0.1)
        tools = [
            # Select appropriate tools for your agent
            create_find_product_by_sku_tool(warehouse_id),
            find_product_details_by_description_tool(...)
        ]
        
        # Create and configure agent
        agent = create_react_agent(model=llm, tools=tools, prompt=prompt)
        
        # Apply guardrails
        for guardrail in guardrails:
            agent = with_guardrails(agent, guardrail)
        
        return agent
    
    return agent
```

### Step-by-Step Implementation

#### 1. Define Your Agent's Purpose

```python
# Example: Customer Support Agent
AGENT_PURPOSE = {
    "name": "customer_support",
    "domain": "Customer service and order management",
    "capabilities": [
        "Order status lookup",
        "Return policy information", 
        "Product recommendations",
        "Store information"
    ],
    "target_queries": [
        "Where is my order?",
        "What's your return policy?",
        "Can you recommend similar products?"
    ]
}
```

#### 2. Select Appropriate Tools

```python
def select_tools_for_customer_support(warehouse_id: str) -> list:
    """Select tools for customer support agent."""
    
    return [
        # Order management
        create_order_lookup_tool(warehouse_id),
        
        # Product information
        create_find_product_by_sku_tool(warehouse_id),
        find_product_details_by_description_tool(
            endpoint_name="vs_endpoint",
            index_name="products_index",
            columns=["product_name", "description", "price"]
        ),
        
        # Policy information
        create_policy_search_tool(),
        
        # Store information
        create_store_locator_tool()
    ]
```

#### 3. Configure Agent Prompt

```python
CUSTOMER_SUPPORT_PROMPT = """You are a helpful customer support representative at BrickMart store {store_num}.
Your goal is to assist customers with their orders, returns, and product questions.

Guidelines:
- Be friendly and professional
- Use the available tools to find accurate information
- If you can't find specific information, offer to connect them with a specialist
- Always provide order numbers and reference information when available

Available tools:
- Order lookup by order number
- Product search and information
- Store policy information
- Store location and hours

Current customer: {user_id}
Store: {store_num}
"""
```

#### 4. Implement Guardrails

```python
def customer_support_guardrails() -> list:
    """Define guardrails for customer support agent."""
    
    return [
        {
            "type": "content_safety",
            "config": {"block_harmful": True}
        },
        {
            "type": "business_rules", 
            "config": {
                "require_order_verification": True,
                "privacy_protection": True
            }
        },
        {
            "type": "quality_check",
            "config": {"min_response_length": 50}
        }
    ]
```

#### 5. Complete Agent Implementation

```python
def customer_support_node(model_config: ModelConfig) -> AgentCallable:
    """Complete customer support agent implementation."""
    
    # Extract configuration
    model = model_config.get("agents").get("customer_support").get("model").get("name")
    warehouse_id = model_config.get("warehouse_id")
    
    @mlflow.trace()
    def customer_support_agent(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
        # Initialize LLM
        llm = ChatDatabricks(model=model, temperature=0.1)
        
        # Format prompt with context
        prompt = CUSTOMER_SUPPORT_PROMPT.format(
            store_num=state["store_num"],
            user_id=state["user_id"]
        )
        
        # Select and configure tools
        tools = select_tools_for_customer_support(warehouse_id)
        
        # Create agent
        agent = create_react_agent(
            model=llm,
            prompt=prompt,
            tools=tools
        )
        
        # Apply guardrails
        guardrails = customer_support_guardrails()
        for guardrail_config in guardrails:
            guardrail = create_guardrail(guardrail_config)
            agent = with_guardrails(agent, guardrail)
        
        # Execute agent
        result = agent.invoke(state)
        
        return {"messages": [result]}
    
    return customer_support_agent
```

### Testing Your Agent

```python
def test_agent(agent_func, test_cases: list):
    """Test your agent with various scenarios."""
    
    for test_case in test_cases:
        print(f"Testing: {test_case['description']}")
        
        state = AgentState(
            messages=[HumanMessage(content=test_case['input'])],
            user_id="test_user",
            store_num="101"
        )
        
        try:
            result = agent_func(state, {})
            print(f"Response: {result['messages'][-1].content}")
            print("Test passed\n")
        except Exception as e:
            print(f"Test failed: {e}\n")

# Example test cases
test_cases = [
    {
        "description": "Order status inquiry",
        "input": "What's the status of order #12345?"
    },
    {
        "description": "Product recommendation",
        "input": "Can you recommend wireless headphones under $100?"
    },
    {
        "description": "Return policy question",
        "input": "What's your return policy for electronics?"
    }
]

test_agent(customer_support_agent, test_cases)
```

---

## Performance & Optimization

### Agent Performance Summary

| Agent Type | Avg Response Time | Success Rate | Primary Use Cases |
|------------|------------------|--------------|-------------------|
| **Product** | 1.2s | 98.5% | Product lookup, specifications |
| **Inventory** | 0.8s | 99.2% | Stock checks, availability |
| **Comparison** | 2.1s | 94.1% | Product analysis, recommendations |
| **DIY** | 3.2s | 91.3% | Tutorials, project guidance |
| **General** | 1.5s | 96.8% | Policies, customer service |

### Optimization Strategies

#### 1. Tool Selection Optimization

Choose the fastest appropriate tools for each query type:

```python
def optimize_tool_selection(query_type: str, performance_requirements: dict) -> list:
    """Select optimal tools based on performance needs."""
    
    if performance_requirements.get("max_latency", float('inf')) < 1.0:
        # High-performance requirements - use fastest tools
        return [
            create_find_product_by_sku_tool(),  # ~200ms avg
            create_find_inventory_by_sku_tool()  # ~200ms avg
        ]
    
    elif query_type == "semantic_search":
        # Balance performance and capability
        return [
            find_product_details_by_description_tool(),  # ~300ms avg
            create_find_product_by_sku_tool()  # Fallback
        ]
    
    else:
        # Full capability - include analysis tools
        return [
            find_product_details_by_description_tool(),
            create_product_comparison_tool(),  # ~1.5s avg
            create_sku_extraction_tool()
        ]
```

#### 2. Parallel Execution

Run independent tools simultaneously to reduce total response time:

```python
import asyncio

async def parallel_tool_execution(tools: list, query: str) -> dict:
    """Execute multiple tools in parallel."""
    
    # Group tools by execution time
    fast_tools = [t for t in tools if t.avg_latency < 500]  # < 500ms
    slow_tools = [t for t in tools if t.avg_latency >= 500]  # >= 500ms
    
    tasks = []
    
    # Execute fast tools first
    if fast_tools:
        fast_tasks = [tool.ainvoke(query) for tool in fast_tools]
        tasks.extend(fast_tasks)
    
    # Execute slow tools in parallel
    if slow_tools:
        slow_tasks = [tool.ainvoke(query) for tool in slow_tools]
        tasks.extend(slow_tasks)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Combine successful results
    combined_results = {}
    for i, result in enumerate(results):
        if not isinstance(result, Exception):
            combined_results[f"tool_{i}"] = result
    
    return combined_results
```

#### 3. Caching Implementation

Cache frequent queries and tool results:

```python
from functools import lru_cache
import redis

# In-memory cache for frequent queries
@lru_cache(maxsize=1000)
def cached_product_lookup(sku: str) -> dict:
    """Cache product lookups in memory."""
    return find_product_by_sku_tool([sku])

# Redis cache for session data
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cached_vector_search(query: str, ttl: int = 3600) -> list:
    """Cache vector search results in Redis."""
    cache_key = f"vector_search:{hash(query)}"
    
    # Try cache first
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return json.loads(cached_result)
    
    # Execute search and cache result
    result = find_product_details_by_description_tool(query)
    redis_client.setex(cache_key, ttl, json.dumps(result))
    
    return result
```

#### 4. Efficient Guardrails

Implement lightweight, efficient safety and quality checks:

```python
def efficient_guardrails():
    """Implement efficient guardrails that don't impact performance."""
    
    @mlflow.trace(span_type="GUARDRAIL", name="efficient_check")
    def quick_guardrail_check(state: AgentState) -> dict:
        last_message = state["messages"][-1].content
        
        # Quick keyword-based safety check (< 1ms)
        if any(word in last_message.lower() for word in BLOCKED_KEYWORDS):
            return safe_fallback_response()
        
        # Quick length check (< 1ms)
        if len(last_message) < 10:
            return request_clarification_response()
        
        # Quick business rule check (< 5ms)
        if needs_disclaimer(last_message):
            return add_disclaimer(last_message)
        
        return state
    
    return quick_guardrail_check
```

### Performance Monitoring

```python
import time
from functools import wraps

def monitor_performance(func):
    """Decorator to monitor agent performance."""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log performance metrics
            mlflow.log_metric("execution_time", execution_time)
            mlflow.log_metric("success", 1)
            
            return result
        
        except Exception as e:
            execution_time = time.time() - start_time
            mlflow.log_metric("execution_time", execution_time)
            mlflow.log_metric("success", 0)
            raise
    
    return wrapper

# Usage
@monitor_performance
def my_optimized_agent(state, config):
    # Agent implementation
    pass
```

### Quick Performance Tips

1. **Use Unity Catalog functions** for exact lookups (fastest)
2. **Implement caching** for repeated queries
3. **Run tools in parallel** when possible
4. **Keep prompts concise** to reduce token usage
5. **Use appropriate model parameters** (lower temperature for factual responses)
6. **Implement circuit breakers** to prevent cascading failures
7. **Monitor and alert** on performance degradation

---

## Related Documentation

- **[Agent Development Patterns](agent-development-patterns.md)** - Advanced implementation patterns and techniques
- **[Agent Performance](agent-performance.md)** - Detailed performance metrics, optimization, and monitoring
- **[Agent Best Practices](agent-best-practices.md)** - Comprehensive guidelines for development and deployment
- **[Agent Reference](../references/agent-reference.md)** - Detailed specifications for all implemented agents
- **[Tools Reference](../references/tools-reference.md)** - Complete technical specifications for all tools 