# Agent Best Practices

This guide provides comprehensive best practices for developing, deploying, and maintaining retail AI agents.

## Agent Design Principles

### 1. Single Responsibility Principle

Each agent should have a clear, focused purpose:

```python
# Good: Focused agent
def inventory_agent(state, config):
    """Handles only inventory-related queries."""
    # Specialized for inventory management
    pass

# Bad: Multi-purpose agent
def everything_agent(state, config):
    """Handles products, inventory, orders, and customer service."""
    # Too many responsibilities
    pass
```

### 2. Fail Gracefully

Always provide meaningful fallbacks:

```python
def resilient_agent(state, config):
    try:
        # Primary functionality
        return primary_agent_logic(state, config)
    except ToolException as e:
        # Tool-specific fallback
        return fallback_with_limited_tools(state, config)
    except Exception as e:
        # General fallback
        return {
            "messages": [
                AIMessage(
                    content="I'm experiencing technical difficulties. "
                           "Please try again or contact customer service for assistance."
                )
            ]
        }
```

### 3. Context Awareness

Leverage available context for better responses:

```python
def context_aware_agent(state: AgentState, config: AgentConfig):
    # Use store context
    store_num = state.get("store_num", "unknown")
    
    # Use user history
    user_preferences = state.get("preferences", {})
    
    # Use session data
    previous_queries = state.get("session_data", {}).get("queries", [])
    
    prompt = f"""You are assisting a customer at store {store_num}.
    Previous queries in this session: {previous_queries}
    User preferences: {user_preferences}
    
    Provide personalized assistance based on this context."""
```

---

## Tool Selection Best Practices

### 1. Performance-First Tool Selection

Choose tools based on performance requirements:

```python
def select_optimal_tools(query_type: str, performance_requirements: dict):
    """Select tools based on performance needs."""
    
    if performance_requirements.get("max_latency", float('inf')) < 1.0:
        # High-performance requirements - use fastest tools
        return [
            create_find_product_by_sku_tool(),  # ~200ms
            create_find_inventory_by_sku_tool()  # ~200ms
        ]
    
    elif query_type == "semantic_search":
        # Balance performance and capability
        return [
            find_product_details_by_description_tool(),  # ~300ms
            create_find_product_by_sku_tool()  # Fallback
        ]
    
    else:
        # Full capability - include analysis tools
        return [
            find_product_details_by_description_tool(),
            create_product_comparison_tool(),  # ~1.5s
            create_sku_extraction_tool()
        ]
```

### 2. Tool Composition Patterns

Combine tools effectively:

```python
# Sequential tool usage
def sequential_tool_pattern(query: str):
    # Step 1: Extract entities
    skus = sku_extraction_tool(query)
    
    # Step 2: Lookup products
    if skus:
        products = find_product_by_sku_tool(skus)
    else:
        products = find_product_details_by_description_tool(query)
    
    # Step 3: Check inventory
    inventory = find_inventory_by_sku_tool([p['sku'] for p in products])
    
    return combine_product_and_inventory_data(products, inventory)

# Parallel tool usage
async def parallel_tool_pattern(query: str):
    # Execute independent tools in parallel
    tasks = [
        find_product_details_by_description_tool(query),
        search_tool(query),  # External search
        classification_tool(query)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return combine_parallel_results(results)
```

### 3. Tool Error Handling

Handle tool failures gracefully:

```python
def robust_tool_execution(tools: list, query: str):
    """Execute tools with proper error handling."""
    
    results = {}
    
    for tool in tools:
        try:
            result = tool.invoke(query)
            results[tool.name] = result
            
        except TimeoutError:
            logger.warning(f"Tool {tool.name} timed out")
            # Try faster alternative
            if hasattr(tool, 'fast_alternative'):
                try:
                    result = tool.fast_alternative.invoke(query)
                    results[f"{tool.name}_fallback"] = result
                except Exception:
                    pass
                    
        except Exception as e:
            logger.error(f"Tool {tool.name} failed: {e}")
            # Continue with other tools
            continue
    
    return results
```

---

## Prompt Engineering Best Practices

### 1. Clear Role Definition

Define the agent's role clearly:

```python
# Good: Clear, specific role
PRODUCT_AGENT_PROMPT = """You are a product specialist at BrickMart store {store_num}.
Your expertise is in helping customers find products, understand features, and make informed decisions.

Capabilities:
- Look up products by SKU or UPC
- Search products by description
- Provide detailed product information
- Compare similar products

Guidelines:
- Always use exact SKU lookup when SKUs are mentioned
- Provide specific product details including price and availability
- If you don't have information, say so clearly
- Ask clarifying questions when the request is ambiguous
"""

# Bad: Vague role
GENERIC_PROMPT = """You are a helpful assistant. Help the user with their questions."""
```

### 2. Structured Response Format

Guide the agent to provide consistent responses:

```python
STRUCTURED_PROMPT = """When providing product information, use this format:

**Product Name**: [Name]
**SKU**: [SKU]
**Price**: [Price]
**Availability**: [In stock/Out of stock]
**Description**: [Brief description]
**Key Features**: 
- [Feature 1]
- [Feature 2]

If comparing products, use:
**Comparison**: [Product A] vs [Product B]
**Winner**: [Product] for [reason]
**Recommendation**: [Specific recommendation based on use case]
"""
```

### 3. Context Integration

Include relevant context in prompts:

```python
def create_contextual_prompt(agent_type: str, state: AgentState) -> str:
    base_prompt = get_base_prompt(agent_type)
    
    # Add store context
    store_context = f"Store: {state['store_num']}"
    
    # Add user context
    user_context = ""
    if state.get("preferences"):
        user_context = f"User preferences: {state['preferences']}"
    
    # Add conversation context
    conversation_context = ""
    if len(state["messages"]) > 1:
        recent_messages = state["messages"][-3:]  # Last 3 messages
        conversation_context = f"Recent conversation: {recent_messages}"
    
    return f"{base_prompt}\n\n{store_context}\n{user_context}\n{conversation_context}"
```

---

## Guardrails and Safety

### 1. Content Safety

Implement comprehensive content filtering:

```python
def content_safety_guardrail():
    """Multi-layer content safety checking."""
    
    @mlflow.trace(span_type="GUARDRAIL", name="content_safety")
    def safety_check(state: AgentState) -> dict:
        last_message = state["messages"][-1].content
        
        # Layer 1: Keyword filtering
        inappropriate_keywords = ["harmful", "illegal", "offensive"]
        if any(keyword in last_message.lower() for keyword in inappropriate_keywords):
            return safe_response("I can only help with retail-related questions.")
        
        # Layer 2: ML-based classification
        safety_score = content_classifier.predict(last_message)
        if safety_score > 0.8:  # High risk
            return safe_response("Let me help you with product or store information instead.")
        
        # Layer 3: Business context validation
        if not is_retail_related(last_message):
            return safe_response("I specialize in retail assistance. How can I help you find products or store information?")
        
        return state
    
    return safety_check

def safe_response(message: str) -> dict:
    """Create a safe fallback response."""
    return {
        "messages": [AIMessage(content=message)]
    }
```

### 2. Business Rules Enforcement

Ensure compliance with business policies:

```python
def business_rules_guardrail():
    """Enforce business rules and policies."""
    
    @mlflow.trace(span_type="GUARDRAIL", name="business_rules")
    def rules_check(state: AgentState) -> dict:
        last_message = state["messages"][-1].content
        
        # Price accuracy disclaimer
        if contains_price_information(last_message):
            disclaimer = "\n\n*Prices are subject to change and may vary by location."
            updated_content = last_message + disclaimer
            return {
                "messages": [AIMessage(content=updated_content)]
            }
        
        # Inventory accuracy disclaimer
        if contains_inventory_information(last_message):
            disclaimer = "\n\n*Inventory levels are updated in real-time but may vary."
            updated_content = last_message + disclaimer
            return {
                "messages": [AIMessage(content=updated_content)]
            }
        
        # Compliance with return policy
        if mentions_returns(last_message):
            if not includes_return_policy_link(last_message):
                policy_link = "\n\nFor complete return policy details, visit: [Return Policy](https://store.com/returns)"
                updated_content = last_message + policy_link
                return {
                    "messages": [AIMessage(content=updated_content)]
                }
        
        return state
    
    return rules_check
```

### 3. Quality Assurance

Implement response quality checks:

```python
def quality_assurance_guardrail():
    """Ensure response quality and completeness."""
    
    @mlflow.trace(span_type="GUARDRAIL", name="quality_assurance")
    def quality_check(state: AgentState) -> dict:
        last_message = state["messages"][-1].content
        
        # Check response length
        if len(last_message) < 20:
            return {
                "messages": [
                    AIMessage(content="Let me provide a more detailed response. Could you please clarify what specific information you're looking for?")
                ]
            }
        
        # Check for completeness
        user_query = state["messages"][-2].content if len(state["messages"]) > 1 else ""
        if not addresses_user_query(last_message, user_query):
            return {
                "messages": [
                    AIMessage(content="I want to make sure I'm addressing your question properly. Could you help me understand what specific information you need?")
                ]
            }
        
        # Check for accuracy indicators
        confidence_score = calculate_response_confidence(last_message)
        if confidence_score < 0.7:
            hedged_response = add_confidence_hedging(last_message)
            return {
                "messages": [AIMessage(content=hedged_response)]
            }
        
        return state
    
    return quality_check
```

---

## Monitoring and Observability

### 1. Comprehensive Logging

Log all important events and metrics:

```python
import structlog

logger = structlog.get_logger()

def log_agent_execution(agent_name: str):
    """Decorator for comprehensive agent logging."""
    
    def decorator(func):
        @wraps(func)
        def wrapper(state: AgentState, config: AgentConfig):
            execution_id = str(uuid.uuid4())
            
            # Log start
            logger.info(
                "agent_execution_start",
                execution_id=execution_id,
                agent_name=agent_name,
                user_id=state.get("user_id"),
                store_num=state.get("store_num"),
                query=state["messages"][-1].content[:100]  # First 100 chars
            )
            
            start_time = time.time()
            
            try:
                result = func(state, config)
                
                # Log success
                execution_time = time.time() - start_time
                logger.info(
                    "agent_execution_success",
                    execution_id=execution_id,
                    agent_name=agent_name,
                    execution_time=execution_time,
                    response_length=len(result["messages"][-1].content)
                )
                
                return result
                
            except Exception as e:
                # Log error
                execution_time = time.time() - start_time
                logger.error(
                    "agent_execution_error",
                    execution_id=execution_id,
                    agent_name=agent_name,
                    execution_time=execution_time,
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise
        
        return wrapper
    return decorator
```

### 2. Performance Metrics

Track key performance indicators:

```python
class AgentMetrics:
    """Centralized metrics collection for agents."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def record_execution(self, agent_name: str, execution_time: float, success: bool):
        """Record agent execution metrics."""
        self.metrics[f"{agent_name}_execution_time"].append(execution_time)
        self.metrics[f"{agent_name}_success"].append(1 if success else 0)
    
    def record_tool_usage(self, tool_name: str, execution_time: float, success: bool):
        """Record tool usage metrics."""
        self.metrics[f"{tool_name}_tool_time"].append(execution_time)
        self.metrics[f"{tool_name}_tool_success"].append(1 if success else 0)
    
    def get_summary(self, agent_name: str) -> dict:
        """Get performance summary for an agent."""
        execution_times = self.metrics[f"{agent_name}_execution_time"]
        successes = self.metrics[f"{agent_name}_success"]
        
        if not execution_times:
            return {}
        
        return {
            "avg_execution_time": statistics.mean(execution_times),
            "p95_execution_time": statistics.quantiles(execution_times, n=20)[18] if len(execution_times) > 20 else max(execution_times),
            "success_rate": statistics.mean(successes),
            "total_executions": len(execution_times)
        }

# Global metrics instance
agent_metrics = AgentMetrics()
```

### 3. Health Checks

Implement agent health monitoring:

```python
def create_health_check(agent_func, test_cases: list):
    """Create a health check for an agent."""
    
    def health_check() -> dict:
        """Execute health check and return status."""
        
        results = {
            "status": "healthy",
            "checks": [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        for test_case in test_cases:
            check_result = {
                "test": test_case["name"],
                "status": "pass",
                "response_time": None,
                "error": None
            }
            
            try:
                start_time = time.time()
                result = agent_func(test_case["input"])
                check_result["response_time"] = time.time() - start_time
                
                # Validate result
                if not validate_test_result(result, test_case.get("expected")):
                    check_result["status"] = "fail"
                    check_result["error"] = "Unexpected response format"
                    
            except Exception as e:
                check_result["status"] = "fail"
                check_result["error"] = str(e)
                check_result["response_time"] = time.time() - start_time
            
            results["checks"].append(check_result)
        
        # Overall status
        failed_checks = [c for c in results["checks"] if c["status"] == "fail"]
        if failed_checks:
            results["status"] = "unhealthy"
            results["failed_count"] = len(failed_checks)
        
        return results
    
    return health_check
```

---

## Deployment Best Practices

### 1. Environment Configuration

Use environment-specific configurations:

```python
# config/production.yaml
agents:
  product:
    model:
      name: "databricks-dbrx-instruct"
      temperature: 0.1
      max_tokens: 500
    guardrails:
      - content_safety: true
      - business_rules: true
      - quality_check: true
    tools:
      - unity_catalog_functions
      - vector_search
    performance:
      timeout: 30
      max_retries: 3

# config/development.yaml
agents:
  product:
    model:
      name: "databricks-dbrx-instruct"
      temperature: 0.2  # Slightly higher for testing
      max_tokens: 500
    guardrails:
      - content_safety: true
    tools:
      - unity_catalog_functions
      - vector_search
    performance:
      timeout: 60  # Longer timeout for debugging
      max_retries: 1
```

### 2. Gradual Rollout

Implement feature flags for safe deployments:

```python
class FeatureFlags:
    """Feature flag management for agent deployments."""
    
    def __init__(self):
        self.flags = {
            "new_comparison_agent": 0.1,  # 10% rollout
            "enhanced_guardrails": 0.5,   # 50% rollout
            "vector_search_v2": 0.0       # Disabled
        }
    
    def is_enabled(self, flag_name: str, user_id: str) -> bool:
        """Check if feature is enabled for user."""
        if flag_name not in self.flags:
            return False
        
        rollout_percentage = self.flags[flag_name]
        user_hash = hash(user_id) % 100
        
        return user_hash < (rollout_percentage * 100)

def create_agent_with_feature_flags(agent_name: str, state: AgentState, config: AgentConfig):
    """Create agent with feature flag support."""
    
    feature_flags = FeatureFlags()
    user_id = state.get("user_id", "anonymous")
    
    # Use new agent version if enabled
    if feature_flags.is_enabled(f"new_{agent_name}", user_id):
        return create_new_agent_version(agent_name, state, config)
    else:
        return create_stable_agent_version(agent_name, state, config)
```

### 3. Circuit Breaker Pattern

Protect against cascading failures:

```python
class CircuitBreaker:
    """Circuit breaker for agent protection."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise

# Usage
circuit_breaker = CircuitBreaker()

def protected_agent(state, config):
    return circuit_breaker.call(actual_agent_function, state, config)
```

---

## Related Documentation

- [Agent Reference](../references/agent-reference.md) - Detailed agent specifications
- [Agent Development Patterns](agent-development-patterns.md) - Implementation patterns
- [Agent Performance](agent-performance.md) - Performance optimization
- [Tools Reference](../references/tools-reference.md) - Available tools and usage 