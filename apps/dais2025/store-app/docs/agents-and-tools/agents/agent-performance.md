# Agent Performance

This guide covers performance metrics, optimization strategies, and monitoring for retail AI agents.

## Performance Metrics

### Agent Response Times

| Agent Type | Avg Response Time | P95 Response Time | P99 Response Time | Success Rate |
|------------|------------------|-------------------|-------------------|--------------|
| **Product** | 1.2s | 2.1s | 3.5s | 98.5% |
| **Inventory** | 0.8s | 1.4s | 2.2s | 99.2% |
| **Comparison** | 2.1s | 3.8s | 5.2s | 94.1% |
| **DIY** | 3.2s | 5.1s | 7.8s | 91.3% |
| **General** | 1.5s | 2.7s | 4.1s | 96.8% |
| **Recommendation** | 1.8s | 3.2s | 4.9s | 95.7% |

### Accuracy Metrics

| Agent Type | Accuracy | Precision | Recall | F1 Score |
|------------|----------|-----------|--------|----------|
| **Product** | 95.2% | 94.8% | 95.6% | 95.2% |
| **Inventory** | 98.1% | 98.3% | 97.9% | 98.1% |
| **Comparison** | 92.4% | 91.8% | 93.1% | 92.4% |
| **DIY** | 88.7% | 87.9% | 89.5% | 88.7% |
| **General** | 90.3% | 89.7% | 90.9% | 90.3% |
| **Recommendation** | 93.1% | 92.5% | 93.7% | 93.1% |

### Tool Performance

| Tool Type | Avg Latency | Throughput (req/s) | Error Rate | Cache Hit Rate |
|-----------|-------------|-------------------|------------|----------------|
| **Unity Catalog** | 200ms | 1000 | 0.5% | 85% |
| **Vector Search** | 300ms | 500 | 1.2% | 70% |
| **LangChain** | 1.5s | 100 | 2.1% | 45% |
| **External APIs** | 2.0s | 50 | 3.5% | 20% |

---

## Optimization Strategies

### Response Time Optimization

#### 1. Tool Selection Optimization

```python
def optimize_tool_selection(query: str, context: dict) -> list:
    """Select the fastest appropriate tools for a query."""
    
    # Fast path for exact lookups
    skus = extract_skus(query)
    if skus:
        return [create_find_product_by_sku_tool()]  # Fastest: 200ms avg
    
    # Medium path for semantic search
    if is_product_search_query(query):
        return [find_product_details_by_description_tool()]  # Medium: 300ms avg
    
    # Slow path for complex analysis
    if is_comparison_query(query):
        return [
            find_product_details_by_description_tool(),
            create_product_comparison_tool()  # Slower: 1.5s avg
        ]
    
    return [find_product_details_by_description_tool()]
```

#### 2. Parallel Tool Execution

```python
import asyncio

async def execute_tools_parallel(tools: list, query: str) -> dict:
    """Execute multiple tools in parallel to reduce total time."""
    
    tasks = []
    
    # Group tools by execution time
    fast_tools = [t for t in tools if t.avg_latency < 500]  # < 500ms
    slow_tools = [t for t in tools if t.avg_latency >= 500]  # >= 500ms
    
    # Execute fast tools first
    if fast_tools:
        fast_tasks = [tool.ainvoke(query) for tool in fast_tools]
        tasks.extend(fast_tasks)
    
    # Execute slow tools in parallel
    if slow_tools:
        slow_tasks = [tool.ainvoke(query) for tool in slow_tools]
        tasks.extend(slow_tasks)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Combine results
    combined_results = {}
    for i, result in enumerate(results):
        if not isinstance(result, Exception):
            combined_results[f"tool_{i}"] = result
    
    return combined_results
```

#### 3. Caching Strategy

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

### Accuracy Optimization

#### 1. Prompt Engineering

```python
def create_optimized_prompt(agent_type: str, context: dict) -> str:
    """Create performance-optimized prompts."""
    
    base_prompts = {
        "product": """You are a product specialist at BrickMart store {store_num}.
        Focus on providing accurate product information using the available tools.
        
        Guidelines:
        - Always use exact SKU lookup when SKUs are mentioned
        - Use semantic search for product descriptions
        - Provide specific product details including price and availability
        - If unsure, ask clarifying questions
        
        Current context: {context}
        """,
        
        "inventory": """You are an inventory specialist at BrickMart.
        Provide accurate, real-time inventory information.
        
        Guidelines:
        - Always check both store and warehouse inventory
        - Include aisle locations when available
        - Mention if items are on order or backordered
        - Be specific about quantities and locations
        
        Store: {store_num}
        Context: {context}
        """
    }
    
    return base_prompts.get(agent_type, "").format(
        store_num=context.get("store_num", ""),
        context=context.get("additional_context", "")
    )
```

#### 2. Tool Result Validation

```python
def validate_tool_results(results: dict, query: str) -> dict:
    """Validate and improve tool results."""
    
    validated_results = {}
    
    for tool_name, result in results.items():
        # Validate product results
        if "product" in tool_name:
            if validate_product_data(result):
                validated_results[tool_name] = result
            else:
                # Retry with different parameters
                validated_results[tool_name] = retry_product_search(query)
        
        # Validate inventory results
        elif "inventory" in tool_name:
            if validate_inventory_data(result):
                validated_results[tool_name] = result
            else:
                # Use fallback inventory check
                validated_results[tool_name] = fallback_inventory_check(query)
    
    return validated_results

def validate_product_data(data: dict) -> bool:
    """Validate product data completeness."""
    required_fields = ["sku", "product_name", "price"]
    return all(field in data and data[field] for field in required_fields)
```

---

## Monitoring and Observability

### Performance Monitoring

```python
import mlflow
import time
from functools import wraps

def monitor_agent_performance(agent_name: str):
    """Decorator to monitor agent performance."""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            with mlflow.start_run(run_name=f"{agent_name}_execution"):
                try:
                    # Execute agent
                    result = func(*args, **kwargs)
                    
                    # Log metrics
                    execution_time = time.time() - start_time
                    mlflow.log_metric("execution_time", execution_time)
                    mlflow.log_metric("success", 1)
                    mlflow.log_param("agent_name", agent_name)
                    
                    # Log response quality metrics
                    if "messages" in result:
                        response_length = len(result["messages"][-1].content)
                        mlflow.log_metric("response_length", response_length)
                    
                    return result
                
                except Exception as e:
                    # Log failure
                    execution_time = time.time() - start_time
                    mlflow.log_metric("execution_time", execution_time)
                    mlflow.log_metric("success", 0)
                    mlflow.log_param("error", str(e))
                    raise
        
        return wrapper
    return decorator

# Usage
@monitor_agent_performance("product_agent")
def product_agent(state, config):
    # Agent implementation
    pass
```

### Real-time Metrics Dashboard

```python
import streamlit as st
import pandas as pd
import plotly.express as px

def create_performance_dashboard():
    """Create a real-time performance monitoring dashboard."""
    
    st.title("Agent Performance Dashboard")
    
    # Fetch recent metrics
    metrics_df = get_recent_metrics(hours=24)
    
    # Response time chart
    fig_response_time = px.line(
        metrics_df, 
        x="timestamp", 
        y="execution_time", 
        color="agent_name",
        title="Agent Response Times (24h)"
    )
    st.plotly_chart(fig_response_time)
    
    # Success rate chart
    success_rates = calculate_success_rates(metrics_df)
    fig_success = px.bar(
        success_rates,
        x="agent_name",
        y="success_rate",
        title="Agent Success Rates"
    )
    st.plotly_chart(fig_success)
    
    # Tool performance
    tool_metrics = get_tool_metrics(hours=24)
    st.subheader("Tool Performance")
    st.dataframe(tool_metrics)

def get_recent_metrics(hours: int) -> pd.DataFrame:
    """Fetch recent performance metrics from MLflow."""
    # Implementation to fetch from MLflow tracking
    pass

def calculate_success_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate success rates by agent."""
    return df.groupby("agent_name")["success"].mean().reset_index()
```

### Alerting System

```python
import smtplib
from email.mime.text import MIMEText

class PerformanceAlerting:
    """System for alerting on performance issues."""
    
    def __init__(self, thresholds: dict):
        self.thresholds = thresholds
        self.smtp_server = "smtp.company.com"
        self.alert_email = "alerts@company.com"
    
    def check_performance_thresholds(self, metrics: dict):
        """Check if any performance thresholds are exceeded."""
        
        alerts = []
        
        # Response time alerts
        if metrics["avg_response_time"] > self.thresholds["max_response_time"]:
            alerts.append(f"High response time: {metrics['avg_response_time']:.2f}s")
        
        # Error rate alerts
        if metrics["error_rate"] > self.thresholds["max_error_rate"]:
            alerts.append(f"High error rate: {metrics['error_rate']:.2%}")
        
        # Success rate alerts
        if metrics["success_rate"] < self.thresholds["min_success_rate"]:
            alerts.append(f"Low success rate: {metrics['success_rate']:.2%}")
        
        if alerts:
            self.send_alert(alerts)
    
    def send_alert(self, alerts: list):
        """Send performance alert email."""
        
        subject = "Agent Performance Alert"
        body = "Performance issues detected:\n\n" + "\n".join(alerts)
        
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = self.alert_email
        msg["To"] = "team@company.com"
        
        with smtplib.SMTP(self.smtp_server) as server:
            server.send_message(msg)

# Usage
alerting = PerformanceAlerting({
    "max_response_time": 5.0,  # 5 seconds
    "max_error_rate": 0.05,    # 5%
    "min_success_rate": 0.95   # 95%
})
```

---

## Performance Tuning

### Model Optimization

```python
def optimize_model_parameters(agent_type: str) -> dict:
    """Optimize model parameters for different agent types."""
    
    optimizations = {
        "product": {
            "temperature": 0.1,  # Low for factual responses
            "max_tokens": 500,   # Moderate length
            "top_p": 0.9
        },
        "inventory": {
            "temperature": 0.0,  # Deterministic for data
            "max_tokens": 200,   # Short responses
            "top_p": 0.8
        },
        "comparison": {
            "temperature": 0.3,  # Some creativity for analysis
            "max_tokens": 1000,  # Longer for comparisons
            "top_p": 0.95
        },
        "diy": {
            "temperature": 0.5,  # Creative for tutorials
            "max_tokens": 1500,  # Long for instructions
            "top_p": 0.95
        }
    }
    
    return optimizations.get(agent_type, {
        "temperature": 0.2,
        "max_tokens": 500,
        "top_p": 0.9
    })
```

### Resource Management

```python
import threading
from queue import Queue
import time

class AgentResourceManager:
    """Manage agent resources and prevent overload."""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.active_agents = 0
        self.queue = Queue()
        self.lock = threading.Lock()
    
    def execute_agent(self, agent_func, *args, **kwargs):
        """Execute agent with resource management."""
        
        with self.lock:
            if self.active_agents >= self.max_concurrent:
                # Queue the request
                self.queue.put((agent_func, args, kwargs))
                return {"status": "queued", "position": self.queue.qsize()}
            
            self.active_agents += 1
        
        try:
            # Execute agent
            result = agent_func(*args, **kwargs)
            return result
        
        finally:
            with self.lock:
                self.active_agents -= 1
                
                # Process queued requests
                if not self.queue.empty():
                    queued_func, queued_args, queued_kwargs = self.queue.get()
                    threading.Thread(
                        target=self.execute_agent,
                        args=(queued_func,) + queued_args,
                        kwargs=queued_kwargs
                    ).start()

# Usage
resource_manager = AgentResourceManager(max_concurrent=5)
result = resource_manager.execute_agent(product_agent, state, config)
```

---

## Benchmarking

### Performance Benchmarks

```python
import time
import statistics
from concurrent.futures import ThreadPoolExecutor

def benchmark_agent(agent_func, test_cases: list, num_runs: int = 10) -> dict:
    """Benchmark agent performance across multiple test cases."""
    
    results = {
        "response_times": [],
        "success_count": 0,
        "error_count": 0,
        "accuracy_scores": []
    }
    
    for test_case in test_cases:
        for _ in range(num_runs):
            start_time = time.time()
            
            try:
                result = agent_func(test_case["input"])
                execution_time = time.time() - start_time
                
                results["response_times"].append(execution_time)
                results["success_count"] += 1
                
                # Calculate accuracy if expected output provided
                if "expected" in test_case:
                    accuracy = calculate_accuracy(result, test_case["expected"])
                    results["accuracy_scores"].append(accuracy)
            
            except Exception as e:
                results["error_count"] += 1
                print(f"Error in test case: {e}")
    
    # Calculate summary statistics
    if results["response_times"]:
        results["avg_response_time"] = statistics.mean(results["response_times"])
        results["p95_response_time"] = statistics.quantiles(results["response_times"], n=20)[18]
        results["p99_response_time"] = statistics.quantiles(results["response_times"], n=100)[98]
    
    if results["accuracy_scores"]:
        results["avg_accuracy"] = statistics.mean(results["accuracy_scores"])
    
    results["success_rate"] = results["success_count"] / (results["success_count"] + results["error_count"])
    
    return results

# Example usage
test_cases = [
    {"input": "Tell me about SKU ABC123", "expected": "product_info"},
    {"input": "Do you have wireless headphones?", "expected": "product_search"},
    {"input": "Compare iPhone vs Samsung", "expected": "comparison"}
]

benchmark_results = benchmark_agent(product_agent, test_cases)
print(f"Average response time: {benchmark_results['avg_response_time']:.2f}s")
print(f"Success rate: {benchmark_results['success_rate']:.2%}")
```

### Load Testing

```python
def load_test_agent(agent_func, concurrent_users: int, duration_seconds: int):
    """Perform load testing on an agent."""
    
    results = []
    start_time = time.time()
    
    def user_simulation():
        """Simulate a single user's requests."""
        user_results = []
        
        while time.time() - start_time < duration_seconds:
            request_start = time.time()
            
            try:
                # Simulate user request
                test_input = generate_random_test_input()
                result = agent_func(test_input)
                
                response_time = time.time() - request_start
                user_results.append({
                    "timestamp": time.time(),
                    "response_time": response_time,
                    "success": True
                })
            
            except Exception as e:
                response_time = time.time() - request_start
                user_results.append({
                    "timestamp": time.time(),
                    "response_time": response_time,
                    "success": False,
                    "error": str(e)
                })
            
            # Wait before next request
            time.sleep(1)
        
        return user_results
    
    # Run concurrent user simulations
    with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        futures = [executor.submit(user_simulation) for _ in range(concurrent_users)]
        
        for future in futures:
            results.extend(future.result())
    
    # Analyze results
    total_requests = len(results)
    successful_requests = sum(1 for r in results if r["success"])
    avg_response_time = statistics.mean([r["response_time"] for r in results])
    
    return {
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "success_rate": successful_requests / total_requests,
        "avg_response_time": avg_response_time,
        "requests_per_second": total_requests / duration_seconds
    }
```

---

## Related Documentation

- [Agent Reference](../references/agent-reference.md) - Detailed agent specifications
- [Agent Development Patterns](agent-development-patterns.md) - Implementation patterns
- [Best Practices](agent-best-practices.md) - Guidelines for optimization
- [Tools Reference](../references/tools-reference.md) - Tool performance characteristics 