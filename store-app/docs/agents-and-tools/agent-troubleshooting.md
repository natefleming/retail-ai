# Agent Troubleshooting

This guide provides comprehensive troubleshooting steps for common issues with retail AI agents, tools, and deployment.

## Common Issues and Solutions

### Agent Not Responding

**Symptoms:**
- Agent execution hangs or times out
- No response from agent calls
- Silent failures

**Diagnostic Steps:**

```python
# Check agent configuration
def diagnose_agent_config(agent_name: str, model_config: dict):
    """Diagnose agent configuration issues."""
    
    print(f"Diagnosing agent: {agent_name}")
    
    # Check model configuration
    agent_config = model_config.get("agents", {}).get(agent_name)
    if not agent_config:
        print(f"ERROR: Agent '{agent_name}' not found in configuration")
        return False
    
    # Check model settings
    model_name = agent_config.get("model", {}).get("name")
    if not model_name:
        print("ERROR: Model name not specified")
        return False
    
    print(f"OK: Model: {model_name}")
    
    # Check tools configuration
    tools = agent_config.get("tools", [])
    print(f"OK: Tools configured: {len(tools)}")
    
    return True

# Test agent execution
def test_agent_execution(agent_func, test_input: str):
    """Test basic agent execution."""
    
    try:
        start_time = time.time()
        
        state = AgentState(
            messages=[HumanMessage(content=test_input)],
            user_id="test_user",
            store_num="101"
        )
        
        result = agent_func(state, {})
        execution_time = time.time() - start_time
        
        print(f"OK: Agent executed successfully in {execution_time:.2f}s")
        print(f"Response: {result['messages'][-1].content[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Agent execution failed: {e}")
        print(f"Error type: {type(e).__name__}")
        return False
```

**Common Solutions:**

1. **Check Model Endpoint:**
```python
# Verify model endpoint is accessible
from databricks.sdk import WorkspaceClient

try:
    w = WorkspaceClient()
    endpoints = w.serving_endpoints.list()
    print("Available endpoints:", [e.name for e in endpoints])
except Exception as e:
    print(f"Cannot access Databricks: {e}")
```

2. **Verify Tool Availability:**
```python
# Test individual tools
def test_tools(tools: list):
    """Test each tool individually."""
    
    for tool in tools:
        try:
            # Simple test input
            result = tool.invoke("test")
            print(f"OK: Tool {tool.name} working")
        except Exception as e:
            print(f"ERROR: Tool {tool.name} failed: {e}")
```

3. **Check Timeout Settings:**
```python
# Increase timeout for debugging
agent_config = {
    "timeout": 120,  # 2 minutes
    "max_retries": 1
}
```

---

### Tool Errors

**Symptoms:**
- Tool execution failures
- Authentication errors
- Data access issues

**Unity Catalog Tool Issues:**

```python
def diagnose_unity_catalog_tools(warehouse_id: str, function_name: str):
    """Diagnose Unity Catalog function issues."""
    
    from databricks.sdk import WorkspaceClient
    
    try:
        w = WorkspaceClient()
        
        # Test warehouse connection
        warehouses = w.warehouses.list()
        warehouse_names = [wh.name for wh in warehouses]
        print(f"Available warehouses: {warehouse_names}")
        
        if warehouse_id not in warehouse_names:
            print(f"ERROR: Warehouse '{warehouse_id}' not found")
            return False
        
        # Test function execution
        statement = f"SELECT * FROM {function_name}(ARRAY('TEST-SKU')) LIMIT 1"
        response = w.statement_execution.execute_statement(
            statement=statement,
            warehouse_id=warehouse_id
        )
        
        print(f"OK: Function {function_name} accessible")
        return True
        
    except Exception as e:
        print(f"ERROR: Unity Catalog error: {e}")
        
        # Common error patterns
        if "PERMISSION_DENIED" in str(e):
            print("💡 Check Unity Catalog permissions")
        elif "FUNCTION_NOT_FOUND" in str(e):
            print("💡 Function may not exist or not accessible")
        elif "WAREHOUSE_NOT_FOUND" in str(e):
            print("💡 Check warehouse ID and permissions")
        
        return False
```

**Vector Search Tool Issues:**

```python
def diagnose_vector_search(endpoint_name: str, index_name: str):
    """Diagnose vector search issues."""
    
    try:
        from databricks.vector_search.client import VectorSearchClient
        
        client = VectorSearchClient()
        
        # Check endpoint
        try:
            endpoint = client.get_endpoint(endpoint_name)
            print(f"OK: Endpoint '{endpoint_name}' found")
        except Exception as e:
            print(f"ERROR: Endpoint error: {e}")
            return False
        
        # Check index
        try:
            index = client.get_index(index_name)
            print(f"OK: Index '{index_name}' found")
            
            # Test search
            results = index.similarity_search(
                query_text="test query",
                columns=["product_name"],
                num_results=1
            )
            print(f"OK: Search working, found {len(results)} results")
            
        except Exception as e:
            print(f"ERROR: Index error: {e}")
            return False
        
        return True
        
    except ImportError:
        print("ERROR: Vector search client not available")
        return False
```

**LangChain Tool Issues:**

```python
def diagnose_langchain_tools(llm_endpoint: str):
    """Diagnose LangChain tool issues."""
    
    try:
        from langchain_community.llms import Databricks
        
        # Test LLM connection
        llm = Databricks(endpoint_name=llm_endpoint)
        
        # Simple test
        response = llm.invoke("Hello, this is a test.")
        print(f"OK: LLM endpoint working: {response[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"ERROR: LangChain LLM error: {e}")
        
        if "ENDPOINT_NOT_FOUND" in str(e):
            print("💡 Check LLM endpoint name and availability")
        elif "PERMISSION_DENIED" in str(e):
            print("💡 Check endpoint permissions")
        
        return False
```

---

### Performance Issues

**Symptoms:**
- Slow response times
- High latency
- Timeouts

**Performance Diagnostics:**

```python
import time
import statistics

def performance_diagnostic(agent_func, test_cases: list, num_runs: int = 5):
    """Comprehensive performance diagnostic."""
    
    results = {
        "response_times": [],
        "tool_times": {},
        "errors": []
    }
    
    for test_case in test_cases:
        for run in range(num_runs):
            start_time = time.time()
            
            try:
                # Monitor tool execution times
                with ToolTimeMonitor() as monitor:
                    result = agent_func(test_case)
                
                total_time = time.time() - start_time
                results["response_times"].append(total_time)
                
                # Collect tool times
                for tool_name, tool_time in monitor.get_times().items():
                    if tool_name not in results["tool_times"]:
                        results["tool_times"][tool_name] = []
                    results["tool_times"][tool_name].append(tool_time)
                
            except Exception as e:
                results["errors"].append(str(e))
    
    # Analysis
    if results["response_times"]:
        avg_time = statistics.mean(results["response_times"])
        p95_time = statistics.quantiles(results["response_times"], n=20)[18]
        
        print(f"Average response time: {avg_time:.2f}s")
        print(f"P95 response time: {p95_time:.2f}s")
        
        # Tool breakdown
        print("\nTool performance:")
        for tool_name, times in results["tool_times"].items():
            avg_tool_time = statistics.mean(times)
            print(f"  {tool_name}: {avg_tool_time:.2f}s avg")
    
    if results["errors"]:
        print(f"\nErrors encountered: {len(results['errors'])}")
        for error in set(results["errors"]):
            print(f"  - {error}")

class ToolTimeMonitor:
    """Monitor tool execution times."""
    
    def __init__(self):
        self.times = {}
        self.start_times = {}
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def start_tool(self, tool_name: str):
        self.start_times[tool_name] = time.time()
    
    def end_tool(self, tool_name: str):
        if tool_name in self.start_times:
            elapsed = time.time() - self.start_times[tool_name]
            self.times[tool_name] = elapsed
    
    def get_times(self) -> dict:
        return self.times
```

**Performance Optimization:**

```python
def optimize_agent_performance(agent_func):
    """Apply performance optimizations."""
    
    # Add caching
    @lru_cache(maxsize=100)
    def cached_agent(query_hash: str, state_json: str):
        state = json.loads(state_json)
        return agent_func(state)
    
    # Add timeout
    def timeout_wrapper(state, config, timeout: int = 30):
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Agent execution timed out after {timeout}s")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            result = agent_func(state, config)
            signal.alarm(0)  # Cancel timeout
            return result
        except TimeoutError:
            signal.alarm(0)
            raise
    
    return timeout_wrapper
```

---

### Authentication and Permissions

**Symptoms:**
- Permission denied errors
- Authentication failures
- Access denied to resources

**Databricks Authentication:**

```python
def diagnose_databricks_auth():
    """Diagnose Databricks authentication issues."""
    
    import os
    from databricks.sdk import WorkspaceClient
    
    # Check environment variables
    required_vars = ["DATABRICKS_HOST", "DATABRICKS_TOKEN"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"ERROR: Missing environment variables: {missing_vars}")
        return False
    
    try:
        w = WorkspaceClient()
        user = w.current_user.me()
        print(f"OK: Authenticated as: {user.user_name}")
        
        # Test permissions
        try:
            warehouses = w.warehouses.list()
            print(f"OK: Can access {len(list(warehouses))} warehouses")
        except Exception as e:
            print(f"ERROR: Cannot access warehouses: {e}")
        
        try:
            endpoints = w.serving_endpoints.list()
            print(f"OK: Can access {len(list(endpoints))} serving endpoints")
        except Exception as e:
            print(f"ERROR: Cannot access serving endpoints: {e}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Authentication failed: {e}")
        
        if "INVALID_TOKEN" in str(e):
            print("💡 Check DATABRICKS_TOKEN value")
        elif "INVALID_HOST" in str(e):
            print("💡 Check DATABRICKS_HOST format (https://...)")
        
        return False
```

**Unity Catalog Permissions:**

```python
def check_unity_catalog_permissions(catalog: str, schema: str):
    """Check Unity Catalog permissions."""
    
    from databricks.sdk import WorkspaceClient
    
    try:
        w = WorkspaceClient()
        
        # Check catalog access
        try:
            catalogs = w.catalogs.list()
            catalog_names = [c.name for c in catalogs]
            
            if catalog in catalog_names:
                print(f"OK: Can access catalog '{catalog}'")
            else:
                print(f"ERROR: Cannot access catalog '{catalog}'")
                print(f"Available catalogs: {catalog_names}")
                return False
        except Exception as e:
            print(f"ERROR: Cannot list catalogs: {e}")
            return False
        
        # Check schema access
        try:
            schemas = w.schemas.list(catalog_name=catalog)
            schema_names = [s.name for s in schemas]
            
            if schema in schema_names:
                print(f"OK: Can access schema '{catalog}.{schema}'")
            else:
                print(f"ERROR: Cannot access schema '{catalog}.{schema}'")
                print(f"Available schemas: {schema_names}")
                return False
        except Exception as e:
            print(f"ERROR: Cannot list schemas: {e}")
            return False
        
        # Check function access
        try:
            functions = w.functions.list(catalog_name=catalog, schema_name=schema)
            function_names = [f.name for f in functions]
            print(f"OK: Can access {len(function_names)} functions")
            
            if function_names:
                print(f"Available functions: {function_names[:5]}...")  # Show first 5
            
        except Exception as e:
            print(f"ERROR: Cannot list functions: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"ERROR: Permission check failed: {e}")
        return False
```

---

### Configuration Issues

**Symptoms:**
- Invalid configuration errors
- Missing configuration values
- Type errors in configuration

**Configuration Validation:**

```python
def validate_model_config(config: dict) -> list:
    """Validate model configuration and return issues."""
    
    issues = []
    
    # Check required top-level keys
    required_keys = ["catalog_name", "database_name", "warehouse_id", "agents"]
    for key in required_keys:
        if key not in config:
            issues.append(f"Missing required key: {key}")
    
    # Validate agents configuration
    if "agents" in config:
        for agent_name, agent_config in config["agents"].items():
            # Check agent structure
            if not isinstance(agent_config, dict):
                issues.append(f"Agent '{agent_name}' config must be a dictionary")
                continue
            
            # Check model configuration
            if "model" not in agent_config:
                issues.append(f"Agent '{agent_name}' missing model configuration")
            else:
                model_config = agent_config["model"]
                if "name" not in model_config:
                    issues.append(f"Agent '{agent_name}' missing model name")
            
            # Check prompt
            if "prompt" not in agent_config:
                issues.append(f"Agent '{agent_name}' missing prompt")
            
            # Validate guardrails
            if "guardrails" in agent_config:
                guardrails = agent_config["guardrails"]
                if not isinstance(guardrails, list):
                    issues.append(f"Agent '{agent_name}' guardrails must be a list")
    
    return issues

def fix_common_config_issues(config: dict) -> dict:
    """Automatically fix common configuration issues."""
    
    fixed_config = config.copy()
    
    # Add default values for missing keys
    defaults = {
        "warehouse_id": "default_warehouse",
        "vector_search": {
            "endpoint_name": "default_endpoint",
            "index_name": "default_index"
        }
    }
    
    for key, default_value in defaults.items():
        if key not in fixed_config:
            fixed_config[key] = default_value
            print(f"Added default value for {key}")
    
    # Fix agent configurations
    if "agents" in fixed_config:
        for agent_name, agent_config in fixed_config["agents"].items():
            # Add default model settings
            if "model" in agent_config and "temperature" not in agent_config["model"]:
                agent_config["model"]["temperature"] = 0.1
                print(f"Added default temperature for {agent_name}")
            
            # Add default guardrails
            if "guardrails" not in agent_config:
                agent_config["guardrails"] = []
                print(f"Added empty guardrails list for {agent_name}")
    
    return fixed_config
```

---

### Deployment Issues

**Symptoms:**
- Application startup failures
- Environment-specific errors
- Resource allocation issues

**Environment Diagnostics:**

```python
def diagnose_environment():
    """Diagnose deployment environment."""
    
    import sys
    import platform
    
    print("Environment Diagnostics:")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    # Check required packages
    required_packages = [
        "databricks-sdk",
        "langchain",
        "mlflow",
        "streamlit"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"OK: {package} installed")
        except ImportError:
            print(f"ERROR: {package} not installed")
    
    # Check environment variables
    required_env_vars = [
        "DATABRICKS_HOST",
        "DATABRICKS_TOKEN",
        "DATABRICKS_WAREHOUSE_ID"
    ]
    
    for var in required_env_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            masked_value = value[:10] + "..." if len(value) > 10 else value
            print(f"OK: {var}: {masked_value}")
        else:
            print(f"ERROR: {var}: Not set")

def health_check_endpoint():
    """Create a health check endpoint for deployment monitoring."""
    
    def health_check() -> dict:
        """Comprehensive health check."""
        
        status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {}
        }
        
        # Check Databricks connection
        try:
            w = WorkspaceClient()
            w.current_user.me()
            status["checks"]["databricks"] = "healthy"
        except Exception as e:
            status["checks"]["databricks"] = f"unhealthy: {e}"
            status["status"] = "unhealthy"
        
        # Check model endpoints
        try:
            endpoints = list(w.serving_endpoints.list())
            status["checks"]["model_endpoints"] = f"healthy ({len(endpoints)} available)"
        except Exception as e:
            status["checks"]["model_endpoints"] = f"unhealthy: {e}"
            status["status"] = "unhealthy"
        
        # Check Unity Catalog
        try:
            catalogs = list(w.catalogs.list())
            status["checks"]["unity_catalog"] = f"healthy ({len(catalogs)} catalogs)"
        except Exception as e:
            status["checks"]["unity_catalog"] = f"unhealthy: {e}"
            status["status"] = "unhealthy"
        
        return status
    
    return health_check
```

---

## Debugging Tools

### Agent Execution Tracer

```python
class AgentTracer:
    """Trace agent execution for debugging."""
    
    def __init__(self):
        self.trace = []
    
    def log_step(self, step_type: str, data: dict):
        """Log a step in agent execution."""
        self.trace.append({
            "timestamp": time.time(),
            "step_type": step_type,
            "data": data
        })
    
    def trace_agent(self, agent_func):
        """Decorator to trace agent execution."""
        
        def traced_agent(state, config):
            self.log_step("agent_start", {
                "user_query": state["messages"][-1].content,
                "user_id": state.get("user_id"),
                "store_num": state.get("store_num")
            })
            
            try:
                result = agent_func(state, config)
                
                self.log_step("agent_success", {
                    "response": result["messages"][-1].content,
                    "execution_time": self.get_execution_time()
                })
                
                return result
                
            except Exception as e:
                self.log_step("agent_error", {
                    "error": str(e),
                    "error_type": type(e).__name__
                })
                raise
        
        return traced_agent
    
    def get_execution_time(self) -> float:
        """Get total execution time."""
        if len(self.trace) >= 2:
            return self.trace[-1]["timestamp"] - self.trace[0]["timestamp"]
        return 0.0
    
    def print_trace(self):
        """Print execution trace."""
        print("Agent Execution Trace:")
        for i, step in enumerate(self.trace):
            timestamp = step["timestamp"]
            step_type = step["step_type"]
            print(f"{i+1}. [{timestamp:.3f}] {step_type}: {step['data']}")

# Usage
tracer = AgentTracer()
traced_agent = tracer.trace_agent(my_agent)
result = traced_agent(state, config)
tracer.print_trace()
```

### Tool Performance Profiler

```python
class ToolProfiler:
    """Profile tool performance for optimization."""
    
    def __init__(self):
        self.profiles = {}
    
    def profile_tool(self, tool):
        """Decorator to profile tool execution."""
        
        original_invoke = tool.invoke
        
        def profiled_invoke(*args, **kwargs):
            start_time = time.time()
            start_memory = self.get_memory_usage()
            
            try:
                result = original_invoke(*args, **kwargs)
                
                execution_time = time.time() - start_time
                memory_used = self.get_memory_usage() - start_memory
                
                self.record_profile(tool.name, {
                    "execution_time": execution_time,
                    "memory_used": memory_used,
                    "success": True
                })
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                self.record_profile(tool.name, {
                    "execution_time": execution_time,
                    "memory_used": 0,
                    "success": False,
                    "error": str(e)
                })
                raise
        
        tool.invoke = profiled_invoke
        return tool
    
    def record_profile(self, tool_name: str, metrics: dict):
        """Record profiling metrics."""
        if tool_name not in self.profiles:
            self.profiles[tool_name] = []
        
        self.profiles[tool_name].append({
            "timestamp": time.time(),
            **metrics
        })
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def get_summary(self) -> dict:
        """Get profiling summary."""
        summary = {}
        
        for tool_name, profiles in self.profiles.items():
            execution_times = [p["execution_time"] for p in profiles if p["success"]]
            success_count = sum(1 for p in profiles if p["success"])
            
            if execution_times:
                summary[tool_name] = {
                    "avg_execution_time": statistics.mean(execution_times),
                    "min_execution_time": min(execution_times),
                    "max_execution_time": max(execution_times),
                    "success_rate": success_count / len(profiles),
                    "total_calls": len(profiles)
                }
        
        return summary
```

---

## 📞 Getting Help

### Support Channels

1. **Documentation**: Check the comprehensive guides
2. **GitHub Issues**: Report bugs and request features
3. **Community Forum**: Ask questions and share solutions
4. **Direct Support**: Contact the development team

### Reporting Issues

When reporting issues, include:

```python
def generate_debug_report():
    """Generate a comprehensive debug report."""
    
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "packages": get_installed_packages()
        },
        "configuration": {
            "databricks_host": os.getenv("DATABRICKS_HOST", "Not set"),
            "warehouse_id": os.getenv("DATABRICKS_WAREHOUSE_ID", "Not set")
        },
        "error_details": {
            # Include specific error information
        },
        "reproduction_steps": [
            # List steps to reproduce the issue
        ]
    }
    
    return report

def get_installed_packages() -> dict:
    """Get versions of key packages."""
    import pkg_resources
    
    key_packages = [
        "databricks-sdk",
        "langchain",
        "mlflow",
        "streamlit"
    ]
    
    versions = {}
    for package in key_packages:
        try:
            version = pkg_resources.get_distribution(package).version
            versions[package] = version
        except pkg_resources.DistributionNotFound:
            versions[package] = "Not installed"
    
    return versions
```

---

## Related Documentation

- [Agent Reference](references/agent-reference.md) - Detailed agent specifications
- [Agent Development Patterns](agents/agent-development-patterns.md) - Implementation patterns
- [Agent Performance](agents/agent-performance.md) - Performance optimization
- [Best Practices](agents/agent-best-practices.md) - Development guidelines 