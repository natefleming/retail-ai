# Python API

## Loading Configuration

```python
from dao_ai.config import AppConfig

# Load configuration from YAML file
config = AppConfig.from_file("config/my_config.yaml")
```

## Accessing Components

```python
# Access agents
agents = config.find_agents()

# Access tools
tools = config.find_tools()

# Access vector stores
vector_stores = config.resources.vector_stores

# Access other resources
llms = config.resources.llms
warehouses = config.resources.warehouses
databases = config.resources.databases
```

## Creating Infrastructure

```python
# Create vector stores
for name, vs in vector_stores.items():
    vs.create()

# Create specific infrastructure components
config.resources.vector_stores["my_store"].create()
```

## Packaging and Deployment

```python
# Package the agent as an MLflow model
config.create_agent(
    additional_pip_reqs=["custom-package==1.0.0"],
    additional_code_paths=["./my_modules"]
)

# Deploy to Databricks Model Serving
config.deploy_agent()
```

## Visualization

```python
# Display graph in notebook
config.display_graph()

# Save graph as image
config.save_image("docs/architecture.png")
```

## Local Testing

```python
from dao_ai.config import AppConfig

# Load configuration
config = AppConfig.from_file("config/my_agent.yaml")

# Create runnable agent
agent = config.as_runnable()

# Test locally
response = agent.invoke({
    "messages": [{"role": "user", "content": "Test question"}],
    "configurable": {
        "thread_id": "test-123",
        "user_id": "test_user"
    }
})

# Print response
print(response)
```

## Advanced Usage

### Custom Tool Creation

```python
from langchain.tools import tool

@tool
def my_custom_tool(query: str) -> str:
    """My custom tool description."""
    # Your custom logic here
    return "Result"
```

### Custom Middleware

Middleware factories in DAO AI return single `AgentMiddleware` instances:

```python
from langchain.agents import AgentMiddleware

def create_my_middleware(**kwargs) -> AgentMiddleware:
    """
    Factory function that creates middleware.
    
    Returns a list for composability - factories can return multiple
    middleware instances when needed (e.g., one per tool).
    """
    
    class MyMiddleware(AgentMiddleware):
        def process_request(self, state):
            # Process before agent execution
            return state
        
        def process_response(self, state):
            # Process after agent execution
            return state
    
    return MyMiddleware()

# Combine multiple middleware instances into a list
all_middleware = [
    create_my_middleware(),
    create_other_middleware(),
]
```

### Custom Hooks

```python
def my_startup_hook():
    """Run on agent startup."""
    print("Initializing agent...")
    # Your initialization logic

def my_shutdown_hook():
    """Run on agent shutdown."""
    print("Cleaning up resources...")
    # Your cleanup logic
```

## Configuration Validation

```python
from dao_ai.config import AppConfig

try:
    config = AppConfig.from_file("config/my_config.yaml")
    print("✅ Configuration is valid!")
except Exception as e:
    print(f"❌ Configuration error: {e}")
```

## Schema Generation

```python
from dao_ai.config import AppConfig

# Generate JSON schema for IDE support
schema = AppConfig.model_json_schema()

# Save to file
import json
with open("schemas/model_config_schema.json", "w") as f:
    json.dump(schema, f, indent=2)
```

---

## Navigation

- [← Previous: CLI Reference](cli-reference.md)
- [↑ Back to Documentation Index](../README.md#-documentation)
- [Next: FAQ →](faq.md)

