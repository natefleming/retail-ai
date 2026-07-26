# Python API

## Loading Configuration

```python
from dao_ai.config import AppConfig

# Load configuration from YAML file
config = AppConfig.from_file("config/my_config.yaml")

# Load with parameter overrides for ${var.NAME} / ${param.NAME} substitution
config = AppConfig.from_file(
    "config/my_config.yaml",
    params={"catalog": "my_catalog", "schema": "my_schema"},
)
```

## Parameter Substitution (Python API)

DAO AI configs can declare a `parameters:` block and reference values
inline with `${var.NAME}` or `${param.NAME}`. Substitution happens at
load time, before `ModelConfig` parses the YAML.

Full reference (YAML syntax, precedence rules, error handling, bridge
pattern, YAML quoting caveat): see
[Parameters (Load-Time Substitution)](configuration-reference.md#parameters-load-time-substitution)
in the Configuration Reference.

### Passing overrides from Python

```python
from dao_ai.config import AppConfig

config = AppConfig.from_file(
    "dao_ai.yaml",
    params={"catalog": "nfleming", "module_id": "09"},
)
```

The `params` argument is keyword-only. Values must be strings; Pydantic
handles downstream coercion (e.g. `"4096"` to `int` for `max_tokens`).

### Inspecting what was substituted

```python
config.source_config_path   # "/path/to/dao_ai.yaml"
config.substitution_vars    # {"catalog": "nfleming", "module_id": "09"}
config.rendered_yaml        # full YAML text with ${var.…} resolved
```

### Handling errors

```python
from dao_ai.config import AppConfig
from dao_ai.config_vars import ConfigVariableError

try:
    config = AppConfig.from_file("dao_ai.yaml")
except ConfigVariableError as err:
    print(f"Cannot load {err.path}")
    print(f"  missing required: {err.missing_required}")
    print(f"  undeclared refs:  {err.undeclared}")
    raise
```

### From a Databricks notebook with widgets

```python
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("module_id", "09")

config = AppConfig.from_file(
    "dao_ai.yaml",
    params={
        "catalog": dbutils.widgets.get("catalog"),
        "module_id": dbutils.widgets.get("module_id"),
    },
)
```

### Skip initialization for tests / inspection

```python
config = AppConfig.from_file(
    "dao_ai.yaml",
    params={"catalog": "test"},
    initialize=False,
)
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
from dao_ai.config import AiSearchVectorStoreModel, LakebaseVectorStoreModel

# `vector_stores` is a discriminated union — each entry is either an
# AiSearchVectorStoreModel or a LakebaseVectorStoreModel. AI Search stores
# use `.create()` (provisions the index endpoint if configured); Lakebase
# stores use `.provision(dimension=...)` (creates extensions + table +
# indexes in Postgres).
for name, vs in vector_stores.items():
    if isinstance(vs, AiSearchVectorStoreModel):
        vs.create()
    elif isinstance(vs, LakebaseVectorStoreModel):
        vs.provision(dimension=1024)  # matches your embedding endpoint

# Or address a specific entry directly by name:
config.resources.vector_stores["my_ai_store"].create()
config.resources.vector_stores["my_lakebase_store"].provision(dimension=1024)
```

## Packaging and Deployment

```python
from dao_ai.config import AppConfig, DeploymentTarget

config = AppConfig.from_file("config/my_agent.yaml")

# Package the agent as an MLflow model
config.create_agent(
    additional_pip_reqs=["custom-package==1.0.0"],
    additional_code_paths=["./my_modules"]
)

# Deploy to Databricks Model Serving (default)
config.deploy_agent()

# Deploy directly to Databricks Apps (no asset bundle needed)
config.deploy_agent(target=DeploymentTarget.APPS)

# Deploy to MCP server (Apps with MCP entrypoint)
config.deploy_agent(target=DeploymentTarget.MCP)
```

### Apps deployment and chat UI

When deploying to Apps with `enable_chat_proxy: true` (the default), the
deployed app automatically clones and builds the Databricks
[e2e-chatbot-app-next](https://github.com/databricks/app-templates/tree/main/e2e-chatbot-app-next)
chat UI at startup.  The Apps runtime has Node.js pre-installed, so no
additional tools are needed on your development machine.

All three deploy flows converge on the same runtime behavior:

- `config.deploy_agent(target=DeploymentTarget.APPS)` (programmatic)
- `dao-ai workflow generate --deploy --run --mode apps` (CLI, runs a notebook that calls `deploy_agent`)
- `dao-ai agent generate` + `databricks bundle deploy` (standalone bundle)

Set `app.enable_chat_proxy: false` in your config to deploy without the chat
UI (backend API only).

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

## User Feedback (Thumbs-Up / Thumbs-Down)

Every dao-ai response exposes the outer MLflow trace_id on
`response.custom_outputs["trace_id"]`. Pass it to `log_user_feedback` to
attach a `user_feedback` assessment to the trace — works the same for
single-agent and multi-agent (supervisor / swarm) flows.

```python
from dao_ai.evaluation import log_user_feedback

resp = await agent.apredict(request)

# Happy path: pull trace_id from the response, not MLflow global state
trace_id = resp.custom_outputs["trace_id"]

log_user_feedback(
    trace_id=trace_id,
    value="up",                       # "up" / "down" / bool
    comment="Multi-agent answer was correct.",
    user_id="user@example.com",
)
```

### Why not read trace_id from MLflow global state?

| Anti-pattern | Failure mode |
|---|---|
| `mlflow.get_last_active_trace_id()` in caller | Races under concurrency — desyncs from the trace this call produced |
| `mlflow.get_current_active_span()` in caller | Returns `None` once the agent function returns; `.trace_id` raises `AttributeError` |
| `mlflow.log_assessment(...)` / legacy `Assessment(...)` | MLflow 2.x preview API, deprecated in MLflow 3 |

### Querying traces with feedback

```python
import mlflow

traces = mlflow.search_traces(locations=[experiment_id], max_results=100)
# `assessments` column has the user_feedback rows
has_pos = traces["assessments"].apply(
    lambda assess: any(
        a.get("assessment_name") == "user_feedback"
        and a.get("feedback", {}).get("value") is True
        for a in (assess or [])
    )
)
positive = traces[has_pos]
```

For SQL-side analysis, see
[`notebooks/16_feedback_demo.py`](../notebooks/16_feedback_demo.py) — it
registers `mlflow.search_traces` results as a Spark temp view and
demonstrates `LATERAL VIEW EXPLODE(assessments)` queries for daily
up/down volume and feedback-by-trace.

---

## Navigation

- [← Previous: CLI Reference](cli-reference.md)
- [↑ Back to Documentation Index](../README.md#-documentation)
- [Next: FAQ →](faq.md)

