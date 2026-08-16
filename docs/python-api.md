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

### Config sources

A config can come from three places. `from_file` accepts any of them; the named
methods validate their own kind, which is useful when you want a mistake to be
caught immediately rather than fall through to a filesystem read.

| Method | Source | Relative `ddl` / `data` / `code_paths` |
|---|---|---|
| `from_file(path)` | Local path — also accepts the other two | Resolved against the config's directory |
| `from_url(url)` | An `http(s)` URL (single YAML) | **Rejected** — no directory to anchor to |
| `from_git(locator)` | A git repository, with its whole tree | Resolved against the config's directory in the checkout |
| `from_source(spec)` | Any of the above, classified for you | Depends on the source |

```python
# A git repository: the project's assets come along, so a config with
# `ddl: data/products.sql` works — which a URL cannot do.
config = AppConfig.from_git(
    "git+https://github.com/org/repo@v1.0#examples/retail/agent.yaml",
    params={"catalog": "my_catalog"},
)

# gh: shorthand; ref and in-repo path are both optional. When the path is a
# directory (or omitted), the config is discovered — ambiguity is an error
# listing the candidates.
config = AppConfig.from_git("gh:org/repo@main#examples/retail")

# Accepts a spec string or a typed source. Construct the source when you need
# options a string can't express.
from dao_ai.git_source import GitSource

config = AppConfig.from_git(
    GitSource("gh:org/private-repo@main", token=my_token, refresh=True)
)
```

`from_git` requires `git` on `PATH`. Resolution is client-side only: a generated
bundle is self-contained, so nothing needs `git` at deploy or run time.

**Trust.** A git locator runs the repository's code — a config can ship Python via
`code_paths` / `src/` and inline tool code — exactly as cloning it and running
dao-ai locally would. The resolved commit SHA is logged on every load. Pin a tag
or SHA for repositories you do not control.

Checkouts are cached per commit under `~/.dao-ai/git` (override with
`$DAO_AI_GIT_CACHE`). A full 40-character SHA is immutable and never re-fetched; a
branch or tag is re-resolved with `git ls-remote` each load and re-fetched only
when it moved. Private repositories authenticate through git itself (ssh-agent,
credential helpers); for headless use set `DAO_AI_GIT_TOKEN` or `GITHUB_TOKEN`,
which is passed to git without ever being written to disk or appearing in a
command line.

`~/.dao-ai` is deliberately not under `$XDG_CACHE_HOME`: a checkout is not
disposable the way a cache is (losing one costs a re-clone), and it mirrors the
project-local `.dao-ai/` name. See the
[CLI reference](cli-reference.md#config-sources-local-url-or-git) for how staging
directories are placed.

### Provisioning a whole project from a notebook

`from_git` brings the repository's tree along, so a notebook with no project code
checked out anywhere can provision the project end to end. Install dao-ai, load
the config from a locator, and create each resource in dependency order —
schemas and volumes before the datasets that land in them, datasets before the UC
functions and vector indexes that read them.

```python
%pip install dao-ai
%restart_python
```

```python
from databricks.sdk import WorkspaceClient

from dao_ai.config import AppConfig

config = AppConfig.from_git(
    "gh:org/repo@v1.0#examples/retail/agent.yaml",
    params={"catalog": "my_catalog", "schema": "my_schema"},
)

w = WorkspaceClient()

for schema in config.schemas.values():
    schema.create(w=w)
for volume in config.resources.volumes.values():
    volume.create(w=w)
for dataset in config.datasets or []:
    dataset.create()
for fn in config.unity_catalog_functions or []:
    fn.create()
for vector_store in config.resources.vector_stores.values():
    vector_store.create()
```

This is the same order the generated workflow notebooks use
(`01_ingest_and_transform`, `02_provision_vector_search`,
`04_unity_catalog_tools`) — running it inline just skips the bundle and the job.

**Where the checkout lands.** On the **driver's local filesystem**, under
`~/.dao-ai/git/<host>/<owner>/<repo>/<sha>/`. Nothing is written to Workspace
Files, DBFS, or a UC volume, and the checkout does not survive a cluster restart.
Pass `cache_dir` to put it somewhere durable:

```python
from pathlib import Path

from dao_ai.git_source import GitSource

config = AppConfig.from_git(
    GitSource(
        "gh:org/repo@v1.0#examples/retail/agent.yaml",
        cache_dir=Path("/Volumes/my_catalog/my_schema/my_volume/dao-ai-git"),
    ),
    params={"catalog": "my_catalog"},
)
```

`$DAO_AI_GIT_CACHE` does the same thing for call sites you do not control. Prefer
a per-user destination: anyone with write access to a shared one can change code
that later lands on `sys.path`.

A volume or `/Workspace` destination is a FUSE mount rather than a POSIX
filesystem, and `git` cannot fetch into one — writing a pack index needs
random-access writes the mount does not support. dao-ai handles that by fetching
into a local temporary directory and copying the finished checkout in, so the only
visible difference is that publication is a copy instead of an atomic rename.

**Relative assets resolve inside the checkout.** A dataset's
`ddl: functions/products.sql` or `data: data/products.csv` is anchored on the
config's own directory in the checkout, so a repository's colocated assets work with no
rewriting. So do `skills/`, `code_paths`, and the colocated `src/` convention —
`from_git` puts them on `sys.path` at load time.

**Seed files are staged into a volume for Spark.** Serverless executors cannot
read driver-local files, so a `csv` / `parquet` / `orc` / `delta` dataset is
copied into a managed volume `<catalog>.<schema>.dao_ai_staging` in the dataset's
own target schema and Spark is handed the `/Volumes/...` path. `json` and `excel`
are read on the driver with pandas and need no staging. A `data:` value already
under `/Volumes/` passes through untouched.

**Requirements.** `git` on the driver's `PATH` and network egress to the git
host. For a private repository set the token from a secret scope rather than
inlining it in the locator, which `parse_git_locator` rejects:

```python
import os

os.environ["DAO_AI_GIT_TOKEN"] = dbutils.secrets.get("my-scope", "git-token")
```

Everything under *Trust* above applies with more force here: provisioning
executes the repository's DDL and Python against your catalog. Pin a tag or a
full SHA for repositories you do not control.

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
models = config.resources.models  # `.llms` is a deprecated alias
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
from dao_ai.config import AppConfig, ServingMode

config = AppConfig.from_file("config/my_agent.yaml")

# Package the agent as an MLflow model
config.create_agent()

# Deploy to Databricks Model Serving (default)
config.deploy_agent()

# Deploy directly to Databricks Apps (no asset bundle needed)
config.deploy_agent(mode=ServingMode.APPS)

# Deploy to MCP server (Apps with MCP entrypoint)
config.deploy_agent(mode=ServingMode.APPS, as_mcp=True)
```

> Extra pip packages and custom code paths are declared on the config, not passed
> to `create_agent()`: set `app.pip_requirements: [...]` and `app.code_paths: [...]`
> in your YAML (or on the `AppConfig` object) and they are threaded into the
> generated bundle / logged model automatically.

### Apps deployment and chat UI

When deploying to Apps with `enable_chat_proxy: true` (the default), the
deployed app automatically clones and builds the Databricks
[e2e-chatbot-app-next](https://github.com/databricks/app-templates/tree/main/e2e-chatbot-app-next)
chat UI at startup.  The Apps runtime has Node.js pre-installed, so no
additional tools are needed on your development machine.

All three deploy flows converge on the same runtime behavior:

- `config.deploy_agent(mode=ServingMode.APPS)` (programmatic)
- `dao-ai agent up --mode apps` (CLI — build → sync → start in one command)
- `dao-ai agent build` + `databricks bundle deploy` (standalone bundle)

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
[`notebooks/07_feedback_demo.py`](https://github.com/natefleming/dao-ai/blob/main/notebooks/07_feedback_demo.py) — it
registers `mlflow.search_traces` results as a Spark temp view and
demonstrates `LATERAL VIEW EXPLODE(assessments)` queries for daily
up/down volume and feedback-by-trace.

---

## Navigation

- [← Previous: CLI Reference](cli-reference.md)
- [↑ Back to Documentation Index](https://github.com/natefleming/dao-ai/blob/main/README.md#-documentation)
- [Next: FAQ →](faq.md)

