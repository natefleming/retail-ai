# Configuration Reference

## Full Configuration Structure

```yaml
# Load-time parameters (${var.NAME} substitution)
parameters:
  param_name:
    description: string          # Human-readable description
    default: string | null       # Omit to make required

# Schema definitions for Unity Catalog
schemas:
  my_schema: &my_schema
    catalog_name: string         # supports ${var.NAME} references
    schema_name: string

# Reusable variables (secrets, env vars) - resolved at RUNTIME
variables:
  api_key: &api_key
    options:
      - env: MY_API_KEY
      - scope: my_scope
        secret: api_key

# Infrastructure resources
resources:
  llms:
    model_name: &model_name
      name: string              # Databricks endpoint name
      temperature: float        # 0.0 - 2.0
      max_tokens: int
      fallbacks: [string]       # Fallback model names
      on_behalf_of_user: bool   # Use caller's permissions

  vector_stores:
    store_name: &store_name
      endpoint:
        name: string
        type: STANDARD | OPTIMIZED_STORAGE
        target_qps: int            # optional, STANDARD only, Public Preview
      index:
        schema: *my_schema
        name: string
      source_table:
        schema: *my_schema
        name: string
      embedding_model: *embedding_model
      embedding_source_column: string
      columns: [string]

  databases:
    # Lakebase (autoscaling)
    lakebase_db: &lakebase_db
      project: string              # Lakebase project name
      branch: string               # optional, auto-resolved if omitted
      client_id: *api_key          # OAuth credentials
      client_secret: *secret
      workspace_host: string

    # Standard PostgreSQL
    postgres_db: &postgres_db
      host: string
      port: int
      database: string
      user: string
      password: string

  warehouses:
    warehouse: &warehouse
      warehouse_id: string         # or omit and provide name instead
      name: string                 # resolves warehouse_id by name if warehouse_id is omitted
      on_behalf_of_user: bool

  genie_rooms:
    genie: &genie
      space_id: string             # or omit and provide name instead
      name: string                 # resolves space_id by title if space_id is omitted

# Retriever configurations
retrievers:
  retriever_name: &retriever_name
    vector_store: *store_name
    columns: [string]
    search_parameters:
      num_results: int
      query_type: ANN | HYBRID

# Tool definitions
tools:
  tool_name: &tool_name
    name: string
    function:
      type: python | factory | unity_catalog | mcp
      name: string              # Import path or UC function name
      args: {}                  # For factory tools
      schema: *my_schema        # For UC tools
      # MCP-specific options
      url: string               # MCP server URL
      connection: *connection   # UC Connection for MCP
      sql: bool                 # Use DBSQL MCP server
      functions: *my_schema     # Use UC Functions MCP
      genie_room: *genie        # Use Genie MCP
      vector_search: *store     # Use Vector Search MCP
      include_tools: [string]   # Tools to load (allowlist, supports glob)
      exclude_tools: [string]   # Tools to exclude (denylist, supports glob)
      human_in_the_loop:        # Optional approval gate
        review_prompt: string
        allowed_decisions: [approve, edit, reject]

# Agent definitions
agents:
  agent_name: &agent_name
    name: string
    description: string
    model: *model_name
    tools: [*tool_name]
    guardrails: [*guardrail_ref]
    prompt: string | *prompt_ref
    handoff_prompt: string      # For swarm routing
    requires: [*agent_name]     # Swarm only: prerequisite agents that must have
                                # run before this agent can be reached. Empty by
                                # default. See architecture.md → Swarm Pattern →
                                # Handoff constraints.
    middleware: [*middleware_ref]
    response_format: *response_format_ref | string | null

# Prompt definitions (MLflow registry)
prompts:
  prompt_name: &prompt_name:
    schema: *my_schema
    name: string
    alias: string | null        # e.g., "production"
    version: int | null
    default_template: string
    tags: {}

# Guardrails (MLflow judge-based or Scorer-based evaluation)
guardrails:
  # Custom judge mode (model + prompt)
  guardrail_name: &guardrail_name
    name: string                    # Guardrail identifier
    model: *judge_llm               # LLM model for the MLflow judge
    prompt: string | *prompt_ref    # Evaluation instructions with {{ inputs }} and {{ outputs }}
    num_retries: int | null         # Max retry attempts (default: 3)
    fail_on_error: bool | null      # Block responses on evaluation error (default: false)
    max_context_length: int | null  # Max tool context chars (default: 8000)

  # Scorer mode (scorer + scorer_args)
  scorer_guardrail: &scorer_guardrail
    name: string                    # Guardrail identifier
    scorer: string                  # FQN of mlflow.genai.scorers.base.Scorer class
    scorer_args: {}                 # Kwargs passed to scorer constructor (default: {})
    num_retries: int | null         # Max retry attempts (default: 3)
    fail_on_error: bool | null      # Block responses on evaluation error (default: false)
    max_context_length: int | null  # Max tool context chars (default: 8000)

# Response format (structured output)
response_formats:
  format_name: &format_name
    response_schema: string | type   # JSON schema string or type reference
    use_tool: bool | null             # null=auto, true=ToolStrategy, false=ProviderStrategy

# Memory configuration
memory: &memory
  checkpointer:
    name: string
    type: memory | postgres | lakebase
    database: *postgres_db      # For postgres
    schema: *my_schema           # For lakebase
    table_name: string           # For lakebase
  store:
    name: string
    type: memory | postgres | lakebase
    database: *postgres_db       # For postgres
    schema: *my_schema            # For lakebase
    table_name: string            # For lakebase
    embedding_model: *embedding_model
    dims: int | null              # Auto-detected from embedding model if omitted
  extraction:                              # Long-term memory extraction
    schemas: [string]                      # Schema names: user_profile, preference, episode
    instructions: string | null            # Custom extraction instructions
    auto_inject: bool                      # Inject memories into prompts (default: true)
    auto_inject_limit: int                 # Max memories to inject (default: 5)
    background_extraction: bool            # Extract in background thread (default: false)
    extraction_model: *llm_model | null    # Separate LLM for extraction
    query_model: *llm_model | null         # Separate LLM for search queries

# Application configuration
app:
  name: string
  description: string
  log_level: DEBUG | INFO | WARNING | ERROR
  
  registered_model:
    schema: *my_schema
    name: string
  
  endpoint_name: string
  
  agents: [*agent_name]
  
  orchestration:
    supervisor:                 # OR swarm, not both
      model: *model_name
      prompt: string
    swarm:
      default_agent: *agent_name
      handoffs:
        agent_a: [agent_b, agent_c]          # agentic handoffs (LLM decides)
        agent_b:
          - agent: agent_c                   # HandoffRouteModel
            is_deterministic: true           # deterministic: always route here
          - agent_a                          # agentic: LLM decides via tool
      middleware: [*middleware_ref]
    memory: *memory
  
  initialization_hooks: [string]
  shutdown_hooks: [string]
  
  permissions:
    - principals: [users]
      entitlements: [CAN_QUERY]
  
  environment_vars:
    KEY: "{{secrets/scope/secret}}"
  
  enable_chat_proxy: true          # default; set false for API-only
```

### Vector Search endpoint capacity (`target_qps`)

**`vector_stores.<name>.endpoint.target_qps`** *(int, optional, Public Preview)* —
Target queries-per-second for the Vector Search endpoint. **STANDARD endpoints only**;
setting this on an `OPTIMIZED_STORAGE` endpoint raises a config-validation error.
Endpoint compute scales linearly with `target_qps`, so cost scales linearly too.
**Honored at endpoint-creation time only** — if the endpoint already exists, this
value is ignored (a debug log entry records the configured value but no API call
is made). To change capacity on a live endpoint, use the Databricks UI, REST API,
or SDK directly. See the [Databricks Vector Search QPS scaling docs](https://docs.databricks.com/aws/en/generative-ai/vector-search) for the underlying capability.

### Chat UI (`enable_chat_proxy`)

Controls whether the deployed Databricks App includes the interactive chat UI
alongside the agent backend.

| Value | Behaviour |
|-------|-----------|
| `true` (default) | The app runs both a Python backend (port 8000) and a Node.js chat frontend (port 3000). The MLflow `AgentServer` proxies browser requests to the frontend. The chat UI is the Databricks [e2e-chatbot-app-next](https://github.com/databricks/app-templates/tree/main/e2e-chatbot-app-next) template, cloned and built automatically at app startup (the Apps runtime has Node.js pre-installed). |
| `false` | The app runs the Python backend only (`dao_ai.apps.server`). No chat UI. Useful for headless API endpoints or Model Serving deployments. |

---

## Parameters (Load-Time Substitution)

Configs can declare typed input parameters and reference them inline with `${var.NAME}` (or its alias `${param.NAME}`). Substitution happens once at load time, **before** MLflow's `ModelConfig` parses the YAML. This makes one YAML re-usable across catalogs, schemas, environments, and workshop modules without duplicating files.

### Declaring parameters

Add a top-level `parameters:` block. Each entry can include a `description` and an optional `default`. Omitting `default` makes the parameter required.

```yaml
parameters:
  catalog:
    description: Unity Catalog catalog name
    default: main
  schema:
    description: Schema for workshop tables
    default: dao_ai
  module_id:
    description: Workshop module identifier
    # no default => required

schemas:
  workshop_schema:
    catalog_name: ${var.catalog}
    schema_name: ${var.schema}

app:
  name: dao_ws_${var.module_id}_orchestration
```

### Reference syntax

Two prefixes are supported as interchangeable aliases:

- `${var.NAME}` - matches the Databricks Asset Bundle convention (recommended).
- `${param.NAME}` - matches the `parameters:` block name.

Both can appear in the same file and resolve against the same declaration. Inline defaults are also supported: `${var.NAME:-fallback}`.

### Resolution precedence

Each reference is resolved in this order:

1. **CLI** `--var name=value` (or `AppConfig.from_file(params={...})`)
2. **Process env** - `NAME` upper-cased with `.` and `-` replaced by `_` (e.g. `${var.app.catalog-name}` reads from `APP_CATALOG_NAME`)
3. **Declared default** - the `default:` entry in the `parameters:` block
4. **Inline default** - `${var.NAME:-fallback}` on the reference itself
5. **Error** - raises `ConfigVariableError`

### Error handling

Two classes of error are caught at load time:

**Missing required** - a declared parameter with no `default` and no override:

```
Config parameter error in dao_ai.yaml:
  missing required: module_id.
  Pass with --var name=value or set the equivalent env var.
```

**Undeclared reference** - a `${var.NAME}` used in the YAML but not in the `parameters:` block (typo protection):

```
Config parameter error in dao_ai.yaml:
  undeclared ${var.NAME} / ${param.NAME} references: catlaog.
  Add them to the top-level parameters: block.
```

### YAML quoting caveat

Substitution is text-level - the value is spliced into the YAML before parsing. If a value may contain YAML-special characters (`:` followed by a space, `#`, `[`, `{`, newlines, quotes), quote the reference:

```yaml
prompt: "${var.user_prompt}"   # safe regardless of value content
label: ${var.label}            # OK only for plain alphanumeric values
```

### Non-recursion

Substitution does not recurse. If a substituted value happens to contain `${var.x}` literally, it is preserved as-is and not re-resolved.

### Bundle behaviour

When `dao-ai generate-bundle` writes the deployable Apps bundle, the emitted config YAML has every reference substituted to a literal value and the `parameters:` block dropped. The deployed app does not need the original `--var` flags.

---

## Dynamic Configuration with AnyVariable

Many configuration fields support dynamic values through the `AnyVariable` type, which allows values to be loaded from environment variables, Databricks secrets, or provide fallback chains.

### Supported Fields

The following fields support `AnyVariable`:

- **SchemaModel**: `catalog_name`, `schema_name`
- **DatabricksAppModel**: `url`
- And many other resource and configuration fields

### Usage Patterns

**Plain String (Static Value)**
```yaml
schemas:
  my_schema:
    catalog_name: production_catalog
    schema_name: analytics
```

**Environment Variable**
```yaml
schemas:
  my_schema:
    catalog_name:
      env: DATABRICKS_CATALOG
    schema_name:
      env: DATABRICKS_SCHEMA
```

**Databricks Secret**
```yaml
schemas:
  my_schema:
    catalog_name:
      scope: my_scope
      secret: catalog_name
```

**Composite with Fallback Chain**
```yaml
schemas:
  my_schema:
    catalog_name:
      options:
        - env: PROD_CATALOG        # Try environment variable first
        - scope: prod_secrets      # Fall back to Databricks secret
          secret: catalog_name
        - default_value: main      # Final fallback
```

**Databricks App URL**
```yaml
resources:
  apps:
    my_app:
      name: dao_ai_app
      url:
        env: DATABRICKS_APP_URL
        default_value: https://my-app.databricksapps.com
```

### Benefits

- **Environment Flexibility**: Same config works across dev/staging/prod
- **Security**: Keep sensitive values in secrets, not config files
- **Portability**: Easy multi-cloud and multi-workspace deployments
- **Resilience**: Fallback chains ensure configuration succeeds
- **Backwards Compatible**: Plain strings still work for static values

### Parameters vs Variables - the Lifecycle Distinction

`parameters:` and `variables:` look similar but solve different problems at different lifecycle stages. Use this table to pick the right one:

| | `parameters:` | `variables:` |
|---|---|---|
| **When resolved** | Load time, by `AppConfig.from_file` | Runtime, when `as_value()` is called inside the deployed app |
| **Source of value** | `--var`, env, declared default, inline `:-default` | `env` / `scope`+`secret` / composite at runtime |
| **Reference syntax** | `${var.NAME}` or `${param.NAME}` (inline string macro) | YAML anchor `*name` (typed mapping spliced into a field) |
| **Scope of effect** | Anywhere in any string in the YAML | Wherever the anchor expands |
| **What ends up in the bundle** | Resolved literal value, declarations dropped | The typed mapping itself, evaluated at runtime |
| **Use for** | Catalog/schema/app names, table prefixes, prompt fragments | Credentials, hostnames, secrets - anything the deployed runtime must read live |

**Rule of thumb:** If the value should travel with the bundle, use `parameters:`. If it must be read from the deployed environment or Databricks Secrets each time the agent runs, use `variables:`.

### Bridge Pattern: Parameters Feeding Variables

`${var.NAME}` references work inside any string field - including fields inside typed `variables:` entries. This lets parameters control _where_ a secret lives without touching the runtime resolution model.

```yaml
parameters:
  secret_scope:
    description: Databricks secrets scope holding service-principal creds
    default: dao_ai
  client_id_secret_key:
    description: Secret key for the SP client id
    default: SP_CLIENT_ID

variables:
  client_id: &client_id
    options:
      - scope: ${var.secret_scope}
        secret: ${var.client_id_secret_key}
      - env: ${var.client_id_secret_key}
```

At load time, `${var.secret_scope}` and `${var.client_id_secret_key}` are text-substituted to their literal values. The resulting `variables:` entry is then parsed normally as a `CompositeVariableModel` with a `SecretVariableModel` and an `EnvironmentVariableModel` - both resolved at runtime using the parameterised scope and key names.

Override at deploy time:

```bash
dao-ai pipeline --deploy -c dao_ai.yaml --var secret_scope=prod_dao_ai --var client_id_secret_key=PROD_SP_CLIENT_ID
```

**What this does NOT do:** You cannot substitute a parameter for an _entire_ typed mapping - only for string fields inside one. This works:

```yaml
variables:
  cred:
    scope: ${var.scope}    # OK - string field inside a typed mapping
    secret: ${var.key}     # OK
```

This does not:

```yaml
variables:
  cred: ${var.whole_thing}  # NO - the typed mapping is not a string
```

---

## MCP Tool Filtering

MCP servers can expose many tools. Use `include_tools` and `exclude_tools` to control which tools are loaded.

### Basic Usage

**Allowlist (Include Only)**
```yaml
tools:
  sql_mcp:
    name: sql_safe
    function:
      type: mcp
      sql: true
      include_tools:
        - execute_query      # Exact name
        - list_tables
        - "query_*"          # Glob pattern
```

**Denylist (Exclude)**
```yaml
tools:
  sql_mcp:
    name: sql_readonly
    function:
      type: mcp
      sql: true
      exclude_tools:
        - "drop_*"           # Glob pattern
        - "delete_*"
        - execute_ddl
```

**Hybrid (Include + Exclude)**
```yaml
tools:
  functions_mcp:
    function:
      type: mcp
      functions: *schema
      include_tools: ["query_*", "get_*"]
      exclude_tools: ["*_sensitive"]  # Exclude overrides include
```

### Pattern Syntax

Supports glob patterns from Python's `fnmatch`:

| Pattern | Description | Example |
|---------|-------------|---------|
| `*` | Any characters | `query_*` → `query_sales`, `query_inventory` |
| `?` | Single character | `tool_?` → `tool_a`, `tool_b` |
| `[abc]` | Char in set | `tool_[123]` → `tool_1`, `tool_2` |
| `[!abc]` | Char NOT in set | `tool_[!abc]` → `tool_d` |

### Precedence Rules

1. **exclude_tools** always takes precedence over include_tools
2. If **include_tools** is specified, only matching tools load (allowlist)
3. If **exclude_tools** is specified, matching tools are blocked (denylist)
4. If neither is specified, all tools load (default behavior)

### Common Patterns

**Read-Only SQL**
```yaml
include_tools: ["query_*", "list_*", "describe_*", "get_*"]
```

**Block Dangerous Operations**
```yaml
exclude_tools: ["drop_*", "delete_*", "truncate_*", "execute_ddl"]
```

**Development Mode**
```yaml
exclude_tools: ["drop_*", "truncate_*"]  # Block only critical ops
```

**Maximum Security**
```yaml
include_tools: ["execute_query", "list_tables"]  # Only these 2
```

### See Also

- Full examples: [`config/examples/02_mcp/filtered_mcp.yaml`](../config/examples/02_mcp/filtered_mcp.yaml)
- MCP documentation: [`config/examples/02_mcp/README.md`](../config/examples/02_mcp/README.md#mcp-tool-filtering)

---

## Deep Agents Middleware

DAO AI provides factory functions for the [Deep Agents](https://pypi.org/project/deepagents/) middleware stack. These are configured in the `middleware` section using `name` (factory import path) and `args` (keyword arguments).

### Factory Configuration Pattern

```yaml
middleware:
  my_middleware: &my_middleware
    name: dao_ai.middleware.<module>.create_<type>_middleware
    args:
      backend_type: state          # state | filesystem | store | volume
      root_dir: /workspace         # Required for backend_type: filesystem
      volume_path: /Volumes/c/s/v  # Required for backend_type: volume
      # ... additional factory-specific args
```

### Available Factories

```yaml
middleware:
  # Task planning -- adds write_todos tool
  todo: &todo
    name: dao_ai.middleware.todo.create_todo_list_middleware
    args:
      system_prompt: string | null       # Custom system prompt (optional)
      tool_description: string | null    # Custom tool description (optional)

  # File operations -- adds ls, read_file, write_file, edit_file, glob, grep
  filesystem: &filesystem
    name: dao_ai.middleware.filesystem.create_filesystem_middleware
    args:
      backend_type: state                # state | filesystem | store | volume
      root_dir: string | null            # Required for filesystem backend
      volume_path: string | null         # Required for volume backend
      tool_token_limit_before_evict: int | null  # Default: 20000, null to disable
      system_prompt: string | null       # Custom system prompt (optional)

  # Subagent spawning -- adds task tool
  subagent: &subagent
    name: dao_ai.middleware.subagent.create_subagent_middleware
    args:
      subagents:                         # List of subagent specifications
        - name: string
          description: string
          system_prompt: string
          model: string | LLMModel dict  # See "Subagent model" note below
          tools: [object]
      backend_type: state
      root_dir: string | null
      volume_path: string | null
      system_prompt: string | null       # Custom system prompt for task tool
      task_description: string | null    # Custom task tool description

  # AGENTS.md memory -- loads context from AGENTS.md files
  memory: &memory
    name: dao_ai.middleware.memory_agents.create_agents_memory_middleware
    args:
      sources: [string]                  # Required: list of AGENTS.md paths
      backend_type: state
      root_dir: string | null
      volume_path: string | null

  # Skill discovery -- discovers SKILL.md files
  skills: &skills
    name: dao_ai.middleware.skills.create_skills_middleware
    args:
      sources: [string]                  # Required: list of skill source paths
      backend_type: state
      root_dir: string | null
      volume_path: string | null

  # Enhanced summarization -- backend offloading + arg truncation
  summarization: &summarization
    name: dao_ai.middleware.summarization.create_deep_summarization_middleware
    args:
      model: string                      # Required: model identifier
      backend_type: state
      root_dir: string | null
      volume_path: string | null
      trigger: [string, int] | null      # e.g. ["tokens", 100000]
      keep: [string, int]                # Default: ["messages", 20]
      history_path_prefix: string        # Default: /conversation_history
      truncate_args_trigger: [string, int] | null
      truncate_args_keep: [string, int]  # Default: ["messages", 20]
      truncate_args_max_length: int      # Default: 2000
```

### Backend Types

| Backend | Description | Required Args |
|---------|-------------|---------------|
| `state` (default) | Ephemeral storage in LangGraph state | None |
| `filesystem` | Real disk storage | `root_dir` |
| `store` | Persistent via LangGraph Store | None |
| `volume` | Databricks Unity Catalog Volume | `volume_path` |

The `volume` backend uses the Databricks SDK `WorkspaceClient.files` API. The `volume_path` must start with `/Volumes/` and can be either a string path (e.g. `/Volumes/catalog/schema/volume`) or reference a `VolumePathModel` from the config.

### Subagent Model

The `model` field in each subagent specification supports multiple formats:

| Format | Description | Example |
|--------|-------------|---------|
| String | `"provider:model"` identifier, passed directly to deepagents | `"openai:gpt-4o-mini"` |
| Dict (LLMModel) | Mapping of `LLMModel` fields, converted to `ChatDatabricks` via `LLMModel.as_chat_model()` | `{name: "my-endpoint", temperature: 0.1}` |
| LLMModel instance | DAO AI `LLMModel` object (Python API only), converted via `as_chat_model()` | `LLMModel(name="my-endpoint")` |
| BaseChatModel instance | LangChain chat model (Python API only), passed through directly | `ChatDatabricks(model="my-endpoint")` |

**YAML example with a Databricks serving endpoint:**

```yaml
subagents:
  - name: analyst
    description: "Data analysis agent"
    system_prompt: "You are a data analyst."
    model:
      name: "databricks-meta-llama-3-3-70b-instruct"
      temperature: 0.1
      max_tokens: 4096
    tools: []
```

### See Also

- Full example: [`config/examples/12_middleware/deepagents_middleware.yaml`](../config/examples/12_middleware/deepagents_middleware.yaml)
- Middleware examples: [`config/examples/12_middleware/README.md`](../config/examples/12_middleware/README.md)

---

## Navigation

- [← Previous: Key Capabilities](key-capabilities.md)
- [↑ Back to Documentation Index](../README.md#-documentation)
- [Next: Examples →](examples.md)

