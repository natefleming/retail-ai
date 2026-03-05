# Configuration Reference

## Full Configuration Structure

```yaml
# Schema definitions for Unity Catalog
schemas:
  my_schema: &my_schema
    catalog_name: string
    schema_name: string

# Reusable variables (secrets, env vars)
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
    postgres_db: &postgres_db
      instance_name: string
      client_id: *api_key       # OAuth credentials
      client_secret: *secret
      workspace_host: string

  warehouses:
    warehouse: &warehouse
      warehouse_id: string
      on_behalf_of_user: bool

  genie_rooms:
    genie: &genie
      space_id: string

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

# Guardrails (MLflow judge-based evaluation)
guardrails:
  guardrail_name: &guardrail_name
    name: string                    # Guardrail identifier
    model: *judge_llm               # LLM model for the MLflow judge
    prompt: string | *prompt_ref    # Evaluation instructions with {{ inputs }} and {{ outputs }}
    num_retries: int | null         # Max retry attempts (default: 3)
    fail_open: bool | null          # Let responses through on error (default: true)
    max_context_length: int | null  # Max tool context chars (default: 8000)
    # Note: {{ inputs }} includes user query + extracted tool context
    # Note: {{ outputs }} includes the agent's response

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
```

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

