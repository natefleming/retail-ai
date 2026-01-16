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
        agent_a: [agent_b, agent_c]
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

## Navigation

- [← Previous: Key Capabilities](key-capabilities.md)
- [↑ Back to Documentation Index](../README.md#-documentation)
- [Next: Examples →](examples.md)

