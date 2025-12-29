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
      model: *model_name
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

## Navigation

- [← Previous: Key Capabilities](key-capabilities.md)
- [↑ Back to Documentation Index](../README.md#-documentation)
- [Next: Examples →](examples.md)

