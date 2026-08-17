# Configuration Reference

## Full Configuration Structure

```yaml
# Load-time parameters (${var.NAME} substitution)
parameters:
  param_name:
    description: string          # Human-readable description
    default: string | null       # Omit to make required
    provided: bool               # default false. When true, the value is
                                 # supplied DYNAMICALLY at run time (like a
                                 # build tool's 'provided' dependency scope) —
                                 # e.g. a Genie space id created by the workflow's
                                 # provision-genie task and forwarded via
                                 # taskValues. An unsupplied `provided` param
                                 # resolves to "" at load time (or its `default`)
                                 # instead of erroring, so ${var.NAME} refs load
                                 # before the real value exists. apps/mcp/
                                 # model_serving deploys (no provisioning step)
                                 # reject an unsupplied, defaultless `provided`
                                 # param rather than ship a broken binding.

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
  # Inference endpoint definitions. Backs every serving endpoint dao-ai
  # calls at runtime: chat LLMs, embeddings, judges, extraction /
  # reflection / query models, and custom agent endpoints. The previous
  # key `resources.llms` and class name `LLMModel` remain as
  # backward-compat aliases — prefer `resources.models` /
  # `InferenceEndpointModel` in new configs.
  models:
    model_name: &model_name
      name: string                # Serving endpoint name (e.g. databricks-claude-opus-4-6)
      description: string         # optional, human-readable
      temperature: float          # 0.0 - 2.0, default 0.1
      max_tokens: int             # default 8192
      fallbacks: [string]         # Fallback endpoint names (or full InferenceEndpointModel configs)
      on_behalf_of_user: bool     # Forward the caller's identity (OBO)
      use_responses_api: bool     # Use Responses API for ResponsesAgent endpoints
      disable_streaming: bool     # Required when output guardrails are enabled; also required when ai_gateway is true and the model uses with_structured_output
      ai_gateway: bool            # dao-ai 0.1.77+: route via /ai-gateway/mlflow/v1/chat/completions instead of /serving-endpoints/<name>/invocations
      best_of_n:                  # optional, dao-ai 0.1.72+
        n: int                    # parallel candidate generations, 1..16
        judge: string | *model_name   # endpoint name or full InferenceEndpointModel
        temperature_override: float   # optional candidate-call temperature
      # Auth fields (all optional — falls back to the App's identity).
      # Use exactly one of: service_principal, (client_id + client_secret),
      # or pat. workspace_host is required only when targeting a different
      # workspace.
      service_principal: *sp_ref  # or inline ServicePrincipalModel
      client_id: *api_key
      client_secret: *secret
      workspace_host: string
      pat: *secret

  # `vector_stores` is a discriminated union — each entry is either an
  # AiSearchVectorStoreModel (Databricks AI Search index) or a
  # LakebaseVectorStoreModel (Postgres table with lakebase_vector /
  # lakebase_text extensions). The `type:` field selects the concrete
  # class; when omitted, defaults to `ai_search` for back-compat with
  # legacy configs. Both types can co-exist under the same dict.
  vector_stores:
    # AI Search store — the historical default. `type: ai_search` is
    # implicit when omitted.
    ai_store: &ai_store
      type: ai_search             # optional (default)
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

    # Lakebase Postgres store — `type: lakebase_search` is required.
    # Auth flows through the nested `database` (DatabaseModel); no
    # `endpoint` / `index` fields.
    lakebase_store: &lakebase_store
      type: lakebase_search        # required for the Lakebase branch
      database: *lakebase_db       # DatabaseModel reference
      schema_name: public          # Postgres schema
      table: string                # Postgres table with vector column
      content_column: string       # text column returned as Document.page_content
      embedding_column: string     # VECTOR(N) column indexed by lakebase_ann
      tsvector_column: string      # optional, required for BM25 / HYBRID
      embedding_model: *embedding_model
      metadata_columns: [string]
      distance_metric: cosine | l2 | ip

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
      agent_id: string             # alias of space_id (Genie Spaces → Genie Agents)
      name: string                 # resolves space_id by title if space_id is omitted
      on_behalf_of_user: bool      # forward the caller's token to Genie (OBO)
      # A room referenced by a GenieAgentModel (see "Genie Agent as a model")
      # MUST be registered here so the deploy emits the genie-space grant and,
      # when on_behalf_of_user is set, the dashboards.genie user_api_scope.

  # Unity Catalog references (used to wire deployment resources and grants)
  tables:
    table_name: &table_name
      schema: *my_schema
      name: string

  volumes:
    volume_name: &volume_name
      schema: *my_schema
      name: string

  functions:
    function_name: &function_name
      schema: *my_schema
      name: string

  # UC Connection references for MCP / external data sources
  connections:
    connection_name: &connection_name
      name: string

  # Other Databricks Apps used as MCP endpoints or tool backends
  apps:
    app_name: &app_name
      name: string

  # Deepagents skills — see "Skills (`resources.skills`)" below for details
  skills:
    skill_name: &skill_name
      name: string                          # Unique skill identifier
      description: string | null            # Surfaced in docs/traces
      path: string | *volume_path_model     # Local path string OR VolumePathModel
                                            # Raw "/Volumes/..." strings auto-promote
                                            # to VolumePathModel by the validator.

# Retriever configurations — discriminated union of AiSearchRetrieverModel
# and LakebaseRetrieverModel. The ``type`` field defaults to ``ai_search``
# when omitted, so existing YAMLs continue to parse unchanged.
retrievers:
  # AI Search retriever (default when ``type`` is omitted)
  products_retriever: &products_retriever
    type: ai_search                             # optional (default)
    vector_store: *store_name                   # AiSearchIndexModel reference
    columns: [string]
    search_parameters:
      num_results: int
      query_type: ANN | HYBRID
    rerank: bool | *rerank_params_model         # optional FlashRank / instruction-aware
    instructed: *instructed_retriever_model     # optional query decomposition

  # Lakebase Postgres retriever
  kb_retriever: &kb_retriever
    type: lakebase_search                       # required for the Lakebase branch
    vector_store: *lakebase_vector_store_model  # LakebaseVectorStoreModel reference
    columns: [string | *column_info]
    search_parameters:
      num_results: int
      query_type: ANN | BM25 | HYBRID

# Tool definitions
tools:
  tool_name: &tool_name
    name: string
    function:
      type: python | factory | unity_catalog | mcp | sql | app | serving_endpoint | a2a
      name: string              # Import path or UC function name
      args: {}                  # For factory tools
      schema: *my_schema        # For UC tools
      # MCP-specific options
      url: string               # MCP server URL
      connection: *connection   # UC Connection for MCP
      sql: bool                 # Use DBSQL MCP server
      functions: *my_schema     # Use UC Functions MCP
      genie_room: *genie        # Use Genie MCP for a single space (per-space URL)
      genie: bool               # Use workspace-wide Genie MCP (all spaces, no space_id)
      vector_search: *store     # Use AI Search MCP (field name kept for backwards compat)
      include_tools: [string]   # Tools to load (allowlist, supports glob)
      exclude_tools: [string]   # Tools to exclude (denylist, supports glob)
      meta:                     # _meta sent on every tool call (MCP spec, public preview on Databricks)
        warehouse_id: string    # DBSQL: pin a specific warehouse
        num_results: int        # AI Search: cap result count
        # ... any other server-specific keys (see Databricks managed MCP docs)
      # type: app — call a Databricks App as a tool
      app: *app_resource        # DatabricksAppModel ref (required for type: app)
      # type: serving_endpoint — call a Model Serving endpoint as a tool
      endpoint: string | *model # Endpoint name (sugar) or InferenceEndpointModel
      # type: app + type: serving_endpoint — OpenAI wire-shape selector
      api: responses | completions | null   # null (default) = lazy discovery
      # type: a2a — call an external A2A agent (Google A2A v0.3)
      auth: forwarded_user_token | databricks_app_sp | string  # see a2a docs
      human_in_the_loop:        # Optional approval gate
        review_prompt: string
        allowed_decisions: [approve, edit, reject]

# Agent definitions
agents:
  agent_name: &agent_name
    name: string
    description: string
    model: *model_name          # InferenceEndpointModel (serving endpoint) OR a
                                # GenieAgentModel (Genie Agent as a streaming brain —
                                # see "Genie Agent as a model" below). A bare Genie
                                # room anchor is auto-wrapped into a GenieAgentModel;
                                # a name-only room cannot be (see that section).
    tools: [*tool_name]
    guardrails: [*guardrail_ref]
    prompt: string | *prompt_ref
    handoff_prompt: string      # For swarm routing
    requires: [*agent_name]     # Swarm only: prerequisite agents that must have
                                # run before this agent can be reached. Empty by
                                # default. See architecture.md → Swarm Pattern →
                                # Handoff constraints.
    middleware: [*middleware_ref]
    skills: [*skill_name]       # SkillModel refs OR inline SkillModel entries.
                                # Each entry produces a SkillsMiddleware appended
                                # to this agent's middleware stack. Works under
                                # supervisor, swarm, and deep_agent.
    response_format: *response_format_ref | string | null
    recursion_limit: int | null # Max LangGraph supersteps per invocation (default 25)

# Prompt definitions (reusable inline prompts)
prompts:
  prompt_name: &prompt_name:
    schema: *my_schema          # Optional UC schema (label only)
    name: string
    template: string            # Prompt text with optional {variable} placeholders
    description: string | null
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

# Named middleware definitions (cross-cutting concerns reusable via anchors)
# Each entry is a MiddlewareModel: a factory function FQN + its kwargs.
# See "Deep Agents Middleware" below for the available factories shipped
# with dao-ai. Custom middleware can be added by pointing `name` at any
# importable factory that returns a LangGraph AgentMiddleware instance.
middleware:
  middleware_name: &middleware_name
    name: string                       # FQN of the middleware factory function
    args: {}                            # Kwargs forwarded to the factory

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
    supervisor:                 # Exactly one of supervisor / swarm / deep_agent
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
    deep_agent:                 # Wraps deepagents.create_deep_agent
      model: *model_name | string | null     # Primary LLM. Strings pass through
                                             # to init_chat_model (e.g. "openai:gpt-4o").
                                             # Defaults to deepagents' default if omitted.
      system_prompt: string | *prompt_ref | null
      tools: [*tool_name]                    # Merged with deepagents' built-in suite
                                             # (todo, filesystem, execute, task)
      middleware: [*middleware_ref]          # User middleware between base & tail stacks
      subagents:                             # Callable via the `task` tool. Three forms:
        - *agent_name                        # (1) string → entry in app.agents
        - name: research                     # (2) inline SubAgentModel
          description: string
          system_prompt: string | *prompt_ref
          model: *model_name | string | null
          tools: [*tool_name]
          middleware: [*middleware_ref]
          skills: [*skill_name]
          permissions: [*filesystem_permission_ref]
          interrupt_on:
            tool_name: true | *human_in_the_loop_ref
          response_format: *response_format_ref | string | null
        - *agent_name                        # (3) full AgentModel inline
      skills: [*skill_name]                  # Skill paths or SkillModel refs exposed
                                             # via deepagents' SkillsMiddleware.
      instruction_files: [string]            # AGENTS.md-style files loaded into the
                                             # system prompt at startup.
      permissions: [*filesystem_permission_ref] # Inherited by sub-agents
      interrupt_on:
        tool_name: true | *human_in_the_loop_ref
      backend:                               # Optional BackendModel for state/store
        type: state | filesystem | store | volume
        root_dir: string | null
        volume_path: string | null
      context_schema: string | null          # FQN of TypedDict/dataclass for run context
      recursion_limit: int | null
      debug: bool
      name: string | null                    # Shows in MLflow trace dashboards
      response_format: *response_format_ref | string | null
    memory: *memory
    output_mode: full_history | last_message # Default: full_history
  
  initialization_hooks: [string]
  shutdown_hooks: [string]
  
  permissions:
    - principals: [users]
      entitlements: [CAN_QUERY]
  
  environment_vars:
    KEY: "{{secrets/scope/secret}}"
  
  enable_chat_proxy: true          # default; set false for API-only
  scale_to_zero: bool              # default: true
  workload_size: Small | Medium | Large   # default: Small
  python_version: string           # default: "3.12"
  # deployment_target field removed — serving mode is chosen at deploy time via --mode (default apps)
  budget_policy_id: string         # Cost-attribution policy id
  code_paths: [string]             # Extra Python files bundled with the model artifact
  pip_requirements: [string]       # Extra pip packages installed in the serving env
  tags: {}                          # Key-value tags on the registered model version
  alias: string                     # Model version alias assigned after registration (e.g. "champion")
  input_example: {}                # Example chat payload logged alongside the model

  # Conversation summarization (long-running chats)
  chat_history:
    model: *summary_llm            # LLM used to generate summaries
    max_tokens: int                # Default 2048; tokens kept after summarization
    max_tokens_before_summary: int | null     # Triggers summarization at this token count
    max_messages_before_summary: int | null   # OR triggers at this message count (mutually exclusive)

  # OTEL trace storage in Unity Catalog Delta tables.
  # Requires an explicit post-deploy link step:
  #   `dao-ai trace link -c my_config.yaml -p <profile>`
  # See `docs/cli-reference.md#trace-commands` for the flow and
  # for the migration playbook. IMPORTANT: once an experiment is linked
  # to a UC destination, Databricks does NOT allow un-linking or
  # changing the destination (verified live — the server rejects
  # `unset_experiment_trace_location`). Changing catalog / schema /
  # table_prefix requires creating a fresh experiment.
  trace_location:
    # Either provide a schema + warehouse (preferred), or pass a single
    # "catalog.schema" string and the warehouse separately.
    schema: *my_schema
    warehouse: *warehouse | string    # WarehouseModel ref OR warehouse-id string
    table_prefix: string | null       # Prefix for the OTEL tables
                                      # (<prefix>_otel_{spans,logs,metrics}).
                                      # null → MLflow uses the experiment id
                                      # as the prefix (backend-assigned).
                                      # PERMANENT once linked — see the note
                                      # above on the trace_location block.

  # Production monitoring via MLflow GenAI scorers
  monitoring:
    sample_rate: float                          # Built-in scorers (default 1.0)
    scorers: [string | *guardrail_ref] | null   # Names/globs/GuardrailModel refs.
                                                # Built-ins: safety, completeness,
                                                # relevance_to_query, tool_call_efficiency.
                                                # null → all built-ins.
    guidelines:                                 # Guidelines-scorer configurations
      - name: string
        guidelines: [string]
    guidelines_sample_rate: float               # Guidelines scorers (default 0.5)

  # Opt-in background agent (Responses-API kickoff/poll/cancel)
  background:
    database: *lakebase_db                       # Persistence backend (Lakebase or Postgres)
    default_enabled: bool                        # default: false
    max_duration_seconds: int                    # default: 1800
    poll_interval_seconds: float                 # default: 1.0
    responses_table_name: string                 # default: dao_ai_responses
    messages_table_name: string                  # default: dao_ai_response_messages

# Offline evaluation (MLflow GenAI scorers)
evaluation:
  model: *judge_llm                # Judge LLM for LLM-based scorers
  table: *table_name               # UC table where eval results are stored
  num_evals: int                    # Number of synthetic samples to generate
  replace: bool                     # default: false; drop+recreate table & dataset
  agent_description: string | null # Used by the question generator
  question_guidelines: string | null
  custom_inputs: {}                 # Extra inputs forwarded to the agent during eval
  guidelines:                       # Guidelines-scorer configs
    - name: string
      guidelines: [string]

# Cache-threshold optimization + training/evaluation datasets
optimizations:
  training_datasets:
    dataset_name: &eval_dataset
      schema: *my_schema
      name: string
      overwrite: bool                            # default: false
      data:                                      # Inline EvaluationDatasetEntry list
        - inputs: {}                             # ChatPayload
          expectations:
            expected_response: string | null
            expected_facts: [string] | null      # Mutually exclusive with expected_response

  cache_threshold_optimizations:
    optimize_cache:
      name: string                                # Bayesian cache-threshold optimization
      # See examples/13_optimization/ for the full schema
```

### AI Gateway routing (`ai_gateway`)

**`resources.models.<name>.ai_gateway`** *(bool, optional, default `false`,
dao-ai 0.1.77+)* — Route this model through the Databricks AI Gateway
(`POST /ai-gateway/mlflow/v1/chat/completions`) instead of the legacy
Model Serving path (`POST /serving-endpoints/<name>/invocations`). When
`true`, `name` is sent as the OpenAI-style model id in the request
body, and dao-ai constructs a `langchain_openai.ChatOpenAI` client
(rather than `databricks_langchain.ChatDatabricks`) pointed at the
gateway base URL. The flag is additive — existing configs are unaffected.

```yaml
resources:
  models:
    gateway_llm: &gateway_llm
      name: databricks-claude-opus-4-6
      ai_gateway: true
      temperature: 0.1
      max_tokens: 1024
```

**Why this exists.** `ChatDatabricks` (≤ `databricks-langchain` 0.19.0)
has no `base_url` override and cannot target the AI Gateway path. AI
Gateway is OpenAI-compatible, so dao-ai swaps to `ChatOpenAI` whenever
the flag is set.

**Constraints.**
- **Chat completions only.** AI Gateway exposes `/chat/completions`; it
  does not implement the Responses API. Combining
  `ai_gateway: true` with `use_responses_api: true` is rejected by the
  Pydantic validator at load time.
- **Structured output requires `disable_streaming: true`.** AI Gateway
  returns `INVALID_PARAMETER_VALUE: Structured output is not currently
  supported with streaming.` when a `with_structured_output` call streams.
  Set `disable_streaming: true` on configs that use structured output.
- **Not for embedding endpoints.** AI Gateway is chat-only; embedding
  endpoints (`databricks-gte-large-en` etc.) continue to use the legacy
  path regardless of the flag.

**Auth.** Every credential mode supported on `InferenceEndpointModel`
(PAT, service principal / OAuth-M2M, `on_behalf_of_user`) flows through
the AI Gateway path. dao-ai uses a callable token provider so the
underlying `openai` SDK re-resolves the bearer token on every request
via `WorkspaceClient.config.authenticate()` — short-lived OBO and SP
tokens stay current automatically.

**Fallbacks.** A model with `ai_gateway: true` can fall back to a legacy
Model Serving endpoint (or vice versa). Both clients are LangChain
`Runnable`s, so `with_fallbacks(...)` composes the heterogeneous list
without further configuration:

```yaml
resources:
  models:
    resilient_llm: &resilient_llm
      name: databricks-claude-opus-4-6
      ai_gateway: true
      fallbacks:
      - databricks-claude-sonnet-4   # legacy Model Serving fallback
```

### AI Search endpoint capacity (`target_qps`)

**`vector_stores.<name>.endpoint.target_qps`** *(int, optional, Public Preview)* —
Target queries-per-second for the AI Search endpoint (formerly Vector Search).
**STANDARD endpoints only**; setting this on an `OPTIMIZED_STORAGE` endpoint raises
a config-validation error. Endpoint compute scales linearly with `target_qps`, so
cost scales linearly too. **Honored at endpoint-creation time only** — if the
endpoint already exists, this value is ignored (a debug log entry records the
configured value but no API call is made). To change capacity on a live endpoint,
use the Databricks UI, REST API, or SDK directly. See the
[Databricks AI Search QPS scaling docs](https://docs.databricks.com/aws/en/generative-ai/vector-search)
for the underlying capability.

### Skills (`resources.skills`)

A **skill** is a directory of Markdown content that teaches a deep-agent (or any agent under supervisor/swarm) how to perform a task. Skills follow the deepagents convention: a `SKILL.md` file with task instructions, optionally accompanied by `AGENTS.md` for memory plus arbitrary supporting files referenced from `SKILL.md`.

Skills are loaded by deepagents' `SkillsMiddleware`. dao-ai exposes them as a first-class config entity so they can be declared once under `resources.skills` and referenced by name from any agent, sub-agent, or deep_agent definition.

#### Object Model

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | yes | Unique identifier used by `SkillsMiddleware` |
| `path` | string \| VolumePathModel | yes | Skill source directory — see "Path forms" below |
| `description` | string | no | Human-readable description, surfaced in docs and traces |

#### Path forms

`path` accepts two shapes:

**1. Local (string)** — a relative path, resolved against the directory holding your config file. The directory is copied into every deploy bundle and into the model artifact via `code_paths`, layout preserved, so the same relative path resolves again on the target.

```yaml
resources:
  skills:
    research_skill:
      name: research
      description: Multi-source research with citations
      path: skills/research                  # relative to the config file
```

Keep it relative. An absolute path is passed through verbatim and will not exist on the deployment target — see "Path resolution" below.

**2. Volume-backed (VolumePathModel)** — a Unity Catalog volume reference. The skill is read directly from `/Volumes/<cat>/<schema>/<vol>/...` at runtime and the volume is wired as a deployment resource for permission grants. Use this when skills are governed centrally.

```yaml
resources:
  volumes:
    skills_volume: &skills_volume
      schema: *governance_schema
      name: dao_ai_skills

  skills:
    research_skill:
      name: research
      description: Multi-source research with citations
      path:
        volume: *skills_volume
        path: research                       # sub-path under the volume
```

A raw absolute string starting with `/Volumes/` is auto-promoted to a `VolumePathModel` by the pre-validator, so you can paste paths from the UC explorer verbatim:

```yaml
resources:
  skills:
    research_skill:
      name: research
      path: /Volumes/governance/skills/dao_ai_skills/research   # auto-promoted
```

#### Referencing skills

Skills can be attached at three levels:

| Where | Field | Accepts |
|---|---|---|
| `agents[].skills` | `list[SkillModel \| str]` | Strings resolved against `resources.skills`, or inline `SkillModel` entries |
| `orchestration.deep_agent.skills` | `list[SkillModel \| str]` | Same |
| `orchestration.deep_agent.subagents[].skills` | `list[SkillModel \| str]` | Same |

Skills attached at **any** of those levels force a filesystem-capable backend when `orchestration.deep_agent.backend` is left unset — including a config whose skills live *only* under `subagents[].skills`, with none on the deep_agent itself. deepagents' default `StateBackend` reads from graph state, so a skill on disk could never load through it whichever level declared it. `examples/13_orchestration/deep_agent_subagent_skills.yaml` is that sub-agent-only shape on its own.

One asymmetry to know: an entry in `app.agents` reused as an implicit sub-agent does **not** carry its `skills` into deepagents — only `subagents[].skills` does. Attach the skill to the `subagents` entry if a sub-agent needs it.

When the same skill is referenced from multiple places, declare it once under `resources.skills` and reuse the YAML anchor:

```yaml
resources:
  skills:
    research_skill: &research_skill
      name: research
      path: skills/research

agents:
  researcher:
    name: researcher
    skills: [*research_skill]                # OR ["research"] to look up by name

app:
  orchestration:
    deep_agent:
      skills: [*research_skill]
      subagents:
        - name: deep_research
          description: Deep multi-source research
          system_prompt: ...
          skills: [*research_skill]
```

#### Path resolution

A local `path` stays **relative everywhere** — in the config you write, in the config a bundle stages, and in the config baked into a registered model. It is turned into a real directory once, when the agent graph is built, by trying these anchors in order and taking the first that contains the skill:

1. The directory of the config file, when the config was loaded from a local path (`AppConfig.from_file`, `from_git`, `from_source`).
2. `$DAO_AI_PROJECT_ROOT`, if set.
3. The current working directory — this is the bundle root under Databricks Apps and the MCP server.
4. Each `sys.path` entry — this is what covers Model Serving, where mlflow prepends `<model_dir>/code`.

Nothing machine-specific is ever written back into the config, so building the graph and then registering the model (what `dao-ai agent create` and the deploy notebooks do in one process) cannot bake a local path into the artifact.

To confirm a deployed agent really loaded its skills, use one of these, in increasing order of strength:

- **Endpoint logs.** `Creating deep_agent graph` reports `skills_count`, `subagent_skills_count`, and `instruction_files_count`; `Resolved skill directory` names each skill and the directory it resolved to inside the container; `Defaulting deep_agent backend to FilesystemBackend` records the backend swap. An unresolvable skill is a `WARNING` naming every anchor tried.
- **MLflow traces.** The `SkillsMiddleware.before_agent` span carries `outputs.skills_metadata`, a list of `{name, description, path}` with the path as resolved *inside the container*, plus `skills_load_errors` when something failed. For a sub-agent's skill this span is nested under that sub-agent's span, which is also how you confirm delegation happened. An empty list is the silent skills-missing failure.
- **Asking the agent** to list its skills with a path for each. Weakest of the three: the middleware injects that list into the system prompt, so a plausible answer proves a path reached the prompt, not that the backend read the file.

#### Progressive disclosure needs a file-reading agent

Skills load lazily. Only each skill's `name` and `description` go into the system prompt; the body of `SKILL.md` is read on demand, and the injected instructions tell the model to call `read_file` on the listed path. A **deep agent** has that tool, so its skills work end to end. A plain `agents[].skills` entry does not add one: the agent can see that the skill exists and what it is for, but cannot read its instructions unless something else in the config supplies a file-reading tool. Attach detailed procedural skills to a deep agent, or keep the `description` self-sufficient for a plain one.

An **absolute** `path` bypasses resolution entirely and is used as given. That is correct for a Databricks FUSE mount — `/Volumes/...`, `/Workspace/...`, `/dbfs/...` — and wrong for anything under your home directory or a checkout, where the deployed agent will find nothing.

If a declared local skill cannot be found at **deploy** time, the deploy fails and names each unresolvable skill: `agent create`, `agent build`, `agent build --as-mcp`, the workflow bundler, and the direct Apps deploy (`agent up --mode apps`) all check. At **runtime** a missing directory is a `WARNING` in the endpoint or app log naming the source and every anchor tried, and the agent serves without that skill — a missing skill degrades the agent, it does not take the endpoint down.

**FUSE paths are exempt from the deploy-time check.** A path under `/Volumes`, `/Workspace`, or `/dbfs` is mounted inside Databricks compute and is normally absent from the machine running the deploy, so whether it exists locally says nothing about whether it will be there at runtime. Those paths are never flagged. The exemption matches per path segment, so a near miss like `/Volumesnotreally/skills` is still treated as a local path and still checked.

#### Deployment behaviour

- **Local skills** are staged by every path that ships a config — the Apps bundler (`agent build`), the MCP bundler (`agent build --as-mcp`), the workflow DAB (under `config/skills/...`, beside the staged config), and the direct Apps deploy, which uploads them beside the config in the app's workspace source — and ship with the model artifact via `code_paths`. No extra grants needed.
- **Volume-backed skills** are never copied. They emit deployment resources (via the underlying `VolumeModel`) so the app's service principal receives `READ_VOLUME` on the backing volume at deploy time, and are read from `/Volumes/...` at runtime.

#### Instruction files (`deep_agent.instruction_files`)

`orchestration.deep_agent.instruction_files` names `AGENTS.md`-style files whose contents are spliced into the system prompt at startup. They follow exactly the contract above — relative in the config, resolved against the same four anchors when the graph is built, staged by every bundler and by the direct Apps deploy, and shipped with the model artifact — with two differences worth knowing:

- An entry names a **file**, not a directory. `instructions/AGENTS.md` is right; `instructions/` is not, and a directory resolves to nothing.
- An entry that lives inside a skill directory (`skills/research/AGENTS.md`) is already carried by that skill's staging, so it is not copied twice.

```yaml
app:
  orchestration:
    deep_agent:
      instruction_files:
        - instructions/AGENTS.md           # relative to the config file
        - skills/research/AGENTS.md        # already shipped with the skill
```

Declaring `instruction_files` also forces a filesystem-capable backend when `backend` is left unset, for the same reason skills do: deepagents' default `StateBackend` reads from graph state, so no path could ever load. As with skills, an unresolvable entry fails the **deploy** by name, and a file missing at **runtime** is a `WARNING` naming every anchor tried while the agent serves without it.

### Chat UI (`enable_chat_proxy`)

Controls whether the deployed Databricks App includes the interactive chat UI
alongside the agent backend.

| Value | Behaviour |
|-------|-----------|
| `true` (default) | The app runs both a Python backend (port 8000) and a Node.js chat frontend (port 3000). The MLflow `AgentServer` proxies browser requests to the frontend. The chat UI is the Databricks [e2e-chatbot-app-next](https://github.com/databricks/app-templates/tree/main/e2e-chatbot-app-next) template, cloned and built automatically at app startup (the Apps runtime has Node.js pre-installed). |
| `false` | The app runs the Python backend only (`dao_ai.apps.server`). No chat UI. Useful for headless API endpoints or Model Serving deployments. |

---

## Parameters (Load-Time Substitution)

Configs can declare typed input parameters and reference them inline with `${param.NAME}` (or its alias `${var.NAME}`). They can also reference Databricks workspace context (host, current user) via `${workspace.*}` using the same convention as Databricks Asset Bundles. Substitution happens once at load time, **before** MLflow's `ModelConfig` parses the YAML, so one config can re-use across catalogs, schemas, environments, workshop modules, and users without duplicating files.

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
  genie_parent_path:
    description: Workspace folder for the Genie space
    default: "/Users/${workspace.current_user.userName}/genie"

schemas:
  workshop_schema:
    catalog_name: ${param.catalog}
    schema_name: ${param.schema}

app:
  name: dao_ws_${param.module_id}_orchestration
```

Inspect declared parameters and their resolved values with [`dao-ai parameters list`](cli-reference.md#inspect-declared-parameters).

### Reference syntax

Two prefixes are supported as interchangeable aliases:

- `${param.NAME}` - matches the `parameters:` block name (recommended).
- `${var.NAME}` - matches the Databricks Asset Bundle convention.

Both can appear in the same file and resolve against the same declaration. Inline defaults are also supported: `${param.NAME:-fallback}` / `${var.NAME:-fallback}`.

### Resolution precedence

Each reference is resolved in this order:

1. **CLI** `--param name=value` (alias `--var`), or `AppConfig.from_file(params={...})`
2. **Process env** - `NAME` upper-cased with `.` and `-` replaced by `_` (e.g. `${param.app.catalog-name}` reads from `APP_CATALOG_NAME`)
3. **Declared default** - the `default:` entry in the `parameters:` block
4. **Inline default** - `${param.NAME:-fallback}` on the reference itself
5. **Error** - raises `ConfigVariableError`

### Workspace variables

In addition to declared `${param.*}` / `${var.*}`, configs may reference Databricks workspace context using the Databricks Asset Bundles namespace:

| Reference | Resolves to | Example |
|---|---|---|
| `${workspace.host}` | Workspace URL, trailing slash stripped | `https://adb-1234.5.azuredatabricks.net` |
| `${workspace.current_user.userName}` | Full email address of the loading user | `nate.fleming@databricks.com` |
| `${workspace.current_user.short_name}` | Email prefix before `@` (dots intact, DABs convention) | `nate.fleming` |
| `${workspace.current_user.domain_friendly_name}` | Email domain after `@` | `databricks.com` |

Workspace references resolve **before** `${param.*}` / `${var.*}`, so they may appear inside a parameter's `default` (as in the `genie_parent_path` example above). The `WorkspaceClient` is built lazily — configs that don't reference any `${workspace.*}` value never trigger an auth call. When a config does reference one, the SCIM `me()` call is memoized across the three derived user paths for one network round-trip per load.

Authentication for the workspace lookup follows the standard Databricks SDK precedence (`DATABRICKS_HOST` / `DATABRICKS_TOKEN` env vars, `DATABRICKS_CONFIG_PROFILE`, or the `DEFAULT` profile in `~/.databrickscfg`). Failures surface as `WorkspaceVariableError` with the original cause attached. Unsupported paths (e.g. `${workspace.current_user.email}`) are rejected at load time with a list of allowed paths.

### Error handling

Three classes of error are caught at load time:

**Missing required** - a declared parameter with no `default` and no override:

```
Config parameter error in dao_ai.yaml:
  missing required: module_id.
  Pass with --param name=value or set the equivalent env var.
```

**Undeclared reference** - a `${param.NAME}` used in the YAML but not in the `parameters:` block (typo protection):

```
Config parameter error in dao_ai.yaml:
  undeclared ${param.NAME} / ${var.NAME} references: catlaog.
  Add them to the top-level parameters: block.
```

**Unsupported workspace path** - a `${workspace.*}` reference outside the supported set:

```
Unsupported ${workspace.*} reference(s) in dao_ai.yaml: current_user.email.
Supported: current_user.domain_friendly_name, current_user.short_name, current_user.userName, host.
```

### YAML quoting caveat

Substitution is text-level - the value is spliced into the YAML before parsing. If a value may contain YAML-special characters (`:` followed by a space, `#`, `[`, `{`, newlines, quotes), quote the reference:

```yaml
prompt: "${param.user_prompt}"   # safe regardless of value content
label: ${param.label}            # OK only for plain alphanumeric values
```

### Non-recursion

Substitution does not recurse. If a substituted value happens to contain `${param.x}` literally, it is preserved as-is and not re-resolved.

### Bundle behaviour

When `dao-ai agent build` writes the deployable Apps bundle, the emitted config YAML has every reference (both `${param.*}` and `${workspace.*}`) substituted to a literal value and the `parameters:` block dropped. The deployed app does not need the original `--param` flags or runtime workspace lookups.

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
| **Source of value** | `--param` (alias `--var`), env, declared default, inline `:-default`, or `${workspace.*}` | `env` / `scope`+`secret` / composite at runtime |
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
dao-ai workflow up -c dao_ai.yaml --param secret_scope=prod_dao_ai --param client_id_secret_key=PROD_SP_CLIENT_ID
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

## SQL Tool (`type: sql`)

Runs a **fixed** SQL statement — optionally with bound parameters — against a SQL
warehouse or a Lakebase / Postgres database. The statement is set at config time;
the LLM cannot author arbitrary SQL. Exactly one of `warehouse:` or `database:` is
required.

```yaml
tools:
  # Warehouse target — uses :name bind markers
  store_lookup:
    name: store_lookup
    function:
      type: sql
      warehouse: *shared_warehouse
      statement: |
        SELECT store_id, name, city FROM retail.ops.stores WHERE store_id = :store_id
      description: "Look up a store by id."
      params:
        - name: store_id
          type: int                     # string | int | float | bool
          description: "The store id."

  # Lakebase / Postgres target — uses %(name)s bind markers, mixed param sources
  category_inventory:
    name: category_inventory
    function:
      type: sql
      database: *retail_database
      statement: |
        SELECT product_name, on_hand FROM inventory
        WHERE store_num = %(store_num)s AND category = %(category)s
      description: "On-hand inventory for a category at the current store."
      params:
        - name: category                # LLM-supplied (source defaults to 'llm')
          type: string
          description: "Product category to filter by."
        - name: store_num               # bound from runtime Context
          source: context
          type: int
          # context_key: store_num      # defaults to the param name
```

**Backend fields (mutually exclusive, exactly one required)**

| Field | Meaning |
|---|---|
| `warehouse:` | `WarehouseModel` — statement runs via the Statement Execution API. Bind markers: `:name`. |
| `database:` | `DatabaseModel` — statement runs via psycopg on the shared Lakebase pool. Bind markers: `%(name)s`. |

**`statement:`** — the SQL to run. Author bind markers in the target backend's
native syntax (`:name` for warehouse, `%(name)s` for Lakebase). Values are bound
natively — never interpolated into the SQL string — so both backends are
injection-safe.

**`params:`** — optional list of `StatementParam`:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | string | — | Marker name as it appears in the statement. |
| `type` | `string`\|`int`\|`float`\|`bool` | `string` | Declared type; shapes the LLM-facing schema. |
| `source` | `llm`\|`context` | `llm` | `llm`: model supplies the value (appears in the tool schema). `context`: bound from runtime `Context`, hidden from the model. |
| `required` | bool | `true` | A missing required value returns an `Error:` string. |
| `default` | any | `null` | Fallback applied when the value is absent. |
| `description` | string | `null` | Shown to the LLM for `source: llm` params. |
| `context_key` | string | `null` | For `source: context`: the `Context` attribute to read (defaults to `name`). |

With no `params` (or only `context` params) the tool exposes **no** LLM-facing
arguments, matching the legacy zero-argument SQL tool.

**Auth (OBO)** — the workspace client is obtained per request via
`workspace_client_from(context)`, so warehouse OBO works on both Model Serving
(user credentials) and Databricks Apps (forwarded user token), in addition to
service principal, PAT, and ambient auth. For Lakebase, Model Serving OBO is
honored today; Databricks Apps OBO for Lakebase is not yet wired through the
shared connection pool.

**Governance** — because `type: sql` inherits the base tool model, it composes with
`human_in_the_loop:` and `audit:` (e.g. gate a mutating `UPDATE`/`DELETE` behind
human approval with a signed audit receipt). See
[`examples/99_complete_applications/hardware_store/hardware_store_lakebase.yaml`](https://github.com/natefleming/dao-ai/blob/main/examples/99_complete_applications/hardware_store/hardware_store_lakebase.yaml)
and [`examples/14_basic_tools/sql_tool_example.yaml`](https://github.com/natefleming/dao-ai/blob/main/examples/14_basic_tools/sql_tool_example.yaml).

## First-Class Agent Tools

dao-ai exposes three first-class function types for calling another agent as a tool. Each picks the target kind explicitly (where the workload lives), so the type discriminator matches what Agent Bricks Supervisor calls `app`, `serving_endpoint`, and `a2a`.

| `type:` | Target | Default wire shape | Discovery (when `api:` is unset) |
|---|---|---|---|
| `app` | Databricks App | OpenAI Responses | `GET <app_url>/agent/info` → reads `agent_api` |
| `serving_endpoint` | Model Serving endpoint (FMAPI or UC ResponsesAgent) | OpenAI Chat Completions | `WorkspaceClient.serving_endpoints.get(name).task` → maps `agent/v1/responses` to `responses`, `llm/v1/chat` to `completions` |
| `a2a` | External A2A agent (Vertex, Crew.ai, ADK, Databricks App with A2A) | Google A2A v0.3 | n/a |

### `type: app` — call a Databricks App

```yaml
resources:
  apps:
    supplier_app: &supplier_app
      name: dao-ai-supplier-app
      on_behalf_of_user: true

tools:
  ask_supplier:
    name: ask_supplier
    function:
      type: app
      app: *supplier_app
      api: responses | completions | null   # null (default) = lazy probe
      description: "Delegate supplier questions to the supplier app."
```

- `app:` is required and must reference a `DatabricksAppModel` (apps with the `mcp-` name prefix are rejected — use `type: mcp`).
- `api:` selects the OpenAI wire shape at the app's `/v1/responses` or `/v1/chat/completions` route.
- When `api:` is **unset**, the dispatcher lazily probes `<app_url>/agent/info` on **first invocation** and caches the result. Falls back to `"responses"` if the probe returns no signal.
- When `api:` is **set**, the probe never runs — explicit value wins, fully offline-safe.
- OBO is auto-derived from `app.on_behalf_of_user`.

### `type: serving_endpoint` — call a Model Serving endpoint

```yaml
tools:
  # FMAPI string sugar — discovery maps llm/v1/chat → completions
  ask_sonnet:
    name: ask_sonnet
    function:
      type: serving_endpoint
      endpoint: databricks-claude-sonnet-4

  # UC-registered ResponsesAgent — discovery maps agent/v1/responses → responses
  query_hardware_store:
    name: query_hardware_store
    function:
      type: serving_endpoint
      endpoint: hardware_store_dao

  # Full InferenceEndpointModel form (lets you set temperature, max_tokens, …)
  ask_sonnet_creative:
    name: ask_sonnet_creative
    function:
      type: serving_endpoint
      endpoint:
        name: databricks-claude-sonnet-4
        temperature: 0.9
        max_tokens: 1024
      api: completions   # explicit — skips discovery
```

- `endpoint:` accepts an endpoint **name string** (sugar; promoted to a minimal `InferenceEndpointModel`) **or** a full `InferenceEndpointModel` when you need `temperature`, `max_tokens`, `ai_gateway`, or `on_behalf_of_user`.
- `api:` defaults to lazy SDK probe via `serving_endpoints.get(name).task`. Falls back to `"completions"` when discovery returns no signal (preserves FMAPI behavior).

### Probe safety

- Both discovery probes are **lazy** (run on first tool invocation only) and **cached per tool instance**.
- Every failure mode (404, 401, 5xx, network error, non-JSON body, unknown future task value, SDK exception) falls back silently to the per-type default with a DEBUG log line.
- Config-load, Pydantic validation, `dao-ai validate`, and bundle packaging make **zero network calls**. dao-ai bundles can be built and deployed even when target apps/endpoints are not yet live.
- On first invocation each dispatcher logs one INFO line: `app_dispatcher resolved api='responses' (discovery) | app='…'` or `serving_endpoint_dispatcher resolved api='completions' (default) | endpoint='…'` — origin is `explicit`, `discovery`, or `default`.

### `type: a2a` — call an external A2A agent

See [`examples/99_complete_applications/procurement_supplier_a2a/README.md`](https://github.com/natefleming/dao-ai/blob/main/examples/99_complete_applications/procurement_supplier_a2a/README.md) for the full A2A protocol example. Available since v0.1.80.

### See Also

- [`examples/10_agent_integrations/app_first_class.yaml`](https://github.com/natefleming/dao-ai/blob/main/examples/10_agent_integrations/app_first_class.yaml)
- [`examples/10_agent_integrations/serving_endpoint_first_class.yaml`](https://github.com/natefleming/dao-ai/blob/main/examples/10_agent_integrations/serving_endpoint_first_class.yaml)
- [`examples/10_agent_integrations/README.md`](https://github.com/natefleming/dao-ai/blob/main/examples/10_agent_integrations/README.md) — routing matrix and migration notes
- [`examples/99_complete_applications/procurement_supplier_a2a/`](https://github.com/natefleming/dao-ai/tree/main/examples/99_complete_applications/procurement_supplier_a2a/) — end-to-end A2A example

---

## Genie tool descriptions

The description of a `type: genie` tool is what a supervisor reads to decide
whether to route a question to *that* Genie space. It is resolved in this order:

1. the tool's own `description`, when set;
2. the room's `description` — either declared under `resources.genie_rooms` or
   back-filled from the live Genie space by discovery (see below), which never
   overwrites a value you declared;
3. a generic fallback naming the space (`"…chat with tabular data about <space name>"`),
   or naming nothing if the space has no name.

**Why this matters.** With neither (1) nor (2), every description-less Genie tool
in a config used to advertise the same subject-free text, so a supervisor with two
Genie tools had no signal to route on and picked more or less at random. Giving
each space a `description` — or letting it be discovered — is the fix.

**Example questions.** Set `include_example_questions: true` and, when the room has
`sample_questions` (else the `question` of each `example_sqls` pair), up to ten are
appended to the tool description as few-shot routing hints:

```
Example questions this tool can answer:
- How many stores are in each state?
- Which SKUs led revenue last quarter?
```

SQL bodies are deliberately never included: the description is read by the
calling LLM for routing, Genie already holds the SQL on its own space, and the
bodies would cost hundreds of tokens on every LLM call for no routing gain.

**The flag is opt-in in every case**, and independent of how the base text was
chosen — the tool description is prompt surface, so nothing lands there that you
did not ask for, and up to ten questions per Genie tool do not ride in every LLM
call by default:

| `include_example_questions` | base text | example block |
|---|---|---|
| `false` (default) | your `description`, else room / discovered / fallback | no |
| `true` | your `description`, else room / discovered / fallback | **yes**, if questions exist |

Setting it alongside your own `description` is the combination the flag exists
for: your wording *and* the auto-appended questions. If the room has example
questions and the flag is off, dao-ai logs an `INFO` line naming the space, so
the available signal is discoverable when you are looking for it.

```yaml
tools:
  store_sales_genie:
    name: store_sales_genie
    function:
      type: genie
      genie_room: *retail_genie
      description: Store sales, inventory and returns for North America.
      include_example_questions: true    # keep my wording, append the questions too
```

Intent survives the deploy bake because `false` and `true` are both ordinary
values. Model Serving dumps the config with `exclude_none=True` into the logged
`model_config` and Apps ships rendered YAML text, so `model_fields_set` — which is
how a boolean would tell "unset" from "explicitly set" — does not round-trip; with
an unconditional default there is nothing to distinguish, and the tool description
comes out byte-identical in local, Apps, and Model Serving.

**`preserve_question`** (default `false`) is a separate knob on the same tool. It
constrains the question sent *to* Genie, not Genie's answer: when Genie is a tool,
the calling LLM writes the `question` argument, and stronger models rephrase or
decompose it — quietly dropping a refining qualifier and changing the scope Genie
answers. Setting it stamps a preserve-exact instruction onto both surfaces the
model reads (the tool description and the `question` argument annotation), while
still allowing a genuinely multi-tool request to be split across tools.

```yaml
      preserve_question: true            # send the user's wording through unedited
```

**Discovery, and why it is identical in all three runtimes.** A room declared
with only a `space_id` — the common case, since Genie titles are not unique —
carries no local text at all. `GenieRoomModel.ensure_resolved()` back-fills
`name`, `description`, and `sample_questions` from the live space. `name` and
`description` feed the tool description directly; `sample_questions` reach it only
with `include_example_questions: true`, and are separately what a room writes back
into its space when it *provisions* one. Each runtime gets the same resolved
values, by a different route:

| Runtime | How the discovered values reach the container |
|---|---|
| Local | `ensure_resolved()` runs during `AppConfig.from_file(initialize=True)`. |
| Model Serving | Discovery runs at deploy time and the resolved config is baked into the logged MLflow `model_config`; the container needs **no** Genie call at model load. |
| Databricks Apps | Discovery runs at **bundle** time and the three fields are written into the config YAML the bundle ships. |

Apps needs its own bake because the bundle carries the rendered config as *text*,
not as a resolved object — nothing `ensure_resolved()` produced at deploy time
would otherwise reach the container. So `dao-ai agent up -m apps` looks each
distinct `space_id` up once with the deploying identity's credentials and writes
the result into the emitted YAML. Fields you declared are never overwritten, a
`${var.…}` space id (one provisioned later in the same run) is skipped, and a
space that cannot be read leaves its room exactly as written — the deploy is
never blocked by discovery.

**What each Genie permission level can discover.** The two halves of discovery
need different grants, which is why the deploy-time bake matters even though the
container can read *something*:

| Grant | `title` / `description` | `sample_questions` |
|---|---|---|
| `CAN_EDIT` | yes | yes (needs the serialized space) |
| `CAN_RUN` | yes | **no** |

A deployed identity usually holds `CAN_RUN` — that is all an app's Genie resource
confers. `_get_space_details()` asks for the serialized space (the only source of
sample questions) and retries without it when that read is denied, so a
`CAN_RUN`-only identity still recovers the space's own description instead of
losing everything to one failed call. The questions still come from the bake — the
only chance to capture them, which is what keeps an opted-in tool's routing hints
and a provisioning room's declared questions from vanishing once deployed.

Tool construction itself never calls the Genie API; if discovery was unavailable,
the fallback text is used rather than an error.

Locally-declared `sample_questions` and `example_sqls` always win over discovered
ones. Only the question fields are hydrated — the other provisioning inputs
(`table_sources`, `function_sources`, …) are left alone, unlike
`GenieRoomModel.refresh()`.

---

## Genie Agent as a model

The Databricks **Genie Agent Mode API** (`POST /api/2.0/genie/agents/{agent_id}/responses`,
Beta) can back an agent's **reasoning model** instead of being wrapped as a tool.
A `GenieAgentModel` streams Genie's output (SQL + result table + narrative) as
`AIMessageChunk`s to the outer response stream, so an agent with `tools: []`
becomes a "Genie specialist" a supervisor can route to like any other sub-agent.

**Tool vs. model.** `type: genie` (the tool) is atomic — a LangGraph tool node
returns one `ToolMessage` after the whole stream completes, so Genie output can't
stream to the end user through the agent's response. A `GenieAgentModel` streams
natively (`stream_mode="messages"`). Both point at the same Genie space (the
32-char `agent_id` is the renamed `space_id`); keep both while the Agent Mode API
is Beta.

```yaml
resources:
  genie_rooms:
    retail_genie: &retail_genie          # register the room HERE (required)
      agent_id: 01f0...                  # alias of space_id
      on_behalf_of_user: true            # optional: run Genie as the caller (OBO)

agents:
  genie_specialist:
    name: genie_specialist
    description: Answers factual questions from the warehouse via Genie.
    model: *retail_genie                 # terse: bare room → GenieAgentModel (timeout 300s)
    # model:                             # explicit wrapper form (custom knobs):
    #   genie_room: *retail_genie
    #   timeout_seconds: 600
    tools: []                            # Genie IS the brain; no tools
    prompt: |
      Relay Genie's answer, preserving SQL, tables, and citations.
```

**Assignment forms.** `model:` accepts either a bare Genie room (a
`genie_rooms` anchor, or a dict carrying any room-only key — `agent_id`/`space_id`
as well as the provisioning fields `warehouse`, `table_sources`,
`text_instructions`, `sample_questions`, … — auto-wrapped into a
`GenieAgentModel` with default `timeout_seconds`) or the explicit
`{genie_room: <room>, timeout_seconds: <int>}` wrapper. A `{name: <endpoint>}`
config carries no room-only key and stays an `InferenceEndpointModel`. A
`GenieAgentModel` is **not** a serving endpoint — do not place it under
`resources.models`.

**One shape stays ambiguous: a name-only room.** `{name: X}` is valid for *both*
union members, so it cannot be coerced by shape and resolves to an
`InferenceEndpointModel`. If that name matches a room registered under
`resources.genie_rooms`, config-load fails with an explanatory error rather than
letting the agent point at a serving endpoint that does not exist. Use the
explicit `model: {genie_room: *your_room_anchor}` wrapper, or give the room an
`agent_id`/`space_id` so it can be assigned bare. Note that a room used as a
model resolves its id **statically** (`space_id`/`agent_id`, else
`DATABRICKS_GENIE_SPACE_ID`) — a name is resolved to an id during
`AppConfig.initialize()`, which is why the room must be registered and shared by
anchor.

**Room registration (required).** A `GenieAgentModel` is a wrapper, not a
deploy resource. Its `genie_room` **must** be registered under
`resources.genie_rooms` so the bundle emits the `genie-space` grant and, when
`on_behalf_of_user: true`, the `dashboards.genie` `user_api_scope`. Config-load
fails with a clear error if an agent's Genie room isn't registered.

**Multi-turn.** The Genie server owns conversation history keyed by a
Genie-issued `conversation_id` (independent of the LangGraph `thread_id`).
`GenieAgentMiddleware` caches it in `session.genie.spaces[agent_id]` — the same
channel `type: genie` uses — reading the prior id before each turn and
persisting the newly-issued one after (via the `merge_session` reducer).

**OBO.** Set `on_behalf_of_user: true` on the **room** (single source of truth).
`GenieAgentMiddleware` builds the per-request client via
`workspace_client_from(context)` — the forwarded `x-forwarded-access-token` on
Databricks Apps, or `ModelServingUserCredentials` on Model Serving. When OBO is
set, also set the room's `workspace_host` unless `DATABRICKS_HOST` is in the
environment (it is on Apps/MS deploys).

### See Also

- [`examples/10_agent_integrations/genie_agent_model.yaml`](https://github.com/natefleming/dao-ai/blob/main/examples/10_agent_integrations/genie_agent_model.yaml)
- [`examples/10_agent_integrations/genie_agent_model_obo.yaml`](https://github.com/natefleming/dao-ai/blob/main/examples/10_agent_integrations/genie_agent_model_obo.yaml) — OBO + deployable App

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

- Full examples: [`examples/02_mcp/filtered_mcp.yaml`](https://github.com/natefleming/dao-ai/blob/main/examples/02_mcp/filtered_mcp.yaml)
- MCP documentation: [`examples/02_mcp/README.md`](https://github.com/natefleming/dao-ai/blob/main/examples/02_mcp/README.md#pattern-4-filtered-mcp-tool-selection)

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
      name: "databricks-gpt-5-4-mini"
      temperature: 0.1
      max_tokens: 4096
    tools: []
```

### See Also

- Full example: [`examples/12_middleware/deepagents_middleware.yaml`](https://github.com/natefleming/dao-ai/blob/main/examples/12_middleware/deepagents_middleware.yaml)
- Middleware examples: [`examples/12_middleware/README.md`](https://github.com/natefleming/dao-ai/blob/main/examples/12_middleware/README.md)

---

## Navigation

- [← Previous: Key Capabilities](key-capabilities.md)
- [↑ Back to Documentation Index](https://github.com/natefleming/dao-ai/blob/main/README.md#-documentation)
- [Next: Examples →](examples.md)

