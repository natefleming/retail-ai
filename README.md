# DAO: Declarative Agent Orchestration

**Production-grade AI agents defined in YAML, powered by LangGraph, deployed on Databricks.**

DAO is an **infrastructure-as-code framework** for building, deploying, and managing multi-agent AI systems. Instead of writing boilerplate Python code to wire up agents, tools, and orchestration, you define everything declaratively in YAML configuration files.

```yaml
# Define an agent in 10 lines of YAML
agents:
  product_expert:
    name: product_expert
    model: *claude_sonnet
    tools:
      - *vector_search_tool
      - *genie_tool
    prompt: |
      You are a product expert. Answer questions about inventory and pricing.
```

---

## Why DAO?

### For Newcomers to AI Agents

**What is an AI Agent?**
An AI agent is more than a chatbot. While a chatbot just responds to messages, an *agent* can:
- **Reason** about what steps to take
- **Use tools** (databases, APIs, search) to gather information
- **Make decisions** about which actions to perform
- **Coordinate** with other agents on complex tasks

**What is Databricks?**
Databricks is a unified data and AI platform that provides:
- **Unity Catalog**: Governed data access and permissions
- **Model Serving**: Deploy ML models as APIs
- **Vector Search**: Semantic similarity search for RAG
- **Genie**: Natural language to SQL for business users
- **MLflow**: Model tracking, versioning, and deployment

DAO brings these capabilities together into a cohesive agent framework.

---

## DAO vs. Databricks Agent Bricks

Databricks offers two approaches to building AI agents. Understanding when to use each is crucial:

| Aspect | **DAO** (This Framework) | **Databricks Agent Bricks** |
|--------|--------------------------|----------------------------|
| **Configuration** | YAML files (infrastructure-as-code) | GUI-based AI Playground |
| **Target Users** | ML Engineers, Platform Teams | Citizen Developers, Analysts |
| **Customization** | Full control over every component | Guided templates and wizards |
| **Version Control** | Git-native, full CI/CD support | Limited versioning |
| **Orchestration** | Multiple patterns (Supervisor, Swarm) | Single-agent or simple routing |
| **Tool Types** | Python, Factory, UC, MCP, Custom | Unity Catalog functions |
| **Caching** | LRU + Semantic (pg_vector) | None built-in |
| **Memory** | PostgreSQL, In-Memory, Custom | Basic checkpointing |
| **Deployment** | Databricks Asset Bundles, MLflow | One-click deployment |
| **Best For** | Production multi-agent systems | Rapid prototyping, POCs |

### When to Use DAO

✅ You need **multiple specialized agents** working together  
✅ You want **version-controlled, reviewable** agent configurations  
✅ You require **custom tools** beyond Unity Catalog functions  
✅ You need **advanced caching** for cost optimization  
✅ You're building a **production system** with CI/CD  
✅ You need **human-in-the-loop** approval workflows  

### When to Use Agent Bricks

✅ You're **prototyping** a new agent idea quickly  
✅ You prefer a **no-code/low-code** approach  
✅ Your use case fits **standard templates** (extraction, Q&A)  
✅ You want **automated optimization** without manual tuning  
✅ You have a **single agent** with simple tool needs  

---

## Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            YAML Configuration                               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐   │
│  │ Schemas │  │ Resources│  │  Tools  │  │ Agents  │  │  Orchestration  │   │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DAO Framework (Python)                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    │
│  │   Config    │  │    Graph    │  │    Nodes    │  │   Tool Factory  │    │
│  │  Loader     │  │   Builder   │  │   Factory   │  │                 │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LangGraph Runtime                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Compiled State Graph                             │   │
│  │   ┌─────────┐    ┌─────────────┐    ┌─────────────────────────┐    │   │
│  │   │ Message │───▶│ Supervisor/ │───▶│    Specialized Agents    │    │   │
│  │   │  Hook   │    │   Swarm     │    │  (Product, Orders, DIY)  │    │   │
│  │   └─────────┘    └─────────────┘    └─────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Databricks Platform                                  │
│  ┌─────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────┐  ┌─────────┐ │
│  │  Model  │  │    Unity    │  │   Vector    │  │  Genie   │  │ MLflow  │ │
│  │ Serving │  │   Catalog   │  │   Search    │  │  Spaces  │  │         │ │
│  └─────────┘  └─────────────┘  └─────────────┘  └──────────┘  └─────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Orchestration Patterns

DAO supports two orchestration patterns for multi-agent coordination:

#### 1. Supervisor Pattern
A central supervisor agent routes queries to specialized agents based on the request content.

```yaml
orchestration:
  supervisor:
    model: *router_llm
    prompt: |
      Route queries to the appropriate specialist agent.
```

```
                    ┌─────────────┐
                    │  Supervisor │
                    └──────┬──────┘
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │   Product   │ │   Orders    │ │     DIY     │
    │    Agent    │ │    Agent    │ │    Agent    │
    └─────────────┘ └─────────────┘ └─────────────┘
```

#### 2. Swarm Pattern
Agents can hand off to each other directly, enabling more fluid multi-step workflows.

```yaml
orchestration:
  swarm:
    model: *default_llm
    default_agent: *general_agent
    handoffs:
      product_agent: [orders_agent, diy_agent]
      orders_agent: [product_agent]
```

```
    ┌─────────────┐     handoff     ┌─────────────┐
    │   Product   │◄───────────────▶│   Orders    │
    │    Agent    │                 │    Agent    │
    └──────┬──────┘                 └──────┬──────┘
           │          handoff              │
           └──────────────┬────────────────┘
                          ▼
                   ┌─────────────┐
                   │     DIY     │
                   │    Agent    │
                   └─────────────┘
```

---

## Key Capabilities

### 1. Multi-Tool Support

DAO supports five types of tools, each suited for different use cases:

| Tool Type | Use Case | Example |
|-----------|----------|---------|
| **Python** | Custom business logic | `dao_ai.tools.current_time_tool` |
| **Factory** | Complex initialization with config | `create_vector_search_tool(retriever=...)` |
| **Unity Catalog** | Governed SQL functions | `catalog.schema.find_product_by_sku` |
| **MCP** | External services via Model Context Protocol | GitHub, Slack, custom APIs |
| **Agent Endpoint** | Call other deployed agents as tools | Chaining agent systems |

```yaml
tools:
  # Python function - direct import
  time_tool:
       function:
         type: python
      name: dao_ai.tools.current_time_tool

  # Factory - initialized with config
  search_tool:
       function:
         type: factory
      name: dao_ai.tools.create_vector_search_tool
         args:
           retriever: *products_retriever

  # Unity Catalog - governed SQL function
  sku_lookup:
       function:
         type: unity_catalog
         name: find_product_by_sku
         schema: *retail_schema

  # MCP - external service integration
  github_mcp:
    function:
      type: mcp
      transport: streamable_http
      connection: *github_connection
```

### 2. Advanced Caching (Genie Queries)

DAO provides **two-tier caching** for Genie natural language queries, dramatically reducing costs and latency. Unlike simple response caching, DAO caches the **generated SQL** and re-executes it against your warehouse—ensuring you always get **fresh data** while avoiding the cost of repeated Genie API calls.

```yaml
genie_tool:
  function:
    type: factory
    name: dao_ai.tools.create_genie_tool
    args:
      genie_room: *retail_genie_room
      
      # L1: Fast O(1) exact match lookup
      lru_cache_parameters:
        warehouse: *warehouse
        capacity: 100                    # Max cached queries
        time_to_live_seconds: 86400      # 1 day (use -1 for never expire)

      # L2: Semantic similarity search via pg_vector
      semantic_cache_parameters:
        database: *postgres_db
        warehouse: *warehouse
        embedding_model: *embedding_model
        similarity_threshold: 0.85       # 0.0-1.0, higher = stricter matching
        time_to_live_seconds: 86400      # 1 day (use -1 for never expire)
```

#### Cache Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Two-Tier Cache Flow                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Question: "What products are low on stock?"                               │
│                          │                                                   │
│                          ▼                                                   │
│   ┌──────────────────────────────────────┐                                  │
│   │     L1: LRU Cache (In-Memory)        │  ◄── O(1) exact string match     │
│   │     • Capacity: 100 entries          │      Fastest lookup              │
│   │     • Hash-based lookup              │                                  │
│   └──────────────────────────────────────┘                                  │
│              │ Miss                                                          │
│              ▼                                                               │
│   ┌──────────────────────────────────────┐                                  │
│   │  L2: Semantic Cache (PostgreSQL)     │  ◄── Vector similarity search    │
│   │     • pg_vector embeddings           │      Catches rephrased questions │
│   │     • L2 distance similarity         │                                  │
│   │     • Partitioned by Genie space ID  │                                  │
│   └──────────────────────────────────────┘                                  │
│              │ Miss                                                          │
│              ▼                                                               │
│   ┌──────────────────────────────────────┐                                  │
│   │       Genie API                       │  ◄── Natural language to SQL    │
│   │       (Expensive call)                │                                  │
│   └──────────────────────────────────────┘                                  │
│              │                                                               │
│              ▼                                                               │
│   ┌──────────────────────────────────────┐                                  │
│   │    Execute SQL via Warehouse         │  ◄── Always fresh data!         │
│   └──────────────────────────────────────┘                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### LRU Cache (L1)

The **LRU (Least Recently Used) Cache** provides instant lookups for exact question matches:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `capacity` | 100 | Maximum number of cached queries |
| `time_to_live_seconds` | 86400 | Cache entry lifetime (-1 = never expire) |
| `warehouse` | Required | Databricks warehouse for SQL execution |

**Best for:** Repeated exact queries, chatbot interactions, dashboard refreshes

#### Semantic Cache (L2)

The **Semantic Cache** uses PostgreSQL with pg_vector to find similar questions even when worded differently:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `similarity_threshold` | 0.85 | Minimum similarity for cache hit (0.0-1.0) |
| `time_to_live_seconds` | 86400 | Cache entry lifetime (-1 = never expire) |
| `embedding_model` | Required | Model for generating question embeddings |
| `database` | Required | PostgreSQL with pg_vector extension |
| `table_name` | `genie_semantic_cache` | Table name for cache storage |

**Best for:** Catching rephrased questions like:
- "What's our inventory status?" ≈ "Show me stock levels"
- "Top selling products this month" ≈ "Best sellers in December"

#### Cache Behavior

1. **SQL Caching, Not Results**: The cache stores the *generated SQL query*, not the query results. On a cache hit, the SQL is re-executed against your warehouse, ensuring **data freshness**.

2. **Refresh on Hit**: When a semantic cache entry is found but expired:
   - The expired entry is deleted
   - A cache miss is returned
   - Genie generates fresh SQL
   - The new SQL is cached

3. **Multi-Instance Aware**: Each LRU cache is per-instance (in Model Serving, each replica has its own). The semantic cache is shared across all instances via PostgreSQL.

4. **Space ID Partitioning**: Cache entries are isolated per Genie space, preventing cross-space cache pollution.

### 3. Vector Search Reranking

DAO supports **two-stage retrieval** with FlashRank reranking to improve search relevance without external API calls:

```yaml
retrievers:
  products_retriever: &products_retriever
    vector_store: *products_vector_store
    columns: [product_id, name, description, price]
    search_parameters:
      num_results: 50        # Retrieve more candidates
      query_type: ANN
    rerank:
      model: ms-marco-MiniLM-L-12-v2   # Local cross-encoder model
      top_n: 5                          # Return top 5 after reranking
```

#### How It Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Two-Stage Retrieval Flow                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Query: "heavy duty outdoor extension cord"                                │
│                          │                                                   │
│                          ▼                                                   │
│   ┌──────────────────────────────────────┐                                  │
│   │   Stage 1: Vector Similarity Search  │  ◄── Fast, approximate matching  │
│   │   • Returns 50 candidates            │      Uses embedding similarity   │
│   │   • Milliseconds latency             │                                  │
│   └──────────────────────────────────────┘                                  │
│              │                                                               │
│              ▼ 50 documents                                                  │
│   ┌──────────────────────────────────────┐                                  │
│   │     Stage 2: Cross-Encoder Rerank    │  ◄── Precise relevance scoring   │
│   │     • FlashRank (local, no API)      │      Query-document interaction  │
│   │     • Returns top 5 most relevant    │                                  │
│   └──────────────────────────────────────┘                                  │
│              │                                                               │
│              ▼ 5 documents (reordered by relevance)                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Why Reranking?

| Approach | Pros | Cons |
|----------|------|------|
| **Vector Search Only** | Fast, scalable | Embedding similarity ≠ relevance |
| **Reranking** | More accurate relevance | Slightly higher latency |
| **Both (Two-Stage)** | Best of both worlds | Optimal quality/speed tradeoff |

Vector embeddings capture semantic similarity but may rank loosely related documents highly. Cross-encoder reranking evaluates query-document pairs directly, dramatically improving result quality for the final user.

#### Available Models

| Model | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| `ms-marco-TinyBERT-L-2-v2` | ⚡⚡⚡ Fastest | Good | High-throughput, latency-sensitive |
| `ms-marco-MiniLM-L-6-v2` | ⚡⚡ Fast | Better | Balanced performance |
| `ms-marco-MiniLM-L-12-v2` | ⚡ Moderate | Best | Default, recommended |
| `rank-T5-flan` | Slower | Excellent | Maximum accuracy |

#### Configuration Options

```yaml
rerank:
  model: ms-marco-MiniLM-L-12-v2    # FlashRank model name
  top_n: 10                          # Documents to return (default: all)
  cache_dir: /tmp/flashrank_cache    # Model weights cache location
  columns: [description, name]       # Columns for Databricks Reranker (optional)
```

**Note:** Model weights are downloaded automatically on first use (~20MB for MiniLM-L-12-v2).

### 4. Human-in-the-Loop Approvals

Add approval gates to sensitive tool calls without changing tool code:

```yaml
tools:
  dangerous_operation:
    function:
      type: python
      name: my_package.dangerous_function
      human_in_the_loop:
        review_prompt: "This operation will modify production data. Approve?"
```

### 5. Memory & State Persistence

Configure conversation memory with in-memory or PostgreSQL backends:

   ```yaml
memory:
  checkpointer:
    name: conversation_checkpointer
    type: postgres
    database: *postgres_db
  
  store:
    name: user_preferences_store
    type: postgres
    database: *postgres_db
    embedding_model: *embedding_model
```

### 6. Hook System

Inject custom logic at key points in the agent lifecycle:

```yaml
app:
  # Run on startup
  initialization_hooks:
    - my_package.hooks.setup_connections
    - my_package.hooks.warmup_caches

  # Run on every message
  message_hooks:
    - dao_ai.hooks.require_user_id_hook
    - my_package.hooks.log_request

  # Run on shutdown
  shutdown_hooks:
    - my_package.hooks.cleanup_resources

agents:
  my_agent:
    # Run before/after agent execution
    pre_agent_hook: my_package.hooks.enrich_context
    post_agent_hook: my_package.hooks.collect_metrics
```

### 7. MLflow Prompt Registry Integration

Store and version prompts externally, enabling prompt engineers to iterate without code changes:

```yaml
prompts:
  product_expert_prompt:
    schema: *retail_schema
    name: product_expert_prompt
    alias: production  # or version: 3
    default_template: |
      You are a product expert...
    tags:
      team: retail
      environment: production

agents:
  product_expert:
    prompt: *product_expert_prompt  # Loaded from MLflow registry
```

### 8. Automated Prompt Optimization

Use DSPy-style optimization to automatically improve prompts:

```yaml
optimizations:
  prompt_optimizations:
    optimize_diy_prompt:
      prompt: *diy_prompt
      agent: *diy_agent
      dataset: *training_dataset
      reflection_model: "openai:/gpt-4"
      num_candidates: 5
```

---

## Quick Start

### Prerequisites

- Python 3.12+
- Databricks workspace with:
  - Unity Catalog access
  - Model Serving enabled
  - (Optional) Vector Search, Genie

### Installation

```bash
# Clone and setup
git clone <repo-url>
cd dao-ai

# Create virtual environment
uv venv
source .venv/bin/activate

# Install
make install
```

### Your First Agent

1. **Create a minimal configuration** (`config/my_agent.yaml`):

```yaml
schemas:
  my_schema: &my_schema
    catalog_name: my_catalog
    schema_name: my_schema

resources:
  llms:
    default_llm: &default_llm
      name: databricks-meta-llama-3-3-70b-instruct

agents:
  assistant: &assistant
    name: assistant
    model: *default_llm
    prompt: |
      You are a helpful assistant.

app:
  name: my_first_agent
  registered_model:
    schema: *my_schema
    name: my_first_agent
  agents:
    - *assistant
  orchestration:
    swarm:
      model: *default_llm
```

2. **Validate your configuration**:

```bash
dao-ai validate -c config/my_agent.yaml
```

3. **Visualize the agent graph**:

```bash
dao-ai graph -c config/my_agent.yaml -o my_agent.png
```

4. **Deploy to Databricks**:

```python
from dao_ai.config import AppConfig

config = AppConfig.from_file("config/my_agent.yaml")
config.create_agent()   # Package as MLflow model
config.deploy_agent()   # Deploy to serving endpoint
```

Or via CLI:
```bash
dao-ai bundle --deploy --run -c config/my_agent.yaml
```

### Interact with Your Agent

```python
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")
response = client.predict(
    endpoint="my_first_agent",
    inputs={
        "messages": [{"role": "user", "content": "Hello!"}],
        "configurable": {"thread_id": "1", "user_id": "demo_user"}
    }
)
print(response["message"]["content"])
```

---

## Configuration Reference

### Full Configuration Structure

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
    prompt: string | *prompt_ref
    handoff_prompt: string      # For swarm routing
    pre_agent_hook: string      # Python function path
    post_agent_hook: string

# Prompt definitions (MLflow registry)
prompts:
  prompt_name: &prompt_name
    schema: *my_schema
    name: string
    alias: string | null        # e.g., "production"
    version: int | null
    default_template: string
    tags: {}

# Memory configuration
memory: &memory
  checkpointer:
    name: string
    type: memory | postgres
    database: *postgres_db
  store:
    name: string
    type: memory | postgres
    database: *postgres_db
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
  message_hooks: [string]
  shutdown_hooks: [string]
  
  permissions:
    - principals: [users]
      entitlements: [CAN_QUERY]
  
  environment_vars:
    KEY: "{{secrets/scope/secret}}"
```

---

## Example Configurations

The `config/examples/` directory contains ready-to-use configurations:

| Example | Description |
|---------|-------------|
| `minimal.yaml` | Simplest possible agent configuration |
| `genie.yaml` | Natural language to SQL with Genie |
| `genie_with_lru_cache.yaml` | Genie with LRU caching |
| `genie_with_semantic_cache.yaml` | Genie with two-tier caching |
| `human_in_the_loop.yaml` | Tool approval workflows |
| `mcp.yaml` | External service integration via MCP |
| `prompt_optimization.yaml` | Automated prompt tuning |
| `vector_search_with_reranking.yaml` | RAG with reranking |
| `deep_research.yaml` | Multi-step research agent |
| `slack.yaml` | Slack integration |

---

## CLI Reference

```bash
# Validate configuration
dao-ai validate -c config/my_config.yaml

# Generate JSON schema for IDE support
dao-ai schema > schemas/model_config_schema.json

# Visualize agent workflow
dao-ai graph -c config/my_config.yaml -o workflow.png

# Deploy with Databricks Asset Bundles
dao-ai bundle --deploy --run -c config/my_config.yaml --profile DEFAULT

# Verbose output (-v through -vvvv)
dao-ai -vvvv validate -c config/my_config.yaml
```

---

## Python API

```python
from dao_ai.config import AppConfig

# Load configuration
config = AppConfig.from_file("config/my_config.yaml")

# Access components
agents = config.find_agents()
tools = config.find_tools()
vector_stores = config.resources.vector_stores

# Create infrastructure
for name, vs in vector_stores.items():
    vs.create()

# Package and deploy
config.create_agent(
    additional_pip_reqs=["custom-package==1.0.0"],
    additional_code_paths=["./my_modules"]
)
config.deploy_agent()

# Visualize
config.display_graph()
config.save_image("docs/architecture.png")
```

---

## Project Structure

```
dao-ai/
├── src/dao_ai/
│   ├── config.py          # Pydantic configuration models
│   ├── graph.py           # LangGraph workflow builder
│   ├── nodes.py           # Agent node factories
│   ├── state.py           # State management
│   ├── tools/             # Tool implementations
│   │   ├── genie.py       # Genie tool with caching
│   │   ├── mcp.py         # MCP integrations
│   │   ├── vector_search.py
│   │   └── ...
│   ├── genie/
│   │   └── cache/         # LRU and Semantic cache
│   ├── memory/            # Checkpointer and store
│   └── hooks/             # Lifecycle hooks
├── config/
│   ├── examples/          # Example configurations
│   └── hardware_store/    # Reference implementation
├── tests/                 # Test suite
└── schemas/               # JSON schemas for validation
```

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Format code: `make format`
6. Submit a pull request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
