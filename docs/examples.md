# Example Configurations

The `examples/` directory contains ready-to-use configurations organized in a **numbered, progressive learning path**. Each directory builds upon the previous, guiding you from basic concepts to production-ready applications.

## 📚 Learning Path

The examples follow a natural progression:

```
01_getting_started → 02_mcp → 03_reranking → 04_genie → 05_memory 
    → 06_on_behalf_of_user → 07_human_in_the_loop → 08_guardrails → 09_structured_output
    → 10_agent_integrations → 11_prompt_engineering → 12_middleware → 13_orchestration
    → 14_basic_tools → 15_complete_applications → 16_instructed_retriever
    → 17_parallel_tools → 18_visualization → 19_background_agents
    → 20_a2a_protocol → 21_lakebase_search
```

Start at `01_getting_started` if you're new, or jump directly to the category that matches your needs.

## Quick Reference

### 🆕 New to DAO AI?
**Start here:**
- [`01_getting_started/minimal.yaml`](../examples/01_getting_started/minimal.yaml) - Simplest possible agent
- [`04_genie/genie_basic.yaml`](../examples/04_genie/genie_basic.yaml) - Natural language to SQL

### 🔧 Need Specific Tools?
**Explore:**
- [`02_mcp/`](../examples/02_mcp/) - Slack, JIRA, MCP integrations
- [`10_agent_integrations/`](../examples/10_agent_integrations/) - Agent Bricks, Kasal, external agent platforms
- [`14_basic_tools/`](../examples/14_basic_tools/) - SQL execution and basic tool patterns

### ⚡ Optimizing Performance?
**Check out:**
- [`04_genie/`](../examples/04_genie/) - LRU and semantic caching strategies

### 💾 Managing State?
**See:**
- [`05_memory/`](../examples/05_memory/) - Conversation history and persistence

### 🛡️ Production Ready?
**Essential patterns:**
- [`06_on_behalf_of_user/`](../examples/06_on_behalf_of_user/) - User-level authentication and access control
- [`07_human_in_the_loop/`](../examples/07_human_in_the_loop/) - Approval workflows
- [`08_guardrails/`](../examples/08_guardrails/) - Safety and compliance
- [`09_structured_output/`](../examples/09_structured_output/) - Enforce JSON schemas
- [`11_prompt_engineering/`](../examples/11_prompt_engineering/) - Prompt management and optimization

### 🛡️ Need Validation & Monitoring?
**Middleware patterns:**
- [`12_middleware/`](../examples/12_middleware/) - Input validation, logging, performance monitoring

### 📊 Visualizations?
**Charts and graphs:**
- [`18_visualization/`](../examples/18_visualization/) - Vega-Lite chart generation via `custom_outputs`

### ⏱️ Background Tasks?
**Background kickoff + poll/stream retrieval (deep research, multi-tool workflows):**
- [`19_background_agents/`](../examples/19_background_agents/) - OpenAI Responses API–compatible `/v1/responses` on Apps + `background=true` on Model Serving, backed by Lakebase

### 🔎 Retrieval from Lakebase Postgres?
**ANN / BM25 / hybrid RRF over a Lakebase table (as a sibling of `ai_search`):**
- [`21_lakebase_search/`](../examples/21_lakebase_search/) - `type: lakebase_search` using the `lakebase_vector` + `lakebase_text` extensions, with filter-operator coverage and a UC OTEL `trace_location` example

### 🏗️ Complete Solutions?
**Full applications:**
- [`15_complete_applications/`](../examples/15_complete_applications/) - Executive assistant, research agent, reservation system

---

## Using Examples

### Validate a Configuration
```bash
dao-ai validate -c examples/01_getting_started/minimal.yaml
```

### Visualize the Workflow
```bash
dao-ai graph -c examples/04_genie/genie_basic.yaml -o genie.png
```

### Chat with an Agent
```bash
dao-ai chat -c examples/02_tools/slack_integration.yaml
```

### Deploy to Databricks
```bash
dao-ai workflow generate --deploy --run -c examples/07_human_in_the_loop/human_in_the_loop.yaml
```

---

## 📂 Directory Guide

### 01. Getting Started [📖 README](../examples/01_getting_started/README.md)

Foundation concepts for beginners.

| Example | Description |
|---------|-------------|
| `minimal.yaml` | Simplest possible agent configuration |
| `genie_basic.yaml` | Natural language to SQL with Databricks Genie |

**Prerequisites:** Databricks workspace, basic YAML knowledge  
**Next:** Learn about tools in `02_tools/`

---

### 02. Tools [📖 README](../examples/02_tools/README.md)

Integrate with external services and Databricks capabilities.

| Example | Description |
|---------|-------------|
| `slack_integration.yaml` | Slack messaging integration |
| `custom_mcp.yaml` | Custom MCP integration (JIRA example) |
| `managed_mcp.yaml` | Managed Model Context Protocol integration |
| `external_mcp.yaml` | External MCP with Unity Catalog connections |
| `filtered_mcp.yaml` | MCP tool filtering (security, performance, access control) |
| `vector_search_with_reranking.yaml` | RAG with FlashRank reranking |
| `genie_with_conversation_id.yaml` | Genie with conversation tracking |

**Prerequisites:** Credentials for external services, Unity Catalog access  
**Next:** Optimize with caching in `04_genie/`

---

### 03. Caching [📖 README](../examples/04_genie/README.md)

Improve performance and reduce costs through intelligent caching.

| Example | Description |
|---------|-------------|
| `genie_lru_cache.yaml` | LRU (Least Recently Used) caching for Genie |
| `genie_context_aware_cache.yaml` | Two-tier context-aware caching with PostgreSQL embeddings |
| `genie_in_memory_context_aware_cache.yaml` | In-memory context-aware caching (no database required) |

**Prerequisites:** PostgreSQL or Lakebase required for `genie_context_aware_cache.yaml` only  
**Next:** Add persistence in `05_memory/`

---

### 05. Memory [📖 README](../examples/05_memory/README.md)

Persistent state management and long-term memory for multi-turn conversations.

| Example | Description |
|---------|-------------|
| `in_memory_basic.yaml` | In-memory conversation state (no persistence, good for testing) |
| `postgres_persistence.yaml` | PostgreSQL-backed conversation persistence |
| `lakebase_persistence.yaml` | Lakebase (Databricks-native) conversation persistence |
| `conversation_summarization.yaml` | Long conversation summarization with Lakebase store |

All examples support the optional `extraction` config block for long-term memory with structured schemas, background extraction, and automatic memory injection.

**Prerequisites:** PostgreSQL or Lakebase database (except `in_memory_basic.yaml`)  
**Next:** User-level access control in `06_on_behalf_of_user/`

---

### 06. On-Behalf-Of User [📖 README](../examples/06_on_behalf_of_user/README.md)

User-level authentication and access control with Unity Catalog.

| Example | Description |
|---------|-------------|
| `obo_basic.yaml` | OBO with UC Functions and Genie Spaces |

**Prerequisites:** Unity Catalog, user credentials  
**Next:** Add approval workflows in `07_human_in_the_loop/`

---

### 07. Human-in-the-Loop [📖 README](../examples/07_human_in_the_loop/README.md)

Approval workflows for sensitive operations.

| Example | Description |
|---------|-------------|
| `human_in_the_loop.yaml` | Tool approval workflows and HITL patterns |

**Prerequisites:** MLflow for HITL checkpointing  
**Next:** Add safety guardrails in `08_guardrails/`

---

### 08. Guardrails [📖 README](../examples/08_guardrails/README.md)

Automated safety and validation using MLflow judges (`mlflow.genai.judges.make_judge`). The prompt determines the evaluation type -- tone, completeness, veracity/groundedness, or any custom criteria. Tool context from the conversation is automatically extracted for veracity checks.

| Example | Description |
|---------|-------------|
| `guardrails_basic.yaml` | Tone, completeness, and veracity guardrails with MLflow judges |
| `guardrails_scorers.yaml` | MLflow Scorer-based guardrails (ToxicLanguage, GibberishText) alongside custom judges |

**Prerequisites:** MLflow with Databricks model endpoints  
**Next:** Enforce schemas in `09_structured_output/`

---

### 09. Structured Output [📖 README](../examples/09_structured_output/README.md)

Enforce response format with JSON schema.

| Example | Description |
|---------|-------------|
| `structured_output.yaml` | Type-safe API responses with automatic validation |

**Prerequisites:** Basic understanding of JSON schemas  
**Next:** External agents in `10_agent_integrations/`

---

### 10. Agent Integrations [📖 README](../examples/10_agent_integrations/README.md)

Call another agent as a tool using the first-class `type: app`, `type: serving_endpoint`, and `type: a2a` function types.

| Example | Description |
|---------|-------------|
| `app_first_class.yaml` | `type: app` — call a Databricks App as a tool (lazy `/agent/info` probe + explicit `api:` override) |
| `serving_endpoint_first_class.yaml` | `type: serving_endpoint` — call FMAPI (Chat Completions) and UC-registered ResponsesAgent endpoints (`task` discovery) |
| `agent_bricks.yaml` | Agent Bricks integration with customer support and product expert agents (`type: serving_endpoint`) |
| `kasal.yaml` | Kasal enterprise agents with financial, compliance, and privacy specialists (`type: serving_endpoint`) |

**What You'll Learn:**
- **First-Class Agent Tools**: Use `type: app`, `type: serving_endpoint`, and `type: a2a` to call other agents as tools — no factory boilerplate required
- **Wire-Shape Selection**: Pick OpenAI Responses vs Chat Completions per tool via `api:`, or let dao-ai discover it lazily on first invocation
- **Per-Target Discovery**: Apps probe `/agent/info`; Model Serving endpoints probe `serving_endpoints.get(name).task` — both lazy, cached, and offline-safe
- **Multi-Agent Orchestration**: Coordinate between specialized external agents
- **Delegation Patterns**: Route tasks to purpose-built specialist agents

**Key Concepts:**
- **Hub-and-Spoke Pattern**: One orchestrator routes to multiple specialists
- **Sequential Workflows**: Chain specialist agents for compliance and validation
- **Parallel Consultation**: Consult multiple agents simultaneously for multi-perspective analysis

**Prerequisites:** Target Databricks App or Model Serving endpoint deployed (or an Agent Bricks / Kasal endpoint configured)  
**Next:** Reuse prompts across agents in `11_prompt_engineering/`

> 📖 See [configuration-reference.md → First-Class Agent Tools](configuration-reference.md#first-class-agent-tools) for the full field reference, discovery rules, and offline-safety guarantees.

---

### 11. Prompt Engineering [📖 README](../examples/11_prompt_engineering/README.md)

Define reusable prompts as first-class config objects and share them across agents.

| Example | Description |
|---------|-------------|
| `prompt_registry.yaml` | Reusable inline prompts referenced via YAML anchors |

**Prerequisites:** None  
**Next:** Add validation and monitoring in `12_middleware/`

**Common Patterns:**
```yaml
tools:
  specialist_tool: &specialist_tool
    name: specialist_agent
    function:
      # First-class type — discovery picks Responses vs Chat Completions
      # based on serving_endpoints.get(name).task.
      type: serving_endpoint
      endpoint: external-agent-endpoint-name
      description: |
        Detailed description of when to use this agent.

agents:
  orchestrator:
    name: main_agent
    tools:
      - *specialist_tool
    prompt: |
      You coordinate tasks and delegate to specialist agents.
      Use the specialist tool for X, Y, Z tasks.
```

**Use Cases:**
- **Customer Service**: Route queries to specialized support, product, and escalation agents
- **Financial Services**: Financial analysis with compliance validation and risk assessment
- **Healthcare**: Clinical guidance with HIPAA compliance and privacy validation
- **Enterprise IT**: Multi-domain technical support with security and access control

**Real-World Examples:**

**Agent Bricks** - Customer service automation:
```yaml
# Customer support agent for handling complaints
customer_support_tool:
  function:
    type: serving_endpoint
    endpoint: *agent_bricks_customer_support
    description: "Handle customer complaints, returns, and issues"

# Product expert for technical questions
product_expert_tool:
  function:
    type: serving_endpoint
    endpoint: *agent_bricks_product_expert
    description: "Technical specs, compatibility, recommendations"

# Main agent routes to specialists
orchestrator:
  tools: [customer_support_tool, product_expert_tool]
```

**Kasal** - Enterprise governance workflows:
```yaml
# Financial analyst with compliance checks
enterprise_coordinator:
  tools:
    - financial_analyst_tool      # Data analysis and forecasting
    - compliance_checker_tool     # Regulatory validation
    - privacy_specialist_tool     # PII and data privacy
  prompt: |
    IMPORTANT: For financial decisions, ALWAYS check with 
    compliance validator before providing recommendations.
    For customer data, ALWAYS consult privacy specialist.
```

**Best Practices:**
- **Clear Agent Responsibilities**: Give each agent a specific, well-defined role
- **Effective Prompting**: Provide complete context when calling specialist agents
- **Error Handling**: Handle agent timeout and failure scenarios gracefully
- **Compliance First**: Use compliance validators before making regulatory decisions
- **Performance**: Cache agent responses when appropriate, use parallel calls

---

### 12. Middleware [📖 README](../examples/12_middleware/README.md)

Cross-cutting concerns for production agents: validation, logging, monitoring, limits, retries, and privacy.

| Example | Description |
|---------|-------------|
| `custom_field_validation.yaml` | Input validation patterns (store numbers, tenant IDs, API keys) |
| `logging_middleware.yaml` | Request logging, performance monitoring, audit trails |
| `combined_middleware.yaml` | Production-ready middleware stacks |
| `limit_middleware.yaml` | Tool call and model call limits to prevent runaway loops |
| `retry_middleware.yaml` | Automatic retry with exponential backoff for transient failures |
| `context_management.yaml` | Context editing to prevent token limit issues |
| `pii_middleware.yaml` | PII detection and protection for privacy compliance |

**Key Concepts:**
- **Input Validation**: Ensure required context fields (store_num, user_id) are provided
- **Request Logging**: Track all interactions for debugging and auditing
- **Performance Monitoring**: Identify bottlenecks and slow operations
- **Audit Trails**: Comprehensive logging for compliance
- **Middleware Composition**: Combine multiple middleware in the correct order
- **Tool/Model Limits**: Prevent excessive API calls and runaway loops
- **Retry Logic**: Automatic retry with backoff for transient failures
- **Context Management**: Clear older tool outputs to stay within token limits
- **PII Protection**: Detect and handle sensitive personal information

**Common Patterns:**
```yaml
middleware:
  store_validation: &store_validation
    name: dao_ai.middleware.create_custom_field_validation_middleware
    args:
      fields:
        - name: store_num
          description: "Your store number"
          example_value: "12345"

agents:
  my_agent:
    middleware:
      - *store_validation
    prompt: |
      Store Number: {store_num}
      ...
```

**Real-World Example:**  
The hardware store application uses store number validation to ensure users provide their store location for inventory lookups. See [`15_complete_applications/hardware_store/hardware_store.yaml`](../examples/15_complete_applications/hardware_store/hardware_store.yaml).

**Prerequisites:** Basic understanding of agents and prompts  
**Next:** Learn multi-agent coordination in `13_orchestration/`

---

### 13. Orchestration [📖 README](../examples/13_orchestration/README.md)

Multi-agent coordination patterns.

| Example | Description |
|---------|-------------|
| `supervisor_pattern.yaml` | Supervisor orchestration pattern |
| `swarm_pattern.yaml` | Swarm orchestration pattern |
| `deterministic_handoff_pattern.yaml` | Deterministic handoff pipeline pattern |

**Prerequisites:** Understanding of multi-agent systems  
**Next:** Try basic tools in `14_basic_tools/`

---

### 14. Basic Tools [📖 README](../examples/14_basic_tools/README.md)

Simple tool integrations for SQL and data operations.

| Example | Description |
|---------|-------------|
| `sql_tool_example.yaml` | First-class `type: sql` execution tools for inventory analysis, including a parameterized statement |

**Prerequisites:** Databricks SQL warehouse  
**Next:** See complete applications in `15_complete_applications/`

---

### 15. Complete Applications [📖 README](../examples/15_complete_applications/README.md)

Full-featured, production-ready agent applications.

| Example | Description |
|---------|-------------|
| `executive_assistant.yaml` | Comprehensive assistant with email, calendar, Slack |
| `deep_research.yaml` | Multi-step research agent with web search |
| `reservations_system.yaml` | Restaurant reservation management system |
| `genie_vector_search_hybrid.yaml` | Combined SQL and AI Search capabilities |
| `genie_and_genie_mcp.yaml` | Multiple Genie instances via MCP (experimental) |

**Prerequisites:** All concepts from previous categories  
**Use:** As reference implementations or starting points

---

### 18. Visualization [📖 README](../examples/18_visualization/README.md)

Generate Vega-Lite chart specs from structured data, delivered to clients via `custom_outputs.visualizations`.

| Example | Description |
|---------|-------------|
| `vega_lite_visualization.yaml` | Bar/line/scatter/area/arc/heatmap charts with SQL data source |

**Prerequisites:** Factory tool pattern (`14_basic_tools`)  
**Use:** Pair with SQL or Genie tools to turn query results into interactive charts

---

### 19. Background Agents [📖 Full docs](background_agents.md)

Responses API–compatible kickoff / poll / cancel for agent runs that exceed the ~5 min Model Serving worker timeout or ~120 s Databricks Apps DPAPI timeout. Persists response state + stream events to Lakebase; the background task runs on a persistent daemon thread so it survives the per-request `asyncio.run()` teardown.

| Example | Description |
|---------|-------------|
| `deep_research.yaml` | Deep-research agent with `app.background` enabled, deployed to both Databricks Apps and Model Serving |

**Prerequisites:** Configured Lakebase project  
**Use:** Deep research, multi-agent workflows, or any single inference that may take more than ~2–5 minutes

---

## Customizing Examples

Each example is a starting point:

1. **Copy** to your config directory: `cp examples/01_getting_started/minimal.yaml config/my_agent.yaml`
2. **Modify** prompts, tools, and settings
3. **Validate**: `dao-ai validate -c config/my_agent.yaml`
4. **Test** locally: `dao-ai chat -c config/my_agent.yaml`
5. **Deploy**: `dao-ai workflow generate --deploy -c config/my_agent.yaml`

For detailed guidance, see the README.md in each category directory.

---

## Contributing Examples

Adding a new example? Follow this guide:

1. **Choose the right category** based on the primary feature demonstrated
2. **Use descriptive names**: `tool_name_variant.yaml` (e.g., `slack_with_approval.yaml`)
3. **Add to the appropriate category** (`01_getting_started` through `13_complete_applications`)
4. **Update this file** with a table entry
5. **Test thoroughly** before submitting

See [Contributing Guide](contributing.md) for details.

---

## Navigation

- [← Previous: Configuration Reference](configuration-reference.md)
- [↑ Back to Documentation Index](../README.md#-documentation)
- [Next: CLI Reference →](cli-reference.md)

