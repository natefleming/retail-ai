# Example Configurations

The `config/examples/` directory contains ready-to-use configurations organized in a **numbered, progressive learning path**. Each directory builds upon the previous, guiding you from basic concepts to production-ready applications.

## 📚 Learning Path

The examples follow a natural progression:

```
01_getting_started → 02_mcp → 04_genie → 05_memory 
    → 06_human_in_the_loop → 09_agent_integrations → 10_prompt_engineering 
    → 11_middleware → 12_orchestration → 13_complete_applications
```

Start at `01_getting_started` if you're new, or jump directly to the category that matches your needs.

## Quick Reference

### 🆕 New to DAO AI?
**Start here:**
- [`01_getting_started/minimal.yaml`](../config/examples/01_getting_started/minimal.yaml) - Simplest possible agent
- [`04_genie/genie_basic.yaml`](../config/examples/04_genie/genie_basic.yaml) - Natural language to SQL

### 🔧 Need Specific Tools?
**Explore:**
- [`02_mcp/`](../config/examples/02_mcp/) - Genie, Vector Search, Slack, JIRA, MCP integrations
- [`09_agent_integrations/`](../config/examples/09_agent_integrations/) - Agent Bricks, Kasal, external agent platforms

### ⚡ Optimizing Performance?
**Check out:**
- [`04_genie/`](../config/examples/04_genie/) - LRU and semantic caching strategies

### 💾 Managing State?
**See:**
- [`05_memory/`](../config/examples/05_memory/) - Conversation history and persistence

### 🛡️ Production Ready?
**Essential patterns:**
- [`06_human_in_the_loop/`](../config/examples/06_human_in_the_loop/) - Guardrails, HITL, structured output
- [`10_prompt_engineering/`](../config/examples/10_prompt_engineering/) - Prompt management and optimization

### 🛡️ Need Validation & Monitoring?
**Middleware patterns:**
- [`11_middleware/`](../config/examples/11_middleware/) - Input validation, logging, performance monitoring

### 🏗️ Complete Solutions?
**Full applications:**
- [`13_complete_applications/`](../config/examples/13_complete_applications/) - Executive assistant, research agent, reservation system

---

## Using Examples

### Validate a Configuration
```bash
dao-ai validate -c config/examples/01_getting_started/minimal.yaml
```

### Visualize the Workflow
```bash
dao-ai graph -c config/examples/04_genie/genie_basic.yaml -o genie.png
```

### Chat with an Agent
```bash
dao-ai chat -c config/examples/02_tools/slack_integration.yaml
```

### Deploy to Databricks
```bash
dao-ai bundle --deploy --run -c config/examples/06_human_in_the_loop/human_in_the_loop.yaml
```

---

## 📂 Directory Guide

### 01. Getting Started [📖 README](../config/examples/01_getting_started/README.md)

Foundation concepts for beginners.

| Example | Description |
|---------|-------------|
| `minimal.yaml` | Simplest possible agent configuration |
| `genie_basic.yaml` | Natural language to SQL with Databricks Genie |

**Prerequisites:** Databricks workspace, basic YAML knowledge  
**Next:** Learn about tools in `02_tools/`

---

### 02. Tools [📖 README](../config/examples/02_tools/README.md)

Integrate with external services and Databricks capabilities.

| Example | Description |
|---------|-------------|
| `slack_integration.yaml` | Slack messaging integration |
| `custom_mcp.yaml` | Custom MCP integration (JIRA example) |
| `managed_mcp.yaml` | Managed Model Context Protocol integration |
| `external_mcp.yaml` | External MCP with Unity Catalog connections |
| `vector_search_with_reranking.yaml` | RAG with FlashRank reranking |
| `genie_with_conversation_id.yaml` | Genie with conversation tracking |

**Prerequisites:** Credentials for external services, Unity Catalog access  
**Next:** Optimize with caching in `04_genie/`

---

### 03. Caching [📖 README](../config/examples/04_genie/README.md)

Improve performance and reduce costs through intelligent caching.

| Example | Description |
|---------|-------------|
| `genie_lru_cache.yaml` | LRU (Least Recently Used) caching for Genie |
| `genie_semantic_cache.yaml` | Two-tier semantic caching with embeddings |

**Prerequisites:** PostgreSQL or Lakebase for semantic cache  
**Next:** Add persistence in `05_memory/`

---

### 04. Memory [📖 README](../config/examples/05_memory/README.md)

Persistent state management for multi-turn conversations.

| Example | Description |
|---------|-------------|
| `conversation_summarization.yaml` | Long conversation summarization with PostgreSQL |

**Prerequisites:** PostgreSQL or Lakebase database  
**Next:** Add safety with `06_human_in_the_loop/`

---

### 05. Quality Control [📖 README](../config/examples/06_human_in_the_loop/README.md)

Production-grade safety, validation, and approval workflows.

| Example | Description |
|---------|-------------|
| `guardrails_basic.yaml` | Content filtering and safety guardrails |
| `human_in_the_loop.yaml` | Tool approval workflows and HITL patterns |
| `structured_output.yaml` | Enforce response format with JSON schema |

**Prerequisites:** MLflow for HITL, guardrail services (optional)  
**Next:** Optimize prompts in `09_prompt_engineering/`

---

### 06. Prompt Engineering [📖 README](../config/examples/09_prompt_engineering/README.md)

Prompt versioning, management, and automated optimization.

| Example | Description |
|---------|-------------|
| `prompt_registry.yaml` | MLflow prompt registry integration |
| `prompt_optimization.yaml` | Automated prompt tuning with GEPA |

**Prerequisites:** MLflow prompt registry, training dataset for optimization  
**Next:** Learn agent integrations in `09_agent_integrations/`

---

### 07. Agent Integrations [📖 README](../config/examples/09_agent_integrations/README.md)

Integrate with external agent platforms like Agent Bricks and Kasal using agent endpoint tools.

| Example | Description |
|---------|-------------|
| `agent_bricks.yaml` | Agent Bricks integration with customer support and product expert agents |
| `kasal.yaml` | Kasal enterprise agents with financial, compliance, and privacy specialists |

**What You'll Learn:**
- **Agent Endpoint Tools**: Call external agents as tools within your DAO-AI agents
- **Multi-Agent Orchestration**: Coordinate between specialized external agents
- **Delegation Patterns**: Route tasks to purpose-built specialist agents
- **Enterprise Integration**: Leverage existing agent infrastructure with governance

**Key Concepts:**
- **Hub-and-Spoke Pattern**: One orchestrator routes to multiple specialists
- **Sequential Workflows**: Chain specialist agents for compliance and validation
- **Parallel Consultation**: Consult multiple agents simultaneously for multi-perspective analysis

**Common Patterns:**
```yaml
resources:
  llms:
    # External agent endpoint configuration
    specialist_agent: &specialist_agent
      name: external-agent-endpoint-name
      description: "Agent capabilities"
      temperature: 0.1
      max_tokens: 1000

tools:
  specialist_tool: &specialist_tool
    name: specialist_agent
    function:
      type: factory
      name: dao_ai.tools.create_agent_endpoint_tool
      args:
        llm: *specialist_agent
        name: specialist
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
    name: dao_ai.tools.create_agent_endpoint_tool
    args:
      llm: *agent_bricks_customer_support
      description: "Handle customer complaints, returns, and issues"

# Product expert for technical questions
product_expert_tool:
  function:
    name: dao_ai.tools.create_agent_endpoint_tool
    args:
      llm: *agent_bricks_product_expert
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

**Prerequisites:** Understanding of agent tools and multi-agent concepts  
**Next:** Learn middleware patterns in `11_middleware/`

---

### 08. Middleware [📖 README](../config/examples/11_middleware/README.md)

Cross-cutting concerns for production agents: validation, logging, and monitoring.

| Example | Description |
|---------|-------------|
| `custom_field_validation.yaml` | Input validation patterns (store numbers, tenant IDs, API keys) |
| `logging_middleware.yaml` | Request logging, performance monitoring, audit trails |
| `combined_middleware.yaml` | Production-ready middleware stacks |

**Key Concepts:**
- **Input Validation**: Ensure required context fields (store_num, user_id) are provided
- **Request Logging**: Track all interactions for debugging and auditing
- **Performance Monitoring**: Identify bottlenecks and slow operations
- **Audit Trails**: Comprehensive logging for compliance
- **Middleware Composition**: Combine multiple middleware in the correct order

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
The hardware store application uses store number validation to ensure users provide their store location for inventory lookups. See [`12_complete_applications/hardware_store.yaml`](../config/examples/12_complete_applications/hardware_store.yaml).

**Prerequisites:** Basic understanding of agents and prompts  
**Next:** Learn multi-agent coordination in `12_orchestration/`

---

### 09. Orchestration [📖 README](../config/examples/12_orchestration/README.md)

Multi-agent coordination patterns.

| Example | Description |
|---------|-------------|
| *(Coming soon)* | Supervisor and swarm orchestration patterns |

**Prerequisites:** Understanding of multi-agent systems  
**Next:** See complete applications in `13_complete_applications/`

---

### 10. Complete Applications [📖 README](../config/examples/13_complete_applications/README.md)

Full-featured, production-ready agent applications.

| Example | Description |
|---------|-------------|
| `executive_assistant.yaml` | Comprehensive assistant with email, calendar, Slack |
| `deep_research.yaml` | Multi-step research agent with web search |
| `reservations_system.yaml` | Restaurant reservation management system |
| `genie_vector_search_hybrid.yaml` | Combined SQL and vector search capabilities |
| `genie_and_genie_mcp.yaml` | Multiple Genie instances via MCP (experimental) |

**Prerequisites:** All concepts from previous categories  
**Use:** As reference implementations or starting points

---

## Customizing Examples

Each example is a starting point:

1. **Copy** to your config directory: `cp config/examples/01_getting_started/minimal.yaml config/my_agent.yaml`
2. **Modify** prompts, tools, and settings
3. **Validate**: `dao-ai validate -c config/my_agent.yaml`
4. **Test** locally: `dao-ai chat -c config/my_agent.yaml`
5. **Deploy**: `dao-ai bundle --deploy -c config/my_agent.yaml`

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

