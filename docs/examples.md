# Example Configurations

The `config/examples/` directory contains ready-to-use configurations organized in a **numbered, progressive learning path**. Each directory builds upon the previous, guiding you from basic concepts to production-ready applications.

## 📚 Learning Path

The examples follow a natural progression:

```
01_getting_started → 02_tools → 03_caching → 04_memory 
    → 05_quality_control → 06_prompt_engineering 
    → 07_orchestration → 08_complete_applications
```

Start at `01_getting_started` if you're new, or jump directly to the category that matches your needs.

## Quick Reference

### 🆕 New to DAO AI?
**Start here:**
- [`01_getting_started/minimal.yaml`](../config/examples/01_getting_started/minimal.yaml) - Simplest possible agent
- [`01_getting_started/genie_basic.yaml`](../config/examples/01_getting_started/genie_basic.yaml) - Natural language to SQL

### 🔧 Need Specific Tools?
**Explore:**
- [`02_tools/`](../config/examples/02_tools/) - Genie, Vector Search, Slack, JIRA, MCP integrations

### ⚡ Optimizing Performance?
**Check out:**
- [`03_caching/`](../config/examples/03_caching/) - LRU and semantic caching strategies

### 💾 Managing State?
**See:**
- [`04_memory/`](../config/examples/04_memory/) - Conversation history and persistence

### 🛡️ Production Ready?
**Essential patterns:**
- [`05_quality_control/`](../config/examples/05_quality_control/) - Guardrails, HITL, structured output
- [`06_prompt_engineering/`](../config/examples/06_prompt_engineering/) - Prompt management and optimization

### 🏗️ Complete Solutions?
**Full applications:**
- [`08_complete_applications/`](../config/examples/08_complete_applications/) - Executive assistant, research agent, reservation system

---

## Using Examples

### Validate a Configuration
```bash
dao-ai validate -c config/examples/01_getting_started/minimal.yaml
```

### Visualize the Workflow
```bash
dao-ai graph -c config/examples/01_getting_started/genie_basic.yaml -o genie.png
```

### Chat with an Agent
```bash
dao-ai chat -c config/examples/02_tools/slack_integration.yaml
```

### Deploy to Databricks
```bash
dao-ai bundle --deploy --run -c config/examples/05_quality_control/human_in_the_loop.yaml
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
| `jira_integration.yaml` | JIRA issue tracking integration |
| `mcp_basic.yaml` | Model Context Protocol integration |
| `mcp_with_uc_connection.yaml` | MCP with Unity Catalog connections |
| `vector_search_with_reranking.yaml` | RAG with FlashRank reranking |
| `genie_with_conversation_id.yaml` | Genie with conversation tracking |

**Prerequisites:** Credentials for external services, Unity Catalog access  
**Next:** Optimize with caching in `03_caching/`

---

### 03. Caching [📖 README](../config/examples/03_caching/README.md)

Improve performance and reduce costs through intelligent caching.

| Example | Description |
|---------|-------------|
| `genie_lru_cache.yaml` | LRU (Least Recently Used) caching for Genie |
| `genie_semantic_cache.yaml` | Two-tier semantic caching with embeddings |

**Prerequisites:** PostgreSQL or Lakebase for semantic cache  
**Next:** Add persistence in `04_memory/`

---

### 04. Memory [📖 README](../config/examples/04_memory/README.md)

Persistent state management for multi-turn conversations.

| Example | Description |
|---------|-------------|
| `conversation_summarization.yaml` | Long conversation summarization with PostgreSQL |

**Prerequisites:** PostgreSQL or Lakebase database  
**Next:** Add safety with `05_quality_control/`

---

### 05. Quality Control [📖 README](../config/examples/05_quality_control/README.md)

Production-grade safety, validation, and approval workflows.

| Example | Description |
|---------|-------------|
| `guardrails_basic.yaml` | Content filtering and safety guardrails |
| `human_in_the_loop.yaml` | Tool approval workflows and HITL patterns |
| `structured_output.yaml` | Enforce response format with JSON schema |

**Prerequisites:** MLflow for HITL, guardrail services (optional)  
**Next:** Optimize prompts in `06_prompt_engineering/`

---

### 06. Prompt Engineering [📖 README](../config/examples/06_prompt_engineering/README.md)

Prompt versioning, management, and automated optimization.

| Example | Description |
|---------|-------------|
| `prompt_registry.yaml` | MLflow prompt registry integration |
| `prompt_optimization.yaml` | Automated prompt tuning with GEPA |

**Prerequisites:** MLflow prompt registry, training dataset for optimization  
**Next:** Scale with orchestration in `07_orchestration/`

---

### 07. Orchestration [📖 README](../config/examples/07_orchestration/README.md)

Multi-agent coordination patterns.

| Example | Description |
|---------|-------------|
| *(Coming soon)* | Supervisor and swarm orchestration patterns |

**Prerequisites:** Understanding of multi-agent systems  
**Next:** See complete applications in `08_complete_applications/`

---

### 08. Complete Applications [📖 README](../config/examples/08_complete_applications/README.md)

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
3. **Add to the appropriate category** (`01_getting_started` through `08_complete_applications`)
4. **Update this file** with a table entry
5. **Test thoroughly** before submitting

See [Contributing Guide](contributing.md) for details.

---

## Navigation

- [← Previous: Configuration Reference](configuration-reference.md)
- [↑ Back to Documentation Index](../README.md#-documentation)
- [Next: CLI Reference →](cli-reference.md)

