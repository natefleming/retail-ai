# DAO AI Example Configurations

Welcome to the DAO AI examples! This directory contains ready-to-use configurations organized in a **numbered, progressive learning path**.

## 🗺️ Learning Path

Follow the numbered directories from 01 to 11 for a structured learning experience:

```
01_getting_started → 02_mcp → 03_reranking → 04_genie → 05_memory 
    → 06_human_in_the_loop → 07_guardrails → 08_structured_output 
    → 09_agent_integrations → 10_prompt_engineering → 11_middleware → 12_orchestration
    → 13_complete_applications → 14_on_behalf_of_user
```

Or jump directly to the category that matches your current need.

---

## 📂 Directory Guide

### [01. Getting Started](01_getting_started/) 
**Foundation concepts for beginners**
- `minimal.yaml` - Simplest possible agent

👉 Start here if you're new to DAO AI

---

### [02. MCP](02_mcp/)
**Integrate with external services**
- Slack, JIRA integrations
- Model Context Protocol (MCP)
- Vector Search with reranking
- Genie with conversation tracking

👉 Learn how to connect agents to tools and services

---

### [03. Reranking](03_reranking/)
**Improve search result relevance**
- FlashRank integration
- Two-stage retrieval
- Semantic reranking patterns

👉 Boost search quality by 20-40% with minimal latency

---

### [04. Genie](04_genie/)
**Natural language to SQL**
- Basic Genie integration
- LRU caching for performance
- Semantic caching with embeddings

👉 Query data with natural language, optimized with caching

---

### [05. Memory](05_memory/)
**Persistent state management**
- Conversation summarization
- PostgreSQL/Lakebase checkpointers
- User preference stores

👉 Add memory for multi-turn conversations

---

### [06. Human-in-the-Loop](06_human_in_the_loop/)
**Approval workflows for sensitive operations**
- Tool approval workflows
- Review prompts and decision handling
- State management for interrupts

👉 Get human approval before executing critical actions

---

### [07. Guardrails](07_guardrails/)
**Automated safety and validation**
- PII detection and content filtering
- Toxicity and bias detection
- Input/output validation

👉 Essential for production safety and compliance

---

### [08. Structured Output](08_structured_output/)
**Enforce JSON schema responses**
- Type-safe API responses
- Data extraction patterns
- Automatic validation

👉 Guarantee consistent, parseable responses

---

### [09. Agent Integrations](09_agent_integrations/)
**Integrate with external agent platforms**
- Agent Bricks integration
- Kasal enterprise agents
- Multi-agent orchestration with specialists

👉 Delegate to purpose-built external agents

---

### [10. Prompt Engineering](10_prompt_engineering/)
**Prompt management and optimization**
- MLflow prompt registry
- GEPA automated optimization
- Version control and A/B testing

👉 Improve prompt quality and maintainability

---

### [11. Middleware](11_middleware/)
**Cross-cutting concerns for agents**
- Custom input validation (store numbers, tenant IDs, API keys)
- Request logging and audit trails
- Performance monitoring and tracking
- Combined middleware stacks

👉 Add validation, logging, and monitoring to your agents

---

### [12. Orchestration](12_orchestration/)
**Multi-agent coordination**
- Supervisor pattern (coming soon)
- Swarm pattern (coming soon)
- Hierarchical agents (coming soon)

👉 Coordinate multiple specialized agents

---

### [13. Complete Applications](13_complete_applications/)
**Production-ready systems**
- Executive assistant
- Deep research agent
- Reservation system
- Hybrid Genie + Vector Search

👉 Reference implementations for real-world applications

---

### [14. On-Behalf-Of User](14_on_behalf_of_user/)
**User-level authentication and access control**
- OBO with UC Functions
- OBO with Genie Spaces
- User permission inheritance
- Multi-tenant patterns

👉 Enable user-level access control and audit trails

---

## 🚀 Quick Start

### Validate a Configuration
```bash
dao-ai validate -c config/examples/01_getting_started/minimal.yaml
```

### Visualize the Agent Workflow
```bash
dao-ai graph -c config/examples/02_mcp/slack_integration.yaml -o agent_graph.png
```

### Chat with an Agent
```bash
dao-ai chat -c config/examples/02_mcp/slack_integration.yaml
```

### Deploy to Databricks
```bash
dao-ai bundle --deploy --run -c config/examples/06_human_in_the_loop/human_in_the_loop.yaml
```

---

## 🎯 Find What You Need

### I want to...

**...learn DAO AI basics**  
→ Start with [`01_getting_started/`](01_getting_started/)

**...connect to Slack/JIRA/other services**  
→ Check [`02_mcp/`](02_mcp/)

**...improve search result quality**  
→ See [`03_reranking/`](03_reranking/)

**...improve performance and reduce costs**  
→ Explore [`04_genie/`](04_genie/)

**...add conversation memory**  
→ See [`05_memory/`](05_memory/)

**...add approval workflows for sensitive actions**  
→ Review [`06_human_in_the_loop/`](06_human_in_the_loop/)

**...add safety and compliance guardrails**  
→ Check [`07_guardrails/`](07_guardrails/)

**...manage and optimize prompts**  
→ Learn from [`10_prompt_engineering/`](10_prompt_engineering/)

**...add validation, logging, or monitoring**  
→ Check [`11_middleware/`](11_middleware/)

**...coordinate multiple agents**  
→ Study [`12_orchestration/`](12_orchestration/)

**...see complete, production-ready examples**  
→ Explore [`13_complete_applications/`](13_complete_applications/)

**...implement user-level access control**  
→ Review [`14_on_behalf_of_user/`](14_on_behalf_of_user/)

---

## 📖 Documentation

- **[Main Documentation](../../docs/)** - Comprehensive guides
- **[Configuration Reference](../../docs/configuration-reference.md)** - Complete YAML reference
- **[Key Capabilities](../../docs/key-capabilities.md)** - Feature deep-dives
- **[CLI Reference](../../docs/cli-reference.md)** - Command-line usage
- **[FAQ](../../docs/faq.md)** - Common questions

---

## 🛠️ Customizing Examples

Each example is a starting point for your own agents:

1. **Copy** the example to your config directory:
   ```bash
   cp config/examples/01_getting_started/minimal.yaml config/my_agent.yaml
   ```

2. **Modify** prompts, tools, and settings for your use case

3. **Validate** your configuration:
   ```bash
   dao-ai validate -c config/my_agent.yaml
   ```

4. **Test** locally:
   ```bash
   dao-ai chat -c config/my_agent.yaml
   ```

5. **Deploy** to Databricks:
   ```bash
   dao-ai bundle --deploy -c config/my_agent.yaml
   ```

---

## 🤝 Contributing

Have an example to share? We'd love to see it!

### Adding a New Example

1. **Choose the right category** (`01_getting_started` through `11_complete_applications`)
2. **Use descriptive naming**: `tool_name_variant.yaml` (e.g., `slack_with_threads.yaml`)
3. **Add inline comments** explaining key concepts
4. **Test thoroughly** with `dao-ai validate` and `dao-ai chat`
5. **Update documentation**:
   - Add entry to the category's README.md
   - Update [`docs/examples.md`](../../docs/examples.md)
6. **Submit a pull request**

See the [Contributing Guide](../../docs/contributing.md) for details.

---

## 💡 Tips for Success

### Start Simple
Begin with `01_getting_started/minimal.yaml` and gradually add complexity.

### Follow the Path
The numbered structure is designed as a learning progression. Follow it!

### Read the READMEs
Each category has a detailed README with prerequisites, tips, and troubleshooting.

### Experiment Locally
Use `dao-ai chat` to test configurations before deploying.

### Use Version Control
Keep your configurations in Git for tracking and collaboration.

### Monitor in Production
Use MLflow to track agent performance and costs.

---

## 📊 Example Complexity Matrix

| Category | Complexity | Time to Learn | Prerequisites |
|----------|------------|---------------|---------------|
| 01_getting_started | ⭐ | 30 min | Basic YAML |
| 02_mcp | ⭐⭐ | 1-2 hrs | Category 01 |
| 03_reranking | ⭐⭐ | 1 hr | Vector search setup |
| 04_genie | ⭐⭐ | 1 hr | Category 02 |
| 05_memory | ⭐⭐⭐ | 2 hrs | Database setup |
| 06_human_in_the_loop | ⭐⭐⭐ | 2 hrs | Checkpointer setup |
| 07_guardrails | ⭐⭐⭐ | 2-3 hrs | Production mindset |
| 09_agent_integrations | ⭐⭐⭐ | 2-3 hrs | Agent endpoints |
| 10_prompt_engineering | ⭐⭐⭐⭐ | 3-4 hrs | MLflow setup |
| 11_middleware | ⭐⭐ | 1-2 hrs | Category 01 |
| 12_orchestration | ⭐⭐⭐⭐ | 4-6 hrs | Multi-agent concepts |
| 13_complete_applications | ⭐⭐⭐⭐⭐ | 6-8 hrs | All above |

---

## 🆘 Getting Help

- **Documentation**: [docs/](../../docs/)
- **Examples Guide**: [docs/examples.md](../../docs/examples.md)
- **FAQ**: [docs/faq.md](../../docs/faq.md)
- **Issues**: [GitHub Issues](https://github.com/your-org/dao-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/dao-ai/discussions)

---

## 📝 Example Naming Conventions

We use descriptive naming to make examples easy to find:

```
[tool/feature]_[variant].yaml

Examples:
- minimal.yaml                (foundational example)
- genie_lru_cache.yaml        (specific caching variant)
- slack_integration.yaml      (integration example)
- external_mcp.yaml           (variant with specific feature)
```

---

## 🎓 Recommended Learning Path

### Week 1: Foundations
- Day 1-2: `01_getting_started/` - Basic concepts
- Day 3-4: `02_mcp/` - Tool integrations
- Day 5: `03_reranking/` - Search optimization

### Week 2: Performance & State
- Day 1: `04_genie/` - Performance optimization
- Day 2-3: `05_memory/` - State management
- Day 4: `06_human_in_the_loop/` - Approval workflows
- Day 5: `07_guardrails/` - Safety and validation

### Week 3: Advanced Patterns
- Day 1: `09_agent_integrations/` - External agent platforms
- Day 2-3: `10_prompt_engineering/` - Prompt management
- Day 4: `11_middleware/` - Validation and monitoring
- Day 5: `12_orchestration/` - Multi-agent coordination

### Week 4: Production
- Day 1-5: `13_complete_applications/` - Full systems

### Week 4: Build Your Own
- Apply learned patterns to your use case
- Deploy to production
- Monitor and iterate

---

**Ready to start?** Head to [`01_getting_started/`](01_getting_started/) and build your first agent! 🚀
