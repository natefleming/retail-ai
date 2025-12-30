# DAO AI Example Configurations

Welcome to the DAO AI examples! This directory contains ready-to-use configurations organized in a **numbered, progressive learning path**.

## 🗺️ Learning Path

Follow the numbered directories from 01 to 08 for a structured learning experience:

```
01_getting_started → 02_tools → 03_caching → 04_memory 
    → 05_quality_control → 06_prompt_engineering 
    → 07_orchestration → 08_complete_applications
```

Or jump directly to the category that matches your current need.

---

## 📂 Directory Guide

### [01. Getting Started](01_getting_started/) 
**Foundation concepts for beginners**
- `minimal.yaml` - Simplest possible agent
- `genie_basic.yaml` - Natural language to SQL

👉 Start here if you're new to DAO AI

---

### [02. Tools](02_tools/)
**Integrate with external services**
- Slack, JIRA integrations
- Model Context Protocol (MCP)
- Vector Search with reranking
- Genie with conversation tracking

👉 Learn how to connect agents to tools and services

---

### [03. Caching](03_caching/)
**Performance optimization**
- LRU (Least Recently Used) caching
- Semantic caching with embeddings
- Two-tier caching strategies

👉 Reduce costs and improve response times by 50-80%

---

### [04. Memory](04_memory/)
**Persistent state management**
- Conversation summarization
- PostgreSQL/Lakebase checkpointers
- User preference stores

👉 Add memory for multi-turn conversations

---

### [05. Quality Control](05_quality_control/)
**Production-grade safety**
- Guardrails (PII, bias, toxicity)
- Human-in-the-Loop (HITL) approval workflows
- Structured output validation

👉 Essential for production deployments

---

### [06. Prompt Engineering](06_prompt_engineering/)
**Prompt management and optimization**
- MLflow prompt registry
- GEPA automated optimization
- Version control and A/B testing

👉 Improve prompt quality and maintainability

---

### [07. Orchestration](07_orchestration/)
**Multi-agent coordination**
- Supervisor pattern (coming soon)
- Swarm pattern (coming soon)
- Hierarchical agents (coming soon)

👉 Coordinate multiple specialized agents

---

### [08. Complete Applications](08_complete_applications/)
**Production-ready systems**
- Executive assistant
- Deep research agent
- Reservation system
- Hybrid Genie + Vector Search

👉 Reference implementations for real-world applications

---

## 🚀 Quick Start

### Validate a Configuration
```bash
dao-ai validate -c config/examples/01_getting_started/minimal.yaml
```

### Visualize the Agent Workflow
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

## 🎯 Find What You Need

### I want to...

**...learn DAO AI basics**  
→ Start with [`01_getting_started/`](01_getting_started/)

**...connect to Slack/JIRA/other services**  
→ Check [`02_tools/`](02_tools/)

**...improve performance and reduce costs**  
→ Explore [`03_caching/`](03_caching/)

**...add conversation memory**  
→ See [`04_memory/`](04_memory/)

**...deploy safely to production**  
→ Review [`05_quality_control/`](05_quality_control/)

**...manage and optimize prompts**  
→ Learn from [`06_prompt_engineering/`](06_prompt_engineering/)

**...coordinate multiple agents**  
→ Study [`07_orchestration/`](07_orchestration/)

**...see complete, production-ready examples**  
→ Explore [`08_complete_applications/`](08_complete_applications/)

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

1. **Choose the right category** (`01_getting_started` through `08_complete_applications`)
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
| 02_tools | ⭐⭐ | 1-2 hrs | Category 01 |
| 03_caching | ⭐⭐ | 1 hr | Category 02 |
| 04_memory | ⭐⭐⭐ | 2 hrs | Database setup |
| 05_quality_control | ⭐⭐⭐ | 2-3 hrs | Production mindset |
| 06_prompt_engineering | ⭐⭐⭐⭐ | 3-4 hrs | MLflow setup |
| 07_orchestration | ⭐⭐⭐⭐ | 4-6 hrs | Multi-agent concepts |
| 08_complete_applications | ⭐⭐⭐⭐⭐ | 6-8 hrs | All above |

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
- genie_basic.yaml            (foundational example)
- genie_lru_cache.yaml        (specific caching variant)
- slack_integration.yaml      (integration example)
- mcp_with_uc_connection.yaml (variant with specific feature)
```

---

## 🎓 Recommended Learning Path

### Week 1: Foundations
- Day 1-2: `01_getting_started/` - Basic concepts
- Day 3-4: `02_tools/` - Tool integrations
- Day 5: `03_caching/` - Performance optimization

### Week 2: Production Features
- Day 1-2: `04_memory/` - State management
- Day 3-4: `05_quality_control/` - Safety and validation
- Day 5: `06_prompt_engineering/` - Prompt management

### Week 3: Advanced Patterns
- Day 1-3: `07_orchestration/` - Multi-agent coordination
- Day 4-5: `08_complete_applications/` - Full systems

### Week 4: Build Your Own
- Apply learned patterns to your use case
- Deploy to production
- Monitor and iterate

---

**Ready to start?** Head to [`01_getting_started/`](01_getting_started/) and build your first agent! 🚀
