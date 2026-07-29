---
layout: home
title: Home
nav_order: 0
---

# DAO: Declarative Agent Orchestration

**Production-grade AI agents defined in YAML, powered by LangGraph, deployed on Databricks.**

DAO is an **infrastructure-as-code framework** for building, deploying, and managing
multi-agent AI systems. Instead of writing boilerplate Python to wire up agents, tools,
and orchestration, you define everything declaratively in YAML configuration files.

```yaml
# Define an agent in 10 lines of YAML
agents:
  product_expert:
    name: product_expert
    model: *claude_sonnet
    tools:
      - *ai_search_tool
      - *genie_tool
    prompt: |
      You are a product expert. Answer questions about inventory and pricing.
```

## Getting started

- [**Why DAO?**](docs/why-dao.md) — what DAO is and how it compares to other platforms.
- [**Architecture**](docs/architecture.md) — how DAO works under the hood.
- [**Key Capabilities**](docs/key-capabilities.md) — 20 features for production agents.

## Reference

- [**Configuration Reference**](docs/configuration-reference.md) — the complete YAML schema.
- [**CLI Reference**](docs/cli-reference.md) — the `dao-ai` command-line interface.
- [**Python API**](docs/python-api.md) — programmatic usage and customization.

## Guides

- [**Examples**](docs/examples.md) — ready-to-use example configurations.
- [**MCP Server**](docs/mcp_server.md) — expose a dao-ai agent as a single MCP tool.
- [**A2A Protocol**](docs/a2a_protocol.md) — Google Agent2Agent endpoints on every Apps deployment.
- [**Background Agents**](docs/background_agents.md) — kickoff / poll / cancel for multi-minute runs.
- [**Auditable Tool Invocations**](docs/audit.md) — tamper-evident approval receipts and audit-trail queries.

## Visual configuration studio

Prefer a visual interface?
[**DAO AI Builder**](https://github.com/natefleming/dao-ai-builder) is a React web app that
provides a graphical interface for creating and editing DAO configurations — explore
capabilities, learn the configuration structure with guided forms, and build agents without
writing YAML by hand. It generates valid configurations that work seamlessly with this
framework.
