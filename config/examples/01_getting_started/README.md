# 01. Getting Started

**Foundation concepts for beginners**

This is your starting point. These examples introduce the core concepts of DAO AI with minimal complexity.

## Examples

| File | Description | Prerequisites |
|------|-------------|---------------|
| `minimal.yaml` | Simplest possible agent configuration | Databricks workspace, LLM endpoint |
| `genie_basic.yaml` | Natural language to SQL with Databricks Genie | Genie space access |

## What You'll Learn

- **Basic YAML structure** - How DAO AI configurations are organized
- **Agent definition** - Defining an agent with a model and prompt
- **Tool integration** - Connecting your first tool (Genie)
- **Simple workflows** - Single-agent, single-tool patterns

## Quick Start

### Test the minimal agent
```bash
dao-ai validate -c config/examples/01_getting_started/minimal.yaml
dao-ai chat -c config/examples/01_getting_started/minimal.yaml
```

### Try Genie SQL queries
```bash
dao-ai chat -c config/examples/01_getting_started/genie_basic.yaml
```

Example query: *"What are the top 10 customers by revenue?"*

## Key Concepts

### Minimal Agent
- Demonstrates the absolute minimum required configuration
- Single agent with one LLM endpoint
- Simple prompt, no tools
- Great for understanding the basic structure

### Genie Basic
- Adds tool integration (Genie SQL)
- Shows how to configure a Databricks tool
- Demonstrates natural language to SQL translation
- Foundation for more complex data agents

## Prerequisites

Before using these examples:
- ✅ Databricks workspace access
- ✅ Model Serving endpoint configured (or using FMAPI)
- ✅ For Genie: Genie space with tables configured

## Next Steps

Once you're comfortable with these basics:

👉 **02_tools/** - Explore more tool integrations (Slack, JIRA, Vector Search, MCP)  
👉 **03_caching/** - Learn to optimize performance with caching  
👉 **05_quality_control/** - Add production safety features

## Troubleshooting

**"Model endpoint not found"**
- Verify your model endpoint name in the YAML
- Check you have access to the endpoint
- Try using FMAPI models (e.g., `databricks-meta-llama-3-3-70b-instruct`)

**"Genie space not accessible"**
- Confirm you have the correct Genie space ID
- Verify permissions to access the space
- Check that tables exist in the space

## Related Documentation

- [Configuration Reference](../../../docs/configuration-reference.md)
- [Key Capabilities](../../../docs/key-capabilities.md)
- [Python API](../../../docs/python-api.md)

