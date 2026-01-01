# 01. Getting Started

**Foundation concepts for beginners**

This is your starting point. These examples introduce the core concepts of DAO AI with minimal complexity.

## Examples

| File | Description | Prerequisites |
|------|-------------|---------------|
| `minimal.yaml` | Simplest possible agent configuration | Databricks workspace, LLM endpoint |

## What You'll Learn

- **Basic YAML structure** - How DAO AI configurations are organized
- **Agent definition** - Defining an agent with a model and prompt
- **Simple workflows** - Single-agent patterns

## Quick Start

### Test the minimal agent
```bash
dao-ai validate -c config/examples/01_getting_started/minimal.yaml
dao-ai chat -c config/examples/01_getting_started/minimal.yaml
```


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

👉 **02_mcp/** - Explore more tool integrations (Slack, JIRA, Vector Search, MCP)  
👉 **04_genie/** - Learn to optimize performance with caching  
👉 **07_human_in_the_loop/** - Add production safety features

## Troubleshooting

**"Model endpoint not found"**
- Verify your model endpoint name in the YAML
- Check you have access to the endpoint
- Try using FMAPI models (e.g., `databricks-claude-sonnet-4`)

**"Genie space not accessible"**
- Confirm you have the correct Genie space ID
- Verify permissions to access the space
- Check that tables exist in the space

## Related Documentation

- [Configuration Reference](../../../docs/configuration-reference.md)
- [Key Capabilities](../../../docs/key-capabilities.md)
- [Python API](../../../docs/python-api.md)

