# 02. Tools

**Integrate with external services and Databricks capabilities**

This category demonstrates how to connect your agents to various tools and services. Each example focuses on a specific tool integration pattern.

## Examples

| File | Description | Prerequisites |
|------|-------------|---------------|
| `slack_integration.yaml` | Slack messaging integration | Slack workspace, bot token |
| `jira_integration.yaml` | JIRA issue tracking | JIRA instance, API token |
| `mcp_basic.yaml` | Model Context Protocol integration | MCP server |
| `mcp_with_uc_connection.yaml` | MCP with Unity Catalog connections | Unity Catalog, MCP connection |
| `vector_search_with_reranking.yaml` | RAG with FlashRank reranking | Vector Search index, FlashRank |
| `genie_with_conversation_id.yaml` | Genie with conversation tracking | Genie space |

## What You'll Learn

- **External service integration** - Connect to Slack, JIRA, and other services
- **Model Context Protocol (MCP)** - Standardized tool integration
- **Unity Catalog connections** - Secure credential management
- **Vector Search** - Semantic search and RAG patterns
- **Reranking** - Improve search relevance with FlashRank
- **Conversation tracking** - Maintain context across interactions

## Quick Start

### Test Slack integration
```bash
# Set your Slack token
export SLACK_BOT_TOKEN="xoxb-your-token"

dao-ai chat -c config/examples/02_tools/slack_integration.yaml
```

Example: *"Send a message to #general saying 'Hello from DAO AI!'"*

### Try vector search with reranking
```bash
dao-ai chat -c config/examples/02_tools/vector_search_with_reranking.yaml
```

Example: *"Find documentation about configuring agents"*

## Integration Patterns

### External APIs (Slack, JIRA)
- **Authentication**: Tokens stored in environment variables or Databricks Secrets
- **Tool definition**: Factory functions create tools from credentials
- **Usage**: Agent calls tools based on natural language requests

### Model Context Protocol (MCP)
- **Standardized interface**: Consistent pattern for external integrations
- **Server-based**: MCP servers expose tools to agents
- **UC Connections**: Secure credential management via Unity Catalog

### Vector Search & RAG
- **Semantic search**: Find relevant information using embeddings
- **Reranking**: Improve precision with FlashRank post-processing
- **Context injection**: Retrieved content added to agent prompts

## Prerequisites

### For Slack (`slack_integration.yaml`)
- Slack workspace with bot created
- Bot token with appropriate scopes
- Channel access for the bot

### For JIRA (`jira_integration.yaml`)
- JIRA instance URL
- API token or OAuth credentials
- Project permissions

### For MCP (`mcp_basic.yaml`, `mcp_with_uc_connection.yaml`)
- MCP server running and accessible
- For UC connections: Unity Catalog connection configured

### For Vector Search (`vector_search_with_reranking.yaml`)
- Databricks Vector Search index configured
- Embedding model endpoint
- FlashRank installed (for reranking)

### For Genie (`genie_with_conversation_id.yaml`)
- Genie space with tables
- Conversation tracking enabled

## Security Best Practices

🔒 **Never commit credentials** to configuration files

**Best practices:**
- Use environment variables for development
- Use Databricks Secrets for production
- Use Unity Catalog connections for enterprise deployments
- Rotate credentials regularly

**Example credential management:**
```yaml
variables:
  slack_token: &slack_token
    options:
      - env: SLACK_BOT_TOKEN          # Development
      - scope: secrets                 # Production
        secret: slack_bot_token
```

## Next Steps

After mastering tool integrations:

👉 **03_caching/** - Optimize tool calls with caching  
👉 **04_memory/** - Add conversation persistence  
👉 **05_quality_control/** - Add approval workflows for sensitive operations

## Troubleshooting

**"Authentication failed"**
- Verify credentials are set correctly
- Check token/API key has required permissions
- Ensure Databricks Secrets scope exists

**"Tool not found"**
- Verify tool factory function is correctly configured
- Check tool name matches agent configuration
- Review tool registration in logs

**"Vector search index not accessible"**
- Confirm index exists and is active
- Verify Unity Catalog permissions
- Check embedding model endpoint is serving

## Related Documentation

- [Tool Development Guide](../../../docs/contributing.md#adding-a-new-tool)
- [Unity Catalog Connections](../../../docs/configuration-reference.md)
- [MCP Documentation](https://modelcontextprotocol.io/)

