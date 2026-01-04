# 02. Tools

**Integrate with external services and Databricks capabilities**

This category demonstrates how to connect your agents to various tools and services. Each example focuses on a specific tool integration pattern.

## Examples

| File | Description | Prerequisites |
|------|-------------|---------------|
| `slack_integration.yaml` | Slack messaging integration | Slack workspace, bot token |
| `custom_mcp.yaml` | Custom MCP integration (JIRA example) | JIRA instance, API token |
| `managed_mcp.yaml` | Managed Model Context Protocol integration | MCP server |
| `external_mcp.yaml` | External MCP with Unity Catalog connections | Unity Catalog, MCP connection |
| `filtered_mcp.yaml` | MCP tool filtering examples | MCP server with multiple tools |
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

dao-ai chat -c config/examples/02_mcp/slack_integration.yaml
```

Example: *"Send a message to #general saying 'Hello from DAO AI!'"*


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

### MCP Tool Filtering
- **Security**: Block dangerous operations (drop, delete, execute DDL)
- **Performance**: Load only relevant tools to reduce context size
- **Access Control**: Filter tools based on user permissions
- **Cost Optimization**: Minimize token usage by reducing tool set

## MCP Tool Filtering

MCP servers can expose many tools. Use `include_tools` and `exclude_tools` to control which tools are loaded from the server.

### Why Filter Tools?

**Security**
- Block dangerous operations (drop_table, delete_data, execute_ddl)
- Prevent unauthorized access to sensitive functions
- Enforce principle of least privilege

**Performance**
- Reduce context window usage
- Faster agent responses with fewer tools to consider
- Lower token costs per request

**Usability**
- Agents make better decisions with focused tool sets
- Reduce tool confusion and selection errors
- Clearer audit trails of available operations

### Filtering Options

#### 1. Include Tools (Allowlist)
Load only specified tools - most secure approach:

```yaml
function:
  type: mcp
  sql: true
  include_tools:
    - execute_query      # Exact name
    - list_tables        # Exact name
    - "query_*"          # Pattern: all query tools
    - "get_*"            # Pattern: all getter tools
```

#### 2. Exclude Tools (Denylist)
Load all tools except specified ones - flexible approach:

```yaml
function:
  type: mcp
  sql: true
  exclude_tools:
    - "drop_*"           # Pattern: block all drop operations
    - "delete_*"         # Pattern: block all delete operations
    - execute_ddl        # Exact name
```

#### 3. Hybrid Filtering
Combine include and exclude for fine-grained control:

```yaml
function:
  type: mcp
  functions: *schema
  include_tools:
    - "query_*"          # Start with all query tools
    - "list_*"           # And all list tools
  exclude_tools:
    - "*_sensitive"      # But exclude sensitive ones
    - "*_admin"          # And admin functions
```

**Important:** `exclude_tools` always takes precedence over `include_tools`

### Pattern Syntax

Supports glob patterns (from Python's `fnmatch`):

| Pattern | Description | Examples |
|---------|-------------|----------|
| `*` | Matches any characters | `query_*` matches `query_sales`, `query_inventory` |
| `?` | Matches single character | `tool_?` matches `tool_a`, `tool_b` but not `tool_ab` |
| `[abc]` | Matches any char in set | `tool_[123]` matches `tool_1`, `tool_2`, `tool_3` |
| `[!abc]` | Matches any char NOT in set | `tool_[!abc]` matches `tool_d`, `tool_1` |

### Common Filtering Patterns

**Read-Only SQL Access**
```yaml
include_tools: ["query_*", "list_*", "describe_*", "show_*", "get_*"]
```

**Block Dangerous Operations**
```yaml
exclude_tools: ["drop_*", "delete_*", "truncate_*", "execute_ddl", "alter_*"]
```

**Development Mode (Safe Defaults)**
```yaml
exclude_tools: ["drop_*", "truncate_*", "execute_ddl"]
```

**Admin Functions Only**
```yaml
include_tools: ["admin_*", "manage_*", "configure_*"]
```

**No Sensitive Data Access**
```yaml
exclude_tools: ["*_sensitive", "*_secret", "*_password", "*_credential"]
```

### Examples in filtered_mcp.yaml

The `filtered_mcp.yaml` file demonstrates 6 different filtering strategies:

1. **sql_safe_tools**: Explicit allowlist of safe operations
2. **sql_readonly**: Block all write operations with patterns
3. **functions_filtered**: Hybrid filtering with include + exclude
4. **query_tools_only**: Pattern-based inclusion for consistency
5. **minimal_tools**: Maximum security with only 3 tools
6. **dev_tools**: Development mode blocking only critical operations

### Best Practices

1. **Start with allowlist (include_tools) for production** - safest approach
2. **Use denylist (exclude_tools) for development** - more flexible
3. **Test your filters** - verify correct tools are loaded via logging
4. **Document your reasoning** - why are you filtering these tools?
5. **Use patterns for consistency** - avoid maintaining long lists
6. **Review regularly** - as MCP servers change, update filters

### Testing Filters

```bash
# Test with filtered MCP configuration
dao-ai chat -c config/examples/02_mcp/filtered_mcp.yaml

# Try these commands to verify filtering:
# 1. "List all available tools" - see what's loaded
# 2. "Drop the users table" - should fail (tool not available)
# 3. "Query sales data" - should work (read operation)
```

The logs will show:
- Original tool count from MCP server
- Filtered tool count after include/exclude
- Final list of available tools

## Prerequisites

### For Slack (`slack_integration.yaml`)
- Slack workspace with bot created
- Bot token with appropriate scopes
- Channel access for the bot

### For Custom MCP (`custom_mcp.yaml`)
- JIRA instance URL
- API token or OAuth credentials
- Project permissions

### For MCP (`managed_mcp.yaml`, `external_mcp.yaml`)
- MCP server running and accessible
- For external MCP: Unity Catalog connection configured

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

👉 **04_genie/** - Optimize tool calls with caching  
👉 **05_memory/** - Add conversation persistence  
👉 **07_human_in_the_loop/** - Add approval workflows for sensitive operations

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

