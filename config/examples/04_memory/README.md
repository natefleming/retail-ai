# 04. Memory

**Persistent state management for multi-turn conversations**

Add conversation memory to enable stateful, context-aware agents that remember past interactions.

## Examples

| File | Description | Storage Backend |
|------|-------------|-----------------|
| `conversation_summarization.yaml` | Long conversation summarization | PostgreSQL/Lakebase |

## What You'll Learn

- **Checkpointers** - Persist conversation state across sessions
- **Stores** - Key-value storage for user preferences
- **Summarization** - Handle long conversations by summarizing history
- **Backend options** - In-memory vs PostgreSQL vs Lakebase

## Memory Backends

### 1. In-Memory
- Fast but temporary (resets on restart)
- Good for: Development, testing

### 2. PostgreSQL
- External database, survives restarts
- Good for: Production with existing PostgreSQL

### 3. Lakebase
- Databricks-managed PostgreSQL with Unity Catalog
- Good for: Production in Databricks ecosystem

## Quick Start

```bash
# Ensure database is configured
export DATABASE_HOST="your-db-host"
export DATABASE_PASSWORD="your-password"

dao-ai chat -c config/examples/04_memory/conversation_summarization.yaml
```

Have a long conversation - notice older messages are summarized to stay within token limits.

## Conversation Summarization

When conversations exceed token limits:

1. **Detect**: Monitor message count and token usage
2. **Summarize**: LLM condenses old messages into summary
3. **Preserve**: Recent messages kept as-is
4. **Continue**: Conversation continues with reduced context

**Benefits:**
- ✅ Handle unlimited conversation length
- ✅ Maintain context without hitting token limits
- ✅ Reduce costs by summarizing old context

## Configuration

### Checkpointer (Conversation State)
```yaml
memory:
  checkpointer:
    database: *postgres_db        # Or Lakebase
    table_name: agent_checkpoints
```

### Store (User Preferences)
```yaml
memory:
  store:
    database: *postgres_db
    table_name: agent_store
    embedding_model: *embed_model  # For semantic search
```

### Summarization
```yaml
chat_history:
  max_tokens: 8000                    # Trigger summarization
  max_messages_before_summary: 20     # Or by message count
  summary_model: *claude_sonnet       # Model for summarization
```

## Prerequisites

- ✅ PostgreSQL or Databricks Lakebase instance
- ✅ Database credentials (OAuth for Lakebase)
- ✅ Tables will be auto-created on first use

## Architecture

```
┌─────────────┐
│   Agent     │
└──────┬──────┘
       │
       ├──→ Checkpointer (Conversation State)
       │    └─→ PostgreSQL/Lakebase Table
       │
       └──→ Store (User Preferences)
            └─→ PostgreSQL/Lakebase Table
```

## Next Steps

👉 **05_quality_control/** - Add safety and validation  
👉 **06_prompt_engineering/** - Optimize prompts for summarization

## Troubleshooting

**"Database connection failed"**
- Verify host and credentials
- Check network connectivity
- For Lakebase: Verify OAuth client ID/secret

**"Table doesn't exist"**
- Tables are auto-created - check permissions
- Verify schema exists in Unity Catalog

## Related Documentation

- [Memory Configuration](../../../docs/key-capabilities.md#conversation-memory--state)
- [Database Configuration](../../../docs/configuration-reference.md)

