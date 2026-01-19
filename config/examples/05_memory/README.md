# 05. Memory

**Conversation persistence across sessions**

Store and retrieve conversation history to maintain context across user sessions.

## Architecture Overview

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#1565c0'}}}%%
flowchart TB
    subgraph Session1["💬 Session 1"]
        U1["👤 User: My name is Alice"]
        A1["🤖 Agent: Nice to meet you, Alice!"]
    end

    subgraph Memory["🧠 Memory System"]
        subgraph Backends["Storage Backend"]
            direction LR
            PG["🐘 PostgreSQL<br/><i>Production</i>"]
            SQLite["📁 SQLite<br/><i>Development</i>"]
            InMem["💾 In-Memory<br/><i>Testing</i>"]
        end
        
        subgraph Data["Stored Data"]
            Thread["<b>thread_id:</b> user_123<br/><b>messages:</b> [...]<br/><b>summary:</b> User is Alice..."]
        end
    end

    subgraph Session2["💬 Session 2 (Later)"]
        U2["👤 User: What's my name?"]
        A2["🤖 Agent: Your name is Alice!"]
    end

    Session1 -->|"Store"| Memory
    Memory -->|"Retrieve"| Session2

    style Session1 fill:#e3f2fd,stroke:#1565c0
    style Memory fill:#e8f5e9,stroke:#2e7d32
    style Session2 fill:#e3f2fd,stroke:#1565c0
```

## Examples

| File | Backend | Description |
|------|---------|-------------|
| [`memory_sqlite.yaml`](./memory_sqlite.yaml) | 📁 SQLite | Local file-based persistence |
| [`memory_postgres.yaml`](./memory_postgres.yaml) | 🐘 PostgreSQL | Production-ready persistence |

## Memory Components

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart TB
    subgraph Memory["🧠 Memory Configuration"]
        subgraph Checkpoint["📍 Checkpointer"]
            direction TB
            CP["<b>checkpointer:</b><br/>━━━━━━━━━━━━━━━━<br/>type: postgres | sqlite<br/>connection_string: ...<br/><br/><i>Stores conversation messages</i>"]
        end
        
        subgraph Store["📦 Store (Optional)"]
            direction TB
            ST["<b>store:</b><br/>━━━━━━━━━━━━━━━━<br/>type: postgres | sqlite<br/>connection_string: ...<br/><br/><i>Stores metadata & summaries</i>"]
        end
        
        subgraph Summarizer["📝 Summarizer (Optional)"]
            direction TB
            SU["<b>summarizer:</b><br/>━━━━━━━━━━━━━━━━<br/>model: *default_llm<br/>max_messages: 100<br/><br/><i>Summarizes long conversations</i>"]
        end
    end

    Checkpoint --> Store
    Store --> Summarizer

    style Checkpoint fill:#e3f2fd,stroke:#1565c0
    style Store fill:#e8f5e9,stroke:#2e7d32
    style Summarizer fill:#fff3e0,stroke:#e65100
```

## Backend Comparison

```mermaid
%%{init: {'theme': 'base'}}%%
graph TB
    subgraph Backends["📊 Backend Comparison"]
        subgraph SQLite["📁 SQLite"]
            S1["✅ Zero setup"]
            S2["✅ Local development"]
            S3["⚠️ Single process"]
            S4["⚠️ Not for production"]
        end
        
        subgraph Postgres["🐘 PostgreSQL"]
            P1["✅ Production-ready"]
            P2["✅ Multi-process safe"]
            P3["✅ Scalable"]
            P4["⚠️ Requires setup"]
        end
        
        subgraph InMemory["💾 In-Memory"]
            I1["✅ Fastest"]
            I2["✅ Testing only"]
            I3["⚠️ Lost on restart"]
        end
    end

    style SQLite fill:#e3f2fd,stroke:#1565c0
    style Postgres fill:#e8f5e9,stroke:#2e7d32
    style InMemory fill:#fff3e0,stroke:#e65100
```

## SQLite Configuration

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart LR
    subgraph Config["📄 memory_sqlite.yaml"]
        YAML["orchestration:<br/>  memory:<br/>    checkpointer:<br/>      type: sqlite<br/>      connection_string:<br/>        sqlite:///memory.db"]
    end

    subgraph File["📁 Local File"]
        DB["memory.db<br/>━━━━━━━━━━━━━━━━<br/>📊 messages table<br/>📊 checkpoints table"]
    end

    Config --> File

    style Config fill:#e3f2fd,stroke:#1565c0
    style File fill:#e8f5e9,stroke:#2e7d32
```

```yaml
app:
  orchestration:
    swarm: true
    memory:
      checkpointer:
        type: sqlite
        connection_string: "sqlite:///memory.db"
      store:
        type: sqlite
        connection_string: "sqlite:///store.db"
```

## PostgreSQL Configuration

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart LR
    subgraph Config["📄 memory_postgres.yaml"]
        YAML["orchestration:<br/>  memory:<br/>    checkpointer:<br/>      type: postgres<br/>      connection_string:<br/>        postgresql://..."]
    end

    subgraph UC["🔐 Unity Catalog Secret"]
        Secret["postgres_conn_string<br/>━━━━━━━━━━━━━━━━<br/>postgresql://user:pass@host/db"]
    end

    subgraph DB["🐘 PostgreSQL"]
        Tables["📊 Tables<br/>━━━━━━━━━━━━━━━━<br/>checkpoints<br/>messages<br/>metadata"]
    end

    Config --> UC
    UC --> DB

    style Config fill:#e3f2fd,stroke:#1565c0
    style UC fill:#fff3e0,stroke:#e65100
    style DB fill:#e8f5e9,stroke:#2e7d32
```

```yaml
app:
  orchestration:
    swarm: true
    memory:
      checkpointer:
        type: postgres
        connection_string: "{{secrets/scope/postgres_conn_string}}"
      store:
        type: postgres
        connection_string: "{{secrets/scope/postgres_conn_string}}"
      summarizer:
        model: *default_llm
        max_messages: 100
```

## Conversation Summarization

```mermaid
%%{init: {'theme': 'base'}}%%
sequenceDiagram
    autonumber
    participant 💬 as Conversation
    participant 🧠 as Memory
    participant 📝 as Summarizer LLM

    💬->>🧠: Message 1...100
    🧠->>🧠: max_messages reached!
    🧠->>📝: Summarize first 50 messages
    📝-->>🧠: "User Alice discussed power tools..."
    🧠->>🧠: Store summary, keep recent 50
    Note over 🧠: Context preserved, size reduced
```

```yaml
memory:
  summarizer:
    model: *default_llm     # LLM for summarization
    max_messages: 100       # Trigger summarization at 100 messages
```

## Quick Start

```bash
# SQLite (development)
dao-ai chat -c config/examples/05_memory/memory_sqlite.yaml \
  --thread-id my_session

# PostgreSQL (production)
dao-ai chat -c config/examples/05_memory/memory_postgres.yaml \
  --thread-id user_123
```

**Test memory:**
```
> My name is Alice
Nice to meet you, Alice!

> [quit and restart]

> What's my name?
Your name is Alice!
```

## Thread ID Usage

```mermaid
%%{init: {'theme': 'base'}}%%
graph TB
    subgraph ThreadIDs["🔑 Thread ID Patterns"]
        TID1["<b>user_123</b><br/><i>Per-user history</i>"]
        TID2["<b>session_abc</b><br/><i>Per-session history</i>"]
        TID3["<b>project_xyz</b><br/><i>Per-project history</i>"]
    end

    style ThreadIDs fill:#e3f2fd,stroke:#1565c0
```

## Prerequisites

| Backend | Requirements |
|---------|--------------|
| 📁 SQLite | None (creates file) |
| 🐘 PostgreSQL | PostgreSQL server, connection string |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Memory not persisting | Check connection_string, file permissions |
| PostgreSQL connection failed | Verify host, port, credentials |
| Context lost | Ensure same thread_id across sessions |

## Next Steps

- **13_orchestration/** - Combine with multi-agent patterns
- **07_human_in_the_loop/** - Stateful approval workflows
- **15_complete_applications/** - Production memory patterns

## Related Documentation

- [Memory Configuration](../../../docs/key-capabilities.md#memory)
- [Orchestration](../13_orchestration/README.md)
