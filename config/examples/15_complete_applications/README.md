# 15. Complete Applications

**Production-ready examples combining multiple features**

End-to-end configurations demonstrating best practices for real-world deployments.

## Architecture Overview

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#1565c0'}}}%%
flowchart TB
    subgraph Complete["🏗️ Complete Application Architecture"]
        subgraph UI["🖥️ User Interface"]
            Chat["💬 Chat UI"]
            API["🔌 REST API"]
        end
        
        subgraph Core["🤖 DAO AI Core"]
            subgraph Orchestration["🎭 Orchestration"]
                Supervisor["👔 Supervisor"]
                Swarm["🐝 Swarm"]
            end
            
            subgraph Agents["👷 Specialized Agents"]
                A1["💬 General"]
                A2["📋 Orders"]
                A3["🔧 DIY"]
                A4["🛒 Product"]
                A5["📦 Inventory"]
                A6["⚖️ Comparison"]
                A7["💡 Recommendation"]
            end
            
            subgraph Features["✨ Features"]
                F1["🧠 Memory"]
                F2["🔒 PII Protection"]
                F3["🛡️ Guardrails"]
                F4["⏸️ HITL"]
            end
        end
        
        subgraph Data["☁️ Databricks Platform"]
            LLM["🧠 LLM Endpoints"]
            VS["🔍 Vector Search"]
            Genie["🧞 Genie Rooms"]
            MCP["🔌 MCP Servers"]
            SQL["🗄️ SQL Warehouse"]
        end
    end

    UI --> Core
    Core --> Data

    style UI fill:#e3f2fd,stroke:#1565c0
    style Orchestration fill:#fff3e0,stroke:#e65100
    style Agents fill:#e8f5e9,stroke:#2e7d32
    style Features fill:#fce4ec,stroke:#c2185b
    style Data fill:#f3e5f5,stroke:#7b1fa2
```

## Examples

| File | Pattern | Description |
|------|---------|-------------|
| [`hardware_store.yaml`](./hardware_store.yaml) | 👔 Supervisor | Multi-agent supervisor with full features |
| [`hardware_store_swarm.yaml`](./hardware_store_swarm.yaml) | 🐝 Swarm | Swarm orchestration with handoffs |
| [`hardware_store_lakebase.yaml`](./hardware_store_lakebase.yaml) | 👔 Supervisor + 🧠 Lakebase | Supervisor with PostgreSQL memory persistence |
| [`hardware_store_instructed.yaml`](./hardware_store_instructed.yaml) | 🎯 Instructed | Hardware store with instructed retrieval |

## Hardware Store Supervisor Architecture

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart TB
    subgraph User["👤 Customer"]
        Query["Do you have Dewalt drills?<br/>What's the price and stock?"]
    end

    subgraph Supervisor["🎯 Supervisor Agent"]
        Router["Routing LLM<br/>Analyzes request<br/>Routes to specialist"]
    end

    subgraph Specialists["👷 Specialized Agents"]
        General["💬 General"]
        Orders["📋 Orders"]
        DIY["🔧 DIY"]
        Product["🛒 Product"]
        Inventory["📦 Inventory"]
        Comparison["⚖️ Comparison"]
        Recommendation["💡 Recommendation"]
    end

    subgraph Features["✨ Applied Features"]
        Memory["🧠 Memory"]
        Middleware["🔒 Middleware"]
        Guard["🛡️ Guardrails"]
    end

    Query --> Router
    Router --> General
    Router --> Orders
    Router --> DIY
    Router --> Product
    Router --> Inventory
    Router --> Comparison
    Router --> Recommendation
    Specialists --> Features

    style Supervisor fill:#fff3e0,stroke:#e65100
    style Specialists fill:#e8f5e9,stroke:#2e7d32
    style Features fill:#e3f2fd,stroke:#1565c0
```

## Hardware Store Swarm Architecture

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart TB
    subgraph User["👤 Customer"]
        Query["Compare Dewalt vs Milwaukee drills<br/>Check stock for both"]
    end

    subgraph Swarm["🐝 Agent Swarm"]
        General["💬 General<br/>Entry Point"]
        Orders["📋 Orders"]
        DIY["🔧 DIY"]
        Product["🛒 Product"]
        Inventory["📦 Inventory<br/>Terminal"]
        Comparison["⚖️ Comparison"]
        Recommendation["💡 Recommendation"]
    end

    subgraph Features["✨ Applied Features"]
        Memory["🧠 Memory"]
        Middleware["🔒 Swarm Middleware"]
    end

    Query --> General
    General -->|handoff| Orders
    General -->|handoff| DIY
    General -->|handoff| Product
    General -->|handoff| Inventory
    General -->|handoff| Comparison
    General -->|handoff| Recommendation
    DIY -->|handoff| Product
    DIY -->|handoff| Inventory
    DIY -->|handoff| Recommendation
    Swarm --> Features

    style General fill:#1565c0,stroke:#0d47a1,color:#fff
    style Inventory fill:#42BA91,stroke:#00875C
    style Swarm fill:#e8f5e9,stroke:#2e7d32
    style Features fill:#e3f2fd,stroke:#1565c0
```

**Swarm Handoff Configuration:**
- **General** (blue, entry point): Can handoff to any agent
- **DIY**: Can handoff to product, inventory, recommendation
- **Inventory** (green): Terminal agent with no outbound handoffs

## Feature Integration

```mermaid
%%{init: {'theme': 'base'}}%%
graph TB
    subgraph Integration["🔗 Feature Integration"]
        subgraph Memory["🧠 Memory"]
            M1["checkpointer: postgres"]
            M2["store: postgres"]
            M3["summarizer: *default_llm"]
        end
        
        subgraph Middleware["🔒 Middleware"]
            MW1["pii_detection: local"]
            MW2["pii_restoration: local"]
            MW3["logger: INFO"]
        end
        
        subgraph Guardrails["🛡️ Guardrails"]
            G1["tone_check"]
            G2["completeness_check"]
            G3["num_retries: 2"]
        end
        
        subgraph Tools["🔧 Tools"]
            T1["Genie MCP"]
            T2["Vector Search"]
            T3["SQL Warehouse"]
        end
    end

    style Memory fill:#e3f2fd,stroke:#1565c0
    style Middleware fill:#e8f5e9,stroke:#2e7d32
    style Guardrails fill:#fff3e0,stroke:#e65100
    style Tools fill:#fce4ec,stroke:#c2185b
```

## Production Checklist

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart TB
    subgraph Checklist["✅ Production Checklist"]
        subgraph Security["🔐 Security"]
            S1["☐ PII middleware enabled"]
            S2["☐ Secrets in Unity Catalog"]
            S3["☐ HITL for sensitive ops"]
        end
        
        subgraph Reliability["🔄 Reliability"]
            R1["☐ PostgreSQL memory"]
            R2["☐ Guardrails configured"]
            R3["☐ Error handling"]
        end
        
        subgraph Observability["📊 Observability"]
            O1["☐ MLflow tracing"]
            O2["☐ Logging middleware"]
            O3["☐ Metrics collection"]
        end
        
        subgraph Scale["📈 Scale"]
            SC1["☐ Load testing"]
            SC2["☐ Rate limiting"]
            SC3["☐ Model registration"]
        end
    end

    style Security fill:#ffebee,stroke:#c62828
    style Reliability fill:#e8f5e9,stroke:#2e7d32
    style Observability fill:#e3f2fd,stroke:#1565c0
    style Scale fill:#fff3e0,stroke:#e65100
```

## Configuration Structure

```yaml
# Complete Application Structure
schemas:
  retail_schema: &retail_schema           # Unity Catalog location

resources:
  llms:
    default_llm: &default_llm             # Primary LLM
    judge_llm: &judge_llm                 # Guardrail evaluator
  vector_stores:
    products_store: &products_store       # Semantic search
  genie_rooms:
    retail_genie: &retail_genie           # Natural language SQL

prompts:
  tone_prompt: &tone_prompt               # Guardrail prompts
  agent_prompts: ...                      # Agent instructions

middleware:
  pii_detection: &pii_detection           # Input protection
  pii_restoration: &pii_restoration       # Output restoration
  logger: &logger                         # Audit logging

guardrails:
  tone_check: &tone_check                 # Response quality
  completeness_check: &completeness_check

tools:
  genie_tool: &genie_tool                 # Data queries
  vector_tool: &vector_tool               # Semantic search
  handoff_tools: ...                      # For swarm pattern

agents:
  general_agent: &general_agent         # General store inquiries
  orders_agent: &orders_agent           # Order tracking
  diy_agent: &diy_agent                 # DIY advice & tutorials
  product_agent: &product_agent         # Product details
  inventory_agent: &inventory_agent     # Stock levels
  comparison_agent: &comparison_agent   # Product comparisons
  recommendation_agent: &recommendation_agent  # Product suggestions

app:
  name: hardware_store_assistant
  agents:
    - *general_agent
    - *orders_agent
    - *diy_agent
    - *product_agent
    - *inventory_agent
    - *comparison_agent
    - *recommendation_agent
  orchestration:
    supervisor:                           # or swarm:
      model: *default_llm
      prompt: "Route to appropriate agent..."
      middleware: [*pii_detection, *pii_restoration]
    memory:
      checkpointer:
        type: postgres
        connection_string: "{{secrets/scope/postgres}}"
```

## Quick Start

```bash
# Validate complete application
dao-ai validate -c config/examples/15_complete_applications/hardware_store.yaml

# Run in chat mode
dao-ai chat -c config/examples/15_complete_applications/hardware_store.yaml

# Visualize architecture
dao-ai graph -c config/examples/15_complete_applications/hardware_store.yaml -o architecture.png

# Deploy to Databricks
dao-ai bundle --deploy -c config/examples/15_complete_applications/hardware_store.yaml
```

## Deployment Options

```mermaid
%%{init: {'theme': 'base'}}%%
graph LR
    subgraph Deploy["🚀 Deployment Options"]
        subgraph Model["📦 MLflow Model"]
            M["dao-ai register<br/>━━━━━━━━━━━━━━━━<br/>Versioned artifact<br/>Model serving ready"]
        end
        
        subgraph App["🖥️ Databricks App"]
            A["dao-ai-builder<br/>━━━━━━━━━━━━━━━━<br/>Web UI<br/>REST API"]
        end
        
        subgraph Endpoint["⚡ Model Serving"]
            E["Serverless Endpoint<br/>━━━━━━━━━━━━━━━━<br/>Auto-scaling<br/>Low latency"]
        end
    end

    style Model fill:#e3f2fd,stroke:#1565c0
    style App fill:#e8f5e9,stroke:#2e7d32
    style Endpoint fill:#fff3e0,stroke:#e65100
```

## Best Practices

```mermaid
%%{init: {'theme': 'base'}}%%
graph TB
    subgraph Best["✅ Best Practices"]
        BP1["🔒 Use PII middleware in production"]
        BP2["🧠 PostgreSQL for multi-process memory"]
        BP3["🛡️ Guardrails for quality control"]
        BP4["📊 Enable MLflow tracing"]
        BP5["⏸️ HITL for write operations"]
        BP6["📝 Version prompts in MLflow Registry"]
    end

    style Best fill:#e8f5e9,stroke:#2e7d32
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Memory not persisting | Check PostgreSQL connection |
| Slow responses | Review guardrail num_retries |
| Wrong agent routing | Improve supervisor prompt |
| PII leaking | Verify middleware order |

## Related Documentation

- [Architecture Overview](../../../docs/architecture.md)
- [Configuration Reference](../../../docs/configuration-reference.md)
- [Deployment Guide](../../../docs/deployment.md)
