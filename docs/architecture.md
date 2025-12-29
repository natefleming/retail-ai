# Architecture

## How It Works (Simple Explanation)

Think of DAO as a three-layer cake:

**1. Your Configuration (Top Layer)** 🎂  
You write a YAML file describing what you want: which AI models, what data to access, what tools agents can use.

**2. DAO Framework (Middle Layer)** 🔧  
DAO reads your YAML and automatically wires everything together using LangGraph (a workflow engine for AI agents).

**3. Databricks Platform (Bottom Layer)** ☁️  
Your deployed agent runs on Databricks, accessing Unity Catalog data, calling AI models, and using other Databricks services.

## Technical Architecture Diagram

For developers and architects, here's the detailed view:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            YAML Configuration                               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐   │
│  │ Schemas │  │ Resources│  │  Tools  │  │ Agents  │  │  Orchestration  │   │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DAO Framework (Python)                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    │
│  │   Config    │  │    Graph    │  │    Nodes    │  │   Tool Factory  │    │
│  │  Loader     │  │   Builder   │  │   Factory   │  │                 │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LangGraph Runtime                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Compiled State Graph                             │   │
│  │   ┌─────────┐    ┌─────────────┐    ┌─────────────────────────┐    │   │
│  │   │ Message │───▶│ Supervisor/ │───▶│    Specialized Agents    │    │   │
│  │   │  Hook   │    │   Swarm     │    │  (Product, Orders, DIY)  │    │   │
│  │   └─────────┘    └─────────────┘    └─────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Databricks Platform                                  │
│  ┌─────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────┐  ┌─────────┐ │
│  │  Model  │  │    Unity    │  │   Vector    │  │  Genie   │  │ MLflow  │ │
│  │ Serving │  │   Catalog   │  │   Search    │  │  Spaces  │  │         │ │
│  └─────────┘  └─────────────┘  └─────────────┘  └──────────┘  └─────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## System-Level Data Flow

This diagram shows how a deployed DAO agent integrates with Databricks services and external systems:

```
                                    ┌──────────────────┐
                                    │      User        │
                                    └────────┬─────────┘
                                             │ HTTP Request
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       Databricks Model Serving                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │                         DAO Agent Runtime                               │    │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │    │
│  │  │                   Agent Orchestration Layer                       │  │    │
│  │  │            (Supervisor / Swarm with multiple agents)              │  │    │
│  │  └────┬──────────────────┬──────────────────┬────────────────┬──────┘  │    │
│  │       │                  │                  │                │         │    │
│  │       │ Tool Call        │ Tool Call        │ Tool Call      │ Store   │    │
│  │       ▼                  ▼                  ▼                ▼         │    │
│  │  ┌──────────┐   ┌─────────────┐   ┌──────────────┐   ┌──────────┐   │    │
│  │  │  Genie   │   │   DBSQL     │   │    Agent     │   │   MCP    │   │    │
│  │  │   Tool   │   │    Tool     │   │  Endpoint    │   │   Tool   │   │    │
│  │  │          │   │             │   │     Tool     │   │          │   │    │
│  │  └────┬─────┘   └──────┬──────┘   └──────┬───────┘   └────┬─────┘   │    │
│  └───────┼────────────────┼─────────────────┼────────────────┼─────────┘    │
└──────────┼────────────────┼─────────────────┼────────────────┼──────────────┘
           │                │                 │                │
           │                │                 │                │
    ┌──────▼─────┐          │          ┌──────▼──────┐        │
    │   Genie    │          │          │   Another   │        │
    │  Service   │          │          │   Agent     │        │
    │            │          │          │  Endpoint   │        │
    └──────┬─────┘          │          └─────────────┘        │
           │                │                                 │
           │ NL → SQL       │ Direct SQL Query                │
           ▼                ▼                                 ▼
    ┌────────────────────────────────────┐           ┌───────────────┐
    │    Databricks SQL / Warehouse      │           │  MCP Server   │
    │  ┌──────────────────────────────┐  │           │  (GitHub,     │
    │  │      Unity Catalog           │  │           │   Slack,      │
    │  │  • Tables & Views            │  │           │   Custom)     │
    │  │  • Functions                 │  │           └───────────────┘
    │  │  • Permissions               │  │
    │  └──────────────────────────────┘  │
    └────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                           State Persistence Layer                               │
│  ┌────────────────────────────────────────────────────────────────────────┐     │
│  │                         Lakebase (Postgres )                           │     │
│  │  ┌──────────────────────────┐     ┌──────────────────────────────┐     │     │
│  │  │  Conversation Checkpoints│     │   User Preferences Store     │     │     │
│  │  │  • Thread state          │     │   • User settings            │     │     │
│  │  │  • Message history       │     │   • Semantic search          │     │     │
│  │  │  • Agent context         │     │   • Key-value storage        │     │     │
│  │  └──────────────────────────┘     └──────────────────────────────┘     │     │
│  │                    ▲                            ▲                       │   │
│  └────────────────────┼────────────────────────────┼────────────────────────┘   │
│                       │                            │                            │
│                       └────────────────────────────┘                            │
│                              Persisted by Agent                                 │
│                       (Unity Catalog governed storage)                          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Explanation

**1. User Interaction**
- User sends a request to the DAO agent via Databricks Model Serving endpoint
- Request includes message, conversation ID, and user context

**2. Agent Processing**
- Agent orchestration layer (Supervisor or Swarm) processes the request
- Determines which tools to invoke based on the user's question
- Multiple agents may collaborate to answer complex queries

**3. Tool Integration Patterns**

**A. Genie Tool → Genie Service → DBSQL**
- Agent invokes Genie tool with natural language question
- Genie service translates NL to SQL query
- Executes against Databricks SQL / Unity Catalog
- Returns structured data results
- *Use case:* "What products are low on stock?"

**B. Direct DBSQL Tool**
- Agent calls SQL warehouse directly with pre-defined SQL
- Executes Unity Catalog functions or queries
- Returns data from governed tables
- *Use case:* Execute a stored procedure or predefined query

**C. Agent Endpoint Tool**
- Agent calls another deployed agent endpoint
- Enables composition and specialization
- Other agent may use different tools/models
- *Use case:* Call a specialized HR agent from a general assistant

**D. MCP Tool**
- Agent communicates with external MCP server
- Supports GitHub, Slack, custom APIs
- Extends agent capabilities beyond Databricks
- *Use case:* Create GitHub issue, send Slack message

**4. State Persistence**
- Conversation state saved to Lakebase checkpointer (Delta table)
- User preferences stored in Lakebase store (Unity Catalog governed)
- Enables multi-turn conversations and personalization
- Survives agent restarts and scales across instances

**5. Security & Governance**
- **On-Behalf-Of User**: Requests execute with caller's permissions
- **Unity Catalog**: Row/column-level security enforced
- **Audit Logs**: All data access tracked per user
- **Isolation**: Conversation state partitioned by user/thread

## Orchestration Patterns

When you have multiple specialized agents, you need to decide how they work together. DAO supports two patterns:

**Think of it like a company:**
- **Supervisor Pattern** = Traditional hierarchy (manager assigns tasks to specialists)
- **Swarm Pattern** = Collaborative team (specialists hand off work to each other)

DAO supports both approaches for multi-agent coordination:

### 1. Supervisor Pattern

**Best for:** Clear separation of responsibilities with centralized control

A central "supervisor" agent reads each user request and decides which specialist agent should handle it. Think of it like a call center manager routing calls to different departments.

**Example use case:** Hardware store assistant
- User asks about product availability → Routes to **Product Agent**
- User asks about order status → Routes to **Orders Agent**  
- User asks for DIY advice → Routes to **DIY Agent**

**Configuration:**

```yaml
orchestration:
  supervisor:
    model: *router_llm
    prompt: |
      Route queries to the appropriate specialist agent based on the content.
```

```
                    ┌─────────────┐
                    │  Supervisor │
                    └──────┬──────┘
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │   Product   │ │   Orders    │ │     DIY     │
    │    Agent    │ │    Agent    │ │    Agent    │
    └─────────────┘ └─────────────┘ └─────────────┘
```

### 2. Swarm Pattern

**Best for:** Complex, multi-step workflows where agents need to collaborate

Agents work more autonomously and can directly hand off tasks to each other. Think of it like a team of specialists who know when to involve their colleagues.

**Example use case:** Complex customer inquiry
1. User: *"I need a drill for a home project, do we have any in stock, and can you suggest how to use it?"*
2. **Product Agent** checks inventory → Finds drill in stock → Hands off to **DIY Agent**
3. **DIY Agent** provides usage instructions → Done

No central supervisor needed — agents decide collaboratively.

**Configuration:**

```yaml
orchestration:
  swarm:
    model: *default_llm
    default_agent: *general_agent    # Where to start
    handoffs:
      product_agent: [orders_agent, diy_agent]  # Product agent can hand off to these
      orders_agent: [product_agent]             # Orders agent can hand off to Product
```

```
    ┌─────────────┐     handoff     ┌─────────────┐
    │   Product   │◄───────────────▶│   Orders    │
    │    Agent    │                 │    Agent    │
    └──────┬──────┘                 └──────┬──────┘
           │          handoff              │
           └──────────────┬────────────────┘
                          ▼
                   ┌─────────────┐
                   │     DIY     │
                   │    Agent    │
                   └─────────────┘
```

---

## Navigation

- [← Previous: Why DAO?](why-dao.md)
- [↑ Back to Documentation Index](../README.md#-documentation)
- [Next: Key Capabilities →](key-capabilities.md)
