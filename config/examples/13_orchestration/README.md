# 13. Orchestration

**Multi-agent coordination patterns**

Coordinate multiple specialized agents to solve complex problems using supervisor or swarm orchestration.

## Architecture Overview

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#1565c0'}}}%%
flowchart TB
    subgraph Patterns["🎭 Two Orchestration Patterns"]
        direction LR
        
        subgraph Supervisor["👔 Supervisor Pattern"]
            direction TB
            S["🎯 Supervisor LLM<br/><i>Analyzes & routes</i>"]
            S --> PA["🛒 Product Agent"]
            S --> IA["📦 Inventory Agent"]
            S --> GA["💬 General Agent"]
        end

        subgraph Swarm["🐝 Swarm Pattern"]
            direction TB
            P["🛒 Product"] <-->|"handoff"| I["📦 Inventory"]
            I <-->|"handoff"| C["⚖️ Comparison"]
            C <-->|"handoff"| P
        end
    end

    style Supervisor fill:#e3f2fd,stroke:#1565c0
    style Swarm fill:#e8f5e9,stroke:#2e7d32
```

## Examples

| File | Pattern | Description |
|------|---------|-------------|
| [`supervisor_pattern.yaml`](./supervisor_pattern.yaml) | 👔 Supervisor | Central LLM routes to specialized agents |
| [`swarm_pattern.yaml`](./swarm_pattern.yaml) | 🐝 Swarm | Agents use handoff tools to transfer |
| [`deterministic_handoff_pattern.yaml`](./deterministic_handoff_pattern.yaml) | 🔗 Deterministic | Pipeline-style predetermined routing |
| [`parallel_fan_out_pattern.yaml`](./parallel_fan_out_pattern.yaml) | 🌊 Parallel fan-out | Source dispatches concurrently to N sibling agents that converge on a shared join |
| [`deep_agent_pattern.yaml`](./deep_agent_pattern.yaml) | 🧠 Deep Agent | Single planner with todo, filesystem, shell, sub-agents (langgraph deepagents) |
| [`deep_agent_with_subagents.yaml`](./deep_agent_with_subagents.yaml) | 🧠 Deep Agent | Three sub_agent declaration forms: anchor, name, inline |
| [`deep_agent_with_skills.yaml`](./deep_agent_with_skills.yaml) | 🧠 Deep Agent | Local + volume-backed skills, AGENTS.md memory, permissions, HITL |

## Pattern Comparison

```mermaid
%%{init: {'theme': 'base'}}%%
graph TB
    subgraph Compare["📊 Pattern Comparison"]
        subgraph SupervisorFeatures["👔 Supervisor"]
            SF1["🎯 Centralized routing"]
            SF2["📋 Single prompt controls all"]
            SF3["🔄 Agents don't talk to each other"]
            SF4["⚡ Lower overhead"]
        end
        
        subgraph SwarmFeatures["🐝 Swarm"]
            WF1["🔀 Distributed decisions"]
            WF2["🛠️ Each agent has handoff tools"]
            WF3["💬 Agents collaborate directly"]
            WF4["🎨 More flexible workflows"]
        end
    end

    style SupervisorFeatures fill:#e3f2fd,stroke:#1565c0
    style SwarmFeatures fill:#e8f5e9,stroke:#2e7d32
```

| Aspect | 👔 Supervisor | 🐝 Swarm | 🧠 Deep Agent |
|--------|--------------|----------|---------------|
| **Control** | Centralized LLM | Distributed agents | Single planner with sub-agents |
| **Routing** | Supervisor prompt | Handoff tools per agent | `task` tool delegation |
| **Configuration** | `orchestration.supervisor` | `orchestration.swarm` | `orchestration.deep_agent` |
| **Built-in tools** | None | Handoff tools | Todo, filesystem, shell, task |
| **Skills** | n/a | n/a | First-class — local + UC volume |
| **Best For** | Clear categories | Fluid collaboration | Long-horizon planning |
| **Overhead** | Single router call | Per-agent logic | Heavier base middleware |

## 🧠 Deep Agent Pattern

A single planning agent equipped with deepagents' built-in tool suite —
`write_todos`, filesystem ops (`ls`, `read_file`, `write_file`, `edit_file`,
`glob`, `grep`), shell `execute`, and the `task` tool that delegates to
sub-agents. Skills (Markdown workflow files) and AGENTS.md memory are
first-class concepts that ship with the model artifact.

```yaml
orchestration:
  deep_agent:
    model: *default_llm                  # LLMModel anchor or "provider:name" string
    system_prompt: |
      You are a planner. Break complex tasks into todos before answering.
    tools: [*current_time]               # ToolModel anchors or string refs
    skills: [*research_skill]            # SkillModel refs (local or volume-backed)
    instruction_files: [skills/.../AGENTS.md]   # AGENTS.md instruction files loaded into prompt
    subagents:                           # three accepted forms
      - *product_specialist              #   1. AgentModel anchor (full carry-over)
      - inventory_specialist             #   2. name lookup in app.agents
      - name: math_helper                #   3. inline SubAgentModel dict
        description: ...
        system_prompt: ...
    permissions:                         # Filesystem permission rules
      - paths: ["/workspace/**"]
        mode: allow
        operations: [read, write]
    interrupt_on:                        # HITL on selected tools
      write_file: true
```

**When to use deep_agent:**
- Long-horizon, multi-step tasks where the agent needs to plan, scratchpad
  intermediate state, and iterate.
- Workflows where pre-authored Markdown skills capture institutional
  knowledge that should ship with the agent.
- Apps where you want a single API surface (one `CompiledStateGraph`) but
  still want delegation to specialists.

**Skills layout** — by convention, skills live in a `skills/` directory at
the project root, organized by vertical (mirrors `functions/`):

```
skills/
└── sporting_goods_store/
    ├── research/
    │   ├── SKILL.md       # what the skill does
    │   └── AGENTS.md      # persistent memory
    └── product-lookup/
        └── SKILL.md
```

Local skills are bundled via `code_paths` (Model Serving) and the app source
(Databricks Apps). Volume-backed skills (`/Volumes/...`) live on Unity Catalog
and are read at runtime — declare them as `SkillModel` with a `volume:` field
so deployment wires the read permission automatically.

---

## 👔 Supervisor Pattern

A central supervisor LLM analyzes requests and routes to specialized worker agents.

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart TB
    subgraph User["👤 User Request"]
        Q["Do you have the Dewalt drill in stock?"]
    end

    subgraph Supervisor["🎯 Supervisor Agent"]
        Analyze["Analyze request...<br/>━━━━━━━━━━━━━━━━<br/>🔍 Stock question detected<br/>📍 Route to: inventory_agent"]
    end

    subgraph Workers["👷 Specialized Workers"]
        direction LR
        Product["🛒 <b>product_agent</b><br/><i>Details, specs, features</i>"]
        Inventory["📦 <b>inventory_agent</b><br/><i>Stock, availability</i>"]
        General["💬 <b>general_agent</b><br/><i>Policies, hours</i>"]
    end

    Q --> Analyze
    Analyze -->|"Route"| Inventory
    Product -.->|"Not selected"| Analyze
    General -.->|"Not selected"| Analyze
    Inventory -->|"Response"| Q

    style Supervisor fill:#fff3e0,stroke:#e65100
    style Inventory fill:#e8f5e9,stroke:#2e7d32
```

### Configuration

```yaml
app:
  agents:
    - *product_agent      # 🛒 Product details
    - *inventory_agent    # 📦 Stock levels
    - *general_agent      # 💬 General inquiries

  orchestration:
    supervisor:
      model: *default_llm
      prompt: |
        You are a routing coordinator. Analyze requests and route to:
        
        - product_agent: Product details, features, specs, pricing
        - inventory_agent: Stock availability, inventory levels
        - general_agent: Store policies, hours, general questions
        
        Route to the single most appropriate agent.
```

### Sequence Diagram

```mermaid
%%{init: {'theme': 'base'}}%%
sequenceDiagram
    autonumber
    participant 👤 as User
    participant 🎯 as Supervisor
    participant 📦 as Inventory Agent
    participant ☁️ as Databricks

    👤->>🎯: Do you have Dewalt drills in stock?
    🎯->>🎯: Analyze: Stock question → inventory_agent
    🎯->>📦: Handle request
    📦->>☁️: Check inventory
    ☁️-->>📦: Stock data
    📦-->>🎯: "Yes, 15 units available"
    🎯-->>👤: We have 15 Dewalt drills in stock!
```

---

## 🐝 Swarm Pattern

Agents dynamically hand off conversations to each other using handoff tools.

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart TB
    subgraph User["👤 User Request"]
        Q["Tell me about Dewalt drill<br/>and check if you have it"]
    end

    subgraph Swarm["🐝 Agent Swarm"]
        direction TB
        
        subgraph Product["🛒 Product Agent"]
            PT["Tools:<br/>• search_products<br/>• <b>transfer_to_inventory</b><br/>• <b>transfer_to_comparison</b>"]
        end
        
        subgraph Inventory["📦 Inventory Agent"]
            IT["Tools:<br/>• check_inventory<br/>• <b>transfer_to_product</b><br/>• <b>transfer_to_comparison</b>"]
        end
        
        subgraph Comparison["⚖️ Comparison Agent"]
            CT["Tools:<br/>• search_products<br/>• <b>transfer_to_product</b><br/>• <b>transfer_to_inventory</b>"]
        end
    end

    Q --> Product
    Product -->|"Need stock info"| Inventory
    Inventory -->|"Need comparison"| Comparison
    Comparison -->|"Back to product"| Product

    style Swarm fill:#e8f5e9,stroke:#2e7d32
```

### Configuration

```yaml
tools:
  # 🔀 Handoff tools for agent-to-agent routing
  transfer_to_inventory: &transfer_to_inventory
    name: transfer_to_inventory
    function:
      type: factory
      name: dao_ai.tools.agent.create_handoff_tool
      args:
        agent_name: inventory_agent

  transfer_to_product: &transfer_to_product
    name: transfer_to_product
    function:
      type: factory
      name: dao_ai.tools.agent.create_handoff_tool
      args:
        agent_name: product_agent

agents:
  product_agent: &product_agent
    name: product_agent
    tools:
      - *search_products
      - *transfer_to_inventory     # Can hand off
      - *transfer_to_comparison    # Can hand off
    prompt: |
      You are a product specialist.
      
      When to hand off:
      - STOCK questions → use transfer_to_inventory
      - COMPARE requests → use transfer_to_comparison
    handoff_prompt: |
      Questions about product details of a SINGLE product.
```

### Sequence Diagram

```mermaid
%%{init: {'theme': 'base'}}%%
sequenceDiagram
    autonumber
    participant 👤 as User
    participant 🛒 as Product Agent
    participant 📦 as Inventory Agent

    👤->>🛒: Tell me about Dewalt drill and stock
    🛒->>🛒: Get product details...
    Note over 🛒: 18V, 1/2" chuck, 500 RPM
    🛒->>🛒: Need stock info → handoff
    🛒->>📦: transfer_to_inventory()
    Note over 📦: Context preserved
    📦->>📦: Check inventory...
    Note over 📦: 15 units available
    📦-->>👤: The Dewalt 18V drill has 1/2" chuck,<br/>500 RPM, and we have 15 in stock!
```

---

## 🔗 Deterministic Handoff Pattern

Agents always transfer control to a predetermined next agent after completing their turn, creating a pipeline-style workflow. Deterministic handoffs can be combined with agentic (tool-based) handoffs.

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart LR
    subgraph Pipeline["🔗 Deterministic Pipeline"]
        direction LR
        T["🏷️ Triage Agent"]
        R["🔧 Resolution Agent"]
        S["📝 Summary Agent"]
        E["⚠️ Escalation Agent"]

        T -->|"deterministic"| R
        R -->|"deterministic"| S
        R -.->|"agentic (optional)"| E
    end

    style Pipeline fill:#fff3e0,stroke:#e65100
```

### Configuration

Use `HandoffRouteModel` with `is_deterministic: true` to declare deterministic routes:

```yaml
orchestration:
  swarm:
    default_agent: triage_agent
    handoffs:
      triage_agent:
        - agent: resolution_agent
          is_deterministic: true       # always hand off here
      resolution_agent:
        - agent: summary_agent
          is_deterministic: true       # always hand off here
        - escalation_agent             # agentic: LLM decides via tool
```

### Sequence Diagram

```mermaid
%%{init: {'theme': 'base'}}%%
sequenceDiagram
    autonumber
    participant User
    participant Triage as Triage Agent
    participant Resolution as Resolution Agent
    participant Summary as Summary Agent

    User->>Triage: I was charged twice
    Triage->>Triage: Classify: billing issue
    Note over Triage,Resolution: Deterministic handoff (no tool call)
    Triage->>Resolution: Classified billing issue
    Resolution->>Resolution: Investigate and resolve
    Note over Resolution,Summary: Deterministic handoff (no tool call)
    Resolution->>Summary: Issue resolved
    Summary-->>User: Summary: duplicate charge refunded
```

---

## 🌊 Parallel Fan-Out Pattern

A source agent invokes multiple sibling agents concurrently in a single LLM
turn, and all siblings converge on a shared **join** agent that synthesizes
their outputs into one response. A cohort is declared as a single handoff
entry with `agents:` (the siblings) and `join:` (the shared reducer).

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart LR
    subgraph FanOut["🌊 Parallel Fan-Out"]
        direction LR
        T["🏷️ Triage"]
        P["💲 Pricing"]
        I["📦 Inventory"]
        Y["📜 Policy"]
        S["📝 Synthesizer<br/>(join)"]

        T -->|"parallel"| P
        T -->|"parallel"| I
        T -->|"parallel"| Y
        P -->|"fan-in"| S
        I -->|"fan-in"| S
        Y -->|"fan-in"| S
    end

    style FanOut fill:#f3e5f5,stroke:#6a1b9a
```

### Configuration

```yaml
orchestration:
  swarm:
    default_agent: triage_agent
    handoffs:
      triage_agent:
      - agents: [pricing_agent, inventory_agent, policy_agent]
        join: synthesizer_agent      # shared join for the cohort
      pricing_agent: []
      inventory_agent: []
      policy_agent: []
      synthesizer_agent: []
```

You can also mix a cohort with regular single-target handoffs on the same source:

```yaml
handoffs:
  triage_agent:
  - agents: [pricing_agent, inventory_agent, policy_agent]
    join: synthesizer_agent
  - escalation_agent                 # regular agentic peer
  - agent: emergency_agent
    is_deterministic: true           # single-target deterministic peer
```

### How it works

- The cohort entry produces one per-sibling parallel handoff tool
  (`handoff_to_pricing_agent`, etc.). The source LLM invokes multiple of
  these in a single turn (Claude & GPT support parallel tool calls
  natively).
- LangGraph runs the targeted siblings in the **same superstep** —
  concurrent execution, not serialized.
- Each sibling has a static edge to the shared join. Because they share
  one join, LangGraph runs the join **exactly once** after all fired
  siblings complete.

### Configuration rules (validated at load time)

- A cohort entry must set both `agents` (list of ≥ 2 distinct siblings)
  and `join`. `agent` (singular) and `agents` are mutually exclusive on
  the same entry.
- `is_deterministic` is not meaningful on a cohort entry — the join is
  always reached deterministically after fan-in.
- The join must not also appear in `agents` (no self-edge).
- A sibling cannot belong to two cohorts with different joins.
- A sibling cannot itself be the source of another cohort (nested fan-out
  is out of scope).
- Cycles containing any parallel or deterministic edge are rejected.

### Sequence Diagram

```mermaid
%%{init: {'theme': 'base'}}%%
sequenceDiagram
    autonumber
    participant User
    participant Triage as Triage
    participant Pricing
    participant Inventory
    participant Policy
    participant Join as Synthesizer

    User->>Triage: Is the drill in stock, what's the price, and what's the return policy?
    Triage->>Triage: Fan-out: invoke 3 handoff tools in one turn
    par concurrent siblings
        Triage->>Pricing: parallel handoff
        and
        Triage->>Inventory: parallel handoff
        and
        Triage->>Policy: parallel handoff
    end
    Pricing-->>Join: pricing answer
    Inventory-->>Join: stock answer
    Policy-->>Join: policy answer
    Join-->>User: One synthesized response
```

---

## When to Use Each Pattern

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart TB
    subgraph Decision["🤔 Which Pattern?"]
        Q1{"Clear task<br/>categories?"}
        Q2{"Need mid-conversation<br/>collaboration?"}
        Q3{"Simple routing<br/>logic?"}
    end

    subgraph Answers["📋 Recommendation"]
        Sup["👔 <b>Supervisor</b><br/>━━━━━━━━━━━━━━━━<br/>• Centralized control<br/>• Clear categories<br/>• Lower complexity"]
        Swa["🐝 <b>Swarm</b><br/>━━━━━━━━━━━━━━━━<br/>• Fluid handoffs<br/>• Agent autonomy<br/>• Complex workflows"]
    end

    Q1 -->|"Yes"| Q3
    Q1 -->|"No"| Q2
    Q2 -->|"Yes"| Swa
    Q2 -->|"No"| Q3
    Q3 -->|"Yes"| Sup
    Q3 -->|"No"| Swa

    style Sup fill:#e3f2fd,stroke:#1565c0
    style Swa fill:#e8f5e9,stroke:#2e7d32
```

## Quick Start

```bash
# Validate patterns
dao-ai validate -c config/examples/13_orchestration/supervisor_pattern.yaml
dao-ai validate -c config/examples/13_orchestration/swarm_pattern.yaml
dao-ai validate -c config/examples/13_orchestration/deep_agent_with_skills.yaml

# Chat with supervisor or deep_agent
dao-ai chat -c config/examples/13_orchestration/supervisor_pattern.yaml
dao-ai chat -c config/examples/13_orchestration/deep_agent_pattern.yaml

# Visualize architecture
dao-ai graph -c config/examples/13_orchestration/supervisor_pattern.yaml -o graph.png

# Generate a Databricks Asset Bundle for a deep_agent app
dao-ai generate-bundle -c config/examples/13_orchestration/deep_agent_with_skills.yaml
```

## Prerequisites

- Understanding of single-agent patterns
- Multiple specialized agents defined
- Clear task decomposition strategy
- For swarm: handoff tools configured

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Wrong agent selected | Improve supervisor/handoff prompts |
| Infinite handoff loops | Add termination conditions |
| Context lost | Configure shared memory |

## Next Steps

- **12_middleware/** - Add cross-cutting concerns
- **15_complete_applications/** - See orchestration in production

## Related Documentation

- [Orchestration Architecture](../../../docs/architecture.md)
- [Multi-Agent Patterns](../../../docs/key-capabilities.md)
