# 13. Orchestration

**Multi-agent coordination patterns**

Coordinate multiple specialized agents to solve complex problems. This category will contain supervisor and swarm orchestration patterns.

## Examples

| File | Description | Pattern |
|------|-------------|---------|
| [`supervisor_pattern.yaml`](./supervisor_pattern.yaml) | Central coordinator routes to specialized agents | Hierarchical delegation |
| [`swarm_pattern.yaml`](./swarm_pattern.yaml) | Agents dynamically hand off to each other | Peer-to-peer handoffs |

## What You'll Learn

- **Supervisor pattern** - Central coordinator delegates to specialized agents
- **Swarm pattern** - Peer-to-peer agent handoffs
- **Hierarchical coordination** - Multi-level agent structures
- **Message routing** - Control information flow between agents
- **State sharing** - Coordinate agent memory and context

## Orchestration Patterns

### Supervisor Pattern
```
         ┌──────────────┐
         │  Supervisor  │
         └───────┬──────┘
                 │
     ┌───────────┼───────────┐
     │           │           │
┌────▼────┐ ┌───▼────┐ ┌───▼────┐
│ Agent 1 │ │Agent 2 │ │ Agent3 │
└─────────┘ └────────┘ └────────┘
```
- **Top-down control**: Supervisor delegates to specialists
- **Clear hierarchy**: Single point of coordination
- **Best for**: Well-defined tasks with specialized agents

### Swarm Pattern
```
┌─────────┐  handoff  ┌─────────┐
│ Agent 1 │◄─────────►│ Agent 2 │
└────┬────┘           └────┬────┘
     │     handoff         │
     └────────►┌─────────┐◄┘
              │ Agent 3 │
              └─────────┘
```
- **Peer-to-peer**: Agents hand off to each other
- **Dynamic routing**: Agents decide next steps
- **Best for**: Fluid workflows, collaborative problem-solving

## Prerequisites

- ✅ Understanding of single-agent patterns (`01_getting_started`)
- ✅ Multiple specialized agents defined
- ✅ Clear task decomposition strategy
- ✅ Shared memory/state infrastructure (`05_memory`)

## When to Use Orchestration

**Use orchestration when:**
- ✅ Tasks require multiple specialized skills
- ✅ Subtasks can be parallelized
- ✅ Different agents have different tool access
- ✅ Complexity exceeds single-agent capabilities

**Stick with single agent when:**
- ❌ Task is straightforward
- ❌ One set of tools suffices
- ❌ Coordination overhead > benefits

## Configuration Pattern

### Supervisor Example

```yaml
app:
  name: my_supervisor_app
  agents:
    - product_agent
    - inventory_agent
    - general_agent
  
  orchestration:
    supervisor:
      model: *default_llm
      prompt: |
        Route requests to the appropriate specialist:
        - product_agent: Product details and specifications
        - inventory_agent: Stock and availability
        - general_agent: General inquiries
```

See [`supervisor_pattern.yaml`](./supervisor_pattern.yaml) for a complete working example.

### Swarm Example

```yaml
app:
  name: my_swarm_app
  agents:
    - product_agent   # Has handoff tools to other agents
    - inventory_agent # Can transfer to product or comparison
    - comparison_agent # Can transfer back to product or inventory

tools:
  transfer_to_inventory:
    function: dao_ai.tools.agent.create_handoff_tool
    args:
      agent_name: inventory_agent
```

See [`swarm_pattern.yaml`](./swarm_pattern.yaml) for a complete working example.

## Design Principles

### 1. Clear Responsibilities
Each agent should have a well-defined role and expertise.

### 2. Minimal State Sharing
Share only necessary context between agents.

### 3. Explicit Handoffs
Make agent transitions clear and trackable.

### 4. Error Handling
Plan for agent failures and deadlocks.

### 5. Observable
Log all agent interactions for debugging.

## Next Steps

👉 **11_complete_applications/** - See orchestration in production systems

## Try These Examples

```bash
# Validate the supervisor pattern
dao-ai validate -c config/examples/13_orchestration/supervisor_pattern.yaml

# Validate the swarm pattern  
dao-ai validate -c config/examples/13_orchestration/swarm_pattern.yaml

# Visualize the supervisor architecture
dao-ai graph -c config/examples/13_orchestration/supervisor_pattern.yaml -o supervisor_graph.png

# Visualize the swarm architecture
dao-ai graph -c config/examples/13_orchestration/swarm_pattern.yaml -o swarm_graph.png
```

## Real-World Examples

For production-ready orchestration examples, see:
- [`config/hardware_store/supervisor.yaml`](../../hardware_store/supervisor.yaml) - Full supervisor implementation
- [`config/hardware_store/swarm.yaml`](../../hardware_store/swarm.yaml) - Full swarm implementation

## Related Documentation

- [Orchestration Architecture](../../../docs/architecture.md)
- [Multi-Agent Patterns](../../../docs/key-capabilities.md)

## Contribute

Have an orchestration pattern to share? We'd love to see it! Check out the [Contributing Guide](../../../docs/contributing.md).

