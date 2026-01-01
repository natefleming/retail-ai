# 10. Agent Integrations

**Integrate with external agent platforms like Agent Bricks and Kasal**

Learn how to use agent endpoint tools to delegate tasks to specialized external agents, creating powerful multi-agent systems with best-in-class specialist capabilities.

---

## 📋 Overview

Agent endpoint tools allow you to call external agent systems (like Agent Bricks or Kasal) as tools within your DAO-AI agents. This enables:

- **Delegation to Specialists**: Route specific tasks to purpose-built external agents
- **Multi-Agent Orchestration**: Coordinate multiple agent systems in a single workflow
- **Enterprise Integration**: Leverage existing agent infrastructure and investments
- **Governance & Compliance**: Use specialized agents with built-in compliance features

### When to Use Agent Integrations

✅ **Use when you need:**
- Domain-specific expertise (finance, legal, customer support)
- Compliance-certified agent responses
- Integration with existing agent platforms
- Coordinated multi-agent workflows
- Separation of concerns across agent systems

❌ **Not needed when:**
- Simple single-agent tasks suffice
- All capabilities can be provided by built-in tools
- Latency is critical (agent-to-agent calls add overhead)

---

## 🗂️ Examples in This Directory

### 1. `agent_bricks.yaml` - Agent Bricks Integration

Demonstrates integration with Agent Bricks, routing customer queries to specialized agents.

**What it shows:**
- Customer support agent delegation
- Product expert consultation
- Intelligent routing between specialists
- Context-aware agent selection

**Key concepts:**
- Agent endpoint tool configuration
- Multi-specialist orchestration
- Dynamic task delegation

**Use for:**
- Customer service automation
- Technical support systems
- Multi-domain expertise applications

---

### 2. `kasal.yaml` - Kasal Enterprise Integration

Shows enterprise-grade agent integration with Kasal, including compliance and governance.

**What it shows:**
- Financial analysis agent integration
- Compliance validation workflows
- Data privacy specialist consultation
- Multi-agent governance patterns

**Key concepts:**
- Compliance-first agent design
- Privacy-aware data handling
- Enterprise agent orchestration
- Regulatory validation workflows

**Use for:**
- Financial services applications
- Healthcare and regulated industries
- Enterprise data processing
- Compliance-sensitive workflows

---

## 🚀 Quick Start

### Prerequisites

1. **External Agent Endpoints**: Deploy and configure your Agent Bricks or Kasal agents
2. **Endpoint URLs**: Obtain the serving endpoint names/URLs for external agents
3. **Authentication**: Configure necessary credentials for external agent access
4. **Databricks Workspace**: Access to create and deploy DAO-AI agents

### Running the Examples

#### Agent Bricks Example

```bash
# Set up Agent Bricks endpoint (replace with your actual endpoints)
export AGENT_BRICKS_CUSTOMER_SUPPORT_ENDPOINT="agent-bricks-customer-support"
export AGENT_BRICKS_PRODUCT_EXPERT_ENDPOINT="agent-bricks-product-expert"

# Run the agent
dao-ai chat -c config/examples/10_agent_integrations/agent_bricks.yaml
```

**Try asking:**
- "A customer is unhappy with a delayed order, how should I handle this?"
- "What's the best drill for a professional contractor?"
- "I need to process a return for a defective power tool"

#### Kasal Example

```bash
# Set up Kasal endpoints (replace with your actual endpoints)
export KASAL_FINANCIAL_ENDPOINT="kasal-financial-analyst"
export KASAL_COMPLIANCE_ENDPOINT="kasal-compliance-checker"
export KASAL_PRIVACY_ENDPOINT="kasal-privacy-specialist"

# Run the agent
dao-ai chat -c config/examples/10_agent_integrations/kasal.yaml
```

**Try asking:**
- "What's our revenue forecast for next quarter?"
- "Can we store customer email addresses in this database?"
- "Does this marketing campaign comply with GDPR?"

---

## 🔧 How Agent Endpoint Tools Work

### Tool Factory Configuration

Agent endpoint tools are created using the `create_agent_endpoint_tool` factory:

```yaml
tools:
  specialist_agent_tool:
    name: my_specialist
    function:
      type: factory
      name: dao_ai.tools.create_agent_endpoint_tool
      args:
        llm: *external_agent_llm        # LLM endpoint configuration
        name: specialist_name            # Tool name for the orchestrator
        description: |                   # Clear description of capabilities
          Detailed description of what this agent can do
          and when to use it.
```

### LLM Configuration for External Agents

External agents are configured as LLM endpoints:

```yaml
resources:
  llms:
    external_agent: &external_agent
      name: external-agent-endpoint-name    # Databricks serving endpoint
      description: "Agent description"      # What this agent does
      temperature: 0.1                      # Model temperature
      max_tokens: 1000                      # Response length limit
```

### Orchestrator Pattern

The orchestrator agent decides when to delegate to specialists:

```yaml
agents:
  orchestrator:
    name: main_agent
    model: *main_llm
    tools:
      - *specialist_tool_1
      - *specialist_tool_2
    prompt: |
      You coordinate between specialist agents.
      Use specialist_tool_1 for X tasks.
      Use specialist_tool_2 for Y tasks.
```

---

## 🏗️ Architecture Patterns

### 1. Hub-and-Spoke Pattern

**Structure**: One orchestrator routes to multiple specialists

```
         ┌─────────────┐
         │ Orchestrator│
         │   Agent     │
         └──────┬──────┘
                │
      ┌─────────┼─────────┐
      │         │         │
   ┌──▼──┐   ┌──▼──┐   ┌──▼──┐
   │Spec │   │Spec │   │Spec │
   │  A  │   │  B  │   │  C  │
   └─────┘   └─────┘   └─────┘
```

**Use for**: Task routing, domain separation

### 2. Sequential Workflow Pattern

**Structure**: Chain of specialist agents in sequence

```
User → Agent A → Agent B → Agent C → Response
```

**Use for**: Compliance validation, multi-step processing

### 3. Parallel Consultation Pattern

**Structure**: Consult multiple agents simultaneously

```
           ┌─── Agent A ───┐
User → Hub ├─── Agent B ───┤ → Synthesize → Response
           └─── Agent C ───┘
```

**Use for**: Multi-perspective analysis, consensus building

---

## ⚙️ Configuration Options

### Tool Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `llm` | LLMModel | External agent endpoint configuration |
| `name` | str | Tool name (how orchestrator references it) |
| `description` | str | Detailed description of agent capabilities |

### LLM Endpoint Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Databricks serving endpoint name |
| `temperature` | float | Model creativity (0.0 = deterministic) |
| `max_tokens` | int | Maximum response length |
| `description` | str | Agent purpose and capabilities |

---

## 📊 Best Practices

### 1. Clear Agent Responsibilities

**✅ Do:**
- Give each agent a specific, well-defined role
- Document agent capabilities clearly in descriptions
- Avoid overlapping responsibilities

**❌ Don't:**
- Create agents with vague, overlapping roles
- Make descriptions too generic
- Assume the orchestrator knows agent capabilities

### 2. Effective Prompting

**✅ Do:**
- Provide complete context when calling specialist agents
- Include relevant background information
- Format requests clearly

**❌ Don't:**
- Send minimal or ambiguous prompts
- Assume specialists have conversation context
- Omit critical details

### 3. Error Handling

**✅ Do:**
- Handle agent timeout and failure scenarios
- Provide fallback options
- Log agent interactions for debugging

**❌ Don't:**
- Assume agent calls always succeed
- Silently ignore errors
- Leave users hanging on failures

### 4. Performance Optimization

**✅ Do:**
- Cache agent responses when appropriate
- Use parallel calls when possible
- Set reasonable token limits

**❌ Don't:**
- Make unnecessary sequential calls
- Request excessive output
- Ignore latency implications

---

## 🔐 Security & Governance

### Authentication

External agent endpoints should use:
- Service principal authentication
- Token-based access control
- Databricks endpoint security

### Compliance Considerations

When using agent integrations for regulated industries:

1. **Audit Trails**: Log all agent interactions
2. **Data Privacy**: Validate PII handling with privacy specialists
3. **Regulatory Compliance**: Use compliance validators before final decisions
4. **Access Control**: Restrict agent access based on user roles

### Example Compliance Workflow

```yaml
# Always check compliance before execution
1. User makes request
2. Orchestrator analyzes request
3. If decision has regulatory implications:
   → Call compliance_validator agent
   → Wait for approval
   → Proceed only if validated
4. Execute with specialist agent
5. Log all interactions for audit
```

---

## 🎯 Common Use Cases

### Customer Support Automation

**Pattern**: Route customer queries to specialized support agents

```yaml
- Customer service agent: Handles complaints, returns
- Product expert agent: Technical questions, recommendations
- Escalation agent: Complex issues requiring human intervention
```

### Financial Services

**Pattern**: Multi-agent financial analysis with compliance

```yaml
- Financial analyst: Data analysis and forecasting
- Compliance validator: Regulatory checks
- Risk assessment: Risk scoring and mitigation
```

### Healthcare & Life Sciences

**Pattern**: Clinical decision support with privacy

```yaml
- Clinical expert: Medical guidance
- Privacy specialist: HIPAA compliance validation
- Research agent: Latest treatment protocols
```

### Enterprise IT

**Pattern**: Multi-domain technical support

```yaml
- Security specialist: Security and access control
- Infrastructure expert: System and network issues
- Application support: Software and integration
```

---

## 🐛 Troubleshooting

### Agent Not Responding

**Problem**: External agent endpoint not returning responses

**Solutions:**
1. Verify endpoint is deployed and active
2. Check authentication credentials
3. Validate endpoint name matches configuration
4. Review serving endpoint logs in Databricks

### Poor Task Routing

**Problem**: Orchestrator calling wrong specialist agents

**Solutions:**
1. Improve tool descriptions (be more specific)
2. Enhance orchestrator prompt with clearer routing rules
3. Add examples in orchestrator prompt
4. Consider adding a routing decision step

### High Latency

**Problem**: Agent responses taking too long

**Solutions:**
1. Use parallel agent calls when possible
2. Reduce max_tokens for specialist agents
3. Optimize specialist agent prompts
4. Consider caching for repeated queries

### Inconsistent Results

**Problem**: Agent responses vary unexpectedly

**Solutions:**
1. Lower temperature (closer to 0.0) for deterministic results
2. Provide more context in agent calls
3. Use structured output formats
4. Add validation steps

---

## 📚 Related Examples

👉 **13_orchestration/** - Advanced multi-agent patterns  
👉 **02_mcp/** - Alternative tool integration approaches  
👉 **15_complete_applications/** - Full production systems  

---

## 🔗 Additional Resources

- [Agent Endpoint Tools Documentation](../../../docs/key-capabilities.md#agent-tools)
- [Multi-Agent Orchestration Guide](../13_orchestration/)
- [Production Deployment Best Practices](../../../docs/contributing.md)

---

## 📝 Next Steps

After mastering agent integrations:

1. **Explore Orchestration**: Learn advanced patterns in `13_orchestration/`
2. **Add Middleware**: Implement cross-cutting concerns from `12_middleware/`
3. **Deploy Production**: See complete systems in `15_complete_applications/`

---

**Questions or Issues?** Check the [FAQ](../../../docs/faq.md) or open an issue on GitHub.
