# 15. Complete Applications

**Full-featured, production-ready agent applications**

Reference implementations of complete, production-grade agent systems. These examples combine multiple patterns from previous categories into cohesive applications.

## Examples

| File | Description | Complexity |
|------|-------------|------------|
| `executive_assistant.yaml` | Comprehensive assistant (email, calendar, Slack) | ⭐⭐⭐⭐⭐ |
| `deep_research.yaml` | Multi-step research agent with web search | ⭐⭐⭐⭐ |
| `reservations_system.yaml` | Restaurant reservation management | ⭐⭐⭐ |
| `genie_vector_search_hybrid.yaml` | Combined SQL and vector search | ⭐⭐⭐⭐ |
| `genie_and_genie_mcp.yaml` | Multiple Genie instances via MCP | ⭐⭐⭐⭐ (experimental) |

## What You'll Learn

- **System integration** - Combine multiple tools and services
- **Production patterns** - Safety, monitoring, error handling
- **Complex workflows** - Multi-step processes and decision trees
- **Real-world architecture** - How to structure production agents

## Application Profiles

### Executive Assistant
**Use case**: Personal productivity and communication  
**Tools**: Email (SMTP), Calendar, Slack, Web search, File management  
**Patterns**: HITL for sensitive ops, structured output, memory

**Capabilities:**
- 📧 Email management (read, send, search)
- 📅 Calendar operations (schedule, find meetings)
- 💬 Slack integration (send messages, check channels)
- 🔍 Web search for information
- 📁 Document management

### Deep Research
**Use case**: Comprehensive research and analysis  
**Tools**: Web search, Vector Search, Document processing  
**Patterns**: Multi-step reasoning, caching, structured output

**Workflow:**
1. Understand research question
2. Search for relevant information
3. Synthesize findings
4. Generate structured report
5. Iterate with follow-up questions

### Reservations System
**Use case**: Restaurant booking management  
**Tools**: Database access, SMS/Email notifications, Calendar  
**Patterns**: Structured output, validation, state management

**Operations:**
- 📞 Take reservations
- 🔍 Check availability
- ✉️ Send confirmations
- 🔄 Handle modifications/cancellations
- 📊 Generate reports

### Genie + Vector Search Hybrid
**Use case**: Comprehensive data access  
**Tools**: Databricks Genie (SQL), Vector Search (semantic)  
**Patterns**: Caching, intelligent routing

**Capabilities:**
- SQL queries for structured data
- Semantic search for unstructured content
- Intelligent query routing
- Combined result synthesis

## Prerequisites

### General
- ✅ All concepts from categories 01-06
- ✅ Production Databricks workspace
- ✅ MLflow for monitoring
- ✅ PostgreSQL or Lakebase for state

### Application-Specific

**Executive Assistant:**
- SMTP credentials or SendGrid API
- Calendar API access (Google Calendar, Outlook)
- Slack workspace and bot token

**Deep Research:**
- Web search API (DuckDuckGo or Serper)
- Vector Search index
- Document processing tools

**Reservations:**
- Database for bookings
- SMS/Email service
- Optional: POS system integration

**Genie + Vector Search:**
- Multiple Genie spaces
- Vector Search indexes
- Embedding models

## Quick Start

### Executive Assistant
```bash
# Set required credentials
export SMTP_PASSWORD="your-password"
export SLACK_BOT_TOKEN="xoxb-token"
export CALENDAR_API_KEY="your-key"

dao-ai chat -c config/examples/11_complete_applications/executive_assistant.yaml
```

Example: *"Check my calendar for tomorrow and send a Slack message to #team with my availability"*

### Deep Research
```bash
dao-ai chat -c config/examples/11_complete_applications/deep_research.yaml
```

Example: *"Research the latest developments in quantum computing and create a summary report"*

### Reservations
```bash
dao-ai chat -c config/examples/11_complete_applications/reservations_system.yaml
```

Example: *"Make a reservation for 4 people tomorrow at 7pm"*

## Production Checklist

Before deploying these applications:

### Security
- [ ] All credentials stored in Databricks Secrets
- [ ] HITL enabled for sensitive operations
- [ ] Input validation for all user queries
- [ ] Rate limiting configured
- [ ] Audit logging enabled

### Reliability
- [ ] Error handling for all tool calls
- [ ] Fallback strategies defined
- [ ] Timeout configuration
- [ ] Retry logic for transient failures
- [ ] State persistence configured

### Monitoring
- [ ] MLflow tracking enabled
- [ ] Custom metrics logged
- [ ] Alert thresholds defined
- [ ] Performance dashboards created
- [ ] Cost tracking configured

### Quality
- [ ] Guardrails enabled
- [ ] Output validation
- [ ] Response time SLAs
- [ ] Accuracy targets defined
- [ ] User feedback loop

## Architecture Patterns

### Layered Architecture
```
┌─────────────────────────────────┐
│      User Interface             │
├─────────────────────────────────┤
│      Agent Orchestration        │
├─────────────────────────────────┤
│      Business Logic             │
├─────────────────────────────────┤
│      Tool Integrations          │
├─────────────────────────────────┤
│      Data & State Layer         │
└─────────────────────────────────┘
```

### Key Design Decisions

1. **Separation of Concerns**: Each tool handles one responsibility
2. **Idempotency**: Operations can be safely retried
3. **State Management**: Clear state transitions and persistence
4. **Error Boundaries**: Failures isolated and handled gracefully
5. **Observability**: Comprehensive logging and tracing

## Customization Guide

These applications are starting points. To adapt for your needs:

### 1. Identify Core Requirements
- What are the must-have features?
- What tools/services need integration?
- What are the safety/compliance requirements?

### 2. Select Base Application
- Choose the example closest to your needs
- Review its architecture and patterns

### 3. Customize Configuration
- Modify agents and tools
- Adjust prompts for your domain
- Configure security and validation

### 4. Test Thoroughly
- Unit test individual tools
- Integration test workflows
- Load test for expected traffic
- Security test for vulnerabilities

### 5. Deploy Gradually
- Start with limited users/beta
- Monitor metrics closely
- Iterate based on feedback
- Scale gradually

## Performance Optimization

### Caching Strategy
```yaml
# Add caching from 04_genie
tools:
  - genie:
      lru_cache: true
      semantic_cache: true
```

### Parallel Execution
```yaml
# Tools that can run in parallel
agents:
  my_agent:
    parallel_tool_calls: true
```

### Model Selection
```yaml
# Use faster models for simple tasks
agents:
  classifier:
    model: *fast_model    # GPT-3.5, Claude Haiku
  reasoner:
    model: *smart_model   # GPT-4, Claude Opus
```

## Cost Management

| Component | Monthly Cost (est.) | Optimization |
|-----------|---------------------|--------------|
| LLM calls | $100-500 | Cache, use appropriate model size |
| Vector Search | $50-200 | Limit query frequency, batch queries |
| Database | $20-100 | Appropriate instance size, connection pooling |
| Tools (APIs) | $0-300 | Rate limits, caching, choose cost-effective providers |

**Total**: $170-1,100/month for moderate usage

## Troubleshooting

### High Latency
- Enable caching (04_genie)
- Use parallel tool calls
- Optimize prompts (reduce tokens)
- Use faster models where appropriate

### High Costs
- Review tool call frequency
- Enable caching
- Use smaller models for simple tasks
- Set token limits

### Low Accuracy
- Improve prompts (11_prompt_engineering)
- Add examples to prompts
- Use more capable models
- Add validation and guardrails

## Next Steps

🎉 **Congratulations!** You've completed the learning path.

### Continue Learning
- Review [Key Capabilities](../../../docs/key-capabilities.md)
- Explore [Python API](../../../docs/python-api.md)
- Read [Architecture Guide](../../../docs/architecture.md)

### Build Your Own
- Identify your use case
- Start with relevant example
- Customize step-by-step
- Deploy to production

### Contribute Back
- Share your application
- Report issues or improvements
- Help other users
- [Contributing Guide](../../../docs/contributing.md)

## Related Documentation

- [Deployment Guide](../../../docs/cli-reference.md#bundle)
- [Production Best Practices](../../../docs/faq.md)
- [MLflow Integration](../../../docs/key-capabilities.md)

---

**Need help?** Check the [FAQ](../../../docs/faq.md) or open a [GitHub issue](https://github.com/your-org/dao-ai/issues).

