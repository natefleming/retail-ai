# Frequently Asked Questions (FAQ)

## General Questions

### How is this different from LangChain/LangGraph directly?

DAO is **built on top of** LangChain and LangGraph. Instead of writing Python code to configure agents, you use YAML files. Think of it as:
- **LangChain/LangGraph**: The engine
- **DAO**: The blueprint system that configures the engine

Benefits:
- ✅ No Python coding required (just YAML)
- ✅ Configurations are easier to review and version control
- ✅ Databricks-specific integrations work out-of-the-box
- ✅ Reusable patterns across your organization

### Do I need to learn Python?

**For basic usage:** No. You only need to write YAML configuration files.

**For advanced usage:** Some Python knowledge helps if you want to:
- Create custom tools
- Write middleware hooks
- Build complex business logic

Most users stick to YAML and use pre-built tools.

### Can I test locally before deploying?

Yes! DAO includes a local testing mode:

```python
from dao_ai.config import AppConfig

config = AppConfig.from_file("config/my_agent.yaml")
agent = config.as_runnable()

# Test locally
response = agent.invoke({
    "messages": [{"role": "user", "content": "Test question"}]
})
print(response)
```

### What's the learning curve?

**If you're new to AI agents:** 1-2 weeks to understand concepts and build your first agent

**If you know LangChain:** 1-2 days to translate your knowledge to YAML configs

**If you're a business user:** Consider starting with [DAO AI Builder](https://github.com/natefleming/dao-ai-builder) (visual interface)

### How do I get help?

1. Check the [`config/examples/`](../config/examples/) directory for working examples
2. Review the documentation for detailed explanations
3. Read the [Configuration Reference](configuration-reference.md) section
4. Open an issue on GitHub

## Deployment Questions

### Can I deploy to multiple environments?

Yes! Use different configuration files for each environment:

```bash
# Development
dao-ai bundle --deploy -c config/dev.yaml --profile dev

# Production
dao-ai bundle --deploy -c config/prod.yaml --profile prod
```

### How do I manage secrets?

DAO supports multiple ways to manage secrets:

1. **Databricks Secrets** (recommended):
```yaml
variables:
  api_key: &api_key
    options:
      - scope: my_scope
        secret: api_key
```

2. **Environment Variables**:
```yaml
variables:
  api_key: &api_key
    options:
      - env: MY_API_KEY
```

### How do I update a deployed agent?

Simply redeploy with the updated configuration:

```bash
dao-ai bundle --deploy --run -c config/my_config.yaml
```

This will update the existing deployment.

## Performance Questions

### How do I optimize agent performance?

1. **Enable caching** for Genie queries (LRU + Semantic cache)
2. **Use reranking** for vector search to improve result quality
3. **Tune similarity thresholds** to balance cache hit rate vs. accuracy
4. **Monitor MLflow traces** to identify bottlenecks
5. **Use appropriate model sizes** (larger models = slower but more accurate)

### What's the typical latency?

Latency depends on your configuration:

- **Simple query with cache hit**: 50-200ms
- **Vector search with reranking**: 200-500ms
- **Genie NL-to-SQL (no cache)**: 2-5 seconds
- **Multi-agent orchestration**: 1-10 seconds (depends on complexity)

### How do I reduce costs?

1. **Enable caching** - Dramatically reduces Genie API calls
2. **Use smaller models** where appropriate
3. **Implement result deduplication** to avoid redundant processing
4. **Set TTLs appropriately** to balance freshness vs. cache hits
5. **Monitor usage** with MLflow tracking

## Troubleshooting

### My agent isn't responding correctly

1. **Check configuration**: Run `dao-ai validate -c config/my_config.yaml`
2. **Review logs**: Look for error messages in the output
3. **Test locally**: Use `dao-ai chat -c config/my_config.yaml` to interact
4. **Examine traces**: Check MLflow for detailed execution traces
5. **Verify permissions**: Ensure your service account has the necessary access

### Cache isn't working

For LRU cache:
- Verify questions are **exactly** the same (case-sensitive)
- Check TTL hasn't expired
- Ensure warehouse configuration is correct

For semantic cache:
- Verify PostgreSQL connection is working
- Check `similarity_threshold` isn't set too high
- Ensure embedding model is accessible
- Review logs for cache hits/misses

### Deployment fails

Common issues:
1. **Missing permissions**: Ensure your profile has access to Model Serving
2. **Invalid configuration**: Run `dao-ai validate` first
3. **Resource conflicts**: Check if endpoint name already exists
4. **Missing dependencies**: Verify all custom packages are available

### Agent is slow

1. **Profile with MLflow**: Identify bottlenecks using traces
2. **Enable caching**: Reduce redundant API calls
3. **Optimize prompts**: Shorter prompts = faster responses
4. **Check model size**: Consider using smaller/faster models
5. **Review middleware**: Disable unnecessary validation in dev

## Platform-Specific Questions

### How does DAO compare to Agent Bricks?

See the detailed comparison in [Why DAO?](why-dao.md#comparing-databricks-ai-agent-platforms)

**Quick summary:**
- **DAO**: Code-first, Git-native, advanced features (caching, middleware)
- **Agent Bricks**: GUI-based, automated optimization, rapid prototyping

### Can I use DAO with Agent Bricks or Kasal?

Yes! All three platforms can interoperate via **agent endpoints**. Deploy agents from any platform to Model Serving and call them as tools in your DAO configuration.

See [Using All Three Together](why-dao.md#using-all-three-together) for examples.

### Does DAO work with external LLMs?

Yes! DAO supports:
- Databricks Foundation Models (native)
- OpenAI models (`openai:/gpt-4`)
- Anthropic models (via Databricks endpoints)
- Custom model endpoints

### How do I migrate from LangChain code to DAO?

1. **Identify components**: Map your code to DAO configuration sections
2. **Create resources**: Define LLMs, databases, vector stores in `resources:`
3. **Define tools**: Convert tool definitions to YAML `tools:` section
4. **Configure agents**: Map agent logic to `agents:` configuration
5. **Set up orchestration**: Choose Supervisor or Swarm pattern
6. **Test**: Validate and test locally before deploying

Need help? Check the [`config/examples/`](../config/examples/) directory for reference implementations.

---

## Navigation

- [← Previous: Python API](python-api.md)
- [↑ Back to Documentation Index](../README.md#-documentation)
- [Next: Contributing →](contributing.md)

