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
agent = config.as_graph()

# Test locally (async)
response = await agent.ainvoke({
    "messages": [{"role": "user", "content": "Test question"}]
})
print(response["messages"][-1].content)
```

See [Lab 1 — Your First DAO-AI Agent](https://github.com/natefleming/dao-ai-workshop/tree/main/L100-foundations/lab-01-first-agent) for the shortest end-to-end example.

### What's the learning curve?

**If you're new to AI agents:** 1-2 weeks to understand concepts and build your first agent. Start with the four self-paced [L100 foundations labs](https://github.com/natefleming/dao-ai-workshop/tree/main/L100-foundations) in the [dao-ai-workshop](https://github.com/natefleming/dao-ai-workshop) — zero to a deployed Databricks App in ~2 hours.

**If you know LangChain:** 1-2 days to translate your knowledge to YAML configs. [Lab 13 — Programmatic Construction](https://github.com/natefleming/dao-ai-workshop/tree/main/L300-advanced/lab-13-programmatic) shows how to build the same `AppConfig` in Python instead of YAML, which is closest to the LangChain-code mental model.

**If you're a business user:** Consider starting with [DAO AI Builder](https://github.com/natefleming/dao-ai-builder) (visual interface).

### How do I get help?

1. Check the [`config/examples/`](../config/examples/) directory for working examples
2. Run through the [dao-ai-workshop](https://github.com/natefleming/dao-ai-workshop) — 25 self-paced labs covering every framework feature, each with a runnable notebook + YAML
3. Review the documentation for detailed explanations
4. Read the [Configuration Reference](configuration-reference.md) section
5. Open an issue on GitHub

## Deployment Questions

### Can I deploy to multiple environments?

Yes! Use different configuration files for each environment:

```bash
# Development
dao-ai pipeline --deploy -c config/dev.yaml --profile dev

# Production
dao-ai pipeline --deploy -c config/prod.yaml --profile prod
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
dao-ai pipeline --deploy --run -c config/my_config.yaml
```

This will update the existing deployment.

## Performance Questions

### How do I optimize agent performance?

1. **Enable caching** for Genie queries (LRU + Context-Aware cache) — see [Lab 12 — Genie Context-Aware Caching](https://github.com/natefleming/dao-ai-workshop/tree/main/L300-advanced/lab-12-genie-caching)
2. **Use reranking** for vector search to improve result quality — see [Lab 6 — Vector Search + FlashRank](https://github.com/natefleming/dao-ai-workshop/tree/main/L200-real-agents/lab-06-vector-search) and [Lab 11 — Instructed Retrieval + LLM Rerank](https://github.com/natefleming/dao-ai-workshop/tree/main/L300-advanced/lab-11-instructed-retrieval)
3. **Tune similarity thresholds** to balance cache hit rate vs. accuracy
4. **Monitor MLflow traces** to identify bottlenecks — see [Lab 24 — UC OTEL Trace Tables](https://github.com/natefleming/dao-ai-workshop/tree/main/L300-advanced/lab-24-uc-trace-location) for durable trace storage
5. **Use appropriate model sizes** (larger models = slower but more accurate)

### What's the typical latency?

Latency depends on your configuration:

- **Simple query with cache hit**: 50-200ms
- **Vector search with reranking**: 200-500ms
- **Genie NL-to-SQL (no cache)**: 2-5 seconds
- **Multi-agent orchestration**: 1-10 seconds (depends on complexity)

### How do I reduce costs?

1. **Enable caching** - Dramatically reduces Genie API calls ([Lab 12](https://github.com/natefleming/dao-ai-workshop/tree/main/L300-advanced/lab-12-genie-caching))
2. **Use smaller models** where appropriate
3. **Implement result deduplication** to avoid redundant processing
4. **Set TTLs appropriately** to balance freshness vs. cache hits
5. **Monitor usage** with MLflow tracking

## Configuration Questions

### What is the difference between `parameters:` and `variables:`?

`parameters:` are inputs to the YAML resolved at **load time** by `AppConfig.from_file`. `variables:` are typed value sources (`env:`, `scope:/secret:`, composites) resolved at **runtime** inside the deployed app.

Rule of thumb:

- Should the value travel with the bundle (catalog name, schema, app name)? Use `parameters:`.
- Should the value be read from the deployed environment or Databricks Secrets each time the agent runs (credentials, hostnames)? Use `variables:`.

See [Parameters vs Variables](configuration-reference.md#parameters-vs-variables---the-lifecycle-distinction) for the full comparison table.

### What happens if I use `${var.NAME}` without declaring it?

If your YAML has a `parameters:` block, any `${var.NAME}` reference not declared in that block raises a `ConfigVariableError` listing all undeclared names. This is intentional typo protection - a misspelled `${var.catlaog}` fails loudly at load time instead of silently resolving to nothing.

If your YAML has no `parameters:` block at all, the undeclared-name check is skipped and the reference falls through to the inline `:-default` or the "missing required" error.

### Can I use a parameter to choose which secret to load?

Yes - this is the recommended **bridge pattern**. `${var.NAME}` references are text-substituted before the YAML is parsed, so they work inside any string field, including fields belonging to typed `variables:` entries:

```yaml
parameters:
  scope:
    default: my_scope

variables:
  api_key: &api_key
    options:
      - scope: ${var.scope}
        secret: api_key
```

This lets the same config target different secret scopes per environment. See [Bridge Pattern](configuration-reference.md#bridge-pattern-parameters-feeding-variables) for a full worked example.

### How do I forward the caller's identity (OBO)?

Set `on_behalf_of_user: true` on any Databricks resource you want the deployed agent to reach *as the calling user* rather than as the agent's own service principal. The Apps runtime forwards the caller's `x-forwarded-access-token` through to that resource for every request.

The flag is accepted by any Databricks resource — most commonly LLMs (`resources.models.*`), downstream Apps (`resources.apps.*`), UC tables (`resources.tables.*`), warehouses, and Vector Search indexes.

```yaml
resources:
  models:
    default_llm:
      name: databricks-claude-sonnet-4-5
      on_behalf_of_user: true          # forward caller identity to the LLM endpoint

  apps:
    upstream_agent:
      name: some-other-app
      on_behalf_of_user: true          # calls the other App as the user, not as the SP
```

**A2A auto-derivation:** if any resource in the config carries `on_behalf_of_user: true` and you haven't explicitly set `a2a.on_behalf_of_user`, dao-ai auto-derives it to `True` and the deployed agent-card emits both `oauth2` (authorizationCode flow, `user_impersonation` scope, workspace's real OIDC URLs) and `bearer` security schemes.

See [Lab 20 — A2A: HITL + OBO](https://github.com/natefleming/dao-ai-workshop/tree/main/L300-advanced/lab-20-a2a-hitl-obo) for the canonical end-to-end demonstration (approve/edit/reject over A2A with OBO). [Lab 10 — Human in the Loop](https://github.com/natefleming/dao-ai-workshop/tree/main/L200-real-agents/lab-10-hitl) covers the standalone HITL primitive that OBO commonly runs alongside.

---

## MLflow Tracing & Monitoring

### How do I route traces to a UC schema?

Declare `app.trace_location:` on the AppConfig. The dao-ai deploy paths (both Databricks Apps and Model Serving) will call `mlflow.set_experiment(experiment_id=..., trace_location=UnityCatalog(...))` at boot, and MLflow will lazily materialize four Delta tables in the target schema — `<prefix>_otel_spans`, `<prefix>_otel_logs`, `<prefix>_otel_metrics`, `<prefix>_otel_annotations` — on the first trace flush.

```yaml
app:
  name: hardware-store
  trace_location:
    schema:                              # UC schema where the OTEL tables live
      catalog_name: retail_consumer_goods
      schema_name: dao_ai_workshop
    warehouse: ${var.warehouse_id}       # SQL warehouse used to materialize tables
    table_prefix: hardware_store         # optional; defaults to the experiment_id
```

Everything about the wiring is identical for Apps and Model Serving — same `_link_experiment_trace_location` call is invoked from both deploy paths. The only asymmetry is what permissions the endpoint's runtime SP needs on the target schema (see next question).

See [Lab 24 — UC OTEL Trace Tables](https://github.com/natefleming/dao-ai-workshop/tree/main/L300-advanced/lab-24-uc-trace-location) for the walkthrough. **Note:** in-process notebook usage of the same config additionally needs `mlflow.langchain.autolog(run_tracer_inline=True)` + `dao_ai.logging.suppress_autolog_context_warnings()` — the deploy runtime does both automatically at boot, but the notebook flow must do them explicitly.

### What extra permissions does Model Serving need for `trace_location`?

The Model Serving endpoint runs as a service principal, and that SP needs write access to the OTEL tables — MLflow inserts spans row-by-row on every trace export.

dao-ai grants the required privileges automatically at deploy time (see `_grant_uc_trace_table_permissions_to_principal`):

- `USE_CATALOG` on the target catalog
- `USE_SCHEMA` on the target schema
- `SELECT` + `MODIFY` on each of the four OTEL tables

**Gotcha:** the *deployer* (the person running `deploy_agent(...)` or `dao-ai pipeline --deploy`) must hold `MANAGE` on the target UC schema for those grants to succeed. If the deployer doesn't have `MANAGE`, ask a metastore admin to run once:

```sql
GRANT USE_CATALOG ON CATALOG <catalog> TO `<endpoint-sp-client-id>`;
GRANT USE_SCHEMA, CREATE_TABLE, MODIFY, SELECT
  ON SCHEMA <catalog>.<schema>
  TO `<endpoint-sp-client-id>`;
```

The workshop README's ["Trace persistence on Databricks Apps"](https://github.com/natefleming/dao-ai-workshop#trace-persistence-on-databricks-apps) section has the exact snippet. Databricks Apps have the same requirement for the App's SP — grants are one-off per app / endpoint per schema.

### How do I point an agent at an existing MLflow experiment?

Use `app.experiment:` with either `id` (numeric experiment ID, wins if both fields are set) or `name` (workspace path):

```yaml
app:
  experiment:
    id: "1952423719449237"              # bind by ID, no lookup
    # OR
    name: "/Shared/traces/hardware-store"  # bind by path; created if missing
    create_if_not_exists: false          # set false to hard-fail if the path doesn't exist
```

Precedence: `id` wins if both are set. When the whole `experiment:` block is omitted, dao-ai auto-creates `/Users/<deployer_email>/<app.name>` — fine for solo development but not what you want if a team shares one experiment (or if the experiment is pre-provisioned by an admin with tighter ACLs).

The canonical worked example is [`config/examples/15_complete_applications/hardware_store.yaml`](../config/examples/15_complete_applications/hardware_store.yaml).

### How do I turn on production monitoring / register scorers?

Declare `app.monitoring:` — at deploy time, dao-ai calls `register_monitoring_scorers` to bind judges to the agent's trace stream. Built-in scorers get their own sample rate; LLM-judge guidelines have a separate one so you can, for example, run cheap heuristic scorers on 100% of traffic and expensive judges on 25%.

```yaml
app:
  name: hardware-store
  monitoring:
    sample_rate: 1.0                     # built-in scorers on 100% of traces
    scorers:                             # names or globs of built-in scorers
      - safety
      - relevance
    guidelines_sample_rate: 0.25         # LLM-judge sampling
    guidelines:
      - name: quality_check
        guidelines:
          - "Responses must be complete and accurate."
          - "Responses must not fabricate SKUs."
```

Monitoring is **independent** of `trace_location:` — it works over MLflow's default trace store as well as UC OTEL tables. When both are configured, monitoring uses the same warehouse to query the UC tables, so assessment results land alongside spans in Unity Catalog.

See [Lab 23 — Production Monitoring with Registered Scorers](https://github.com/natefleming/dao-ai-workshop/tree/main/L300-advanced/lab-23-production-monitoring) for the runtime side. For adjacent evaluation surfaces, [Lab 22 — Offline Evaluation](https://github.com/natefleming/dao-ai-workshop/tree/main/L300-advanced/lab-22-offline-evaluation) covers `mlflow.genai.evaluate()` on curated datasets, and [Lab 21 — User Feedback](https://github.com/natefleming/dao-ai-workshop/tree/main/L300-advanced/lab-21-feedback) covers attaching thumbs-up/thumbs-down assessments to live traces.

---

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

For context-aware cache:
- Verify PostgreSQL connection is working
- Check `similarity_threshold` isn't set too high
- Ensure embedding model is accessible
- Review logs for cache hits/misses

See [Lab 12](https://github.com/natefleming/dao-ai-workshop/tree/main/L300-advanced/lab-12-genie-caching) for the reference config that pairs L1 (LRU exact-match) with L2 (embedding-similarity) over a Genie tool.

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

Need help? Check the [`config/examples/`](../config/examples/) directory, or work through [Lab 13 — Programmatic Construction](https://github.com/natefleming/dao-ai-workshop/tree/main/L300-advanced/lab-13-programmatic), which builds the same `AppConfig` in pure Python instead of YAML — closest to the LangChain-code mental model.

---

## Navigation

- [← Previous: Python API](python-api.md)
- [↑ Back to Documentation Index](../README.md#-documentation)
- [Next: Contributing →](contributing.md)

