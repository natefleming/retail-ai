# Frequently Asked Questions (FAQ)

## Contents

**General**
- [How is this different from LangChain/LangGraph directly?](#how-is-this-different-from-langchainlanggraph-directly)
- [Do I need to learn Python?](#do-i-need-to-learn-python)
- [Can I test locally before deploying?](#can-i-test-locally-before-deploying)
- [What's the learning curve?](#whats-the-learning-curve)
- [How do I get help?](#how-do-i-get-help)

**Deployment**
- [Can I deploy to multiple environments?](#can-i-deploy-to-multiple-environments)
- [How do I manage secrets?](#how-do-i-manage-secrets)
- [How do I update a deployed agent?](#how-do-i-update-a-deployed-agent)
- [How do I deploy to Databricks Apps?](#how-do-i-deploy-to-databricks-apps)

**Performance**
- [How do I optimize agent performance?](#how-do-i-optimize-agent-performance)
- [What's the typical latency?](#whats-the-typical-latency)
- [How do I reduce costs?](#how-do-i-reduce-costs)

**Configuration**
- [What is the difference between `parameters:` and `variables:`?](#what-is-the-difference-between-parameters-and-variables)
- [What happens if I use `${var.NAME}` without declaring it?](#what-happens-if-i-use-varname-without-declaring-it)
- [Can I use a parameter to choose which secret to load?](#can-i-use-a-parameter-to-choose-which-secret-to-load)
- [How do I forward the caller's identity (OBO)?](#how-do-i-forward-the-callers-identity-obo)
- [How do I add human-in-the-loop approval to my tool calls?](#how-do-i-add-human-in-the-loop-approval-to-my-tool-calls)
- [How do I use Genie with dao-ai?](#how-do-i-use-genie-with-dao-ai)
- [How do I use Unity AI Gateway?](#how-do-i-use-unity-ai-gateway)
- [How do I use the MLflow Prompt Registry?](#how-do-i-use-the-mlflow-prompt-registry)
- [How do I give my agent persistent memory / chat history?](#how-do-i-give-my-agent-persistent-memory-chat-history)
- [How do I orchestrate multiple agents?](#how-do-i-orchestrate-multiple-agents)
- [How do I orchestrate a parallel fan-out pattern?](#how-do-i-orchestrate-a-parallel-fan-out-pattern)
- [How do I add guardrails to my agent?](#how-do-i-add-guardrails-to-my-agent)
- [How do I add tools to my agent (UC functions, REST, MCP)?](#how-do-i-add-tools-to-my-agent-uc-functions-rest-mcp)
- [How do I do RAG / vector search with reranking?](#how-do-i-do-rag-vector-search-with-reranking)
- [How do I run long-running tasks?](#how-do-i-run-long-running-tasks)

**MLflow Tracing & Monitoring**
- [How do I route traces to a UC schema?](#how-do-i-route-traces-to-a-uc-schema)
- [What extra permissions does Model Serving need for `trace_location`?](#what-extra-permissions-does-model-serving-need-for-trace_location)
- [How do I point an agent at an existing MLflow experiment?](#how-do-i-point-an-agent-at-an-existing-mlflow-experiment)
- [How do I turn on production monitoring / register scorers?](#how-do-i-turn-on-production-monitoring-register-scorers)

**Troubleshooting**
- [My agent isn't responding correctly](#my-agent-isnt-responding-correctly)
- [Cache isn't working](#cache-isnt-working)
- [Deployment fails](#deployment-fails)
- [Agent is slow](#agent-is-slow)

**Platform-specific**
- [How does DAO compare to Agent Bricks?](#how-does-dao-compare-to-agent-bricks)
- [Can I use DAO with Agent Bricks or Kasal?](#can-i-use-dao-with-agent-bricks-or-kasal)
- [Does DAO work with external LLMs?](#does-dao-work-with-external-llms)
- [How do I migrate from LangChain code to DAO?](#how-do-i-migrate-from-langchain-code-to-dao)

---

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

**Learn more:** [`docs/why-dao.md`](why-dao.md) · [`docs/key-capabilities.md`](key-capabilities.md) · [`docs/architecture.md`](architecture.md)

### Do I need to learn Python?

**For basic usage:** No. You only need to write YAML configuration files.

**For advanced usage:** Some Python knowledge helps if you want to:
- Create custom tools
- Write middleware hooks
- Build complex business logic

Most users stick to YAML and use pre-built tools.

**Learn more:** [`docs/python-api.md`](python-api.md) · [`config/examples/01_getting_started/minimal.yaml`](../config/examples/01_getting_started/minimal.yaml)

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

**Learn more:** [`docs/python-api.md`](python-api.md) · [`config/examples/01_getting_started/`](../config/examples/01_getting_started/)

### What's the learning curve?

**If you're new to AI agents:** 1-2 weeks to understand concepts and build your first agent. Start with the four self-paced [L100 foundations labs](https://github.com/natefleming/dao-ai-workshop/tree/main/L100-foundations) in the [dao-ai-workshop](https://github.com/natefleming/dao-ai-workshop) — zero to a deployed Databricks App in ~2 hours.

**If you know LangChain:** 1-2 days to translate your knowledge to YAML configs. [Lab 13 — Programmatic Construction](https://github.com/natefleming/dao-ai-workshop/tree/main/L300-advanced/lab-13-programmatic) shows how to build the same `AppConfig` in Python instead of YAML, which is closest to the LangChain-code mental model.

**If you're a business user:** Consider starting with [DAO AI Builder](https://github.com/natefleming/dao-ai-builder) (visual interface).

**Learn more:** [`docs/key-capabilities.md`](key-capabilities.md) · [`docs/examples.md`](examples.md) · [`config/examples/01_getting_started/`](../config/examples/01_getting_started/)

### How do I get help?

1. Check the [`config/examples/`](../config/examples/) directory for working examples
2. Run through the [dao-ai-workshop](https://github.com/natefleming/dao-ai-workshop) — 25 self-paced labs covering every framework feature, each with a runnable notebook + YAML
3. Review the documentation for detailed explanations — see the [docs index in the top-level README](../README.md#-documentation)
4. Read the [Configuration Reference](configuration-reference.md) section
5. Open an issue on GitHub

**Learn more:** [`docs/examples.md`](examples.md) · [`config/examples/README.md`](../config/examples/README.md)

## Deployment Questions

### Can I deploy to multiple environments?

Yes! Use different configuration files for each environment:

```bash
# Development
dao-ai pipeline --deploy -c config/dev.yaml --profile dev

# Production
dao-ai pipeline --deploy -c config/prod.yaml --profile prod
```

**Learn more:** [`docs/cli-reference.md`](cli-reference.md) · [`docs/configuration-reference.md`](configuration-reference.md) (parameters + variables lifecycle)

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

**Learn more:** [`docs/configuration-reference.md`](configuration-reference.md#variables) · [`config/examples/01_getting_started/`](../config/examples/01_getting_started/)

### How do I update a deployed agent?

Simply redeploy with the updated configuration:

```bash
dao-ai pipeline --deploy --run -c config/my_config.yaml
```

This will update the existing deployment.

**Learn more:** [`docs/cli-reference.md`](cli-reference.md)

### How do I deploy to Databricks Apps?

Two paths, both driven by the same YAML — pick whichever fits your workflow.

**Path 1 — one-call Python (what every workshop lab uses):**

```python
from dao_ai.config import AppConfig, DeploymentTarget

config = AppConfig.from_file("config/my_agent.yaml", params={...})
config.deploy_agent(target=DeploymentTarget.APPS)
print(f"Deployed app: {config.app.name}")
```

`deploy_agent(target=APPS)` generates the Asset Bundle, uploads source + `requirements.txt`, deploys the app, waits for compute ACTIVE, and (if `app.trace_location:` is set) grants the App SP the OTEL-table permissions.  See [Lab 1 — Your First DAO-AI Agent](https://github.com/natefleming/dao-ai-workshop/tree/main/L100-foundations/lab-01-first-agent) for the shortest working example.

**Path 2 — `dao-ai generate-bundle` (Asset Bundle you can inspect / edit / check into Git):**

```bash
dao-ai generate-bundle -c config/my_agent.yaml -o ./my-bundle
cd my-bundle
databricks bundle deploy
databricks bundle run <app-name>
```

`generate-bundle` writes a complete, deployable Databricks Apps bundle directory (`databricks.yaml`, `app.yaml`, `pyproject.toml`, scaffolding). Useful when you want the bundle under version control, need to hand-tune anything the generator produced, or want to deploy from CI outside of Python. Add `--development` to bundle local dao-ai source instead of the pinned PyPI wheel; add `--force` to overwrite an existing output directory.

**Learn more:** [`docs/cli-reference.md`](cli-reference.md) · [`docs/python-api.md`](python-api.md)

## Performance Questions

### How do I optimize agent performance?

1. **Enable caching** for Genie queries (LRU + Context-Aware cache) — see [Lab 12 — Genie Context-Aware Caching](https://github.com/natefleming/dao-ai-workshop/tree/main/L300-advanced/lab-12-genie-caching)
2. **Use reranking** for vector search to improve result quality — see [Lab 6 — Vector Search + FlashRank](https://github.com/natefleming/dao-ai-workshop/tree/main/L200-real-agents/lab-06-vector-search) and [Lab 11 — Instructed Retrieval + LLM Rerank](https://github.com/natefleming/dao-ai-workshop/tree/main/L300-advanced/lab-11-instructed-retrieval)
3. **Tune similarity thresholds** to balance cache hit rate vs. accuracy
4. **Monitor MLflow traces** to identify bottlenecks — see [Lab 24 — UC OTEL Trace Tables](https://github.com/natefleming/dao-ai-workshop/tree/main/L300-advanced/lab-24-uc-trace-location) for durable trace storage
5. **Use appropriate model sizes** (larger models = slower but more accurate)

**Learn more:** [`docs/genie_context_aware_cache_prompt_history.md`](genie_context_aware_cache_prompt_history.md) · [`config/examples/04_genie/`](../config/examples/04_genie/) · [`config/examples/03_reranking/`](../config/examples/03_reranking/)

### What's the typical latency?

Latency depends on your configuration:

- **Simple query with cache hit**: 50-200ms
- **Vector search with reranking**: 200-500ms
- **Genie NL-to-SQL (no cache)**: 2-5 seconds
- **Multi-agent orchestration**: 1-10 seconds (depends on complexity)

**Learn more:** [`docs/architecture.md`](architecture.md) · [`docs/key-capabilities.md`](key-capabilities.md)

### How do I reduce costs?

1. **Enable caching** - Dramatically reduces Genie API calls ([Lab 12](https://github.com/natefleming/dao-ai-workshop/tree/main/L300-advanced/lab-12-genie-caching))
2. **Use smaller models** where appropriate
3. **Implement result deduplication** to avoid redundant processing
4. **Set TTLs appropriately** to balance freshness vs. cache hits
5. **Monitor usage** with MLflow tracking

**Learn more:** [`docs/genie_context_aware_cache_prompt_history.md`](genie_context_aware_cache_prompt_history.md) · [`config/examples/04_genie/genie_context_aware_cache.yaml`](../config/examples/04_genie/genie_context_aware_cache.yaml)

## Configuration Questions

### What is the difference between `parameters:` and `variables:`?

`parameters:` are inputs to the YAML resolved at **load time** by `AppConfig.from_file`. `variables:` are typed value sources (`env:`, `scope:/secret:`, composites) resolved at **runtime** inside the deployed app.

Rule of thumb:

- Should the value travel with the bundle (catalog name, schema, app name)? Use `parameters:`.
- Should the value be read from the deployed environment or Databricks Secrets each time the agent runs (credentials, hostnames)? Use `variables:`.

See [Parameters vs Variables](configuration-reference.md#parameters-vs-variables---the-lifecycle-distinction) for the full comparison table.

**Learn more:** [`docs/configuration-reference.md`](configuration-reference.md) · [`config/examples/01_getting_started/`](../config/examples/01_getting_started/)

### What happens if I use `${var.NAME}` without declaring it?

If your YAML has a `parameters:` block, any `${var.NAME}` reference not declared in that block raises a `ConfigVariableError` listing all undeclared names. This is intentional typo protection - a misspelled `${var.catlaog}` fails loudly at load time instead of silently resolving to nothing.

If your YAML has no `parameters:` block at all, the undeclared-name check is skipped and the reference falls through to the inline `:-default` or the "missing required" error.

**Learn more:** [`docs/configuration-reference.md`](configuration-reference.md#parameters)

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

**Learn more:** [`docs/configuration-reference.md`](configuration-reference.md#bridge-pattern-parameters-feeding-variables)

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

**Learn more:** [`docs/a2a_protocol.md`](a2a_protocol.md) · [`config/examples/06_on_behalf_of_user/`](../config/examples/06_on_behalf_of_user/) · [`config/examples/07_human_in_the_loop/`](../config/examples/07_human_in_the_loop/) · [`config/examples/20_a2a_protocol/a2a_hitl_obo.yaml`](../config/examples/20_a2a_protocol/a2a_hitl_obo.yaml)

### How do I add human-in-the-loop approval to my tool calls?

Add a `human_in_the_loop:` block to any `tools.<name>:` entry. The tool call is intercepted before execution and paused as an *interrupt* on the agent's LangGraph state — the client (a UI, a Slack app, an A2A caller, ...) sees an `input-required` state, decides `approve | edit | reject`, and resumes the graph with that decision.

```yaml
tools:
  refund_order:
    type: unity_catalog_function
    function: {schema: *ops_schema, name: refund_order}
    human_in_the_loop:
      review_prompt: |
        This action refunds a customer order and is irreversible.
        Confirm the order_id and amount before approving.
      allowed_decisions:
        - approve                       # run the tool as-is with the LLM-generated args
        - edit                          # let the reviewer edit the args before running
        - reject                        # block the tool call
        # - respond                     # add this to let the reviewer answer the user directly instead of running the tool
```

`allowed_decisions` defaults to `[approve, edit, reject]` if omitted; `respond` is opt-in when you want the reviewer to reply on the agent's behalf instead of executing the tool. `review_prompt` is shown to the human alongside the pending tool-call arguments.

See [Lab 10 — Human in the Loop](https://github.com/natefleming/dao-ai-workshop/tree/main/L200-real-agents/lab-10-hitl) for the standalone primitive and [Lab 20 — A2A: HITL + OBO](https://github.com/natefleming/dao-ai-workshop/tree/main/L300-advanced/lab-20-a2a-hitl-obo) for HITL over the A2A protocol (approve/edit/reject via DataPart resume, SSE streaming).

**Learn more:** [`config/examples/07_human_in_the_loop/`](../config/examples/07_human_in_the_loop/) · [`config/examples/20_a2a_protocol/a2a_hitl_obo.yaml`](../config/examples/20_a2a_protocol/a2a_hitl_obo.yaml) · [`docs/a2a_protocol.md`](a2a_protocol.md)

### How do I use Genie with dao-ai?

Declare a `genie_rooms:` entry under `resources:` referencing an existing Genie Space ID, then reference it from a `tools:` entry using `type: genie`. The LLM sees a callable tool (typically named `ask_genie`); under the hood dao-ai POSTs to the Genie Space's Conversation API on each call.

```yaml
parameters:
  genie_space_id:
    description: Databricks Genie Space ID (copy from the Space URL).

resources:
  genie_rooms:
    products_genie: &products_genie
      space_id: ${var.genie_space_id}

tools:
  ask_genie:
    type: genie
    genie_room: *products_genie

agents:
  greeter:
    tools: [*ask_genie]
    prompt: |
      Route product-data questions to `ask_genie`.
```

Create the Genie Space in the workspace UI first and copy its ID from the URL. See [Lab 3 — NL Analytics with Genie](https://github.com/natefleming/dao-ai-workshop/tree/main/L100-foundations/lab-03-genie) for the walkthrough, [Lab 12 — Genie Context-Aware Caching](https://github.com/natefleming/dao-ai-workshop/tree/main/L300-advanced/lab-12-genie-caching) for layering L1/L2 cache over the same tool, and [Lab 16 — Declarative Genie Space Provisioning](https://github.com/natefleming/dao-ai-workshop/tree/main/L300-advanced/lab-16-genie-provisioning) for provisioning the Space itself from YAML instead of the UI.

**Learn more:** [`docs/genie_context_aware_cache_prompt_history.md`](genie_context_aware_cache_prompt_history.md) · [`config/examples/04_genie/`](../config/examples/04_genie/) (basic, context-aware cache, threshold optimization) · [`config/examples/16_test_scenarios/genie_provisioning_only.yaml`](../config/examples/16_test_scenarios/genie_provisioning_only.yaml)

### How do I use Unity AI Gateway?

Set `ai_gateway: true` on an LLM resource. dao-ai will route chat completions through the Databricks AI Gateway path (`https://<host>/ai-gateway/mlflow/v1/chat/completions`) instead of the direct Model Serving path (`/serving-endpoints/<name>/invocations`). This is the standard way to pick up AI Gateway features — usage tracking, guardrails, PII redaction, and rate limiting — without changing any Python code.

```yaml
resources:
  models:
    default_llm:
      name: databricks-claude-sonnet-4-5
      ai_gateway: true                # route through AI Gateway
      temperature: 0.1
      max_tokens: 2048
```

**Constraints (enforced at load time):**
- `ai_gateway: true` is incompatible with `use_responses_api: true` on the same resource (the AI Gateway path exposes chat-completions only). dao-ai raises a validation error if both are set.
- OBO (`on_behalf_of_user: true`) + `ai_gateway: true` is permitted but relatively new — verify end-to-end trace propagation in your workspace before shipping.

Canonical example: [`config/examples/01_getting_started/ai_gateway.yaml`](../config/examples/01_getting_started/ai_gateway.yaml). Also used across `config/examples/15_complete_applications/commerce_supervisor/`.

**Learn more:** [`docs/key-capabilities.md`](key-capabilities.md) · [`config/examples/15_complete_applications/commerce_supervisor/commerce_supervisor.yaml`](../config/examples/15_complete_applications/commerce_supervisor/commerce_supervisor.yaml)

### How do I use the MLflow Prompt Registry?

Declare a top-level `prompts:` block containing `PromptModel` entries — each pins a prompt in the MLflow Prompt Registry by `schema.name`, optionally with `alias` (e.g. `champion`) or `version` (numeric). Reference the entry from an `agents:` or `guardrails:` block via a YAML anchor.

```yaml
schemas:
  workshop_schema: &workshop_schema
    catalog_name: main
    schema_name: dao_ai

prompts:
  support_prompt: &support_prompt
    schema: *workshop_schema
    name: support_prompt
    description: Main system prompt for the SaaS support agent.
    alias: champion                    # or: version: 3
    default_template: |                # inline text used only when auto_register=true
      You are a tier-1 SaaS support assistant. Be accurate and concise.
    auto_register: true                # register default_template if not in the registry

agents:
  saas_support:
    model: *default_llm
    prompt: *support_prompt            # resolves to prompts:/main.dao_ai.support_prompt@champion
```

- `alias` and `version` are mutually exclusive. If both are omitted, dao-ai loads `@latest`.
- `auto_register: true` writes `default_template` to the registry on first deploy; set `false` (default) if a prompt owner manages versions out-of-band.
- The same `PromptModel` also plugs into `guardrails.<name>.prompt` for LLM-judge guardrails.

See [Lab 8 — Production Prompts and Guardrails](https://github.com/natefleming/dao-ai-workshop/tree/main/L200-real-agents/lab-08-prompts-guardrails). The lab walks from an inline-string prompt (`01_inline_support.yaml`) → a Prompt-Registry-backed prompt (`02_support_with_managed_prompts.yaml`) → the same setup with an added judge guardrail (`03_support_with_guardrails.yaml`).

**Learn more:** [`docs/key-capabilities.md`](key-capabilities.md) · [`config/examples/11_prompt_engineering/`](../config/examples/11_prompt_engineering/)

### How do I give my agent persistent memory / chat history?

There are two independent knobs: a top-level `memory:` block (checkpointer for per-thread state + store for cross-thread facts + extraction LLM that writes those facts), and an `app.chat_history:` block for automatic summarization of long conversations. Both back onto Databricks Lakebase Postgres in production; you can start with the in-memory driver for local testing.

```yaml
resources:
  databases:
    lakebase: &lakebase
      type: lakebase_autoscaling
      project: retail-consumer-goods
      description: "Lakebase used for persistent memory."

memory:
  checkpointer:
    database: *lakebase                  # per-thread LangGraph state
  store:
    database: *lakebase                  # cross-thread facts
    schemas: [user_profile, preference]
  extraction:
    model: *default_llm                  # LLM pipeline that writes facts to the store

app:
  name: saas-helpdesk
  chat_history:
    model: *summarization_llm            # LLM used to summarize long conversations
    max_tokens: 500                      # tokens to keep after each summarization pass
    max_tokens_before_summary: 1500      # summarize when the running conversation exceeds this
    # OR: max_messages_before_summary: 20  # message-count trigger (mutually exclusive with tokens)
```

`chat_history` keeps long conversations under the model's context budget without losing the thread — `max_tokens` (default 2048) is the "keep" threshold; you supply *either* `max_tokens_before_summary` or `max_messages_before_summary` as the trigger. `store` + `extraction` are the "long-term memory" — the extractor LLM writes structured facts to the store on each turn and future turns retrieve them.

See [Lab 7 — Persistent Memory + Chat Summarization](https://github.com/natefleming/dao-ai-workshop/tree/main/L200-real-agents/lab-07-memory) for the runnable walkthrough.

**Learn more:** [`docs/key-capabilities.md`](key-capabilities.md) · [`config/examples/05_memory/`](../config/examples/05_memory/) (in-memory, Lakebase, conversation-summarization variants)

### How do I orchestrate multiple agents?

dao-ai supports three orchestration patterns. Pick based on how deterministic the routing needs to be:

- **Supervisor** — a central agent decides which sub-agent handles each turn. Best when routing is knowable up front (tier-1 vs tier-2 support, e.g.).
- **Swarm** — sub-agents hand control to each other via `handoff` tools. Best for open-ended flows where the next agent is a runtime decision.
- **Deep agent** — a planning agent driven by [`deepagents.create_deep_agent`](https://github.com/langchain-ai/deepagents) with `todo` / `filesystem` / `shell` tools, Skills (a directory of Markdown), and sub-agents callable via a `task` tool. Best for open-ended research / build workflows.

```yaml
agents:
  tier1_support: {model: *fast_llm, prompt: "..."}
  tier2_engineer: {model: *technical_llm, prompt: "..."}

# Exactly one of `supervisor:`, `swarm:`, or `deep_agent:` — dao-ai auto-picks
# a router based on the agent count when none of the three are set.
orchestration:
  supervisor:
    model: *default_llm                 # the routing LLM (the supervisor's own model)
    prompt: "..."                        # optional; routing instructions
    # OR use the shorthand for swarm defaults:
    # swarm: true
```

See [Lab 9 — Multi-agent Orchestration](https://github.com/natefleming/dao-ai-workshop/tree/main/L200-real-agents/lab-09-orchestration) for supervisor + swarm side by side, [Lab 17 — Deep Agent Orchestration](https://github.com/natefleming/dao-ai-workshop/tree/main/L300-advanced/lab-17-deep-agents) for the planning + Skills pattern, and [Lab 18 — Skills-only Deep Agent](https://github.com/natefleming/dao-ai-workshop/tree/main/L300-advanced/lab-18-skills-only-deep-agent) for the minimum-viable deep agent (zero sub-agents, one Skill).

**Learn more:** [`docs/key-capabilities.md`](key-capabilities.md) · [`docs/architecture.md`](architecture.md) · [`config/examples/13_orchestration/`](../config/examples/13_orchestration/) (deep-agent patterns) · [`config/examples/15_complete_applications/commerce_supervisor/`](../config/examples/15_complete_applications/commerce_supervisor/) and [`commerce_swarm/`](../config/examples/15_complete_applications/commerce_swarm/)

### How do I orchestrate a parallel fan-out pattern?

Declare a cohort as one handoff entry with `agents:` (the siblings) and
`join:` (the shared reducer). When the source's LLM invokes multiple
parallel handoff tools in a single turn, LangGraph runs the targeted
siblings in the **same superstep** (true concurrent execution) and runs
the join **exactly once** after all fired siblings complete. The end user
sees one final response from the join.

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

You can mix a cohort with regular single-target handoffs on the same
source:

```yaml
handoffs:
  triage_agent:
  - agents: [pricing_agent, inventory_agent, policy_agent]
    join: synthesizer_agent
  - escalation_agent                 # regular agentic peer
  - agent: emergency_agent
    is_deterministic: true           # single-target deterministic peer
```

**When to reach for it:**
- Multi-source retrieval / research (query several specialists, synthesize).
- Independent enrichment steps (price + inventory + policy → response).
- Judge / critic patterns (N candidates → one selector).

**Rules dao-ai enforces at load time:**
- A cohort entry must set both `agents` (list of ≥ 2 distinct siblings)
  and `join`. `agent` (singular) and `agents` are mutually exclusive on
  one entry.
- `is_deterministic` is not meaningful on a cohort entry — the join is
  always reached deterministically after fan-in.
- The join agent must not also appear in `agents` (no self-edge).
- A sibling cannot belong to two cohorts with different joins.
- A sibling cannot itself be the source of another cohort (nested
  fan-out is out of scope; the outer join would become unreachable).
- Cycles containing any parallel or deterministic edge are rejected up
  front so you don't burn compute on a runaway loop.

Prompt tip: because the LLM decides which siblings to invoke, tell the
source agent explicitly in its prompt to "call ALL parallel handoff tools
in a single turn" when that's the intent. If the LLM invokes only a subset,
that's a valid degenerate case — only those siblings run, then the join.
If it invokes zero, the source terminates and the join does not run.

See [`config/examples/13_orchestration/parallel_fan_out_pattern.yaml`](../config/examples/13_orchestration/parallel_fan_out_pattern.yaml)
for a complete deployable example.

**Learn more:** [`config/examples/13_orchestration/`](../config/examples/13_orchestration/) · [`docs/key-capabilities.md`](key-capabilities.md)

### How do I add guardrails to my agent?

Declare a top-level `guardrails:` block. Two modes:

1. **LLM-judge guardrail** — supply a judge `model` and a `prompt` (inline or from the Prompt Registry). dao-ai builds a `JudgeScorer` via `mlflow.genai.judges.make_judge`.
2. **Scorer-based guardrail** — supply a `scorer` class (any `mlflow.genai.scorers.base.Scorer` — built-in `ToxicLanguage`, `DetectPII`, `RelevanceToQuery`, etc.).

```yaml
guardrails:
  accuracy:
    model: *judge_llm
    prompt: *accuracy_guardrail_prompt   # PromptModel from the prompts: block
    num_retries: 2                       # retry the agent up to N times if the guardrail fails

  no_pii:
    scorer: mlflow.genai.scorers.DetectPII
```

Wire the guardrail(s) into an agent via `agents.<name>.guardrails: [...]`. Failed guardrails can either block the response (default) or trigger a retry (`num_retries`).

See [Lab 8 — Production Prompts and Guardrails](https://github.com/natefleming/dao-ai-workshop/tree/main/L200-real-agents/lab-08-prompts-guardrails) — the third YAML (`03_support_with_guardrails.yaml`) adds a judge guardrail to the prompts flow.

**Learn more:** [`docs/key-capabilities.md`](key-capabilities.md) · [`config/examples/08_guardrails/`](../config/examples/08_guardrails/)

### How do I add tools to my agent (UC functions, REST, MCP)?

dao-ai treats every tool the same way to the LLM (`agents.<name>.tools: [...]`), but you declare them by **type** so the framework knows how to build the LangChain tool underneath. Four common shapes:

**Unity Catalog SQL functions:**
```yaml
unity_catalog_functions:
  - function: {schema: *workshop_schema, name: find_product_by_sku}
    ddl: ./functions/find_product_by_sku.sql

tools:
  find_product_by_sku:
    type: unity_catalog_function
    function: {schema: *workshop_schema, name: find_product_by_sku}
```

**REST API:**
```yaml
tools:
  get_weather:
    type: rest
    method: GET
    url: https://api.weather.example.com/v1/forecast
    params: {lat: "{{lat}}", lon: "{{lon}}"}
```

**MCP server (managed — dao-ai discovers every tool the server exposes):**
```yaml
tools:
  functions_mcp: &functions_mcp
    name: functions_mcp
    function:
      type: mcp
      functions: *workshop_schema         # managed MCP: every UC function in this schema
      # Other source types (pick one):
      # sql: true                          # managed serverless DBSQL MCP
      # connection: *some_uc_connection    # UC Connection-backed (OAuth handled for you)
      # url: https://<host>/mcp/<name>     # external MCP by direct URL
      # Optional filtering (glob-supported; exclude_tools always wins over include_tools):
      # include_tools: ["get_*", "list_*"]
      # exclude_tools: ["drop_*", "delete_*"]
```

MCP tools live under `tools:` (not `resources:`). Each tool has `name:` + `function: { type: mcp, ... }`. The `function` block picks one of four sources: `functions:` (a UC schema — every function in it becomes a tool), `sql: true` (managed serverless DBSQL executor), `connection:` (a UC Connection that handles OAuth for you), or `url:` (an external MCP endpoint you address directly). `include_tools` / `exclude_tools` accept glob patterns; `exclude_tools` always wins.

**Genie space** (covered separately above via `type: genie`).

See the workshop's tool-grounding progression: [Lab 2 — Grounding with Unity Catalog Tools](https://github.com/natefleming/dao-ai-workshop/tree/main/L100-foundations/lab-02-uc-tools) → [Lab 4 — Schema-wide Tool Discovery with MCP](https://github.com/natefleming/dao-ai-workshop/tree/main/L100-foundations/lab-04-mcp) → [Lab 5 — External Integrations via REST](https://github.com/natefleming/dao-ai-workshop/tree/main/L200-real-agents/lab-05-rest).

**Learn more:** [`docs/key-capabilities.md`](key-capabilities.md) · [`docs/mcp_server.md`](mcp_server.md) · [`config/examples/14_basic_tools/`](../config/examples/14_basic_tools/) (REST, Slack) · [`config/examples/02_mcp/`](../config/examples/02_mcp/) (custom / external / filtered MCP)

### How do I do RAG / vector search with reranking?

Declare a `vector_stores:` resource pointing at a Databricks Vector Search index over a Delta table with Change Data Feed enabled, then wire it into a `retrievers:` block that composes ANN search with a rerank stage.

```yaml
resources:
  vector_stores:
    kb_articles: &kb_articles
      endpoint: {name: dao_ai_workshop_vs}
      source_table: {schema: *workshop_schema, name: kb_articles}
      primary_key: article_id
      embedding_source_column: body
      embedding_model: {name: databricks-gte-large-en}
      columns: [article_id, title, topic, body]

retrievers:
  kb_retriever:
    vector_store: *kb_articles
    search_parameters:
      num_results: 50
      query_type: HYBRID              # dense + BM25
    rerank:
      model: ${var.reranker_model}    # FlashRank cross-encoder
      top_n: 5
```

Reference the retriever from an `agents:` block (`agents.<name>.retrievers: [*kb_retriever]`) — the agent grounds each turn on the top-`top_n` reranked results.

For filter-heavy queries (*"Milwaukee power tools under $100"*), layer an **instructed retriever** on top: query decomposition into structured filters + a residual semantic query, then LLM-based rerank with natural-language instructions. See [Lab 6 — Knowledge-base Retrieval with Vector Search](https://github.com/natefleming/dao-ai-workshop/tree/main/L200-real-agents/lab-06-vector-search) for the base pattern and [Lab 11 — Instructed Retrieval](https://github.com/natefleming/dao-ai-workshop/tree/main/L300-advanced/lab-11-instructed-retrieval) for the filter-decomposition variant.

**Learn more:** [`docs/key-capabilities.md`](key-capabilities.md) · [`config/examples/03_reranking/`](../config/examples/03_reranking/) · [`config/examples/16_instructed_retriever/`](../config/examples/16_instructed_retriever/)

### How do I run long-running tasks?

Enable **background agents** via `app.background:` on the AppConfig. The deployed app then exposes an OpenAI Responses-API-shaped kickoff / poll / cancel surface: a POST kicks off work and returns immediately with a `response_id`; subsequent GETs poll for status; a DELETE cancels an in-flight run. Response state is persisted in a Lakebase-backed responses store so a run survives app restarts.

```yaml
resources:
  databases:
    lakebase: &lakebase
      type: lakebase_autoscaling
      project: retail-consumer-goods

app:
  name: research-agent
  background:
    database: *lakebase                   # Lakebase used for durable kickoff/poll/cancel state
    max_duration_seconds: 1800            # hard cap on any single background run
    poll_interval_seconds: 1.0            # internal poll cadence for streaming retrieve
    # default_enabled: true               # treat all requests as background even without background: true
```

Clients then use any OpenAI-compatible SDK against the deployed app's `/responses` endpoint (kickoff → poll → cancel). Best for research agents, batch data enrichment, or any workload that outlives a single request/response cycle.

See [Lab 15 — Background Agents](https://github.com/natefleming/dao-ai-workshop/tree/main/L300-advanced/lab-15-background) for the runnable walkthrough.

**Learn more:** [`docs/background_agents.md`](background_agents.md) · [`config/examples/19_background_agents/deep_research.yaml`](../config/examples/19_background_agents/deep_research.yaml) · [`config/examples/20_a2a_protocol/a2a_background.yaml`](../config/examples/20_a2a_protocol/a2a_background.yaml) (background over A2A)

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

**Learn more:** [`docs/configuration-reference.md`](configuration-reference.md) · [`config/examples/15_complete_applications/hardware_store.yaml`](../config/examples/15_complete_applications/hardware_store.yaml)

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

**Learn more:** [`docs/configuration-reference.md`](configuration-reference.md)

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

**Learn more:** [`docs/configuration-reference.md`](configuration-reference.md)

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

**Learn more:** [`docs/key-capabilities.md`](key-capabilities.md) · [`config/examples/08_guardrails/`](../config/examples/08_guardrails/)

---

## Troubleshooting

### My agent isn't responding correctly

1. **Check configuration**: Run `dao-ai validate -c config/my_config.yaml`
2. **Review logs**: Look for error messages in the output
3. **Test locally**: Use `dao-ai chat -c config/my_config.yaml` to interact
4. **Examine traces**: Check MLflow for detailed execution traces
5. **Verify permissions**: Ensure your service account has the necessary access

**Learn more:** [`docs/cli-reference.md`](cli-reference.md)

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

**Learn more:** [`docs/genie_context_aware_cache_prompt_history.md`](genie_context_aware_cache_prompt_history.md) · [`config/examples/04_genie/`](../config/examples/04_genie/)

### Deployment fails

Common issues:
1. **Missing permissions**: Ensure your profile has access to Model Serving
2. **Invalid configuration**: Run `dao-ai validate` first
3. **Resource conflicts**: Check if endpoint name already exists
4. **Missing dependencies**: Verify all custom packages are available

**Learn more:** [`docs/cli-reference.md`](cli-reference.md) · [`docs/configuration-reference.md`](configuration-reference.md)

### Agent is slow

1. **Profile with MLflow**: Identify bottlenecks using traces
2. **Enable caching**: Reduce redundant API calls
3. **Optimize prompts**: Shorter prompts = faster responses
4. **Check model size**: Consider using smaller/faster models
5. **Review middleware**: Disable unnecessary validation in dev

**Learn more:** [`docs/architecture.md`](architecture.md) · [`config/examples/04_genie/`](../config/examples/04_genie/) · [`config/examples/03_reranking/`](../config/examples/03_reranking/)

## Platform-Specific Questions

### How does DAO compare to Agent Bricks?

See the detailed comparison in [Why DAO?](why-dao.md#comparing-databricks-ai-agent-platforms)

**Quick summary:**
- **DAO**: Code-first, Git-native, advanced features (caching, middleware)
- **Agent Bricks**: GUI-based, automated optimization, rapid prototyping

**Learn more:** [`docs/why-dao.md`](why-dao.md) · [`config/examples/10_agent_integrations/agent_bricks.yaml`](../config/examples/10_agent_integrations/agent_bricks.yaml)

### Can I use DAO with Agent Bricks or Kasal?

Yes! All three platforms can interoperate via **agent endpoints**. Deploy agents from any platform to Model Serving and call them as tools in your DAO configuration.

See [Using All Three Together](why-dao.md#using-all-three-together) for examples.

**Learn more:** [`docs/why-dao.md`](why-dao.md) · [`config/examples/10_agent_integrations/`](../config/examples/10_agent_integrations/) (A2A, Agent Bricks, external agents)

### Does DAO work with external LLMs?

Yes! DAO supports:
- Databricks Foundation Models (native)
- OpenAI models (`openai:/gpt-4`)
- Anthropic models (via Databricks endpoints)
- Custom model endpoints

**Learn more:** [`docs/key-capabilities.md`](key-capabilities.md) · [`config/examples/01_getting_started/`](../config/examples/01_getting_started/)

### How do I migrate from LangChain code to DAO?

1. **Identify components**: Map your code to DAO configuration sections
2. **Create resources**: Define LLMs, databases, vector stores in `resources:`
3. **Define tools**: Convert tool definitions to YAML `tools:` section
4. **Configure agents**: Map agent logic to `agents:` configuration
5. **Set up orchestration**: Choose Supervisor or Swarm pattern
6. **Test**: Validate and test locally before deploying

Need help? Check the [`config/examples/`](../config/examples/) directory, or work through [Lab 13 — Programmatic Construction](https://github.com/natefleming/dao-ai-workshop/tree/main/L300-advanced/lab-13-programmatic), which builds the same `AppConfig` in pure Python instead of YAML — closest to the LangChain-code mental model.

**Learn more:** [`docs/python-api.md`](python-api.md) · [`docs/architecture.md`](architecture.md)

---

## Navigation

- [← Previous: Python API](python-api.md)
- [↑ Back to Documentation Index](../README.md#-documentation)
- [Next: Contributing →](contributing.md)

