# MCP Server for dao-ai

The `dao_ai.mcp` package turns any dao-ai agent config into a
[Model Context Protocol (MCP)](https://modelcontextprotocol.io) server
hosted on Databricks Apps. The emitted server exposes **one** MCP tool:
the whole dao-ai agent graph. Any MCP client — Claude Desktop, Cursor,
Databricks Multi-Agent Supervisor, another dao-ai agent — can call the
agent as a single tool over
[Streamable HTTP](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports).

This is the **server** side. `dao_ai.apps.mcp` is the **client** side
(primitives for *consuming* external MCP servers from a dao-ai agent). The
two are independent.

---

## Why

Customers integrating dao-ai with an external agent framework
(LangGraph, ADK, OpenAI Agents SDK, MAS, IDE assistants) don't want to
re-implement dao-ai's orchestration — they want to plug the dao-ai agent
in as a single high-level capability. Individual tools (Genie, Vector
Search, UC functions) already have first-class Databricks MCP surfaces or
UC exposure; publishing them a second time from a dao-ai MCP server is
duplicative.

`dao-ai generate-mcp` bundles a Databricks App that runs the same graph
`generate-bundle` deploys — but with an MCP entrypoint instead of an
MLflow AgentServer HTTP entrypoint. OBO tokens from the caller flow
through the graph unchanged: downstream Genie / Vector Search / UC
function calls run as the caller, not as the App's service principal.

> **Migrating from an older release?** The MCP server was rewritten to
> an agent-as-tool model in PR #154 (one MCP tool per app instead of
> per-tool fan-out). See the "agent-as-tool refactor" entry in
> [`CHANGELOG.md`](../CHANGELOG.md) for the full set of removed modules
> (`dao_ai.mcp.service`, `dao_ai.mcp.adapters/`, `AppModel.mcp_only`)
> and the requirement that `config.app.name` is now mandatory.

---

## Quickstart

```bash
# 1. Install the optional MCP extra
pip install 'dao-ai[mcp]'

# 2. Generate a deploy-ready bundle from your dao-ai config
dao-ai generate-mcp \
  -c my_agent.yaml \
  -o ./my-agent-mcp \
  -p <profile>

# 3. Deploy to Databricks Apps
cd my-agent-mcp
uv sync
databricks bundle deploy -t dev -p <profile>
databricks bundle run <app-name> -t dev -p <profile>
```

The deployed App name is `config.app.name`. The MCP tool exposed at
`/mcp` is a slugified form of the same value; its description is
`config.app.description`. Point any MCP client at
`https://<app-url>/mcp`.

---

## Configuration requirements

Your dao-ai config needs:

- `app.name` — Databricks App name **and** MCP tool name (slugified).
- `app.description` — recommended; surfaced as the MCP tool description.
- `app.deployment_target: apps` — the MCP server runs on Databricks Apps.
- At least one agent (or an `orchestration.deep_agent` block) — the
  server calls `AppConfig.as_responses_agent()` at boot.
- Any resources the agent needs (Genie rooms, Vector Search indexes,
  Lakebase, warehouses, models) — same as `generate-bundle`.

The MCP server prefers `mcp-`-prefixed app names because Databricks
Multi-Agent Supervisor pattern-matches that prefix when auto-discovering
MCP-hosted Apps across an account.

---

## Architecture

```
┌───────────────────────────────┐
│   FastAPI + RequestContext    │  /healthz, /readyz
│       middleware              │  captures every request header into a
│        │                      │  contextvar (x-forwarded-access-token,
│        ▼                      │  x-request-id, ...)
│   FastMCP (Streamable HTTP)   │  /mcp   (stateless, json_response)
│   ── session_manager driven   │
│      by FastAPI.lifespan      │
│        │                      │
│        ▼                      │
│   register_agent_as_tool      │  builds config.as_responses_agent()
│        │                      │  once at boot; registers one tool
│        ▼                      │  named after config.app.name
│   invoke_agent(input) ────────┼─► dao-ai agent graph
│                               │      · headers forwarded via
│                               │        custom_inputs.configurable.headers
│                               │      · downstream OBO honored by
│                               │        IsDatabricksResource.workspace_client_from
└───────────────────────────────┘
```

The single MCP tool accepts either a plain string (wrapped into a single
user turn internally) or a Responses-style input array. It returns the
final assistant message text via the MCP `tools/call` response.

When the server is configured with `app.mcp_server:` (see
[Server-side capabilities](#server-side-capabilities)) it can additionally
emit `notifications/progress` and `notifications/message` frames during
a call. Clients that opt in via `McpFunctionModel.capabilities` (see
[Consuming this MCP server from another dao-ai agent](#consuming-this-mcp-server-from-another-dao-ai-agent))
receive those envelopes and forward them to their outer response stream.
See [`docs/mcp-callbacks.md`](./mcp-callbacks.md) for the wire format.

### Response shape

Every `tools/call` response is a `CallToolResult` with:

| Field | Purpose |
|---|---|
| `content[0].text` | Plain-text final assistant message — for legacy MCP clients that ignore `structuredContent`. |
| `structuredContent` | `AgentInvocationResult` (schema advertised on `outputSchema`): `final_message`, `trace_id` (fully-qualified UC location — e.g. `trace:/catalog.schema.prefix/<hex>`), `conversation_id` (resolved conversation key — see [Stateful conversations](#stateful-conversations)), `thread_id` (alias of `conversation_id` for LangGraph-native callers), `confidence` (reserved). |
| `_meta.databricks.trace_id` | Same UC-qualified trace id as `structuredContent.trace_id`. Copy so schema-unaware callers can still jump to the MLflow trace. |
| `_meta.conversation_id` | Same resolved conversation key as `structuredContent.conversation_id`. Emitted unnamespaced (not under `databricks.`) to stay symmetric with the inbound `_meta.conversation_id` channel — it is a cross-cutting concept, not a Databricks-specific field. |
| `_meta.databricks.experiment_id` | Bound experiment id from the runtime `MLFLOW_EXPERIMENT_ID` env. |
| `_meta.databricks.model` | Primary agent's `model.name` from the config. |
| `_meta.databricks.latency_ms` | Wall-clock around `agent.apredict()`. |
| `_meta.databricks.request_id` | Server-assigned request id (matches server logs). |
| `_meta.databricks.obo_present` | `true` when the caller forwarded an `x-forwarded-access-token`. Useful for verifying OBO passthrough without a diagnostic tool. |
| `isError` | `true` when `agent.apredict()` raises. Content still reaches the caller LLM (unlike a JSON-RPC error which strips content). |

### Experiment provisioning

`generate-mcp` emits an MLflow experiment resource in the DAB — parity
with `generate-bundle`. Behaviour:

- If `config.app.experiment` is set → binds by literal experiment id
  (and if `manage_permissions: false`, requests `CAN_READ` only).
- Otherwise → declares
  `experiments.<app-name>-experiment.name = /Users/${workspace.current_user.userName}/<app-name>`
  in the top-level `experiments:` block and binds via
  `${resources.experiments.<key>.id}` so DABs materializes + grants
  `CAN_EDIT` to the App SP.

The emitted `app.yaml` sets `MLFLOW_EXPERIMENT_ID: valueFrom: experiment`
(camelCase — the Apps runtime consumes this file directly, DABs isn't in
the loop). When `config.app.trace_location` is set,
`MLFLOW_TRACING_SQL_WAREHOUSE_ID` is also injected as a literal id, and
the trace warehouse is added to the App's `resources` list so the
platform grants the App SP `CAN_USE` on it.

### OBO passthrough

Verified end-to-end. To observe: check the MCP server's app logs for a
line like:

```
mcp.agent_tool.invoke.headers | request_id=<uuid> | obo_present=True |
  obo_token_fingerprint=<sha256[:16]> | obo_token_subject=<sub-claim> |
  header_count=<n>
```

`obo_token_subject` is a best-effort decode of the JWT `sub` claim (no
signature verification — diagnostic only). Compare across calls:

- Direct HTTP call from a Databricks user → `sub` = user email.
- Nested call from a dao-ai consumer app with
  `on_behalf_of_user: true` on the MCP tool → `sub` = user email
  (identity flows through).
- Nested call with `on_behalf_of_user: false` → `sub` = the consumer
  App's SP `client_id`.

The raw token itself is never logged.

### Server-side capabilities

When `config.app.mcp_server:` is set, dao-ai's MCP server advertises
extra capabilities beyond the single agent-as-tool.

```yaml
app:
  name: mcp-dao-ai-retail
  description: Retail assistant exposed as an MCP tool.
  mcp_server:
    progress: true                        # notifications/progress from LangGraph astream_events
    logging: true                         # forward Python logger records as notifications/message
    resources:                            # static resources listed on resources/list
      - uri: dao-ai://prompts/system
        name: system-prompt
        description: Curated system prompt shipped with the server.
        mime_type: text/plain
        content: |
          You are a retail-aware assistant. Answer concisely.
    prompts:                              # prompt templates listed on prompts/list
      - name: greet_customer
        description: Greeting for a returning customer.
        template: "Welcome back, {customer_name}!"
        arguments:
          - name: customer_name
            description: The customer's first name.
            required: true
```

| Field | Meaning |
|---|---|
| `progress` | Emit `notifications/progress` from `LangGraph.astream_events` during the agent call. Requires the caller to supply a `progressToken` via `_meta` on `tools/call`. Default `true`. |
| `logging` | Route Python `logger` records into `notifications/message` on the active FastMCP session. Silent no-op when no session context is bound. Default `true`. |
| `resources` | Static resources published via `resources/list`. Empty list means no resources are advertised. |
| `prompts` | Prompt templates published via `prompts/list`. Clients call `prompts/get` with argument values; the server returns the rendered template as a single user-role message. Placeholders use Python format-string syntax (`{name}`). |

Enabling `progress` or `logging` opts the FastMCP transport into
stateful streamable-HTTP (`stateless_http=False`) so notifications can
be correlated to the caller's session.

When `mcp_server:` is unset (the default), the server publishes only the
single agent-as-tool surface — no resources, no prompts, no
notifications.

### Stateful conversations

By default the MCP surface is **stateless**: each `tools/call` on
`invoke_agent` is treated as an isolated turn and the LangGraph
checkpointer sees a fresh UUID every time. To maintain conversation
history across turns, supply a stable **conversation key** on every call
via one of four channels (highest-precedence wins):

1. **Tool argument** — `conversation_id` (or the `thread_id` alias)
   on `invoke_agent`. This is the discoverable, schema-advertised
   channel. Most MCP hosts and orchestrator clients should use this.
   ```python
   # mcp Python client
   result = await session.call_tool(
       "invoke_agent",
       arguments={"input": "what did I say earlier?", "conversation_id": "abc-123"},
   )
   ```

2. **`_meta` on the `tools/call` request** — MCP-native side channel.
   `RequestParams.Meta` allows arbitrary extra fields and is surfaced
   to FastMCP tool handlers via `ctx.request_context.meta`. Preferred
   over the HTTP header for MCP-native callers because it travels in
   the JSON-RPC message rather than transport headers.
   ```json
   {
     "jsonrpc": "2.0",
     "method": "tools/call",
     "params": {
       "name": "invoke_agent",
       "arguments": {"input": "what did I say earlier?"},
       "_meta": {"conversation_id": "abc-123"}
     }
   }
   ```

3. **`X-Databricks-Conversation-Id` HTTP header** — useful for reverse
   proxies or gateways that want to inject the key without touching
   the JSON-RPC payload. **Note**: this is a dao-ai convention, not a
   Databricks-wide standard. It matches the existing `x-databricks-*`
   header family for consistency.
   ```bash
   curl -H "X-Databricks-Conversation-Id: abc-123" \
        -H "Content-Type: application/json" \
        <mcp-url>/mcp
   ```

4. **Nothing supplied** — the agent generates a UUID downstream. The
   resolved id is still echoed on the response so the caller can
   capture it and pin subsequent turns to the same conversation.

The resolved conversation key is always echoed on the response, so a
caller that lets the server generate the id can persist the returned
value and reuse it next turn:

- `structuredContent.conversation_id` (Databricks-native name)
- `structuredContent.thread_id` (alias — same value)
- `_meta.conversation_id` (unnamespaced, symmetric with inbound `_meta.conversation_id`; other `_meta.databricks.*` fields stay namespaced because those are Databricks-specific observability data)

The `mcp.agent_tool.invoke.headers` server log records
`conversation_id_source ∈ {"arg", "meta", "header", null}` and the
resolved id so operators can distinguish stateless vs stateful calls
without decoding trace state.

#### Observability: querying MLflow traces by conversation

The resolved conversation key is written to the trace's
``mlflow.trace.session`` metadata by the ResponsesAgent — a standard
MLflow field, so downstream trace-search tools (MLflow UI, the REST
``mlflow/traces`` endpoint, ``search_traces`` filters) can group turns
by conversation with no additional wiring. For example, all turns from
a single conversation share the same ``mlflow.trace.session`` value
even though each turn produces its own trace with a distinct
``trace_id``. Verified against a deployed fevm App: a two-turn HEADER
flight and a two-turn META flight each produced two traces sharing the
same session, while a control flight with no supplied id produced two
traces with distinct auto-generated sessions.

#### What server-side state actually requires

`conversation_id` is a **key**, not the storage. Persisting per-turn
state requires the deployed dao-ai instance to have a Lakebase-backed
checkpointer configured on the AppConfig `memory` block. Without it,
supplying a `conversation_id` is a no-op beyond consistent trace
correlation (multiple turns will share the same trace `thread_id`
attribute but no LangGraph state is loaded).

#### Trust model

A bare `conversation_id` is bearer-equivalent: any caller with access
to this endpoint can read or write any conversation whose id they know.
This matches how `conversation_id` on Databricks Apps `/invocations`
already works. Keep the MCP endpoint behind OBO / SP auth and treat
generated ids as secrets in client code.

#### What is NOT used, and why

- **`Mcp-Session-Id`** (the Streamable HTTP transport header) is
  **not** used for conversation continuity. It is per-HTTP-connection,
  in-memory only by default, and the server issues a new one on client
  reconnect unless the client explicitly re-sends the previous value
  *and* the server still remembers it. Tying conversation state to it
  would silently reset user memory on every reconnect. Use one of the
  four channels above instead.

---

## OBO / user headers

Every incoming HTTP request has its headers captured by
`RequestContextMiddleware` and stored in a contextvar. The tool
implementation copies them into
`ResponsesAgentRequest.custom_inputs.configurable.headers`, which flows
into the agent graph's `Context`. From there, dao-ai's existing
`IsDatabricksResource.workspace_client_from(context)` pattern picks up
`x-forwarded-access-token` and issues OBO'd clients for any resource
that declares `on_behalf_of_user: true`.

This means an MCP client that forwards a user bearer token gets:

- Genie calls run as the user
- Vector Search / UC function calls run as the user
- Lakebase writes attributed to the user

with **no code changes** in the agent config beyond the usual
`on_behalf_of_user` flags.

---

## Deployment artifacts

`dao-ai generate-mcp` emits:

```
output/
├── databricks.yaml    # DAB with bundle.engine: direct; App + bound resources
├── resources/app.yml  # command: ["python", "-m", "dao_ai.mcp.server"]
├── pyproject.toml     # dao-ai[mcp]
├── requirements.txt   # dao-ai[mcp]  (Apps build phase installs from this)
├── <your-config>.yaml # rendered with parameters: stripped
└── README.md          # generated deploy snippet + exposed tool name
```

The Apps build phase runs `pip install -r requirements.txt`, which
installs `dao-ai[mcp]` (dao-ai plus the fastapi + uvicorn extras) from
public PyPI. The runtime command `["python", "-m", "dao_ai.mcp.server"]`
launches the server via module invocation — no console script required.

### CLI flags

| Flag | Meaning |
|---|---|
| `-c / --config` | Path to your dao-ai config YAML. |
| `-o / --output-dir` | Where to write the bundle. |
| `--force` | Overwrite existing files in the output directory. |
| `--development` | Build the local dao-ai wheel and bundle it under `output/dist/`; the generated pyproject installs from there. Use when dao-ai changes haven't shipped to PyPI yet. |
| `-p / --profile` | Databricks CLI profile — drives `_resolve_all_resources` so generated bundle paths (e.g. Lakebase database IDs) come from the target workspace. Always pass this when your config references resources resolved at generate time. |
| `--var KEY=VALUE` / `--param KEY=VALUE` | Override declared `${var.KEY}` substitutions. Repeatable. |

### App resource bindings

`generate-mcp` derives App resource bindings via
`dao_ai.apps.resources.generate_app_resources` and converts them with
`dao_ai.apps.bundle._convert_to_bundle_resources`. The resulting
`databricks.yml` declares whatever bindings the agent needs — genie
space, sql warehouse, postgres, uc securable, serving endpoint, secret —
identical to what `generate-bundle` emits.

Lakebase `database_instances` are **not** auto-provisioned. The bundle
assumes the Lakebase project already exists.

---

## Verifying a deploy

```bash
APP_URL=$(databricks apps get <app-name> -p <profile> --output json | jq -r .url)
TOKEN=$(databricks auth token -p <profile> | jq -r .access_token)

# Boot probe
curl -sS -H "Authorization: Bearer $TOKEN" "$APP_URL/readyz"

# MCP handshake — initialize + list tools
curl -sS -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"smoke","version":"1"}}}' \
  "$APP_URL/mcp"

curl -sS -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  "$APP_URL/mcp"

# Call the agent
curl -sS -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"<slugified-app-name>","arguments":{"input":"What are our top-selling SKUs?"}}}' \
  "$APP_URL/mcp"
```

The Apps logs (`databricks apps logs <app-name> -p <profile>`) include
structured loguru events for boot (`mcp.agent_tool.registered`) and
every tool call (dao-ai's existing graph / span emissions).

---

## Consuming this MCP server from another dao-ai agent

Use dao-ai's first-class `type: mcp` tool to call the deployed server
from a consumer config. `type: app` is *not* the right shape — the
`AppToolModel` validator explicitly rejects apps whose name starts with
`mcp-` (which every dao-ai MCP server does, since the default deployed
name is `mcp-dao-ai-<...>` for MAS discovery) and directs you here.

```yaml
resources:
  apps:
    retail_mcp: &retail_mcp
      name: mcp-dao-ai-retail          # the deployed MCP app's name
      on_behalf_of_user: true          # forwards the caller's user token

tools:
  retail_agent:                        # your binding name (free choice)
    type: mcp
    app: *retail_mcp                   # points at the mcp- app above
```

Alternate URL-direct form (useful when the target is outside your
workspace or you want explicit SP auth):

```yaml
tools:
  retail_agent:
    type: mcp
    url: https://mcp-dao-ai-retail.<workspace>.azuredatabricksapps.com/mcp/
    client_id: *client_id
    client_secret: *client_secret
    workspace_host: *workspace_host
```

Neither form takes a `tool:` argument — the MCP server exposes its tool
set on the MCP protocol and the client discovers it. dao-ai's MCP server
registers a single agent-as-tool named from the slugified server-side
`app.name`.

The consumer's OBO flow automatically forwards `x-forwarded-access-token`
to the deployed MCP server, which then propagates it to the nested
agent's downstream calls. Traces from both tiers land in the consumer's
MLflow experiment (set `app.trace_location` on both sides for a unified
UC OTEL table).

### Client capabilities

Opt into advanced MCP behaviors by setting `capabilities:` on the
consumer-side `McpFunctionModel`. Every field is opt-in; leaving
`capabilities:` unset preserves the classic `MultiServerMCPClient`
path with no callbacks or interceptors (byte-for-byte compatible with
pre-capabilities dao-ai).

```yaml
tools:
  retail_agent:
    type: mcp
    app: *retail_mcp
    capabilities:
      progress: true                    # consume notifications/progress
      logging: true                     # consume notifications/message
      structured_output: true           # prefer CallToolResult.structuredContent (default true)
      elicitation: hitl                 # server elicit → LangGraph interrupt
      sampling:                         # server sampling/createMessage
        endpoint: *reasoning_endpoint   # LLM used to satisfy the request
        max_iterations: 3
        allow_tool_use: false
      roots:                            # URI roots advertised on roots/list
        - uri: databricks:///Volumes/prod/main/retail
          name: retail-volume
```

| Field | Meaning |
|---|---|
| `progress` | Consume `notifications/progress`, forward as MLflow span events and (during streaming) as `response.output_item.added` envelopes on the outer stream. Default `false`. |
| `logging` | Consume `notifications/message` (and any custom `notifications/<method>`). Same dual-emission as `progress`. Default `false`. |
| `structured_output` | Prefer `CallToolResult.structuredContent` and expand `resource_link` items into MLflow span attributes via a `ToolCallInterceptor`. Additive — falls back to text extraction when structured content is absent. Default `true`. |
| `elicitation` | Handle server-initiated `elicitation/create`. `hitl` raises a LangGraph interrupt (surfaces via the standard HITL flow, resumes via `custom_inputs.decisions`); `reject` returns `action="cancel"` without prompting. Default `None`. |
| `sampling` | Handle server-initiated `sampling/createMessage` by routing to the referenced inference endpoint through AI Gateway. `max_iterations` caps nested sampling calls; `allow_tool_use` (default `false`) gates whether a sampling call may itself request tool use. |
| `roots` | URI roots advertised to the server on `roots/list`. Empty list disables the capability. |

Setting `sampling` or a non-empty `roots` drops the client to a raw
`mcp.client.session.ClientSession` transport since those callbacks are
not surfaced by langchain-mcp-adapters. `progress`, `logging`,
`elicitation`, and `structured_output` all work under the classic
adapter path.

For the wire format of progress and logging envelopes on the outer
response stream, see [`docs/mcp-callbacks.md`](./mcp-callbacks.md).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `POST /mcp/` returns 307. | The MCP route is at `/mcp` (no trailing slash); requests to `/mcp/` get FastAPI's slash-redirect. | Strip the trailing slash, or follow the redirect. |
| `Task group is not initialized` 500 on every request. | FastMCP's session manager wasn't started. | Already handled in `dao_ai.mcp.server.build_app` via FastAPI lifespan. |
| `INVALID_PARAMETER_VALUE: Failed to register tools from Databricks App MCP server '<name>'` from MAS. | MAS POSTs to `<app-url>/mcp` (no trailing slash) and treats anything other than 200 as failure. | Confirm with `curl -X POST <app-url>/mcp -d '{"jsonrpc":"2.0","id":1,"method":"initialize",...}'`. Expect a 200. |
| Downstream tool spans show the App SP identity, not the caller. | The caller isn't forwarding a bearer token, or the resource models don't set `on_behalf_of_user: true`. | Ensure the MCP client sets `Authorization: Bearer <user-token>` and the config sets `on_behalf_of_user: true` on the relevant Genie / VS / warehouse / database models. |
| `trace_id=null` on every response. | App isn't bound to an MLflow experiment. | Set `app.trace_location` (UC schema) and `app.experiment` in the config. |

---

## Related modules

- `dao_ai.mcp.agent_tool` — the single-tool registration surface.
- `dao_ai.mcp.server` — FastAPI + FastMCP entrypoint.
- `dao_ai.mcp.generate` — `write_mcp_bundle` bundle emission.
- `dao_ai.apps.bundle.write_bundle` — the `dao-ai generate-bundle` entry
  point for the standard AgentServer HTTP deployment. `write_mcp_bundle`
  mirrors its shape.
- `dao_ai.apps.mcp` — **client** primitives for consuming external MCP
  servers from a dao-ai agent.
