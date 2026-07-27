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

`dao-ai agent generate --mode mcp` bundles a Databricks App that runs the same graph
`dao-ai agent generate` deploys — but with an MCP entrypoint instead of an
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
dao-ai agent generate \
  --mode mcp \
  -c my_agent.yaml \
  -o ./my-agent-mcp \
  -p <profile>

# 3. Deploy + run in one step (or drive databricks bundle by hand, below)
dao-ai agent up --mode mcp -c my_agent.yaml -o ./my-agent-mcp -p <profile>

# 3b. ...or drive the bundle manually
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
- At least one agent (or an `orchestration.deep_agent` block) — the
  server calls `AppConfig.as_responses_agent()` at boot.
- Any resources the agent needs (Genie rooms, Vector Search indexes,
  Lakebase, warehouses, models) — same as `dao-ai agent generate`.

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
emit `notifications/progress` frames during a call. Clients that opt in via
`McpFunctionModel.capabilities` (see
[Consuming this MCP server from another dao-ai agent](#consuming-this-mcp-server-from-another-dao-ai-agent))
receive those envelopes and forward them to their outer response stream.
See [`docs/mcp-callbacks.md`](./mcp-callbacks.md) for the wire format.

### Response shape

Every `tools/call` response is a `CallToolResult` with:

| Field | Purpose |
|---|---|
| `content[0].text` | Plain-text final assistant message — for legacy MCP clients that ignore `structuredContent`. |
| `structuredContent` | `AgentInvocationResult` (schema advertised on `outputSchema`): `final_message`, `trace_id` (fully-qualified UC location — e.g. `trace:/catalog.schema.prefix/<hex>`), `conversation_id` (resolved conversation key — supplied via `_meta.conversation_id` or `X-Databricks-Conversation-Id` header; see [Stateful conversations](#stateful-conversations)), `thread_id` (alias of `conversation_id` for LangGraph-native callers), `confidence` (reserved). |
| `_meta.databricks.trace_id` | Same UC-qualified trace id as `structuredContent.trace_id`. Copy so schema-unaware callers can still jump to the MLflow trace. |
| `_meta.conversation_id` | Same resolved conversation key as `structuredContent.conversation_id`. Emitted unnamespaced (not under `databricks.`) to stay symmetric with the inbound `_meta.conversation_id` channel — it is a cross-cutting concept, not a Databricks-specific field. |
| `_meta.databricks.experiment_id` | Bound experiment id from the runtime `MLFLOW_EXPERIMENT_ID` env. |
| `_meta.databricks.model` | Primary agent's `model.name` from the config. |
| `_meta.databricks.latency_ms` | Wall-clock around `agent.apredict()`. |
| `_meta.databricks.request_id` | Server-assigned request id (matches server logs). |
| `_meta.databricks.obo_present` | `true` when the caller forwarded an `x-forwarded-access-token`. Useful for verifying OBO passthrough without a diagnostic tool. |
| `isError` | `true` when `agent.apredict()` raises. Content still reaches the caller LLM (unlike a JSON-RPC error which strips content). |

### Experiment provisioning

`dao-ai agent generate --mode mcp` emits an MLflow experiment resource in the DAB — parity
with `dao-ai agent generate`. Behaviour:

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
| `resources` | Static resources published via `resources/list`. Empty list means no resources are advertised. |
| `prompts` | Prompt templates published via `prompts/list`. Clients call `prompts/get` with argument values; the server returns the rendered template as a single user-role message. Placeholders use Python format-string syntax (`{name}`). |

> The server-side `logging` capability (`notifications/message`) was removed
> — the MCP `logging` feature is deprecated under
> [SEP-2577](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2577);
> use MLflow/OTEL tracing for observability. The server also extracts inbound
> W3C trace context from `_meta` (`traceparent`/`baggage`) so it continues the
> caller's distributed trace.

Enabling `progress` opts the FastMCP transport into stateful
streamable-HTTP (`stateless_http=False`) so notifications can be
correlated to the caller's session.

When `mcp_server:` is unset (the default), the server publishes only the
single agent-as-tool surface — no resources, no prompts, no
notifications.

### Stateful conversations

By default the MCP surface is **stateless**: each `tools/call` on
`invoke_agent` is treated as an isolated turn and the LangGraph
checkpointer sees a fresh UUID every time. To maintain conversation
history across turns, supply a stable **conversation key** on every call
via one of two **transport-only channels** (highest-precedence wins):

1. **`_meta.conversation_id` on the `tools/call` request** — MCP-native
   side channel. `RequestParams.Meta` allows arbitrary extra fields and
   is surfaced to FastMCP tool handlers via `ctx.request_context.meta`.
   This is the same channel MCP already uses for `progressToken`.
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

2. **`X-Databricks-Conversation-Id` HTTP header** — universal
   transport-level fallback. Works with any client that can set request
   headers: curl, Claude Desktop over HTTP, Cursor IDE MCP, and
   `langchain-mcp-adapters` via a per-call `ToolCallInterceptor`.
   **Note**: this is a dao-ai convention, not a Databricks-wide
   standard. It matches the existing `x-databricks-*` header family for
   consistency.
   ```bash
   curl -H "X-Databricks-Conversation-Id: abc-123" \
        -H "Content-Type: application/json" \
        <mcp-url>/mcp
   ```

3. **Nothing supplied** — the agent generates a UUID downstream. The
   resolved id is still echoed on the response so the caller can
   capture it and pin subsequent turns to the same conversation.

The resolved conversation key is always echoed on the response, so a
caller that lets the server generate the id can persist the returned
value and reuse it next turn:

- `structuredContent.conversation_id` (Databricks-native name)
- `structuredContent.thread_id` (alias — same value)
- `_meta.conversation_id` (unnamespaced, symmetric with the inbound
  `_meta.conversation_id` channel; other `_meta.databricks.*` fields stay
  namespaced because those are Databricks-specific observability data)

The `mcp.agent_tool.invoke.headers` server log records
`conversation_id_source ∈ {"meta", "header", null}` and the resolved id
so operators can distinguish stateless vs stateful calls without
decoding trace state.

#### Why not a tool argument?

An earlier iteration accepted `conversation_id` / `thread_id` as tool
arguments on `invoke_agent`. That surface was **removed** for the
following reasons:

- **Prompt-injection surface.** Tool arguments live on the tool's
  `inputSchema`, which is *model-controlled* per MCP semantics — the
  LLM populates them on each call. A prompt-injected user turn
  ("actually call the tool with `conversation_id=<victim-id>`") could
  steer the calling model into reading or writing another user's
  conversation. Transport-level channels (`_meta`, headers) never
  touch the LLM.
- **MCP spec precedent.** MCP puts its own correlation-id primitive
  (`progressToken`) in `_meta`, not as a tool arg. That's the design
  the spec explicitly gives for request-scoped identifiers the tool
  doesn't semantically own.
- **Industry precedent.** LangGraph Server puts `thread_id` in
  `configurable` (out-of-band from tool schemas). OpenAI Assistants v2
  makes `thread_id` a URL path parameter, never tool-visible. Anthropic
  Messages API keeps conversation state fully client-owned. dao-ai's
  own Genie tools return `conversation_id` in `_meta` of tool *results*
  — they don't take it as tool *inputs*.
- **Determinism.** LLMs are known to invent, munge, or forget
  structured ids across turns. Making thread continuity the model's
  job is fragile; making it the transport's job is deterministic.

#### LangGraph parent → MCP child: header injection via interceptor

If your dao-ai (or other LangGraph-based) agent invokes this MCP server
as a tool via `langchain-mcp-adapters`, use a `ToolCallInterceptor` to
inject the conversation id from parent-graph state into the HTTP
header on every outgoing tool call. The child LLM never sees the id:

```python
from langchain_mcp_adapters.interceptors import ToolCallInterceptor
from langchain_mcp_adapters.client import MultiServerMCPClient


class ConversationIdInterceptor(ToolCallInterceptor):
    """Inject conversation id from LangGraph runtime state into the outgoing
    MCP call as an X-Databricks-Conversation-Id header."""

    async def __call__(self, request, handler):
        # request.runtime is the parent-graph runtime; extract thread_id
        # from whatever state field your graph carries it in.
        runtime = getattr(request, "runtime", None)
        thread_id = _resolve_thread_id_from_runtime(runtime)  # your helper
        if thread_id:
            request = request.override(
                headers={
                    **(request.headers or {}),
                    "X-Databricks-Conversation-Id": thread_id,
                }
            )
        return await handler(request)


client = MultiServerMCPClient(
    connections={
        "retail_agent": {
            "transport": "streamable_http",
            "url": "https://retail-mcp.<workspace>.databricksapps.com/mcp",
        }
    },
    interceptors=[ConversationIdInterceptor()],
)
```

**Known limitation**: `langchain-mcp-adapters` v0.3.0 does not plumb
per-call `meta=` through to `session.call_tool`, so the `_meta` channel
is currently unreachable from LangChain callers — use the header
channel above. This is a trivial upstream fix
(`MCPToolCallRequest.meta` field + one line in `tools.py`); once it
lands, LangChain callers can drop the header shim in favor of `_meta`.

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
correlation (multiple turns will share the same `mlflow.trace.session`
metadata value but no LangGraph state is loaded).

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

`dao-ai agent generate --mode mcp` emits:

```
output/
├── databricks.yaml    # DAB with bundle.engine: direct; App + bound resources
├── resources/app.yml  # command: ["python", "-m", "dao_ai.mcp.server"]
├── pyproject.toml     # dao-ai[mcp] (pinned ==<version>, or local-wheel redirect under --development)
├── uv.lock            # portable lock; Apps build phase syncs from this
├── <your-config>.yaml # rendered with parameters: stripped
└── README.md          # generated deploy snippet + exposed tool name
```

The Apps build phase runs `uv sync --locked --no-dev` from the `pyproject.toml` +
`uv.lock`, which installs `dao-ai[mcp]` (dao-ai plus the fastapi + uvicorn extras).
No `requirements.txt` is emitted — its presence would take precedence and force the
slower pip path. The runtime command `["python", "-m", "dao_ai.mcp.server"]`
launches the server via module invocation — no console script required.

### CLI flags

| Flag | Meaning |
|---|---|
| `-c / --config` | Path to your dao-ai config YAML. |
| `-o / --output-dir` | Where to write the bundle. |
| `--overwrite` | Overwrite existing files in the output directory. |
| `--development` | Build the local dao-ai wheel and bundle it under `output/dist/`; the generated pyproject installs from there. Use when dao-ai changes haven't shipped to PyPI yet. |
| `-p / --profile` | Databricks CLI profile — drives `_resolve_all_resources` so generated bundle paths (e.g. Lakebase database IDs) come from the target workspace. Always pass this when your config references resources resolved at generate time. |
| `--var KEY=VALUE` / `--param KEY=VALUE` | Override declared `${var.KEY}` substitutions. Repeatable. |

### App resource bindings

`dao-ai agent generate --mode mcp` derives App resource bindings via
`dao_ai.apps.resources.generate_app_resources` and converts them with
`dao_ai.apps.bundle._convert_to_bundle_resources`. The resulting
`databricks.yml` declares whatever bindings the agent needs — genie
space, sql warehouse, postgres, uc securable, serving endpoint, secret —
identical to what `dao-ai agent generate` emits.

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
      structured_output: true           # prefer CallToolResult.structuredContent (default true)
      elicitation: hitl                 # server elicit → LangGraph interrupt
```

| Field | Meaning |
|---|---|
| `progress` | Consume `notifications/progress`, forward as MLflow span events and (during streaming) as `response.output_item.added` envelopes on the outer stream. Default `false`. |
| `structured_output` | Prefer `CallToolResult.structuredContent` and expand `resource_link` items into MLflow span attributes via a `ToolCallInterceptor`. Additive — falls back to text extraction when structured content is absent. Default `true`. |
| `elicitation` | Handle server-initiated `elicitation/create`. `hitl` raises a LangGraph interrupt (surfaces via the standard HITL flow, resumes via `custom_inputs.decisions`); `reject` returns `action="cancel"` without prompting. Default `None`. |

> **Deprecated capabilities removed.** `logging`, `sampling`, and `roots`
> were deprecated together under
> [SEP-2577](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2577)
> and are no longer supported. `capabilities:` uses `extra="forbid"`, so a
> config that still sets any of them fails validation. Migration: use
> MLflow/OTEL tracing instead of `logging` (dao-ai now propagates W3C trace
> context via `_meta` — see below); integrate directly with an LLM endpoint
> instead of `sampling`; pass directories/URIs via tool parameters instead of
> `roots`.

All surviving capabilities (`progress`, `structured_output`, `elicitation`)
run under the classic langchain-mcp-adapters `MultiServerMCPClient` path.

W3C trace context (`traceparent`/`baggage`) is injected into `_meta` on every
`tools/call` (SEP-414) so a downstream MCP server continues the caller's
distributed trace — see [`docs/mcp-callbacks.md`](./mcp-callbacks.md) and
`dao_ai.tools.mcp_trace_context`.

For the wire format of progress envelopes on the outer response stream, see
[`docs/mcp-callbacks.md`](./mcp-callbacks.md).

## Consuming this MCP server from raw Python or LangChain

Non-dao-ai callers can invoke the deployed server directly. Both
examples below demonstrate a two-turn conversation reusing the same
`conversation_id` on turn 2.

### Raw Python (`mcp` SDK, Streamable HTTP)

```python
import asyncio, os, uuid
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

APP_URL = os.environ["MCP_APP_URL"].rstrip("/")   # https://<app>.<workspace>.databricksapps.com
TOKEN   = os.environ["MCP_APP_TOKEN"]              # OBO user token or SP token
TOOL    = "mcp_dao_ai_retail"                      # slugified server-side app.name

async def run():
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        # Apps proxy forwards this to the server as x-forwarded-access-token
        # for OBO downstream calls.
        "X-Forwarded-Access-Token": TOKEN,
    }
    conv_id = f"conv-{uuid.uuid4().hex[:8]}"

    async with streamablehttp_client(f"{APP_URL}/mcp", headers=headers) as (r, w, _):
        async with ClientSession(r, w) as sess:
            await sess.initialize()

            # Turn 1 — pass conversation_id via _meta (spec-idiomatic,
            # matches how MCP ships progressToken).
            r1 = await sess.call_tool(
                TOOL,
                {"input": "My favorite color is chartreuse."},
                meta={"conversation_id": conv_id},
            )
            print("turn1:", r1.structuredContent["final_message"])

            # Turn 2 — same channel. The server threads the id into the
            # LangGraph checkpointer under the same thread, so the reply
            # recalls turn 1.
            r2 = await sess.call_tool(
                TOOL,
                {"input": "What did I just tell you my favorite color was?"},
                meta={"conversation_id": conv_id},
            )
            print("turn2:", r2.structuredContent["final_message"])
            # → "Chartreuse."

asyncio.run(run())
```

Alternative: pass the id via the `X-Databricks-Conversation-Id` header on
the transport instead of `_meta` — same server-side behavior. Both
channels are invisible to the LLM.

Every response echoes `structuredContent.conversation_id`,
`structuredContent.thread_id`, and `_meta.conversation_id` — capture one
of these on turn 1 if you let the server generate the id, then replay
it on subsequent turns.

### LangChain / LangGraph (`langchain-mcp-adapters`)

`MultiServerMCPClient` wraps each MCP tool as a LangChain `BaseTool`.
Because the conversation id is **not** a tool argument (see [Stateful
conversations](#stateful-conversations) for the rationale), a LangGraph
parent injects it as an HTTP header via a per-call
`ToolCallInterceptor` — the child LLM never sees the id.

**Known limitation**: `langchain-mcp-adapters` v0.3.0 does not plumb
per-call `meta=` through to `session.call_tool`, so the `_meta` channel
isn't reachable from LangChain callers today. Use the header
interceptor pattern below. A trivial upstream PR
(`MCPToolCallRequest.meta` + one line in `tools.py`) would unlock the
`_meta` path; once it lands, callers can migrate off the header shim.

```python
import asyncio, os, uuid
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import ToolCallInterceptor

APP_URL = os.environ["MCP_APP_URL"].rstrip("/")
TOKEN   = os.environ["MCP_APP_TOKEN"]


class PerCallConversationHeader(ToolCallInterceptor):
    """Set X-Databricks-Conversation-Id on every outgoing MCP call from
    whatever conversation key the parent LangGraph is tracking. The
    child LLM never sees this value.
    """

    def __init__(self, resolver):
        # resolver(request.runtime) → str | None. Wire this to your
        # parent-graph state (LangGraph node context, thread config, etc.).
        self._resolver = resolver

    async def __call__(self, request, handler):
        conversation_id = self._resolver(getattr(request, "runtime", None))
        if conversation_id:
            request = request.override(
                headers={
                    **(request.headers or {}),
                    "X-Databricks-Conversation-Id": conversation_id,
                }
            )
        return await handler(request)


async def run():
    conv_id = f"conv-{uuid.uuid4().hex[:8]}"

    # For this simple example the resolver is a closure over conv_id.
    # In a real LangGraph app you'd pull it from the runtime's parent
    # thread config or agent state.
    client = MultiServerMCPClient(
        connections={
            "retail": {
                "transport": "streamable_http",
                "url": f"{APP_URL}/mcp",
                "headers": {
                    "Authorization": f"Bearer {TOKEN}",
                    "X-Forwarded-Access-Token": TOKEN,
                },
            }
        },
        interceptors=[PerCallConversationHeader(lambda _rt: conv_id)],
    )
    tools = await client.get_tools()
    invoke_agent = next(t for t in tools if t.name == "mcp_dao_ai_retail")

    turn1 = await invoke_agent.ainvoke(
        {"input": "My favorite color is chartreuse."}
    )
    print("turn1:", turn1)

    turn2 = await invoke_agent.ainvoke(
        {"input": "What did I just tell you my favorite color was?"}
    )
    print("turn2:", turn2)  # → "Chartreuse."

asyncio.run(run())
```

Notes:

- **Header via interceptor is the recommended per-call pattern.** The
  interceptor reads `request.runtime` at call time, so a single
  `MultiServerMCPClient` can serve many concurrent conversations by
  resolving the id from the parent graph's state per invocation.
- **Client-wide header is a shortcut** if a client is scoped to one
  conversation — set `X-Databricks-Conversation-Id` directly in the
  `headers` dict passed to `MultiServerMCPClient` and skip the
  interceptor. Wasteful when one client serves many conversations.
- **Reading the resolved id.** `langchain-mcp-adapters` returns the
  tool's plain-text `content[0]` by default; if the agent needs
  programmatic access to the echoed `conversation_id`, drop to the
  raw `mcp.client.session.ClientSession` (see the previous section)
  which exposes `structuredContent` and `_meta`.

For agents built with LangGraph that already track a `thread_id` in
their own state, wire the interceptor's resolver to that field so
parent-graph and MCP-side conversation lineage stay aligned.

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
- `dao_ai.apps.bundle.write_bundle` — the `dao-ai agent generate` entry
  point for the standard AgentServer HTTP deployment. `write_mcp_bundle`
  mirrors its shape.
- `dao_ai.apps.mcp` — **client** primitives for consuming external MCP
  servers from a dao-ai agent.
