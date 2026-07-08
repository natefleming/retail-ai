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
final assistant message text — non-streaming; MCP progress events aren't
emitted.

### Response shape

Every `tools/call` response is a `CallToolResult` with:

| Field | Purpose |
|---|---|
| `content[0].text` | Plain-text final assistant message — for legacy MCP clients that ignore `structuredContent`. |
| `structuredContent` | `AgentInvocationResult` (schema advertised on `outputSchema`): `final_message`, `trace_id` (fully-qualified UC location — e.g. `trace:/catalog.schema.prefix/<hex>`), `confidence` (reserved). |
| `_meta.databricks.trace_id` | Same UC-qualified trace id as `structuredContent.trace_id`. Copy so schema-unaware callers can still jump to the MLflow trace. |
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
├── databricks.yml     # DAB with bundle.engine: direct; App + bound resources
├── app.yaml           # command: ["dao-ai-mcp-server"]
├── pyproject.toml     # dao-ai[mcp]>=<version>
├── <your-config>.yaml # rendered with parameters: stripped
└── README.md          # generated deploy snippet + exposed tool name
```

After `generate-mcp`, run `uv sync` in the output directory to produce
`uv.lock`. Databricks Apps' native uv support then activates at deploy:
BUILD runs `uv sync --locked --no-dev`, and the runtime command
`["dao-ai-mcp-server"]` invokes the console script from `.venv/bin/`.

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

## Consuming the deployed MCP server from another dao-ai agent

Use dao-ai's first-class `type: app` MCP tool in a consumer config to
call this deployed server:

```yaml
tools:
  retail_agent:
    type: app
    args:
      app: <deployed-mcp-app-name>
      tool: <slugified-app-name>
```

The consumer's OBO flow automatically forwards
`x-forwarded-access-token` to the deployed MCP server, which then
propagates it to the nested agent's downstream calls. Traces from both
tiers land in the consumer's MLflow experiment (set
`app.trace_location` on both sides for a unified UC OTEL table).

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
