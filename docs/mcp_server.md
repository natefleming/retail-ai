# MCP Server for dao-ai

The `dao_ai.mcp` package turns any dao-ai config defining `create_genie_toolkit` or `create_vector_search_tool` factories into a [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server hosted on Databricks Apps. Any MCP client — Claude Desktop, Cursor, agent platforms, dao-ai itself — can connect to a deployed app and call those tools natively over [Streamable HTTP](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports).

This is the **server** side. dao-ai's existing `dao_ai.apps.mcp` module is the client side (primitives for *consuming* external MCP servers from a dao-ai agent). The two are independent.

---

## Why

You already have:

- a dao-ai config that wires Genie spaces, vector-search retrievers, the cache chain, and OBO/SP auth via Pydantic models;
- factories like `dao_ai.tools.create_genie_toolkit` and `dao_ai.tools.create_vector_search_tool` that produce battle-tested tool implementations (LRU + pg_vector semantic cache, query decomposition + RRF + FlashRank + instruction-aware reranking + verifier).

`dao-ai generate-mcp` lets you ship those exact tools as an MCP endpoint without writing or maintaining a parallel server. One command, one app, all the dao-ai retrieval features at the other end of an MCP client.

---

## Quickstart

```bash
# 1. Install the optional MCP extra
pip install 'dao-ai[mcp]'

# 2. Generate a deploy-ready bundle from an existing dao-ai config
dao-ai generate-mcp \
  -c config/examples/15_complete_applications/sporting_goods_store_mcp.yaml \
  -o ./sporting-goods-mcp \
  --var "warehouse_id=<id>" \
  --var "merchandising_genie_space_id=<uuid>" \
  --var "sales_pricing_genie_space_id=<uuid>" \
  --var "vector_search_endpoint=<endpoint-name>" \
  --var "products_index_name=<catalog>.<schema>.products_description_index"

# 3. Deploy to Databricks Apps
cd sporting-goods-mcp
databricks bundle deploy -t dev -p <profile>
databricks bundle run mcp_dao_ai -t dev -p <profile>
```

The default Databricks App name is `mcp-dao-ai` (with the underscore-form `mcp_dao_ai` for the bundle resource key). The `mcp-` prefix is a discovery signal for Databricks Multi-Agent Supervisor (MAS), which pattern-matches it when enumerating MCP-hosted Apps across an account. Override the name explicitly by setting `app.name` in your dao-ai config.

The bundle ships with `bundle.engine: direct`. The app exposes a single Streamable HTTP endpoint at `/mcp/` and serves `/healthz` + `/readyz` for platform probes.

Point any MCP client at `https://<app-url>/mcp/`.

---

## Architecture

### Discovery

The server boots, reads its config via `dao_ai.config.AppConfig.from_file(initialize=False)`, walks `config.tools`, and dispatches each entry by its `function.name` to a registered adapter. Tools whose factory has no adapter are skipped with a warning.

```
┌───────────────────────────────┐
│   FastAPI + RequestContext    │  /healthz, /readyz
│       middleware              │
│        │                      │
│        ▼                      │
│   FastMCP (Streamable HTTP)   │  /mcp/   (stateless, json_response)
│   ── session_manager driven   │
│      by FastAPI.lifespan      │
│        │                      │
│        ▼                      │
│   register_tools_from_config  │  walks AppConfig.tools
│        │                      │
│        ▼                      │
│   adapter.register()          │  per factory_name
└─────────────┬─────────────────┘
              │
              ├─► Genie adapter   ── dao_ai.tools.create_genie_toolkit
              │     rebuilds: Genie → GenieService
              │                       → PostgresContextAwareGenieService
              │                       → LRUCacheService
              │     registers: ask_<name> + <name>_feedback (MCP tools)
              │
              ├─► Vector Search adapter ── dao_ai.tools.create_vector_search_tool
              │     invokes the factory, wraps the returned StructuredTool
              │     registers: <name>
              │
              └─► (your custom adapter)
```

### Adapter registry

Adapters live under `dao_ai/mcp/adapters/`. Each one is a module that calls `register_adapter(...)` at import time:

```python
from dao_ai.mcp.adapters import McpAdapter, register_adapter

def _register(mcp, tool_name, args, workspace_client):
    # build tool implementation from args, register on mcp
    ...

register_adapter(McpAdapter(
    factory_name="dao_ai.tools.create_my_thing",
    register=_register,
))
```

Add the import to `dao_ai/mcp/service.py` (alongside the shipped `genie` and `vector_search` imports), and your factory is now exposed as an MCP tool whenever a config references it.

### Names and descriptions

The MCP tool's **name** is the YAML key (`tools.<name>`). The **description** comes from `args.description` in the YAML — or from the LangChain tool object the factory returns. No MCP-side prefixes or suffixes are added; what the LLM sees is exactly what dao-ai's factory advertises.

### `_meta` contract

Every tool response includes a structured `_meta` block in the JSON payload — invisible to the model, visible to the MCP client.

**`ask_<name>` (Genie):**
```json
{
  "_meta": {
    "tool_name": "merchandising_analytics",
    "space_id": "01f1539922fb...",
    "cache_hit": true,
    "served_by": "merchandising_analytics-lru",
    "latency_ms": 4841,
    "message_id": "01f...",
    "cache_entry_id": null,
    "conversation_id": "01f...",
    "trace_id": null
  }
}
```

**`<name>_feedback` (Genie):**
```json
{ "_meta": { "tool_name": "...", "conversation_id": "...", "message_id": "...", "rating": "NEGATIVE", "was_cache_hit": true } }
```

**Vector search:**
```json
{ "_meta": { "tool_name": "product_vector_search", "result_count": 20, "latency_ms": 4557, "trace_id": null } }
```

The client honors these `_meta` fields on inbound `tools/call`:
- `progressToken` → progress notifications during long cache misses.
- `dao-ai/conversation_id` → forwarded to `ask_question` for multi-turn.
- `dao-ai/disable_cache` → walks straight to the underlying `GenieService`, skipping cache layers.

---

## Configuration

The MCP server's YAML is a (subset of a) dao-ai `AppConfig`. It needs:

- `parameters:` — declared `${var.NAME}` references (CLI overrides, env-var fallback, defaults).
- `resources.genie_rooms`, `resources.warehouses`, `resources.databases`, `resources.vector_stores` — the dao-ai resource models, identical to what the agent runtime consumes.
- `retrievers:` — only needed for vector-search tools.
- `tools:` — one entry per MCP tool to expose. Each `tools.<name>.function` must be a `factory` whose `name` matches a registered adapter (`dao_ai.tools.create_genie_toolkit` or `dao_ai.tools.create_vector_search_tool`).

The `app:` and `agents:` blocks are **intentionally omitted** — the MCP server has no agent runtime to configure. Server name and log level come from env vars `DAO_AI_MCP_SERVER_NAME` (default `mcp-dao-ai` — chosen so Databricks Multi-Agent Supervisor's `mcp-` discovery prefix matches) and `DAO_AI_MCP_LOG_LEVEL` (default `INFO`).

See `config/examples/15_complete_applications/sporting_goods_store_mcp.yaml` for a worked example.

### Cache parameters

A `create_genie_toolkit` tool entry can configure both cache layers:

```yaml
tools:
  merchandising_analytics:
    name: merchandising_analytics
    function:
      type: factory
      name: dao_ai.tools.create_genie_toolkit
      args:
        name: merchandising_analytics
        description: "Query merchandising analytics..."
        genie_room: *merch_room
        lru_cache_parameters:        # outer layer — exact match
          warehouse: *wh
          capacity: 100
          time_to_live_seconds: 3600
        context_aware_cache_parameters:  # middle layer — semantic match
          database: *retail_db
          warehouse: *wh
          embedding_model: *embed
          similarity_threshold: 0.85
          time_to_live_seconds: 86400
```

The chain composes outer-to-inner: `LRUCacheService → PostgresContextAwareGenieService → GenieService`. Each lookup short-circuits on hit. Both layers are optional — drop a block to disable that layer.

The Postgres semantic cache needs `CREATE` privilege on its target schema in the Lakebase database. Most Apps deployments require an explicit grant in your Lakebase project; the cache will gracefully fall back to the underlying `GenieService` on permission errors but won't memoize.

### Vector-search retriever

The vector-search adapter passes the entire `args.retriever` block straight through to `create_vector_search_tool`. Anything that factory supports — query decomposition, FlashRank, instruction-aware reranking, query routing, result verification, metadata filters — works without code changes.

---

## Deployment

`dao-ai generate-mcp` emits:

```
output/
├── databricks.yml        # bundle.engine: direct; App + bound resources
├── app.yaml              # command: ["uv", "run", "dao-ai-mcp-server"]
├── pyproject.toml        # dao-ai[mcp]>=<version>
├── requirements.txt      # uv
├── <your-config>.yaml    # rendered with parameters: stripped
└── README.md             # generated deploy snippet
```

### CLI flags

| Flag | Meaning |
|---|---|
| `-c / --config` | Path to your dao-ai config YAML. |
| `-o / --output-dir` | Where to write the bundle. |
| `--force` | Overwrite existing files in the output directory. |
| `--development` | Build the local dao-ai wheel and bundle it under `output/dist/`; the generated pyproject installs from there. Use this when the MCP server code hasn't shipped to PyPI yet. |
| `-p / --profile` | Databricks CLI profile — drives `_resolve_all_resources` so generated bundle paths (e.g. Lakebase database IDs) come from the target workspace. **Always pass this if your config references resources resolved at generate-time.** |
| `--var KEY=VALUE` / `--param KEY=VALUE` | Override declared `${var.KEY}` substitutions. Repeatable. |

### App resource bindings

`generate-mcp` derives App resource bindings via `dao_ai.apps.resources.generate_app_resources` and converts them with `dao_ai.apps.bundle._convert_to_bundle_resources`. The resulting `databricks.yml` declares:

- `genie_space` (CAN_RUN) — one per `create_genie_toolkit` entry.
- `uc_securable` (TABLE / SELECT) — one per VS index. Vector-search indexes use the `TABLE` securable type since Databricks Apps' resource schema doesn't have a native `VECTOR_SEARCH_INDEX` type.
- `sql_warehouse` (CAN_USE) — for Genie SQL execution and cached-SQL re-execution.
- `postgres` (CAN_CONNECT_AND_CREATE) — for the Lakebase autoscaling project backing the semantic cache.
- `serving_endpoint` (CAN_QUERY) — for embedding and decomposition LLM endpoints.
- `secret` (READ) — for any secret-scope refs in the config.

Lakebase `database_instances` are **not** auto-provisioned. The MCP bundle assumes the Lakebase project already exists (or that you'll add an `database_instances` declaration manually for the rare new-project case). This avoids accidentally re-creating shared instances across deploys.

### Auth

Defaults to **App SP** auth via Databricks Apps' auto-injected `DATABRICKS_CLIENT_ID` / `DATABRICKS_CLIENT_SECRET`. To use **OBO** (the requesting user's identity), set `on_behalf_of_user: true` on the relevant resource model (`GenieRoomModel`, `WarehouseModel`, `VectorStoreModel`, `DatabaseModel`). The MCP server captures `x-forwarded-access-token` per-request and dao-ai's existing `IsDatabricksResource.workspace_client_from(context)` machinery uses it when the flag is set.

The request-context middleware tags every log line with `obo_present=<true|false>` so you can confirm header propagation at a glance.

---

## Custom adapters

To expose a third dao-ai factory as an MCP tool, drop a new module under `dao_ai/mcp/adapters/`:

```python
# dao_ai/mcp/adapters/my_thing.py
from typing import Any
from databricks.sdk import WorkspaceClient
from mcp.server.fastmcp import FastMCP

from dao_ai.mcp.adapters import McpAdapter, register_adapter

FACTORY = "dao_ai.tools.create_my_thing"

def _register(
    mcp: FastMCP,
    tool_name: str,
    args: dict[str, Any],
    workspace_client: WorkspaceClient,
) -> None:
    description = args.get("description") or f"dao-ai tool '{tool_name}'."

    @mcp.tool(name=tool_name, description=description)
    async def my_tool(query: str) -> dict[str, Any]:
        ...  # call into your factory's tool, shape the response
        return {"result": ..., "_meta": {"tool_name": tool_name}}

register_adapter(McpAdapter(factory_name=FACTORY, register=_register))
```

Add the side-effect import to `dao_ai/mcp/service.py`:

```python
from dao_ai.mcp.adapters import my_thing as _my_thing_adapter  # noqa: F401
```

Now any `tools.<name>` whose factory is `dao_ai.tools.create_my_thing` becomes an MCP tool when the server boots.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| App boots but `/healthz` returns empty body. | Databricks Apps' edge proxy intercepts `/healthz` for its own platform health check. | Use `/readyz` (or any other path) to verify the app's own routes. |
| `Task group is not initialized` 500 on every request. | FastMCP's session manager wasn't started. | Already fixed in `dao_ai.mcp.server.build_app` via FastAPI lifespan. |
| 307 on `POST /mcp` (no trailing slash). | Standard ASGI redirect to `/mcp/`. | Make sure your client follows redirects, or POST to `/mcp/` directly. |
| MCP returns no SQL → LRU never caches (`Not caching: response has no SQL query`). | Genie returned a free-text response, not a SQL query. Often means the question isn't SQL-eliciting or the Genie space lacks the relevant tables. | Use a more concrete question matching the space's data sources. |
| `permission denied for schema public` at semantic-cache init. | The App SP can connect to Lakebase but doesn't have `CREATE` on `public`. | Grant `CREATE` to the App SP in the target schema, or pre-create `genie_context_aware_cache` and `genie_prompt_history`. |
| Lakebase resource path wrong on `bundle deploy` (`database does not exist`). | `dao-ai generate-mcp` resolved against the wrong workspace's Lakebase project. | Pass `-p <profile>` to `dao-ai generate-mcp` so the resource resolver targets the right workspace. |
| `trace_id=null` in every `_meta`. | App isn't bound to an MLflow experiment. | Set `MLFLOW_EXPERIMENT_ID` env var on the App resource (mirror `generate-bundle`'s experiment-resource pattern). Future enhancement. |

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
  "$APP_URL/mcp/"

curl -sS -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  "$APP_URL/mcp/"

# Call a tool
curl -sS -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"merchandising_analytics","arguments":{"question":"Count products per department"}}}' \
  "$APP_URL/mcp/"
```

The Apps logs (`databricks apps logs <app-name> -p <profile>`) include structured loguru events for every tool call — `mcp.genie.query.start/done`, `mcp.vs.start/done`, `mcp.adapter.*.registered`, and the underlying dao-ai cache-layer events (`Cache MISS`, `Cache HIT | layer=...-lru | cache_age_seconds=...`). Use those to trace cache progression and instructed-retrieval pipeline activity.

---

## Related modules

- `dao_ai.apps.mcp` — **client** primitives for consuming external MCP servers from a dao-ai agent (security helpers, etc.). Independent of `dao_ai.mcp`.
- `dao_ai.tools.create_genie_toolkit`, `dao_ai.tools.create_vector_search_tool` — the underlying factories the MCP server delegates to.
- `dao_ai.apps.bundle.write_bundle` — the corresponding `dao-ai generate-bundle` entry point for agent deployments. `write_mcp_bundle` mirrors its shape.
