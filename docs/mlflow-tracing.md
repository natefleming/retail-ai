# MLflow tracing for dao-ai deployments

dao-ai emits MLflow traces from every agent turn. On Databricks the traces land in Unity Catalog OTEL tables when `app.trace_location` is configured; otherwise they land in the control-plane experiment store. This page covers the deploy + runtime flow, the CLI tools that back it, and the diagnostic loop when spans don't appear where you expect.

## Deploy paths and where the trace machinery lives

dao-ai supports three deploy targets and each drives MLflow tracing differently:

| Target | Startup entrypoint | Trace destination binding |
|---|---|---|
| Databricks Apps (`dao-ai agent generate`) | `src/dao_ai/apps/handlers.py` | `link_experiment_trace_location` at import time + `apply_runtime_trace_destination` populates `_MLFLOW_TRACE_USER_DESTINATION` ContextVar so the OTEL exporter picks the correct UC table. |
| MCP server on Apps (`dao-ai agent generate --mode mcp`) | `src/dao_ai/mcp/server.py` | Same as Databricks Apps — imports `apply_runtime_trace_destination` after `mlflow.set_experiment`. |
| Model Serving (`agents.deploy`) | `src/dao_ai/apps/model_serving.py` | **No** in-container MLflow calls. `agents.deploy()` sets `MLFLOW_EXPERIMENT_ID`, `MLFLOW_TRACING_DESTINATION`, and `MLFLOW_TRACING_SQL_WAREHOUSE_ID` on the endpoint; MLflow's `_get_span_processors` resolves the destination from those env vars. See the header comment in `model_serving.py` for the rationale (MS containers can't reliably call `mlflow.set_experiment` without OAuth-config crashes). |

The Apps and MCP-server paths intentionally diverge from Model Serving: they call `mlflow.set_experiment(trace_location=UnityCatalog(...))` because their containers have ambient OAuth. Model Serving relies purely on env-driven routing so the container can boot even when the model wasn't logged with the experiment as a resource dependency.

## `trace_location` configuration

```yaml
app:
  name: my_app
  trace_location:
    schema: *my_uc_schema                     # SchemaModel — catalog + schema
    warehouse: "your-sql-warehouse-id"        # string OR WarehouseModel ref
    table_prefix: my_app_traces               # optional — see below
```

Semantics:

- **With `table_prefix`**: OTEL tables become `<catalog>.<schema>.<prefix>_otel_{spans,logs,metrics,annotations}` plus `<prefix>_trace_metadata` and `<prefix>_trace_unified`. Explicit and stable across experiments.
- **Without `table_prefix`**: MLflow uses the experiment id as the auto-prefix (e.g. `<catalog>.<schema>.1978128188192999_otel_spans`). Simpler config; each experiment is siloed.
- **`trace_location` unset**: traces persist to the control-plane experiment artifact store. Simpler still, but the artifact endpoint (`us-east-1.storage.cloud.databricks.com`) is currently unreachable from Databricks Apps containers — spans silently drop. Model Serving deploys are unaffected.

**`table_prefix` is permanent per experiment.** Once an experiment has been linked to a UC trace destination with a specific prefix, MLflow rejects any attempt to change catalog / schema / table_prefix ("already contains traces"). To change any of the three, provision a fresh experiment:

```bash
dao-ai trace create --name /Shared/my-app/dao-ai-v2 -p <profile>
# then reference the new experiment id under app.experiment.id in your config,
# or rev app.name so the auto-declared experiment path is distinct.
```

## The `dao-ai trace link` CLI

Between `bundle deploy` and `bundle run` you must run:

```bash
databricks bundle deploy --target dev -p <profile>
dao-ai trace link -c my_config.yaml -p <profile>
databricks bundle run <app-name> --target dev -p <profile>
```

The verb is idempotent — safe on every deploy — but load-bearing on re-deploys and after `trace_location` changes because the app's own runtime linkage attempt is rejected once the experiment already has traces. Running the link from the operator machine (deterministic timing, full OAuth) avoids that race.

`dao-ai agent generate` prints a one-line reminder in its "Next steps" output when `trace_location` is configured.

## Runtime ContextVar sync — `apply_runtime_trace_destination`

`dao-ai trace link` writes the experiment ↔ trace_location tag on the tracking server. That's necessary but not sufficient — MLflow's OTEL span exporter reads a client-side ContextVar `mlflow.tracing.provider._MLFLOW_TRACE_USER_DESTINATION` to decide which table to write to. Three things can populate the ContextVar:

1. `mlflow.set_experiment(..., trace_location=UnityCatalog(...))` — inside MLflow, this calls `_sync_trace_destination_and_provider` which writes the correct UnityCatalog into the ContextVar.
2. `MLFLOW_TRACING_DESTINATION` env var — MLflow's env parser reads this as a 2-part `catalog.schema` string and populates the ContextVar with the **deprecated `UCSchemaLocation`** (whose `full_otel_spans_table_name` defaults to the hard-coded `mlflow_experiment_trace_otel_spans`). `dao-ai agent generate` emits this env var whenever `trace_location` is set.
3. `mlflow.tracing.set_destination(...)` — the public API, which explicitly rejects the `UnityCatalog(table_prefix=...)` form.

The problem: when `dao-ai trace link` skips re-linking on the "already linked" path, MLflow never calls `_sync_trace_destination_and_provider`. So the ContextVar keeps whatever `MLFLOW_TRACING_DESTINATION` set — the deprecated UCSchemaLocation. The exporter then targets `mlflow_experiment_trace_otel_spans` (which doesn't exist on the prefixed schema) and every span export silently fails with `TABLE_DOES_NOT_EXIST`.

`dao_ai.providers.databricks.apply_runtime_trace_destination(config)` closes that gap:

- **With `table_prefix`**: writes a valid `UnityCatalog(catalog, schema, table_prefix)` directly into `_MLFLOW_TRACE_USER_DESTINATION`. The exporter picks `<prefix>_otel_spans`.
- **Without `table_prefix`**: **clears** the ContextVar (`set(None)`). Constructing `UnityCatalog(catalog, schema)` without a prefix raises at export time — clearing lets MLflow's `_resolve_experiment_uc_location` (in `mlflow/tracing/provider.py`) fall back to the experiment tags and construct the correct UnityCatalog with the backend-computed prefix.
- **`trace_location` unset**: no-op. Traces go to the control-plane store.

Both `apps/handlers.py` and `mcp/server.py` invoke it right after `link_experiment_trace_location`. `model_serving.py` does not — Model Serving relies entirely on env-driven routing per the divergence documented above.

## Verifying traces landed

```bash
# From your response's custom_outputs.trace_id (e.g.
# `trace:/retail_consumer_goods.sporting_goods_store.my_app/<hex>`), pull the hex
# suffix and query the prefixed table:
databricks api post /api/2.0/sql/statements --profile <profile> --json '{
  "statement": "SELECT name, kind, status.code FROM retail_consumer_goods.sporting_goods_store.my_app_otel_spans WHERE trace_id = '\''<hex>'\'' ORDER BY start_time_unix_nano LIMIT 100",
  "warehouse_id": "<warehouse-id>",
  "wait_timeout": "30s"
}'
```

When traces are landing correctly you'll see the full agent flow — root `predict` span, `LangGraph` orchestration spans, per-node model / tool spans — all with `STATUS_CODE_OK`.

## Diagnostic loop when traces don't land

1. **App logs** — `databricks apps logs <app-name>` and grep for `mlflow.tracing.export` warnings. A `TABLE_DOES_NOT_EXIST: mlflow_experiment_trace_otel_spans` line means the exporter fell back to the deprecated UCSchemaLocation path — `apply_runtime_trace_destination` didn't run (or its provider reset didn't take effect).
2. **`dao-ai trace link` log** — the CLI reports either "Linked experiment to UC trace location" or "Experiment already linked to matching UC trace destination, skipping". The second is fine on re-deploys; only worrying if you're changing catalog / schema / table_prefix on an existing experiment (which is rejected — you need a fresh experiment).
3. **App startup log** — grep for `Set MLflow runtime trace destination`. Confirms `apply_runtime_trace_destination` ran. If absent, either `trace_location` is unset or the app crashed before reaching that line.
4. **Experiment lifecycle** — check with `databricks api get /api/2.0/mlflow/experiments/get --json '{"experiment_id":"<id>"}'`. Trashed experiments accept trace metadata writes but silently drop OTEL spans. Restore with `databricks api post /api/2.0/mlflow/experiments/restore --json '{"experiment_id":"<id>"}'`.
5. **App SP grants** — the app service principal needs `USE_CATALOG` + `USE_SCHEMA` + `CREATE_TABLE` + `MODIFY` + `SELECT` on the trace-location schema (one-time per app SP). `dao-ai agent generate`'s "Next steps" prints the grant commands.
6. **Async flush** — MLflow batches OTEL exports. Allow 60–120 s between the inference call and querying the tables. Async logging can be disabled per-app with `MLFLOW_ENABLE_ASYNC_TRACE_LOGGING=false` in the app env.

## Distributed tracing across MCP (`_meta` propagation)

When a dao-ai agent calls a downstream MCP server (via an `mcp` tool with
`capabilities:` set), it injects W3C trace context into the `_meta` block of
each `tools/call`
([SEP-414](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/414)):

- `traceparent` — `00-<32-hex trace id>-<16-hex span id>-01`, minted from the
  active MLflow span. The hex is the OTel-native id that also lands in the UC
  `_otel_spans.trace_id` column, so the propagated context resolves to the same
  trace row.
- `baggage` — carries `mlflow.trace_id=<id>` for dao-ai-native correlation.

A dao-ai MCP **server** extracts these keys from the inbound `_meta` and stamps
them onto its own trace (`mcp.trace_context.*` span attributes), so a client's
trace and the server's trace correlate. This replaces the former custom
`x-dao-ai-trace-id` HTTP header. Implementation:
`dao_ai.tools.mcp_trace_context`. Note trace context is independent of OBO —
the OBO token continues to ride the `x-forwarded-access-token` HTTP header
untouched.

## Testing all six combinations

Success criteria for a dao-ai release: MLflow traces work consistently across the six configurations below. `apply_runtime_trace_destination` handles the client-side ContextVar so all Apps + MCP-server paths route consistently; Model Serving uses env-driven routing exclusively.

| Deploy target | `trace_location` | `table_prefix` | Expected trace destination |
|---|---|---|---|
| Databricks Apps | unset | — | Control-plane experiment store (fails on Apps today — network gap) |
| Databricks Apps | set | unset | `<catalog>.<schema>.<experiment_id>_otel_spans` |
| Databricks Apps | set | set | `<catalog>.<schema>.<prefix>_otel_spans` |
| Model Serving | unset | — | Control-plane experiment store |
| Model Serving | set | unset | `<catalog>.<schema>.<experiment_id>_otel_spans` |
| Model Serving | set | set | `<catalog>.<schema>.<prefix>_otel_spans` |

MCP-server-on-Apps mirrors the Databricks Apps rows.

## Known gaps (as of Jul 2026)

- **Apps + control-plane store**: broken by the `us-east-1.storage.cloud.databricks.com` network reachability gap. Configure `trace_location` on Apps to route around it.
- **Empty `_otel_spans` on Apps deployments with a custom `table_prefix`** — RESOLVED (Jul 2026). Root cause was a dao-ai bug, not a platform gap: `apply_runtime_trace_destination` set the `UnityCatalog` ContextVar destination without populating its `_otel_spans_table_name`/`_otel_logs_table_name` fields. `UnityCatalog.full_otel_spans_table_name` returns those fields verbatim (unlike `UCSchemaLocation`, which auto-qualifies), so `mlflow.tracing.utils.get_active_spans_table_name()` returned `None` and the OTEL exporter silently skipped every span. When the fields hold a bare table name the trace server rejects it with `Invalid full table name` — an error hidden by the Databricks SDK round-trip logger crashing on the BytesIO span payload (`object of type '_io.BytesIO' has no len()`). Fix: set the fields to the **fully-qualified** `<catalog>.<schema>.<prefix>_otel_spans`. Verified end-to-end on Databricks Apps (18 spans landed). Model Serving was never affected — it uses the platform's env-driven export, not this client-side UC exporter path.
