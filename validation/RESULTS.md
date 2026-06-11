# OBO Scope Validation Campaign — Results

**Date:** 2026-06-11
**Workspace:** FEVM (`fevm-nfleming-stable-aws.cloud.databricks.com`)
**Branch:** `feat/obo-scopes-ai-gateway`

## Phase 0 — Canonical scope strings (probed against FEVM Apps API)

Probed every candidate string via `databricks apps update <probe> --json '{"user_api_scopes":["<scope>"]}'` on a throwaway probe app. The Apps API either echoes the scope back (ACCEPTED) or returns `The specified scope <X> is not a valid scope.` (REJECTED).

### Accepted (canonical)

| Scope | Resource surface it grants |
|---|---|
| `sql` | SQL warehouse / statement execution |
| `genie` | Genie spaces (conversational analytics) |
| `files` | UC Volume / workspace files |
| `vector-search` | Vector Search endpoints + indexes |
| `ai-gateway` | AI Gateway-fronted endpoint invocation |
| `model-serving` | Model Serving endpoint management |
| `serving.serving-endpoints` | Invoke Model Serving endpoints |
| `postgres` | Lakebase Postgres database access |
| `workspace.workspace` | Workspace folder / notebook files |
| `mcp.external` | External MCP server (UC Connection) |
| `mcp.functions` | Managed UC Function MCP server |
| `mcp.genie` | Managed Genie MCP server |
| `mcp.vectorsearch` | Managed Vector Search MCP server |
| `catalog.connections` | UC Connections / Lakehouse Federation |
| `catalog.catalogs:read` | UC catalogs (read) |
| `catalog.schemas:read` | UC schemas (read) |
| `catalog.tables:read` | UC tables (read) |

### Accepted aliases (legacy names — kept in `VALID_USER_API_SCOPES` to avoid breaking older configs)

- `files.files` (alias for `files`)
- `dashboards.genie` (alias for `genie`)
- `vectorsearch.vector-search-indexes` (alias for `vector-search`)
- `vectorsearch.vector-search-endpoints` (alias for `vector-search`)

### Rejected (NOT valid OBO scopes)

| Scope | Why |
|---|---|
| `vector-search.search` | Wrong string — actual scope is `vector-search` |
| `catalog.volumes` | Not exposed as user OBO scope; use `files` for volume access |
| `apps.apps` | Not exposed as user OBO scope |
| `iam.access-control:read` | Auto-granted default; cannot be set manually |
| `iam.current-user:read` | Auto-granted default; cannot be set manually |

## Phase 1 — Code changes

See commit `418ef1f` on branch `feat/obo-scopes-ai-gateway`.

- `src/dao_ai/apps/resources.py` — `VALID_USER_API_SCOPES` rewritten with canonical + aliases; `API_SCOPE_TO_USER_SCOPE` (1:1) replaced with `API_SCOPE_TO_USER_SCOPES` (1:set) for additive MCP companion emission; `generate_user_api_scopes` gates `ai-gateway` on `InferenceEndpointModel.on_behalf_of_user AND .ai_gateway`.
- `src/dao_ai/config.py` — `ConnectionModel.api_scopes` and `McpFunctionModel.api_scopes` no longer hand-roll `mcp.*` strings (companions are derived).
- `src/dao_ai/providers/databricks.py` — `build_auth_policy` now delegates `api_scopes` generation to `generate_user_api_scopes`, unifying Apps and Model Serving emission.

## Phase 2 — Validation matrix (12 scenarios, both surfaces)

`validation/scenarios.py` defines 12 scenarios covering: AI Gateway gating (4 cases — all 4 combinations of `on_behalf_of_user` × `ai_gateway`), each OBO resource type in isolation (warehouse, genie, vector-search, volume, connection, lakebase, table+function), and a "mixed all OBO with gateway" scenario combining every resource type.

### Surface 1: `generate_user_api_scopes(config)` — Apps path

| # | Scenario | Result | Emitted scopes |
|---|---|---|---|
| 1 | llm-sp-no-gw | ✅ | `[]` |
| 2 | llm-obo-no-gw | ✅ | `[serving.serving-endpoints]` |
| 3 | llm-obo-with-gw | ✅ | `[ai-gateway, serving.serving-endpoints]` |
| 4 | llm-sp-with-gw-NEGATIVE | ✅ | `[]` — `ai-gateway` correctly suppressed |
| 5 | warehouse-obo | ✅ | `[mcp.functions, sql]` |
| 6 | genie-obo | ✅ | `[genie, mcp.genie]` |
| 7 | vector-search-obo | ✅ | `[mcp.vectorsearch, serving.serving-endpoints, vector-search]` |
| 8 | volume-obo | ✅ | `[files]` |
| 9 | connection-obo | ✅ | `[catalog.connections, mcp.external, serving.serving-endpoints]` |
| 10 | lakebase-obo | ✅ | `[postgres]` |
| 11 | table-and-function-obo | ✅ | `[catalog.catalogs:read, catalog.schemas:read, catalog.tables:read, mcp.functions, sql]` |
| 12 | mixed-all-obo-with-gw | ✅ | `[ai-gateway, catalog.catalogs:read, catalog.connections, catalog.schemas:read, catalog.tables:read, files, genie, mcp.external, mcp.functions, mcp.genie, mcp.vectorsearch, postgres, serving.serving-endpoints, sql, vector-search]` |

Run: `uv run python validation/run_unit_check.py`.

### Surface 2: `build_auth_policy(config).user_auth_policy.api_scopes` — Model Serving path

Every scenario produced the **identical** scope set on the Model Serving path. The unification via `build_auth_policy → generate_user_api_scopes` means both surfaces are guaranteed to emit the same strings (verified per-scenario in `run_serving_policy_check.py`).

Run: `uv run python validation/run_serving_policy_check.py`.

### Apps platform acceptance — real `PATCH /api/2.0/apps/...` against FEVM

For every scenario with a non-empty scope set, sent the expected set in a real Apps update call and confirmed the platform echoed back the exact set with no `INVALID_SCOPE` rejection.

| Scenario | Apps platform response |
|---|---|
| llm-obo-no-gw | ✓ 1 scope accepted |
| llm-obo-with-gw | ✓ 2 scopes accepted |
| warehouse-obo | ✓ 2 scopes accepted |
| genie-obo | ✓ 2 scopes accepted |
| vector-search-obo | ✓ 3 scopes accepted |
| volume-obo | ✓ 1 scope accepted |
| connection-obo | ✓ 3 scopes accepted |
| lakebase-obo | ✓ 1 scope accepted |
| table-and-function-obo | ✓ 5 scopes accepted |
| mixed-all-obo-with-gw | ✓ 15 scopes accepted |

Run: `python validation/run_apps_probe.py --profile fevm --app scope-probe-nf`.

### Real example config — `config/examples/06_on_behalf_of_user/obo_basic.yaml`

Loaded a real (existing) dao-ai OBO example config through both surfaces and confirmed:

- Apps `user_api_scopes` = `[catalog.catalogs:read, catalog.schemas:read, catalog.tables:read, genie, mcp.functions, mcp.genie, serving.serving-endpoints, sql]`
- Model Serving `UserAuthPolicy.api_scopes` = same list (identical)
- FEVM Apps API accepted the full set in a real update call

## Phase 2b — Real end-to-end deploys against FEVM existing resources

Two real dao-ai bundles built via `dao-ai generate-bundle --development` and deployed to FEVM via `databricks bundle deploy`. Both used existing FEVM resources (Genie space `01f164d91cb71e63a36d9545d86c7424` "dao-ai genie provisioning test", vector index `retail_consumer_goods.hardware_store.products_index`, UC functions `find_product_by_sku/upc`).

| App | `ai_gateway` flag | Deployed `user_api_scopes` (read from `/api/2.0/apps`) |
|---|---|---|
| `obo-validation-gw-on` | `true` | `[ai-gateway, catalog.catalogs:read, catalog.schemas:read, catalog.tables:read, genie, mcp.functions, mcp.genie, mcp.vectorsearch, serving.serving-endpoints, sql, vector-search]` |
| `obo-validation-gw-off` | `false` | `[catalog.catalogs:read, catalog.schemas:read, catalog.tables:read, genie, mcp.functions, mcp.genie, serving.serving-endpoints, sql]` |

Both apps reached `compute=ACTIVE` and `deploy=SUCCEEDED`. Logs show: OBO middleware enabled, agent created with the expected tool counts (gw-on: 3 tools, gw-off: 2), uvicorn listening on 8000. Expected `_create_obo_uc_tool` warnings appeared at startup ("User does not have USE CATALOG …") — the **app SP correctly cannot introspect** OBO-marked functions, which is the intended OBO behavior (introspection happens with the user token at runtime).

### JWT scope-claim decoding from live `x-forwarded-access-token`

A real inference probe (`POST /invocations` with my user token) was made against each running app. The dao-ai server reflects the request headers into `custom_outputs.configurable.headers`, including the platform's `x-forwarded-access-token` JWT. Decoding the `scope` claim:

```
gw-on:  ai-gateway catalog.catalogs:read catalog.schemas:read catalog.tables:read
        genie iam.access-control:read iam.current-user:read mcp.functions
        mcp.genie mcp.vectorsearch serving.serving-endpoints sql vector-search
        → contains ai-gateway? True

gw-off: catalog.catalogs:read catalog.schemas:read catalog.tables:read genie
        iam.access-control:read iam.current-user:read mcp.functions mcp.genie
        serving.serving-endpoints sql
        → contains ai-gateway? False
```

The JWT claim matches the deployed app's `user_api_scopes` plus the two platform-auto-granted defaults (`iam.access-control:read`, `iam.current-user:read`). **AI Gateway gating works end-to-end**: dao-ai config → bundle generation → Apps platform → user token claim.

### MLflow traces

Both apps produced complete MLflow traces in their experiments:

- `gw-on` experiment `3360342003324839`: trace `tr-c0a358f6b90dd31c569efdaca2272d78` — state `OK`, 3.413s, 9 spans, 15525 tokens. Request `"hi"`, response is Claude's retail assistant greeting.
- `gw-on` second trace `tr-ae483066379d51c2bdd58ed65fb990ce` — `What is 2+2?`, response confirms inference path works.
- `mlflow.user` tag = the app SP UUID — this is the trace **author**, not the OBO end user. End-user identity lives in the JWT `sub` claim (`nate.fleming@databricks.com` here), which the app code can read off `x-forwarded-access-token` and propagate to its own request log if needed.

### Apps platform acceptance — bundle path proven

The `databricks bundle deploy` path goes: `dao-ai generate-bundle` → `databricks.yaml` → `databricks bundle deploy` → `POST /api/2.0/apps` (create or update). The deployed apps' actual `user_api_scopes` (read back from `/api/2.0/apps`) match what `generate_user_api_scopes` produced. End-to-end contract proven on the path users actually deploy with.

## Cleanup

- Probe app `scope-probe-nf` deleted (Phase 0/2).
- `obo-validation-gw-on` and `obo-validation-gw-off` apps + experiments deleted via `databricks bundle destroy`.
- Local bundle dirs `/tmp/obo-val-gw-*` removed.

## Test counts

- Unit + integration tests in `tests/dao_ai/test_auth_policy.py` and `tests/dao_ai/test_apps_obo_partition.py`: **48 passed** (36 pre-existing, 12 new for ai-gateway gating, MCP companion pairing, postgres OBO, canonical-string invariant).
- `tests/dao_ai/test_mcp.py`: **8 passed, 1 skipped** (2 tests updated to reflect new `ConnectionModel.api_scopes` shape).
- Full suite: **2249 passed**, 8 failed (all pre-existing real-API tests unrelated to scope changes — Genie and reranking integration tests that hit live indexes), 56 skipped.
