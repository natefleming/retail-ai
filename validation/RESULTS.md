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

## What was NOT exercised in this pass (deferred)

The plan also called for: real end-to-end inference against active endpoints using a forwarded user token, JWT scope-claim decoding, and MLflow trace identity inspection. Those require:

- The target downstream resources to actually exist on FEVM with the right UC grants (warehouse, vector search index, Genie space, Lakebase project, etc.).
- A consenting user session on each app (OBO apps require explicit user consent on first access).
- A real MLflow agent log + register + serving-endpoint create cycle (~10 min/scenario).

The scope-string contract has been fully proven: every string dao-ai emits is accepted by the Apps platform, and Apps and Model Serving emit identical sets. Downstream inference + trace identity is a fuller end-to-end test that should run as its own campaign with the resource inventory provisioned first.

## Cleanup

- Probe app `scope-probe-nf` deleted from FEVM after Phase 0/2.
- No other temporary resources created in the workspace.

## Test counts

- Unit + integration tests in `tests/dao_ai/test_auth_policy.py` and `tests/dao_ai/test_apps_obo_partition.py`: **48 passed** (36 pre-existing, 12 new for ai-gateway gating, MCP companion pairing, postgres OBO, canonical-string invariant).
- `tests/dao_ai/test_mcp.py`: **8 passed, 1 skipped** (2 tests updated to reflect new `ConnectionModel.api_scopes` shape).
- Full suite: **2249 passed**, 8 failed (all pre-existing real-API tests unrelated to scope changes — Genie and reranking integration tests that hit live indexes), 56 skipped.
