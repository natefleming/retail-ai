# Walmart Pharmacy → Genie One (dao-ai `--as-mcp --with-connection`)

A single, fully parameterized dao-ai agent that Walmart deploys **as an MCP server** and
registers with the **Unity AI Gateway** so the **Genie One mobile app** can add it as a tool.
The agent fronts three Genie rooms (one per Walmart Pharmacy metric view) plus three Databricks
managed `system.ai.*` MCP services (Microsoft 365, Atlassian, Google Drive) and a built-in web
search.

Walmart cannot use partner-provider (Anthropic/Claude) models, so the agent's reasoning model
is a Databricks Model Serving endpoint serving an **OpenAI-flavored** model (`llm` parameter,
default `databricks-gpt-oss-120b`), routed through the Unity AI Gateway (`use_ai_gateway: true`).

## Architecture

```
Genie One (mobile)
      │  (MCP)
      ▼
mcp-walmart-pharmacy-genie-one          ← this agent, deployed as an MCP server
      │
      ├─ clinical_outcome_genie   → Genie room: Clinical Outcome Snapshot
      ├─ business_growth_genie    → Genie room: Core Business Growth
      ├─ digital_account_genie    → Genie room: Digital Account Snapshot
      ├─ microsoft_365_mcp        → system.ai.microsoft_365   (Unity AI Gateway, OBO)
      ├─ atlassian_mcp            → system.ai.atlassian        (Unity AI Gateway, OBO)
      ├─ google_drive_mcp         → system.ai.google_drive     (Unity AI Gateway, OBO)
      └─ web_search_mcp           → built-in DuckDuckGo search (zero-config)
```

Everything the agent calls is a Unity-AI-Gateway-registered MCP server — including the agent
itself once deployed with `--with-connection`.

## The three Genie rooms / metric views

All three describe Walmart Health & Wellness **Pharmacy** over one source table, hard-filtered to
the PHARMACY line of business, sharing an **LOB → Business Unit → Region → Market → Store**
hierarchy. The metric-view YAML is in `metric_views/` (source repointed to
`${catalog}.${schema}.market_bus_growth_metric`):

| Room | Metric view | Contents | Semantics |
|------|-------------|----------|-----------|
| Clinical Outcome Snapshot | `clinical_outcome_snapshot.yml` | OutcomesOne Successful Completion Rate, Completion Rate, Validation Rate | YTD direct-column snapshots — do not recompute/aggregate across time or hierarchy |
| Core Business Growth | `pharmacy_core_business_growth.yml` | Refill rates, messaging adoption, first-time-ready, out-of-stock, NPS, 5-star, script sales/counts, immunizations, testing & treatments, new digital accounts | Row-based, with TY / LY / point-change / YoY-% |
| Digital Account Snapshot | `digital_account_snapshot.yml` | Digital Population; Active Patients with Digital / New / Continuing accounts | Rolling 365-day direct-column snapshots — no time filters or cross-entity aggregation |

## Parameters (all overridable at deploy via `--param`)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `catalog` | `retail_consumer_goods` | Catalog for the metric views + AI Gateway MCP service |
| `schema` | `walmart_pharmacy_poc` | Schema within the catalog |
| `llm` | `databricks-gpt-oss-120b` | OpenAI-flavored serving endpoint (the agent's brain) |
| `clinical_space_id` | placeholder | Genie space ID for the Clinical Outcome room |
| `growth_space_id` | placeholder | Genie space ID for the Core Business Growth room |
| `digital_space_id` | placeholder | Genie space ID for the Digital Account room |
| `microsoft_365_service` | `system.ai.microsoft_365` | UC MCP service securable |
| `atlassian_service` | `system.ai.atlassian` | UC MCP service securable |
| `google_drive_service` | `system.ai.google_drive` | UC MCP service securable |

## Deploy

```bash
dao-ai agent up --as-mcp --with-connection \
  -c walmart_pharmacy_genie_one.yaml -p <profile> \
  --param catalog=<catalog> --param schema=<schema> \
  --param llm=<openai_flavored_serving_endpoint> \
  --param clinical_space_id=<id> \
  --param growth_space_id=<id> \
  --param digital_space_id=<id>
```

`--with-connection` requires `--as-mcp`. On deploy dao-ai grants the app SP CAN_USE on its own
app, creates the HTTP/MCP connection (`mcp_walmart_pharmacy_genie_one_conn`), registers the MCP
service (`mcp_walmart_pharmacy_genie_one`) with the Unity AI Gateway, and grants
USE_CONNECTION + EXECUTE. It then appears under Catalog → Unity AI Gateway → MCP servers; add the
connection to a Genie One chat (one-time, in the UI) to call the agent.

## Identity & auth

Every tool is **on-behalf-of-user (OBO)** (`on_behalf_of_user: true`): the deployed MCP app
forwards the caller's token (`x-forwarded-access-token`), so tools run as the end user — Genie
respects each user's data access, and the `system.ai.*` tools use each user's linked SaaS
credential. What each user needs:

- **Genie room tools** — `CAN RUN` on the three Genie spaces (and read on the metric views).
- **`system.ai.*` MCP tools** — served by the Unity AI Gateway at
  `/ai-gateway/mcp-services/<catalog.schema.service>`; each user must have **linked** the
  underlying SaaS account (Microsoft 365 / Atlassian / Google) via its OAuth consent flow —
  visit `…/explore/data/mcp-services/system/ai/<service>` once per user.

### Startup tool discovery (important)

The agent lists each MCP server's tools once **at app startup**, under the **app service
principal** — before any user context exists. Managed services differ here:

- `system.ai.microsoft_365` and `system.ai.google_drive` return their tool catalog to any
  principal, so they bind at startup and work for linked users.
- `system.ai.atlassian` (and GitHub) require the *caller* to have linked the account even to
  **list** tools, so the app SP's startup `tools/list` 403s. dao-ai now **skips** such an OBO
  tool with a warning instead of crashing the app; it becomes available once the caller links
  it, or once its schema is supplied at deploy time. Tracked in
  [dao-ai#305](https://github.com/natefleming/dao-ai/issues/305).

Validated on fevm: the deployed app boots, routes to all three Genie rooms (OBO), and calls
Microsoft 365 (`sharepoint_search`) as the forwarded user; Atlassian is skipped at startup
until linked.

## The `service:` field

This example uses a dao-ai MCP tool field, `service:`, that targets a three-part Unity Catalog
MCP service securable and routes to the Unity AI Gateway:

```yaml
tools:
  microsoft_365_mcp:
    name: microsoft_365_mcp
    function:
      type: mcp
      service: system.ai.microsoft_365   # → {host}/ai-gateway/mcp-services/system.ai.microsoft_365
      on_behalf_of_user: true
```

There is **no** `system.ai.web_search` managed service, so web search uses dao-ai's built-in
zero-config `type: search` tool. Swap it for a `service:`/`connection:` MCP tool if you
provision a web-search MCP server.

> **Platform note:** this POC deploys with `--as-mcp` (Databricks **Apps**), where the
> `service:` tools work end-to-end (the app SP lists them at startup; the `ai-gateway` OBO scope
> lets calls run as the forwarded end user). Deploying a `service:` tool to **Model Serving** is
> not yet supported — the served model has no user identity at load, so the startup `tools/list`
> fails. Tracked in [dao-ai#305](https://github.com/natefleming/dao-ai/issues/305).

## Standing up example rooms to test (fevm)

The metric views' production source (`wmt-hnw-pharmacy-catalog-prod…`) does not exist on fevm, so
build synthetic data first:

1. **Synthetic source table** — `python generate_synthetic_data.py` builds
   `${catalog}.${schema}.market_bus_growth_metric` (the wide table all three views read).
2. **Metric views** — create the three UC metric views from `metric_views/*.yml`.
3. **Genie spaces** — create one Genie space per metric view; capture the three `space_id`s.
4. **Deploy** — run the deploy command above with the three space IDs.
5. **Register with Genie One** — add the MCP connection to a Genie One chat in the UI.
