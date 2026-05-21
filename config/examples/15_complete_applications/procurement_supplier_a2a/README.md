# Procurement <-> Supplier (A2A)

End-to-end demo of two dao-ai apps speaking the
[Google A2A (Agent-to-Agent) protocol](https://a2a-protocol.org).

```text
[user] -> procurement app -- HTTP A2A --> supplier app -> Foundation Model API
```

| App | YAML | Role |
|-----|------|------|
| `dao-ai-supplier-a2a` | `supplier.yaml` | Wholesale-supplier specialist. Answers SKU, pricing, lead-time, MOQ, and stock questions from an embedded catalog. Exposes A2A by default. |
| `dao-ai-procurement-a2a` | `procurement.yaml` | Procurement-officer agent. Holds one tool — `query_supplier` — built from `dao_ai.tools.create_a2a_agent_tool` in **AppResource mode**: the supplier app is passed in directly via `app: *supplier_app`, and the tool resolves both the endpoint and the auth mode from it. |

Both apps run on Foundation Model API only — no Unity Catalog tables,
Vector Search indexes, or Genie rooms required.

---

## Why AppResource mode

The procurement app declares the supplier as a first-class Databricks
App resource and hands that resource straight to the A2A tool:

```yaml
resources:
  apps:
    supplier_app: &supplier_app
      name: dao-ai-supplier-a2a
      on_behalf_of_user: true

tools:
  query_supplier:
    function:
      type: factory
      name: dao_ai.tools.create_a2a_agent_tool
      args:
        app: *supplier_app
```

The factory then:

1. Looks up the supplier app's deployed URL via
   `DatabricksAppModel.url` (no `SUPPLIER_A2A_ENDPOINT` env var or
   secret needed).
2. Picks the auth mode from `supplier_app.on_behalf_of_user`:

| `supplier_app.on_behalf_of_user` | Auth used | What the supplier sees |
|---|---|---|
| `true` (this demo) | **`forwarded_user_token`** — read `runtime.context.headers["x-forwarded-access-token"]` per call | The calling end user. Combined with `supplier_llm.on_behalf_of_user: true`, the supplier's LLM call runs as that user. |
| `false` / unset | **`databricks_app_sp`** — mint a fresh M2M header from the ambient `WorkspaceClient().config.authenticate()` per call | The procurement service principal. Useful for server-to-server / scheduled pipelines where there's no calling user. |

You can still pin `auth_type:` explicitly in the tool args to override
the app-derived default (e.g. force App-SP M2M against an OBO-tagged
app, or attach a custom static bearer).

The tool also keeps its original **manual-endpoint mode** for external /
non-Databricks A2A agents (Vertex, third-party Crew.ai / LangGraph,
public agents). See
`config/examples/10_agent_integrations/a2a_agent.yaml` for that
variant.

---

## Deployment workflow

The supplier MUST be deployed first so its URL is resolvable by
`DatabricksAppModel.url` when the procurement app boots.

### 1. Deploy the supplier app

```bash
cd <repo root>

uv run dao-ai validate \
  -c config/examples/15_complete_applications/procurement_supplier_a2a/supplier.yaml

uv run dao-ai pipeline --deploy --run \
  -c config/examples/15_complete_applications/procurement_supplier_a2a/supplier.yaml
```

### 2. (Optional) Sanity-check the supplier

```bash
SUPPLIER_URL=$(databricks apps get dao-ai-supplier-a2a --output json | jq -r .url)
TOKEN=$(databricks auth token | jq -r .access_token)
curl -sf "$SUPPLIER_URL/.well-known/agent-card.json" \
  -H "Authorization: Bearer $TOKEN" | jq '.name, .version'
```

### 3. Deploy the procurement app

```bash
uv run dao-ai validate \
  -c config/examples/15_complete_applications/procurement_supplier_a2a/procurement.yaml

uv run dao-ai pipeline --deploy --run \
  -c config/examples/15_complete_applications/procurement_supplier_a2a/procurement.yaml
```

No env vars, no secret scopes, no manual token minting — the
`resources.apps.supplier_app` binding gives the procurement SP
`CAN_VIEW` on the supplier (the platform may prompt you to approve
this on first deploy in the Apps UI), and the A2A tool handles the
rest at runtime.

### 4. Try it end-to-end

```bash
PROC_URL=$(databricks apps get dao-ai-procurement-a2a --output json | jq -r .url)
TOKEN=$(databricks auth token | jq -r .access_token)

# Responses-API form.
curl -sf -X POST "$PROC_URL/invocations" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user",
       "content": "Quote 1,200 of ACM-HB-08 and confirm whether the lead time works for an Aug-15 build."}
    ]
  }' | jq

# A2A form.
curl -sf "$PROC_URL/.well-known/agent-card.json" \
  -H "Authorization: Bearer $TOKEN" | jq
curl -sf -X POST "$PROC_URL/a2a" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "demo-1",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [
          {"kind": "text",
           "text": "Quote 1,200 of ACM-HB-08 and confirm the lead time works for an Aug-15 build."}
        ]
      }
    }
  }' | jq
```

`examples/a2a/client.py` in the dao-ai repo is a slightly fuller A2A
client; it works against either app.

---

## Identity propagation (with AppResource mode)

| Hop | Auth in this demo (`supplier_app.on_behalf_of_user: true`) | Auth with flag unset / false |
|-----|---|---|
| user → procurement LLM | ✅ User OBO (`procurement_llm.on_behalf_of_user: true`) | ✅ User OBO |
| procurement app → supplier app | ✅ User bearer forwarded by the A2A tool (`forwarded_user_token`) | 🅼 Procurement-SP OAuth M2M (`databricks_app_sp`) — fresh per call via `WorkspaceClient().config.authenticate()` |
| supplier app → supplier LLM | ✅ User OBO (`supplier_llm.on_behalf_of_user: true` sees the same user) | ⚠️ Procurement-SP OBO — the supplier's OBO-tagged LLM resolves as the procurement SP rather than a human |

Notes:

* Cross-app OBO is **not yet** supported at the Databricks Apps
  platform layer — the `apps.apps` scope has no corresponding
  `user_api_scope` (see `src/dao_ai/apps/resources.py:124`). The
  forwarding above happens **at the dao-ai tool layer** and is
  implemented in `dao_ai.tools.a2a_agent.create_a2a_agent_tool`.
* Setting `on_behalf_of_user: true` on the procurement `apps.<key>`
  entry is therefore not a no-op: the dao-ai factory uses it to pick
  the right auth mode even though the platform itself doesn't emit a
  user scope for it.

---

## When to use which mode

* **`on_behalf_of_user: true`** → end-to-end user attribution. Pick
  this for user-facing apps where the audit trail must point at the
  human. Inbound caller must be a user (or service that proxies a user
  bearer), otherwise the `forwarded_user_token` mode raises a clear
  runtime error.
* **`on_behalf_of_user: false` / unset** → App-SP M2M. Pick this for
  server-to-server pipelines, scheduled jobs invoking the procurement
  app, or anywhere there's no calling user (e.g. batch reprocessing).
  The ambient `WorkspaceClient` mints a fresh M2M header per call from
  the auto-injected `DATABRICKS_CLIENT_ID` / `DATABRICKS_CLIENT_SECRET`
  env vars that the Apps runtime provides.

To swap modes on the fly: change the flag in `procurement.yaml`,
re-deploy. No code edits needed.

---

## Iterating

```bash
# Local chat against the supplier alone (no A2A involved).
uv run dao-ai chat \
  -c config/examples/15_complete_applications/procurement_supplier_a2a/supplier.yaml

# Local chat against the procurement agent. Requires the supplier app
# to be deployed (so DatabricksAppModel.url can resolve at startup).
uv run dao-ai chat \
  -c config/examples/15_complete_applications/procurement_supplier_a2a/procurement.yaml
```

For deploy iteration on either app, prefer
`databricks jobs repair-run` on the failed task over a full
`--deploy --run` cycle.

---

## Files

| File | Purpose |
|------|---------|
| `supplier.yaml` | Supplier-side dao-ai config (embedded catalog, no UC deps). |
| `procurement.yaml` | Procurement-side dao-ai config (one A2A tool using AppResource mode). |
| `README.md` | You are here. |
