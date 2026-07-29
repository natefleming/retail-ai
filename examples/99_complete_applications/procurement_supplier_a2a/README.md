# Procurement <-> Supplier (App-to-App via Responses API)

End-to-end demo of two dao-ai apps where the procurement agent calls the
supplier agent as a tool via the **MLflow Responses API** (the canonical
contract for `mlflow.agents` ResponsesAgent deployments — including every
dao-ai app).

```text
[user] -> procurement app -- HTTPS /responses --> supplier app -> Foundation Model API
```

| App | YAML | Role |
|-----|------|------|
| `dao-ai-supplier-a2a` | `supplier.yaml` | Wholesale-supplier specialist. Answers SKU, pricing, lead-time, MOQ, and stock questions from an embedded catalog. Exposes the Responses API (and A2A) by default. |
| `dao-ai-procurement-a2a` | `procurement.yaml` | Procurement-officer agent. Holds one tool — `query_supplier` — declared as a first-class `type: app` tool with `app: *supplier_app`. dao-ai dispatches the call via `DatabricksOpenAI(workspace_client=...).responses.create(model='apps/<name>', …)`. OBO is auto-derived from the bound app. |

Both apps run on Foundation Model API only — no Unity Catalog tables,
Vector Search indexes, or Genie rooms required.

---

## How it works

The procurement app declares the supplier as a first-class Databricks
App resource and hands that resource straight to the agent tool:

```yaml
resources:
  apps:
    supplier_app: &supplier_app
      name: dao-ai-supplier-a2a
      on_behalf_of_user: true

tools:
  query_supplier:
    function:
      type: app
      app: *supplier_app
```

`type: app` dispatches via the OpenAI Responses API (the default when
`api:` is unset and `/agent/info` resolves to "responses"):

```python
ws = supplier_app.workspace_client_from(context)         # OBO-aware
client = DatabricksOpenAI(workspace_client=ws)
client.responses.create(
    model="apps/dao-ai-supplier-a2a",
    input=[{"role": "user", "content": prompt}],
)
```

The factory:

1. Builds a `WorkspaceClient` per call. If `supplier_app.on_behalf_of_user`
   is true, that client uses the calling user's `x-forwarded-access-token`
   (forwarded by the Apps proxy) — so the supplier sees the end user.
   Otherwise it uses the procurement app's service principal.
2. Wraps the client in `DatabricksOpenAI`, which speaks the OpenAI API
   against the workspace's serving-endpoint proxy. The `apps/<name>`
   model prefix tells the proxy to forward to the supplier app's
   `POST /v1/responses` route.
3. Extracts the assistant's reply via `response.output_text`.

For explicit Google A2A protocol against the same app (session-continuity,
agent-card discovery, HITL-over-A2A), use `type: a2a` with `app:`
instead. For external A2A agents (Vertex, Crew.ai, ADK), see
`examples/10_agent_integrations/a2a_agent.yaml`.

---

## Deployment workflow

The supplier must be deployed before the procurement app makes its
first `query_supplier` tool call. Routing via the workspace's
serving-endpoint proxy + the `apps/<name>` model prefix means the
factory never needs to resolve the supplier's URL itself — the proxy
handles that. In practice: deploy supplier first, then procurement,
then exercise.

Each app is generated into its own bundle directory so the outputs
don't clobber each other.

### 1. Deploy the supplier app

```bash
cd <repo root>

# Generate the supplier bundle.
uv run dao-ai agent generate \
  -c examples/99_complete_applications/procurement_supplier_a2a/supplier.yaml \
  -o ../output-supplier-a2a \
  -p DEFAULT

# Deploy + start.
cd ../output-supplier-a2a
databricks bundle deploy --target dev -p DEFAULT
databricks bundle run dao-ai-supplier-a2a --target dev -p DEFAULT
cd -
```

### 2. (Optional) Sanity-check the supplier

```bash
SUPPLIER_URL=$(databricks apps get dao-ai-supplier-a2a -p DEFAULT --output json | jq -r .url)
TOKEN=$(databricks auth token -p DEFAULT | jq -r .access_token)
curl -sf "$SUPPLIER_URL/.well-known/agent-card.json" \
  -H "Authorization: Bearer $TOKEN" | jq '.name, .version'
```

### 3. Deploy the procurement app

```bash
# Generate the procurement bundle into a separate output dir.
uv run dao-ai agent generate \
  -c examples/99_complete_applications/procurement_supplier_a2a/procurement.yaml \
  -o ../output-procurement-a2a \
  -p DEFAULT

# Deploy + start.
cd ../output-procurement-a2a
databricks bundle deploy --target dev -p DEFAULT
databricks bundle run dao-ai-procurement-a2a --target dev -p DEFAULT
cd -
```

No env vars, no secret scopes, no manual token minting — the
`resources.apps.supplier_app` binding gives the procurement SP
`CAN_USE` on the supplier (the platform may prompt you to approve
this on first deploy in the Apps UI), and the A2A tool handles the
rest at runtime.

### 4. Try it end-to-end

```bash
PROC_URL=$(databricks apps get dao-ai-procurement-a2a -p DEFAULT --output json | jq -r .url)
TOKEN=$(databricks auth token -p DEFAULT | jq -r .access_token)

# Responses-API form (uses ``input``, not ``messages``).
curl -sf -X POST "$PROC_URL/invocations" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
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

`examples/19_a2a_protocol/client/client.py` in the dao-ai repo is a slightly fuller A2A
client; it works against either app.

---

## Verifying the supplier was actually called

The agent's prompt forbids answering supplier-domain questions from
memory (rule 1 in `procurement.yaml`), and all SKUs in the supplier
catalog are fictional (`ACM-*`). So any reply containing a real
catalog fact must have come from a live A2A call. Pick a prompt with a
unique tell:

| Prompt | Catalog fact only the supplier knows |
|---|---|
| "What is the current stock and lead time on ACM-DR-3?" | Stock = **72**, lead time = **14 days** |
| "Quote me 2,500 of ACM-LT-25 at the highest bulk tier." | Tier-2 price = **$0.37**, MOQ 100, stock 3,150 |
| "List every SKU Acme carries with its lead time." | Should enumerate all 7 SKUs in `supplier.yaml`'s embedded table |

To watch it happen in real time, tail the procurement app's log stream
in a second terminal — `create_a2a_agent_tool` logs an
`Invoking A2A agent` line before every hop:

```bash
PROC_URL=$(databricks apps get dao-ai-procurement-a2a -p DEFAULT --output json | jq -r .url)
TOKEN=$(databricks auth token -p DEFAULT | jq -r .access_token)

curl -sN -H "Authorization: Bearer $TOKEN" "$PROC_URL/logz/stream?source=APP" \
  | grep --line-buffered -E "Invoking A2A agent|query_supplier|dao_ai.a2a"
```

You should see something like:

```
INFO  dao_ai.tools.a2a_agent:a2a_agent - Invoking A2A agent
  endpoint=https://dao-ai-supplier-a2a-<workspace-id>.aws.databricksapps.com
  prompt_chars=… context_id=… user_id=… streaming=True
```

For a negative control, ask a non-supplier question (e.g. *"What's our
internal PO approval policy?"*) — rule 4 says the agent should answer
directly. No `Invoking A2A agent` line should appear.

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

## Agent Card customization

Both apps publish a v0.3.x A2A **Agent Card** at
`/.well-known/agent-card.json`. The card is auto-derived from the rest of
the config — every field below has a fallback, so omitting the `app.a2a`
block produces a working (if minimal) card. Populate `app.a2a` to ship a
richer, discovery-friendly card.

### Auto-derivation map

| Card field | Source when `app.a2a` is unset |
|---|---|
| `name` | `app.name` |
| `description` | `app.description` (trailing whitespace stripped) |
| `version` | Installed `dao-ai` package version |
| `url` | `$DATABRICKS_APP_URL/a2a` at startup, else `/a2a` |
| `protocolVersion`, `preferredTransport` | a2a-sdk defaults (`0.3.0`, `JSONRPC`) |
| `defaultInputModes` / `defaultOutputModes` | `["text/plain", "application/json"]` |
| `capabilities.streaming` | `true` (a2a-sdk supports `message/stream`) |
| `capabilities.pushNotifications` | `false` (dao-ai has no push-notification webhook) |
| `capabilities.stateTransitionHistory` | `true` iff `a2a.task_store.database` is set |
| `skills` | One `AgentSkill` per entry in `agents` — `id`/`name` = agent name, `description` from agent, `tags=["dao-ai", "sub-agent"]`, no `examples` |
| `securitySchemes` | Auto: scans config for any `on_behalf_of_user: true` → emits `oauth2` (authorization-code, `user_impersonation` scope) + `bearer`; else single `bearer` (PAT/M2M) |
| `security` | One OR-alternative per scheme; `oauth2` requirement lists the scopes declared on the scheme (e.g. `["user_impersonation"]`); other schemes list `[]` |
| `provider`, `documentationUrl`, `iconUrl` | Omitted unless set under `app.a2a` |

### Override block (`app.a2a`)

```yaml
app:
  a2a:
    # Service provider — strongly recommended for production. Both fields required.
    provider:
      organization: Databricks Field Engineering
      url: https://github.com/databrickslabs/dao-ai

    # Optional URLs surfaced on the card.
    documentation_url: https://example.com/docs
    icon_url: https://example.com/icon.png

    # Capability advertisements. None of these change runtime behavior;
    # they just describe what the server supports to A2A clients.
    streaming: true               # default true
    push_notifications: false     # default false; flip only after wiring a notifier
    state_transition_history: null  # null (default) auto-derives from task_store.database

    # Force-toggle the OBO security model independent of resource-level
    # on_behalf_of_user flags. null (default) → auto-derive.
    on_behalf_of_user: null

    # Default I/O modes for skills that don't declare their own.
    default_input_modes:  [text/plain, application/json]
    default_output_modes: [text/plain, application/json]

    # Per-skill overrides. When provided, replaces the auto-derived
    # one-skill-per-agent list. Use to ship human-readable names,
    # tags, examples, and per-skill I/O modes.
    skills:
      - id: supplier_agent                          # programmatic id
        name: Acme Industrial Supply — Catalog Specialist   # human-readable
        description: >-
          What this skill does, surfaced in discovery clients.
        tags: [dao-ai, supplier, procurement, catalog]
        examples:
          - Quote 750 units of ACM-HB-08 and confirm lead time.
        input_modes:  [text/plain, application/json]
        output_modes: [text/plain, application/json]

    # Full override of the security schemes block. Typed against
    # a2a-sdk's SecurityScheme discriminated union, so malformed schemes
    # fail at config load. See ``dao_ai.apps.a2a.security`` for ready-made
    # bearer/oauth2/OIDC/apiKey constants and factories.
    security_schemes: null
```

This demo populates `provider`, `documentation_url`, and the `skills` list
on both apps (`supplier.yaml`, `procurement.yaml`) — diff against the
auto-derived card to see the effect.

### OBO security model

Both apps in this demo set `on_behalf_of_user: true` on their LLM, which
causes the auto-derivation to emit **both** an `oauth2` and a `bearer`
scheme:

- **`oauth2`** — authorization-code flow against the workspace's
  `/oidc/v1/authorize` and `/oidc/v1/token` endpoints, with a single
  `user_impersonation` scope. This is documentation for A2A callers:
  "the resources behind this agent will act as the caller, and here is
  the OAuth2 dance that produces the right token."
- **`bearer`** — HTTP bearer scheme with an OBO-aware `bearerFormat`
  hint. This is the wire-level shape: clients send
  `Authorization: Bearer <token>`. The Databricks Apps proxy unwraps it
  and re-forwards as `x-forwarded-access-token` to dao-ai.

The card's `security` array lists these as **two OR-alternatives** —
clients satisfy one or the other (in practice they're the same token at
different layers).

To suppress OBO advertisement (run the apps as their service principal
end-to-end), set `app.a2a.on_behalf_of_user: false` and also flip the
resource-level `on_behalf_of_user` flags on the LLM / app resources.

### v1.0 spec migration

dao-ai pins `a2a-sdk>=0.3.0,<1.0.0`. A2A v1.0 restructures the card
(`supportedInterfaces`, `protocolBinding`) and is tracked as a separate
upgrade — no action required for users of this demo today.

---

## Iterating

```bash
# Local chat against the supplier alone (no A2A involved).
uv run dao-ai chat \
  -c examples/99_complete_applications/procurement_supplier_a2a/supplier.yaml

# Local chat against the procurement agent. The supplier app does NOT
# need to be deployed for the chat to start (the A2A tool resolves the
# supplier URL lazily on first call). It does need to be deployed
# before you ask any supplier-domain question, otherwise the first
# tool call errors out.
uv run dao-ai chat \
  -c examples/99_complete_applications/procurement_supplier_a2a/procurement.yaml
```

For deploy iteration on either app, re-run `agent generate` against
the same output directory and redeploy:

```bash
# Supplier:
uv run dao-ai agent generate \
  -c examples/99_complete_applications/procurement_supplier_a2a/supplier.yaml \
  -o ../output-supplier-a2a -p DEFAULT
(cd ../output-supplier-a2a && databricks bundle deploy --target dev -p DEFAULT)

# Procurement:
uv run dao-ai agent generate \
  -c examples/99_complete_applications/procurement_supplier_a2a/procurement.yaml \
  -o ../output-procurement-a2a -p DEFAULT
(cd ../output-procurement-a2a && databricks bundle deploy --target dev -p DEFAULT)
```

---

## Files

| File | Purpose |
|------|---------|
| `supplier.yaml` | Supplier-side dao-ai config (embedded catalog, no UC deps). |
| `procurement.yaml` | Procurement-side dao-ai config (one A2A tool using AppResource mode). |
| `README.md` | You are here. |
