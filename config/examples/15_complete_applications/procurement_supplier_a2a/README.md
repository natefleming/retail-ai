# Procurement <-> Supplier (A2A)

End-to-end demo of two dao-ai apps speaking the
[Google A2A (Agent-to-Agent) protocol](https://a2a-protocol.org).

```text
[user] -> procurement app -- HTTP A2A --> supplier app -> Foundation Model API
```

| App | YAML | Role |
|-----|------|------|
| `dao-ai-supplier-a2a` | `supplier.yaml` | Wholesale-supplier specialist. Answers SKU, pricing, lead-time, MOQ, and stock questions from an embedded catalog. Exposes A2A by default. |
| `dao-ai-procurement-a2a` | `procurement.yaml` | Procurement-officer agent. Holds one tool — `query_supplier` — built from `dao_ai.tools.create_a2a_agent_tool`. Delegates supplier questions; adds procurement recommendation on top. |

Both apps run on Foundation Model API only — no Unity Catalog tables,
Vector Search indexes, or Genie rooms required.

---

## Deployment workflow

The supplier MUST be deployed first so its URL is known before the
procurement app is configured.

### 1. Deploy the supplier app

```bash
cd <repo root>

# (Validate first — purely local; no workspace round-trip.)
uv run dao-ai validate \
  -c config/examples/15_complete_applications/procurement_supplier_a2a/supplier.yaml

# Deploy.
uv run dao-ai pipeline --deploy --run \
  -c config/examples/15_complete_applications/procurement_supplier_a2a/supplier.yaml
```

### 2. Capture the supplier URL

```bash
SUPPLIER_URL=$(databricks apps get dao-ai-supplier-a2a --output json | jq -r .url)
echo "$SUPPLIER_URL"
# Sanity-check the Agent Card is being served.
curl -sf "$SUPPLIER_URL/.well-known/agent-card.json" | jq '.name, .version'
```

### 3. Provide endpoint + token to the procurement app

The procurement app reads `SUPPLIER_A2A_ENDPOINT` and `SUPPLIER_A2A_TOKEN`
from env vars OR the `procurement_supplier_a2a` secret scope. Either form
works — env vars are easiest for local iteration; secrets are required
for production Apps deployments.

**Env-var form (local iteration / `dao-ai chat`):**

```bash
export SUPPLIER_A2A_ENDPOINT="$SUPPLIER_URL"
# Use the procurement service principal's OAuth M2M token.
export SUPPLIER_A2A_TOKEN=$(databricks auth token --profile <procurement-m2m-profile> | jq -r .access_token)
```

**Secret-scope form (deployed Apps):**

```bash
databricks secrets create-scope procurement_supplier_a2a
databricks secrets put-secret procurement_supplier_a2a SUPPLIER_A2A_ENDPOINT \
  --string-value "$SUPPLIER_URL"
databricks secrets put-secret procurement_supplier_a2a SUPPLIER_A2A_TOKEN \
  --string-value "$(databricks auth token --profile <procurement-m2m-profile> | jq -r .access_token)"
```

The procurement app's SP also needs `CAN_VIEW` on the supplier app — this
is requested automatically by the `resources.apps.supplier_app` binding
in `procurement.yaml`; review and approve it in the Apps UI on first
deploy.

### 4. Deploy the procurement app

```bash
uv run dao-ai validate \
  -c config/examples/15_complete_applications/procurement_supplier_a2a/procurement.yaml

uv run dao-ai pipeline --deploy --run \
  -c config/examples/15_complete_applications/procurement_supplier_a2a/procurement.yaml
```

### 5. Try it end-to-end

Pick a transport — the procurement app speaks both A2A and the MLflow
Responses contract.

**Over the Responses API (chat-style):**

```bash
PROC_URL=$(databricks apps get dao-ai-procurement-a2a --output json | jq -r .url)
TOKEN=$(databricks auth token | jq -r .access_token)

curl -sf -X POST "$PROC_URL/invocations" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user",
       "content": "Quote 1,200 of ACM-HB-08 and confirm whether the lead time works for an Aug-15 build."}
    ]
  }' | jq
```

**Over A2A (treating the procurement app as a remote A2A agent):**

```bash
PROC_URL=$(databricks apps get dao-ai-procurement-a2a --output json | jq -r .url)
TOKEN=$(databricks auth token | jq -r .access_token)

# Inspect the procurement app's Agent Card.
curl -sf "$PROC_URL/.well-known/agent-card.json" \
  -H "Authorization: Bearer $TOKEN" | jq

# Send a message.
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

## What's happening under the hood

1. The procurement agent receives the user message.
2. Its prompt mandates calling `query_supplier` for supplier-domain
   questions, so the LLM emits a tool call.
3. `query_supplier` is a `StructuredTool` built by
   `dao_ai.tools.create_a2a_agent_tool`. It:
   - Resolves the supplier's Agent Card at
     `${SUPPLIER_A2A_ENDPOINT}/.well-known/agent-card.json`.
   - Opens an A2A `message/send` stream (JSON-RPC 2.0 over SSE).
   - Forwards `runtime.context.thread_id` as the A2A `Message.context_id`
     so multi-turn state persists server-side on the supplier.
   - Drains the stream, aggregates every agent text part, and returns
     one string to the LLM.
4. The procurement agent layers a procurement recommendation on top of
   the supplier's quote and returns the combined reply.

MLflow traces span both apps. The procurement trace shows a `query_supplier`
tool span with embedded A2A streaming attributes
(`dao_ai.a2a.endpoint_url`, `dao_ai.a2a.stream.terminal_state`, etc.);
the supplier trace shows the LLM call serving the request.

---

## OBO and identity propagation

| Hop | OBO today? | How |
|-----|------------|-----|
| user -> procurement LLM | **Yes** | `procurement_llm.on_behalf_of_user: true` — the Apps proxy forwards `x-forwarded-access-token`; dao-ai routes the LLM call as the user. |
| procurement app -> supplier app | **No (SP only)** | Cross-app OBO is not supported by Databricks Apps yet (see comment at `src/dao_ai/apps/resources.py:124`). The A2A call uses the procurement SP's OAuth M2M bearer. |
| supplier app -> supplier LLM | **Yes** | `supplier_llm.on_behalf_of_user: true`. *If* the procurement-side A2A tool ever forwards a user token, the chain remains user-attributed end-to-end. |

When cross-app OBO ships, set `on_behalf_of_user: true` on
`resources.apps.supplier_app` in `procurement.yaml` and switch the
A2A tool to forward the user token — both apps already have the OBO
flag set on their LLMs.

---

## Iterating

For tight feedback loops use `dao-ai chat` against either config:

```bash
# Local chat against the supplier alone (no A2A involved).
uv run dao-ai chat \
  -c config/examples/15_complete_applications/procurement_supplier_a2a/supplier.yaml

# Local chat against the procurement agent. Requires the supplier app
# to be deployed and SUPPLIER_A2A_ENDPOINT/TOKEN env vars set.
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
| `procurement.yaml` | Procurement-side dao-ai config (single A2A tool). |
| `README.md` | You are here. |
