# WhatsApp Channel (Meta Cloud API)

## Overview

The WhatsApp channel lets end users chat with a dao-ai agent over WhatsApp
without any gateway service in between. When `app.channels.whatsapp` is
configured, two routes are mounted on the existing Databricks Apps FastAPI
server:

| Route | Purpose |
| ----- | ------- |
| `GET  /channels/whatsapp/webhook`  | Meta verification handshake. Returns `hub.challenge` when the verify token matches. |
| `POST /channels/whatsapp/webhook`  | Inbound message delivery. Verifies the HMAC, dedupes on Meta's message id, returns **200 in <20 s**, and dispatches the agent call in the background. |

Inbound messages enter the **same** agent invocation path used by
`/v1/responses` and `/invocations` — `dao_ai.apps.handlers.non_streaming`
under the hood — so guardrails, middleware, OBO header propagation,
memory, tracing, and long-running offloading all apply unchanged. Replies
go out via `https://graph.facebook.com/{version}/{phone_number_id}/messages`.

The channel is **Apps-only**. Databricks Model Serving cannot host
inbound webhooks because it doesn't expose a stable public HTTPS URL.

## Why a first-class channel?

Slack and Teams are modelled in dao-ai as **outbound tools**
(`tools/slack.py`, `tools/ms_teams.py`) — agents can post messages but
can't *receive* them. WhatsApp needs the inbound half:

- HMAC-SHA256 signature verification on every delivery.
- Idempotent dedup against Meta's retries.
- ACKing 200 within 20 s regardless of agent latency.
- Persistent mapping from a WhatsApp sender (`wa_id`) to a LangGraph
  `thread_id` so multi-turn conversations resume on the same memory.

All of that belongs at the HTTP layer alongside `_mount_long_running_routes`
and `_mount_a2a_routes`, not inside a tool.

## What you get out of the box

When you set `app.channels.whatsapp` and deploy to Databricks Apps, the
framework wires up:

- **Verification handshake** — pass-through of Meta's `hub.challenge` for
  any GET whose `hub.verify_token` query value matches your secret. Uses
  `hmac.compare_digest` to defeat timing attacks.
- **Signature verification** — every POST is rejected with 403 unless the
  `X-Hub-Signature-256: sha256=<hex>` header is a valid HMAC of the raw
  body against your Meta App Secret. Constant-time compare again.
- **Dedup** — every accepted message id is inserted into a Lakebase table
  with a `UNIQUE` constraint. Meta retries silently no-op.
- **Thread continuity** — `wa_id → thread_id` is upserted in Lakebase, so
  the LangGraph checkpointer keys on the same `thread_id` across messages
  from the same user. Any existing `memory.checkpointer` config works
  unchanged.
- **Outbound chunking** — assistant replies longer than the configured
  `max_outbound_chunk_chars` (default 4000) are split on paragraph/sentence
  boundaries before being POSTed to the Graph API. WhatsApp's hard limit
  is 4096.
- **PII redaction** — `wa_id` is SHA-256 hashed (16-char prefix) before
  being emitted as an MLflow trace attribute. Disable only in dev.
- **Companion outbound tool** — `dao_ai.tools.whatsapp.create_send_whatsapp_message_tool`
  for agent-initiated proactive messages (within the 24h customer service
  window, or via a pre-approved template).

## Configuration

A minimal config (no Lakebase — in-memory dedup, OK for smoke tests):

```yaml
app:
  name: whatsapp-bot
  deployment_target: apps
  agents:
    - *my_agent
  channels:
    whatsapp:
      verify_token:
        scope: whatsapp_test
        secret: WHATSAPP_VERIFY_TOKEN
      app_secret:
        scope: whatsapp_test
        secret: WHATSAPP_APP_SECRET
      access_token:
        scope: whatsapp_test
        secret: WHATSAPP_ACCESS_TOKEN
      phone_number_id: "1234567890123456"   # Meta-issued (NOT the E.164 phone)
```

A production config (Lakebase-backed dedup + thread mapping, shared with
the LangGraph checkpointer):

```yaml
resources:
  databases:
    whatsapp_database: &whatsapp_database
      project: "retail-consumer-goods"
      on_behalf_of_user: false

memory:
  checkpointer:
    name: whatsapp_checkpointer
    database: *whatsapp_database

app:
  name: whatsapp-concierge
  deployment_target: apps
  agents:
    - *concierge
  channels:
    whatsapp:
      verify_token:    { scope: retail, secret: WHATSAPP_VERIFY_TOKEN }
      app_secret:      { scope: retail, secret: WHATSAPP_APP_SECRET }
      access_token:    { scope: retail, secret: WHATSAPP_ACCESS_TOKEN }
      phone_number_id: "1234567890123456"
      graph_api_version: "v22.0"
      webhook_path: "/channels/whatsapp/webhook"
      database: *whatsapp_database
      default_thread_strategy: wa_id            # one thread per sender
      max_outbound_chunk_chars: 4000
      redact_phone_in_traces: true
```

See `config/examples/21_channels/whatsapp.yaml` for a runnable example
with the full schema layered on the standard dao-ai config blocks.

### Configuration reference

Every field except `database` and `default_thread_strategy` is typed as
[`AnyVariable`](configuration-reference.md), the standard dao-ai value
type. That means each one accepts the full union of value sources — pick
the one that fits the field's sensitivity and rotation needs:

| Form                                | Use it for | Resolves at |
| ----------------------------------- | ---------- | ----------- |
| `{scope: …, secret: …}`             | Credentials in production | Request time (rotation without restart) |
| `{env: NAME}`                       | Local dev, FEVM testing | Process start |
| `{options: [...]}` (composite)      | Fallback chains (secret → env → literal) | First non-None at request time |
| `"literal-string"` / `4000` / `true`| Tests, non-sensitive fields | Config load |

| Field                       | Type                       | Default                                                  | Notes |
| --------------------------- | -------------------------- | -------------------------------------------------------- | ----- |
| `verify_token`              | `AnyVariable`              | (required)                                               | Meta webhook verify token. Use a secret scope in prod. |
| `app_secret`                | `AnyVariable`              | (required)                                               | Meta App Secret for HMAC. Use a secret scope in prod. |
| `access_token`              | `AnyVariable`              | (required)                                               | Bearer for outbound Graph API. Use a secret scope in prod. |
| `phone_number_id`           | `AnyVariable`              | (required)                                               | Meta-issued phone-number id. Not the E.164 phone. |
| `graph_api_version`         | `AnyVariable`              | `v22.0`                                                  | Resolved value should match `v\d+\.\d+`. |
| `webhook_path`              | `AnyVariable`              | `/channels/whatsapp/webhook`                             | Resolved value must start with `/` (checked at mount time). |
| `database`                  | `DatabaseModel?`           | falls back to `app.long_running.database`, then in-memory | Persist dedup + threads. NOT AnyVariable — typed config block. |
| `dedup_table_name`          | `AnyVariable`              | `dao_ai_whatsapp_inbound_dedup`                          | Created idempotently. |
| `threads_table_name`        | `AnyVariable`              | `dao_ai_whatsapp_threads`                                | Created idempotently. |
| `default_thread_strategy`   | `Literal[…]`               | `wa_id`                                                  | `wa_id` / `wa_id+phone_number_id` / `static`. NOT AnyVariable — must be one of the three. |
| `static_thread_id`          | `AnyVariable?`             | None                                                     | Required iff strategy is `static`. |
| `max_outbound_chunk_chars`  | `AnyVariable`              | `4000`                                                   | Resolves to int. WhatsApp's hard cap is 4096. |
| `redact_phone_in_traces`    | `AnyVariable`              | `true`                                                   | Resolves to bool. SHA-256 hash `wa_id` for trace attrs. |

## Request lifecycle

```text
WhatsApp user → Meta Cloud API
             ↓ POST {app}/channels/whatsapp/webhook
             ↓ (X-Hub-Signature-256: sha256=<hex>)
             ↓
  whatsapp_inbound() handler:
   1. verify_signature() — HMAC-SHA256 vs raw body + app_secret
   2. ChannelStore.record_message() — UNIQUE message_id (drops Meta retries)
   3. ChannelStore.get_or_create_thread() — wa_id → thread_id
   4. asyncio.create_task(_dispatch_one)   ← fire-and-forget
   5. return 200 {"status":"received"}     ← <20s guaranteed
             ↓
  _dispatch_one() (background):
   • Build ResponsesAgentRequest:
       input=[{"role":"user","content": text}]
       custom_inputs.configurable = {thread_id, channel="whatsapp",
                                     wa_id, phone_number_id, message_id}
   • non_streaming(request)  ← reuses /v1/responses handler
       ↳ MLflow tracing, OBO headers, middleware, memory, guardrails
   • send_text(reply)  ← POST graph.facebook.com/{ver}/{pid}/messages
       with 4096-char-aware chunking
```

## Outbound (agent-initiated) messages

For proactive notifications, give your agent the
`send_whatsapp_message` tool. It uses a Unity Catalog HTTP Connection to
`graph.facebook.com` so the access token is managed in UC instead of an
env var.

```yaml
resources:
  connections:
    whatsapp_connection: &whatsapp_connection
      name: meta_whatsapp_cloud_api      # UC HTTP Connection (Bearer auth)
      on_behalf_of_user: false

tools:
  send_whatsapp:
    name: send_whatsapp_message
    function:
      type: factory
      name: dao_ai.tools.create_send_whatsapp_message_tool
      args:
        connection: *whatsapp_connection
        phone_number_id: "1234567890123456"
        graph_api_version: "v22.0"

agents:
  notifier:
    name: notifier
    tools: [*send_whatsapp]
    prompt: |
      You can notify users by phone via send_whatsapp_message. Recipient
      phone numbers are E.164 without '+', e.g. '14155552671'.
```

⚠️ Outside the 24-hour customer service window, Meta blocks free-form
text. Pre-approved templates are the only way to message a user
proactively; template authoring is a Meta Business Manager workflow,
not a dao-ai concern.

## Production-readiness checklist

These are baked in, but worth understanding:

- **Signature verification** — `hmac.compare_digest` against the raw body.
  Log+drop on failure; never raise to Meta or 5xx will trigger retries.
- **Idempotency** — `UNIQUE` constraint on `message_id`. The handler is
  safe under Meta's at-least-once delivery.
- **<20s ACK** — agent dispatch is `asyncio.create_task`; webhook returns
  before the agent runs. Long agent runs are unconstrained by Meta's
  retry timer.
- **Secret rotation** — secrets resolve at request time, not boot, so
  `databricks secrets put-secret …` takes effect on the next inbound
  message without an app restart.
- **Public access** — Databricks Apps require workspace auth by default.
  Meta's GET verification is unauthenticated; grant `CAN_USE` to
  `All workspace users` AND enable unauthenticated access on the app, or
  Meta's *Verify and save* will fail with a 302 to login.
- **24-hour window** — inbound replies are always in-window. Proactive
  outbound (the tool) outside the window requires a pre-approved template.
- **PII / tracing** — phone numbers are PII. `redact_phone_in_traces`
  (default `true`) hashes `wa_id` before it lands in MLflow traces.
- **Rate limits** — Meta enforces tiered throughput per phone-number id.
  Watch for HTTP 429s in the outbound-send logs.

## End-to-end test recipe

### 1. Meta side (one-time)

1. Create a Meta Developer App at developers.facebook.com → *Other* → *Business*.
2. *Add Product* → *WhatsApp* → *Set up*. You get a free test number.
3. Under *WhatsApp → API Setup*, copy the **Phone Number ID** (NOT the
   E.164 phone), and add your personal WhatsApp number to the verified
   recipient list.
4. Under *App Settings → Basic*, copy the **App Secret**.
5. Generate a temporary access token from *API Setup* (24h lifespan; use
   a System User token for prod).
6. Invent a random verify token: `openssl rand -hex 32`.

### 2. Databricks side

```bash
# Create secret scope + seed the three Meta secrets
databricks secrets create-scope whatsapp_test --profile <prof>
databricks secrets put-secret whatsapp_test WHATSAPP_VERIFY_TOKEN  --string-value "$VERIFY_TOKEN"
databricks secrets put-secret whatsapp_test WHATSAPP_APP_SECRET    --string-value "$APP_SECRET"
databricks secrets put-secret whatsapp_test WHATSAPP_ACCESS_TOKEN  --string-value "$ACCESS_TOKEN"

# Generate + deploy the bundle
dao-ai generate-bundle -c config/whatsapp_test.yaml -o ./deploy-whatsapp --development --force
cd deploy-whatsapp
databricks bundle deploy --profile <prof>
databricks bundle run whatsapp-test-nf --profile <prof>
# → outputs the App URL
```

### 3. Wire Meta to Databricks

In *Meta Developers → Your App → WhatsApp → Configuration → Webhook*:

| Field         | Value                                                              |
| ------------- | ------------------------------------------------------------------ |
| Callback URL  | `<app-url>/channels/whatsapp/webhook`                              |
| Verify token  | the same string in `WHATSAPP_VERIFY_TOKEN`                         |

Click *Verify and Save* → expect a green check. Under *Webhook fields*,
subscribe to **`messages`** only.

### 4. Verify with curl (sanity check)

```bash
# GET handshake should echo the challenge
curl -i "$APP_URL/channels/whatsapp/webhook?hub.mode=subscribe&hub.verify_token=$VERIFY_TOKEN&hub.challenge=ping"
# → HTTP 200, body: ping

# GET with wrong token should be rejected
curl -i "$APP_URL/channels/whatsapp/webhook?hub.mode=subscribe&hub.verify_token=WRONG&hub.challenge=ping"
# → HTTP 403

# POST with tampered HMAC should be rejected
curl -i -X POST "$APP_URL/channels/whatsapp/webhook" \
  -H 'Content-Type: application/json' \
  -H 'X-Hub-Signature-256: sha256=deadbeef' \
  -d '{"entry":[]}'
# → HTTP 403
```

### 5. Send a real message

Message the test number from your verified phone. Expect:

- Agent reply on the same WhatsApp thread within seconds.
- New rows in `dao_ai_whatsapp_inbound_dedup` and `dao_ai_whatsapp_threads`.
- MLflow trace under the experiment with `channel=whatsapp` and a
  redacted `wa_id_hash`.

## Troubleshooting

| Symptom | Likely cause |
| ------- | ------------ |
| Meta *Verify and Save* fails with 302/login HTML | Databricks App requires workspace auth. Grant `All workspace users → CAN_USE` and enable unauthenticated access on the app. |
| 403 *Invalid signature* on every POST after verify works | App Secret mismatch. Signing uses the **App Secret**, not the access token. |
| 502 *Bad Gateway* immediately after deploy | The App container is still building the chat UI on first boot (a few minutes). Wait, then retry. |
| Inbound 200s but no reply on WhatsApp | (a) Recipient not on the verified test-number list, (b) outside the 24h window with no template, (c) access token missing `whatsapp_business_messaging` scope. |
| Agent runs but `reply` is empty | Final assistant message had no text. Check the MLflow trace and the agent prompt. |
| 500 *psycopg* errors in App logs | Service principal lacks `USE CATALOG`/`USE SCHEMA` on the Lakebase database. Grant via UC. |

## What's *not* in scope here

- **Meta Business Verification** — required before you can message
  non-test numbers. Filed entirely in Meta Business Manager.
- **Message templates** — content authoring is a Meta dashboard
  workflow. dao-ai gives you the API hook; your content team writes the
  template.
- **Inbound media** (images / audio / video) — initial scope is text
  in/out. Inbound non-text messages are logged and skipped. Extending is
  a matter of mapping more `message.type` cases in
  `_parse_inbound_text`.
- **Multi-tenancy** — one `WhatsAppChannelModel` instance equals one
  Meta phone number. Multiple numbers per app would need a list — defer
  until needed.

## See also

- `src/dao_ai/apps/channels/whatsapp.py` — route handlers + outbound sender.
- `src/dao_ai/apps/channels/store.py` — Lakebase dedup + thread mapping.
- `src/dao_ai/tools/whatsapp.py` — outbound `send_whatsapp_message` tool.
- `config/examples/21_channels/whatsapp.yaml` — runnable example.
- `tests/dao_ai/apps/channels/test_whatsapp.py` — 26 unit tests.
- [Long-Running Agents](long_running_agents.md) — recommended add-on so
  the inbound webhook ACK isn't bounded by the agent's wall-clock budget.
- [A2A Protocol](a2a_protocol.md) — the other inbound surface mounted on
  the same FastAPI app.
