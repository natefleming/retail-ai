# A2A Protocol — Worked Example

This directory pairs a deploy-ready dao-ai config with a Python A2A client
that exercises the full capability surface.

## Files

* [`../../config/examples/20_a2a_protocol/a2a_minimal.yaml`](../../config/examples/20_a2a_protocol/a2a_minimal.yaml)
  — Minimal agent config (no Unity Catalog tables / Vector Search / Genie
  required). Foundation Model API only.
* [`client.py`](client.py) — End-to-end A2A client that fetches the Agent
  Card, calls `message/send`, streams `message/stream`, and demonstrates
  the HITL `input-required` flow.

## Deploy

```bash
# Deploy the minimal agent to your fevm (or any) workspace.
dao-ai generate-workflow \
    -c config/examples/20_a2a_protocol/a2a_minimal.yaml \
    --deploy --run \
    --deployment-target apps \
    --profile fevm

# Find the app URL once deployment finishes.
databricks apps list --profile fevm | grep a2a-minimal
```

## Run the client

```bash
# Against the deployed app
APP_URL="https://<your-app>.cloud.databricksapps.com"
TOKEN="$(databricks auth token --profile fevm | jq -r .access_token)"

uv run python examples/a2a/client.py \
    --base-url "$APP_URL" \
    --bearer-token "$TOKEN"
```

```bash
# Against a local dev server
DAO_AI_CONFIG_PATH=config/examples/20_a2a_protocol/a2a_minimal.yaml \
    uv run python -m dao_ai.apps.server &

uv run python examples/a2a/client.py --base-url http://localhost:8000
```

## What you'll see

```text
[1/4] GET .../.well-known/agent-card.json
  name        : a2a-minimal
  description : Minimal dao-ai agent demonstrating Google A2A protocol support.
  url         : https://.../a2a
  skills      : ['greeter']
  security    : ['bearer']
  capabilities: {'streaming': True, ...}

[2/4] POST .../a2a  (method=message/send, text='Say hi in one sentence.')
  task id  : 7f3b…
  state    : completed
  text     : Hi there — happy to help!

[3/4] POST .../a2a  (method=message/stream, …)
  STATUS   submitted     final=False
  STATUS   working       final=False
  ARTIFACT 'Hi there — happy to help!'
  STATUS   completed     final=True

[4/4] HITL demo (interrupt → input-required → resume)
  initial state: completed
  (agent did not request input — done.)
```

The minimal agent doesn't have HITL tools, so the demo skips the resume
step. To exercise HITL end-to-end, swap in `config/examples/07_human_in_the_loop/human_in_the_loop.yaml`
and re-deploy — that agent triggers `interrupt()` for sensitive tool
calls, and the client will resume with `decisions=[{"type":"approve"}]`.

## Things to try next

* **Same conversation across protocols.** Use the same `contextId` in
  an A2A call and a `conversation_id` in an OpenAI Responses call. The
  LangGraph checkpointer keys both off the same `thread_id`, so the
  conversation continues seamlessly.
* **Custom inputs.** Send a `DataPart` alongside a `TextPart` —
  arbitrary keys land in dao-ai's `custom_inputs`. Use this for store
  number, user attributes, feature flags, etc.
* **Lakebase task persistence.** Set `app.background.database` and
  redeploy. A2A tasks now survive worker restarts (auto-selected via
  `app.a2a.task_store: auto`).
