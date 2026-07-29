# 20 - A2A Protocol (Agent-to-Agent)

Expose a dao-ai agent over the Google **A2A (Agent2Agent)** protocol so other
agents can discover and call it. On every Databricks Apps deployment the app
automatically mounts:

- `GET  /.well-known/agent-card.json` — the A2A **Agent Card** (capabilities + auth schemes)
- `POST /a2a` — A2A JSON-RPC 2.0 (`message/send`, `tasks/*`)
- `POST /invocations` — MLflow Responses / chat-completions

A2A is enabled by default on Apps deployments; set `a2a.enabled: false` to opt out.

## Examples

| File | Description |
|------|-------------|
| [`a2a_minimal.yaml`](./a2a_minimal.yaml) | Smallest A2A-enabled agent — a `greeter` on Foundation Model API. No UC tables, Vector Search, or Genie required. In-memory task store. |
| [`a2a_hitl_obo.yaml`](./a2a_hitl_obo.yaml) | A2A with human-in-the-loop approval and on-behalf-of-user identity forwarding — the Agent Card advertises both oauth2 and bearer schemes. |
| [`a2a_background.yaml`](./a2a_background.yaml) | A2A with a Lakebase-persisted task store + HITL + OBO, sharing one connection pool with the checkpointer and the background responses store. |
| [`client/`](./client/) | A standalone A2A client for exercising a deployed agent — see [`client/README.md`](./client/README.md). |

## Prerequisites

- **`a2a_minimal.yaml`** — only Foundation Model API access (default on any workspace).
- **`a2a_hitl_obo.yaml` / `a2a_background.yaml`** — a Lakebase project + service-principal
  secrets for the persisted task store, and OBO-capable serving endpoints.

## Quick Start

```bash
# Validate
dao-ai validate -c examples/19_a2a_protocol/a2a_minimal.yaml

# Deploy to Databricks Apps (mounts the A2A endpoints)
dao-ai agent up -c examples/19_a2a_protocol/a2a_minimal.yaml --mode apps
```

Then fetch the Agent Card and send a message (see [`client/README.md`](./client/README.md)
for a ready-made client):

```bash
curl -s https://<app-url>/.well-known/agent-card.json | jq .
```

## Task Store

A2A tasks default to an in-memory store (fine for `a2a_minimal`). To persist tasks
across restarts, point `a2a.task_store.database` at a Lakebase `DatabaseModel`:

```yaml
app:
  a2a:
    enabled: true
    task_store:
      database: *a2a_demo_db          # same anchor as memory + background → one shared pool
      table: dao_ai_a2a_tasks
```

## Related

- On-behalf-of-user identity: [`06_on_behalf_of_user/`](../06_on_behalf_of_user/)
- Human-in-the-loop approvals: [`07_human_in_the_loop/`](../07_human_in_the_loop/)
- Background kickoff/poll/cancel: [`18_background_agents/`](../18_background_agents/)
