# 19 - Background Agents

Run long-running agent tasks with a **kickoff → poll → cancel** lifecycle instead
of a single blocking request. A caller starts the task, gets back an id
immediately, polls for progress/results, and can cancel before completion.
Task state is persisted in Lakebase so it survives across poll requests.

The same config works on both deployment targets:

- **Databricks Apps** — OpenAI Responses-API-compatible routes (`POST /v1/responses`,
  `GET /v1/responses/{id}`, `POST /v1/responses/{id}/cancel`).
- **Model Serving** — same semantics via `/invocations` with `background=true`
  in `custom_inputs`.

## Prerequisites

- A Lakebase project for durable task/checkpoint state (set via the
  `lakebase_project` parameter; default `retail-consumer-goods`).
- Service-principal secrets in the configured secret scope
  (`RETAIL_AI_DATABRICKS_CLIENT_ID` / `_SECRET`) for the Lakebase connection.

## Quick Start

```bash
# Validate
dao-ai validate -c examples/19_background_agents/background_research.yaml

# Deploy to Databricks Apps (Responses-API routes)
dao-ai agent up -c examples/19_background_agents/background_research.yaml --mode apps

# Or deploy to Model Serving (background=true via custom_inputs)
dao-ai agent up -c examples/19_background_agents/background_research.yaml --mode model_serving
```

## How It Works

1. The `app.background` block turns on background execution and points at a
   Lakebase `database` for the long-running responses store.
2. A caller kicks off a run (background request); the agent starts working and
   the call returns a response **id** without waiting for completion.
3. The caller polls by id to read status and, when finished, the result.
4. The caller may cancel an in-flight run by id.
5. Runs are bounded by `max_duration_seconds`; polling cadence is hinted by
   `poll_interval_seconds`.

## `app.background` Reference

```yaml
app:
  background:
    database: *retail_database        # Lakebase-backed durable task/result store
    default_enabled: false            # background off unless the caller opts in per request
    max_duration_seconds: 1800        # hard cap on a single run (< 30 min)
    poll_interval_seconds: 1.0        # suggested client poll interval
```

| Field | Purpose |
|---|---|
| `database` | `DatabaseModel` (Lakebase) that persists task state + results across polls. Shares its pool with `memory.checkpointer.database` when it's the same anchor. |
| `default_enabled` | Whether requests run in the background by default; when `false`, callers opt in per request. |
| `max_duration_seconds` | Upper bound on run time before the task is stopped. |
| `poll_interval_seconds` | Hint for how often clients should poll for status. |

## Related

- Full feature docs: [`docs/background_agents.md`](../../docs/background_agents.md)
- Background execution over the A2A protocol:
  [`examples/20_a2a_protocol/a2a_background.yaml`](../20_a2a_protocol/a2a_background.yaml)
