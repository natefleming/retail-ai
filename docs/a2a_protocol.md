# Google A2A (Agent2Agent) Protocol Support

## Overview

dao-ai agents deployed to **Databricks Apps** automatically expose
[Google's A2A v0.3 protocol](https://a2a-protocol.org) alongside the
existing OpenAI Responses contract. The same agent — same configuration,
same LangGraph, same checkpointer — answers both protocols on the same
FastAPI app:

| Route                                 | Protocol                                 |
| ------------------------------------- | ---------------------------------------- |
| `POST /invocations`                   | MLflow Responses (OpenAI-style)          |
| `POST /v1/responses*`                 | OpenAI Responses long-running (opt-in)   |
| `POST /v1/chat/completions`           | OpenAI Chat Completions proxy            |
| **`GET  /.well-known/agent-card.json`** | **A2A Agent Card discovery**           |
| **`POST /a2a`**                       | **A2A JSON-RPC 2.0 (message/send, message/stream, tasks/get, tasks/list, tasks/cancel)** |

Model Serving deployments only mount `/invocations` — A2A is Apps-only
because the MLflow Model Serving runtime cannot serve arbitrary routes.

A2A is **enabled by default** on every Databricks Apps deployment. No
config change is required; opt out via `app.a2a.enabled: false`.

## Why A2A?

A2A is the open, vendor-neutral protocol for agent-to-agent collaboration
governed by the Linux Foundation. Once an agent speaks A2A, it can:

- Be discovered automatically by any A2A-aware client (via the well-known
  Agent Card).
- Stream task progress over Server-Sent Events.
- Resume from human-in-the-loop interrupts using the same `taskId`.
- Participate in multi-agent workflows alongside agents built on other
  stacks (LangGraph, AutoGen, OpenAI Agents SDK, Strands, etc.).

## What you get out of the box

When you deploy to Databricks Apps (the default — or pass `--mode apps`), your agent's
Agent Card automatically advertises:

- **`name`** — `app.name`.
- **`description`** — `app.description`.
- **`url`** — `$DATABRICKS_APP_URL/a2a` if the env var is set, otherwise
  the relative path `/a2a`.
- **`skills`** — one [`AgentSkill`](https://a2a-protocol.org) per entry
  in `app.agents`. Each sub-agent's `name` and `description` are
  surfaced verbatim.
- **`securitySchemes`** — a single `bearer` HTTP scheme, validated at
  config-load time against a2a-sdk's typed `SecurityScheme` discriminated
  union. The `bearerFormat` description is conditioned on
  `app.a2a.on_behalf_of_user` to tell A2A clients whether the deployment
  supports OBO. See [Security scheme recipes](#security-scheme-recipes)
  for ready-made Databricks-flavored schemes.
- **`capabilities`** — streaming enabled, state transition history
  enabled, push notifications disabled.

Every field is overridable from config. See [Configuration](#configuration).

## Configuration

A minimal config that enables A2A (and gets sensible defaults for
everything):

```yaml
app:
  name: my_agent
  description: My helpful assistant.
  # deploy with: dao-ai agent deploy --mode apps
  agents:
    - *my_agent
```

Yes, that's it. `app.a2a` is implicitly enabled.

To customise:

```yaml
app:
  name: my_agent
  description: My helpful assistant.
  # deploy with: dao-ai agent deploy --mode apps
  a2a:
    enabled: true                        # default; set false to skip mounting
    server_url: null                     # default: derive from $DATABRICKS_APP_URL
    on_behalf_of_user: true              # advisory hint; surfaces in Agent Card
    task_store:                          # default: empty (no database → in-memory)
      database: *my_lakebase_db          # DatabaseModel reference; omit for in-memory
      table: dao_ai_a2a_tasks            # default; only used when database is set
    default_input_modes: [text/plain, application/json]
    default_output_modes: [text/plain, application/json]
    skills:                              # default: derive from app.agents
      - id: classify
        name: Classify
        description: Classify customer emails into categories.
        tags: [email, classification]
        examples:
          - Is this an urgent support request?
    security_schemes:                    # default: bearer
      databricks_oauth:
        type: oauth2
        flows:
          authorizationCode:
            authorizationUrl: ${workspace.host}/oidc/v1/authorize
            tokenUrl:         ${workspace.host}/oidc/v1/token
            scopes:
              all-apis: Call all Databricks workspace APIs
  agents:
    - *my_agent
```

The `${workspace.host}` token is resolved at config-load time by dao-ai's
built-in workspace-variable substitution — no hardcoding required. See
[Security scheme recipes](#security-scheme-recipes) for the full set of
ready-made schemes.

## Semantic mappings

dao-ai's existing capabilities flow through A2A without any agent-side
changes. The mapping is:

| dao-ai concept                                  | A2A concept                                        |
| ----------------------------------------------- | -------------------------------------------------- |
| `thread_id` / `conversation_id`                 | `contextId`                                        |
| In-flight invocation                            | `Task` (`taskId`)                                  |
| LangGraph `interrupt()` raised                  | `TaskStatusUpdateEvent(state=INPUT_REQUIRED, final=True)` with payload as `DataPart` |
| `custom_inputs["decisions"]: list[Decision]`    | Resume `Message` carrying `DataPart` `{"decisions": [...]}` |
| Free-text resume (existing LLM-parser path)     | Resume `Message` with only a `TextPart` — `handle_interrupt_response()` parses it the same way |
| `custom_inputs` (arbitrary dict)                | `DataPart` on inbound `Message`                    |
| OBO via `x-forwarded-access-token`              | Read from request headers; injected into `configurable.headers` (same as Responses path) |
| Structured agent output                         | `TaskArtifactUpdateEvent` with `DataPart`          |
| Streamed AIMessage chunks                       | `TaskStatusUpdateEvent(state=WORKING)`             |
| Terminal AIMessage                              | `TaskStatusUpdateEvent(state=COMPLETED, final=True)` + final `TaskArtifactUpdateEvent` |

## Task persistence

The A2A `Task` lifecycle is a separate concept from the LangGraph
conversation state. dao-ai picks where to store the task lifecycle
metadata based on `app.a2a.task_store`, which mirrors the
`CheckpointerModel` / `StoreModel` idiom: an optional `database` toggles
the backing store.

| `app.a2a.task_store` shape                          | Behavior                                                                                                                            |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Omitted / `{}` (default)                            | **In-memory** (`InMemoryTaskStore`). Process-local; tasks lost on restart. Suitable for short demos and dev loops.                  |
| `{ database: <DatabaseModel> }`                     | **Lakebase-backed** (`LakebaseTaskStore`). Tasks persist in `dao_ai_a2a_tasks` (or `table:`-override) on the referenced database.   |

A2A task persistence is **independent** of `app.background` — the two
concepts (A2A task lifecycle vs Responses-API kickoff/poll/cancel) are
configured separately. To share a connection pool with the background
store and the LangGraph Postgres checkpointer, point all three at the
same `DatabaseModel` anchor; `AsyncPostgresPoolManager` dedupes by
connection-string value.

The Lakebase table is `(task_id PRIMARY KEY, context_id, state,
task_json, created_at, updated_at)`. `LakebaseTaskStore.ensure_schema()`
creates the table idempotently on first use (skipped when a provisioning
notebook has already created it).

Example — Lakebase-backed task store sharing the background pool:

```yaml
resources:
  databases:
    a2a_db: &a2a_db
      name: dao-ai-a2a-demo
      project: dao-ai-a2a-demo

memory: &memory
  checkpointer:
    name: a2a_checkpointer
    database: *a2a_db          # shared pool anchor

app:
  background:
    database: *a2a_db          # shared pool anchor
  a2a:
    task_store:
      database: *a2a_db        # shared pool anchor
      table: dao_ai_a2a_tasks  # default; override if multiple apps share the DB
```

## HITL over A2A

The dao-ai HITL contract is preserved over A2A unchanged:

1. The agent reaches a LangGraph `interrupt()` mid-execution.
2. A2A emits a terminal `TaskStatusUpdateEvent` with
   `state=INPUT_REQUIRED`. The `status.message` body contains a
   `DataPart` with `{"interrupts": [...]}` so machine clients can act
   on it programmatically.
3. The client supplies the resume payload by sending another
   `message/send` for the same `taskId` + `contextId`. Two shapes work:
   * **Structured**: a `DataPart` with
     `{"decisions": [{"type": "approve"}, ...]}` — bypasses the LLM
     parser and is the canonical machine-to-machine contract.
   * **Free-text**: a `TextPart` carrying a natural-language response.
     dao-ai's `handle_interrupt_response()` LLM parser converts it to
     decisions, same as on the Responses path.
4. The agent resumes with `Command(resume={"decisions": ...})` and
   transitions back to `WORKING`, eventually `COMPLETED`.

## OBO over A2A

Databricks Apps forwards the caller's bearer token via the
`x-forwarded-access-token` header. dao-ai's A2A executor reads it from
the FastAPI request (via a2a-sdk's `DefaultCallContextBuilder`) and
injects it into `configurable.headers`, exactly mirroring the
Responses-path handler. Downstream tools (UC functions, AI Search,
Genie, model invocations) that have `on_behalf_of_user: true` see the
end-user's token unchanged.

### Agent Card auto-derivation

`app.a2a.on_behalf_of_user` is a three-state flag:

| Value | Meaning | Agent Card emits |
|---|---|---|
| `null` (default) | Auto-derive from resources | `oauth2` + `bearer` schemes iff **any** resource in the config has `on_behalf_of_user: true` |
| `true` (explicit) | Force-advertise OBO | `oauth2` + `bearer` schemes |
| `false` (explicit) | Force-suppress OBO | single PAT/M2M `bearer` scheme |

When effective OBO is True, the Agent Card emits **both**:

* `oauth2` — the declarative auth flow (`authorizationCode` against
  `${workspace.host}/oidc/v1/authorize` + `oidc/v1/token`, scope
  `user_impersonation`). This is what OAuth2-aware A2A clients consume.
* `bearer` — the wire-level shape callers actually send. The
  Apps proxy forwards via `x-forwarded-access-token` regardless of which
  scheme the client thinks it's satisfying.

A2A's `security` requirement array lists both schemes; clients pick
whichever their auth machinery supports. Workspace host comes from
`$DATABRICKS_HOST` (auto-set in the Apps runtime). If unresolvable at
boot, the Agent Card falls back to the OBO-aware `bearer` scheme alone
and logs a warning.

### Example: auto-derived OBO from a resource

```yaml
resources:
  models:
    default_llm: &default_llm
      name: databricks-gpt-5-4-mini
      on_behalf_of_user: true       # resource-level OBO

agents:
  - name: my_agent
    model: *default_llm

app:
  name: my-app
  # deploy with: dao-ai agent deploy --mode apps
  agents: [...]
  # No a2a block — Agent Card auto-emits oauth2 + bearer
```

This produces the same Agent Card as setting
`app.a2a.on_behalf_of_user: true` explicitly. The convenience factory
`oauth2_databricks_obo()` is the building block used internally.

To declare a custom OAuth2 scheme set (different scopes, additional
flows, etc.), set `app.a2a.security_schemes` explicitly — see
[Security scheme recipes](#security-scheme-recipes).

## Security scheme recipes

`A2AModel.security_schemes` is typed against a2a-sdk's
`SecurityScheme` discriminated union, so any dict you write in YAML is
validated at config-load time (malformed schemes fail at boot, not at
first request). dao-ai ships ready-made schemes in
`dao_ai.apps.a2a.security` for the most common Databricks-flavored
cases.

### Python (programmatic config)

```python
from dao_ai.apps.a2a.security import (
    BEARER_DATABRICKS_PAT,
    BEARER_DATABRICKS_M2M,
    BEARER_DATABRICKS_OBO,
    api_key_header,
    oauth2_databricks_authorization_code,
    oauth2_databricks_client_credentials,
    oauth2_databricks_obo,
    openid_connect_databricks,
)
from dao_ai.config import A2AModel

a2a = A2AModel(
    security_schemes={
        "bearer": BEARER_DATABRICKS_OBO,
        "oauth2": oauth2_databricks_obo(),  # host auto-resolved from $DATABRICKS_HOST
    }
)
```

Each factory takes an optional `host` argument; when omitted, the host
is resolved from `$DATABRICKS_HOST` (auto-set in the Databricks Apps
runtime) or the ambient `WorkspaceClient` config.

### YAML (declarative config)

The same recipes inlined in YAML — `${workspace.host}` is resolved at
config load by dao-ai's built-in substitution.

#### Bearer (Databricks PAT)

```yaml
a2a:
  security_schemes:
    bearer:
      type: http
      scheme: bearer
      bearerFormat: Databricks PAT
      description: Bearer token issued by a Databricks workspace user.
```

#### Bearer (Databricks OAuth M2M / service principal)

```yaml
a2a:
  security_schemes:
    bearer:
      type: http
      scheme: bearer
      bearerFormat: Databricks OAuth M2M
      description: |
        Bearer token minted via OAuth M2M client-credentials for a
        Databricks service principal.
```

#### Bearer (OBO via Apps proxy)

Equivalent to setting `app.a2a.on_behalf_of_user: true`:

```yaml
a2a:
  on_behalf_of_user: true   # short form — derives a bearer scheme like the one below
```

…or set the scheme explicitly:

```yaml
a2a:
  security_schemes:
    bearer:
      type: http
      scheme: bearer
      bearerFormat: Databricks OAuth (forwarded by Apps proxy via x-forwarded-access-token; OBO supported)
      description: |
        Databricks Apps forwards the calling user's bearer token via
        the x-forwarded-access-token header.
```

#### API key in a header

```yaml
a2a:
  security_schemes:
    api_key:
      type: apiKey
      in: header
      name: X-API-Key
```

#### OAuth2 authorization_code (three-legged)

```yaml
a2a:
  security_schemes:
    oauth2:
      type: oauth2
      flows:
        authorizationCode:
          authorizationUrl: ${workspace.host}/oidc/v1/authorize
          tokenUrl:         ${workspace.host}/oidc/v1/token
          scopes:
            all-apis: Full Databricks REST API surface.
```

#### OAuth2 client_credentials (M2M)

```yaml
a2a:
  security_schemes:
    oauth2_m2m:
      type: oauth2
      flows:
        clientCredentials:
          tokenUrl: ${workspace.host}/oidc/v1/token
          scopes:
            all-apis: Full Databricks REST API surface.
```

#### OAuth2 OBO (advertise user_impersonation)

```yaml
a2a:
  security_schemes:
    oauth2_obo:
      type: oauth2
      flows:
        authorizationCode:
          authorizationUrl: ${workspace.host}/oidc/v1/authorize
          tokenUrl:         ${workspace.host}/oidc/v1/token
          scopes:
            user_impersonation: Act on behalf of the calling user.
```

#### OpenID Connect discovery

```yaml
a2a:
  security_schemes:
    oidc:
      type: openIdConnect
      openIdConnectUrl: ${workspace.host}/oidc/.well-known/openid-configuration
```

## End-to-end example

See `examples/20_a2a_protocol/client/client.py` for a Python script that:

1. Fetches the Agent Card.
2. Calls `message/send` and prints the artifact.
3. Calls `message/stream` and prints each SSE event.
4. Demonstrates a HITL flow: triggers an interrupt, then resumes with a
   `DataPart` decisions payload.

And `examples/20_a2a_protocol/a2a_minimal.yaml` for the deploy-
ready agent config.

## Disabling A2A

```yaml
app:
  a2a:
    enabled: false
```

This is the only knob that prevents the routes from being mounted. The
A2A executor and task store are not constructed when disabled, so the
opt-out is also a perf/cost-of-zero opt-out.

## Spec compliance

dao-ai targets [A2A v0.3](https://a2a-protocol.org) (matching the
installed `a2a-sdk` 0.3.x pin in dao-ai's `pyproject.toml`). The
implementation passes the protocol's mandatory verbs:

* `message/send` (single-shot)
* `message/stream` (SSE)
* `tasks/get`
* `tasks/list`
* `tasks/cancel`
* `tasks/subscribe`

Out-of-scope for the initial release (defer to future work):

* gRPC binding (HTTP + JSON-RPC over SSE is sufficient for v0.3 compliance).
* Signed Agent Cards (an enterprise v1.0 feature; dao-ai pins to v0.3).
* `FilePart` input/output (TextPart + DataPart are supported; FilePart
  is currently ignored on input).
* Bridging the A2A `taskId` to the dao-ai long-running `response_id` —
  in Phase 1 they're parallel views over the same checkpointer state.
