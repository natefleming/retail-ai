# Databricks notebook source
# MAGIC %md
# MAGIC # Background Agents demo
# MAGIC
# MAGIC A **background agent** runs a long task server-side: you *kick it off*, get
# MAGIC a response id back immediately, *poll* for the result, and can *cancel* it
# MAGIC mid-run.
# MAGIC
# MAGIC This notebook drives a background-enabled dao-ai agent deployed as a
# MAGIC **Databricks App**. It can optionally deploy the app for you (from the
# MAGIC `background_research.yaml` example), then exercises the lifecycle:
# MAGIC **kickoff → poll → cancel**, plus a look at the streaming event surface.
# MAGIC
# MAGIC ### Client note
# MAGIC The app exposes OpenAI Responses API–compatible routes under `/v1`
# MAGIC (`POST /v1/responses`, `GET /v1/responses/{id}`, `POST /v1/responses/{id}/cancel`).
# MAGIC We resolve the app's URL from the app name via the Databricks SDK, then point
# MAGIC the stock `OpenAI` client at `{app_url}/v1`.
# MAGIC
# MAGIC > **Auth:** the Databricks Apps front door only accepts an OAuth access token
# MAGIC > *audience-scoped to the app's `oauth2_app_client_id`*. The notebook's ambient
# MAGIC > credential is not app-scoped and gets a bare `401`, so we exchange the notebook
# MAGIC > token for an app-scoped one via `POST {host}/oidc/v1/token` (token-exchange
# MAGIC > grant). See Databricks docs: *dev-tools/databricks-apps/connect-local*.
# MAGIC
# MAGIC > The `databricks-openai` `DatabricksOpenAI` client with `model="apps/<name>"`
# MAGIC > routes `responses.create` to the app, but its base URL omits the `/v1`
# MAGIC > prefix, so `responses.retrieve`/`cancel` (the poll + cancel legs) 404.
# MAGIC > Building `OpenAI(base_url=f"{app_url}/v1")` supports the whole lifecycle.

# COMMAND ----------

# MAGIC %uv pip install --quiet databricks-sdk databricks-openai openai
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

dbutils.widgets.text(name="app_name", defaultValue="background-dao", label="Databricks App name")
dbutils.widgets.dropdown(name="deploy_app", defaultValue="false", choices=["true", "false"], label="Deploy the app first?")
dbutils.widgets.text(name="config_path", defaultValue="../examples/18_background_agents/background_research.yaml", label="Config (for deploy)")
dbutils.widgets.text(name="lakebase_project", defaultValue="", label="Lakebase project override (optional)")
dbutils.widgets.text(name="prompt", defaultValue="List 5 reasons to use Databricks Lakebase.", label="Prompt")
dbutils.widgets.text(name="thread_id", defaultValue="nb_bg_demo_1", label="Conversation thread_id")
dbutils.widgets.text(name="poll_interval_seconds", defaultValue="2", label="Poll interval (s)")
dbutils.widgets.text(name="max_poll_seconds", defaultValue="240", label="Max poll wait (s)")

app_name: str = dbutils.widgets.get("app_name")
deploy_app: bool = dbutils.widgets.get("deploy_app") == "true"
config_path: str = dbutils.widgets.get("config_path")
lakebase_project: str = dbutils.widgets.get("lakebase_project").strip()
prompt: str = dbutils.widgets.get("prompt")
thread_id: str = dbutils.widgets.get("thread_id")
poll_interval_seconds: float = float(dbutils.widgets.get("poll_interval_seconds"))
max_poll_seconds: float = float(dbutils.widgets.get("max_poll_seconds"))

print("App name:", app_name)
print("Deploy first:", deploy_app)

# COMMAND ----------

# MAGIC %md
# MAGIC ## (Optional) Deploy the target app
# MAGIC
# MAGIC Set **Deploy the app first? = true** to deploy the example config to Apps
# MAGIC via the dao-ai SDK before running the demo. This is blocking (~2–5 min) and
# MAGIC deploys `background_research.yaml`, whose `app.name` (`background_dao`)
# MAGIC becomes the App **`background-dao`**. Leave `false` if it's already up
# MAGIC (deploy once from a shell with
# MAGIC `dao-ai agent up -c examples/18_background_agents/background_research.yaml --mode apps`).

# COMMAND ----------

if deploy_app:
    from dao_ai.config import AppConfig, ServingMode

    params: dict[str, str] | None = {"lakebase_project": lakebase_project} if lakebase_project else None
    deploy_config: AppConfig = AppConfig.from_file(config_path, params=params)
    deploy_config.deploy_agent(mode=ServingMode.APPS)
    print("Deployed. app.name =", deploy_config.app.name)
else:
    print(f"Skipping deploy (using existing app: {app_name})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Client
# MAGIC
# MAGIC Resolve the app URL + OAuth token from the app name, then build an
# MAGIC OpenAI-compatible client against `{app_url}/v1`.

# COMMAND ----------

import time

import requests
from databricks.sdk import WorkspaceClient
from openai import OpenAI, Stream
from openai.types.responses import Response, ResponseStreamEvent

w = WorkspaceClient()
app = w.apps.get(app_name)
app_url: str = app.url.rstrip("/")

# The Databricks Apps front door (*.databricksapps.com) only accepts an OAuth
# access token that is audience-scoped to the app's client id. The notebook's
# ambient credential is NOT scoped to the app, so we exchange it for an
# app-scoped token via /oidc/v1/token (the documented Apps auth recipe).
app_client_id: str | None = app.oauth2_app_client_id
if not app_client_id:
    raise RuntimeError(
        f"App {app_name!r} has no oauth2_app_client_id; cannot mint an "
        "app-scoped token. Ensure the app is fully deployed."
    )

host: str = w.config.host.rstrip("/")
notebook_token: str = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .apiToken()
    .get()
)
token: str = requests.post(
    f"{host}/oidc/v1/token",
    data={
        "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
        "subject_token": notebook_token,
        "subject_token_type": "urn:databricks:params:oauth:token-type:personal-access-token",
        "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
        "scope": "all-apis",
        "audience": app_client_id,
    },
).json()["access_token"]

client = OpenAI(base_url=f"{app_url}/v1", api_key=token)
print("App URL:", app_url)

TERMINAL: set[str] = {"completed", "failed", "cancelled", "incomplete"}


def messages(text: str) -> list[dict[str, str]]:
    # dao-ai's ResponsesAgentRequest expects `input` as a list of messages.
    return [{"role": "user", "content": text}]


def custom_inputs(thread: str) -> dict[str, dict]:
    # Passed through extra_body → how dao-ai receives the configurable thread_id.
    return {"custom_inputs": {"configurable": {"thread_id": thread}}}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Kick off + poll
# MAGIC
# MAGIC `background=True` returns immediately with a `resp_...` id and
# MAGIC `status="in_progress"`. Poll with `responses.retrieve(id)` until terminal.
# MAGIC This agent runs fully async — the result materializes on the polled
# MAGIC response's `output_text`, not on the initial kickoff.

# COMMAND ----------

kickoff: Response = client.responses.create(
    model=app_name,
    input=messages(prompt),
    background=True,
    extra_body=custom_inputs(thread_id),
)
response_id: str = kickoff.id
print(f"kickoff id={response_id} status={kickoff.status}")

deadline: float = time.monotonic() + max_poll_seconds
final: Response = kickoff
while final.status not in TERMINAL and time.monotonic() < deadline:
    time.sleep(poll_interval_seconds)
    final = client.responses.retrieve(response_id)
    print(f"  poll: status={final.status}")

print(f"\nfinal status={final.status}")
print(final.output_text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Cancel a running task
# MAGIC
# MAGIC Kick off a longer task, then `responses.cancel(id)` before it finishes and
# MAGIC confirm the status flips to `cancelled`.

# COMMAND ----------

kickoff = client.responses.create(
    model=app_name,
    input=messages("Write a detailed 1000-word essay on Databricks Lakebase."),
    background=True,
    extra_body=custom_inputs("nb_bg_cancel_1"),
)
response_id = kickoff.id
print(f"kickoff id={response_id} status={kickoff.status}")

cancelled: Response = client.responses.cancel(response_id)
print(f"after cancel: status={cancelled.status}")

time.sleep(1)
after: Response = client.responses.retrieve(response_id)
print(f"retrieve-after-cancel: status={after.status}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Stream a background task's events (OpenAI client)
# MAGIC
# MAGIC After kicking a task off, stream its stored events back with
# MAGIC `responses.retrieve(id, stream=True)`. The `starting_after` cursor resumes
# MAGIC after the last event you saw — on reconnect you don't re-receive events.
# MAGIC The stream ends when the task reaches a terminal status.

# COMMAND ----------

from openai.types.responses import ResponseOutputMessage

kickoff: Response = client.responses.create(
    model=app_name,
    input=messages("List 10 things Databricks Lakebase is good for."),
    background=True,
    extra_body=custom_inputs("nb_bg_stream_1"),
)
response_id: str = kickoff.id
print(f"kickoff id={response_id}")

retrieve_stream: Stream[ResponseStreamEvent] = client.responses.retrieve(
    response_id, stream=True, starting_after=0
)

streamed_text: str = ""
saw_delta: bool = False
event: ResponseStreamEvent
for event in retrieve_stream:
    if event.type == "response.output_text.delta":
        streamed_text += event.delta
        saw_delta = True
        print(event.delta, end="", flush=True)  # render tokens as they arrive
    elif (
        not saw_delta
        and event.type == "response.output_item.done"
        and isinstance(event.item, ResponseOutputMessage)
    ):
        # Fallback for agents that emit output in one piece (no token deltas).
        for part in event.item.content:
            if part.type == "output_text":
                streamed_text += part.text
                print(part.text, end="", flush=True)

print()  # trailing newline after the streamed output

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Stream the agent's output
# MAGIC
# MAGIC The same `OpenAI(/v1)` client streams the agent's **output** as it is
# MAGIC produced — pass `stream=True` with `background=False` so the run executes
# MAGIC synchronously and emits token deltas (rather than a single background
# MAGIC status event). We accumulate `output_text.delta` events (token deltas) and
# MAGIC fall back to the text on a final `output_item.done` event for agents that
# MAGIC emit output in one piece.

# COMMAND ----------

from openai.types.responses import ResponseOutputMessage

stream: Stream[ResponseStreamEvent] = client.responses.create(
    model=app_name,
    input=messages("Give me 3 quick tips for using Databricks Lakebase."),
    stream=True,
    background=False,
    extra_body=custom_inputs("nb_bg_stream_output_1"),
)

streamed_text: str = ""
saw_delta: bool = False
event: ResponseStreamEvent
for event in stream:
    if event.type == "response.output_text.delta":
        streamed_text += event.delta
        saw_delta = True
        print(event.delta, end="", flush=True)  # render tokens as they arrive
    elif (
        not saw_delta
        and event.type == "response.output_item.done"
        and isinstance(event.item, ResponseOutputMessage)
    ):
        # Fallback for agents that emit output in one piece (no token deltas).
        for part in event.item.content:
            if part.type == "output_text":
                streamed_text += part.text
                print(part.text, end="", flush=True)

print()  # trailing newline after the streamed output
