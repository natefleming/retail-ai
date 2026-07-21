# Databricks notebook source
# MAGIC %md
# MAGIC # Background Agents — full feature demo
# MAGIC
# MAGIC Exercises **every** feature of dao-ai's background agent wrapper against
# MAGIC two deployment targets:
# MAGIC
# MAGIC | # | Flow | Databricks Apps | Model Serving |
# MAGIC |---|---|---|---|
# MAGIC | 1 | **Sync passthrough** (no `background`) | ✓ | ✓ |
# MAGIC | 2 | **Sync streaming** (SSE, no `background`) | ✓ | ✓ |
# MAGIC | 3 | **Background kickoff + non-stream poll** | ✓ | ✓ |
# MAGIC | 4 | **Background kickoff + stream retrieve** | ✓ | ✗ (Apps only) |
# MAGIC | 5 | **Cancel a running background task** | ✓ | ✓ |
# MAGIC
# MAGIC Apps exposes strict OpenAI Responses API routes (`POST /v1/responses`,
# MAGIC `GET /v1/responses/{id}`, `POST /v1/responses/{id}/cancel`) when
# MAGIC `app.background` is configured. Model Serving only exposes
# MAGIC `/invocations` (and `/invocations-stream` for sync SSE), so retrieve and
# MAGIC cancel ride on `custom_inputs.operation` there.
# MAGIC
# MAGIC Both targets run the same `BackgroundResponsesAgent` wrapper. Background
# MAGIC work runs on a persistent daemon thread (`_BackgroundLoop`) so the task
# MAGIC survives Model Serving's per-request `asyncio.run()` teardown.

# COMMAND ----------

# MAGIC %pip install --quiet requests httpx
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Setup
# MAGIC
# MAGIC Two endpoints and a bearer token. Adjust the env vars if your deployment
# MAGIC lives elsewhere. Inside a Databricks notebook, `dbutils` can also supply
# MAGIC the token — but this notebook uses a single portable path so it runs
# MAGIC identically from a workstation shell.

# COMMAND ----------

import json
import os
import subprocess
import time
from typing import Any

import httpx
import requests

APP_URL = os.environ.get(
    "APP_URL",
    "https://background-dao-1444828305810485.aws.databricksapps.com",
).rstrip("/")
MS_ENDPOINT = os.environ.get("MS_ENDPOINT", "background_dao")
DATABRICKS_HOST = os.environ.get(
    "DATABRICKS_HOST", "https://e2-demo-field-eng.cloud.databricks.com"
).rstrip("/")

TOKEN = os.environ.get("DATABRICKS_TOKEN")
if not TOKEN:
    _tok = subprocess.check_output(
        [
            "databricks",
            "--profile",
            os.environ.get("DATABRICKS_CONFIG_PROFILE", "aws-field-eng"),
            "auth",
            "token",
        ]
    )
    TOKEN = json.loads(_tok)["access_token"]

HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
MS_URL = f"{DATABRICKS_HOST}/serving-endpoints/{MS_ENDPOINT}/invocations"

# Pretty-print helper used throughout the notebook.
def _short(text: str, limit: int = 400) -> str:
    t = (text or "").replace("\n", " ")
    return t if len(t) <= limit else t[:limit] + "…"


def _first_output_text(body: dict) -> str:
    for item in body.get("output", []) or []:
        if item.get("type") == "message":
            for part in item.get("content", []) or []:
                if part.get("type") == "output_text":
                    return part.get("text", "")
    return ""


print("App URL: ", APP_URL)
print("MS URL:  ", MS_URL)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Sync passthrough (no `background`)
# MAGIC
# MAGIC When `background` is omitted/false AND `custom_inputs.operation` isn't
# MAGIC set, the wrapper simply delegates to the inner `LanggraphResponsesAgent`
# MAGIC — i.e. the agent behaves exactly as it would without long-running
# MAGIC configured. This section proves both targets still do ordinary
# MAGIC request/response work.

# COMMAND ----------

# --- 1a. Apps: POST /invocations (no background) ---
r = requests.post(
    f"{APP_URL}/invocations",
    headers=HEADERS,
    json={
        "input": [{"role": "user", "content": "What is Databricks Lakebase? One sentence."}],
        "custom_inputs": {"configurable": {"thread_id": "nb_sync_apps_1"}},
    },
    timeout=120,
)
r.raise_for_status()
apps_sync = r.json()
print("[Apps sync] HTTP 200")
print(" output_text:", _short(_first_output_text(apps_sync), 200))

# COMMAND ----------

# --- 1b. Model Serving: POST /invocations (no background) ---
r = requests.post(
    MS_URL,
    headers=HEADERS,
    json={
        "input": [{"role": "user", "content": "What is Databricks Lakebase? One sentence."}],
        "custom_inputs": {"configurable": {"thread_id": "nb_sync_ms_1"}},
    },
    timeout=120,
)
r.raise_for_status()
ms_sync = r.json()
print("[MS sync] HTTP 200, databricks_request_id=", ms_sync["id"])
print(" output_text:", _short(_first_output_text(ms_sync), 200))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Sync streaming (no `background`)
# MAGIC
# MAGIC Standard MLflow `predict_stream` — Server-Sent Events stream of
# MAGIC `ResponsesAgentStreamEvent`s. On both targets, streaming is requested
# MAGIC by setting `"stream": true` in the request body to `/invocations`;
# MAGIC the AgentServer returns SSE instead of a JSON envelope.
# MAGIC
# MAGIC This is an existing dao-ai feature — the background wrapper doesn't change it.
# MAGIC The cell confirms the wrapper's passthrough behaves identically to a
# MAGIC deployment without `app.background`.

# COMMAND ----------

def _iter_sse(resp: httpx.Response) -> "list[dict[str, Any]]":
    """Parse an SSE response body into a flat list of event dicts.

    Tolerates SSE framing boilerplate (``: keepalive`` comments, ``data: [DONE]``
    sentinels, blank ``data:`` lines) by skipping non-JSON payloads.
    """
    evts: list[dict[str, Any]] = []
    for line in resp.iter_lines():
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            evts.append(json.loads(payload))
        except json.JSONDecodeError:
            # Skip non-JSON heartbeat/comment lines.
            continue
    return evts

# --- 2a. Apps: POST /invocations with stream=true ---
with httpx.stream(
    "POST",
    f"{APP_URL}/invocations",
    headers=HEADERS,
    json={
        "input": [{"role": "user", "content": "Count to 5 slowly."}],
        "stream": True,
        "custom_inputs": {"configurable": {"thread_id": "nb_stream_apps_1"}},
    },
    timeout=None,
) as resp:
    apps_stream_events = _iter_sse(resp)

event_types = [ev.get("type") for ev in apps_stream_events]
for ev in apps_stream_events:
    if ev.get("type") == "response.output_text.delta":
        print(" delta:", _short(ev.get("delta", ""), 80))
print("\n[Apps sync stream] received", len(apps_stream_events),
      "events; types:", sorted(set(event_types)))

# COMMAND ----------

# --- 2b. Model Serving: POST /invocations with stream=true ---
with httpx.stream(
    "POST",
    MS_URL,
    headers=HEADERS,
    json={
        "input": [{"role": "user", "content": "Count to 5 slowly."}],
        "stream": True,
        "custom_inputs": {"configurable": {"thread_id": "nb_stream_ms_1"}},
    },
    timeout=None,
) as resp:
    ms_stream_events = _iter_sse(resp)

event_types = [ev.get("type") for ev in ms_stream_events]
for ev in ms_stream_events:
    if ev.get("type") == "response.output_text.delta":
        print(" delta:", _short(ev.get("delta", ""), 80))
print("\n[MS sync stream] received", len(ms_stream_events),
      "events; types:", sorted(set(event_types)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Background kickoff + non-streaming poll
# MAGIC
# MAGIC The core long-running flow:
# MAGIC
# MAGIC 1. **Kickoff** — `background=true` makes the wrapper generate a
# MAGIC    `resp_…` id, persist a row in `dao_ai_responses`, spawn the agent
# MAGIC    work on its `_BackgroundLoop` thread, and return immediately with
# MAGIC    `status: "in_progress"`.
# MAGIC 2. **Poll** — retrieve the response by id until status is terminal.
# MAGIC 3. **Final output** — `output` is populated by MLflow's
# MAGIC    `responses_agent_output_reducer` from the stream events the
# MAGIC    background task persisted.

# COMMAND ----------

# --- 3a. Apps: POST /v1/responses (background) + GET /v1/responses/{id} ---
r = requests.post(
    f"{APP_URL}/v1/responses",
    headers=HEADERS,
    json={
        "input": [{"role": "user", "content": "List 5 reasons to use Lakebase."}],
        "background": True,
        "custom_inputs": {"configurable": {"thread_id": "nb_bg_apps_1"}},
    },
    timeout=30,
)
r.raise_for_status()
kickoff = r.json()
rid = kickoff["id"]
print("[Apps bg] kickoff id=", rid, "status=", kickoff["status"])

for attempt in range(120):
    r = requests.get(f"{APP_URL}/v1/responses/{rid}", headers=HEADERS, timeout=30)
    r.raise_for_status()
    body = r.json()
    print(f"  poll {attempt:02d}: status={body['status']}")
    if body["status"] in {"completed", "failed", "cancelled"}:
        break
    time.sleep(2)

print("\n[Apps bg] final:\n", _short(_first_output_text(body), 500))

# COMMAND ----------

# --- 3b. Model Serving: POST /invocations (background) + operation=retrieve ---
def _ms(body: dict) -> dict:
    r = requests.post(MS_URL, headers=HEADERS, json=body, timeout=120)
    r.raise_for_status()
    return r.json()

kickoff = _ms(
    {
        "input": [{"role": "user", "content": "List 5 reasons to use Lakebase."}],
        "background": True,
        "custom_inputs": {"configurable": {"thread_id": "nb_bg_ms_1"}},
    }
)
rid = kickoff["id"]
print("[MS bg] kickoff id=", rid, "status=", kickoff["status"])

for attempt in range(120):
    body = _ms(
        {
            "input": [],
            "custom_inputs": {"operation": "retrieve", "response_id": rid},
        }
    )
    print(f"  poll {attempt:02d}: status={body['status']}")
    if body["status"] in {"completed", "failed", "cancelled"}:
        break
    time.sleep(2)

print("\n[MS bg] final:\n", _short(_first_output_text(body), 500))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Background kickoff + streaming retrieve (Apps only)
# MAGIC
# MAGIC `GET /v1/responses/{id}?stream=true&cursor=0` returns the stored
# MAGIC stream events as SSE. Each event's
# MAGIC `custom_outputs.background.status` is authoritative — stop when
# MAGIC it's terminal.
# MAGIC
# MAGIC Streaming retrieve is Apps-only because Model Serving can't mount
# MAGIC custom SSE routes. A background run started on MS **can** still be
# MAGIC polled non-stream (section 3b).

# COMMAND ----------

r = requests.post(
    f"{APP_URL}/v1/responses",
    headers=HEADERS,
    json={
        "input": [{"role": "user", "content": "Name 3 features of Delta Lake."}],
        "background": True,
        "custom_inputs": {"configurable": {"thread_id": "nb_stream_bg_apps_1"}},
    },
    timeout=30,
)
r.raise_for_status()
rid = r.json()["id"]
print("[Apps bg stream] kickoff id=", rid)

with httpx.stream(
    "GET",
    f"{APP_URL}/v1/responses/{rid}?stream=true&cursor=0",
    headers=HEADERS,
    timeout=None,
) as resp:
    event_count = 0
    terminal = None
    for line in resp.iter_lines():
        if not line or not line.startswith("data:"):
            continue
        ev = json.loads(line[len("data:") :].strip())
        event_count += 1
        background = (ev.get("custom_outputs") or {}).get("background") or {}
        status = background.get("status", "?")
        print(f"  evt#{event_count:03d} type={ev.get('type'):<28} status={status}")
        if status in {"completed", "failed", "cancelled"}:
            terminal = status
            break

print(f"\n[Apps bg stream] {event_count} events, terminal={terminal}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Cancel a running background task
# MAGIC
# MAGIC Cancellation is best-effort same-pod — the server always marks the
# MAGIC response `cancelled` in Lakebase, and *if* the current pod holds the
# MAGIC `asyncio.Task` it also calls `.cancel()` on it. Cross-pod cancel is
# MAGIC out of scope (matches the 1DD).
# MAGIC
# MAGIC Kickoff a long prompt, immediately cancel, then retrieve to confirm
# MAGIC the status is stable.

# COMMAND ----------

# --- 5a. Apps: POST /v1/responses/{id}/cancel ---
r = requests.post(
    f"{APP_URL}/v1/responses",
    headers=HEADERS,
    json={
        "input": [{"role": "user", "content": "Write a 1000-word essay on Lakebase."}],
        "background": True,
        "custom_inputs": {"configurable": {"thread_id": "nb_cancel_apps_1"}},
    },
    timeout=30,
)
r.raise_for_status()
rid = r.json()["id"]
print("[Apps cancel] kickoff id=", rid)

r = requests.post(f"{APP_URL}/v1/responses/{rid}/cancel", headers=HEADERS, timeout=30)
r.raise_for_status()
print("[Apps cancel] response status=", r.json()["status"])

time.sleep(1)
r = requests.get(f"{APP_URL}/v1/responses/{rid}", headers=HEADERS, timeout=30)
r.raise_for_status()
print("[Apps cancel] retrieve-after-cancel status=", r.json()["status"])

# COMMAND ----------

# --- 5b. Model Serving: custom_inputs.operation = "cancel" ---
kickoff = _ms(
    {
        "input": [{"role": "user", "content": "Write a 1000-word essay on Lakebase."}],
        "background": True,
        "custom_inputs": {"configurable": {"thread_id": "nb_cancel_ms_1"}},
    }
)
rid = kickoff["id"]
print("[MS cancel] kickoff id=", rid)

cancel = _ms(
    {
        "input": [],
        "custom_inputs": {"operation": "cancel", "response_id": rid},
    }
)
print("[MS cancel] cancel response status=", cancel["status"])

time.sleep(1)
retrieve = _ms(
    {
        "input": [],
        "custom_inputs": {"operation": "retrieve", "response_id": rid},
    }
)
print("[MS cancel] retrieve-after-cancel status=", retrieve["status"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Optional: resume from a cursor (streaming)
# MAGIC
# MAGIC Streaming retrieve accepts a `cursor` query parameter — the last
# MAGIC `sequence_number` the client saw. On reconnect, the server resumes
# MAGIC after that cursor so the client doesn't re-receive events.

# COMMAND ----------

# Kick off, read a few events, disconnect, then resume at the cursor.
r = requests.post(
    f"{APP_URL}/v1/responses",
    headers=HEADERS,
    json={
        "input": [{"role": "user", "content": "List 10 synonyms for 'happy'."}],
        "background": True,
        "custom_inputs": {"configurable": {"thread_id": "nb_resume_apps_1"}},
    },
    timeout=30,
)
r.raise_for_status()
rid = r.json()["id"]
print("[resume] kickoff id=", rid)

# First pass: read 3 events and break
last_cursor = 0
with httpx.stream(
    "GET",
    f"{APP_URL}/v1/responses/{rid}?stream=true&cursor=0",
    headers=HEADERS,
    timeout=None,
) as resp:
    for i, line in enumerate(resp.iter_lines()):
        if not line or not line.startswith("data:"):
            continue
        ev = json.loads(line[len("data:") :].strip())
        last_cursor = (ev.get("custom_outputs") or {}).get("background", {}).get(
            "cursor", last_cursor
        )
        print(f"  first-pass evt#{i:02d} cursor={last_cursor}")
        if i >= 2:
            break

# Reconnect starting after last_cursor
print(f"\n[resume] reconnecting with cursor={last_cursor}")
with httpx.stream(
    "GET",
    f"{APP_URL}/v1/responses/{rid}?stream=true&cursor={last_cursor}",
    headers=HEADERS,
    timeout=None,
) as resp:
    for i, line in enumerate(resp.iter_lines()):
        if not line or not line.startswith("data:"):
            continue
        ev = json.loads(line[len("data:") :].strip())
        status = (ev.get("custom_outputs") or {}).get("background", {}).get("status")
        print(f"  resumed evt#{i:02d} status={status}")
        if status in {"completed", "failed", "cancelled"}:
            break

# COMMAND ----------

# MAGIC %md
# MAGIC ## (Reference) OpenAI Responses client
# MAGIC
# MAGIC The routes on Apps are OpenAI Responses API–compatible, so the stock
# MAGIC OpenAI Python client works against the Apps endpoint with
# MAGIC `base_url=f"{APP_URL}/v1"`:
# MAGIC
# MAGIC ```python
# MAGIC from openai import OpenAI
# MAGIC
# MAGIC client = OpenAI(base_url=f"{APP_URL}/v1", api_key=TOKEN)
# MAGIC
# MAGIC resp = client.responses.create(
# MAGIC     model="databricks-long-running",          # ignored by the server
# MAGIC     input="Research long-running agent infra at Databricks",
# MAGIC     background=True,
# MAGIC     extra_body={"custom_inputs": {"configurable": {"thread_id": "demo"}}},
# MAGIC )
# MAGIC retrieved = client.responses.retrieve(resp.id)
# MAGIC client.responses.cancel(resp.id)
# MAGIC ```
# MAGIC
# MAGIC Model Serving doesn't mount strict routes, so against MS the OpenAI
# MAGIC client can't be used directly — use `requests` / `httpx` (as
# MAGIC demonstrated throughout this notebook) or the `databricks-sdk`
# MAGIC `ServingEndpoints` extension.
