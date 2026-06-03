# Databricks notebook source
# MAGIC %md
# MAGIC # dao-ai Multi-Agent Feedback Demo
# MAGIC
# MAGIC Capture thumbs-up / thumbs-down feedback against a dao-ai response. Works
# MAGIC the same for in-process invocation, the deployed Model Serving endpoint,
# MAGIC and the deployed Databricks App.
# MAGIC
# MAGIC **The contract**
# MAGIC
# MAGIC - Every dao-ai response carries `custom_outputs["trace_id"]`.
# MAGIC - In multi-agent supervisor/swarm flows, that trace_id is the **outer**
# MAGIC   root trace whose children include every sub-agent hop.
# MAGIC - Pass it to `dao_ai.evaluation.log_user_feedback(...)` to attach an
# MAGIC   assessment to that exact trace.

# COMMAND ----------

# MAGIC %pip install -q --upgrade -r ../requirements.txt nest-asyncio>=1.6.0
# MAGIC %restart_python

# COMMAND ----------

import asyncio
import json
import os
import sys

import nest_asyncio
nest_asyncio.apply()

# Force synchronous trace export so the demo's tight loop (invoke → log
# feedback in the next cell) is deterministic. Must be set BEFORE
# importing mlflow.
os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "false"

sys.path.insert(0, "../src")

# COMMAND ----------

dbutils.widgets.text(
    name="config-path",
    defaultValue="../config/examples/15_complete_applications/hardware_store.yaml",
)
config_path: str = dbutils.widgets.get("config-path")
print(f"Config path: {config_path}")

# COMMAND ----------

import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.types.responses import ResponsesAgentRequest

from dao_ai.config import AppConfig
from dao_ai.evaluation import log_user_feedback

mlflow.langchain.autolog(run_tracer_inline=True)

# COMMAND ----------

config: AppConfig = AppConfig.from_file(path=config_path)
print(f"App:       {config.app.name}")
print(f"Endpoint:  {config.app.endpoint_name}")
print(f"Agents:    {[a.name for a in config.app.agents]}")

w = WorkspaceClient()
USER = w.current_user.me().user_name
print(f"User: {USER}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. In-process invocation
# MAGIC
# MAGIC Build the agent locally from config, call `apredict`, read `trace_id`
# MAGIC from `custom_outputs`, log feedback.

# COMMAND ----------

agent = config.as_responses_agent()

req = ResponsesAgentRequest(
    input=[
        {"role": "user", "content": "Can you recommend a lamp for my oak side table?"}
    ],
    custom_inputs={
        "configurable": {"user_id": USER, "store_num": "87887"},
        "session": {},
    },
)

resp = asyncio.run(agent.apredict(req))
trace_id_in_process = resp.custom_outputs["trace_id"]
print("assistant:", resp.output[0].model_dump()["content"][0]["text"][:300])
print("trace_id:", trace_id_in_process)

log_user_feedback(
    trace_id=trace_id_in_process,
    value="up",
    comment="In-process apredict: multi-agent answer was good.",
    user_id=USER,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Deployed Model Serving endpoint
# MAGIC
# MAGIC POST to `/serving-endpoints/{endpoint}/invocations`. Same response
# MAGIC shape, same `custom_outputs["trace_id"]` contract.

# COMMAND ----------

endpoint_name: str = config.app.endpoint_name

ms_response: dict = w.api_client.do(
    "POST",
    f"/serving-endpoints/{endpoint_name}/invocations",
    body={
        "input": [{"role": "user", "content": "What aisle has 1/2 inch PVC elbows?"}],
        "custom_inputs": {
            "configurable": {"user_id": USER, "store_num": "87887"},
            "session": {},
        },
    },
    headers={"Content-Type": "application/json"},
)
trace_id_ms = ms_response["custom_outputs"]["trace_id"]
print("assistant:", ms_response["output"][0]["content"][0]["text"][:300])
print("trace_id:", trace_id_ms)

log_user_feedback(
    trace_id=trace_id_ms,
    value="up",
    comment="Model Serving endpoint: returned the right aisle.",
    user_id=USER,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Deployed Databricks App
# MAGIC
# MAGIC The App is also a `ResponsesAgent` server. Hit `{app_url}/invocations`
# MAGIC with a Databricks bearer token — same JSON shape, same contract.

# COMMAND ----------

import requests

# Resolve the deployed App URL by name
app_name: str = config.app.name.replace("_", "-")
app_info = w.apps.get(name=app_name)
app_url: str = app_info.url
print(f"App: {app_name} → {app_url}")

token: str = w.config.authenticate()["Authorization"].removeprefix("Bearer ").strip()
r = requests.post(
    f"{app_url}/invocations",
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    json={
        "input": [{"role": "user", "content": "Do you carry gluten-free pasta?"}],
        "custom_inputs": {
            "configurable": {"user_id": USER, "store_num": "87887"},
            "session": {},
        },
    },
    timeout=120,
)
r.raise_for_status()
app_response = r.json()
trace_id_app = app_response["custom_outputs"]["trace_id"]
print("assistant:", app_response["output"][0]["content"][0]["text"][:300])
print("trace_id:", trace_id_app)

log_user_feedback(
    trace_id=trace_id_app,
    value="down",
    comment="Databricks App: answer was off-topic for a hardware store.",
    user_id=USER,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Verify the trace is the OUTER multi-agent root
# MAGIC
# MAGIC Inspect the span tree under each trace_id. The root span should be of
# MAGIC type `AGENT`; children should include the supervisor, middleware,
# MAGIC and handoff into one or more sub-agents.

# COMMAND ----------

for label, tid in [
    ("in-process ", trace_id_in_process),
    ("model serv. ", trace_id_ms),
    ("app         ", trace_id_app),
]:
    trace = mlflow.get_trace(tid)
    spans = trace.search_spans()
    root_spans = [s for s in spans if s.parent_id is None]
    print(
        f"[{label}] trace_id={tid} | spans={len(spans)} | "
        f"root={root_spans[0].name!r} ({root_spans[0].span_type})"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Search & verify feedback rows

# COMMAND ----------

import pandas as pd

traces = mlflow.search_traces(max_results=100)
print(f"total traces: {len(traces)}")


def has_user_feedback(assessments) -> bool:
    return any(
        a.get("assessment_name") == "user_feedback" for a in (assessments or [])
    )


feedback_mask = traces["assessments"].apply(has_user_feedback)
print(f"traces with user_feedback assessments: {int(feedback_mask.sum())}")
traces[feedback_mask][["trace_id", "request_time"]].head()

# COMMAND ----------

# MAGIC %md
# MAGIC SQL view — flatten `assessments` and pull just the `user_feedback` rows.

# COMMAND ----------

spark.createDataFrame(traces).createOrReplaceTempView("dao_ai_traces")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   trace_id,
# MAGIC   a.feedback.value      AS feedback_value,
# MAGIC   a.rationale           AS comment,
# MAGIC   a.source.source_id    AS user_id
# MAGIC FROM dao_ai_traces
# MAGIC LATERAL VIEW EXPLODE(assessments) AS a
# MAGIC WHERE a.assessment_name = 'user_feedback'
# MAGIC ORDER BY request_time DESC
