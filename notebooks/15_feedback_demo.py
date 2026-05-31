# Databricks notebook source
# MAGIC %md
# MAGIC # dao-ai Multi-Agent Feedback Demo
# MAGIC
# MAGIC This notebook shows the **correct** pattern for capturing thumbs-up / thumbs-down
# MAGIC feedback against a dao-ai multi-agent response — and the anti-patterns that go
# MAGIC wrong under concurrency or after the agent function returns.
# MAGIC
# MAGIC **Golden rule:** read `trace_id` out of `response.custom_outputs["trace_id"]`.
# MAGIC Never read it from MLflow global state in the caller.
# MAGIC
# MAGIC Why this notebook exists:
# MAGIC - dao-ai's `LanggraphResponsesAgent` runs a multi-agent graph (supervisor / swarm).
# MAGIC   The trace_id we want is the **outer** root trace that spans every sub-agent hop,
# MAGIC   not any single agent leg.
# MAGIC - `apredict()` / `apredict_stream()` now both attach that outer trace_id to
# MAGIC   `custom_outputs` so the caller never needs to read MLflow global state.

# COMMAND ----------

# MAGIC %pip install -q --upgrade -r ../requirements.txt nest-asyncio>=1.6.0
# MAGIC %restart_python

# COMMAND ----------

import asyncio
import os
import sys

import nest_asyncio
nest_asyncio.apply()

# Force synchronous trace export so log_feedback never races the async exporter.
# `setdefault` won't override a job-context "true" so use direct assignment.
# Set BEFORE importing mlflow / dao_ai.
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

mlflow.langchain.autolog()

# COMMAND ----------

config: AppConfig = AppConfig.from_file(path=config_path)
print(f"App: {config.app.name}")
print(f"Endpoint: {config.app.endpoint_name}")
print(f"Agents: {[a.name for a in config.app.agents]}")

# Resolve the experiment the deployed agent writes traces to
EXPERIMENT_PATH = (
    config.app.registered_model.experiment_name
    if hasattr(config.app.registered_model, "experiment_name")
    and config.app.registered_model.experiment_name
    else f"/Shared/{config.app.name}"
)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_PATH) or mlflow.set_experiment(
    EXPERIMENT_PATH
)
EXPERIMENT_ID = experiment.experiment_id
mlflow.set_experiment(experiment_id=EXPERIMENT_ID)
print(f"Experiment: {EXPERIMENT_PATH} → id={EXPERIMENT_ID}")

USER = WorkspaceClient().current_user.me().user_name
print(f"User: {USER}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Happy path — in-process multi-agent invocation
# MAGIC
# MAGIC 1. Build the graph + ResponsesAgent from config.
# MAGIC 2. `await agent.apredict(request)` — `mlflow.langchain.autolog()` opens the root trace.
# MAGIC 3. Read `trace_id` from `response.custom_outputs["trace_id"]`.
# MAGIC 4. `log_user_feedback(trace_id=..., value="up"/"down", ...)`.

# COMMAND ----------

agent = config.as_responses_agent()

req_positive = ResponsesAgentRequest(
    input=[
        {"role": "user", "content": "Can you recommend a lamp for my oak side table?"}
    ],
    custom_inputs={
        "configurable": {
            "user_id": USER,
            "store_num": "87887",
        },
        "session": {},
    },
)

resp_positive = asyncio.run(agent.apredict(req_positive))
trace_id_positive = resp_positive.custom_outputs["trace_id"]
print("assistant:", resp_positive.output[0].model_dump()["content"][0]["text"][:300])
print("trace_id:", trace_id_positive)

# Flush async trace logging so log_feedback sees the trace.
mlflow.flush_trace_async_logging()
log_user_feedback(
    trace_id=trace_id_positive,
    value="up",
    comment="Multi-agent answer was relevant and concise.",
    user_id=USER,
)
print("Logged positive feedback ✓")

# COMMAND ----------

req_negative = ResponsesAgentRequest(
    input=[
        {
            "role": "user",
            "content": "Do you carry left-handed metric crescent wrenches for stainless?",
        }
    ],
    custom_inputs={
        "configurable": {
            "user_id": USER,
            "store_num": "87887",
        },
        "session": {},
    },
)

resp_negative = asyncio.run(agent.apredict(req_negative))
trace_id_negative = resp_negative.custom_outputs["trace_id"]
print("assistant:", resp_negative.output[0].model_dump()["content"][0]["text"][:300])
print("trace_id:", trace_id_negative)

mlflow.flush_trace_async_logging()
log_user_feedback(
    trace_id=trace_id_negative,
    value="down",
    comment="Wrong — the response missed that the part doesn't exist.",
    user_id=USER,
)
print("Logged negative feedback ✓")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Deployed endpoint variant — query Model Serving
# MAGIC
# MAGIC Once `databricks.agents.deploy(...)` has shipped the agent, you can hit the served
# MAGIC endpoint directly. The deployed endpoint returns the same `custom_outputs.trace_id`
# MAGIC shape, so the feedback-logging pattern is identical.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
endpoint_name = config.app.endpoint_name

# Note: w.serving_endpoints.query() doesn't pass a ResponsesAgent body shape
# cleanly — it wraps under "inputs" and the endpoint rejects with
# "Invalid input. One of 'instances' and 'inputs' must be specified".
# Use the raw POST to /serving-endpoints/{name}/invocations instead.
served_dict = w.api_client.do(
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
served_trace_id = (served_dict.get("custom_outputs") or {}).get("trace_id")
print("Served endpoint trace_id:", served_trace_id)

if served_trace_id:
    mlflow.flush_trace_async_logging()
    log_user_feedback(
        trace_id=served_trace_id,
        value="up",
        comment="Served endpoint round-trip worked.",
        user_id=USER,
    )
    print("Logged feedback against deployed-endpoint trace ✓")
else:
    print(
        "No trace_id in custom_outputs — either the endpoint is running an older "
        "dao-ai build or autolog isn't enabled in the served model."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Multi-agent verification
# MAGIC
# MAGIC The supervisor pattern routes one user turn through up to N sub-agents.
# MAGIC `mlflow.langchain.autolog()` opens a single root trace per `apredict()` call;
# MAGIC sub-agent invocations become child spans under that root. The trace_id in
# MAGIC `custom_outputs` IS that root trace — so feedback attaches at the correct level.

# COMMAND ----------

trace = mlflow.get_trace(trace_id_positive)
spans = trace.search_spans()
print(f"Total spans in trace: {len(spans)}")
print()
print("Top-level span (root of the multi-agent turn):")
root = next((s for s in spans if s.parent_id is None), None)
if root:
    print(f"  name={root.name!r} span_type={root.span_type}")
print()
print("Sub-agent / tool spans nested under the root (first 15):")
for s in spans[:15]:
    indent = "  " if s.parent_id else ""
    print(f"  {indent}{s.span_type:>15} | {s.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Anti-patterns — DO NOT COPY

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4a. Reading `get_last_active_trace_id()` from the caller
# MAGIC
# MAGIC Races with concurrent invocations and with autolog flushes from sibling calls.
# MAGIC "Last" is not "this".

# COMMAND ----------

req_a = ResponsesAgentRequest(
    input=[{"role": "user", "content": "What time does the store open?"}],
    custom_inputs={
        "configurable": {"user_id": USER, "store_num": "87887"},
        "session": {},
    },
)
resp_a = asyncio.run(agent.apredict(req_a))

# WRONG: read from MLflow global state instead of the response
wrong_trace_id = mlflow.get_last_active_trace_id()
right_trace_id = resp_a.custom_outputs["trace_id"]
print(" wrong  (global state):", wrong_trace_id)
print(" right  (custom_outputs):", right_trace_id)
print(
    " — In a single-threaded notebook these may coincide.\n"
    " — Under concurrent invocations they diverge and feedback attaches to the wrong trace."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4b. Reading `get_current_active_span()` from the caller
# MAGIC
# MAGIC The span opened inside `apredict()` is closed once the function returns.
# MAGIC There is no active span in the caller's stack.

# COMMAND ----------

# `resp_a` was already produced above. After apredict() returns, no span is active here.
active = mlflow.get_current_active_span()
print("active span in caller:", active)  # → None

try:
    bad = active.trace_id  # noqa: F841 — demonstrates AttributeError on None
except AttributeError as e:
    print("AttributeError as expected:", e)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4c. Legacy `log_assessment` / `Assessment(...)` shape
# MAGIC
# MAGIC MLflow 2.x preview API. MLflow 3 split this into `log_feedback` (human / LLM-judge
# MAGIC assessments) and `log_expectation` (ground truth) with typed kwargs.

# COMMAND ----------

try:
    from mlflow.entities import Assessment  # may not exist or may have different signature

    legacy = Assessment(  # type: ignore[call-arg]
        name="user_feedback",
        value=True,
        rationale="legacy shape",
    )
    mlflow.log_assessment(trace_id=right_trace_id, assessment=legacy)  # type: ignore[attr-defined]
except (ImportError, AttributeError, TypeError) as e:
    print("Legacy shape failed as expected:", type(e).__name__, e)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Search & verify
# MAGIC
# MAGIC Confirm the happy-path traces show up with `user_feedback` assessments attached.

# COMMAND ----------

import pandas as pd

all_traces = mlflow.search_traces(locations=[EXPERIMENT_ID], max_results=50)
print(f"Total traces returned: {len(all_traces)}")
all_traces[["trace_id", "request_time"]].head(10)

# COMMAND ----------

def has_positive_feedback(assessments) -> bool:
    return any(
        a.get("assessment_name") == "user_feedback"
        and a.get("feedback", {}).get("value") is True
        for a in (assessments or [])
    )


def has_negative_feedback(assessments) -> bool:
    return any(
        a.get("assessment_name") == "user_feedback"
        and a.get("feedback", {}).get("value") is False
        for a in (assessments or [])
    )


positive_mask = all_traces["assessments"].apply(has_positive_feedback)
negative_mask = all_traces["assessments"].apply(has_negative_feedback)
print(f"Positive feedback traces: {int(positive_mask.sum())}")
print(f"Negative feedback traces: {int(negative_mask.sum())}")
all_traces[positive_mask | negative_mask][["trace_id"]].head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. SQL — query traces with feedback
# MAGIC
# MAGIC Register the `search_traces` DataFrame as a Spark temp view and query it via SQL.
# MAGIC `request_time` is BIGINT epoch-ms — wrap with `timestamp_millis(...)` before truncating.

# COMMAND ----------

spark.createDataFrame(all_traces).createOrReplaceTempView("dao_ai_traces")
spark.sql("SELECT COUNT(*) AS trace_count FROM dao_ai_traces").show()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   trace_id,
# MAGIC   request_time,
# MAGIC   a.assessment_name,
# MAGIC   a.feedback.value      AS feedback_value,
# MAGIC   a.rationale            AS comment,
# MAGIC   a.source.source_id     AS user_id
# MAGIC FROM dao_ai_traces
# MAGIC LATERAL VIEW EXPLODE(assessments) AS a
# MAGIC WHERE a.assessment_name = 'user_feedback'
# MAGIC ORDER BY request_time DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Daily up/down volume (operator dashboard)
# MAGIC SELECT
# MAGIC   DATE(timestamp_millis(request_time))                        AS day,
# MAGIC   SUM(CASE WHEN a.feedback.value = TRUE  THEN 1 ELSE 0 END)   AS thumbs_up,
# MAGIC   SUM(CASE WHEN a.feedback.value = FALSE THEN 1 ELSE 0 END)   AS thumbs_down,
# MAGIC   COUNT(*)                                                    AS feedback_events
# MAGIC FROM dao_ai_traces
# MAGIC LATERAL VIEW EXPLODE(assessments) AS a
# MAGIC WHERE a.assessment_name = 'user_feedback'
# MAGIC GROUP BY DATE(timestamp_millis(request_time))
# MAGIC ORDER BY day DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Section | Pattern | Outcome |
# MAGIC |---|---|---|
# MAGIC | 1 (happy path) | `trace_id` from `custom_outputs`, in-process multi-agent | feedback attaches to outer trace |
# MAGIC | 2 (deployed) | same pattern via Model Serving endpoint | identical contract on the wire |
# MAGIC | 3 (verification) | inspect span tree under the trace_id | confirms it's the OUTER root, not a sub-agent leg |
# MAGIC | 4a | `get_last_active_trace_id()` in caller | ⚠️ fragile under concurrency |
# MAGIC | 4b | `get_current_active_span()` in caller | ❌ `AttributeError` on `None` |
# MAGIC | 4c | deprecated `Assessment` / `log_assessment` | ❌ ImportError / AttributeError |
# MAGIC
# MAGIC The Databricks App + the Node.js feedback route in
# MAGIC `e2e-chatbot-app-next/server/src/routes/feedback.ts` both write to the same
# MAGIC MLflow trace store the SQL queries above read from — one source of truth.
