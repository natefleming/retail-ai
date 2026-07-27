# Apply nest_asyncio FIRST before any other imports
# This allows dao-ai's async/sync patterns to work in Model Serving
# where there may already be an event loop running (e.g., notebook context)
import time  # noqa: E402

import nest_asyncio

nest_asyncio.apply()

_t_start = time.monotonic()

import mlflow  # noqa: E402
from mlflow.models import ModelConfig  # noqa: E402
from mlflow.pyfunc import ResponsesAgent  # noqa: E402

from dao_ai.config import AppConfig  # noqa: E402
from dao_ai.logging import (  # noqa: E402
    configure_logging,
    suppress_autolog_context_warnings,
)

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

model_config: ModelConfig = ModelConfig()
config: AppConfig = AppConfig(**model_config.to_dict())

log_level: str = config.app.log_level

configure_logging(level=log_level)

config.initialize()

# In-container experiment / trace-destination binding for Model Serving.
#
# Historically this file called ``mlflow.set_experiment(experiment_id=…)``
# followed by ``link_experiment_trace_location(...)``. Both hit
# ``GET /api/2.0/mlflow/experiments/get`` under the hood, which the
# serverless Model Serving container's ambient auth cannot satisfy without
# the served model having been logged with the experiment declared as a
# resource dependency (see mlflow ``databricks_utils.py`` /
# ``DatabricksModelServingConfigProvider``). Result was model-load crash:
#
#   mlflow.exceptions.RestException: INTERNAL_ERROR ... < 400 Bad Request
#   < Unable to load OAuth Config
#
# The correct pattern (per Databricks MLflow eng — Aravind Segu in
# #mlflow-3, Serena Ruan on es-1538671-swiggy, Shaylan Dias / Sid
# Murching in #agents) is to make no in-container experiment binding
# call at all. Both experiment placement and UC trace_location routing
# are driven by env vars that ``agents.deploy()`` sets on the endpoint:
#
#   * ``MLFLOW_EXPERIMENT_ID`` — active experiment for autolog runs.
#   * ``MLFLOW_TRACING_SQL_WAREHOUSE_ID`` — warehouse used for the
#     UC-table export path.
#
# ``MLFLOW_TRACING_DESTINATION`` is deliberately NOT set — it would parse
# as a legacy ``UCSchemaLocation`` with hardcoded default table name
# ``mlflow_experiment_trace_otel_spans`` and shadow the experiment-linked
# UnityCatalog resolved via ``_resolve_experiment_uc_location``.
#
# MLflow's ``_get_span_processors`` reads the remaining env vars and picks
# the right processor + exporter (``DatabricksUCTableSpanExporter`` when a
# UC destination is present, ``MlflowV3SpanExporter`` otherwise).
#
# An earlier iteration of this file wrapped ``set_experiment`` in a
# try/except, then later swapped to ``set_destination(MlflowExperiment
# Location(...))``. Both moved the crash aside but the ``set_destination``
# path forced ``MlflowV3SpanProcessor`` (control-plane) and silently
# bypassed the UC exporter path — verified empirically: with
# ``trace_location`` configured, the OTEL span table was 0 rows while
# ``mlflow.get_trace(...)`` returned the trace from the experiment's
# default store. Removing the call restores UC routing.
#
# The Apps entrypoint (``handlers.py``) keeps the original
# ``set_experiment`` + ``link_experiment_trace_location`` pattern —
# Apps containers have full ambient OAuth so those REST calls succeed,
# and Apps's tracing writer path relies on the explicit linkage.
# The two entrypoints legitimately diverge here.

mlflow.langchain.autolog(run_tracer_inline=True)
suppress_autolog_context_warnings()

from loguru import logger  # noqa: E402

logger.info(
    "Config loaded, creating ResponsesAgent",
    elapsed_ms=round((time.monotonic() - _t_start) * 1000),
)

_t_agent = time.monotonic()
app: ResponsesAgent = config.as_responses_agent()
logger.info(
    "ResponsesAgent created",
    agent_elapsed_ms=round((time.monotonic() - _t_agent) * 1000),
    total_elapsed_ms=round((time.monotonic() - _t_start) * 1000),
)

mlflow.models.set_model(app)
logger.info(
    "Model registered with MLflow via set_model - READY",
    total_elapsed_ms=round((time.monotonic() - _t_start) * 1000),
)
