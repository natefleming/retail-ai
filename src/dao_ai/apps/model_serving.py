# Apply nest_asyncio FIRST before any other imports
# This allows dao-ai's async/sync patterns to work in Model Serving
# where there may already be an event loop running (e.g., notebook context)
import os  # noqa: E402
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

mlflow.langchain.autolog(run_tracer_inline=True)
suppress_autolog_context_warnings()

model_config: ModelConfig = ModelConfig()
config: AppConfig = AppConfig(**model_config.to_dict())

log_level: str = config.app.log_level

configure_logging(level=log_level)

config.initialize()

# Configure UC-based trace destination if trace_location is set.
# Uses mlflow.set_experiment(trace_location=UnityCatalog(...)) — the post-3.11
# blessed API. Replaces the older
# mlflow.tracing.set_destination(UCSchemaLocation(...)) which emits a
# deprecation warning on every call.
if config.app and config.app.trace_location:
    from mlflow.entities import UnityCatalog  # noqa: E402

    _loc = config.app.trace_location
    _trace_loc_kwargs: dict[str, object] = {
        "catalog_name": _loc.catalog_name,
        "schema_name": _loc.schema_name,
    }
    _table_prefix = _loc.resolved_table_prefix
    if _table_prefix:
        _trace_loc_kwargs["table_prefix"] = _table_prefix
    _experiment_id_env: str | None = os.environ.get("MLFLOW_EXPERIMENT_ID")
    if _experiment_id_env:
        mlflow.set_experiment(
            experiment_id=_experiment_id_env,
            trace_location=UnityCatalog(**_trace_loc_kwargs),
        )

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
