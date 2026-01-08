# Apply nest_asyncio FIRST before any other imports
# This allows dao-ai's async/sync patterns to work in Model Serving
# where there may already be an event loop running (e.g., notebook context)
import nest_asyncio

nest_asyncio.apply()

import mlflow  # noqa: E402
from mlflow.models import ModelConfig  # noqa: E402
from mlflow.pyfunc import ResponsesAgent  # noqa: E402

from dao_ai.config import AppConfig  # noqa: E402
from dao_ai.logging import configure_logging  # noqa: E402

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

mlflow.langchain.autolog()

model_config: ModelConfig = ModelConfig()
config: AppConfig = AppConfig(**model_config.to_dict())

log_level: str = config.app.log_level

configure_logging(level=log_level)

app: ResponsesAgent = config.as_responses_agent()

mlflow.models.set_model(app)
