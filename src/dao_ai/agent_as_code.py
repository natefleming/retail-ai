import mlflow
from mlflow.models import ModelConfig
from mlflow.pyfunc import ResponsesAgent

from dao_ai.config import AppConfig
from dao_ai.logging import configure_logging

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

mlflow.langchain.autolog()

model_config: ModelConfig = ModelConfig()
config: AppConfig = AppConfig(**model_config.to_dict())

log_level: str = config.app.log_level

configure_logging(level=log_level)

app: ResponsesAgent = config.as_responses_agent()

mlflow.models.set_model(app)
