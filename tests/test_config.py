import sys

from mlflow.models import ModelConfig

from retail_ai.config import AppConfig


def test_app_config(model_config: ModelConfig):
    app_config = AppConfig(**model_config.to_dict())
    print(app_config.model_dump_json(indent=2), file=sys.stderr)
    assert app_config is not None
