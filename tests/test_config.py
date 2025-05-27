from retail_ai.config import AppConfig, EntitlementEnum, AppPermission, App, EvaluationTable, Evaluation, DatasetFormat, Dataset, Resources

import pytest
import json

from mlflow.models import ModelConfig
import sys

def test_app_config(model_config: ModelConfig):
    app_config = AppConfig(**model_config.to_dict())
    print(app_config.model_dump_json(indent=2), file=sys.stderr)
    assert(app_config is not None)

