import pytest
from retail_ai.tools import create_tools
from mlflow.models import ModelConfig


def test_create_tools(model_config: ModelConfig) -> None:

    tools_config = model_config.get("tools")

    tools = create_tools(tools_config)

    assert tools is not None