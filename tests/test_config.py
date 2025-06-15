import sys

import pytest
import yaml
from mlflow.models import ModelConfig

from retail_ai.config import AppConfig


@pytest.mark.unit
def test_app_config(model_config: ModelConfig) -> None:
    app_config = AppConfig(**model_config.to_dict())
    print(app_config.model_dump_json(indent=2), file=sys.stderr)
    assert app_config is not None


@pytest.mark.unit
def test_app_config_should_serialize(config: AppConfig) -> None:
    yaml.safe_dump(config.model_dump())
    assert True


@pytest.mark.unit
def test_app_config_tools_should_be_correct_type(
    model_config: ModelConfig, config: AppConfig
) -> None:
    for tool_name, tool in config.tools.items():
        assert tool_name in model_config.get("tools"), (
            f"Tool {tool_name} not found in model_config"
        )
        expected_type = None
        for _, expected_tool in model_config.get("tools").items():
            if expected_tool["name"] == tool.name:
                expected_type = expected_tool["function"]["type"]
                break
        assert expected_type is not None, (
            f"Expected type for tool '{tool_name}' not found in model_config"
        )
        actual_type = tool.function.type
        assert actual_type == expected_type, (
            f"Function type mismatch for tool '{tool_name}': "
            f"expected '{expected_type}', got '{actual_type}'"
        )
