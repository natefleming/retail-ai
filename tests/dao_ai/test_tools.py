from typing import Sequence

import pytest

from dao_ai.config import AppConfig, FunctionType, ToolModel
from dao_ai.tools import create_tools

excluded_tools: Sequence[str] = [
    "vector_search",
    "genie",
    "find_product_details_by_description",
]


@pytest.mark.unit
def test_create_tools(config: AppConfig) -> None:
    tool_models: list[ToolModel] = config.find_tools(
        lambda tool: not any(excluded in tool.name for excluded in excluded_tools)
        and tool.function.type != FunctionType.UNITY_CATALOG
    )

    tools = create_tools(tool_models)

    assert tools is not None
