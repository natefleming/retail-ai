from retail_ai.config import AppConfig, ToolModel
from retail_ai.tools import create_tools


def test_create_tools(config: AppConfig) -> None:
    tools_models: dict[str, ToolModel] = config.tools
    tools_config = [
        v
        for k, v in tools_models.items()
        if "_uc_" not in k and "vector_search" not in k
    ]
    tools = create_tools(tools_config)

    assert tools is not None
