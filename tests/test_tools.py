from mlflow.models import ModelConfig

from retail_ai.tools import create_tools


def test_create_tools(model_config: ModelConfig) -> None:
    tools_config = model_config.get("tools")
    tools_config = {
        k: v
        for k, v in tools_config.items()
        if "_uc_" not in k and "vector_search_tool" not in k
    }

    tools = create_tools(tools_config)

    assert tools is not None
