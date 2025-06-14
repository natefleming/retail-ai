import sys

import pytest
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from mlflow.models import ModelConfig

from retail_ai.config import AppConfig
from retail_ai.graph import create_retail_ai_graph

logger.remove()
logger.add(sys.stderr, level="INFO")


@pytest.mark.key("unit")
def test_create_retail_ai_graph(model_config: ModelConfig) -> None:
    """
    Test the creation of the retail AI graph with a valid model configuration.
    """
    # Ensure the model_config has the required structure

    # Create the graph
    config: AppConfig = AppConfig(**model_config.to_dict())
    graph: CompiledStateGraph = create_retail_ai_graph(config=config)

    assert graph is not None
    assert isinstance(graph, CompiledStateGraph)
