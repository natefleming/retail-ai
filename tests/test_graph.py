import sys

from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from mlflow.models import ModelConfig

from retail_ai.graph import create_retail_ai_graph

logger.remove()
logger.add(sys.stderr, level="INFO")


def test_create_retail_ai_graph(model_config: ModelConfig) -> None:
    """
    Test the creation of the retail AI graph with a valid model configuration.
    """
    # Ensure the model_config has the required structure

    # Create the graph
    graph: CompiledStateGraph = create_retail_ai_graph(model_config)

    assert graph is not None
    assert isinstance(graph, CompiledStateGraph)
