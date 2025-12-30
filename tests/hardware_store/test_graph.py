import pytest
from conftest import has_databricks_env
from langgraph.graph.state import CompiledStateGraph
from mlflow.models import ModelConfig

from dao_ai.config import AppConfig
from dao_ai.graph import create_dao_ai_graph
from dao_ai.logging import configure_logging

configure_logging(level="INFO")


@pytest.mark.unit
@pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
def test_create_dao_ai_graph(model_config: ModelConfig) -> None:
    """
    Test the creation of the retail AI graph with a valid model configuration.
    """
    # Ensure the model_config has the required structure

    # Create the graph
    config: AppConfig = AppConfig(**model_config.to_dict())
    graph: CompiledStateGraph = create_dao_ai_graph(config=config)

    assert graph is not None
    assert isinstance(graph, CompiledStateGraph)
