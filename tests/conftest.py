import os

# Disable MLflow tracing BEFORE importing mlflow to prevent the async trace logging
# queue from being initialized. This avoids the slow "Flushing the async trace logging
# queue" message at the end of test runs.
os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "false"
os.environ["MLFLOW_TRACE_SAMPLING_RATIO"] = "0"

import sys
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import MagicMock

import mlflow
import pytest
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from mlflow.models import ModelConfig
from mlflow.pyfunc import ChatModel

from dao_ai.config import AppConfig, LLMModel
from dao_ai.graph import create_dao_ai_graph
from dao_ai.logging import configure_logging
from dao_ai.models import create_agent

configure_logging(level="INFO")

# Also disable tracing programmatically as a backup
mlflow.tracing.disable()


root_dir: Path = Path(__file__).parents[1]
src_dir: Path = root_dir / "src"
test_dir: Path = root_dir / "tests"
data_dir: Path = test_dir / "data"
config_dir: Path = test_dir / "config"

sys.path.insert(0, str(test_dir.resolve()))
sys.path.insert(0, str(src_dir.resolve()))

env_path: str = find_dotenv()
logger.info(f"Loading environment variables from: {env_path}")
_ = load_dotenv(env_path)


def pytest_configure(config):
    """Configure custom pytest markers."""
    markers: Sequence[str] = [
        "unit: mark test as a unit test (fast, isolated, no external dependencies)",
        "system: mark test as a system test (slower, may use external resources)",
        "integration: mark test as integration test (tests component interactions)",
        "slow: mark test as slow running (> 1 second)",
    ]
    for marker in markers:
        config.addinivalue_line("markers", marker)


def has_databricks_env() -> bool:
    required_vars: Sequence[str] = [
        "DATABRICKS_TOKEN",
        "DATABRICKS_HOST",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_REGISTRY_URI",
        "MLFLOW_EXPERIMENT_ID",
    ]
    return all(var in os.environ for var in required_vars)


def has_postgres_env() -> bool:
    required_vars: Sequence[str] = [
        "PG_HOST",
        "PG_PORT",
        "PG_USER",
        "PG_PASSWORD",
        "PG_DATABASE",
    ]
    return "PG_CONNECTION_STRING" in os.environ or all(
        var in os.environ for var in required_vars
    )


def has_retail_ai_env() -> bool:
    required_vars: Sequence[str] = [
        "RETAIL_AI_DATABRICKS_HOST",
        "RETAIL_AI_DATABRICKS_CLIENT_ID",
        "RETAIL_AI_DATABRICKS_CLIENT_SECRET",
        "RETAIL_AI_GENIE_SPACE_ID",
    ]
    return all(var in os.environ for var in required_vars)


@pytest.fixture
def development_config() -> Path:
    return config_dir / "test_model_config.yaml"


@pytest.fixture
def data_path() -> Path:
    return data_dir


@pytest.fixture
def model_config(development_config: Path) -> ModelConfig:
    return ModelConfig(development_config=development_config)


@pytest.fixture
def config(model_config: ModelConfig) -> AppConfig:
    return AppConfig(**model_config.to_dict())


@pytest.fixture
def graph(config: AppConfig) -> CompiledStateGraph:
    graph: CompiledStateGraph = create_dao_ai_graph(config=config)
    return graph


@pytest.fixture
def chat_model(graph: CompiledStateGraph) -> ChatModel:
    app: ChatModel = create_agent(graph)
    return app


def add_databricks_resource_attrs(mock: Any) -> Any:
    """
    Add IsDatabricksResource attributes to a mock object.

    Since IsDatabricksResource is now a BaseModel with validators,
    mocks with spec=LLMModel (or similar) need these attributes set.

    Args:
        mock: The MagicMock object to configure

    Returns:
        The same mock with IsDatabricksResource attributes added
    """
    # Set all IsDatabricksResource attributes to None/False
    mock.on_behalf_of_user = False
    mock.service_principal = None
    mock.client_id = None
    mock.client_secret = None
    mock.workspace_host = None
    mock.pat = None
    return mock


def create_mock_databricks_resource(spec_class: type) -> MagicMock:
    """
    Create a MagicMock for any IsDatabricksResource subclass.

    Args:
        spec_class: The class to use as spec (e.g., LLMModel, WarehouseModel)

    Returns:
        A MagicMock with IsDatabricksResource attributes properly set
    """
    mock = MagicMock(spec=spec_class)
    add_databricks_resource_attrs(mock)
    return mock


def create_mock_llm_model() -> MagicMock:
    """
    Create a properly configured mock for LLMModel.

    Returns a MagicMock with:
    - spec=LLMModel for attribute checking
    - IsDatabricksResource attributes properly set
    - as_chat_model() returning a mock that can invoke
    - name attribute for logging
    """
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="This is a test summary.")

    mock_llm_model = MagicMock(spec=LLMModel)
    mock_llm_model.as_chat_model.return_value = mock_llm
    mock_llm_model.name = "test-model"

    # Add IsDatabricksResource attributes
    add_databricks_resource_attrs(mock_llm_model)

    return mock_llm_model


@pytest.fixture
def mock_llm_model() -> MagicMock:
    """
    Shared fixture for a mock LLMModel.

    Use this fixture instead of creating your own mock_llm_model to ensure
    all IsDatabricksResource attributes are properly set.
    """
    return create_mock_llm_model()
