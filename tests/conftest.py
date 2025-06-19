import os
import sys
from pathlib import Path
from typing import Sequence

import pytest
from dotenv import find_dotenv, load_dotenv
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from mlflow.models import ModelConfig
from mlflow.pyfunc import ChatModel

from dao_ai.config import AppConfig
from dao_ai.graph import create_dao_ai_graph
from dao_ai.models import create_agent

logger.remove()
logger.add(sys.stderr, level="INFO")


root_dir: Path = Path(__file__).parents[1]
src_dir: Path = root_dir / "src"
test_dir: Path = root_dir / "tests"
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


@pytest.fixture
def development_config() -> Path:
    return config_dir / "test_model_config.yaml"


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
