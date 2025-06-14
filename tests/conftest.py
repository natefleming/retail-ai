import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import pytest
from dotenv import find_dotenv, load_dotenv
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from mlflow.models import ModelConfig
from mlflow.pyfunc import ChatModel

from retail_ai.config import AppConfig
from retail_ai.graph import create_retail_ai_graph
from retail_ai.models import create_agent

logger.remove()
logger.add(sys.stderr, level="INFO")


root_dir: Path = Path(__file__).parents[1]
src_dir: Path = root_dir / "src"
test_dir: Path = root_dir / "tests"
config_dir: Path = root_dir / "config"

sys.path.insert(0, str(test_dir.resolve()))
sys.path.insert(0, str(src_dir.resolve()))

_ = load_dotenv(find_dotenv())


def has_databricks_env() -> bool:
    required_vars: Sequence[str] = [
        "DATABRICKS_TOKEN",
        "DATABRICKS_HOST",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_REGISTRY_URI",
        "MLFLOW_EXPERIMENT_ID",
    ]
    return all(var in os.environ for var in required_vars)


@pytest.fixture
def development_config() -> Path:
    return config_dir / "model_config.yaml"


@pytest.fixture
def model_config(development_config: Path) -> ModelConfig:
    return ModelConfig(development_config=development_config)


@pytest.fixture
def config(model_config: ModelConfig) -> AppConfig:
    return AppConfig(**model_config.to_dict())


@pytest.fixture
def graph(config: AppConfig) -> CompiledStateGraph:
    graph: CompiledStateGraph = create_retail_ai_graph(config=config)
    return graph


@pytest.fixture
def chat_model(graph: CompiledStateGraph) -> ChatModel:
    app: ChatModel = create_agent(graph)
    return app


@pytest.fixture(scope="session")
def weather_server_mcp():
    """Start the weather MCP server for testing and clean up after tests."""
    # Path to the weather server script
    weather_server_path = test_dir / "weather_server_mcp.py"

    # Start the weather server subprocess
    process = subprocess.Popen(
        [sys.executable, str(weather_server_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Give the server a moment to start
    time.sleep(2)

    # Check if the process is still running (didn't crash immediately)
    if process.poll() is not None:
        stdout, stderr = process.communicate()
        raise RuntimeError(
            f"Weather server failed to start. stdout: {stdout}, stderr: {stderr}"
        )

    yield process

    # Cleanup: terminate the process
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
