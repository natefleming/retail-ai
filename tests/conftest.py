import subprocess
import sys
import time
from pathlib import Path

import pytest
from mlflow.models import ModelConfig

from retail_ai.config import AppConfig

root_dir: Path = Path(__file__).parents[1]
src_dir: Path = root_dir / "retail_ai"
test_dir: Path = root_dir / "tests"
config_dir: Path = root_dir / "config"

sys.path.insert(0, str(test_dir.resolve()))
sys.path.insert(0, str(root_dir.resolve()))


@pytest.fixture
def development_config() -> Path:
    return config_dir / "model_config.yaml"


@pytest.fixture
def model_config(development_config: Path) -> ModelConfig:
    return ModelConfig(development_config=development_config)


@pytest.fixture
def config(model_config: ModelConfig) -> AppConfig:
    return AppConfig(**model_config.to_dict())


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
