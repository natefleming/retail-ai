import sys
from pathlib import Path

import pytest
from mlflow.models import ModelConfig

from retail_ai.config import AppConfig

root_dir: Path = Path(__file__).parents[1]
src_dir: Path = root_dir / "retail_ai"
test_dir: Path = root_dir / "tests"

sys.path.insert(0, str(test_dir.resolve()))
sys.path.insert(0, str(root_dir.resolve()))


@pytest.fixture
def development_config() -> Path:
    return root_dir / "model_config.yaml"


@pytest.fixture
def model_config(development_config: Path) -> ModelConfig:
    """
    Fixture to provide a sample ModelConfig for testing.
    """
    return ModelConfig(development_config=development_config)


@pytest.fixture
def config(model_config: ModelConfig) -> AppConfig:
    return AppConfig(**model_config.to_dict())
