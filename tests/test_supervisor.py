import sys
from typing import Any, Sequence

import pytest
from loguru import logger
from mlflow.models import ModelConfig

from retail_ai.supervisor import Supervisor

logger.remove()
logger.add(sys.stderr, level="INFO")


@pytest.fixture
def supervisor(model_config: ModelConfig) -> Supervisor:
    supervisor = Supervisor()
    agents: dict[str, Any] = model_config.get("app").get("agents")
    for agent in agents:
        name: str = agent["name"]
        supervisor.register(name, agent)
    return supervisor


def test_supervisor_initialization() -> None:
    """
    Test that Supervisor initializes with an empty agents dictionary.
    """
    supervisor = Supervisor()
    assert supervisor.agents == {}


def test_supervisor_register_agent(model_config: ModelConfig) -> None:
    """
    Test that Supervisor can register an agent and update the agents dictionary.
    """
    supervisor = Supervisor()
    agents: dict[str, Any] = model_config.get("app").get("agents")
    for agent in agents:
        name = agent["name"]
        supervisor.register(name, agent)

    assert len(supervisor.agents) == len(agents)


def test_supervisor_default_agent(
    supervisor: Supervisor, model_config: ModelConfig
) -> None:
    agents = model_config.get("app").get("agents")
    allowed_routes: Sequence[str] = sorted([agent["name"] for agent in agents])
    assert supervisor.allowed_routes == allowed_routes


def test_supervisor_prompt(supervisor: Supervisor) -> None:
    prompt: str = supervisor.prompt
    logger.info(prompt)
    assert prompt is not None
