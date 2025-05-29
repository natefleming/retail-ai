import sys
from typing import Any, Sequence

import pytest
from loguru import logger

from retail_ai.config import AgentModel, AppConfig
from retail_ai.orchestration import Supervisor

logger.remove()
logger.add(sys.stderr, level="INFO")


@pytest.fixture
def supervisor(config: AppConfig) -> Supervisor:
    agents: dict[str, Any] = config.app.agents
    supervisor = Supervisor(agents=agents)
    return supervisor


def test_supervisor_initialization() -> None:
    """
    Test that Supervisor initializes with an empty agents dictionary.
    """
    supervisor = Supervisor()
    assert supervisor.agent_registry == {}


def test_supervisor_register_agent(config: AppConfig) -> None:
    """
    Test that Supervisor can register an agent and update the agents dictionary.
    """

    supervisor = Supervisor()
    agents: Sequence[AgentModel] = config.app.agents
    for agent in agents:
        supervisor.register(agent)

    assert len(supervisor.agent_registry) == len(agents)


def test_supervisor_allowed_routes(supervisor: Supervisor, config: AppConfig) -> None:
    agents = config.app.agents
    allowed_routes: Sequence[str] = sorted(agent.name for agent in agents)
    assert supervisor.allowed_routes == allowed_routes


def test_supervisor_prompt(supervisor: Supervisor) -> None:
    prompt: str = supervisor.prompt
    logger.info(prompt)
    assert prompt is not None
