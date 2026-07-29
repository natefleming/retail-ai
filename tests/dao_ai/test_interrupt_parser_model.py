"""Resolution-order tests for ``AppConfig.interrupt_parser_model`` (F8).

The HITL interrupt parser must use a model derived from the deployment's
config — explicit override, else supervisor, else swarm default agent, else
first agent — and return None (GA fallback) only when no agents are declared.
Each resolved path is logged with its source.

These exercise the resolver in isolation via a lightweight stand-in for
AppConfig (real AppConfig instantiation drags in many required fields); the
resolver only touches ``self.orchestration`` and ``self.agents``. Pydantic
forbids instance attribute injection, so ``as_chat_model`` is patched at the
class level to echo the model's endpoint name — letting each test assert
*which* model was selected.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dao_ai.config import (
    AgentModel,
    AppConfig,
    LLMModel,
    OrchestrationModel,
    SupervisorModel,
    SwarmModel,
)


def _agent(name: str, endpoint: str) -> AgentModel:
    return AgentModel(name=name, model=LLMModel(name=endpoint), prompt="p")


def _resolve(orchestration, agents) -> object:
    """Call the unbound resolver against a stand-in exposing just the two
    attrs it reads. ``as_chat_model`` is patched to return the endpoint name
    so the caller can assert which model won."""
    stub = MagicMock()
    stub.app.orchestration = orchestration
    stub.agents = {a.name: a for a in agents}
    # AgentModel has no as_chat_model of its own — the model is reached via
    # ``agent.model.as_chat_model()`` (an InferenceEndpointModel/LLMModel).
    with patch.object(LLMModel, "as_chat_model", lambda self: f"chat:{self.name}"):
        return AppConfig.interrupt_parser_model(stub)


@pytest.mark.unit
class TestInterruptParserModelResolution:
    def test_explicit_override_wins(self) -> None:
        orch = OrchestrationModel(
            supervisor=SupervisorModel(model=LLMModel(name="router")),
            interrupt_model=LLMModel(name="pinned-parser"),
        )
        assert _resolve(orch, [_agent("a", "agent-model")]) == "chat:pinned-parser"

    def test_supervisor_model_used_when_no_override(self) -> None:
        orch = OrchestrationModel(
            supervisor=SupervisorModel(model=LLMModel(name="router-model"))
        )
        assert _resolve(orch, [_agent("a", "agent-model")]) == "chat:router-model"

    def test_swarm_default_agent_model(self) -> None:
        first = _agent("first", "first-model")
        chosen = _agent("chosen", "chosen-model")
        orch = OrchestrationModel(swarm=SwarmModel(default_agent=chosen))
        assert _resolve(orch, [first, chosen]) == "chat:chosen-model"

    def test_first_agent_when_swarm_default_is_name_string(self) -> None:
        first = _agent("first", "first-model")
        other = _agent("other", "other-model")
        orch = OrchestrationModel(swarm=SwarmModel(default_agent="other"))
        assert _resolve(orch, [first, other]) == "chat:first-model"

    def test_first_agent_when_no_orchestration(self) -> None:
        assert _resolve(None, [_agent("first", "first-model")]) == "chat:first-model"

    def test_none_when_no_agents(self) -> None:
        orch = OrchestrationModel(swarm=SwarmModel())
        assert _resolve(orch, []) is None
