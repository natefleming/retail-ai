"""
Tests for the `call_limit` agent field and ModelCallLimitModel config model.

These verify the config-model layer: that a bare integer normalizes to a
ModelCallLimitModel, that the object form parses, that invalid values are
rejected, that the model-call variant defaults to and restricts exit_behavior
to end/error, and that an agent's call_limit registers a
ModelCallLimitMiddleware.
"""

from unittest.mock import MagicMock, patch

import pytest
from conftest import create_mock_llm_model
from langchain.agents.middleware import ModelCallLimitMiddleware
from pydantic import ValidationError

from dao_ai.config import (
    AgentModel,
    ModelCallLimitModel,
    SearchToolModel,
    ToolModel,
)


class TestModelCallLimitModel:
    """Validation tests for the ModelCallLimitModel config model."""

    def test_run_limit_only(self):
        model = ModelCallLimitModel(run_limit=3)
        assert model.run_limit == 3
        assert model.thread_limit is None
        # Unlike ToolCallLimitModel, the model-call variant defaults to 'end'.
        assert model.exit_behavior == "end"

    def test_thread_limit_only(self):
        model = ModelCallLimitModel(thread_limit=10)
        assert model.thread_limit == 10
        assert model.run_limit is None

    def test_all_fields(self):
        model = ModelCallLimitModel(run_limit=2, thread_limit=8, exit_behavior="error")
        assert model.run_limit == 2
        assert model.thread_limit == 8
        assert model.exit_behavior == "error"

    def test_requires_at_least_one_limit(self):
        with pytest.raises(ValidationError, match="At least one of run_limit"):
            ModelCallLimitModel()

    @pytest.mark.parametrize("bad", [0, -1, -100])
    def test_run_limit_must_be_positive(self, bad):
        with pytest.raises(ValidationError):
            ModelCallLimitModel(run_limit=bad)

    def test_continue_exit_behavior_rejected(self):
        """'continue' is valid for tool-call limits but NOT model-call limits."""
        with pytest.raises(ValidationError):
            ModelCallLimitModel(run_limit=3, exit_behavior="continue")

    def test_invalid_exit_behavior_rejected(self):
        with pytest.raises(ValidationError):
            ModelCallLimitModel(run_limit=3, exit_behavior="halt")

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            ModelCallLimitModel(run_limit=3, unknown=True)

    def test_turn_limit_alias(self):
        model = ModelCallLimitModel(turn_limit=5)
        assert model.run_limit == 5


class TestAgentCallLimitNormalization:
    """The call_limit field on AgentModel normalizes bare ints."""

    def test_bare_int_normalizes_to_model(self):
        agent = AgentModel(name="a", model=create_mock_llm_model(), call_limit=10)
        assert isinstance(agent.call_limit, ModelCallLimitModel)
        assert agent.call_limit.run_limit == 10
        assert agent.call_limit.exit_behavior == "end"
        assert agent.call_limit.thread_limit is None

    def test_object_form_parses(self):
        agent = AgentModel(
            name="a",
            model=create_mock_llm_model(),
            call_limit={"run_limit": 8, "thread_limit": 40, "exit_behavior": "error"},
        )
        assert isinstance(agent.call_limit, ModelCallLimitModel)
        assert agent.call_limit.run_limit == 8
        assert agent.call_limit.thread_limit == 40
        assert agent.call_limit.exit_behavior == "error"

    def test_default_is_none(self):
        agent = AgentModel(name="a", model=create_mock_llm_model())
        assert agent.call_limit is None

    def test_bare_zero_rejected(self):
        with pytest.raises(ValidationError):
            AgentModel(name="a", model=create_mock_llm_model(), call_limit=0)

    def test_bool_not_treated_as_int(self):
        with pytest.raises(ValidationError):
            AgentModel(name="a", model=create_mock_llm_model(), call_limit=True)


class TestAgentCallLimitWiring:
    """An agent's call_limit auto-registers a ModelCallLimitMiddleware."""

    @patch("dao_ai.nodes.create_agent")
    def test_agent_node_includes_model_call_limit_middleware(self, mock_create_agent):
        from dao_ai.nodes import create_agent_node

        mock_compiled_agent = MagicMock()
        mock_compiled_agent.name = "test_agent"
        mock_create_agent.return_value = mock_compiled_agent

        agent_model = AgentModel(
            name="test_agent",
            model=create_mock_llm_model(),
            tools=[ToolModel(name="search", function=SearchToolModel())],
            call_limit={"run_limit": 6, "exit_behavior": "end"},
        )

        create_agent_node(agent=agent_model)

        mock_create_agent.assert_called_once()
        middleware_list = mock_create_agent.call_args[1].get("middleware", [])

        limit_mws = [
            m for m in middleware_list if isinstance(m, ModelCallLimitMiddleware)
        ]
        assert len(limit_mws) == 1
        assert limit_mws[0].run_limit == 6

    @patch("dao_ai.nodes.create_agent")
    def test_agent_node_no_model_call_limit_when_not_configured(
        self, mock_create_agent
    ):
        from dao_ai.nodes import create_agent_node

        mock_compiled_agent = MagicMock()
        mock_compiled_agent.name = "test_agent"
        mock_create_agent.return_value = mock_compiled_agent

        agent_model = AgentModel(
            name="test_agent",
            model=create_mock_llm_model(),
            tools=[ToolModel(name="search", function=SearchToolModel())],
        )

        create_agent_node(agent=agent_model)

        mock_create_agent.assert_called_once()
        middleware_list = mock_create_agent.call_args[1].get("middleware", [])

        has_limit = any(
            isinstance(m, ModelCallLimitMiddleware) for m in middleware_list
        )
        assert not has_limit
