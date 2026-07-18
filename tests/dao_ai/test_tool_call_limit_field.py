"""
Tests for the `call_limit` tool field and ToolCallLimitModel config model.

These verify the config-model layer: that a bare integer normalizes to a
ToolCallLimitModel, that the object form parses, that invalid values are
rejected, and that the field is inherited by the tool function subtypes.
"""

from unittest.mock import MagicMock, patch

import pytest
from conftest import create_mock_llm_model
from langchain.agents.middleware import ToolCallLimitMiddleware
from pydantic import ValidationError

from dao_ai.config import (
    AgentModel,
    GenieToolModel,
    PythonFunctionModel,
    SearchToolModel,
    ToolCallLimitModel,
    ToolModel,
)


class TestToolCallLimitModel:
    """Validation tests for the ToolCallLimitModel config model."""

    def test_run_limit_only(self):
        model = ToolCallLimitModel(run_limit=3)
        assert model.run_limit == 3
        assert model.thread_limit is None
        assert model.exit_behavior == "continue"

    def test_thread_limit_only(self):
        model = ToolCallLimitModel(thread_limit=10)
        assert model.thread_limit == 10
        assert model.run_limit is None

    def test_all_fields(self):
        model = ToolCallLimitModel(run_limit=2, thread_limit=8, exit_behavior="error")
        assert model.run_limit == 2
        assert model.thread_limit == 8
        assert model.exit_behavior == "error"

    def test_requires_at_least_one_limit(self):
        with pytest.raises(ValidationError, match="At least one of run_limit"):
            ToolCallLimitModel()

    def test_requires_at_least_one_limit_with_only_exit_behavior(self):
        with pytest.raises(ValidationError, match="At least one of run_limit"):
            ToolCallLimitModel(exit_behavior="error")

    @pytest.mark.parametrize("bad", [0, -1, -100])
    def test_run_limit_must_be_positive(self, bad):
        with pytest.raises(ValidationError):
            ToolCallLimitModel(run_limit=bad)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_thread_limit_must_be_positive(self, bad):
        with pytest.raises(ValidationError):
            ToolCallLimitModel(thread_limit=bad)

    def test_invalid_exit_behavior_rejected(self):
        with pytest.raises(ValidationError):
            ToolCallLimitModel(run_limit=3, exit_behavior="halt")

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            ToolCallLimitModel(run_limit=3, unknown=True)


class TestCallLimitFieldNormalization:
    """The call_limit field on tool functions normalizes bare ints."""

    def test_bare_int_normalizes_to_model(self):
        fn = PythonFunctionModel(name="dao_ai.tools.current_time_tool", call_limit=3)
        assert isinstance(fn.call_limit, ToolCallLimitModel)
        assert fn.call_limit.run_limit == 3
        assert fn.call_limit.exit_behavior == "continue"
        assert fn.call_limit.thread_limit is None

    def test_object_form_parses(self):
        fn = PythonFunctionModel(
            name="dao_ai.tools.current_time_tool",
            call_limit={"run_limit": 2, "thread_limit": 10, "exit_behavior": "error"},
        )
        assert isinstance(fn.call_limit, ToolCallLimitModel)
        assert fn.call_limit.run_limit == 2
        assert fn.call_limit.thread_limit == 10
        assert fn.call_limit.exit_behavior == "error"

    def test_default_is_none(self):
        fn = PythonFunctionModel(name="dao_ai.tools.current_time_tool")
        assert fn.call_limit is None

    def test_bare_zero_rejected_via_normalization(self):
        """A bare 0 int normalizes into ToolCallLimitModel(run_limit=0) and fails gt=0."""
        with pytest.raises(ValidationError):
            PythonFunctionModel(name="dao_ai.tools.current_time_tool", call_limit=0)

    def test_bool_not_treated_as_int(self):
        """bool is a subclass of int; True must not become run_limit=1."""
        with pytest.raises(ValidationError):
            PythonFunctionModel(name="dao_ai.tools.current_time_tool", call_limit=True)


class TestCallLimitInheritedBySubtypes:
    """call_limit is defined on BaseFunctionModel, so every subtype has it."""

    def test_search_tool_has_call_limit(self):
        fn = SearchToolModel(call_limit=4)
        assert isinstance(fn.call_limit, ToolCallLimitModel)
        assert fn.call_limit.run_limit == 4

    def test_genie_tool_has_call_limit(self):
        fn = GenieToolModel(
            name="retail_genie_tool",
            genie_room={"space_id": "01f0abc"},
            call_limit={"run_limit": 2, "exit_behavior": "error"},
        )
        assert isinstance(fn.call_limit, ToolCallLimitModel)
        assert fn.call_limit.run_limit == 2
        assert fn.call_limit.exit_behavior == "error"

    def test_python_tool_defaults_none(self):
        fn = PythonFunctionModel(name="dao_ai.tools.current_time_tool")
        assert fn.call_limit is None


class TestCallLimitAgentWiring:
    """A tool's call_limit auto-registers ToolCallLimitMiddleware on the agent."""

    @patch("dao_ai.nodes.create_agent")
    def test_agent_node_includes_call_limit_middleware(self, mock_create_agent):
        """An agent whose tool has call_limit gets a ToolCallLimitMiddleware."""
        from dao_ai.nodes import create_agent_node

        mock_compiled_agent = MagicMock()
        mock_compiled_agent.name = "test_agent"
        mock_create_agent.return_value = mock_compiled_agent

        agent_model = AgentModel(
            name="test_agent",
            model=create_mock_llm_model(),
            tools=[
                ToolModel(name="search", function=SearchToolModel(call_limit=3)),
            ],
        )

        create_agent_node(agent=agent_model)

        mock_create_agent.assert_called_once()
        middleware_list = mock_create_agent.call_args[1].get("middleware", [])

        limit_mws = [
            m for m in middleware_list if isinstance(m, ToolCallLimitMiddleware)
        ]
        assert len(limit_mws) == 1
        assert limit_mws[0].tool_name == "duckduckgo_search"
        assert limit_mws[0].run_limit == 3

    @patch("dao_ai.nodes.create_agent")
    def test_agent_node_no_call_limit_when_not_configured(self, mock_create_agent):
        """No tool has call_limit -> no ToolCallLimitMiddleware."""
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

        has_limit = any(isinstance(m, ToolCallLimitMiddleware) for m in middleware_list)
        assert not has_limit
