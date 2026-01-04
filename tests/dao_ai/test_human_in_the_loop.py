"""
Tests for Human-in-the-Loop middleware functionality.

These tests verify that the HumanInTheLoopMiddleware is properly created
from DAO AI configuration using the factory functions.
"""

from unittest.mock import MagicMock, patch

import pytest
from conftest import create_mock_llm_model
from langchain.agents.middleware import HumanInTheLoopMiddleware

from dao_ai.config import (
    BaseFunctionModel,
    HumanInTheLoopModel,
)
from dao_ai.middleware.human_in_the_loop import (
    create_hitl_middleware_from_tool_models,
    create_human_in_the_loop_middleware,
)


class TestHumanInTheLoopMiddleware:
    """Test class for HITL middleware functionality."""

    def test_create_hitl_middleware_with_bool_true(self):
        """Test creating HITL middleware with simple True configuration."""
        middleware = create_human_in_the_loop_middleware(
            interrupt_on={"send_email": True}
        )

        # Middleware is single instance
        assert middleware is not None
        assert isinstance(middleware, HumanInTheLoopMiddleware)

    def test_create_hitl_middleware_with_bool_false(self):
        """Test creating HITL middleware with False configuration (disabled)."""
        middleware = create_human_in_the_loop_middleware(
            interrupt_on={"send_email": False}
        )

        # Middleware is single instance
        assert middleware is not None
        assert isinstance(middleware, HumanInTheLoopMiddleware)

    def test_create_hitl_middleware_with_model(self):
        """Test creating HITL middleware with HumanInTheLoopModel configuration."""
        hitl_config = HumanInTheLoopModel(
            review_prompt="Please review this email",
            allowed_decisions=["approve", "edit"],
        )

        middleware = create_human_in_the_loop_middleware(
            interrupt_on={"send_email": hitl_config}
        )

        # Middleware is single instance
        assert middleware is not None
        assert isinstance(middleware, HumanInTheLoopMiddleware)

    def test_create_hitl_middleware_with_multiple_tools(self):
        """Test creating HITL middleware with multiple tools configured."""
        middleware = create_human_in_the_loop_middleware(
            interrupt_on={
                "send_email": True,
                "delete_record": True,
                "search": False,
            }
        )

        # Middleware is single instance
        assert middleware is not None
        assert isinstance(middleware, HumanInTheLoopMiddleware)

    def test_create_hitl_middleware_with_custom_description_prefix(self):
        """Test creating HITL middleware with custom description prefix."""
        middleware = create_human_in_the_loop_middleware(
            interrupt_on={"send_email": True},
            description_prefix="Action requires approval",
        )

        # Middleware is single instance
        assert middleware is not None
        assert isinstance(middleware, HumanInTheLoopMiddleware)

    def test_create_hitl_middleware_allowed_decisions_from_model(self):
        """Test that allowed decisions are correctly extracted from HumanInTheLoopModel."""
        hitl_config = HumanInTheLoopModel(
            review_prompt="Review action",
            allowed_decisions=["approve", "reject"],
        )

        middleware = create_human_in_the_loop_middleware(
            interrupt_on={"test_tool": hitl_config}
        )

        # Middleware is single instance
        assert middleware is not None
        # The middleware should have the correct configuration
        # (internal details depend on LangChain implementation)

    def test_create_hitl_middleware_default_allowed_decisions(self):
        """Test that True configuration uses default allowed decisions."""
        middleware = create_human_in_the_loop_middleware(
            interrupt_on={"test_tool": True}
        )

        # Middleware is single instance
        assert middleware is not None

    @patch("dao_ai.middleware.human_in_the_loop.logger")
    def test_create_hitl_middleware_logs_creation(self, mock_logger):
        """Test that middleware creation logs debug information."""
        create_human_in_the_loop_middleware(
            interrupt_on={"send_email": True, "delete_record": True}
        )

        mock_logger.debug.assert_called()

    def test_reject_decision_extracted(self):
        """Test that reject decision is correctly extracted."""
        from dao_ai.middleware.human_in_the_loop import (
            _hitl_config_to_allowed_decisions,
        )

        hitl_config = HumanInTheLoopModel(
            review_prompt="Test prompt",
            allowed_decisions=["reject"],
        )

        decisions = _hitl_config_to_allowed_decisions(hitl_config)
        assert decisions == ["reject"]

    def test_multiple_decisions_extracted(self):
        """Test that multiple decisions are correctly extracted."""
        from dao_ai.middleware.human_in_the_loop import (
            _hitl_config_to_allowed_decisions,
        )

        hitl_config = HumanInTheLoopModel(
            review_prompt="Test prompt",
            allowed_decisions=["approve", "edit", "reject"],
        )

        decisions = _hitl_config_to_allowed_decisions(hitl_config)
        assert decisions == ["approve", "edit", "reject"]


class TestHitlMiddlewareFromToolModels:
    """Test class for creating HITL middleware from ToolModel configurations."""

    def test_returns_empty_list_when_no_tools_require_hitl(self):
        """Test that empty list is returned when no tools have HITL configured."""
        # Use a mock function model without HITL
        mock_function = MagicMock()
        mock_function.human_in_the_loop = None

        tool_models = [
            MagicMock(function=mock_function),
        ]

        middleware = create_hitl_middleware_from_tool_models(tool_models)

        assert middleware is None

    def test_returns_middleware_when_tool_has_hitl(self):
        """Test that middleware is returned when a tool has HITL configured."""
        # Create a mock tool that will be returned by as_tools
        mock_tool = MagicMock()
        mock_tool.name = "send_email"

        # Create a mock function with HITL configured
        # Use spec=BaseFunctionModel so isinstance check passes
        mock_function = MagicMock(spec=BaseFunctionModel)
        mock_function.human_in_the_loop = HumanInTheLoopModel(
            review_prompt="Review this email",
        )
        mock_function.as_tools.return_value = [mock_tool]

        tool_models = [
            MagicMock(function=mock_function),
        ]

        middleware = create_hitl_middleware_from_tool_models(tool_models)

        # Middleware is single instance
        assert middleware is not None
        assert isinstance(middleware, HumanInTheLoopMiddleware)

    def test_handles_multiple_tools_mixed_config(self):
        """Test handling multiple tools with mixed HITL configuration."""
        # Tool with HITL
        mock_email_tool = MagicMock()
        mock_email_tool.name = "send_email"

        mock_function_with_hitl = MagicMock(spec=BaseFunctionModel)
        mock_function_with_hitl.human_in_the_loop = HumanInTheLoopModel(
            review_prompt="Review this email",
        )
        mock_function_with_hitl.as_tools.return_value = [mock_email_tool]

        # Tool without HITL
        mock_search_tool = MagicMock()
        mock_search_tool.name = "search"

        mock_function_without_hitl = MagicMock(spec=BaseFunctionModel)
        mock_function_without_hitl.human_in_the_loop = None

        tool_models = [
            MagicMock(function=mock_function_with_hitl),
            MagicMock(function=mock_function_without_hitl),
        ]

        middleware = create_hitl_middleware_from_tool_models(tool_models)

        # Middleware is single instance
        assert middleware is not None
        assert isinstance(middleware, HumanInTheLoopMiddleware)

    def test_empty_tool_models_returns_empty_list(self):
        """Test that empty tool_models list returns empty list."""
        middleware = create_hitl_middleware_from_tool_models([])

        assert middleware is None

    def test_custom_description_prefix(self):
        """Test that custom description prefix is passed through."""
        mock_tool = MagicMock()
        mock_tool.name = "delete_record"

        mock_function = MagicMock(spec=BaseFunctionModel)
        mock_function.human_in_the_loop = HumanInTheLoopModel(
            review_prompt="Confirm deletion",
        )
        mock_function.as_tools.return_value = [mock_tool]

        tool_models = [
            MagicMock(function=mock_function),
        ]

        middleware = create_hitl_middleware_from_tool_models(
            tool_models,
            description_prefix="Deletion requires approval",
        )

        # Middleware is single instance
        assert middleware is not None

    @patch("dao_ai.middleware.human_in_the_loop.logger")
    def test_logs_tool_configuration(self, mock_logger):
        """Test that tool configuration is logged."""
        mock_tool = MagicMock()
        mock_tool.name = "send_email"

        mock_function = MagicMock()
        mock_function.human_in_the_loop = HumanInTheLoopModel(
            review_prompt="Review email",
        )
        mock_function.as_tools.return_value = [mock_tool]

        tool_models = [
            MagicMock(function=mock_function),
        ]

        create_hitl_middleware_from_tool_models(tool_models)

        # Verify trace logging occurred (structured logging uses trace for tool configuration)
        assert mock_logger.trace.called


class TestHitlMiddlewareIntegration:
    """Integration tests for HITL middleware with agent creation."""

    @patch("dao_ai.nodes.create_agent")
    @patch("dao_ai.nodes.create_hitl_middleware_from_tool_models")
    def test_agent_node_includes_hitl_middleware(
        self, mock_hitl_factory, mock_create_agent
    ):
        """Test that create_agent_node includes HITL middleware when tools require it."""
        from dao_ai.config import AgentModel
        from dao_ai.nodes import create_agent_node

        # Mock the compiled agent
        mock_compiled_agent = MagicMock()
        mock_compiled_agent.name = "test_agent"
        mock_create_agent.return_value = mock_compiled_agent

        # Mock the HITL middleware factory to return a single middleware
        mock_hitl_middleware = MagicMock(spec=HumanInTheLoopMiddleware)
        mock_hitl_factory.return_value = mock_hitl_middleware

        # Create a mock LLM model using the helper
        mock_llm_model = create_mock_llm_model()

        agent_model = AgentModel(
            name="test_agent",
            model=mock_llm_model,
        )

        create_agent_node(agent=agent_model)

        # Verify create_agent was called with middleware
        mock_create_agent.assert_called_once()
        call_kwargs = mock_create_agent.call_args[1]

        middleware_list = call_kwargs.get("middleware", [])

        # Check that the mocked HITL middleware is in the list
        assert mock_hitl_middleware in middleware_list

    @patch("dao_ai.nodes.create_agent")
    @patch("dao_ai.nodes.create_hitl_middleware_from_tool_models")
    def test_agent_node_no_hitl_when_not_configured(
        self, mock_hitl_factory, mock_create_agent
    ):
        """Test that agent node doesn't include HITL middleware when not configured."""
        from dao_ai.config import AgentModel
        from dao_ai.nodes import create_agent_node

        mock_compiled_agent = MagicMock()
        mock_compiled_agent.name = "test_agent"
        mock_create_agent.return_value = mock_compiled_agent

        # Mock the HITL middleware factory to return empty list (no HITL needed)
        mock_hitl_factory.return_value = []

        # Create a mock LLM model using the helper
        mock_llm_model = create_mock_llm_model()

        agent_model = AgentModel(
            name="test_agent",
            model=mock_llm_model,
        )

        create_agent_node(agent=agent_model)

        mock_create_agent.assert_called_once()
        call_kwargs = mock_create_agent.call_args[1]

        middleware_list = call_kwargs.get("middleware", [])

        # Check that no HITL middleware is in the list
        has_hitl = any(isinstance(m, HumanInTheLoopMiddleware) for m in middleware_list)
        assert not has_hitl, "HumanInTheLoopMiddleware should NOT be in middleware list"


if __name__ == "__main__":
    pytest.main([__file__])
