"""
Tests for tool call limit middleware factory.
"""

import pytest
from langchain.agents.middleware import ToolCallLimitMiddleware

from dao_ai.config import PythonFunctionModel, ToolModel
from dao_ai.middleware import create_tool_call_limit_middleware


class TestCreateToolCallLimitMiddleware:
    """Tests for the create_tool_call_limit_middleware factory function."""

    def test_create_with_run_limit_only(self):
        """Test creating middleware with only run_limit specified."""
        middlewares = create_tool_call_limit_middleware(run_limit=5)

        assert isinstance(middlewares, list)
        assert len(middlewares) == 1
        middleware = middlewares[0]
        assert isinstance(middleware, ToolCallLimitMiddleware)
        assert middleware.tool_name is None  # Global limit
        assert middleware.run_limit == 5
        assert middleware.thread_limit is None
        assert middleware.exit_behavior == "continue"

    def test_create_with_thread_limit_only(self):
        """Test creating middleware with only thread_limit specified."""
        middlewares = create_tool_call_limit_middleware(thread_limit=10)

        assert isinstance(middlewares, list)
        assert len(middlewares) == 1
        middleware = middlewares[0]
        assert isinstance(middleware, ToolCallLimitMiddleware)
        assert middleware.tool_name is None  # Global limit
        assert middleware.run_limit is None
        assert middleware.thread_limit == 10
        assert middleware.exit_behavior == "continue"

    def test_create_with_both_limits(self):
        """Test creating middleware with both run_limit and thread_limit."""
        middlewares = create_tool_call_limit_middleware(
            thread_limit=20,
            run_limit=10,
        )

        assert isinstance(middlewares, list)
        assert len(middlewares) == 1
        middleware = middlewares[0]
        assert isinstance(middleware, ToolCallLimitMiddleware)
        assert middleware.run_limit == 10
        assert middleware.thread_limit == 20
        assert middleware.exit_behavior == "continue"

    def test_create_tool_specific_limit(self):
        """Test creating middleware for a specific tool."""
        middlewares = create_tool_call_limit_middleware(
            tool="search_web",
            run_limit=3,
        )

        assert isinstance(middlewares, list)
        assert len(middlewares) == 1
        middleware = middlewares[0]
        assert isinstance(middleware, ToolCallLimitMiddleware)
        assert middleware.tool_name == "search_web"
        assert middleware.run_limit == 3

    def test_create_with_continue_behavior(self):
        """Test creating middleware with 'continue' exit behavior (default)."""
        middlewares = create_tool_call_limit_middleware(
            run_limit=5,
            exit_behavior="continue",
        )

        assert middlewares[0].exit_behavior == "continue"

    def test_create_with_error_behavior(self):
        """Test creating middleware with 'error' exit behavior for strict enforcement."""
        middlewares = create_tool_call_limit_middleware(
            run_limit=2,
            exit_behavior="error",
        )

        assert middlewares[0].exit_behavior == "error"

    def test_create_with_end_behavior(self):
        """Test creating middleware with 'end' exit behavior for graceful termination."""
        middlewares = create_tool_call_limit_middleware(
            tool="single_tool",
            run_limit=5,
            exit_behavior="end",
        )

        assert middlewares[0].exit_behavior == "end"

    def test_raises_error_without_limits(self):
        """Test that creating middleware without limits raises ValueError."""
        with pytest.raises(
            ValueError, match="At least one of thread_limit or run_limit"
        ):
            create_tool_call_limit_middleware()

    def test_default_exit_behavior(self):
        """Test that default exit_behavior is 'continue' for graceful handling."""
        middlewares = create_tool_call_limit_middleware(run_limit=5)

        assert middlewares[0].exit_behavior == "continue"

    def test_multiple_limiters_configuration(self):
        """Test creating multiple limiters for different tools."""
        global_limiters = create_tool_call_limit_middleware(
            thread_limit=20,
            run_limit=10,
        )

        search_limiters = create_tool_call_limit_middleware(
            tool="search_web",
            run_limit=3,
            exit_behavior="continue",
        )

        strict_limiters = create_tool_call_limit_middleware(
            tool="execute_sql",
            run_limit=2,
            exit_behavior="error",
        )

        # Verify all limiters are configured correctly
        global_limiter = global_limiters[0]
        assert global_limiter.tool_name is None
        assert global_limiter.run_limit == 10
        assert global_limiter.thread_limit == 20

        search_limiter = search_limiters[0]
        assert search_limiter.tool_name == "search_web"
        assert search_limiter.run_limit == 3
        assert search_limiter.exit_behavior == "continue"

        strict_limiter = strict_limiters[0]
        assert strict_limiter.tool_name == "execute_sql"
        assert strict_limiter.run_limit == 2
        assert strict_limiter.exit_behavior == "error"

    def test_graceful_termination_default(self):
        """
        Test that the default configuration supports graceful termination.

        With exit_behavior='continue', the agent can recover from limit errors
        and try alternative approaches.
        """
        middlewares = create_tool_call_limit_middleware(run_limit=5)

        # Default is 'continue' which allows graceful recovery
        assert middlewares[0].exit_behavior == "continue"

    def test_factory_accepts_all_parameters(self):
        """Test that factory accepts all supported parameters with type hints."""
        middlewares = create_tool_call_limit_middleware(
            tool="test_tool",
            thread_limit=15,
            run_limit=5,
            exit_behavior="error",
        )

        middleware = middlewares[0]
        assert middleware.tool_name == "test_tool"
        assert middleware.thread_limit == 15
        assert middleware.run_limit == 5
        assert middleware.exit_behavior == "error"

    def test_create_with_tool_model(self):
        """Test creating middleware with ToolModel instead of string."""
        # Create a simple ToolModel
        tool_model = ToolModel(
            name="my_tool",
            function=PythonFunctionModel(name="dao_ai.tools.say_hello_tool"),
        )

        # Create middleware from ToolModel
        result = create_tool_call_limit_middleware(
            tool=tool_model,
            run_limit=5,
        )

        # Should return a list of middlewares
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(m, ToolCallLimitMiddleware) for m in result)
        # Each middleware should have the limits configured
        for middleware in result:
            assert middleware.run_limit == 5

    def test_create_with_dict(self):
        """Test creating middleware with dict instead of ToolModel."""
        # Create tool as dict
        tool_dict = {
            "name": "my_tool",
            "function": {"name": "dao_ai.tools.say_hello_tool"},
        }

        # Create middleware from dict
        result = create_tool_call_limit_middleware(
            tool=tool_dict,
            run_limit=5,
        )

        # Should return a list of middlewares
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(m, ToolCallLimitMiddleware) for m in result)
        for middleware in result:
            assert middleware.run_limit == 5

    def test_create_with_invalid_dict(self):
        """Test that invalid dict raises helpful error."""
        # Dict missing required fields
        invalid_dict = {"invalid": "data"}

        with pytest.raises(ValueError, match="Failed to construct ToolModel from dict"):
            create_tool_call_limit_middleware(
                tool=invalid_dict,
                run_limit=5,
            )

    def test_returns_list_for_composition(self):
        """Test that factory always returns list for easy composition."""
        # All variations should return lists
        global_limits = create_tool_call_limit_middleware(run_limit=10)
        tool_limits = create_tool_call_limit_middleware(tool="test", run_limit=5)

        assert isinstance(global_limits, list)
        assert isinstance(tool_limits, list)

        # Should be composable with +
        all_middlewares = global_limits + tool_limits
        assert len(all_middlewares) == 2
