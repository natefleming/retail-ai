"""
Tests for tool call limit middleware factory.
"""

import pytest
from langchain.agents.middleware import ToolCallLimitMiddleware

from dao_ai.config import (
    PythonFunctionModel,
    SearchToolModel,
    ToolCallLimitModel,
    ToolModel,
)
from dao_ai.middleware import (
    create_tool_call_limit_middleware,
    create_tool_call_limit_middlewares_from_tool_models,
)


class TestCreateToolCallLimitMiddleware:
    """Tests for the create_tool_call_limit_middleware factory function."""

    def test_create_with_run_limit_only(self):
        """Test creating middleware with only run_limit specified."""
        middleware = create_tool_call_limit_middleware(run_limit=5)

        # Middleware is single instance
        assert middleware is not None
        middleware = middleware
        assert isinstance(middleware, ToolCallLimitMiddleware)
        assert middleware.tool_name is None  # Global limit
        assert middleware.run_limit == 5
        assert middleware.thread_limit is None
        assert middleware.exit_behavior == "continue"

    def test_create_with_thread_limit_only(self):
        """Test creating middleware with only thread_limit specified."""
        middleware = create_tool_call_limit_middleware(thread_limit=10)

        # Middleware is single instance
        assert middleware is not None
        middleware = middleware
        assert isinstance(middleware, ToolCallLimitMiddleware)
        assert middleware.tool_name is None  # Global limit
        assert middleware.run_limit is None
        assert middleware.thread_limit == 10
        assert middleware.exit_behavior == "continue"

    def test_create_with_both_limits(self):
        """Test creating middleware with both run_limit and thread_limit."""
        middleware = create_tool_call_limit_middleware(
            thread_limit=20,
            run_limit=10,
        )

        # Middleware is single instance
        assert middleware is not None
        middleware = middleware
        assert isinstance(middleware, ToolCallLimitMiddleware)
        assert middleware.run_limit == 10
        assert middleware.thread_limit == 20
        assert middleware.exit_behavior == "continue"

    def test_create_tool_specific_limit(self):
        """Test creating middleware for a specific tool."""
        middleware = create_tool_call_limit_middleware(
            tool="search_web",
            run_limit=3,
        )

        # Middleware is single instance
        assert middleware is not None
        middleware = middleware
        assert isinstance(middleware, ToolCallLimitMiddleware)
        assert middleware.tool_name == "search_web"
        assert middleware.run_limit == 3

    def test_create_with_continue_behavior(self):
        """Test creating middleware with 'continue' exit behavior (default)."""
        middleware = create_tool_call_limit_middleware(
            run_limit=5,
            exit_behavior="continue",
        )

        assert middleware.exit_behavior == "continue"

    def test_create_with_error_behavior(self):
        """Test creating middleware with 'error' exit behavior for strict enforcement."""
        middleware = create_tool_call_limit_middleware(
            run_limit=2,
            exit_behavior="error",
        )

        assert middleware.exit_behavior == "error"

    def test_create_with_end_behavior(self):
        """Test creating middleware with 'end' exit behavior for graceful termination."""
        middleware = create_tool_call_limit_middleware(
            tool="single_tool",
            run_limit=5,
            exit_behavior="end",
        )

        assert middleware.exit_behavior == "end"

    def test_raises_error_without_limits(self):
        """Test that creating middleware without limits raises ValueError."""
        with pytest.raises(
            ValueError, match="At least one of thread_limit or run_limit"
        ):
            create_tool_call_limit_middleware()

    def test_default_exit_behavior(self):
        """Test that default exit_behavior is 'continue' for graceful handling."""
        middleware = create_tool_call_limit_middleware(run_limit=5)

        assert middleware.exit_behavior == "continue"

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
        assert global_limiters.tool_name is None
        assert global_limiters.run_limit == 10
        assert global_limiters.thread_limit == 20

        assert search_limiters.tool_name == "search_web"
        assert search_limiters.run_limit == 3
        assert search_limiters.exit_behavior == "continue"

        assert strict_limiters.tool_name == "execute_sql"
        assert strict_limiters.run_limit == 2
        assert strict_limiters.exit_behavior == "error"

    def test_graceful_termination_default(self):
        """
        Test that the default configuration supports graceful termination.

        With exit_behavior='continue', the agent can recover from limit errors
        and try alternative approaches.
        """
        middleware = create_tool_call_limit_middleware(run_limit=5)

        # Default is 'continue' which allows graceful recovery
        assert middleware.exit_behavior == "continue"

    def test_factory_accepts_all_parameters(self):
        """Test that factory accepts all supported parameters with type hints."""
        middleware = create_tool_call_limit_middleware(
            tool="test_tool",
            thread_limit=15,
            run_limit=5,
            exit_behavior="error",
        )

        middleware = middleware
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

        # Should return a single middleware instance
        assert isinstance(result, ToolCallLimitMiddleware)
        assert result.run_limit == 5

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

        # Should return a single middleware instance
        assert isinstance(result, ToolCallLimitMiddleware)
        assert result.run_limit == 5

    def test_create_with_invalid_dict(self):
        """Test that invalid dict raises helpful error."""
        # Dict missing required fields
        invalid_dict = {"invalid": "data"}

        with pytest.raises(ValueError, match="Failed to construct ToolModel from dict"):
            create_tool_call_limit_middleware(
                tool=invalid_dict,
                run_limit=5,
            )

    def test_returns_single_instance(self):
        """Test that factory returns single middleware instances."""
        # All variations should return single instances
        global_limits = create_tool_call_limit_middleware(run_limit=10)
        tool_limits = create_tool_call_limit_middleware(tool="test", run_limit=5)

        assert isinstance(global_limits, ToolCallLimitMiddleware)
        assert isinstance(tool_limits, ToolCallLimitMiddleware)

        # Should be composable into list manually
        all_middlewares = [global_limits, tool_limits]
        assert len(all_middlewares) == 2


class TestCreateToolCallLimitMiddlewaresFromToolModels:
    """Tests for the create_tool_call_limit_middlewares_from_tool_models scan."""

    def test_no_tools_returns_empty(self):
        """No tool models at all -> empty list."""
        assert create_tool_call_limit_middlewares_from_tool_models([]) == []

    def test_no_call_limit_returns_empty(self):
        """Tools without a call_limit produce no middleware."""
        tool_models = [
            ToolModel(
                name="time",
                function=PythonFunctionModel(name="dao_ai.tools.current_time_tool"),
            ),
            ToolModel(name="search", function=SearchToolModel()),
        ]

        assert create_tool_call_limit_middlewares_from_tool_models(tool_models) == []

    def test_bare_int_call_limit(self):
        """A bare-int call_limit (normalized on the model) yields run_limit."""
        tool_models = [
            ToolModel(
                name="search",
                function=SearchToolModel(call_limit=3),
            ),
        ]

        middlewares = create_tool_call_limit_middlewares_from_tool_models(tool_models)

        assert len(middlewares) == 1
        mw = middlewares[0]
        assert isinstance(mw, ToolCallLimitMiddleware)
        # SearchToolModel resolves to the duckduckgo_search runtime tool name.
        assert mw.tool_name == "duckduckgo_search"
        assert mw.run_limit == 3
        assert mw.thread_limit is None
        assert mw.exit_behavior == "continue"

    def test_object_call_limit(self):
        """An object call_limit passes through all fields."""
        tool_models = [
            ToolModel(
                name="t",
                function=PythonFunctionModel(
                    name="dao_ai.tools.current_time_tool",
                    call_limit=ToolCallLimitModel(
                        run_limit=2,
                        thread_limit=10,
                        exit_behavior="error",
                    ),
                ),
            ),
        ]

        middlewares = create_tool_call_limit_middlewares_from_tool_models(tool_models)

        assert len(middlewares) == 1
        mw = middlewares[0]
        assert mw.tool_name == "current_time_tool"
        assert mw.run_limit == 2
        assert mw.thread_limit == 10
        assert mw.exit_behavior == "error"

    def test_only_limited_tools_produce_middleware(self):
        """Mixed list: only tools with call_limit are limited."""
        tool_models = [
            ToolModel(name="search", function=SearchToolModel(call_limit=5)),
            ToolModel(
                name="time",
                function=PythonFunctionModel(name="dao_ai.tools.current_time_tool"),
            ),
        ]

        middlewares = create_tool_call_limit_middlewares_from_tool_models(tool_models)

        assert len(middlewares) == 1
        assert middlewares[0].tool_name == "duckduckgo_search"

    def test_multiple_limited_tools(self):
        """Multiple tools each with a call_limit produce one middleware each."""
        tool_models = [
            ToolModel(name="search", function=SearchToolModel(call_limit=5)),
            ToolModel(
                name="t",
                function=PythonFunctionModel(
                    name="dao_ai.tools.current_time_tool",
                    call_limit=2,
                ),
            ),
        ]

        middlewares = create_tool_call_limit_middlewares_from_tool_models(tool_models)

        assert len(middlewares) == 2
        by_name = {mw.tool_name: mw for mw in middlewares}
        assert by_name["duckduckgo_search"].run_limit == 5
        assert by_name["current_time_tool"].run_limit == 2
