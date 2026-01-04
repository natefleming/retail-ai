"""
Tests for tool retry middleware factory.
"""

import pytest
from langchain.agents.middleware import ToolRetryMiddleware

from dao_ai.config import PythonFunctionModel, ToolModel
from dao_ai.middleware import create_tool_retry_middleware


class TestCreateToolRetryMiddleware:
    """Tests for the create_tool_retry_middleware factory function."""

    def test_create_with_defaults(self):
        """Test creating middleware with default parameters."""
        middleware = create_tool_retry_middleware()

        # Middleware is single instance
        assert middleware is not None
        middleware = middleware
        assert isinstance(middleware, ToolRetryMiddleware)
        assert middleware.max_retries == 3
        assert middleware.backoff_factor == 2.0
        assert middleware.initial_delay == 1.0
        assert middleware.on_failure == "continue"

    def test_create_with_custom_retries(self):
        """Test creating middleware with custom max_retries."""
        middleware = create_tool_retry_middleware(max_retries=5)

        assert middleware.max_retries == 5

    def test_create_with_custom_backoff(self):
        """Test creating middleware with custom backoff settings."""
        middleware = create_tool_retry_middleware(
            backoff_factor=1.5,
            initial_delay=0.5,
        )

        middleware = middleware
        assert middleware.backoff_factor == 1.5
        assert middleware.initial_delay == 0.5

    def test_create_with_max_delay(self):
        """Test creating middleware with max_delay cap."""
        middleware = create_tool_retry_middleware(
            max_retries=10,
            max_delay=60.0,
        )

        assert middleware.max_delay == 60.0

    def test_create_with_jitter(self):
        """Test creating middleware with jitter enabled."""
        middleware = create_tool_retry_middleware(jitter=True)

        assert middleware.jitter is True

    def test_create_with_string_tools(self):
        """Test creating middleware for specific tools by name."""
        middleware = create_tool_retry_middleware(
            tools=["search_web", "query_database"],
        )

        # Middleware should be created successfully with tool filtering
        # Middleware is single instance
        assert middleware is not None
        assert isinstance(middleware, ToolRetryMiddleware)

    def test_create_with_tool_model(self):
        """Test creating middleware with ToolModel."""
        tool_model = ToolModel(
            name="my_tool",
            function=PythonFunctionModel(name="dao_ai.tools.say_hello_tool"),
        )

        middleware = create_tool_retry_middleware(tools=[tool_model])

        # Should create middleware successfully
        # Middleware is single instance
        assert middleware is not None
        assert isinstance(middleware, ToolRetryMiddleware)

    def test_create_with_dict_tool(self):
        """Test creating middleware with dict tool config."""
        tool_dict = {
            "name": "my_tool",
            "function": {"name": "dao_ai.tools.say_hello_tool"},
        }

        middleware = create_tool_retry_middleware(tools=[tool_dict])

        # Middleware is single instance
        assert middleware is not None

    def test_create_with_error_on_failure(self):
        """Test creating middleware with error on_failure behavior."""
        middleware = create_tool_retry_middleware(on_failure="error")

        assert middleware.on_failure == "error"

    def test_create_with_exception_tuple(self):
        """Test creating middleware with specific exception types."""
        middleware = create_tool_retry_middleware(
            retry_on=(TimeoutError, ConnectionError),
        )

        assert middleware.retry_on == (TimeoutError, ConnectionError)

    def test_create_with_retry_callable(self):
        """Test creating middleware with callable retry_on."""

        def should_retry(error: Exception) -> bool:
            return "rate_limit" in str(error).lower()

        middleware = create_tool_retry_middleware(retry_on=should_retry)

        assert middleware.retry_on is should_retry

    def test_create_with_custom_on_failure_callable(self):
        """Test creating middleware with callable on_failure."""

        def format_error(error: Exception) -> str:
            return f"Tool failed: {error}"

        middleware = create_tool_retry_middleware(on_failure=format_error)

        assert middleware.on_failure is format_error

    def test_create_with_invalid_dict(self):
        """Test that invalid dict raises helpful error."""
        invalid_dict = {"invalid": "data"}

        with pytest.raises(ValueError, match="Failed to construct ToolModel from dict"):
            create_tool_retry_middleware(tools=[invalid_dict])

    def test_create_all_tools_when_none(self):
        """Test that None tools applies to all tools."""
        middleware = create_tool_retry_middleware(tools=None)

        # Should create successfully without tool filtering
        # Middleware is single instance
        assert middleware is not None
        assert isinstance(middleware, ToolRetryMiddleware)

    def test_returns_list_for_composition(self):
        """Test that factory returns list for easy composition."""
        middleware = create_tool_retry_middleware()

        # Middleware is single instance
        assert middleware is not None
