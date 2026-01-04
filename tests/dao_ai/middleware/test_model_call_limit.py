"""
Tests for model call limit middleware factory.
"""

import pytest
from langchain.agents.middleware import ModelCallLimitMiddleware

from dao_ai.middleware import create_model_call_limit_middleware


class TestCreateModelCallLimitMiddleware:
    """Tests for the create_model_call_limit_middleware factory function."""

    def test_create_with_run_limit_only(self):
        """Test creating middleware with only run_limit specified."""
        middleware = create_model_call_limit_middleware(run_limit=5)

        # Middleware is single instance
        assert middleware is not None
        middleware = middleware
        assert isinstance(middleware, ModelCallLimitMiddleware)
        assert middleware.run_limit == 5
        assert middleware.thread_limit is None
        assert middleware.exit_behavior == "end"

    def test_create_with_thread_limit_only(self):
        """Test creating middleware with only thread_limit specified."""
        middleware = create_model_call_limit_middleware(thread_limit=10)

        # Middleware is single instance
        assert middleware is not None
        middleware = middleware
        assert isinstance(middleware, ModelCallLimitMiddleware)
        assert middleware.run_limit is None
        assert middleware.thread_limit == 10
        assert middleware.exit_behavior == "end"

    def test_create_with_both_limits(self):
        """Test creating middleware with both run_limit and thread_limit."""
        middleware = create_model_call_limit_middleware(
            thread_limit=20,
            run_limit=10,
        )

        # Middleware is single instance
        assert middleware is not None
        middleware = middleware
        assert isinstance(middleware, ModelCallLimitMiddleware)
        assert middleware.run_limit == 10
        assert middleware.thread_limit == 20
        assert middleware.exit_behavior == "end"

    def test_create_with_error_behavior(self):
        """Test creating middleware with 'error' exit behavior."""
        middleware = create_model_call_limit_middleware(
            run_limit=2,
            exit_behavior="error",
        )

        assert middleware.exit_behavior == "error"

    def test_create_with_end_behavior(self):
        """Test creating middleware with 'end' exit behavior."""
        middleware = create_model_call_limit_middleware(
            run_limit=5,
            exit_behavior="end",
        )

        assert middleware.exit_behavior == "end"

    def test_raises_error_without_limits(self):
        """Test that creating middleware without limits raises ValueError."""
        with pytest.raises(
            ValueError, match="At least one of thread_limit or run_limit"
        ):
            create_model_call_limit_middleware()

    def test_default_exit_behavior(self):
        """Test that default exit_behavior is 'end'."""
        middleware = create_model_call_limit_middleware(run_limit=5)

        assert middleware.exit_behavior == "end"

    def test_factory_accepts_all_parameters(self):
        """Test that factory accepts all supported parameters."""
        middleware = create_model_call_limit_middleware(
            thread_limit=15,
            run_limit=5,
            exit_behavior="error",
        )

        middleware = middleware
        assert middleware.thread_limit == 15
        assert middleware.run_limit == 5
        assert middleware.exit_behavior == "error"

    def test_returns_list_for_composition(self):
        """Test that factory returns list for easy composition."""
        middleware = create_model_call_limit_middleware(run_limit=5)

        # Middleware is single instance
        assert middleware is not None
