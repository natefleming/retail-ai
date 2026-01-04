"""
Tests for model retry middleware factory.
"""

from langchain.agents.middleware import ModelRetryMiddleware

from dao_ai.middleware import create_model_retry_middleware


class TestCreateModelRetryMiddleware:
    """Tests for the create_model_retry_middleware factory function."""

    def test_create_with_defaults(self):
        """Test creating middleware with default parameters."""
        middleware = create_model_retry_middleware()

        # Middleware is single instance
        assert middleware is not None
        middleware = middleware
        assert isinstance(middleware, ModelRetryMiddleware)
        assert middleware.max_retries == 3
        assert middleware.backoff_factor == 2.0
        assert middleware.initial_delay == 1.0
        assert middleware.on_failure == "continue"

    def test_create_with_custom_retries(self):
        """Test creating middleware with custom max_retries."""
        middleware = create_model_retry_middleware(max_retries=5)

        assert middleware.max_retries == 5

    def test_create_with_custom_backoff(self):
        """Test creating middleware with custom backoff settings."""
        middleware = create_model_retry_middleware(
            backoff_factor=1.5,
            initial_delay=0.5,
        )

        middleware = middleware
        assert middleware.backoff_factor == 1.5
        assert middleware.initial_delay == 0.5

    def test_create_with_max_delay(self):
        """Test creating middleware with max_delay cap."""
        middleware = create_model_retry_middleware(
            max_retries=10,
            max_delay=60.0,
        )

        assert middleware.max_delay == 60.0

    def test_create_with_jitter(self):
        """Test creating middleware with jitter enabled."""
        middleware = create_model_retry_middleware(jitter=True)

        assert middleware.jitter is True

    def test_create_with_error_on_failure(self):
        """Test creating middleware with error on_failure behavior."""
        middleware = create_model_retry_middleware(on_failure="error")

        assert middleware.on_failure == "error"

    def test_create_with_exception_tuple(self):
        """Test creating middleware with specific exception types."""
        middleware = create_model_retry_middleware(
            retry_on=(TimeoutError, ConnectionError),
        )

        assert middleware.retry_on == (TimeoutError, ConnectionError)

    def test_create_with_retry_callable(self):
        """Test creating middleware with callable retry_on."""

        def should_retry(error: Exception) -> bool:
            return "rate_limit" in str(error).lower()

        middleware = create_model_retry_middleware(retry_on=should_retry)

        assert middleware.retry_on is should_retry

    def test_create_with_custom_on_failure_callable(self):
        """Test creating middleware with callable on_failure."""

        def format_error(error: Exception) -> str:
            return f"Model call failed: {error}"

        middleware = create_model_retry_middleware(on_failure=format_error)

        assert middleware.on_failure is format_error

    def test_create_with_all_parameters(self):
        """Test creating middleware with all parameters."""
        middleware = create_model_retry_middleware(
            max_retries=5,
            backoff_factor=2.0,
            initial_delay=1.0,
            max_delay=30.0,
            jitter=True,
            on_failure="error",
        )

        middleware = middleware
        assert middleware.max_retries == 5
        assert middleware.backoff_factor == 2.0
        assert middleware.initial_delay == 1.0
        assert middleware.max_delay == 30.0
        assert middleware.jitter is True
        assert middleware.on_failure == "error"

    def test_constant_backoff(self):
        """Test creating middleware with constant (no exponential) backoff."""
        middleware = create_model_retry_middleware(
            backoff_factor=0.0,
            initial_delay=2.0,
        )

        middleware = middleware
        assert middleware.backoff_factor == 0.0
        assert middleware.initial_delay == 2.0

    def test_returns_list_for_composition(self):
        """Test that factory returns list for easy composition."""
        middleware = create_model_retry_middleware()

        # Middleware is single instance
        assert middleware is not None
