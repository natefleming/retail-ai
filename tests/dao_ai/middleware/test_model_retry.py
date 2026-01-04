"""
Tests for model retry middleware factory.
"""

from langchain.agents.middleware import ModelRetryMiddleware

from dao_ai.middleware import create_model_retry_middleware


class TestCreateModelRetryMiddleware:
    """Tests for the create_model_retry_middleware factory function."""

    def test_create_with_defaults(self):
        """Test creating middleware with default parameters."""
        middlewares = create_model_retry_middleware()

        assert isinstance(middlewares, list)
        assert len(middlewares) == 1
        middleware = middlewares[0]
        assert isinstance(middleware, ModelRetryMiddleware)
        assert middleware.max_retries == 3
        assert middleware.backoff_factor == 2.0
        assert middleware.initial_delay == 1.0
        assert middleware.on_failure == "continue"

    def test_create_with_custom_retries(self):
        """Test creating middleware with custom max_retries."""
        middlewares = create_model_retry_middleware(max_retries=5)

        assert middlewares[0].max_retries == 5

    def test_create_with_custom_backoff(self):
        """Test creating middleware with custom backoff settings."""
        middlewares = create_model_retry_middleware(
            backoff_factor=1.5,
            initial_delay=0.5,
        )

        middleware = middlewares[0]
        assert middleware.backoff_factor == 1.5
        assert middleware.initial_delay == 0.5

    def test_create_with_max_delay(self):
        """Test creating middleware with max_delay cap."""
        middlewares = create_model_retry_middleware(
            max_retries=10,
            max_delay=60.0,
        )

        assert middlewares[0].max_delay == 60.0

    def test_create_with_jitter(self):
        """Test creating middleware with jitter enabled."""
        middlewares = create_model_retry_middleware(jitter=True)

        assert middlewares[0].jitter is True

    def test_create_with_error_on_failure(self):
        """Test creating middleware with error on_failure behavior."""
        middlewares = create_model_retry_middleware(on_failure="error")

        assert middlewares[0].on_failure == "error"

    def test_create_with_exception_tuple(self):
        """Test creating middleware with specific exception types."""
        middlewares = create_model_retry_middleware(
            retry_on=(TimeoutError, ConnectionError),
        )

        assert middlewares[0].retry_on == (TimeoutError, ConnectionError)

    def test_create_with_retry_callable(self):
        """Test creating middleware with callable retry_on."""

        def should_retry(error: Exception) -> bool:
            return "rate_limit" in str(error).lower()

        middlewares = create_model_retry_middleware(retry_on=should_retry)

        assert middlewares[0].retry_on is should_retry

    def test_create_with_custom_on_failure_callable(self):
        """Test creating middleware with callable on_failure."""

        def format_error(error: Exception) -> str:
            return f"Model call failed: {error}"

        middlewares = create_model_retry_middleware(on_failure=format_error)

        assert middlewares[0].on_failure is format_error

    def test_create_with_all_parameters(self):
        """Test creating middleware with all parameters."""
        middlewares = create_model_retry_middleware(
            max_retries=5,
            backoff_factor=2.0,
            initial_delay=1.0,
            max_delay=30.0,
            jitter=True,
            on_failure="error",
        )

        middleware = middlewares[0]
        assert middleware.max_retries == 5
        assert middleware.backoff_factor == 2.0
        assert middleware.initial_delay == 1.0
        assert middleware.max_delay == 30.0
        assert middleware.jitter is True
        assert middleware.on_failure == "error"

    def test_constant_backoff(self):
        """Test creating middleware with constant (no exponential) backoff."""
        middlewares = create_model_retry_middleware(
            backoff_factor=0.0,
            initial_delay=2.0,
        )

        middleware = middlewares[0]
        assert middleware.backoff_factor == 0.0
        assert middleware.initial_delay == 2.0

    def test_returns_list_for_composition(self):
        """Test that factory returns list for easy composition."""
        middlewares = create_model_retry_middleware()

        assert isinstance(middlewares, list)
        assert len(middlewares) == 1
