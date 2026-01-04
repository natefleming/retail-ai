"""
Tests for chat history and summarization middleware functionality.

These tests verify that the SummarizationMiddleware is properly configured
from DAO AI's ChatHistoryModel configuration.
"""

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import ValidationError

from dao_ai.config import (
    AgentModel,
    AppModel,
    ChatHistoryModel,
    OrchestrationModel,
    RegisteredModelModel,
    SupervisorModel,
)
from dao_ai.middleware.summarization import create_summarization_middleware


# Helper functions
def create_test_messages(count: int, prefix: str = "Message") -> list[BaseMessage]:
    """Create a list of test messages."""
    messages = []
    for i in range(count):
        if i % 2 == 0:
            messages.append(HumanMessage(content=f"{prefix} {i}", id=f"human-{i}"))
        else:
            messages.append(AIMessage(content=f"{prefix} {i}", id=f"ai-{i}"))
    return messages


@pytest.fixture
def base_app_model(mock_llm_model):
    """Base app model for testing."""
    return AppModel(
        name="test_app",
        registered_model=RegisteredModelModel(name="test_model"),
        orchestration=OrchestrationModel(
            supervisor=SupervisorModel(model=mock_llm_model)
        ),
        agents=[AgentModel(name="test_agent", model=mock_llm_model)],
        chat_history=ChatHistoryModel(model=mock_llm_model, max_tokens=2048),
    )


class TestSummarizationMiddleware:
    """Test class for summarization middleware functionality."""

    def test_summarization_middleware_creation_with_default_params(
        self, base_app_model
    ):
        """Test that summarization middleware can be created with default parameters."""
        middleware = create_summarization_middleware(base_app_model.chat_history)
        # Middleware is single instance
        assert middleware is not None

    def test_summarization_middleware_with_max_tokens_only(self, mock_llm_model):
        """Test summarization middleware with only max_tokens parameter."""
        chat_history = ChatHistoryModel(model=mock_llm_model, max_tokens=512)
        middleware = create_summarization_middleware(chat_history)
        # Middleware is single instance
        assert middleware is not None

    def test_summarization_middleware_with_max_tokens_before_summary(
        self, mock_llm_model
    ):
        """Test summarization middleware with max_tokens_before_summary parameter."""
        chat_history = ChatHistoryModel(
            model=mock_llm_model, max_tokens=2048, max_tokens_before_summary=6000
        )
        middleware = create_summarization_middleware(chat_history)
        # Middleware is single instance
        assert middleware is not None

    def test_summarization_middleware_with_max_messages_before_summary(
        self, mock_llm_model
    ):
        """Test summarization middleware with max_messages_before_summary parameter."""
        chat_history = ChatHistoryModel(
            model=mock_llm_model, max_tokens=2048, max_messages_before_summary=10
        )
        middleware = create_summarization_middleware(chat_history)
        # Middleware is single instance
        assert middleware is not None

    def test_summarization_middleware_with_all_parameters(self, mock_llm_model):
        """Test summarization middleware with all parameters configured."""
        chat_history = ChatHistoryModel(
            model=mock_llm_model,
            max_tokens=2048,
            max_tokens_before_summary=6000,
            max_messages_before_summary=15,
        )
        middleware = create_summarization_middleware(chat_history)
        # Middleware is single instance
        assert middleware is not None

    def test_summarization_middleware_trigger_tokens(self, mock_llm_model):
        """Test that trigger uses tokens when max_tokens_before_summary is set."""
        chat_history = ChatHistoryModel(
            model=mock_llm_model, max_tokens=2048, max_tokens_before_summary=6000
        )

        middleware = create_summarization_middleware(chat_history)
        middleware = middleware

        # Verify middleware has correct trigger
        assert middleware.trigger == ("tokens", 6000)

    def test_summarization_middleware_trigger_messages(self, mock_llm_model):
        """Test that trigger uses messages when max_messages_before_summary is set."""
        chat_history = ChatHistoryModel(
            model=mock_llm_model, max_tokens=2048, max_messages_before_summary=10
        )

        middleware = create_summarization_middleware(chat_history)
        middleware = middleware

        # Verify middleware has correct trigger
        assert middleware.trigger == ("messages", 10)

    def test_summarization_middleware_prefers_tokens_over_messages(
        self, mock_llm_model
    ):
        """Test that max_tokens_before_summary takes precedence over max_messages_before_summary."""
        max_tokens_before_summary = 6000
        max_messages_before_summary = 15

        chat_history = ChatHistoryModel(
            model=mock_llm_model,
            max_tokens=2048,
            max_tokens_before_summary=max_tokens_before_summary,
            max_messages_before_summary=max_messages_before_summary,
        )

        middleware = create_summarization_middleware(chat_history)
        middleware = middleware

        # Verify that max_tokens_before_summary is used when both are present
        assert middleware.trigger == ("tokens", max_tokens_before_summary)

    def test_summarization_middleware_keep_parameter(self, mock_llm_model):
        """Test that keep parameter is set correctly."""
        max_tokens = 2048

        chat_history = ChatHistoryModel(
            model=mock_llm_model,
            max_tokens=max_tokens,
            max_tokens_before_summary=6000,
        )

        middleware = create_summarization_middleware(chat_history)
        middleware = middleware

        # Verify keep is set to max_tokens
        assert middleware.keep == ("tokens", max_tokens)

    def test_chat_history_model_default_values(self, mock_llm_model):
        """Test that ChatHistoryModel has correct default values."""
        chat_history = ChatHistoryModel(model=mock_llm_model)

        assert chat_history.max_tokens == 2048
        assert chat_history.max_tokens_before_summary is None
        assert chat_history.max_messages_before_summary is None

    def test_chat_history_model_custom_values(self, mock_llm_model):
        """Test that ChatHistoryModel accepts custom values."""
        max_tokens = 4096
        max_tokens_before_summary = 8000
        max_messages_before_summary = 25

        chat_history = ChatHistoryModel(
            model=mock_llm_model,
            max_tokens=max_tokens,
            max_tokens_before_summary=max_tokens_before_summary,
            max_messages_before_summary=max_messages_before_summary,
        )

        assert chat_history.max_tokens == max_tokens
        assert chat_history.max_tokens_before_summary == max_tokens_before_summary
        assert chat_history.max_messages_before_summary == max_messages_before_summary

    @patch("dao_ai.middleware.summarization.logger")
    def test_summarization_middleware_logs_parameters(
        self, mock_logger, mock_llm_model
    ):
        """Test that summarization middleware logs its parameters during creation."""
        max_tokens = 2048
        max_tokens_before_summary = 6000
        max_messages_before_summary = 15

        chat_history = ChatHistoryModel(
            model=mock_llm_model,
            max_tokens=max_tokens,
            max_tokens_before_summary=max_tokens_before_summary,
            max_messages_before_summary=max_messages_before_summary,
        )

        create_summarization_middleware(chat_history)

        # Verify that debug logging was called with the parameters (structured logging)
        mock_logger.debug.assert_called_with(
            "Creating summarization middleware",
            max_tokens=max_tokens,
            max_tokens_before_summary=max_tokens_before_summary,
            max_messages_before_summary=max_messages_before_summary,
        )

    def test_chat_history_model_rejects_zero_max_tokens(self, mock_llm_model):
        """Test edge case with zero max_tokens should raise validation error."""
        with pytest.raises(ValidationError, match="greater than 0"):
            ChatHistoryModel(model=mock_llm_model, max_tokens=0)

    def test_chat_history_model_rejects_negative_max_tokens(self, mock_llm_model):
        """Test that negative max_tokens raises validation error."""
        with pytest.raises(ValidationError, match="greater than 0"):
            ChatHistoryModel(model=mock_llm_model, max_tokens=-100)

    def test_chat_history_model_rejects_zero_max_tokens_before_summary(
        self, mock_llm_model
    ):
        """Test that zero max_tokens_before_summary raises validation error."""
        with pytest.raises(ValidationError, match="greater than 0"):
            ChatHistoryModel(
                model=mock_llm_model, max_tokens=2048, max_tokens_before_summary=0
            )

    def test_chat_history_model_rejects_zero_max_messages_before_summary(
        self, mock_llm_model
    ):
        """Test that zero max_messages_before_summary raises validation error."""
        with pytest.raises(ValidationError, match="greater than 0"):
            ChatHistoryModel(
                model=mock_llm_model, max_tokens=2048, max_messages_before_summary=0
            )

    def test_summarization_middleware_with_large_values(self, mock_llm_model):
        """Test summarization middleware with large parameter values."""
        chat_history = ChatHistoryModel(
            model=mock_llm_model,
            max_tokens=10000,
            max_tokens_before_summary=50000,
            max_messages_before_summary=1000,
        )

        middleware = create_summarization_middleware(chat_history)
        # Middleware is single instance
        assert middleware is not None

    def test_summarization_middleware_model_conversion(self, mock_llm_model):
        """Test that the LLM model is properly converted to chat model."""
        chat_history = ChatHistoryModel(model=mock_llm_model, max_tokens=2048)

        middleware = create_summarization_middleware(chat_history)
        middleware = middleware

        # Verify that as_chat_model() was called on the LLM model
        mock_llm_model.as_chat_model.assert_called_once()

        # Verify the middleware has the converted model
        assert middleware.model == mock_llm_model.as_chat_model.return_value

    def test_summarization_middleware_default_trigger_fallback(self, mock_llm_model):
        """Test that trigger falls back to max_tokens * 10 when not specified."""
        max_tokens = 2048

        chat_history = ChatHistoryModel(model=mock_llm_model, max_tokens=max_tokens)

        middleware = create_summarization_middleware(chat_history)
        middleware = middleware

        # Verify that trigger defaults to max_tokens * 10
        assert middleware.trigger == ("tokens", max_tokens * 10)
