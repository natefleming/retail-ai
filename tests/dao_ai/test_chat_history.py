from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import ValidationError

from dao_ai.config import (
    AgentModel,
    AppModel,
    ChatHistoryModel,
    LLMModel,
    OrchestrationModel,
    RegisteredModelModel,
    SupervisorModel,
)
from dao_ai.nodes import summarization_node


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
def mock_llm_model():
    """Mock LLM model for testing."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="This is a test summary.")

    mock_llm_model = MagicMock(spec=LLMModel)
    mock_llm_model.as_chat_model.return_value = mock_llm
    return mock_llm_model


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
        chat_history=ChatHistoryModel(model=mock_llm_model, max_tokens=256),
    )


class TestSummarizationNode:
    """Test class for summarization node functionality."""

    def test_summarization_node_creation_with_default_params(self, base_app_model):
        """Test that summarization node can be created with default parameters."""
        node = summarization_node(base_app_model)
        assert node is not None

    def test_summarization_node_with_max_tokens_only(self, mock_llm_model):
        """Test summarization node with only max_tokens parameter."""
        app_model = AppModel(
            name="test_app",
            registered_model=RegisteredModelModel(name="test_model"),
            orchestration=OrchestrationModel(
                supervisor=SupervisorModel(model=mock_llm_model)
            ),
            agents=[AgentModel(name="test_agent", model=mock_llm_model)],
            chat_history=ChatHistoryModel(model=mock_llm_model, max_tokens=512),
        )

        node = summarization_node(app_model)
        assert node is not None

    def test_summarization_node_with_max_tokens_before_summary(self, mock_llm_model):
        """Test summarization node with max_tokens_before_summary parameter."""
        app_model = AppModel(
            name="test_app",
            registered_model=RegisteredModelModel(name="test_model"),
            orchestration=OrchestrationModel(
                supervisor=SupervisorModel(model=mock_llm_model)
            ),
            agents=[AgentModel(name="test_agent", model=mock_llm_model)],
            chat_history=ChatHistoryModel(
                model=mock_llm_model, max_tokens=256, max_tokens_before_summary=1000
            ),
        )

        node = summarization_node(app_model)
        assert node is not None

    def test_summarization_node_with_max_messages_before_summary(self, mock_llm_model):
        """Test summarization node with max_messages_before_summary parameter."""
        app_model = AppModel(
            name="test_app",
            registered_model=RegisteredModelModel(name="test_model"),
            orchestration=OrchestrationModel(
                supervisor=SupervisorModel(model=mock_llm_model)
            ),
            agents=[AgentModel(name="test_agent", model=mock_llm_model)],
            chat_history=ChatHistoryModel(
                model=mock_llm_model, max_tokens=256, max_messages_before_summary=10
            ),
        )

        node = summarization_node(app_model)
        assert node is not None

    def test_summarization_node_with_max_summary_tokens(self, mock_llm_model):
        """Test summarization node with max_summary_tokens parameter."""
        app_model = AppModel(
            name="test_app",
            registered_model=RegisteredModelModel(name="test_model"),
            orchestration=OrchestrationModel(
                supervisor=SupervisorModel(model=mock_llm_model)
            ),
            agents=[AgentModel(name="test_agent", model=mock_llm_model)],
            chat_history=ChatHistoryModel(
                model=mock_llm_model,
                max_tokens=256,
                max_tokens_before_summary=1000,
                max_summary_tokens=100,
            ),
        )

        node = summarization_node(app_model)
        assert node is not None

    def test_summarization_node_with_all_parameters(self, mock_llm_model):
        """Test summarization node with all parameters configured."""
        app_model = AppModel(
            name="test_app",
            registered_model=RegisteredModelModel(name="test_model"),
            orchestration=OrchestrationModel(
                supervisor=SupervisorModel(model=mock_llm_model)
            ),
            agents=[AgentModel(name="test_agent", model=mock_llm_model)],
            chat_history=ChatHistoryModel(
                model=mock_llm_model,
                max_tokens=512,
                max_tokens_before_summary=2000,
                max_messages_before_summary=15,
                max_summary_tokens=150,
            ),
        )

        node = summarization_node(app_model)
        assert node is not None

    @patch("dao_ai.nodes.SummarizationNode")
    def test_summarization_node_uses_token_counter_for_tokens(
        self, mock_summarization_node, mock_llm_model
    ):
        """Test that token counter is used when max_tokens_before_summary is set."""
        app_model = AppModel(
            name="test_app",
            registered_model=RegisteredModelModel(name="test_model"),
            orchestration=OrchestrationModel(
                supervisor=SupervisorModel(model=mock_llm_model)
            ),
            agents=[AgentModel(name="test_agent", model=mock_llm_model)],
            chat_history=ChatHistoryModel(
                model=mock_llm_model, max_tokens=256, max_tokens_before_summary=1000
            ),
        )

        summarization_node(app_model)

        # Verify SummarizationNode was called with token counter
        mock_summarization_node.assert_called_once()
        call_kwargs = mock_summarization_node.call_args[1]

        # When max_tokens_before_summary is set, token_counter should be count_tokens_approximately
        from langchain_core.messages.utils import count_tokens_approximately

        assert call_kwargs["token_counter"] == count_tokens_approximately

    @patch("dao_ai.nodes.SummarizationNode")
    def test_summarization_node_uses_len_counter_for_messages(
        self, mock_summarization_node, mock_llm_model
    ):
        """Test that len counter is used when max_tokens_before_summary is not set."""
        app_model = AppModel(
            name="test_app",
            registered_model=RegisteredModelModel(name="test_model"),
            orchestration=OrchestrationModel(
                supervisor=SupervisorModel(model=mock_llm_model)
            ),
            agents=[AgentModel(name="test_agent", model=mock_llm_model)],
            chat_history=ChatHistoryModel(
                model=mock_llm_model, max_tokens=256, max_messages_before_summary=10
            ),
        )

        summarization_node(app_model)

        # Verify SummarizationNode was called with len counter
        mock_summarization_node.assert_called_once()
        call_kwargs = mock_summarization_node.call_args[1]

        # When max_tokens_before_summary is not set, token_counter should be len
        assert call_kwargs["token_counter"] == len

    @patch("dao_ai.nodes.SummarizationNode")
    def test_summarization_node_parameters_passed_correctly(
        self, mock_summarization_node, mock_llm_model
    ):
        """Test that all parameters are passed correctly to SummarizationNode."""
        max_tokens = 512
        max_tokens_before_summary = 2000
        max_messages_before_summary = 15
        max_summary_tokens = 150

        app_model = AppModel(
            name="test_app",
            registered_model=RegisteredModelModel(name="test_model"),
            orchestration=OrchestrationModel(
                supervisor=SupervisorModel(model=mock_llm_model)
            ),
            agents=[AgentModel(name="test_agent", model=mock_llm_model)],
            chat_history=ChatHistoryModel(
                model=mock_llm_model,
                max_tokens=max_tokens,
                max_tokens_before_summary=max_tokens_before_summary,
                max_messages_before_summary=max_messages_before_summary,
                max_summary_tokens=max_summary_tokens,
            ),
        )

        summarization_node(app_model)

        # Verify SummarizationNode was called with correct parameters
        mock_summarization_node.assert_called_once()
        call_kwargs = mock_summarization_node.call_args[1]

        assert call_kwargs["max_tokens"] == max_tokens
        assert call_kwargs["max_tokens_before_summary"] == max_tokens_before_summary
        assert call_kwargs["max_summary_tokens"] == max_summary_tokens
        assert call_kwargs["input_messages_key"] == "messages"
        assert call_kwargs["output_messages_key"] == "summarized_messages"

    @patch("dao_ai.nodes.SummarizationNode")
    def test_summarization_node_prefers_tokens_over_messages(
        self, mock_summarization_node, mock_llm_model
    ):
        """Test that max_tokens_before_summary takes precedence over max_messages_before_summary."""
        max_tokens_before_summary = 2000
        max_messages_before_summary = 15

        app_model = AppModel(
            name="test_app",
            registered_model=RegisteredModelModel(name="test_model"),
            orchestration=OrchestrationModel(
                supervisor=SupervisorModel(model=mock_llm_model)
            ),
            agents=[AgentModel(name="test_agent", model=mock_llm_model)],
            chat_history=ChatHistoryModel(
                model=mock_llm_model,
                max_tokens=256,
                max_tokens_before_summary=max_tokens_before_summary,
                max_messages_before_summary=max_messages_before_summary,
            ),
        )

        summarization_node(app_model)

        # Verify that max_tokens_before_summary is used when both are present
        mock_summarization_node.assert_called_once()
        call_kwargs = mock_summarization_node.call_args[1]

        assert call_kwargs["max_tokens_before_summary"] == max_tokens_before_summary

    @patch("dao_ai.nodes.SummarizationNode")
    def test_summarization_node_falls_back_to_messages(
        self, mock_summarization_node, mock_llm_model
    ):
        """Test that max_messages_before_summary is used when max_tokens_before_summary is None."""
        max_messages_before_summary = 15

        app_model = AppModel(
            name="test_app",
            registered_model=RegisteredModelModel(name="test_model"),
            orchestration=OrchestrationModel(
                supervisor=SupervisorModel(model=mock_llm_model)
            ),
            agents=[AgentModel(name="test_agent", model=mock_llm_model)],
            chat_history=ChatHistoryModel(
                model=mock_llm_model,
                max_tokens=256,
                max_tokens_before_summary=None,
                max_messages_before_summary=max_messages_before_summary,
            ),
        )

        summarization_node(app_model)

        # Verify that max_messages_before_summary is used when max_tokens_before_summary is None
        mock_summarization_node.assert_called_once()
        call_kwargs = mock_summarization_node.call_args[1]

        assert call_kwargs["max_tokens_before_summary"] == max_messages_before_summary

    def test_summarization_node_validates_required_model(self):
        """Test that summarization node requires a model parameter."""
        with pytest.raises(ValueError):
            AppModel(
                name="test_app",
                registered_model=RegisteredModelModel(name="test_model"),
                orchestration=OrchestrationModel(
                    supervisor=SupervisorModel(model=mock_llm_model)
                ),
                agents=[AgentModel(name="test_agent", model=mock_llm_model)],
                chat_history=ChatHistoryModel(
                    # model is missing - should raise validation error
                    max_tokens=256
                ),
            )

    def test_chat_history_model_default_values(self, mock_llm_model):
        """Test that ChatHistoryModel has correct default values."""
        chat_history = ChatHistoryModel(model=mock_llm_model)

        assert chat_history.max_tokens == 256
        assert chat_history.max_tokens_before_summary is None
        assert chat_history.max_messages_before_summary is None
        assert chat_history.max_summary_tokens == 255

    def test_chat_history_model_custom_values(self, mock_llm_model):
        """Test that ChatHistoryModel accepts custom values."""
        max_tokens = 1024
        max_tokens_before_summary = 5000
        max_messages_before_summary = 25
        max_summary_tokens = 200

        chat_history = ChatHistoryModel(
            model=mock_llm_model,
            max_tokens=max_tokens,
            max_tokens_before_summary=max_tokens_before_summary,
            max_messages_before_summary=max_messages_before_summary,
            max_summary_tokens=max_summary_tokens,
        )

        assert chat_history.max_tokens == max_tokens
        assert chat_history.max_tokens_before_summary == max_tokens_before_summary
        assert chat_history.max_messages_before_summary == max_messages_before_summary
        assert chat_history.max_summary_tokens == max_summary_tokens

    @patch("dao_ai.nodes.logger")
    def test_summarization_node_logs_parameters(self, mock_logger, mock_llm_model):
        """Test that summarization node logs its parameters during creation."""
        max_tokens = 512
        max_tokens_before_summary = 2000
        max_messages_before_summary = 15
        max_summary_tokens = 150

        app_model = AppModel(
            name="test_app",
            registered_model=RegisteredModelModel(name="test_model"),
            orchestration=OrchestrationModel(
                supervisor=SupervisorModel(model=mock_llm_model)
            ),
            agents=[AgentModel(name="test_agent", model=mock_llm_model)],
            chat_history=ChatHistoryModel(
                model=mock_llm_model,
                max_tokens=max_tokens,
                max_tokens_before_summary=max_tokens_before_summary,
                max_messages_before_summary=max_messages_before_summary,
                max_summary_tokens=max_summary_tokens,
            ),
        )

        summarization_node(app_model)

        # Verify that debug logging was called with the parameters
        mock_logger.debug.assert_called_with(
            f"Creating summarization node with max_tokens: {max_tokens}, "
            f"max_tokens_before_summary: {max_tokens_before_summary}, "
            f"max_messages_before_summary: {max_messages_before_summary}, "
            f"max_summary_tokens: {max_summary_tokens}"
        )

    def test_summarization_node_with_zero_max_tokens(self, mock_llm_model):
        """Test edge case with zero max_tokens should raise validation error."""
        with pytest.raises(
            ValidationError, match="max_summary_tokens .* must be less than max_tokens"
        ):
            ChatHistoryModel(model=mock_llm_model, max_tokens=0)

    def test_chat_history_model_validator(self, mock_llm_model):
        """Test the validator that ensures max_summary_tokens < max_tokens."""
        # Valid case: max_summary_tokens < max_tokens
        valid_chat_history = ChatHistoryModel(
            model=mock_llm_model, max_tokens=1000, max_summary_tokens=500
        )
        assert valid_chat_history.max_tokens == 1000
        assert valid_chat_history.max_summary_tokens == 500

        # Invalid case: max_summary_tokens >= max_tokens
        with pytest.raises(
            ValidationError, match="max_summary_tokens .* must be less than max_tokens"
        ):
            ChatHistoryModel(
                model=mock_llm_model,
                max_tokens=256,
                max_summary_tokens=256,  # Equal, should fail
            )

        with pytest.raises(
            ValidationError, match="max_summary_tokens .* must be less than max_tokens"
        ):
            ChatHistoryModel(
                model=mock_llm_model,
                max_tokens=256,
                max_summary_tokens=300,  # Greater, should fail
            )

    def test_summarization_node_with_large_values(self, mock_llm_model):
        """Test summarization node with large parameter values."""
        app_model = AppModel(
            name="test_app",
            registered_model=RegisteredModelModel(name="test_model"),
            orchestration=OrchestrationModel(
                supervisor=SupervisorModel(model=mock_llm_model)
            ),
            agents=[AgentModel(name="test_agent", model=mock_llm_model)],
            chat_history=ChatHistoryModel(
                model=mock_llm_model,
                max_tokens=10000,
                max_tokens_before_summary=50000,
                max_messages_before_summary=1000,
                max_summary_tokens=5000,
            ),
        )

        node = summarization_node(app_model)
        assert node is not None

    @patch("dao_ai.nodes.SummarizationNode")
    def test_summarization_node_model_conversion(
        self, mock_summarization_node, mock_llm_model
    ):
        """Test that the LLM model is properly converted to chat model."""
        app_model = AppModel(
            name="test_app",
            registered_model=RegisteredModelModel(name="test_model"),
            orchestration=OrchestrationModel(
                supervisor=SupervisorModel(model=mock_llm_model)
            ),
            agents=[AgentModel(name="test_agent", model=mock_llm_model)],
            chat_history=ChatHistoryModel(model=mock_llm_model, max_tokens=256),
        )

        summarization_node(app_model)

        # Verify that as_chat_model() was called on the LLM model
        mock_llm_model.as_chat_model.assert_called_once()

        # Verify that the converted model was passed to SummarizationNode
        mock_summarization_node.assert_called_once()
        call_kwargs = mock_summarization_node.call_args[1]
        assert call_kwargs["model"] == mock_llm_model.as_chat_model.return_value
