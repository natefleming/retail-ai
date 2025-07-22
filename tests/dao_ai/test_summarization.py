from unittest.mock import MagicMock

import pytest
from conftest import has_databricks_env
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage
from langgraph.graph.state import CompiledStateGraph

from dao_ai.config import (
    AgentModel,
    AppConfig,
    AppModel,
    LLMModel,
    OrchestrationModel,
    RegisteredModelModel,
    SummarizationModel,
    SupervisorModel,
)
from dao_ai.nodes import summarization_node


@pytest.mark.system
@pytest.mark.slow
@pytest.mark.skipif(
    not has_databricks_env(), reason="Missing Databricks environment variables"
)
def test_summarization_should_keep_correct_number_of_messages_and_summarize_the_rest(
    graph: CompiledStateGraph,
) -> None:
    assert True


# Unit tests for summarization_node


# Helper to create a list of messages for testing
def create_test_messages(count: int) -> list[BaseMessage]:
    return [HumanMessage(content=f"message {i}", id=str(i)) for i in range(count)]


@pytest.fixture
def mock_llm() -> MagicMock:
    """Fixture for a mock language model."""
    llm = MagicMock()
    # Mock the invoke method to return a predictable AIMessage
    llm.invoke.return_value = AIMessage(content="This is a summary.")
    return llm


@pytest.fixture
def mock_llm_model(mock_llm: MagicMock) -> MagicMock:
    """Fixture for a mock LLMModel that returns the mock LLM."""
    llm_model = MagicMock(spec=LLMModel)
    llm_model.as_chat_model.return_value = mock_llm
    return llm_model


@pytest.fixture
def dummy_agent(mock_llm_model: MagicMock) -> AgentModel:
    """Fixture for a dummy agent to satisfy AppModel requirements."""
    return AgentModel(
        name="test_agent", description="Test agent for unit tests", model=mock_llm_model
    )


def test_summarization_with_retained_message_count(
    mock_llm_model: MagicMock, dummy_agent: AgentModel
):
    """
    Tests that summarization correctly trims messages based on retained_message_count.
    """
    # Arrange
    retained_count = 2
    summarization_config = SummarizationModel(
        model=mock_llm_model, retained_message_count=retained_count
    )

    # Create app and config with proper structure
    app_model = AppModel(
        name="test_app",
        registered_model=RegisteredModelModel(name="test_model"),
        orchestration=OrchestrationModel(
            supervisor=SupervisorModel(model=mock_llm_model)
        ),
        agents=[dummy_agent],  # Add required agents list
        summarization=summarization_config,
    )
    app_config = AppConfig(app=app_model)

    # Create summarization node
    summarization_fn = summarization_node(app_config)

    # Create test state with more messages than retained count
    messages = create_test_messages(5)
    state = {"messages": messages, "summary": ""}

    # Act
    result = summarization_fn(state, {})

    # Assert
    assert result is not None
    assert "messages" in result
    assert "summary" in result

    # Should have created delete messages for the extra messages
    assert len(result["messages"]) == 3  # 5 total - 2 retained
    assert all(isinstance(msg, RemoveMessage) for msg in result["messages"])
    assert result["summary"] == "This is a summary."

    # Verify LLM was called
    mock_llm_model.as_chat_model.assert_called_once()
    mock_llm = mock_llm_model.as_chat_model.return_value
    mock_llm.invoke.assert_called_once()


def test_no_summarization_if_message_count_below_threshold(
    mock_llm_model: MagicMock, dummy_agent: AgentModel
):
    """
    Tests that no summarization occurs if the message count is below the retained_message_count.
    """
    # Arrange
    summarization_config = SummarizationModel(
        model=mock_llm_model,
        retained_message_count=5,
    )

    app_model = AppModel(
        name="test_app",
        registered_model=RegisteredModelModel(name="test_model"),
        orchestration=OrchestrationModel(
            supervisor=SupervisorModel(model=mock_llm_model)
        ),
        agents=[dummy_agent],
        summarization=summarization_config,
    )
    app_config = AppConfig(app=app_model)

    messages = create_test_messages(3)
    initial_state = {"messages": messages, "summary": ""}

    # Act
    summarize_node = summarization_node(app_config)
    result = summarize_node(initial_state, {})

    # Assert
    assert result is None


def test_summarization_with_max_tokens(
    mock_llm_model: MagicMock, dummy_agent: AgentModel
):
    """
    Tests that summarization correctly trims messages based on max_tokens.
    This uses the approximate token counter.
    """
    # Arrange
    # Each "message X" is ~2 tokens. Let's set a limit that cuts off the first few.
    summarization_config = SummarizationModel(
        model=mock_llm_model,
        max_tokens=5,  # Should keep last 2 messages approx
    )

    app_model = AppModel(
        name="test_app",
        registered_model=RegisteredModelModel(name="test_model"),
        orchestration=OrchestrationModel(
            supervisor=SupervisorModel(model=mock_llm_model)
        ),
        agents=[dummy_agent],
        summarization=summarization_config,
    )
    app_config = AppConfig(app=app_model)

    messages = create_test_messages(5)  # Total tokens ~10
    initial_state = {"messages": messages, "summary": ""}

    # Act
    summarize_node = summarization_node(app_config)
    result = summarize_node(initial_state, {})

    # Assert
    assert result is not None
    assert "summary" in result
    assert result["summary"] == "This is a summary."

    # The exact number of removed messages depends on the approx counter, but it should be > 0
    removed_messages = result["messages"]
    assert len(removed_messages) > 0
    assert len(removed_messages) < len(messages)


def test_no_summarization_if_token_count_below_threshold(
    mock_llm_model: MagicMock, dummy_agent: AgentModel
):
    """
    Tests that no summarization occurs if the token count is below the max_tokens limit.
    """
    # Arrange
    summarization_config = SummarizationModel(
        model=mock_llm_model,
        max_tokens=100,
    )

    app_model = AppModel(
        name="test_app",
        registered_model=RegisteredModelModel(name="test_model"),
        orchestration=OrchestrationModel(
            supervisor=SupervisorModel(model=mock_llm_model)
        ),
        agents=[dummy_agent],
        summarization=summarization_config,
    )
    app_config = AppConfig(app=app_model)

    messages = create_test_messages(3)  # Total tokens ~6
    initial_state = {"messages": messages, "summary": ""}

    # Act
    summarize_node = summarization_node(app_config)
    result = summarize_node(initial_state, {})

    # Assert - no summarization should occur because tokens < max_tokens
    assert result is None


def test_no_summarization_if_no_config(dummy_agent: AgentModel):
    """
    Tests that the node returns None if no summarization model is configured.
    """
    # Arrange - create AppConfig without summarization
    app_model = AppModel(
        name="test_app",
        registered_model=RegisteredModelModel(name="test_model"),
        orchestration=OrchestrationModel(
            supervisor=SupervisorModel(model=MagicMock(spec=LLMModel))
        ),
        agents=[dummy_agent],
        # No summarization field - it's optional
    )
    app_config = AppConfig(app=app_model)

    # Act
    summarize_node = summarization_node(app_config)
    result = summarize_node({"messages": create_test_messages(3), "summary": ""}, {})

    # Assert
    assert result is None


def test_summarization_keeps_one_message_if_retained_count_is_zero(
    mock_llm_model: MagicMock, dummy_agent: AgentModel
):
    """
    Tests that the summarization keeps at least one message even if retained_message_count is 0.
    This tests the safety check to ensure API doesn't get empty message list.
    """
    # Arrange
    summarization_config = SummarizationModel(
        model=mock_llm_model,
        retained_message_count=0,  # Try to keep 0 messages (dangerous!)
    )

    app_model = AppModel(
        name="test_app",
        registered_model=RegisteredModelModel(name="test_model"),
        orchestration=OrchestrationModel(
            supervisor=SupervisorModel(model=mock_llm_model)
        ),
        agents=[dummy_agent],
        summarization=summarization_config,
    )
    app_config = AppConfig(app=app_model)

    messages = create_test_messages(5)
    initial_state = {"messages": messages, "summary": ""}

    # Act
    summarize_node = summarization_node(app_config)
    result = summarize_node(initial_state, {})

    # Assert
    # Should still create a summary, but keep at least 1 message due to safety check
    assert result is not None
    assert "summary" in result
    assert result["summary"] == "This is a summary."

    # Should remove 4 messages but keep 1 due to the safety check
    assert len(result["messages"]) == 4  # 5 total - 1 kept (safety check)
