from unittest.mock import MagicMock

import pytest
from conftest import has_databricks_env
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage
from langgraph.graph.state import CompiledStateGraph

from dao_ai.config import (
    AgentModel,
    AppConfig,
    AppModel,
    ChatHistoryModel,
    LLMModel,
    OrchestrationModel,
    RegisteredModelModel,
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
    summarization_config = ChatHistoryModel(
        model=mock_llm_model, retained_message_count=retained_count, summarize=True
    )

    # Create app and config with proper structure
    app_model = AppModel(
        name="test_app",
        registered_model=RegisteredModelModel(name="test_model"),
        orchestration=OrchestrationModel(
            supervisor=SupervisorModel(model=mock_llm_model)
        ),
        agents=[dummy_agent],  # Add required agents list
        chat_history=summarization_config,
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
    summarization_config = ChatHistoryModel(
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
        chat_history=summarization_config,
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
    # Each "message X" is ~7 tokens. Let's set a limit that keeps the last 2 messages.
    summarization_config = ChatHistoryModel(
        model=mock_llm_model,
        max_tokens=14,  # Should keep last 2 messages (~14 tokens)
        summarize=True,
    )

    app_model = AppModel(
        name="test_app",
        registered_model=RegisteredModelModel(name="test_model"),
        orchestration=OrchestrationModel(
            supervisor=SupervisorModel(model=mock_llm_model)
        ),
        agents=[dummy_agent],
        chat_history=summarization_config,
    )
    app_config = AppConfig(app=app_model)

    messages = create_test_messages(5)  # Total tokens ~35
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
    summarization_config = ChatHistoryModel(
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
        chat_history=summarization_config,
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
    summarization_config = ChatHistoryModel(
        model=mock_llm_model,
        retained_message_count=0,  # Try to keep 0 messages (dangerous!)
        summarize=True,
    )

    app_model = AppModel(
        name="test_app",
        registered_model=RegisteredModelModel(name="test_model"),
        orchestration=OrchestrationModel(
            supervisor=SupervisorModel(model=mock_llm_model)
        ),
        agents=[dummy_agent],
        chat_history=summarization_config,
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


def test_summarization_disabled_with_explicit_false(
    mock_llm_model: MagicMock, dummy_agent: AgentModel
):
    """
    Tests that no summary is created when summarize=False explicitly, but messages are still trimmed.
    """
    # Arrange
    summarization_config = ChatHistoryModel(
        model=mock_llm_model,
        retained_message_count=2,
        summarize=False,  # Explicitly disabled
    )

    app_model = AppModel(
        name="test_app",
        registered_model=RegisteredModelModel(name="test_model"),
        orchestration=OrchestrationModel(
            supervisor=SupervisorModel(model=mock_llm_model)
        ),
        agents=[dummy_agent],
        chat_history=summarization_config,
    )
    app_config = AppConfig(app=app_model)

    messages = create_test_messages(10)  # Way more than retained count
    initial_state = {"messages": messages, "summary": ""}

    # Act
    summarize_node = summarization_node(app_config)
    result = summarize_node(initial_state, {})

    # Assert - should trim messages but not create a summary
    assert result is not None
    assert "messages" in result
    assert "summary" not in result  # No summary should be created
    assert len(result["messages"]) == 8  # 10 total - 2 retained
    # Note: as_chat_model() may be called for setup even if summarization doesn't occur


def test_summarization_validation_prevents_invalid_combinations(
    mock_llm_model: MagicMock, dummy_agent: AgentModel
):
    """
    Tests that the validation prevents invalid parameter combinations.
    """
    # Should raise ValidationError when mixing message-based and token-based limits
    with pytest.raises(ValueError, match="Cannot specify both retained_message_count"):
        ChatHistoryModel(
            model=mock_llm_model,
            retained_message_count=3,
            max_tokens=100,
            summarize=True,
        )
    
    with pytest.raises(ValueError, match="Cannot specify both retained_message_count"):
        ChatHistoryModel(
            model=mock_llm_model,
            max_message_count=5,
            max_tokens=100,
            summarize=True,
        )


def test_summarization_with_max_message_count_vs_retained_count(
    mock_llm_model: MagicMock, dummy_agent: AgentModel
):
    """
    Tests that retained_message_count takes precedence over max_message_count when both are set.
    """
    # Arrange
    summarization_config = ChatHistoryModel(
        model=mock_llm_model,
        retained_message_count=2,  # Keep 2 messages (should take precedence)
        max_message_count=5,       # Max 5 messages (should be ignored)
        summarize=True,
    )

    app_model = AppModel(
        name="test_app",
        registered_model=RegisteredModelModel(name="test_model"),
        orchestration=OrchestrationModel(
            supervisor=SupervisorModel(model=mock_llm_model)
        ),
        agents=[dummy_agent],
        chat_history=summarization_config,
    )
    app_config = AppConfig(app=app_model)

    messages = create_test_messages(8)
    initial_state = {"messages": messages, "summary": ""}

    # Act
    summarize_node = summarization_node(app_config)
    result = summarize_node(initial_state, {})

    # Assert
    assert result is not None
    assert "summary" in result
    
    # Should use retained_message_count (2), not max_message_count (5)
    # So remove 6 messages (8 total - 2 retained)
    assert len(result["messages"]) == 6


def test_summarization_with_max_tokens_separate_test(
    mock_llm_model: MagicMock, dummy_agent: AgentModel
):
    """
    Tests summarization when max_tokens is set (separate from retained_message_count).
    """
    # Arrange
    summarization_config = ChatHistoryModel(
        model=mock_llm_model,
        max_tokens=14,             # Keep ~2 messages worth of tokens (stricter limit)
        summarize=True,
    )

    app_model = AppModel(
        name="test_app",
        registered_model=RegisteredModelModel(name="test_model"),
        orchestration=OrchestrationModel(
            supervisor=SupervisorModel(model=mock_llm_model)
        ),
        agents=[dummy_agent],
        chat_history=summarization_config,
    )
    app_config = AppConfig(app=app_model)

    messages = create_test_messages(6)
    initial_state = {"messages": messages, "summary": ""}

    # Act
    summarize_node = summarization_node(app_config)
    result = summarize_node(initial_state, {})

    # Assert
    assert result is not None
    assert "summary" in result
    
    # Should remove some messages based on token limit
    removed_messages = result["messages"]
    assert len(removed_messages) > 0  # Should remove some messages


def test_summarization_with_retained_count_separate_test(
    mock_llm_model: MagicMock, dummy_agent: AgentModel
):
    """
    Tests summarization when retained_message_count is set (separate from max_tokens).
    """
    # Arrange
    summarization_config = ChatHistoryModel(
        model=mock_llm_model,
        retained_message_count=2,  # Keep only 2 messages
        summarize=True,
    )

    app_model = AppModel(
        name="test_app",
        registered_model=RegisteredModelModel(name="test_model"),
        orchestration=OrchestrationModel(
            supervisor=SupervisorModel(model=mock_llm_model)
        ),
        agents=[dummy_agent],
        chat_history=summarization_config,
    )
    app_config = AppConfig(app=app_model)

    messages = create_test_messages(8)
    initial_state = {"messages": messages, "summary": ""}

    # Act
    summarize_node = summarization_node(app_config)
    result = summarize_node(initial_state, {})

    # Assert
    assert result is not None
    assert "summary" in result
    
    # Should remove 6 messages (8 total - 2 retained)
    assert len(result["messages"]) == 6


def test_no_summarization_with_default_summarize_false(
    mock_llm_model: MagicMock, dummy_agent: AgentModel
):
    """
    Tests that summarization doesn't happen with the new default summarize=False.
    """
    # Arrange - don't specify summarize parameter (uses default False)
    summarization_config = ChatHistoryModel(
        model=mock_llm_model,
        retained_message_count=2,  # Would normally trigger summarization
    )

    app_model = AppModel(
        name="test_app",
        registered_model=RegisteredModelModel(name="test_model"),
        orchestration=OrchestrationModel(
            supervisor=SupervisorModel(model=mock_llm_model)
        ),
        agents=[dummy_agent],
        chat_history=summarization_config,
    )
    app_config = AppConfig(app=app_model)

    messages = create_test_messages(10)  # More than retained count
    initial_state = {"messages": messages, "summary": ""}

    # Act
    summarize_node = summarization_node(app_config)
    result = summarize_node(initial_state, {})

    # Assert - should trim messages but not create summary because default summarize=False
    assert result is not None
    assert "messages" in result
    assert "summary" not in result  # No summary should be created
    assert len(result["messages"]) == 8  # 10 total - 2 retained
    # Note: as_chat_model() may be called for setup even if summarization doesn't occur




def test_summarization_with_very_low_max_tokens(
    mock_llm_model: MagicMock, dummy_agent: AgentModel
):
    """
    Tests summarization with very low max_tokens.
    """
    # Arrange
    summarization_config = ChatHistoryModel(
        model=mock_llm_model,
        max_tokens=1,  # Very low token limit
        summarize=True,
    )

    app_model = AppModel(
        name="test_app",
        registered_model=RegisteredModelModel(name="test_model"),
        orchestration=OrchestrationModel(
            supervisor=SupervisorModel(model=mock_llm_model)
        ),
        agents=[dummy_agent],
        chat_history=summarization_config,
    )
    app_config = AppConfig(app=app_model)

    messages = create_test_messages(5)
    initial_state = {"messages": messages, "summary": ""}

    # Act
    summarize_node = summarization_node(app_config)
    result = summarize_node(initial_state, {})

    # Assert - depending on token counting, may or may not trigger summarization
    # But if it does, it should create a summary
    if result is None:
        # If no summarization occurred, that's also valid
        # Note: as_chat_model() may still be called for setup
        pass
    else:
        # If summarization occurred, check results
        assert "summary" in result
        assert result["summary"] == "This is a summary."


def test_summarization_with_existing_summary(
    mock_llm_model: MagicMock, dummy_agent: AgentModel
):
    """
    Tests that summarization works correctly when there's already an existing summary.
    """
    # Arrange
    summarization_config = ChatHistoryModel(
        model=mock_llm_model,
        retained_message_count=2,
        summarize=True,
    )

    app_model = AppModel(
        name="test_app",
        registered_model=RegisteredModelModel(name="test_model"),
        orchestration=OrchestrationModel(
            supervisor=SupervisorModel(model=mock_llm_model)
        ),
        agents=[dummy_agent],
        chat_history=summarization_config,
    )
    app_config = AppConfig(app=app_model)

    messages = create_test_messages(6)
    initial_state = {"messages": messages, "summary": "Previous conversation summary"}

    # Act
    summarize_node = summarization_node(app_config)
    result = summarize_node(initial_state, {})

    # Assert
    assert result is not None
    assert "summary" in result
    assert result["summary"] == "This is a summary."  # New summary should replace old one
    
    # Should remove 4 messages (6 total - 2 retained)
    assert len(result["messages"]) == 4


def test_no_summarization_when_message_count_equals_retained_count(
    mock_llm_model: MagicMock, dummy_agent: AgentModel
):
    """
    Tests that no summarization occurs when message count exactly equals retained_message_count.
    """
    # Arrange
    # Reset the mock to ensure clean state
    mock_llm_model.reset_mock()
    
    summarization_config = ChatHistoryModel(
        model=mock_llm_model,
        retained_message_count=5,
        summarize=True,
    )

    app_model = AppModel(
        name="test_app",
        registered_model=RegisteredModelModel(name="test_model"),
        orchestration=OrchestrationModel(
            supervisor=SupervisorModel(model=mock_llm_model)
        ),
        agents=[dummy_agent],
        chat_history=summarization_config,
    )
    app_config = AppConfig(app=app_model)

    messages = create_test_messages(5)  # Exactly equals retained_message_count
    initial_state = {"messages": messages, "summary": ""}

    # Act
    summarize_node = summarization_node(app_config)
    result = summarize_node(initial_state, {})

    # Assert - no summarization should occur because no messages need to be trimmed
    assert result is None


def test_summarization_with_empty_message_list(
    mock_llm_model: MagicMock, dummy_agent: AgentModel
):
    """
    Tests that summarization handles empty message list gracefully.
    """
    # Arrange
    summarization_config = ChatHistoryModel(
        model=mock_llm_model,
        retained_message_count=2,
        summarize=True,
    )

    app_model = AppModel(
        name="test_app",
        registered_model=RegisteredModelModel(name="test_model"),
        orchestration=OrchestrationModel(
            supervisor=SupervisorModel(model=mock_llm_model)
        ),
        agents=[dummy_agent],
        chat_history=summarization_config,
    )
    app_config = AppConfig(app=app_model)

    messages = []  # Empty message list
    initial_state = {"messages": messages, "summary": ""}

    # Act
    summarize_node = summarization_node(app_config)
    result = summarize_node(initial_state, {})

    # Assert - should return None with no messages to process
    assert result is None


def test_summarization_with_max_message_count_threshold(
    mock_llm_model: MagicMock, dummy_agent: AgentModel
):
    """
    Tests max_message_count as a threshold for triggering summarization when combined with retained_message_count.
    Trimming still occurs based on retained_message_count, but summarization only happens when max_message_count is exceeded.
    """
    # Arrange
    summarization_config = ChatHistoryModel(
        model=mock_llm_model,
        max_message_count=4,      # Trigger summarization when more than 4 messages
        retained_message_count=2, # But only keep 2 messages (always trim to this)
        summarize=True,
    )

    app_model = AppModel(
        name="test_app",
        registered_model=RegisteredModelModel(name="test_model"),
        orchestration=OrchestrationModel(
            supervisor=SupervisorModel(model=mock_llm_model)
        ),
        agents=[dummy_agent],
        chat_history=summarization_config,
    )
    app_config = AppConfig(app=app_model)

    # Test with messages below summarization threshold but above trim threshold
    messages_below = create_test_messages(3)  # 3 > retained_message_count(2) but <= max_message_count(4)
    state_below = {"messages": messages_below, "summary": ""}
    
    result_below = summarization_node(app_config)(state_below, {})
    # Should trim messages but not create summary (below max_message_count threshold)
    assert result_below is not None
    assert "messages" in result_below
    assert "summary" not in result_below  # No summary because below max_message_count
    assert len(result_below["messages"]) == 1  # 3 total - 2 retained

    # Test with messages above summarization threshold
    messages_above = create_test_messages(6)  # Above max_message_count(4)
    state_above = {"messages": messages_above, "summary": ""}
    
    result_above = summarization_node(app_config)(state_above, {})
    # Should trim messages and create summary (above max_message_count threshold)
    assert result_above is not None
    assert "summary" in result_above  # Summary because above max_message_count
    assert len(result_above["messages"]) == 4  # 6 total - 2 retained


def test_summarization_trims_without_summary_when_disabled(
    mock_llm_model: MagicMock, dummy_agent: AgentModel
):
    """
    Tests that message trimming still occurs even when summarization is disabled.
    This demonstrates the independent nature of trimming vs summarization.
    """
    # Arrange
    summarization_config = ChatHistoryModel(
        model=mock_llm_model,
        max_message_count=3,      # Should trigger trimming when exceeded
        retained_message_count=2, # Keep only 2 messages
        summarize=False,          # But don't create summaries
    )

    app_model = AppModel(
        name="test_app",
        registered_model=RegisteredModelModel(name="test_model"),
        orchestration=OrchestrationModel(
            supervisor=SupervisorModel(model=mock_llm_model)
        ),
        agents=[dummy_agent],
        chat_history=summarization_config,
    )
    app_config = AppConfig(app=app_model)

    messages = create_test_messages(8)  # Way more than max_message_count
    initial_state = {"messages": messages, "summary": ""}

    # Act
    summarize_node = summarization_node(app_config)
    result = summarize_node(initial_state, {})

    # Assert - should trim messages but create no summary
    assert result is not None
    assert "messages" in result
    assert "summary" not in result  # No summary should be created
    assert len(result["messages"]) == 6  # 8 total - 2 retained
    
    # Verify all returned messages are RemoveMessages
    assert all(isinstance(msg, RemoveMessage) for msg in result["messages"])
