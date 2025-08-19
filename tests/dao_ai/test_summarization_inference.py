"""Tests for inference with summarization node enabled."""

import unittest
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.runtime import Runtime

from dao_ai.config import (
    AgentModel,
    AppModel,
    ChatHistoryModel,
    LLMModel,
    OrchestrationModel,
    RegisteredModelModel,
    SupervisorModel,
)
from dao_ai.nodes import call_agent_with_summarized_messages, create_agent_node
from dao_ai.state import Context


def create_test_messages(count: int, prefix: str = "Message") -> list[BaseMessage]:
    """Create a list of test messages for testing."""
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
    mock_llm.invoke.return_value = AIMessage(content="This is a test response.")

    mock_llm_model = MagicMock(spec=LLMModel)
    mock_llm_model.as_chat_model.return_value = mock_llm
    return mock_llm_model


@pytest.fixture
def mock_runtime():
    """Mock runtime for testing."""
    runtime = MagicMock(spec=Runtime)
    runtime.context = MagicMock(spec=Context)
    runtime.context.user_id = "test_user"
    runtime.context.thread_id = "test_thread"
    return runtime


@pytest.fixture
def app_model_with_chat_history(mock_llm_model):
    """App model with chat history configured."""
    return AppModel(
        name="test_app",
        registered_model=RegisteredModelModel(name="test_model"),
        orchestration=OrchestrationModel(
            supervisor=SupervisorModel(model=mock_llm_model)
        ),
        agents=[AgentModel(name="test_agent", model=mock_llm_model)],
        chat_history=ChatHistoryModel(
            model=mock_llm_model,
            max_tokens=512,
            max_tokens_before_summary=1000,
            max_messages_before_summary=10,
            max_summary_tokens=150,
        ),
    )


@pytest.fixture
def app_model_without_chat_history(mock_llm_model):
    """App model without chat history configured."""
    return AppModel(
        name="test_app",
        registered_model=RegisteredModelModel(name="test_model"),
        orchestration=OrchestrationModel(
            supervisor=SupervisorModel(model=mock_llm_model)
        ),
        agents=[AgentModel(name="test_agent", model=mock_llm_model)],
        chat_history=None,
    )


class TestSummarizationInference:
    """Tests for inference with summarization enabled."""

    def test_call_agent_with_summarized_messages_basic(self, mock_runtime):
        """Test basic functionality of call_agent_with_summarized_messages."""
        # Mock agent
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Agent response", id="response-1")]
        }

        # Create the function
        call_agent_func = call_agent_with_summarized_messages(mock_agent)

        # Test with summarized messages
        test_messages = create_test_messages(5, "Summarized")
        state = {"summarized_messages": test_messages}

        result = call_agent_func(state, mock_runtime)

        # Verify agent was called with correct input
        mock_agent.invoke.assert_called_once_with(
            input={"messages": test_messages}, context=mock_runtime.context
        )

        # Verify result contains only agent response (LangGraph state handles combining)
        assert "messages" in result
        assert len(result["messages"]) == 1  # 1 agent response
        # Check that the message is the agent response
        assert result["messages"][0].content == "Agent response"

    def test_call_agent_with_summarized_messages_empty_state(self, mock_runtime):
        """Test call_agent_with_summarized_messages with empty state."""
        # Mock agent
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent.invoke.return_value = {"messages": []}

        # Create the function
        call_agent_func = call_agent_with_summarized_messages(mock_agent)

        # Test with empty state
        state = {}

        result = call_agent_func(state, mock_runtime)

        # Verify agent was called with empty messages
        mock_agent.invoke.assert_called_once_with(
            input={"messages": []}, context=mock_runtime.context
        )

        # Verify result
        assert "messages" in result
        assert result["messages"] == []

    def test_call_agent_with_summarized_messages_no_summarized_key(self, mock_runtime):
        """Test call_agent_with_summarized_messages when summarized_messages key is missing."""
        # Mock agent
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Agent response")]
        }

        # Create the function
        call_agent_func = call_agent_with_summarized_messages(mock_agent)

        # Test with state missing summarized_messages key
        state = {"other_key": "other_value"}

        call_agent_func(state, mock_runtime)

        # Verify agent was called with empty messages (default value)
        mock_agent.invoke.assert_called_once_with(
            input={"messages": []}, context=mock_runtime.context
        )

    @patch("dao_ai.nodes.create_react_agent")
    def test_create_agent_node_with_chat_history(
        self, mock_create_react_agent, app_model_with_chat_history, mock_runtime
    ):
        """Test create_agent_node with chat history enabled."""
        # Mock the compiled agent
        mock_compiled_agent = MagicMock()
        mock_compiled_agent.name = "test_agent"
        mock_create_react_agent.return_value = mock_compiled_agent

        # Mock workflow compilation
        with patch("dao_ai.nodes.StateGraph") as mock_state_graph:
            mock_workflow = MagicMock()
            mock_workflow.compile.return_value = MagicMock()
            mock_state_graph.return_value = mock_workflow

            agent_model = app_model_with_chat_history.agents[0]
            create_agent_node(app_model_with_chat_history, agent_model)

            # Verify that StateGraph was used (indicating chat history workflow)
            mock_state_graph.assert_called_once()
            mock_workflow.add_node.assert_any_call("summarization", unittest.mock.ANY)
            mock_workflow.add_node.assert_any_call("agent", unittest.mock.ANY)
            mock_workflow.add_edge.assert_called_with("summarization", "agent")
            mock_workflow.set_entry_point.assert_called_with("summarization")

    @patch("dao_ai.nodes.create_react_agent")
    def test_create_agent_node_without_chat_history(
        self, mock_create_react_agent, app_model_without_chat_history
    ):
        """Test create_agent_node without chat history."""
        # Mock the compiled agent
        mock_compiled_agent = MagicMock()
        mock_compiled_agent.name = "test_agent"
        mock_create_react_agent.return_value = mock_compiled_agent

        agent_model = app_model_without_chat_history.agents[0]
        node = create_agent_node(app_model_without_chat_history, agent_model)

        # Verify that the compiled agent is returned directly (no workflow)
        assert node == mock_compiled_agent

    @patch("dao_ai.nodes.summarization_node")
    @patch("dao_ai.nodes.create_react_agent")
    def test_summarization_inference_integration(
        self,
        mock_create_react_agent,
        mock_summarization_node,
        app_model_with_chat_history,
        mock_runtime,
    ):
        """Integration test for summarization node with inference."""
        # Mock the compiled agent
        mock_compiled_agent = MagicMock()
        mock_compiled_agent.name = "test_agent"
        mock_compiled_agent.invoke.return_value = {
            "messages": [AIMessage(content="Final agent response")]
        }
        mock_create_react_agent.return_value = mock_compiled_agent

        # Mock the summarization node
        mock_summ_node = MagicMock()
        mock_summarization_node.return_value = mock_summ_node

        # Mock StateGraph and workflow
        with patch("dao_ai.nodes.StateGraph") as mock_state_graph:
            mock_workflow = MagicMock()
            mock_compiled_workflow = MagicMock()
            mock_workflow.compile.return_value = mock_compiled_workflow
            mock_state_graph.return_value = mock_workflow

            # Create agent node with chat history
            agent_model = app_model_with_chat_history.agents[0]
            create_agent_node(app_model_with_chat_history, agent_model)

            # Verify summarization node was created
            mock_summarization_node.assert_called_once_with(app_model_with_chat_history)

            # Verify workflow was set up correctly
            assert mock_workflow.add_node.call_count == 2
            mock_workflow.add_node.assert_any_call("summarization", mock_summ_node)

            # Verify the agent node function was added
            agent_calls = [
                call
                for call in mock_workflow.add_node.call_args_list
                if call[0][0] == "agent"
            ]
            assert len(agent_calls) == 1

            # Verify workflow structure
            mock_workflow.add_edge.assert_called_with("summarization", "agent")
            mock_workflow.set_entry_point.assert_called_with("summarization")
            mock_workflow.compile.assert_called_with(name="test_agent")

    @patch("dao_ai.nodes.logger")
    def test_call_agent_with_summarized_messages_logging(
        self, mock_logger, mock_runtime
    ):
        """Test that call_agent_with_summarized_messages logs correctly."""
        # Mock agent
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent.invoke.return_value = {
            "messages": [
                AIMessage(content="Response 1"),
                AIMessage(content="Response 2"),
            ]
        }

        # Create the function
        call_agent_func = call_agent_with_summarized_messages(mock_agent)

        # Test with summarized messages
        test_messages = create_test_messages(3, "Test")
        state = {"summarized_messages": test_messages}

        call_agent_func(state, mock_runtime)

        # Verify logging calls
        mock_logger.debug.assert_any_call(
            "Calling agent test_agent with summarized messages"
        )
        mock_logger.debug.assert_any_call("Found 3 summarized messages")
        mock_logger.debug.assert_any_call(
            "Agent returned 2 messages"
        )  # 2 agent response messages

    def test_call_agent_with_different_message_types(self, mock_runtime):
        """Test call_agent_with_summarized_messages with different message types."""
        # Mock agent
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Agent response")]
        }

        # Create the function
        call_agent_func = call_agent_with_summarized_messages(mock_agent)

        # Test with mixed message types
        test_messages = [
            HumanMessage(content="Human message 1"),
            AIMessage(content="AI message 1"),
            HumanMessage(content="Human message 2"),
        ]
        state = {"summarized_messages": test_messages}

        result = call_agent_func(state, mock_runtime)

        # Verify agent was called with all message types
        mock_agent.invoke.assert_called_once_with(
            input={"messages": test_messages}, context=mock_runtime.context
        )

        # Verify result
        assert "messages" in result
        assert len(result["messages"]) == 1  # 1 agent response
        # Check that the message is the agent response
        assert result["messages"][0].content == "Agent response"

    def test_agent_response_format(self, mock_runtime):
        """Test that agent response is properly formatted."""
        # Mock agent with different response formats
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"

        # Test case 1: Normal response with messages
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Normal response")],
            "other_data": "should_be_ignored",
        }

        call_agent_func = call_agent_with_summarized_messages(mock_agent)
        state = {"summarized_messages": [HumanMessage(content="Test")]}

        result = call_agent_func(state, mock_runtime)
        # Should contain only the agent response (LangGraph state handles combining)
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "Normal response"  # agent response

        # Test case 2: Response with empty messages
        mock_agent.invoke.return_value = {"messages": [], "other_data": "ignored"}

        result = call_agent_func(state, mock_runtime)
        # Should contain empty messages (no agent response)
        assert len(result["messages"]) == 0

        # Test case 3: Response without messages key
        mock_agent.invoke.return_value = {"other_data": "no_messages"}

        result = call_agent_func(state, mock_runtime)
        # Should contain no messages (no agent response, no messages key)
        assert len(result["messages"]) == 0


if __name__ == "__main__":
    pytest.main([__file__])
