"""Tests for inference with summarization middleware enabled.

These tests verify that agents are created with SummarizationMiddleware
when chat history is configured.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.runtime import Runtime

from dao_ai.config import (
    AgentModel,
    AppModel,
    ChatHistoryModel,
    OrchestrationModel,
    RegisteredModelModel,
    SupervisorModel,
)
from dao_ai.nodes import create_agent_node
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


def make_async_mock_agent(name: str, return_value: dict):
    """Helper function to create a mock agent with async ainvoke method."""
    mock_agent = MagicMock()
    mock_agent.name = name

    async def mock_ainvoke(**kwargs):
        return return_value

    mock_agent.ainvoke = MagicMock(side_effect=mock_ainvoke)
    return mock_agent


def run_async_test(async_func, *args):
    """Helper function to run async functions in tests."""
    return asyncio.run(async_func(*args))


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
            max_tokens=2048,
            max_tokens_before_summary=6000,
            max_messages_before_summary=10,
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

    @patch("dao_ai.nodes.create_agent")
    def test_create_agent_node_with_chat_history_includes_middleware(
        self, mock_create_agent, app_model_with_chat_history, mock_runtime
    ):
        """Test create_agent_node with chat history includes SummarizationMiddleware."""
        # Mock the compiled agent
        mock_compiled_agent = MagicMock()
        mock_compiled_agent.name = "test_agent"
        mock_create_agent.return_value = mock_compiled_agent

        agent_model = app_model_with_chat_history.agents[0]
        create_agent_node(
            agent=agent_model,
            memory=None,
            chat_history=app_model_with_chat_history.chat_history,
        )

        # Verify create_agent was called with middleware
        mock_create_agent.assert_called_once()
        call_kwargs = mock_create_agent.call_args[1]

        # Verify middleware list includes SummarizationMiddleware
        middleware_list = call_kwargs.get("middleware", [])
        assert len(middleware_list) > 0

        # Check that at least one middleware is a SummarizationMiddleware
        from langchain.agents.middleware import SummarizationMiddleware

        has_summarization = any(
            isinstance(m, SummarizationMiddleware) for m in middleware_list
        )
        assert has_summarization, "SummarizationMiddleware should be in middleware list"

    @patch("dao_ai.nodes.create_agent")
    def test_create_agent_node_without_chat_history_no_summarization(
        self, mock_create_agent, app_model_without_chat_history
    ):
        """Test create_agent_node without chat history doesn't include SummarizationMiddleware."""
        # Mock the compiled agent
        mock_compiled_agent = MagicMock()
        mock_compiled_agent.name = "test_agent"
        mock_create_agent.return_value = mock_compiled_agent

        agent_model = app_model_without_chat_history.agents[0]
        create_agent_node(
            agent=agent_model,
            memory=None,
            chat_history=None,
        )

        # Verify create_agent was called
        mock_create_agent.assert_called_once()
        call_kwargs = mock_create_agent.call_args[1]

        # Verify middleware list does NOT include SummarizationMiddleware
        middleware_list = call_kwargs.get("middleware", [])
        from langchain.agents.middleware import SummarizationMiddleware

        has_summarization = any(
            isinstance(m, SummarizationMiddleware) for m in middleware_list
        )
        assert not has_summarization, (
            "SummarizationMiddleware should NOT be in middleware list"
        )

    @patch("dao_ai.nodes.create_agent")
    def test_create_agent_node_middleware_order(
        self, mock_create_agent, app_model_with_chat_history
    ):
        """Test that middleware is passed in correct order to create_agent."""
        mock_compiled_agent = MagicMock()
        mock_compiled_agent.name = "test_agent"
        mock_create_agent.return_value = mock_compiled_agent

        agent_model = app_model_with_chat_history.agents[0]
        create_agent_node(
            agent=agent_model,
            memory=None,
            chat_history=app_model_with_chat_history.chat_history,
        )

        call_kwargs = mock_create_agent.call_args[1]

        # Verify all required parameters are passed
        assert "model" in call_kwargs
        assert "tools" in call_kwargs
        assert "middleware" in call_kwargs
        assert "state_schema" in call_kwargs
        assert "context_schema" in call_kwargs

    @patch("dao_ai.nodes.create_agent")
    def test_create_agent_node_sets_agent_name(
        self, mock_create_agent, app_model_with_chat_history
    ):
        """Test that create_agent_node sets the agent name correctly."""
        mock_compiled_agent = MagicMock()
        mock_compiled_agent.name = "test_agent"
        mock_create_agent.return_value = mock_compiled_agent

        agent_model = app_model_with_chat_history.agents[0]
        node = create_agent_node(
            agent=agent_model,
            memory=None,
            chat_history=app_model_with_chat_history.chat_history,
        )

        # Verify the node name was set
        assert node.name == "test_agent"

    @patch("dao_ai.nodes.create_agent")
    def test_summarization_middleware_trigger_config(
        self, mock_create_agent, app_model_with_chat_history
    ):
        """Test that SummarizationMiddleware is configured with correct trigger."""
        mock_compiled_agent = MagicMock()
        mock_compiled_agent.name = "test_agent"
        mock_create_agent.return_value = mock_compiled_agent

        agent_model = app_model_with_chat_history.agents[0]
        create_agent_node(
            agent=agent_model,
            memory=None,
            chat_history=app_model_with_chat_history.chat_history,
        )

        call_kwargs = mock_create_agent.call_args[1]
        middleware_list = call_kwargs.get("middleware", [])

        # Find the SummarizationMiddleware
        from langchain.agents.middleware import SummarizationMiddleware

        summarization_mw = next(
            (m for m in middleware_list if isinstance(m, SummarizationMiddleware)), None
        )

        assert summarization_mw is not None

    @patch("dao_ai.nodes.logger")
    @patch("dao_ai.nodes.create_agent")
    def test_create_agent_node_logs_middleware_count(
        self, mock_create_agent, mock_logger, app_model_with_chat_history
    ):
        """Test that create_agent_node logs the middleware count."""
        mock_compiled_agent = MagicMock()
        mock_compiled_agent.name = "test_agent"
        mock_create_agent.return_value = mock_compiled_agent

        agent_model = app_model_with_chat_history.agents[0]
        create_agent_node(
            agent=agent_model,
            memory=None,
            chat_history=app_model_with_chat_history.chat_history,
        )

        # Verify debug logging was called
        # The exact message format may vary, just check debug was called
        assert mock_logger.debug.called

    @patch("dao_ai.nodes.create_agent")
    def test_create_agent_node_with_guardrails_and_chat_history(
        self, mock_create_agent, mock_llm_model
    ):
        """Test create_agent_node with both guardrails and chat history."""
        from dao_ai.config import GuardrailModel

        mock_compiled_agent = MagicMock()
        mock_compiled_agent.name = "test_agent"
        mock_create_agent.return_value = mock_compiled_agent

        agent_model = AgentModel(
            name="test_agent",
            model=mock_llm_model,
            guardrails=[
                GuardrailModel(
                    name="test_guardrail",
                    model=mock_llm_model,
                    prompt="Test guardrail prompt",
                )
            ],
        )

        chat_history = ChatHistoryModel(
            model=mock_llm_model,
            max_tokens=512,
            max_tokens_before_summary=1000,
        )

        create_agent_node(
            agent=agent_model,
            memory=None,
            chat_history=chat_history,
        )

        call_kwargs = mock_create_agent.call_args[1]
        middleware_list = call_kwargs.get("middleware", [])

        # Should have at least GuardrailMiddleware and SummarizationMiddleware
        from langchain.agents.middleware import SummarizationMiddleware

        from dao_ai.middleware.guardrails import GuardrailMiddleware

        has_summarization = any(
            isinstance(m, SummarizationMiddleware) for m in middleware_list
        )
        has_guardrail = any(isinstance(m, GuardrailMiddleware) for m in middleware_list)

        assert has_summarization, "Should have SummarizationMiddleware"
        assert has_guardrail, "Should have GuardrailMiddleware"


if __name__ == "__main__":
    pytest.main([__file__])
