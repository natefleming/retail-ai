"""Tests for Human-in-the-Loop (HITL) in LanggraphResponsesAgent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mlflow.types.responses import ResponsesAgentRequest
from mlflow.types.responses_helpers import Message

from dao_ai.models import LanggraphResponsesAgent


@pytest.fixture
def mock_graph():
    """Create a mock CompiledStateGraph."""
    graph = MagicMock()
    graph.ainvoke = AsyncMock()
    graph.astream = AsyncMock()
    graph.aget_state = AsyncMock()

    # By default, return a non-interrupted state
    mock_snapshot = MagicMock()
    mock_snapshot.interrupts = ()  # Empty tuple = not interrupted
    graph.aget_state.return_value = mock_snapshot

    return graph


@pytest.fixture
def responses_agent(mock_graph):
    """Create a LanggraphResponsesAgent with mock graph."""
    return LanggraphResponsesAgent(mock_graph)


class MockInterrupt:
    """Mock interrupt object matching LangGraph structure."""

    def __init__(self, action_requests, interrupt_id: str = "test-interrupt-id"):
        self.value = {"action_requests": action_requests}
        self.id = interrupt_id


def test_predict_without_interrupt(responses_agent, mock_graph):
    """Test normal prediction without HITL interrupt."""
    # Mock graph response without interrupt
    mock_graph.ainvoke.return_value = {
        "messages": [
            MagicMock(
                content="Here is the information you requested.",
                type="ai",
            )
        ]
    }

    request = ResponsesAgentRequest(
        input=[
            Message(role="user", content="What is the weather?"),
        ],
        custom_inputs={
            "configurable": {
                "thread_id": "test_123",
                "user_id": "test_user",
            }
        },
    )

    with patch(
        "dao_ai.models.get_state_snapshot_async", new_callable=AsyncMock
    ) as mock_state:
        mock_state.return_value = None

        response = responses_agent.predict(request)

    # Verify response
    assert len(response.output) == 1
    assert (
        response.output[0].content[0]["text"]
        == "Here is the information you requested."
    )
    assert "interrupts" not in response.custom_outputs
    assert "thread_id" in response.custom_outputs["configurable"]


def test_predict_with_interrupt(responses_agent, mock_graph):
    """Test prediction with HITL interrupt (pending action)."""

    # Mock graph response with interrupt including review_configs
    class MockInterruptWithConfig:
        def __init__(
            self,
            action_requests,
            review_configs,
            interrupt_id: str = "test-interrupt-id",
        ):
            self.value = {
                "action_requests": action_requests,
                "review_configs": review_configs,
            }
            self.id = interrupt_id

    mock_interrupt = MockInterruptWithConfig(
        action_requests=[
            {
                "name": "send_email",
                "arguments": {
                    "to": "test@example.com",
                    "subject": "Test",
                    "body": "Test body",
                },
                "description": "Tool execution pending approval",
            }
        ],
        review_configs=[
            {
                "action_name": "send_email",
                "allowed_decisions": ["approve", "edit", "reject"],
            }
        ],
    )

    mock_graph.ainvoke.return_value = {
        "messages": [
            MagicMock(
                content="I can help you send that email. Approval required.",
                type="ai",
            )
        ],
        "__interrupt__": [mock_interrupt],
    }

    request = ResponsesAgentRequest(
        input=[
            Message(role="user", content="Send email to test@example.com"),
        ],
        custom_inputs={
            "configurable": {
                "thread_id": "test_123",
                "user_id": "test_user",
            }
        },
    )

    with patch(
        "dao_ai.models.get_state_snapshot_async", new_callable=AsyncMock
    ) as mock_state:
        mock_state.return_value = None

        response = responses_agent.predict(request)

    # Verify interrupt is surfaced in response
    assert "interrupts" in response.custom_outputs
    assert len(response.custom_outputs["interrupts"]) == 1

    interrupt = response.custom_outputs["interrupts"][0]
    assert "action_requests" in interrupt
    assert len(interrupt["action_requests"]) == 1

    action_request = interrupt["action_requests"][0]
    assert action_request["name"] == "send_email"
    assert action_request["arguments"]["to"] == "test@example.com"
    assert "description" in action_request

    # Verify review_configs are included
    assert "review_configs" in interrupt
    assert len(interrupt["review_configs"]) == 1
    assert interrupt["review_configs"][0]["action_name"] == "send_email"
    assert interrupt["review_configs"][0]["allowed_decisions"] == [
        "approve",
        "edit",
        "reject",
    ]

    # Verify action requests are shown to the user
    output_text = response.output[0].content[0]["text"]
    assert "1. send_email" in output_text
    assert "(no arguments)" in output_text
    assert (
        "natural language" in output_text
    )  # Verify it mentions natural language response option


def test_predict_resume_with_approval(responses_agent, mock_graph):
    """Test resuming interrupted graph with approval decision."""
    from langgraph.types import Command

    # Mock graph response after resume
    mock_graph.ainvoke.return_value = {
        "messages": [
            MagicMock(
                content="Email sent successfully.",
                type="ai",
            )
        ]
    }

    request = ResponsesAgentRequest(
        input=[],
        custom_inputs={
            "configurable": {
                "thread_id": "test_123",
                "user_id": "test_user",
            },
            "decisions": [{"type": "approve"}],
        },
    )

    with patch(
        "dao_ai.models.get_state_snapshot_async", new_callable=AsyncMock
    ) as mock_state:
        mock_state.return_value = None

        response = responses_agent.predict(request)

    # Verify Command was used to resume
    call_args = mock_graph.ainvoke.call_args
    assert isinstance(call_args[0][0], Command)
    assert call_args[0][0].resume == {"decisions": [{"type": "approve"}]}

    # Verify response
    assert len(response.output) == 1
    assert response.output[0].content[0]["text"] == "Email sent successfully."
    assert "interrupts" not in response.custom_outputs


def test_predict_resume_with_rejection(responses_agent, mock_graph):
    """Test resuming interrupted graph with rejection decision."""
    from langgraph.types import Command

    # Mock graph response after resume with rejection
    mock_graph.ainvoke.return_value = {
        "messages": [
            MagicMock(
                content="Understood, I will not send the email.",
                type="ai",
            )
        ]
    }

    request = ResponsesAgentRequest(
        input=[],
        custom_inputs={
            "configurable": {
                "thread_id": "test_123",
                "user_id": "test_user",
            },
            "decisions": [
                {
                    "type": "reject",
                    "message": "Email content needs review",
                }
            ],
        },
    )

    with patch(
        "dao_ai.models.get_state_snapshot_async", new_callable=AsyncMock
    ) as mock_state:
        mock_state.return_value = None

        response = responses_agent.predict(request)

    # Verify Command was used to resume with rejection
    call_args = mock_graph.ainvoke.call_args
    assert isinstance(call_args[0][0], Command)
    assert call_args[0][0].resume["decisions"][0]["type"] == "reject"
    assert (
        call_args[0][0].resume["decisions"][0]["message"]
        == "Email content needs review"
    )

    # Verify response
    assert len(response.output) == 1
    assert "interrupts" not in response.custom_outputs


def test_predict_multiple_interrupts(responses_agent, mock_graph):
    """Test handling multiple pending actions in a single response."""

    # Mock graph response with multiple interrupts including review_configs
    class MockInterruptWithConfig:
        def __init__(
            self,
            action_requests,
            review_configs,
            interrupt_id: str = "test-interrupt-id",
        ):
            self.value = {
                "action_requests": action_requests,
                "review_configs": review_configs,
            }
            self.id = interrupt_id

    mock_interrupt1 = MockInterruptWithConfig(
        action_requests=[
            {
                "name": "send_email",
                "arguments": {"to": "user1@example.com"},
                "description": "Send email 1",
            }
        ],
        review_configs=[
            {
                "action_name": "send_email",
                "allowed_decisions": ["approve", "reject"],
            }
        ],
        interrupt_id="interrupt-1",
    )

    mock_interrupt2 = MockInterruptWithConfig(
        action_requests=[
            {
                "name": "send_email",
                "arguments": {"to": "user2@example.com"},
                "description": "Send email 2",
            }
        ],
        review_configs=[
            {
                "action_name": "send_email",
                "allowed_decisions": ["approve", "edit", "reject"],
            }
        ],
        interrupt_id="interrupt-2",
    )

    mock_graph.ainvoke.return_value = {
        "messages": [
            MagicMock(
                content="Multiple approvals required.",
                type="ai",
            )
        ],
        "__interrupt__": [mock_interrupt1, mock_interrupt2],
    }

    request = ResponsesAgentRequest(
        input=[
            Message(role="user", content="Send emails to two users"),
        ],
        custom_inputs={
            "configurable": {
                "thread_id": "test_123",
                "user_id": "test_user",
            }
        },
    )

    with patch(
        "dao_ai.models.get_state_snapshot_async", new_callable=AsyncMock
    ) as mock_state:
        mock_state.return_value = None

        response = responses_agent.predict(request)

    # Verify both interrupts are surfaced
    assert "interrupts" in response.custom_outputs
    assert len(response.custom_outputs["interrupts"]) == 2

    # Verify first interrupt
    assert "action_requests" in response.custom_outputs["interrupts"][0]
    assert (
        response.custom_outputs["interrupts"][0]["action_requests"][0]["arguments"][
            "to"
        ]
        == "user1@example.com"
    )

    # Verify second interrupt
    assert "action_requests" in response.custom_outputs["interrupts"][1]
    assert (
        response.custom_outputs["interrupts"][1]["action_requests"][0]["arguments"][
            "to"
        ]
        == "user2@example.com"
    )

    # Verify action requests are shown to the user
    output_text = response.output[0].content[0]["text"]
    assert "1. send_email" in output_text
    assert "2. send_email" in output_text
    assert (
        "natural language" in output_text
    )  # Verify it mentions natural language response option


def test_predict_stream_with_interrupt(responses_agent, mock_graph):
    """Test streaming with HITL interrupt."""

    async def mock_astream(*args, **kwargs):
        """Mock async stream generator with interrupt."""
        # Yield some messages
        yield (
            ("agent",),
            "messages",
            [MagicMock(content="Processing...", type="ai")],
        )

        # Yield interrupt in updates mode with review_configs
        class MockInterruptWithConfig:
            def __init__(
                self,
                action_requests,
                review_configs,
                interrupt_id: str = "test-interrupt-id",
            ):
                self.value = {
                    "action_requests": action_requests,
                    "review_configs": review_configs,
                }
                self.id = interrupt_id

        interrupt = MockInterruptWithConfig(
            action_requests=[
                {
                    "name": "send_email",
                    "arguments": {"to": "test@example.com"},
                    "description": "Approval required",
                }
            ],
            review_configs=[
                {
                    "action_name": "send_email",
                    "allowed_decisions": ["approve", "edit", "reject"],
                }
            ],
        )
        yield (
            ("agent",),
            "updates",
            {"__interrupt__": [interrupt]},
        )

    # Mock astream to return the generator
    mock_graph.astream = MagicMock(side_effect=lambda *args, **kwargs: mock_astream())

    request = ResponsesAgentRequest(
        input=[
            Message(role="user", content="Send email"),
        ],
        custom_inputs={
            "configurable": {
                "thread_id": "test_123",
                "user_id": "test_user",
            }
        },
    )

    with patch(
        "dao_ai.models.get_state_snapshot_async", new_callable=AsyncMock
    ) as mock_state:
        mock_state.return_value = None

        events = list(responses_agent.predict_stream(request))

    # Find the final event with custom_outputs
    final_event = [e for e in events if e.type == "response.output_item.done"][0]

    # Verify interrupt is surfaced
    assert "interrupts" in final_event.custom_outputs
    assert len(final_event.custom_outputs["interrupts"]) == 1

    interrupt = final_event.custom_outputs["interrupts"][0]
    assert "action_requests" in interrupt
    assert interrupt["action_requests"][0]["name"] == "send_email"
