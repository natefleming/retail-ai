"""Tests for Human-in-the-Loop (HITL) in LanggraphResponsesAgent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.errors import GraphInterrupt
from langgraph.types import Interrupt
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


def _make_interrupt(
    action_name: str = "send_email",
    args: dict | None = None,
    description: str = "Tool execution pending approval",
    allowed_decisions: list[str] | None = None,
    interrupt_id: str = "test-interrupt-id",
) -> Interrupt:
    """Create a real LangGraph Interrupt with HITL request payload."""
    if args is None:
        args = {"to": "test@example.com", "subject": "Test", "body": "Test body"}
    if allowed_decisions is None:
        allowed_decisions = ["approve", "edit", "reject"]
    return Interrupt(
        value={
            "action_requests": [
                {"name": action_name, "args": args, "description": description}
            ],
            "review_configs": [
                {
                    "action_name": action_name,
                    "allowed_decisions": allowed_decisions,
                }
            ],
        },
        id=interrupt_id,
    )


def test_predict_with_graph_interrupt_exception(responses_agent, mock_graph):
    """Test that GraphInterrupt raised by ainvoke() is caught and interrupt data
    is recovered from the checkpointer via aget_state()."""
    mock_interrupt = _make_interrupt(interrupt_id="gi-interrupt-1")

    mock_graph.ainvoke.side_effect = GraphInterrupt(interrupts=(mock_interrupt,))

    mock_snapshot = MagicMock()
    mock_snapshot.values = {
        "messages": [
            MagicMock(
                content="I can help you send that email. Approval required.",
                type="ai",
            )
        ]
    }
    mock_snapshot.interrupts = (mock_interrupt,)
    mock_graph.aget_state = AsyncMock(return_value=mock_snapshot)

    request = ResponsesAgentRequest(
        input=[
            Message(role="user", content="Send email to test@example.com"),
        ],
        custom_inputs={
            "configurable": {
                "thread_id": "test_gi_exception",
                "user_id": "test_user",
            }
        },
    )

    with patch(
        "dao_ai.models.get_state_snapshot_async", new_callable=AsyncMock
    ) as mock_state:
        mock_state.return_value = None
        response = responses_agent.predict(request)

    assert "interrupts" in response.custom_outputs
    assert len(response.custom_outputs["interrupts"]) == 1

    interrupt_data = response.custom_outputs["interrupts"][0]
    assert interrupt_data["action_requests"][0]["name"] == "send_email"
    assert interrupt_data["review_configs"][0]["allowed_decisions"] == [
        "approve",
        "edit",
        "reject",
    ]

    output_text = response.output[0].content[0]["text"]
    assert "send_email" in output_text
    assert "natural language" in output_text


def test_predict_interrupt_via_aget_state_fallback(responses_agent, mock_graph):
    """Test that when ainvoke() returns without __interrupt__ in the dict,
    the post-invocation aget_state() check detects and surfaces the interrupt."""
    mock_interrupt = _make_interrupt(
        action_name="create_ticket",
        args={"title": "Missing dataset", "priority": "high"},
        description="JIRA ticket pending approval",
        interrupt_id="fallback-interrupt-1",
    )

    mock_graph.ainvoke.return_value = {
        "messages": [
            MagicMock(
                content="No matching tables found. Filing a request.",
                type="ai",
            )
        ]
    }

    # First aget_state call (pre-invocation check) returns no interrupts;
    # second call (post-invocation fallback) returns the interrupt.
    pre_snapshot = MagicMock()
    pre_snapshot.interrupts = ()
    post_snapshot = MagicMock()
    post_snapshot.interrupts = (mock_interrupt,)
    mock_graph.aget_state = AsyncMock(side_effect=[pre_snapshot, post_snapshot])

    request = ResponsesAgentRequest(
        input=[
            Message(role="user", content="Find bird species tables"),
        ],
        custom_inputs={
            "configurable": {
                "thread_id": "test_fallback",
                "user_id": "test_user",
            }
        },
    )

    with patch(
        "dao_ai.models.get_state_snapshot_async", new_callable=AsyncMock
    ) as mock_state:
        mock_state.return_value = None
        response = responses_agent.predict(request)

    assert "interrupts" in response.custom_outputs
    assert len(response.custom_outputs["interrupts"]) == 1

    interrupt_data = response.custom_outputs["interrupts"][0]
    assert interrupt_data["action_requests"][0]["name"] == "create_ticket"
    assert interrupt_data["review_configs"][0]["allowed_decisions"] == [
        "approve",
        "edit",
        "reject",
    ]

    output_text = response.output[0].content[0]["text"]
    assert "create_ticket" in output_text


def test_predict_stream_with_graph_interrupt_exception(responses_agent, mock_graph):
    """Test that GraphInterrupt raised by astream() is caught and interrupt data
    is recovered from aget_state() in the streaming path."""
    mock_interrupt = _make_interrupt(interrupt_id="stream-gi-interrupt-1")

    async def mock_astream_raises(*args, **kwargs):
        yield (
            ("agent",),
            "messages",
            [MagicMock(content="Processing your request...", type="ai")],
        )
        raise GraphInterrupt(interrupts=(mock_interrupt,))

    mock_graph.astream = MagicMock(
        side_effect=lambda *args, **kwargs: mock_astream_raises()
    )

    mock_snapshot = MagicMock()
    mock_snapshot.values = {
        "messages": [
            MagicMock(content="Processing your request...", type="ai")
        ],
        "structured_response": None,
    }
    mock_snapshot.interrupts = (mock_interrupt,)
    mock_graph.aget_state = AsyncMock(return_value=mock_snapshot)

    request = ResponsesAgentRequest(
        input=[
            Message(role="user", content="Send email"),
        ],
        custom_inputs={
            "configurable": {
                "thread_id": "test_stream_gi",
                "user_id": "test_user",
            }
        },
    )

    with patch(
        "dao_ai.models.get_state_snapshot_async", new_callable=AsyncMock
    ) as mock_state:
        mock_state.return_value = None
        events = list(responses_agent.predict_stream(request))

    final_event = [e for e in events if e.type == "response.output_item.done"][0]

    assert "interrupts" in final_event.custom_outputs
    assert len(final_event.custom_outputs["interrupts"]) == 1

    interrupt_data = final_event.custom_outputs["interrupts"][0]
    assert interrupt_data["action_requests"][0]["name"] == "send_email"
    assert interrupt_data["review_configs"][0]["allowed_decisions"] == [
        "approve",
        "edit",
        "reject",
    ]


def test_predict_stream_interrupt_via_aget_state_fallback(
    responses_agent, mock_graph
):
    """Test that when astream() yields messages but no __interrupt__ update event,
    the post-stream aget_state() check detects and surfaces the interrupt."""
    mock_interrupt = _make_interrupt(
        action_name="create_ticket",
        args={"title": "Missing dataset"},
        interrupt_id="stream-fallback-interrupt-1",
    )

    async def mock_astream_no_interrupt(*args, **kwargs):
        yield (
            ("agent",),
            "messages",
            [MagicMock(content="Searching catalog...", type="ai")],
        )
        yield (
            ("agent",),
            "updates",
            {"agent": {"messages": [MagicMock(content="Done.", type="ai")]}},
        )

    mock_graph.astream = MagicMock(
        side_effect=lambda *args, **kwargs: mock_astream_no_interrupt()
    )

    # First aget_state call (pre-invocation check) returns no interrupts;
    # second call (post-stream fallback) returns the interrupt.
    pre_snapshot = MagicMock()
    pre_snapshot.interrupts = ()
    post_snapshot = MagicMock()
    post_snapshot.values = {"structured_response": None}
    post_snapshot.interrupts = (mock_interrupt,)
    mock_graph.aget_state = AsyncMock(side_effect=[pre_snapshot, post_snapshot])

    request = ResponsesAgentRequest(
        input=[
            Message(role="user", content="Find bird species tables"),
        ],
        custom_inputs={
            "configurable": {
                "thread_id": "test_stream_fallback",
                "user_id": "test_user",
            }
        },
    )

    with patch(
        "dao_ai.models.get_state_snapshot_async", new_callable=AsyncMock
    ) as mock_state:
        mock_state.return_value = None
        events = list(responses_agent.predict_stream(request))

    final_event = [e for e in events if e.type == "response.output_item.done"][0]

    assert "interrupts" in final_event.custom_outputs
    assert len(final_event.custom_outputs["interrupts"]) == 1

    interrupt_data = final_event.custom_outputs["interrupts"][0]
    assert interrupt_data["action_requests"][0]["name"] == "create_ticket"
