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
        "messages": [MagicMock(content="Processing your request...", type="ai")],
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


def test_predict_stream_interrupt_via_aget_state_fallback(responses_agent, mock_graph):
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


# ---------------------------------------------------------------------------
# Integration tests with real LangGraph graph + InMemorySaver
# ---------------------------------------------------------------------------


class _FakeToolChatModel:
    """Minimal chat model that supports bind_tools and returns predetermined messages.

    Implemented as a wrapper around BaseChatModel to satisfy create_agent's
    requirement for tool-binding support.
    """

    pass  # Defined below to avoid import at module top level


def _build_hitl_agent():
    """Build a real LangGraph agent with HITL middleware and InMemorySaver.

    Returns (agent, send_email_tool) for use in integration tests.
    """
    from langchain.agents import create_agent
    from langchain.agents.middleware import HumanInTheLoopMiddleware
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import InMemorySaver

    @tool
    def send_email(to: str, subject: str, body: str) -> str:
        """Send an email to someone."""
        return f"Email sent to {to}"

    class FakeToolChatModel(BaseChatModel):
        responses: list[AIMessage]
        call_count: int = 0

        @property
        def _llm_type(self) -> str:
            return "fake-tool-chat"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            idx = min(self.call_count, len(self.responses) - 1)
            self.call_count += 1
            return ChatResult(generations=[ChatGeneration(message=self.responses[idx])])

        def bind_tools(self, tools, **kwargs):
            return self

    model = FakeToolChatModel(
        responses=[
            AIMessage(
                content="Sending the email now.",
                tool_calls=[
                    {
                        "name": "send_email",
                        "args": {
                            "to": "test@example.com",
                            "subject": "Test",
                            "body": "Hello",
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Email sent successfully!"),
        ],
    )

    graph = create_agent(
        model=model,
        tools=[send_email],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "send_email": {
                        "allowed_decisions": ["approve", "edit", "reject"],
                        "description": "Review email before sending",
                    }
                },
            ),
        ],
        checkpointer=InMemorySaver(),
    )

    return graph, send_email


def test_integration_ainvoke_returns_interrupt():
    """Integration test: ainvoke() on a real graph returns __interrupt__ in the dict."""
    graph, _ = _build_hitl_agent()

    config = {"configurable": {"thread_id": "integ-ainvoke-1"}}
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "Send email to test"}]},
        config=config,
    )

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "__interrupt__" in result, (
        f"Expected __interrupt__ in result. Keys: {list(result.keys())}"
    )
    assert len(result["__interrupt__"]) == 1

    interrupt = result["__interrupt__"][0]
    assert interrupt.value["action_requests"][0]["name"] == "send_email"
    assert interrupt.value["review_configs"][0]["allowed_decisions"] == [
        "approve",
        "edit",
        "reject",
    ]


def test_integration_ainvoke_resume_with_approve():
    """Integration test: approve resumes graph and the tool executes."""
    from langgraph.types import Command

    graph, _ = _build_hitl_agent()

    config = {"configurable": {"thread_id": "integ-resume-approve"}}

    r1 = graph.invoke(
        {"messages": [{"role": "user", "content": "Send email"}]},
        config=config,
    )
    assert "__interrupt__" in r1

    r2 = graph.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config,
    )

    assert "__interrupt__" not in r2
    assert r2["messages"][-1].content == "Email sent successfully!"


def test_integration_ainvoke_resume_with_reject():
    """Integration test: reject prevents tool execution and feeds back to agent."""
    from langgraph.types import Command

    graph, _ = _build_hitl_agent()

    config = {"configurable": {"thread_id": "integ-resume-reject"}}

    r1 = graph.invoke(
        {"messages": [{"role": "user", "content": "Send email"}]},
        config=config,
    )
    assert "__interrupt__" in r1

    r2 = graph.invoke(
        Command(
            resume={
                "decisions": [{"type": "reject", "message": "Do not send this email"}]
            }
        ),
        config=config,
    )

    assert "__interrupt__" not in r2
    last_content = r2["messages"][-1].content
    assert isinstance(last_content, str) and len(last_content) > 0


def test_integration_ainvoke_resume_with_edit():
    """Integration test: edit modifies tool args before execution."""
    from langgraph.types import Command

    graph, _ = _build_hitl_agent()

    config = {"configurable": {"thread_id": "integ-resume-edit"}}

    r1 = graph.invoke(
        {"messages": [{"role": "user", "content": "Send email"}]},
        config=config,
    )
    assert "__interrupt__" in r1

    r2 = graph.invoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": "edit",
                        "edited_action": {
                            "name": "send_email",
                            "args": {
                                "to": "edited@example.com",
                                "subject": "Edited Subject",
                                "body": "Edited body",
                            },
                        },
                    }
                ]
            }
        ),
        config=config,
    )

    assert "__interrupt__" not in r2
    # The tool should have executed with the edited args
    tool_msg = [m for m in r2["messages"] if m.type == "tool"]
    assert len(tool_msg) > 0, "Expected a tool message after edit+execute"
    assert "edited@example.com" in tool_msg[-1].content


def test_integration_responses_agent_surfaces_interrupt():
    """Integration test: LanggraphResponsesAgent surfaces interrupt in custom_outputs."""
    graph, _ = _build_hitl_agent()
    agent = LanggraphResponsesAgent(graph)

    request = ResponsesAgentRequest(
        input=[Message(role="user", content="Send email to test@example.com")],
        custom_inputs={
            "configurable": {
                "thread_id": "integ-responses-1",
                "user_id": "test_user",
            }
        },
    )

    response = agent.predict(request)

    assert response.custom_outputs is not None
    assert "interrupts" in response.custom_outputs, (
        f"No interrupts in custom_outputs. Keys: {list(response.custom_outputs.keys())}"
    )
    assert len(response.custom_outputs["interrupts"]) == 1

    interrupt = response.custom_outputs["interrupts"][0]
    assert interrupt["action_requests"][0]["name"] == "send_email"
    assert interrupt["review_configs"][0]["allowed_decisions"] == [
        "approve",
        "edit",
        "reject",
    ]

    output_text = response.output[0].content[0]["text"]
    assert "send_email" in output_text


def test_integration_output_text_shows_user_options():
    """Integration test: the user-facing output text communicates available decisions."""
    graph, _ = _build_hitl_agent()
    agent = LanggraphResponsesAgent(graph)

    request = ResponsesAgentRequest(
        input=[Message(role="user", content="Send email")],
        custom_inputs={
            "configurable": {
                "thread_id": "integ-output-options",
                "user_id": "test_user",
            }
        },
    )

    response = agent.predict(request)

    output_text = response.output[0].content[0]["text"]

    assert "Action Approval Required" in output_text, (
        "Missing 'Action Approval Required' header in output"
    )
    assert "send_email" in output_text, "Missing tool name in output"
    assert "approve" in output_text, "Missing 'approve' option in output"
    assert "edit" in output_text, "Missing 'edit' option in output"
    assert "reject" in output_text, "Missing 'reject' option in output"
    assert "natural language" in output_text.lower(), (
        "Missing natural language instructions in output"
    )
    assert "decisions" in output_text, (
        "Missing structured decisions instructions in output"
    )


def test_integration_responses_agent_approve_continuation():
    """Integration test: approve via ResponsesAgent resumes and completes."""
    graph, _ = _build_hitl_agent()
    agent = LanggraphResponsesAgent(graph)

    # Step 1: trigger interrupt
    request = ResponsesAgentRequest(
        input=[Message(role="user", content="Send email")],
        custom_inputs={
            "configurable": {
                "thread_id": "integ-ra-approve",
                "user_id": "test_user",
            }
        },
    )
    r1 = agent.predict(request)
    assert "interrupts" in r1.custom_outputs

    # Step 2: resume with approve
    resume_request = ResponsesAgentRequest(
        input=[],
        custom_inputs={
            "configurable": {
                "thread_id": "integ-ra-approve",
                "user_id": "test_user",
            },
            "decisions": [{"type": "approve"}],
        },
    )
    r2 = agent.predict(resume_request)
    assert "interrupts" not in r2.custom_outputs
    assert "Email sent successfully!" in r2.output[0].content[0]["text"]


def test_integration_responses_agent_reject_continuation():
    """Integration test: reject via ResponsesAgent feeds back without executing tool."""
    graph, _ = _build_hitl_agent()
    agent = LanggraphResponsesAgent(graph)

    # Step 1: trigger interrupt
    request = ResponsesAgentRequest(
        input=[Message(role="user", content="Send email")],
        custom_inputs={
            "configurable": {
                "thread_id": "integ-ra-reject",
                "user_id": "test_user",
            }
        },
    )
    r1 = agent.predict(request)
    assert "interrupts" in r1.custom_outputs

    # Step 2: resume with reject
    resume_request = ResponsesAgentRequest(
        input=[],
        custom_inputs={
            "configurable": {
                "thread_id": "integ-ra-reject",
                "user_id": "test_user",
            },
            "decisions": [{"type": "reject", "message": "Not now, thanks"}],
        },
    )
    r2 = agent.predict(resume_request)
    assert "interrupts" not in r2.custom_outputs
    output_text = r2.output[0].content[0]["text"]
    assert len(output_text) > 0, "Expected non-empty response after reject"


def test_integration_responses_agent_edit_continuation():
    """Integration test: edit via ResponsesAgent modifies args and executes."""
    graph, _ = _build_hitl_agent()
    agent = LanggraphResponsesAgent(graph)

    # Step 1: trigger interrupt
    request = ResponsesAgentRequest(
        input=[Message(role="user", content="Send email")],
        custom_inputs={
            "configurable": {
                "thread_id": "integ-ra-edit",
                "user_id": "test_user",
            }
        },
    )
    r1 = agent.predict(request)
    assert "interrupts" in r1.custom_outputs

    # Step 2: resume with edit
    resume_request = ResponsesAgentRequest(
        input=[],
        custom_inputs={
            "configurable": {
                "thread_id": "integ-ra-edit",
                "user_id": "test_user",
            },
            "decisions": [
                {
                    "type": "edit",
                    "edited_action": {
                        "name": "send_email",
                        "args": {
                            "to": "new-recipient@example.com",
                            "subject": "Updated Subject",
                            "body": "Updated body",
                        },
                    },
                }
            ],
        },
    )
    r2 = agent.predict(resume_request)
    assert "interrupts" not in r2.custom_outputs
    output_text = r2.output[0].content[0]["text"]
    assert len(output_text) > 0, "Expected non-empty response after edit"


# ---------------------------------------------------------------------------
# Async integration tests (Apps deployment path)
#
# These call apredict() / apredict_stream() directly, matching how the
# Databricks Apps AgentServer invokes the agent. Uses asyncio.run() as
# a test harness since the project does not have pytest-asyncio configured.
# ---------------------------------------------------------------------------


def _run_async(coro):
    """Run an async coroutine in a new event loop (test helper)."""
    import asyncio

    return asyncio.run(coro)


def test_async_apredict_surfaces_interrupt():
    """Apps path: apredict() surfaces interrupt with user options in output."""

    async def _test():
        graph, _ = _build_hitl_agent()
        agent = LanggraphResponsesAgent(graph)

        request = ResponsesAgentRequest(
            input=[Message(role="user", content="Send email to test@example.com")],
            custom_inputs={
                "configurable": {
                    "thread_id": "async-integ-interrupt",
                    "user_id": "test_user",
                }
            },
        )

        response = await agent.apredict(request)

        assert response.custom_outputs is not None
        assert "interrupts" in response.custom_outputs, (
            f"No interrupts in custom_outputs. Keys: {list(response.custom_outputs.keys())}"
        )
        assert len(response.custom_outputs["interrupts"]) == 1

        interrupt = response.custom_outputs["interrupts"][0]
        assert interrupt["action_requests"][0]["name"] == "send_email"
        assert interrupt["review_configs"][0]["allowed_decisions"] == [
            "approve",
            "edit",
            "reject",
        ]

        output_text = response.output[0].content[0]["text"]
        assert "Action Approval Required" in output_text
        assert "approve" in output_text
        assert "edit" in output_text
        assert "reject" in output_text

    _run_async(_test())


def test_async_apredict_approve_continuation():
    """Apps path: approve via apredict() resumes and completes."""

    async def _test():
        graph, _ = _build_hitl_agent()
        agent = LanggraphResponsesAgent(graph)

        request = ResponsesAgentRequest(
            input=[Message(role="user", content="Send email")],
            custom_inputs={
                "configurable": {
                    "thread_id": "async-integ-approve",
                    "user_id": "test_user",
                }
            },
        )
        r1 = await agent.apredict(request)
        assert "interrupts" in r1.custom_outputs

        resume_request = ResponsesAgentRequest(
            input=[],
            custom_inputs={
                "configurable": {
                    "thread_id": "async-integ-approve",
                    "user_id": "test_user",
                },
                "decisions": [{"type": "approve"}],
            },
        )
        r2 = await agent.apredict(resume_request)
        assert "interrupts" not in r2.custom_outputs
        assert "Email sent successfully!" in r2.output[0].content[0]["text"]

    _run_async(_test())


def test_async_apredict_reject_continuation():
    """Apps path: reject via apredict() feeds back without executing tool."""

    async def _test():
        graph, _ = _build_hitl_agent()
        agent = LanggraphResponsesAgent(graph)

        request = ResponsesAgentRequest(
            input=[Message(role="user", content="Send email")],
            custom_inputs={
                "configurable": {
                    "thread_id": "async-integ-reject",
                    "user_id": "test_user",
                }
            },
        )
        r1 = await agent.apredict(request)
        assert "interrupts" in r1.custom_outputs

        resume_request = ResponsesAgentRequest(
            input=[],
            custom_inputs={
                "configurable": {
                    "thread_id": "async-integ-reject",
                    "user_id": "test_user",
                },
                "decisions": [{"type": "reject", "message": "Do not send"}],
            },
        )
        r2 = await agent.apredict(resume_request)
        assert "interrupts" not in r2.custom_outputs
        assert len(r2.output[0].content[0]["text"]) > 0

    _run_async(_test())


def test_async_apredict_edit_continuation():
    """Apps path: edit via apredict() modifies args and executes."""

    async def _test():
        graph, _ = _build_hitl_agent()
        agent = LanggraphResponsesAgent(graph)

        request = ResponsesAgentRequest(
            input=[Message(role="user", content="Send email")],
            custom_inputs={
                "configurable": {
                    "thread_id": "async-integ-edit",
                    "user_id": "test_user",
                }
            },
        )
        r1 = await agent.apredict(request)
        assert "interrupts" in r1.custom_outputs

        resume_request = ResponsesAgentRequest(
            input=[],
            custom_inputs={
                "configurable": {
                    "thread_id": "async-integ-edit",
                    "user_id": "test_user",
                },
                "decisions": [
                    {
                        "type": "edit",
                        "edited_action": {
                            "name": "send_email",
                            "args": {
                                "to": "async-edited@example.com",
                                "subject": "Async Edit",
                                "body": "Async body",
                            },
                        },
                    }
                ],
            },
        )
        r2 = await agent.apredict(resume_request)
        assert "interrupts" not in r2.custom_outputs
        assert len(r2.output[0].content[0]["text"]) > 0

    _run_async(_test())


def test_async_apredict_stream_surfaces_interrupt():
    """Apps path: apredict_stream() surfaces interrupt in final event."""

    async def _test():
        graph, _ = _build_hitl_agent()
        agent = LanggraphResponsesAgent(graph)

        request = ResponsesAgentRequest(
            input=[Message(role="user", content="Send email")],
            custom_inputs={
                "configurable": {
                    "thread_id": "async-stream-integ",
                    "user_id": "test_user",
                }
            },
        )

        events = []
        async for event in agent.apredict_stream(request):
            events.append(event)

        final_events = [e for e in events if e.type == "response.output_item.done"]
        assert len(final_events) == 1, (
            f"Expected 1 final event, got {len(final_events)}"
        )

        final_event = final_events[0]
        assert "interrupts" in final_event.custom_outputs, (
            f"No interrupts in streaming custom_outputs. "
            f"Keys: {list(final_event.custom_outputs.keys())}"
        )
        assert len(final_event.custom_outputs["interrupts"]) == 1

        interrupt = final_event.custom_outputs["interrupts"][0]
        assert interrupt["action_requests"][0]["name"] == "send_email"
        assert interrupt["review_configs"][0]["allowed_decisions"] == [
            "approve",
            "edit",
            "reject",
        ]

    _run_async(_test())


# ---------------------------------------------------------------------------
# Swarm-architecture integration tests
#
# These simulate the deployed architecture where agent subgraphs are wrapped
# in create_agent_node_handler inside a parent orchestration graph, verifying
# that HITL interrupts propagate through the handler correctly.
# ---------------------------------------------------------------------------


def _build_swarm_hitl_agent():
    """Build a parent graph wrapping an agent subgraph via
    create_agent_node_handler, mirroring the swarm/supervisor deployment.

    Returns (parent_graph, agent_subgraph).
    """
    from langchain.agents import create_agent
    from langchain.agents.middleware import HumanInTheLoopMiddleware
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    from dao_ai.orchestration.core import create_agent_node_handler
    from dao_ai.state import AgentState, Context

    @tool
    def send_email(to: str, subject: str, body: str) -> str:
        """Send an email to someone."""
        return f"Email sent to {to}"

    class FakeToolChatModel(BaseChatModel):
        """Stateless chat model: returns a tool call when no tool result is
        present in the conversation; returns a final text response otherwise.
        Each call generates a fresh AIMessage with a unique ID to avoid
        add_messages deduplication across multi-turn conversations."""
        tool_call_response: AIMessage
        final_response: AIMessage

        @property
        def _llm_type(self) -> str:
            return "fake-swarm-chat"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            import uuid

            from langchain_core.messages import ToolMessage

            has_tool_result = any(isinstance(m, ToolMessage) for m in messages)
            template = self.final_response if has_tool_result else self.tool_call_response
            msg = AIMessage(
                content=template.content,
                tool_calls=list(template.tool_calls) if template.tool_calls else [],
                id=str(uuid.uuid4()),
            )
            return ChatResult(generations=[ChatGeneration(message=msg)])

        def bind_tools(self, tools, **kwargs):
            return self

    model = FakeToolChatModel(
        tool_call_response=AIMessage(
            content="Sending the email now.",
            tool_calls=[
                {
                    "name": "send_email",
                    "args": {
                        "to": "test@example.com",
                        "subject": "Test",
                        "body": "Hello",
                    },
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
        final_response=AIMessage(content="Email sent successfully!"),
    )

    agent_subgraph = create_agent(
        name="test_agent",
        model=model,
        tools=[send_email],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "send_email": {
                        "allowed_decisions": ["approve", "edit", "reject"],
                        "description": "Review email before sending",
                    }
                },
            ),
        ],
        checkpointer=InMemorySaver(),
        state_schema=AgentState,
        context_schema=Context,
    )

    handler = create_agent_node_handler(
        agent_name="test_agent",
        agent=agent_subgraph,
        output_mode="last_message",
    )

    workflow = StateGraph(
        AgentState,
        input=AgentState,
        output=AgentState,
        context_schema=Context,
    )
    workflow.add_node("test_agent", handler)
    workflow.add_edge(START, "test_agent")
    workflow.add_edge("test_agent", END)

    parent_graph = workflow.compile(checkpointer=InMemorySaver())

    return parent_graph, agent_subgraph


def test_swarm_ainvoke_returns_interrupt():
    """Swarm arch: parent graph ainvoke returns __interrupt__ when subgraph
    HITL middleware fires."""

    async def _test():
        graph, _ = _build_swarm_hitl_agent()

        from dao_ai.state import Context

        config = {"configurable": {"thread_id": "swarm-integ-1", "user_id": "test"}}
        context = Context(thread_id="swarm-integ-1", user_id="test")

        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": "Send email to test"}]},
            config=config,
            context=context,
        )

        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert "__interrupt__" in result, (
            f"Expected __interrupt__ in result. Keys: {list(result.keys())}"
        )
        assert len(result["__interrupt__"]) == 1

        interrupt = result["__interrupt__"][0]
        assert interrupt.value["action_requests"][0]["name"] == "send_email"

    _run_async(_test())


def test_swarm_approve_continuation():
    """Swarm arch: approve via Command(resume=...) resumes and completes."""

    async def _test():
        from langgraph.types import Command

        from dao_ai.state import Context

        graph, _ = _build_swarm_hitl_agent()

        config = {"configurable": {"thread_id": "swarm-integ-approve", "user_id": "test"}}
        context = Context(thread_id="swarm-integ-approve", user_id="test")

        r1 = await graph.ainvoke(
            {"messages": [{"role": "user", "content": "Send email"}]},
            config=config,
            context=context,
        )
        assert "__interrupt__" in r1

        r2 = await graph.ainvoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config=config,
            context=context,
        )

        assert "__interrupt__" not in r2
        assert r2["messages"][-1].content == "Email sent successfully!"

    _run_async(_test())


def test_swarm_second_invocation_triggers_interrupt():
    """Regression test: HITL must trigger on BOTH the first AND second query
    when using the same thread_id. The subgraph's InMemorySaver must not
    accumulate state across separate parent-graph invocations."""

    async def _test():
        from langgraph.types import Command

        from dao_ai.state import Context

        graph, _ = _build_swarm_hitl_agent()

        config = {"configurable": {"thread_id": "swarm-integ-second", "user_id": "test"}}
        context = Context(thread_id="swarm-integ-second", user_id="test")

        # --- First query: interrupt + approve ---
        r1 = await graph.ainvoke(
            {"messages": [{"role": "user", "content": "Send email to alice"}]},
            config=config,
            context=context,
        )
        assert "__interrupt__" in r1, "First invocation must trigger interrupt"

        r2 = await graph.ainvoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config=config,
            context=context,
        )
        assert "__interrupt__" not in r2, "Resume should complete without interrupt"
        assert "Email sent" in r2["messages"][-1].content

        # --- Second query (same thread): must ALSO interrupt ---
        r3 = await graph.ainvoke(
            {"messages": [{"role": "user", "content": "Send email to bob"}]},
            config=config,
            context=context,
        )
        assert "__interrupt__" in r3, (
            "Second invocation must also trigger interrupt. "
            f"Keys: {list(r3.keys())}"
        )

        # Approve the second interrupt
        r4 = await graph.ainvoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config=config,
            context=context,
        )
        assert "__interrupt__" not in r4, "Second resume should complete without interrupt"
        assert "Email sent" in r4["messages"][-1].content

    _run_async(_test())


def test_swarm_responses_agent_surfaces_interrupt():
    """Swarm arch: LanggraphResponsesAgent surfaces interrupt from a
    handler-wrapped subgraph in custom_outputs."""
    graph, _ = _build_swarm_hitl_agent()
    agent = LanggraphResponsesAgent(graph)

    request = ResponsesAgentRequest(
        input=[Message(role="user", content="Send email to test@example.com")],
        custom_inputs={
            "configurable": {
                "thread_id": "swarm-ra-integ-1",
                "user_id": "test_user",
            }
        },
    )

    response = agent.predict(request)

    assert response.custom_outputs is not None
    assert "interrupts" in response.custom_outputs, (
        f"No interrupts in custom_outputs. Keys: {list(response.custom_outputs.keys())}"
    )
    assert len(response.custom_outputs["interrupts"]) == 1

    interrupt = response.custom_outputs["interrupts"][0]
    assert interrupt["action_requests"][0]["name"] == "send_email"


def test_swarm_async_apredict_surfaces_interrupt():
    """Swarm arch: apredict() surfaces interrupt with user options in output."""

    async def _test():
        graph, _ = _build_swarm_hitl_agent()
        agent = LanggraphResponsesAgent(graph)

        request = ResponsesAgentRequest(
            input=[Message(role="user", content="Send email to test@example.com")],
            custom_inputs={
                "configurable": {
                    "thread_id": "swarm-async-integ-1",
                    "user_id": "test_user",
                }
            },
        )

        response = await agent.apredict(request)

        assert response.custom_outputs is not None
        assert "interrupts" in response.custom_outputs, (
            f"No interrupts in custom_outputs. Keys: {list(response.custom_outputs.keys())}"
        )
        assert len(response.custom_outputs["interrupts"]) == 1

        interrupt = response.custom_outputs["interrupts"][0]
        assert interrupt["action_requests"][0]["name"] == "send_email"
        assert interrupt["review_configs"][0]["allowed_decisions"] == [
            "approve",
            "edit",
            "reject",
        ]

        output_text = response.output[0].content[0]["text"]
        assert "Action Approval Required" in output_text

    _run_async(_test())


def test_swarm_async_apredict_stream_surfaces_interrupt():
    """Swarm arch: apredict_stream() surfaces interrupt in final event."""

    async def _test():
        graph, _ = _build_swarm_hitl_agent()
        agent = LanggraphResponsesAgent(graph)

        request = ResponsesAgentRequest(
            input=[Message(role="user", content="Send email")],
            custom_inputs={
                "configurable": {
                    "thread_id": "swarm-async-stream-integ",
                    "user_id": "test_user",
                }
            },
        )

        events = []
        async for event in agent.apredict_stream(request):
            events.append(event)

        final_events = [e for e in events if e.type == "response.output_item.done"]
        assert len(final_events) == 1, (
            f"Expected 1 final event, got {len(final_events)}"
        )

        final_event = final_events[0]
        assert "interrupts" in final_event.custom_outputs, (
            f"No interrupts in streaming custom_outputs. "
            f"Keys: {list(final_event.custom_outputs.keys())}"
        )
        assert len(final_event.custom_outputs["interrupts"]) == 1

        interrupt = final_event.custom_outputs["interrupts"][0]
        assert interrupt["action_requests"][0]["name"] == "send_email"

    _run_async(_test())
