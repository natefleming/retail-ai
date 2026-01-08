"""Unit tests for LanggraphResponsesAgent structured output handling (no actual inference)."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage
from mlflow.types.responses import Message, ResponsesAgentRequest
from pydantic import BaseModel, Field

from dao_ai.models import LanggraphResponsesAgent


# Test schemas
class TestSchema(BaseModel):
    """Test schema for unit tests."""

    value: str = Field(description="A test value")
    count: int = Field(description="A count")


@dataclass
class TestDataclass:
    """Test dataclass."""

    name: str
    amount: float


@pytest.fixture
def mock_graph():
    """Create a mock CompiledStateGraph."""
    graph = Mock()
    graph.checkpointer = None  # No checkpointer by default
    return graph


def test_responses_agent_extracts_pydantic_structured_response(mock_graph):
    """Test that ResponsesAgent extracts and serializes Pydantic structured output (default: message content)."""
    # Create agent without config (defaults to as_custom_output=False)
    agent = LanggraphResponsesAgent(mock_graph)

    # Mock the graph response with structured_response
    test_response = TestSchema(value="test", count=42)
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "messages": [AIMessage(content="Here's your structured data")],
            "structured_response": test_response,
        }
    )

    # Mock aget_state to return non-interrupted state
    mock_state = Mock()
    mock_state.interrupts = ()  # Empty tuple means not interrupted
    mock_graph.aget_state = AsyncMock(return_value=mock_state)

    # Mock _build_custom_outputs_async (used by apredict)
    with patch.object(
        agent,
        "_build_custom_outputs_async",
        new_callable=AsyncMock,
        return_value={"configurable": {}},
    ):
        # Mock _convert methods
        with patch.object(
            agent, "_convert_request_to_langchain_messages", return_value=[]
        ):
            with patch.object(
                agent,
                "_convert_request_to_context",
                return_value=Mock(
                    thread_id="test",
                    model_dump=Mock(return_value={"thread_id": "test"}),
                ),
            ):
                with patch.object(
                    agent, "_extract_session_from_request", return_value={}
                ):
                    # Create request
                    request = ResponsesAgentRequest(
                        input=[Message(role="user", content="test")]
                    )

                    # Predict
                    response = agent.predict(request)

                    # Default: structured_response in message content as JSON (not custom_outputs)
                    assert response.output is not None
                    message_text = response.output[0].content[0]["text"]

                    import json

                    structured = json.loads(message_text)
                    assert isinstance(structured, dict)
                    assert structured == {"value": "test", "count": 42}


def test_responses_agent_extracts_dataclass_structured_response(mock_graph):
    """Test that ResponsesAgent extracts and serializes dataclass structured output."""
    # Create agent
    agent = LanggraphResponsesAgent(mock_graph)

    # Mock the graph response with dataclass structured_response
    test_response = TestDataclass(name="test", amount=99.99)
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "messages": [AIMessage(content="Here's your data")],
            "structured_response": test_response,
        }
    )

    # Mock aget_state to return non-interrupted state
    mock_state = Mock()
    mock_state.interrupts = ()  # Empty tuple means not interrupted
    mock_graph.aget_state = AsyncMock(return_value=mock_state)

    # Mock _build_custom_outputs_async (used by apredict)
    with patch.object(
        agent,
        "_build_custom_outputs_async",
        new_callable=AsyncMock,
        return_value={"configurable": {}},
    ):
        # Mock _convert methods
        with patch.object(
            agent, "_convert_request_to_langchain_messages", return_value=[]
        ):
            with patch.object(
                agent,
                "_convert_request_to_context",
                return_value=Mock(
                    thread_id="test",
                    model_dump=Mock(return_value={"thread_id": "test"}),
                ),
            ):
                with patch.object(
                    agent, "_extract_session_from_request", return_value={}
                ):
                    # Create request
                    request = ResponsesAgentRequest(
                        input=[Message(role="user", content="test")]
                    )

                    # Predict
                    response = agent.predict(request)

                    # Default: structured_response in message content as JSON
                    assert response.output is not None
                    message_text = response.output[0].content[0]["text"]

                    import json

                    structured = json.loads(message_text)
                    assert isinstance(structured, dict)
                    assert structured == {"name": "test", "amount": 99.99}


def test_responses_agent_no_structured_response(mock_graph):
    """Test that ResponsesAgent handles absence of structured_response."""
    # Create agent
    agent = LanggraphResponsesAgent(mock_graph)

    # Mock the graph response WITHOUT structured_response
    mock_graph.ainvoke = AsyncMock(
        return_value={"messages": [AIMessage(content="Regular response")]}
    )

    # Mock aget_state to return non-interrupted state
    mock_state = Mock()
    mock_state.interrupts = ()  # Empty tuple means not interrupted
    mock_graph.aget_state = AsyncMock(return_value=mock_state)

    # Mock _build_custom_outputs_async (used by apredict)
    with patch.object(
        agent,
        "_build_custom_outputs_async",
        new_callable=AsyncMock,
        return_value={"configurable": {}},
    ):
        # Mock _convert methods
        with patch.object(
            agent, "_convert_request_to_langchain_messages", return_value=[]
        ):
            with patch.object(
                agent,
                "_convert_request_to_context",
                return_value=Mock(
                    thread_id="test",
                    model_dump=Mock(return_value={"thread_id": "test"}),
                ),
            ):
                with patch.object(
                    agent, "_extract_session_from_request", return_value={}
                ):
                    # Create request
                    request = ResponsesAgentRequest(
                        input=[Message(role="user", content="test")]
                    )

                    # Predict
                    response = agent.predict(request)

                    # Verify structured_response is NOT in custom_outputs
                    assert response.custom_outputs is not None
                    assert "structured_response" not in response.custom_outputs


def test_responses_agent_dict_structured_response(mock_graph):
    """Test that ResponsesAgent handles dict structured_response (JSON schema) - default: message content."""
    # Create agent without config (defaults to as_custom_output=False)
    agent = LanggraphResponsesAgent(mock_graph)

    # Mock the graph response with dict structured_response
    test_response = {"key": "value", "number": 123}
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "messages": [AIMessage(content="Here's your JSON")],
            "structured_response": test_response,
        }
    )

    # Mock aget_state to return non-interrupted state
    mock_state = Mock()
    mock_state.interrupts = ()  # Empty tuple means not interrupted
    mock_graph.aget_state = AsyncMock(return_value=mock_state)

    # Mock _build_custom_outputs_async (used by apredict)
    with patch.object(
        agent,
        "_build_custom_outputs_async",
        new_callable=AsyncMock,
        return_value={"configurable": {}},
    ):
        # Mock _convert methods
        with patch.object(
            agent, "_convert_request_to_langchain_messages", return_value=[]
        ):
            with patch.object(
                agent,
                "_convert_request_to_context",
                return_value=Mock(
                    thread_id="test",
                    model_dump=Mock(return_value={"thread_id": "test"}),
                ),
            ):
                with patch.object(
                    agent, "_extract_session_from_request", return_value={}
                ):
                    # Create request
                    request = ResponsesAgentRequest(
                        input=[Message(role="user", content="test")]
                    )

                    # Predict
                    response = agent.predict(request)

                    # Default: structured_response in message content as JSON
                    assert response.output is not None
                    message_text = response.output[0].content[0]["text"]

                    import json

                    structured = json.loads(message_text)
                    assert structured == test_response


def test_responses_agent_streaming_extracts_structured_response(mock_graph):
    """Test that ResponsesAgent streaming places structured_response in message content (default)."""
    # Create agent without config (defaults to as_custom_output=False)
    agent = LanggraphResponsesAgent(mock_graph)

    # Mock the streaming response
    async def mock_astream(*args, **kwargs):
        # Simulate streaming messages
        yield (("agent",), "messages", [AIMessage(content="Chunk 1")])
        yield (("agent",), "messages", [AIMessage(content=" Chunk 2")])

    mock_graph.astream = mock_astream

    # Mock aget_state for interrupt check (first call) and final state (second call)
    test_response = TestSchema(value="streamed", count=99)

    # First call: non-interrupted state
    initial_state = Mock()
    initial_state.interrupts = ()  # Not interrupted

    # Second call: final state with structured_response
    final_state = Mock()
    final_state.values = {
        "messages": [AIMessage(content="Chunk 1 Chunk 2")],
        "structured_response": test_response,
    }

    # Configure aget_state to return different values on successive calls
    mock_graph.aget_state = AsyncMock(side_effect=[initial_state, final_state])
    mock_graph.checkpointer = (
        Mock()
    )  # Enable checkpointer for structured_response extraction

    # Mock _build_custom_outputs_async (used by apredict_stream)
    with patch.object(
        agent,
        "_build_custom_outputs_async",
        new_callable=AsyncMock,
        return_value={"configurable": {}},
    ):
        # Mock _convert methods
        with patch.object(
            agent, "_convert_request_to_langchain_messages", return_value=[]
        ):
            with patch.object(
                agent,
                "_convert_request_to_context",
                return_value=Mock(
                    thread_id="test",
                    model_dump=Mock(return_value={"thread_id": "test"}),
                ),
            ):
                with patch.object(
                    agent, "_extract_session_from_request", return_value={}
                ):
                    # Create request
                    request = ResponsesAgentRequest(
                        input=[Message(role="user", content="test")]
                    )

                    # Stream
                    events = list(agent.predict_stream(request))

                    # Find final event
                    final_event = None
                    for event in events:
                        if event.type == "response.output_item.done":
                            final_event = event
                            break

                    assert final_event is not None

                    # Default: structured_response in message content as JSON
                    message_text = final_event.item["content"][0]["text"]
                    # Should contain both streamed text and JSON
                    assert "Chunk 1 Chunk 2" in message_text

                    import json

                    # Extract JSON from end of message
                    json_start = message_text.rfind("{")
                    structured = json.loads(message_text[json_start:])
                    assert structured == {"value": "streamed", "count": 99}


def test_responses_agent_streaming_no_structured_response(mock_graph):
    """Test that ResponsesAgent streaming handles absence of structured_response."""
    # Create agent
    agent = LanggraphResponsesAgent(mock_graph)

    # Mock the streaming response
    async def mock_astream(*args, **kwargs):
        yield (("agent",), "messages", [AIMessage(content="Regular response")])

    mock_graph.astream = mock_astream

    # Mock aget_state to return final state WITHOUT structured_response
    mock_state = Mock()
    mock_state.values = {"messages": [AIMessage(content="Regular response")]}
    mock_graph.aget_state = AsyncMock(return_value=mock_state)
    mock_graph.checkpointer = Mock()  # Enable checkpointer

    # Mock _build_custom_outputs_async (used by apredict_stream)
    with patch.object(
        agent,
        "_build_custom_outputs_async",
        new_callable=AsyncMock,
        return_value={"configurable": {}},
    ):
        # Mock _convert methods
        with patch.object(
            agent, "_convert_request_to_langchain_messages", return_value=[]
        ):
            with patch.object(
                agent,
                "_convert_request_to_context",
                return_value=Mock(
                    thread_id="test",
                    model_dump=Mock(return_value={"thread_id": "test"}),
                ),
            ):
                with patch.object(
                    agent, "_extract_session_from_request", return_value={}
                ):
                    # Create request
                    request = ResponsesAgentRequest(
                        input=[Message(role="user", content="test")]
                    )

                    # Stream
                    events = list(agent.predict_stream(request))

                    # Find final event
                    final_event = None
                    for event in events:
                        if event.type == "response.output_item.done":
                            final_event = event
                            break

                    assert final_event is not None

                    # Verify structured_response is NOT in custom_outputs
                    assert final_event.custom_outputs is not None
                    assert "structured_response" not in final_event.custom_outputs
