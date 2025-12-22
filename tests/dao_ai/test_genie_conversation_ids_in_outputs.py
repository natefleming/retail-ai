"""
Tests for including genie conversation IDs in custom_outputs for ResponsesAgent.

This test module validates that genie conversation IDs are properly extracted
from the graph state and included in the custom_outputs of ResponsesAgent responses.

The output structure is:
{
    "configurable": {"thread_id": "...", "conversation_id": "...", ...},
    "session": {
        "genie": {
            "spaces": {
                "<space_id>": {
                    "conversation_id": "<conv_id>",
                    "cache_hit": false,
                    "follow_up_questions": []
                }
            }
        }
    }
}
"""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
)
from mlflow.types.responses_helpers import Message

from dao_ai.models import (
    LanggraphResponsesAgent,
    get_genie_conversation_ids_from_state,
    get_state_snapshot,
)


class TestGenieConversationIdsInOutputs:
    """Test suite for genie conversation IDs in custom_outputs."""

    def test_get_state_snapshot_success(self):
        """Test successful retrieval of state snapshot."""
        # Mock the graph and checkpointer
        mock_graph = MagicMock(spec=CompiledStateGraph)
        mock_graph.checkpointer = MagicMock()

        # Mock state snapshot with new structure
        mock_state_snapshot = MagicMock()
        mock_state_snapshot.values = {
            "genie_conversation_ids": {
                "space_123": "conv_abc",
                "space_456": "conv_def",
            }
        }

        # Mock async aget_state
        async def mock_aget_state(*args, **kwargs):
            return mock_state_snapshot

        mock_graph.aget_state = mock_aget_state

        # Test the function
        result = get_state_snapshot(mock_graph, "thread_123")

        # Verify
        assert result is not None
        assert result.values["genie_conversation_ids"] == {
            "space_123": "conv_abc",
            "space_456": "conv_def",
        }

    def test_get_genie_conversation_ids_from_state_success(self):
        """Test successful extraction of genie_conversation_ids from state snapshot."""
        # Mock state snapshot
        mock_state_snapshot = MagicMock()
        mock_state_snapshot.values = {
            "genie_conversation_ids": {
                "space_123": "conv_abc",
                "space_456": "conv_def",
            }
        }

        # Test the function
        result = get_genie_conversation_ids_from_state(mock_state_snapshot)

        # Verify
        assert result == {
            "space_123": "conv_abc",
            "space_456": "conv_def",
        }

    def test_get_state_snapshot_no_checkpointer(self):
        """Test when graph has no checkpointer."""
        mock_graph = MagicMock(spec=CompiledStateGraph)
        mock_graph.checkpointer = None

        result = get_state_snapshot(mock_graph, "thread_123")

        assert result is None

    def test_get_genie_conversation_ids_from_state_none_snapshot(self):
        """Test when state snapshot is None."""
        result = get_genie_conversation_ids_from_state(None)

        assert result == {}

    def test_get_state_snapshot_no_state(self):
        """Test when no state exists for thread_id."""
        mock_graph = MagicMock(spec=CompiledStateGraph)
        mock_graph.checkpointer = MagicMock()

        async def mock_aget_state(*args, **kwargs):
            return None

        mock_graph.aget_state = mock_aget_state

        result = get_state_snapshot(mock_graph, "thread_123")

        assert result is None

    def test_get_genie_conversation_ids_from_state_empty_dict(self):
        """Test when state exists but genie_conversation_ids is empty."""
        mock_state_snapshot = MagicMock()
        mock_state_snapshot.values = {"genie_conversation_ids": {}}

        result = get_genie_conversation_ids_from_state(mock_state_snapshot)

        assert result == {}

    def test_get_genie_conversation_ids_from_state_missing_key(self):
        """Test when state exists but genie_conversation_ids key is missing."""
        mock_state_snapshot = MagicMock()
        mock_state_snapshot.values = {"messages": [], "other_field": "value"}

        result = get_genie_conversation_ids_from_state(mock_state_snapshot)

        assert result == {}

    def test_get_state_snapshot_exception_handling(self):
        """Test exception handling in get_state_snapshot."""
        mock_graph = MagicMock(spec=CompiledStateGraph)
        mock_graph.checkpointer = MagicMock()

        async def mock_aget_state(*args, **kwargs):
            raise Exception("Database error")

        mock_graph.aget_state = mock_aget_state

        result = get_state_snapshot(mock_graph, "thread_123")

        assert result is None

    def test_get_genie_conversation_ids_from_state_exception_handling(self):
        """Test exception handling in get_genie_conversation_ids_from_state."""
        mock_state_snapshot = MagicMock()
        mock_state_snapshot.values.get.side_effect = Exception("Access error")

        result = get_genie_conversation_ids_from_state(mock_state_snapshot)

        assert result == {}

    def test_responses_agent_predict_includes_genie_conversation_ids(self):
        """Test that predict() includes genie conversation IDs in session.genie.spaces."""
        # Create mock graph
        mock_graph = MagicMock(spec=CompiledStateGraph)
        mock_graph.checkpointer = MagicMock()

        # Mock ainvoke response
        async def mock_ainvoke(*args, **kwargs):
            return {
                "messages": [
                    HumanMessage(content="Test question"),
                    AIMessage(content="Test response"),
                ]
            }

        mock_graph.ainvoke = mock_ainvoke

        # Mock state with genie_conversation_ids
        mock_state_snapshot = MagicMock()
        mock_state_snapshot.values = {
            "genie_conversation_ids": {"space_123": "conv_abc"}
        }

        async def mock_aget_state(*args, **kwargs):
            return mock_state_snapshot

        mock_graph.aget_state = mock_aget_state

        # Create agent
        agent = LanggraphResponsesAgent(mock_graph)

        # Create request with configurable structure
        request = ResponsesAgentRequest(
            input=[Message(role="user", content="Test question")],
            custom_inputs={
                "configurable": {"thread_id": "test_thread_123", "user_id": "test_user"}
            },
        )

        # Execute
        response = agent.predict(request)

        # Verify
        assert isinstance(response, ResponsesAgentResponse)
        # Check new output structure: session.genie.spaces
        assert "session" in response.custom_outputs
        assert "genie" in response.custom_outputs["session"]
        assert "spaces" in response.custom_outputs["session"]["genie"]
        spaces = response.custom_outputs["session"]["genie"]["spaces"]
        assert "space_123" in spaces
        assert spaces["space_123"]["conversation_id"] == "conv_abc"

    def test_responses_agent_predict_no_genie_conversation_ids(self):
        """Test that predict() works when no genie_conversation_ids exist."""
        # Create mock graph
        mock_graph = MagicMock(spec=CompiledStateGraph)
        mock_graph.checkpointer = MagicMock()

        # Mock ainvoke response
        async def mock_ainvoke(*args, **kwargs):
            return {
                "messages": [
                    HumanMessage(content="Test question"),
                    AIMessage(content="Test response"),
                ]
            }

        mock_graph.ainvoke = mock_ainvoke

        # Mock state without genie_conversation_ids
        async def mock_aget_state(*args, **kwargs):
            mock_snapshot = MagicMock()
            mock_snapshot.values = {}
            return mock_snapshot

        mock_graph.aget_state = mock_aget_state

        # Create agent
        agent = LanggraphResponsesAgent(mock_graph)

        # Create request with configurable structure
        request = ResponsesAgentRequest(
            input=[Message(role="user", content="Test question")],
            custom_inputs={
                "configurable": {"thread_id": "test_thread_123", "user_id": "test_user"}
            },
        )

        # Execute
        response = agent.predict(request)

        # Verify - session.genie.spaces should be empty or not have any spaces
        assert isinstance(response, ResponsesAgentResponse)
        session = response.custom_outputs.get("session", {})
        genie = session.get("genie", {})
        spaces = genie.get("spaces", {})
        assert spaces == {}

    def test_responses_agent_predict_stream_includes_genie_conversation_ids(self):
        """Test that predict_stream() includes genie conversation IDs in final event."""
        # Create mock graph
        mock_graph = MagicMock(spec=CompiledStateGraph)
        mock_graph.checkpointer = MagicMock()

        # Mock astream response
        async def mock_astream(*args, **kwargs):
            yield (
                ("agent",),
                "messages",
                [AIMessage(content="Test response chunk")],
            )

        mock_graph.astream = mock_astream

        # Mock state with genie_conversation_ids
        mock_state_snapshot = MagicMock()
        mock_state_snapshot.values = {
            "genie_conversation_ids": {"space_456": "conv_def"}
        }

        async def mock_aget_state(*args, **kwargs):
            return mock_state_snapshot

        mock_graph.aget_state = mock_aget_state

        # Create agent
        agent = LanggraphResponsesAgent(mock_graph)

        # Create request with configurable structure
        request = ResponsesAgentRequest(
            input=[Message(role="user", content="Test question")],
            custom_inputs={
                "configurable": {"thread_id": "test_thread_456", "user_id": "test_user"}
            },
        )

        # Execute and collect events
        events = list(agent.predict_stream(request))

        # Verify - last event should be the done event with custom_outputs
        assert len(events) > 0
        last_event = events[-1]
        assert last_event.type == "response.output_item.done"
        assert hasattr(last_event, "custom_outputs")
        # Check new output structure: session.genie.spaces
        assert "session" in last_event.custom_outputs
        assert "genie" in last_event.custom_outputs["session"]
        assert "spaces" in last_event.custom_outputs["session"]["genie"]
        spaces = last_event.custom_outputs["session"]["genie"]["spaces"]
        assert "space_456" in spaces
        assert spaces["space_456"]["conversation_id"] == "conv_def"

    def test_responses_agent_predict_stream_no_genie_conversation_ids(self):
        """Test that predict_stream() works when no genie_conversation_ids exist."""
        # Create mock graph
        mock_graph = MagicMock(spec=CompiledStateGraph)
        mock_graph.checkpointer = MagicMock()

        # Mock astream response
        async def mock_astream(*args, **kwargs):
            yield (
                ("agent",),
                "messages",
                [AIMessage(content="Test response chunk")],
            )

        mock_graph.astream = mock_astream

        # Mock state without genie_conversation_ids
        async def mock_aget_state(*args, **kwargs):
            mock_snapshot = MagicMock()
            mock_snapshot.values = {}
            return mock_snapshot

        mock_graph.aget_state = mock_aget_state

        # Create agent
        agent = LanggraphResponsesAgent(mock_graph)

        # Create request with configurable structure
        request = ResponsesAgentRequest(
            input=[Message(role="user", content="Test question")],
            custom_inputs={
                "configurable": {"thread_id": "test_thread_456", "user_id": "test_user"}
            },
        )

        # Execute and collect events
        events = list(agent.predict_stream(request))

        # Verify - last event should not have genie spaces
        assert len(events) > 0
        last_event = events[-1]
        assert last_event.type == "response.output_item.done"
        assert hasattr(last_event, "custom_outputs")
        session = last_event.custom_outputs.get("session", {})
        genie = session.get("genie", {})
        spaces = genie.get("spaces", {})
        assert spaces == {}
