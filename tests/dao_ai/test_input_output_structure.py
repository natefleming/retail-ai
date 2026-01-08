"""
Tests for the input/output structure with thread_id/conversation_id interchangeability.

This module tests the input/output structure:
- thread_id and conversation_id are interchangeable (conversation_id takes precedence)
- Both are in configurable section (same value)
- session contains only accumulated state (genie conversations, etc.)
- Custom outputs can be copy-pasted as inputs
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mlflow.types.agent import ChatContext
from mlflow.types.responses import ResponsesAgentRequest
from mlflow.types.responses_helpers import Message

from dao_ai.models import LanggraphResponsesAgent
from dao_ai.state import (
    Context,
    GenieSpaceState,
    GenieState,
    SessionState,
)

# =============================================================================
# State Model Tests
# =============================================================================


@pytest.mark.unit
class TestGenieSpaceState:
    """Tests for GenieSpaceState model."""

    def test_create_with_defaults(self) -> None:
        """Test creating GenieSpaceState with only required fields."""
        state = GenieSpaceState(conversation_id="conv_123")

        assert state.conversation_id == "conv_123"
        assert state.cache_hit is False
        assert state.cache_key is None
        assert state.follow_up_questions == []
        assert state.last_query is None
        assert state.last_query_time is None

    def test_create_with_all_fields(self) -> None:
        """Test creating GenieSpaceState with all fields."""
        from datetime import datetime

        now = datetime.now()
        state = GenieSpaceState(
            conversation_id="conv_123",
            cache_hit=True,
            cache_key="key_abc",
            follow_up_questions=["What about Q2?", "Show by region"],
            last_query="Total sales",
            last_query_time=now,
        )

        assert state.conversation_id == "conv_123"
        assert state.cache_hit is True
        assert state.cache_key == "key_abc"
        assert state.follow_up_questions == ["What about Q2?", "Show by region"]
        assert state.last_query == "Total sales"
        assert state.last_query_time == now


@pytest.mark.unit
class TestGenieState:
    """Tests for GenieState model."""

    def test_empty_state(self) -> None:
        """Test empty GenieState."""
        state = GenieState()
        assert state.spaces == {}
        assert state.get_conversation_id("space_123") is None

    def test_get_conversation_id(self) -> None:
        """Test getting conversation ID for a space."""
        state = GenieState(
            spaces={
                "space_123": GenieSpaceState(conversation_id="conv_456"),
            }
        )

        assert state.get_conversation_id("space_123") == "conv_456"
        assert state.get_conversation_id("nonexistent") is None

    def test_update_space(self) -> None:
        """Test updating/creating a space."""
        state = GenieState()

        state.update_space(
            space_id="space_123",
            conversation_id="conv_456",
            cache_hit=True,
            follow_up_questions=["Q1", "Q2"],
            last_query="What's the total?",
        )

        assert "space_123" in state.spaces
        space = state.spaces["space_123"]
        assert space.conversation_id == "conv_456"
        assert space.cache_hit is True
        assert space.follow_up_questions == ["Q1", "Q2"]
        assert space.last_query == "What's the total?"
        assert space.last_query_time is not None


@pytest.mark.unit
class TestSessionState:
    """Tests for SessionState model."""

    def test_empty_session(self) -> None:
        """Test empty SessionState."""
        session = SessionState()
        assert session.genie is not None
        assert session.genie.spaces == {}

    def test_session_with_genie(self) -> None:
        """Test SessionState with Genie data."""
        session = SessionState(
            genie=GenieState(
                spaces={
                    "space_a": GenieSpaceState(conversation_id="conv_1"),
                    "space_b": GenieSpaceState(conversation_id="conv_2"),
                }
            )
        )

        assert session.genie.get_conversation_id("space_a") == "conv_1"
        assert session.genie.get_conversation_id("space_b") == "conv_2"


# =============================================================================
# LanggraphResponsesAgent Context Conversion Tests
# =============================================================================


@pytest.mark.unit
class TestContextConversion:
    """Tests for LanggraphResponsesAgent._convert_request_to_context."""

    def _create_agent(self) -> LanggraphResponsesAgent:
        """Create a mock LanggraphResponsesAgent."""
        mock_graph = MagicMock()
        return LanggraphResponsesAgent(mock_graph)

    def test_conversation_id_maps_to_thread_id(self) -> None:
        """Test that conversation_id in input maps to thread_id internally."""
        agent = self._create_agent()

        request = ResponsesAgentRequest(
            input=[Message(role="user", content="Hello")],
            custom_inputs={
                "configurable": {
                    "conversation_id": "my_conv_123",
                    "user_id": "test_user",
                }
            },
        )

        context = agent._convert_request_to_context(request)

        # conversation_id should be mapped to thread_id
        assert context.thread_id == "my_conv_123"
        assert context.user_id == "test_user"

    def test_thread_id_works_on_input(self) -> None:
        """Test that thread_id can also be used on input (interchangeable)."""
        agent = self._create_agent()

        request = ResponsesAgentRequest(
            input=[Message(role="user", content="Hello")],
            custom_inputs={
                "configurable": {
                    "thread_id": "my_thread_456",
                    "user_id": "test_user",
                }
            },
        )

        context = agent._convert_request_to_context(request)

        # thread_id should work directly
        assert context.thread_id == "my_thread_456"
        assert context.user_id == "test_user"

    def test_conversation_id_takes_precedence(self) -> None:
        """Test that conversation_id takes precedence if both are provided."""
        agent = self._create_agent()

        request = ResponsesAgentRequest(
            input=[Message(role="user", content="Hello")],
            custom_inputs={
                "configurable": {
                    "thread_id": "thread_loses",
                    "conversation_id": "conv_wins",
                    "user_id": "test_user",
                }
            },
        )

        context = agent._convert_request_to_context(request)

        # conversation_id should take precedence (Databricks vocabulary)
        assert context.thread_id == "conv_wins"

    def test_user_id_normalization(self) -> None:
        """Test that user_id with dots is normalized."""
        agent = self._create_agent()

        request = ResponsesAgentRequest(
            input=[Message(role="user", content="Hello")],
            custom_inputs={
                "configurable": {
                    "conversation_id": "conv_1",
                    "user_id": "nate.fleming@databricks.com",
                }
            },
        )

        context = agent._convert_request_to_context(request)

        # Dots should be replaced with underscores
        assert context.user_id == "nate_fleming@databricks_com"

    def test_chat_context_conversation_id(self) -> None:
        """Test conversation_id from ChatContext (Databricks)."""
        agent = self._create_agent()

        # ChatContext comes from Databricks when deployed
        request = ResponsesAgentRequest(
            input=[Message(role="user", content="Hello")],
            context=ChatContext(
                conversation_id="db_conv_123",
                user_id="db_user",
            ),
        )

        context = agent._convert_request_to_context(request)

        assert context.thread_id == "db_conv_123"
        assert context.user_id == "db_user"

    def test_custom_inputs_override_chat_context(self) -> None:
        """Test that custom_inputs.configurable overrides ChatContext."""
        agent = self._create_agent()

        request = ResponsesAgentRequest(
            input=[Message(role="user", content="Hello")],
            context=ChatContext(
                conversation_id="db_conv_123",
                user_id="db_user",
            ),
            custom_inputs={
                "configurable": {
                    "conversation_id": "override_conv",
                    "user_id": "override_user",
                }
            },
        )

        context = agent._convert_request_to_context(request)

        # custom_inputs should take precedence
        assert context.thread_id == "override_conv"
        assert context.user_id == "override_user"

    def test_custom_fields_as_top_level_attributes(self) -> None:
        """Test that custom fields are added as top-level context attributes."""
        agent = self._create_agent()

        request = ResponsesAgentRequest(
            input=[Message(role="user", content="Hello")],
            custom_inputs={
                "configurable": {
                    "conversation_id": "conv_1",
                    "user_id": "test_user",
                    "store_num": "87887",
                    "custom_field": "custom_value",
                }
            },
        )

        context = agent._convert_request_to_context(request)

        assert context.store_num == "87887"
        assert context.custom_field == "custom_value"

    def test_generates_thread_id_if_not_provided(self) -> None:
        """Test that a thread_id is generated if not provided."""
        agent = self._create_agent()

        request = ResponsesAgentRequest(
            input=[Message(role="user", content="Hello")],
            custom_inputs={
                "configurable": {
                    "user_id": "test_user",
                }
            },
        )

        context = agent._convert_request_to_context(request)

        # Should generate a UUID
        assert context.thread_id is not None
        assert len(context.thread_id) == 36  # UUID format


# =============================================================================
# Session Extraction Tests
# =============================================================================


@pytest.mark.unit
class TestSessionExtraction:
    """Tests for LanggraphResponsesAgent._extract_session_from_request."""

    def _create_agent(self) -> LanggraphResponsesAgent:
        """Create a mock LanggraphResponsesAgent."""
        mock_graph = MagicMock()
        return LanggraphResponsesAgent(mock_graph)

    def test_empty_session(self) -> None:
        """Test extraction with no session data."""
        agent = self._create_agent()

        request = ResponsesAgentRequest(
            input=[Message(role="user", content="Hello")],
            custom_inputs={
                "configurable": {"conversation_id": "conv_1"},
            },
        )

        session = agent._extract_session_from_request(request)

        assert session == {}

    def test_new_session_structure(self) -> None:
        """Test extraction from new session.genie.spaces structure."""
        agent = self._create_agent()

        request = ResponsesAgentRequest(
            input=[Message(role="user", content="Hello")],
            custom_inputs={
                "configurable": {"conversation_id": "conv_1"},
                "session": {
                    "genie": {
                        "spaces": {
                            "space_123": {"conversation_id": "genie_conv_456"},
                            "space_789": {"conversation_id": "genie_conv_012"},
                        }
                    }
                },
            },
        )

        session = agent._extract_session_from_request(request)

        assert "genie_conversation_ids" in session
        assert session["genie_conversation_ids"]["space_123"] == "genie_conv_456"
        assert session["genie_conversation_ids"]["space_789"] == "genie_conv_012"

    def test_legacy_genie_conversation_ids(self) -> None:
        """Test extraction from legacy genie_conversation_ids format."""
        agent = self._create_agent()

        request = ResponsesAgentRequest(
            input=[Message(role="user", content="Hello")],
            custom_inputs={
                "configurable": {"conversation_id": "conv_1"},
                "genie_conversation_ids": {
                    "space_a": "conv_a",
                    "space_b": "conv_b",
                },
            },
        )

        session = agent._extract_session_from_request(request)

        assert "genie_conversation_ids" in session
        assert session["genie_conversation_ids"]["space_a"] == "conv_a"
        assert session["genie_conversation_ids"]["space_b"] == "conv_b"


# =============================================================================
# Custom Outputs Tests
# =============================================================================


@pytest.mark.unit
class TestCustomOutputs:
    """Tests for custom_outputs structure."""

    def test_build_custom_outputs_basic(self) -> None:
        """Test building custom_outputs with basic context."""
        import asyncio

        mock_graph = MagicMock()
        mock_graph.aget_state = AsyncMock(return_value=None)
        mock_graph.checkpointer = None  # No checkpointer

        agent = LanggraphResponsesAgent(mock_graph)

        context = Context(
            user_id="test_user",
            thread_id="conv_123",
            store_num="87887",
        )

        # Run async method in sync test using asyncio.run()
        outputs = asyncio.run(
            agent._build_custom_outputs_async(
                context=context,
                thread_id="conv_123",
            )
        )

        # Check configurable section - only thread_id (conversation_id is in session)
        assert "configurable" in outputs
        cfg = outputs["configurable"]
        assert cfg["thread_id"] == "conv_123"
        assert "conversation_id" not in cfg  # conversation_id is now in session
        assert cfg["user_id"] == "test_user"
        assert cfg["store_num"] == "87887"

        # Session contains conversation_id (alias of thread_id) and accumulated state
        assert "session" in outputs
        session = outputs["session"]
        assert session["conversation_id"] == "conv_123"  # conversation_id is in session

    def test_outputs_can_be_used_as_inputs(self) -> None:
        """Test that outputs can be directly used as inputs (copy-paste)."""
        import asyncio

        mock_graph = MagicMock()
        mock_graph.aget_state = AsyncMock(return_value=None)
        mock_graph.checkpointer = None  # No checkpointer

        agent = LanggraphResponsesAgent(mock_graph)

        context = Context(
            user_id="test_user",
            thread_id="conv_123",
            store_num="87887",
        )

        # Generate outputs using asyncio.run()
        outputs = asyncio.run(
            agent._build_custom_outputs_async(
                context=context,
                thread_id="conv_123",
            )
        )

        # Use outputs as inputs to a new request
        new_request = ResponsesAgentRequest(
            input=[Message(role="user", content="Follow up question")],
            custom_inputs=outputs,  # Direct copy-paste
        )

        # Convert to context - should work seamlessly
        new_context = agent._convert_request_to_context(new_request)

        assert new_context.thread_id == "conv_123"
        assert new_context.user_id == "test_user"
        assert new_context.store_num == "87887"
