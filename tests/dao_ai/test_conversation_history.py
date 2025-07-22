from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall

from dao_ai.nodes import _deserialize_messages, _serialize_messages


class TestMessageSerialization:
    """Test suite for message serialization and deserialization functionality."""

    def test_serialize_human_message(self):
        """Test serializing a simple HumanMessage."""
        # Arrange
        messages = [HumanMessage(content="Hello, world!", id="msg_1")]

        # Act
        result = _serialize_messages(messages)

        # Assert
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0]["type"] == "human"
        assert result["messages"][0]["data"]["content"] == "Hello, world!"
        assert result["messages"][0]["data"]["id"] == "msg_1"

    def test_serialize_ai_message(self):
        """Test serializing an AIMessage."""
        # Arrange
        messages = [AIMessage(content="Hello back!", id="msg_2")]

        # Act
        result = _serialize_messages(messages)

        # Assert
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0]["type"] == "ai"
        assert result["messages"][0]["data"]["content"] == "Hello back!"
        assert result["messages"][0]["data"]["id"] == "msg_2"

    def test_serialize_system_message(self):
        """Test serializing a SystemMessage."""
        # Arrange
        messages = [SystemMessage(content="You are a helpful assistant.", id="msg_3")]

        # Act
        result = _serialize_messages(messages)

        # Assert
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0]["type"] == "system"
        assert (
            result["messages"][0]["data"]["content"] == "You are a helpful assistant."
        )
        assert result["messages"][0]["data"]["id"] == "msg_3"

    def test_serialize_tool_message(self):
        """Test serializing a ToolMessage."""
        # Arrange
        messages = [
            ToolMessage(
                content="Tool result: 42", id="msg_4", tool_call_id="tool_call_123"
            )
        ]

        # Act
        result = _serialize_messages(messages)

        # Assert
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0]["type"] == "tool"
        assert result["messages"][0]["data"]["content"] == "Tool result: 42"
        assert result["messages"][0]["data"]["id"] == "msg_4"
        assert result["messages"][0]["data"]["tool_call_id"] == "tool_call_123"

    def test_serialize_ai_message_with_tool_calls(self):
        """Test serializing an AIMessage with tool calls."""
        # Arrange
        tool_call = ToolCall(
            name="get_weather", args={"location": "San Francisco"}, id="call_123"
        )
        messages = [
            AIMessage(
                content="I'll check the weather for you.",
                id="msg_5",
                tool_calls=[tool_call],
            )
        ]

        # Act
        result = _serialize_messages(messages)

        # Assert
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0]["type"] == "ai"
        assert (
            result["messages"][0]["data"]["content"]
            == "I'll check the weather for you."
        )
        assert result["messages"][0]["data"]["id"] == "msg_5"
        assert "tool_calls" in result["messages"][0]["data"]
        assert len(result["messages"][0]["data"]["tool_calls"]) == 1
        assert result["messages"][0]["data"]["tool_calls"][0]["name"] == "get_weather"
        assert result["messages"][0]["data"]["tool_calls"][0]["args"] == {
            "location": "San Francisco"
        }
        assert result["messages"][0]["data"]["tool_calls"][0]["id"] == "call_123"

    def test_serialize_multiple_messages(self):
        """Test serializing multiple messages of different types."""
        # Arrange
        messages = [
            SystemMessage(content="You are helpful.", id="msg_1"),
            HumanMessage(content="What's the weather?", id="msg_2"),
            AIMessage(content="Let me check that.", id="msg_3"),
            ToolMessage(content="Sunny, 75°F", id="msg_4", tool_call_id="call_1"),
            AIMessage(content="It's sunny and 75°F!", id="msg_5"),
        ]

        # Act
        result = _serialize_messages(messages)

        # Assert
        assert "messages" in result
        assert len(result["messages"]) == 5
        assert result["messages"][0]["type"] == "system"
        assert result["messages"][1]["type"] == "human"
        assert result["messages"][2]["type"] == "ai"
        assert result["messages"][3]["type"] == "tool"
        assert result["messages"][4]["type"] == "ai"

    def test_serialize_empty_message_list(self):
        """Test serializing an empty message list."""
        # Arrange
        messages = []

        # Act
        result = _serialize_messages(messages)

        # Assert
        assert "messages" in result
        assert len(result["messages"]) == 0

    def test_deserialize_human_message(self):
        """Test deserializing a HumanMessage."""
        # Arrange
        serialized = {
            "messages": [
                {
                    "type": "human",
                    "data": {
                        "content": "Hello, world!",
                        "id": "msg_1",
                        "additional_kwargs": {},
                        "response_metadata": {},
                        "type": "human",
                        "name": None,
                        "example": False,
                    },
                }
            ]
        }

        # Act
        result = _deserialize_messages(serialized)

        # Assert
        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "Hello, world!"
        assert result[0].id == "msg_1"

    def test_deserialize_ai_message(self):
        """Test deserializing an AIMessage."""
        # Arrange
        serialized = {
            "messages": [
                {
                    "type": "ai",
                    "data": {
                        "content": "Hello back!",
                        "id": "msg_2",
                        "additional_kwargs": {},
                        "response_metadata": {},
                        "type": "ai",
                        "name": None,
                        "example": False,
                        "tool_calls": [],
                        "invalid_tool_calls": [],
                    },
                }
            ]
        }

        # Act
        result = _deserialize_messages(serialized)

        # Assert
        assert len(result) == 1
        assert isinstance(result[0], AIMessage)
        assert result[0].content == "Hello back!"
        assert result[0].id == "msg_2"

    def test_deserialize_system_message(self):
        """Test deserializing a SystemMessage."""
        # Arrange
        serialized = {
            "messages": [
                {
                    "type": "system",
                    "data": {
                        "content": "You are a helpful assistant.",
                        "id": "msg_3",
                        "additional_kwargs": {},
                        "response_metadata": {},
                        "type": "system",
                        "name": None,
                        "example": False,
                    },
                }
            ]
        }

        # Act
        result = _deserialize_messages(serialized)

        # Assert
        assert len(result) == 1
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "You are a helpful assistant."
        assert result[0].id == "msg_3"

    def test_deserialize_tool_message(self):
        """Test deserializing a ToolMessage."""
        # Arrange
        serialized = {
            "messages": [
                {
                    "type": "tool",
                    "data": {
                        "content": "Tool result: 42",
                        "id": "msg_4",
                        "tool_call_id": "tool_call_123",
                        "additional_kwargs": {},
                        "response_metadata": {},
                        "type": "tool",
                        "name": None,
                        "example": False,
                    },
                }
            ]
        }

        # Act
        result = _deserialize_messages(serialized)

        # Assert
        assert len(result) == 1
        assert isinstance(result[0], ToolMessage)
        assert result[0].content == "Tool result: 42"
        assert result[0].id == "msg_4"
        assert result[0].tool_call_id == "tool_call_123"

    def test_deserialize_ai_message_with_tool_calls(self):
        """Test deserializing an AIMessage with tool calls."""
        # Arrange
        serialized = {
            "messages": [
                {
                    "type": "ai",
                    "data": {
                        "content": "I'll check the weather for you.",
                        "id": "msg_5",
                        "additional_kwargs": {},
                        "response_metadata": {},
                        "type": "ai",
                        "name": None,
                        "example": False,
                        "tool_calls": [
                            {
                                "name": "get_weather",
                                "args": {"location": "San Francisco"},
                                "id": "call_123",
                            }
                        ],
                        "invalid_tool_calls": [],
                    },
                }
            ]
        }

        # Act
        result = _deserialize_messages(serialized)

        # Assert
        assert len(result) == 1
        assert isinstance(result[0], AIMessage)
        assert result[0].content == "I'll check the weather for you."
        assert result[0].id == "msg_5"
        assert len(result[0].tool_calls) == 1
        assert result[0].tool_calls[0]["name"] == "get_weather"
        assert result[0].tool_calls[0]["args"] == {"location": "San Francisco"}
        assert result[0].tool_calls[0]["id"] == "call_123"

    def test_deserialize_multiple_messages(self):
        """Test deserializing multiple messages of different types."""
        # Arrange
        serialized = {
            "messages": [
                {
                    "type": "system",
                    "data": {
                        "content": "You are helpful.",
                        "id": "msg_1",
                        "additional_kwargs": {},
                        "response_metadata": {},
                        "type": "system",
                        "name": None,
                        "example": False,
                    },
                },
                {
                    "type": "human",
                    "data": {
                        "content": "What's the weather?",
                        "id": "msg_2",
                        "additional_kwargs": {},
                        "response_metadata": {},
                        "type": "human",
                        "name": None,
                        "example": False,
                    },
                },
                {
                    "type": "ai",
                    "data": {
                        "content": "Let me check that.",
                        "id": "msg_3",
                        "additional_kwargs": {},
                        "response_metadata": {},
                        "type": "ai",
                        "name": None,
                        "example": False,
                        "tool_calls": [],
                        "invalid_tool_calls": [],
                    },
                },
                {
                    "type": "tool",
                    "data": {
                        "content": "Sunny, 75°F",
                        "id": "msg_4",
                        "tool_call_id": "call_1",
                        "additional_kwargs": {},
                        "response_metadata": {},
                        "type": "tool",
                        "name": None,
                        "example": False,
                    },
                },
                {
                    "type": "ai",
                    "data": {
                        "content": "It's sunny and 75°F!",
                        "id": "msg_5",
                        "additional_kwargs": {},
                        "response_metadata": {},
                        "type": "ai",
                        "name": None,
                        "example": False,
                        "tool_calls": [],
                        "invalid_tool_calls": [],
                    },
                },
            ]
        }

        # Act
        result = _deserialize_messages(serialized)

        # Assert
        assert len(result) == 5
        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[1], HumanMessage)
        assert isinstance(result[2], AIMessage)
        assert isinstance(result[3], ToolMessage)
        assert isinstance(result[4], AIMessage)

        assert result[0].content == "You are helpful."
        assert result[1].content == "What's the weather?"
        assert result[2].content == "Let me check that."
        assert result[3].content == "Sunny, 75°F"
        assert result[4].content == "It's sunny and 75°F!"

    def test_deserialize_empty_message_list(self):
        """Test deserializing an empty message list."""
        # Arrange
        serialized = {"messages": []}

        # Act
        result = _deserialize_messages(serialized)

        # Assert
        assert len(result) == 0

    def test_deserialize_missing_messages_key(self):
        """Test deserializing when 'messages' key is missing."""
        # Arrange
        serialized = {}

        # Act
        result = _deserialize_messages(serialized)

        # Assert
        assert len(result) == 0

    def test_round_trip_serialization(self):
        """Test that messages can be serialized and then deserialized correctly."""
        # Arrange
        original_messages = [
            SystemMessage(content="System prompt", id="sys_1"),
            HumanMessage(content="User question", id="human_1"),
            AIMessage(content="AI response", id="ai_1"),
            ToolMessage(content="Tool output", id="tool_1", tool_call_id="call_1"),
        ]

        # Act
        serialized = _serialize_messages(original_messages)
        deserialized = _deserialize_messages(serialized)

        # Assert
        assert len(deserialized) == len(original_messages)
        for original, restored in zip(original_messages, deserialized):
            assert isinstance(restored, type(original))
            assert original.content == restored.content
            assert original.id == restored.id
            if hasattr(original, "tool_call_id"):
                assert original.tool_call_id == restored.tool_call_id

    def test_round_trip_with_tool_calls(self):
        """Test round-trip serialization with tool calls."""
        # Arrange
        tool_call = ToolCall(name="search", args={"query": "python"}, id="search_123")
        original_messages = [
            AIMessage(
                content="I'll search for that.",
                id="ai_with_tools",
                tool_calls=[tool_call],
            )
        ]

        # Act
        serialized = _serialize_messages(original_messages)
        deserialized = _deserialize_messages(serialized)

        # Assert
        assert len(deserialized) == 1
        assert isinstance(deserialized[0], AIMessage)
        assert deserialized[0].content == "I'll search for that."
        assert len(deserialized[0].tool_calls) == 1
        assert deserialized[0].tool_calls[0]["name"] == "search"
        assert deserialized[0].tool_calls[0]["args"] == {"query": "python"}
        assert deserialized[0].tool_calls[0]["id"] == "search_123"

    @patch("dao_ai.nodes.logger")
    def test_serialization_logging(self, mock_logger):
        """Test that serialization includes proper logging."""
        # Arrange
        messages = [HumanMessage(content="Test", id="test_1")]

        # Act
        _serialize_messages(messages)

        # Assert
        mock_logger.debug.assert_called()
        mock_logger.trace.assert_called()

    @patch("dao_ai.nodes.logger")
    def test_deserialization_logging(self, mock_logger):
        """Test that deserialization includes proper logging."""
        # Arrange
        serialized = {
            "messages": [
                {
                    "type": "human",
                    "data": {
                        "content": "Test",
                        "id": "test_1",
                        "additional_kwargs": {},
                        "response_metadata": {},
                        "type": "human",
                        "name": None,
                        "example": False,
                    },
                }
            ]
        }

        # Act
        _deserialize_messages(serialized)

        # Assert
        mock_logger.debug.assert_called()
        mock_logger.trace.assert_called()

    def test_message_with_additional_metadata(self):
        """Test serialization/deserialization with additional message metadata."""
        # Arrange
        messages = [
            HumanMessage(
                content="Test message",
                id="test_msg",
                additional_kwargs={"timestamp": "2023-01-01T00:00:00Z"},
            )
        ]

        # Act
        serialized = _serialize_messages(messages)
        deserialized = _deserialize_messages(serialized)

        # Assert
        assert len(deserialized) == 1
        assert isinstance(deserialized[0], HumanMessage)
        assert deserialized[0].content == "Test message"
        assert deserialized[0].id == "test_msg"
        # Note: additional_kwargs might not be preserved depending on LangChain implementation
