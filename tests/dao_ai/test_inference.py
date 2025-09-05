from typing import Any, Sequence
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from conftest import has_databricks_env
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from mlflow.pyfunc import ChatModel
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from mlflow.types.responses_helpers import Message

from dao_ai.models import (
    LanggraphResponsesAgent,
    _process_config_messages,
    _process_config_messages_stream,
    _process_langchain_messages,
    _process_mlflow_response_messages,
    _process_mlflow_response_messages_stream,
    process_messages,
    process_messages_stream,
)


@pytest.mark.system
@pytest.mark.slow
@pytest.mark.skipif(
    not has_databricks_env(), reason="Missing Databricks environment variables"
)
def test_inference(chat_model: ChatModel) -> None:
    messages: Sequence[BaseMessage] = [
        HumanMessage(content="What is the weather like today?"),
    ]
    custom_inputs: dict[str, Any] = {
        "configurable": {
            "user_id": "user123",
            "thread_id": "1",
        }
    }
    response: dict[str, Any] | Any = process_messages(
        chat_model, messages, custom_inputs
    )
    print(response)
    assert response is not None


@pytest.mark.system
@pytest.mark.slow
@pytest.mark.skipif(
    not has_databricks_env(), reason="Missing Databricks environment variables"
)
def test_inference_missing_user_id(chat_model: ChatModel) -> None:
    messages: Sequence[BaseMessage] = [
        HumanMessage(content="What is the weather like today?"),
    ]
    custom_inputs: dict[str, Any] = {
        "configurable": {
            "thread_id": "1",
        }
    }
    response: dict[str, Any] | Any = process_messages(
        chat_model, messages, custom_inputs
    )
    print(response)
    assert response is not None


# Unit tests for process_messages and process_messages_stream variations


def test_process_langchain_messages():
    """Test _process_langchain_messages with LangChain BaseMessage objects."""
    # Create mock app with AsyncMock for ainvoke
    mock_app = MagicMock()
    mock_response = {"messages": [AIMessage(content="Test response")]}
    mock_app.ainvoke = AsyncMock(return_value=mock_response)

    # Test data
    messages = [HumanMessage(content="Hello")]
    custom_inputs = {"configurable": {"user_id": "test_user"}}

    # Test the function
    result = _process_langchain_messages(mock_app, messages, custom_inputs)

    assert result == mock_response
    mock_app.ainvoke.assert_called_once_with(
        {"messages": messages}, config=custom_inputs
    )


def test_process_mlflow_response_messages():
    """Test _process_mlflow_response_messages with ResponsesAgent objects."""
    # Create mock ResponsesAgent
    mock_graph = MagicMock()
    mock_app = LanggraphResponsesAgent(mock_graph)
    mock_response = ResponsesAgentResponse(output=[])
    mock_app.predict = Mock(return_value=mock_response)

    # Test data
    request = ResponsesAgentRequest(
        input=[Message(role="user", content="Hello", type="message")]
    )

    # Test the function
    result = _process_mlflow_response_messages(mock_app, request)

    assert result == mock_response
    mock_app.predict.assert_called_once_with(request)


def test_process_mlflow_response_messages_stream():
    """Test _process_mlflow_response_messages_stream with ResponsesAgent objects."""
    # Create mock ResponsesAgent
    mock_graph = MagicMock()
    mock_app = LanggraphResponsesAgent(mock_graph)
    mock_events = [
        ResponsesAgentStreamEvent(type="response.text.delta", delta="Hello"),
        ResponsesAgentStreamEvent(type="response.text.delta", delta=" world"),
        ResponsesAgentStreamEvent(type="response.done"),
    ]
    mock_app.predict_stream = Mock(return_value=iter(mock_events))

    # Test data
    request = ResponsesAgentRequest(
        input=[Message(role="user", content="Hello", type="message")]
    )

    # Test the function
    result = list(_process_mlflow_response_messages_stream(mock_app, request))

    assert len(result) == 3
    assert all(isinstance(event, ResponsesAgentStreamEvent) for event in result)
    mock_app.predict_stream.assert_called_once_with(request)


def test_process_messages_routing_responses_agent():
    """Test process_messages correctly routes ResponsesAgent requests."""
    # Create mock ResponsesAgent
    mock_graph = MagicMock()
    mock_app = LanggraphResponsesAgent(mock_graph)
    mock_response = ResponsesAgentResponse(output=[])
    mock_app.predict = Mock(return_value=mock_response)

    # Test data - ResponsesAgentRequest should route to ResponsesAgent path
    request = ResponsesAgentRequest(
        input=[Message(role="user", content="Hello", type="message")]
    )

    # Test the function
    result = process_messages(mock_app, request)

    assert result == mock_response
    mock_app.predict.assert_called_once_with(request)


def test_process_messages_stream_routing_responses_agent():
    """Test process_messages_stream correctly routes ResponsesAgent requests."""
    # Create mock ResponsesAgent
    mock_graph = MagicMock()
    mock_app = LanggraphResponsesAgent(mock_graph)
    mock_events = [
        ResponsesAgentStreamEvent(type="response.text.delta", delta="Hello"),
        ResponsesAgentStreamEvent(type="response.done"),
    ]
    mock_app.predict_stream = Mock(return_value=iter(mock_events))

    # Test data - ResponsesAgentRequest should route to ResponsesAgent stream path
    request = ResponsesAgentRequest(
        input=[Message(role="user", content="Hello", type="message")]
    )

    # Test the function
    result = list(process_messages_stream(mock_app, request))

    assert len(result) == 2
    assert all(isinstance(event, ResponsesAgentStreamEvent) for event in result)
    mock_app.predict_stream.assert_called_once_with(request)


def test_process_config_messages_responses_agent():
    """Test _process_config_messages correctly handles LanggraphResponsesAgent with dict input."""

    # Create mock ResponsesAgent
    mock_graph = MagicMock()
    mock_app = LanggraphResponsesAgent(mock_graph)
    mock_response = ResponsesAgentResponse(output=[])
    mock_app.predict = Mock(return_value=mock_response)

    # Test data - dict format should be converted to ResponsesAgentRequest
    messages_dict = [
        {"role": "user", "content": "Hello from config", "type": "message"}
    ]
    custom_inputs = {"user_id": "test_user", "session_id": "test_session"}

    # Test the function
    result = _process_config_messages(mock_app, messages_dict, custom_inputs)

    assert result == mock_response
    mock_app.predict.assert_called_once()

    # Verify the call was made with proper ResponsesAgentRequest
    call_args = mock_app.predict.call_args[0][0]
    assert isinstance(call_args, ResponsesAgentRequest)
    assert len(call_args.input) == 1
    assert call_args.input[0].role == "user"
    assert call_args.input[0].content == "Hello from config"
    assert call_args.custom_inputs == custom_inputs


def test_process_config_messages_stream_responses_agent():
    """Test _process_config_messages_stream correctly handles LanggraphResponsesAgent with dict input."""

    # Create mock ResponsesAgent
    mock_graph = MagicMock()
    mock_app = LanggraphResponsesAgent(mock_graph)
    mock_events = [
        ResponsesAgentStreamEvent(type="response.text.delta", delta="Config"),
        ResponsesAgentStreamEvent(type="response.text.delta", delta=" stream"),
        ResponsesAgentStreamEvent(type="response.done"),
    ]
    mock_app.predict_stream = Mock(return_value=iter(mock_events))

    # Test data - dict format should be converted to ResponsesAgentRequest
    messages_dict = [
        {"role": "user", "content": "Hello from config stream", "type": "message"}
    ]
    custom_inputs = {"user_id": "test_user", "conversation_id": "test_conv"}

    # Test the function
    result = list(
        _process_config_messages_stream(mock_app, messages_dict, custom_inputs)
    )

    assert len(result) == 3
    assert all(isinstance(event, ResponsesAgentStreamEvent) for event in result)
    mock_app.predict_stream.assert_called_once()

    # Verify the call was made with proper ResponsesAgentRequest
    call_args = mock_app.predict_stream.call_args[0][0]
    assert isinstance(call_args, ResponsesAgentRequest)
    assert len(call_args.input) == 1
    assert call_args.input[0].role == "user"
    assert call_args.input[0].content == "Hello from config stream"
    assert call_args.custom_inputs == custom_inputs


def test_process_messages_with_dict_input_responses_agent():
    """Test process_messages correctly routes dict input to LanggraphResponsesAgent."""
    # Create mock ResponsesAgent
    mock_graph = MagicMock()
    mock_app = LanggraphResponsesAgent(mock_graph)
    mock_response = ResponsesAgentResponse(output=[])
    mock_app.predict = Mock(return_value=mock_response)

    # Test data - dict format should route through _process_config_messages
    messages_dict = [
        {"role": "user", "content": "Hello from dict routing", "type": "message"}
    ]
    custom_inputs = {"user_id": "test_user", "thread_id": "test_thread"}

    # Test the function
    result = process_messages(mock_app, messages_dict, custom_inputs)

    assert result == mock_response
    mock_app.predict.assert_called_once()

    # Verify end-to-end conversion: dict -> ResponsesAgentRequest
    call_args = mock_app.predict.call_args[0][0]
    assert isinstance(call_args, ResponsesAgentRequest)
    assert len(call_args.input) == 1
    assert call_args.input[0].role == "user"
    assert call_args.input[0].content == "Hello from dict routing"
    assert call_args.custom_inputs == custom_inputs


def test_process_messages_stream_with_dict_input_responses_agent():
    """Test process_messages_stream correctly routes dict input to LanggraphResponsesAgent."""
    # Create mock ResponsesAgent
    mock_graph = MagicMock()
    mock_app = LanggraphResponsesAgent(mock_graph)
    mock_events = [
        ResponsesAgentStreamEvent(type="response.text.delta", delta="Dict"),
        ResponsesAgentStreamEvent(type="response.text.delta", delta=" routing"),
        ResponsesAgentStreamEvent(type="response.done"),
    ]
    mock_app.predict_stream = Mock(return_value=iter(mock_events))

    # Test data - dict format should route through _process_config_messages_stream
    messages_dict = [
        {"role": "user", "content": "Hello from dict stream routing", "type": "message"}
    ]
    custom_inputs = {"user_id": "test_user", "session_id": "test_session"}

    # Test the function
    result = list(process_messages_stream(mock_app, messages_dict, custom_inputs))

    assert len(result) == 3
    assert all(isinstance(event, ResponsesAgentStreamEvent) for event in result)
    mock_app.predict_stream.assert_called_once()

    # Verify end-to-end conversion: dict -> ResponsesAgentRequest
    call_args = mock_app.predict_stream.call_args[0][0]
    assert isinstance(call_args, ResponsesAgentRequest)
    assert len(call_args.input) == 1
    assert call_args.input[0].role == "user"
    assert call_args.input[0].content == "Hello from dict stream routing"
    assert call_args.custom_inputs == custom_inputs
