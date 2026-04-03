"""Tests for agent endpoint tool OBO support."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from dao_ai.config import LLMModel
from dao_ai.tools.agent import create_agent_endpoint_tool


@pytest.fixture
def mock_llm_no_obo() -> LLMModel:
    """Create an LLMModel with OBO disabled."""
    return LLMModel(name="test-endpoint", on_behalf_of_user=False)


@pytest.fixture
def mock_llm_obo() -> LLMModel:
    """Create an LLMModel with OBO enabled."""
    return LLMModel(name="test-endpoint", on_behalf_of_user=True)


@pytest.mark.unit
def test_create_agent_endpoint_tool_basic(mock_llm_no_obo: LLMModel) -> None:
    """Test that the factory function creates a tool with correct attributes."""
    tool = create_agent_endpoint_tool(mock_llm_no_obo)

    assert tool is not None
    assert tool.name == "agent_endpoint"
    assert "language model endpoint" in tool.description


@pytest.mark.unit
def test_create_agent_endpoint_tool_custom_name(mock_llm_no_obo: LLMModel) -> None:
    """Test creating a tool with custom name and description."""
    tool = create_agent_endpoint_tool(
        mock_llm_no_obo, name="my_agent", description="Custom agent tool"
    )

    assert tool.name == "my_agent"
    assert "Custom agent tool" in tool.description


@pytest.mark.unit
def test_create_agent_endpoint_tool_from_dict() -> None:
    """Test creating a tool from a dict instead of LLMModel."""
    tool = create_agent_endpoint_tool({"name": "test-endpoint"})

    assert tool is not None
    assert tool.name == "agent_endpoint"


@pytest.mark.unit
def test_agent_endpoint_tool_no_obo_uses_as_chat_model(
    mock_llm_no_obo: LLMModel,
) -> None:
    """When OBO is disabled, the tool should use llm.as_chat_model()."""
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = AIMessage(content="test response")

    with patch.object(LLMModel, "as_chat_model", return_value=mock_model):
        tool = create_agent_endpoint_tool(mock_llm_no_obo)
        result = asyncio.run(tool.ainvoke({"prompt": "hello"}))

    assert isinstance(result, AIMessage)
    assert result.content == "test response"


@pytest.mark.unit
def test_agent_endpoint_tool_obo_uses_workspace_client_from(
    mock_llm_obo: LLMModel,
) -> None:
    """When OBO is enabled, the tool should use workspace_client_from(context)."""
    mock_ws = MagicMock()
    mock_ws.config.auth_type = "pat"

    mock_chat_model = AsyncMock()
    mock_chat_model.ainvoke.return_value = AIMessage(content="obo response")

    with (
        patch.object(LLMModel, "workspace_client_from", return_value=mock_ws),
        patch(
            "dao_ai.tools.agent.ChatDatabricks", return_value=mock_chat_model
        ) as mock_chat_cls,
    ):
        tool = create_agent_endpoint_tool(mock_llm_obo)
        result = asyncio.run(tool.ainvoke({"prompt": "hello"}))

    mock_chat_cls.assert_called_once_with(
        model="test-endpoint",
        temperature=mock_llm_obo.temperature,
        max_tokens=mock_llm_obo.max_tokens,
        use_responses_api=mock_llm_obo.use_responses_api,
        workspace_client=mock_ws,
    )
    assert isinstance(result, AIMessage)
    assert result.content == "obo response"
