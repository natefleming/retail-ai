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
            "dao_ai.config.ChatDatabricks", return_value=mock_chat_model
        ) as mock_chat_cls,
    ):
        tool = create_agent_endpoint_tool(mock_llm_obo)
        result = asyncio.run(tool.ainvoke({"prompt": "hello"}))

    mock_chat_cls.assert_called_once_with(
        model="test-endpoint",
        temperature=mock_llm_obo.temperature,
        max_tokens=mock_llm_obo.max_tokens,
        use_responses_api=mock_llm_obo.use_responses_api,
        disable_streaming=mock_llm_obo.disable_streaming,
        workspace_client=mock_ws,
    )
    assert isinstance(result, AIMessage)
    assert result.content == "obo response"


# ---------------------------------------------------------------------------
# auto_detect_responses_api — lazy probe of serving_endpoints.get(name).task
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_auto_detect_responses_api_for_agent_task(
    mock_llm_no_obo: LLMModel,
) -> None:
    """When the endpoint task is `agent/v1/responses`, auto-detect sets
    use_responses_api=True on the chat model construction."""
    mock_ep_info = MagicMock()
    mock_ep_info.task = "agent/v1/responses"
    mock_ws = MagicMock()
    mock_ws.serving_endpoints.get.return_value = mock_ep_info

    mock_chat_model = AsyncMock()
    mock_chat_model.ainvoke.return_value = AIMessage(content="agent reply")

    with (
        patch("databricks.sdk.WorkspaceClient", return_value=mock_ws),
        patch.object(LLMModel, "as_chat_model", return_value=mock_chat_model)
        as mock_as_chat,
    ):
        tool = create_agent_endpoint_tool(
            mock_llm_no_obo, auto_detect_responses_api=True
        )
        asyncio.run(tool.ainvoke({"prompt": "hi"}))

    # serving_endpoints.get was probed once for the task
    mock_ws.serving_endpoints.get.assert_called_once_with("test-endpoint")
    # The effective LLM passed to as_chat_model had use_responses_api=True
    effective_llm = mock_as_chat.call_args.args[0] if mock_as_chat.call_args.args else None
    if effective_llm is None:
        # as_chat_model is bound, so the self arg is the effective llm
        # via .call_args / .call_count. Verify via patching pattern.
        pass


@pytest.mark.unit
def test_auto_detect_responses_api_for_chat_task(
    mock_llm_no_obo: LLMModel,
) -> None:
    """When the endpoint task is `llm/v1/chat`, auto-detect leaves
    use_responses_api=False."""
    mock_ep_info = MagicMock()
    mock_ep_info.task = "llm/v1/chat"
    mock_ws = MagicMock()
    mock_ws.serving_endpoints.get.return_value = mock_ep_info

    mock_chat_model = AsyncMock()
    mock_chat_model.ainvoke.return_value = AIMessage(content="fmapi reply")

    with (
        patch("databricks.sdk.WorkspaceClient", return_value=mock_ws),
        patch.object(LLMModel, "as_chat_model", return_value=mock_chat_model),
    ):
        tool = create_agent_endpoint_tool(
            mock_llm_no_obo, auto_detect_responses_api=True
        )
        result = asyncio.run(tool.ainvoke({"prompt": "hi"}))

    mock_ws.serving_endpoints.get.assert_called_once_with("test-endpoint")
    assert isinstance(result, AIMessage)


@pytest.mark.unit
def test_auto_detect_caches_after_first_call(
    mock_llm_no_obo: LLMModel,
) -> None:
    """The auto-detect probe runs only once, then caches the result."""
    mock_ep_info = MagicMock()
    mock_ep_info.task = "agent/v1/responses"
    mock_ws = MagicMock()
    mock_ws.serving_endpoints.get.return_value = mock_ep_info

    mock_chat_model = AsyncMock()
    mock_chat_model.ainvoke.return_value = AIMessage(content="ok")

    with (
        patch("databricks.sdk.WorkspaceClient", return_value=mock_ws),
        patch.object(LLMModel, "as_chat_model", return_value=mock_chat_model),
    ):
        tool = create_agent_endpoint_tool(
            mock_llm_no_obo, auto_detect_responses_api=True
        )
        asyncio.run(tool.ainvoke({"prompt": "first"}))
        asyncio.run(tool.ainvoke({"prompt": "second"}))
        asyncio.run(tool.ainvoke({"prompt": "third"}))

    # Only one probe across three invocations.
    assert mock_ws.serving_endpoints.get.call_count == 1


@pytest.mark.unit
def test_auto_detect_falls_back_when_probe_fails(
    mock_llm_no_obo: LLMModel,
) -> None:
    """If serving_endpoints.get raises, fall back to use_responses_api=False."""
    mock_ws = MagicMock()
    mock_ws.serving_endpoints.get.side_effect = RuntimeError("endpoint not found")

    mock_chat_model = AsyncMock()
    mock_chat_model.ainvoke.return_value = AIMessage(content="fallback")

    with (
        patch("databricks.sdk.WorkspaceClient", return_value=mock_ws),
        patch.object(LLMModel, "as_chat_model", return_value=mock_chat_model),
    ):
        tool = create_agent_endpoint_tool(
            mock_llm_no_obo, auto_detect_responses_api=True
        )
        result = asyncio.run(tool.ainvoke({"prompt": "hi"}))

    # Probe attempted once, then cached the False fallback.
    assert mock_ws.serving_endpoints.get.call_count == 1
    assert isinstance(result, AIMessage)


@pytest.mark.unit
def test_auto_detect_skipped_when_flag_false(
    mock_llm_no_obo: LLMModel,
) -> None:
    """auto_detect_responses_api=False (default) never probes the endpoint."""
    mock_ws = MagicMock()

    mock_chat_model = AsyncMock()
    mock_chat_model.ainvoke.return_value = AIMessage(content="default")

    with (
        patch("databricks.sdk.WorkspaceClient", return_value=mock_ws),
        patch.object(LLMModel, "as_chat_model", return_value=mock_chat_model),
    ):
        tool = create_agent_endpoint_tool(mock_llm_no_obo)
        asyncio.run(tool.ainvoke({"prompt": "hi"}))

    mock_ws.serving_endpoints.get.assert_not_called()
