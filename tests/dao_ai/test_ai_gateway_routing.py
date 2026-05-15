"""Unit tests for AI Gateway routing on InferenceEndpointModel.

Covers the `ai_gateway: bool` flag introduced for the revamped Databricks
AI Gateway (`/ai-gateway/mlflow/v1/chat/completions`). Verifies:

- Pydantic validator rejects `ai_gateway` + `use_responses_api`
- Default flag value leaves the legacy ChatDatabricks path intact
- `ai_gateway=True` routes both `as_chat_model()` and `as_open_ai_client()`
  through ChatOpenAI with the AI Gateway base_url
- `_resolve_ai_gateway_credentials()` extracts bearer tokens uniformly
  across PAT, service principal, and OBO modes via the SDK's
  `Config.authenticate()`
- Heterogeneous fallback lists (AI Gateway primary + legacy fallback)
  compose without raising
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from dao_ai.config import InferenceEndpointModel

# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def test_default_ai_gateway_is_false() -> None:
    model = InferenceEndpointModel(name="databricks-meta-llama-3-3-70b-instruct")
    assert model.ai_gateway is False


def test_ai_gateway_true_with_responses_api_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        InferenceEndpointModel(
            name="databricks-claude-opus-4-6",
            ai_gateway=True,
            use_responses_api=True,
        )
    assert "use_responses_api" in str(exc_info.value)


def test_ai_gateway_true_with_obo_currently_allowed() -> None:
    """OBO + ai_gateway is permitted in v1 pending live verification."""
    model = InferenceEndpointModel(
        name="databricks-claude-opus-4-6",
        ai_gateway=True,
        on_behalf_of_user=True,
    )
    assert model.ai_gateway is True
    assert model.on_behalf_of_user is True


# ---------------------------------------------------------------------------
# Credential resolver
# ---------------------------------------------------------------------------


def _stub_workspace_client(host: str, token: str) -> MagicMock:
    wc = MagicMock()
    wc.config.host = host
    wc.config.authenticate.return_value = {"Authorization": f"Bearer {token}"}
    return wc


def test_resolve_credentials_returns_host_and_provider() -> None:
    model = InferenceEndpointModel(name="x", ai_gateway=True)
    wc = _stub_workspace_client(
        host="https://adb-984752964297111.11.azuredatabricks.net/",
        token="dapi-test-token",
    )
    with patch.object(
        type(model), "workspace_client", new_callable=lambda: property(lambda self: wc)
    ):
        host, provider = model._resolve_ai_gateway_credentials()
    assert host == "https://adb-984752964297111.11.azuredatabricks.net"
    assert callable(provider)
    assert provider() == "dapi-test-token"


def test_token_provider_refreshes_per_call() -> None:
    """Token provider must call authenticate() each time so rotated/refreshed
    OBO and OAuth-M2M tokens are picked up without re-instantiating the client."""
    model = InferenceEndpointModel(name="x", ai_gateway=True)
    wc = MagicMock()
    wc.config.host = "https://example.databricks.com"
    wc.config.authenticate.side_effect = [
        {"Authorization": "Bearer token-v1"},
        {"Authorization": "Bearer token-v2"},
    ]
    with patch.object(
        type(model), "workspace_client", new_callable=lambda: property(lambda self: wc)
    ):
        provider = model._ai_gateway_token_provider()
        assert provider() == "token-v1"
        assert provider() == "token-v2"


def test_resolve_credentials_raises_when_no_bearer() -> None:
    model = InferenceEndpointModel(name="x", ai_gateway=True)
    wc = MagicMock()
    wc.config.host = "https://example.databricks.com"
    wc.config.authenticate.return_value = {"X-Custom": "nope"}
    with patch.object(
        type(model), "workspace_client", new_callable=lambda: property(lambda self: wc)
    ):
        _, provider = model._resolve_ai_gateway_credentials()
        with pytest.raises(RuntimeError, match="bearer token"):
            provider()


# ---------------------------------------------------------------------------
# as_chat_model routing
# ---------------------------------------------------------------------------


def test_as_chat_model_default_uses_chat_databricks() -> None:
    model = InferenceEndpointModel(
        name="databricks-meta-llama-3-3-70b-instruct",
        temperature=0.2,
        max_tokens=512,
    )
    with (
        patch("dao_ai.config.ChatDatabricks") as mock_chat_databricks,
        patch("dao_ai.config.AIGatewayChatOpenAI") as mock_chat_openai,
    ):
        mock_chat_databricks.return_value = MagicMock(name="chat_databricks_instance")
        result = model.as_chat_model()
    mock_chat_databricks.assert_called_once()
    mock_chat_openai.assert_not_called()
    kwargs = mock_chat_databricks.call_args.kwargs
    assert kwargs["model"] == "databricks-meta-llama-3-3-70b-instruct"
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 512
    assert result is mock_chat_databricks.return_value


def test_as_chat_model_ai_gateway_uses_chat_openai() -> None:
    model = InferenceEndpointModel(
        name="databricks-claude-opus-4-6",
        ai_gateway=True,
        temperature=0.1,
        max_tokens=1024,
    )
    fake_provider = lambda: "dapi-test-token"  # noqa: E731
    with (
        patch.object(
            InferenceEndpointModel,
            "_resolve_ai_gateway_credentials",
            return_value=(
                "https://adb-984752964297111.11.azuredatabricks.net",
                fake_provider,
            ),
        ),
        patch("dao_ai.config.AIGatewayChatOpenAI") as mock_chat_openai,
        patch("dao_ai.config.ChatDatabricks") as mock_chat_databricks,
    ):
        mock_chat_openai.return_value = MagicMock(name="chat_openai_instance")
        result = model.as_chat_model()
    mock_chat_databricks.assert_not_called()
    mock_chat_openai.assert_called_once()
    kwargs = mock_chat_openai.call_args.kwargs
    assert kwargs["model"] == "databricks-claude-opus-4-6"
    assert (
        kwargs["base_url"]
        == "https://adb-984752964297111.11.azuredatabricks.net/ai-gateway/mlflow/v1"
    )
    # api_key must be the callable provider — captured strings can't refresh
    # OBO / OAuth-M2M tokens. The openai SDK invokes the callable per request.
    assert kwargs["api_key"] is fake_provider
    assert kwargs["api_key"]() == "dapi-test-token"
    assert kwargs["temperature"] == 0.1
    assert kwargs["max_tokens"] == 1024
    assert kwargs["streaming"] is True  # disable_streaming defaults False
    assert result is mock_chat_openai.return_value


def test_as_chat_model_ai_gateway_disables_streaming_when_requested() -> None:
    model = InferenceEndpointModel(
        name="databricks-claude-opus-4-6",
        ai_gateway=True,
        disable_streaming=True,
    )
    with (
        patch.object(
            InferenceEndpointModel,
            "_resolve_ai_gateway_credentials",
            return_value=("https://x", lambda: "tok"),
        ),
        patch("dao_ai.config.AIGatewayChatOpenAI") as mock_chat_openai,
    ):
        model.as_chat_model()
    assert mock_chat_openai.call_args.kwargs["streaming"] is False


# ---------------------------------------------------------------------------
# as_open_ai_client routing
# ---------------------------------------------------------------------------


def test_as_open_ai_client_ai_gateway_uses_direct_chat_openai() -> None:
    model = InferenceEndpointModel(
        name="databricks-claude-opus-4-6",
        ai_gateway=True,
        temperature=0.1,
        max_tokens=2048,
    )
    fake_provider = lambda: "tok"  # noqa: E731
    with (
        patch.object(
            InferenceEndpointModel,
            "_resolve_ai_gateway_credentials",
            return_value=("https://host", fake_provider),
        ),
        patch("dao_ai.config.AIGatewayChatOpenAI") as mock_chat_openai,
    ):
        mock_chat_openai.return_value = MagicMock()
        model.as_open_ai_client()
    mock_chat_openai.assert_called_once_with(
        model="databricks-claude-opus-4-6",
        base_url="https://host/ai-gateway/mlflow/v1",
        api_key=fake_provider,
        temperature=0.1,
        max_tokens=2048,
    )


# ---------------------------------------------------------------------------
# Heterogeneous fallbacks compose
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# OBO simulation
# ---------------------------------------------------------------------------


def test_chat_model_for_workspace_client_uses_ai_gateway_when_flag_set() -> None:
    """OBO-style construction (per-request workspace_client) must route through
    AI Gateway when ai_gateway=True on the config — not silently fall back to
    ChatDatabricks. This is the bug that motivated the shared factory."""
    model = InferenceEndpointModel(
        name="databricks-claude-opus-4-6",
        ai_gateway=True,
        on_behalf_of_user=True,
        temperature=0.0,
        max_tokens=128,
    )
    obo_wc = MagicMock()
    obo_wc.config.host = "https://obo-host.databricks.com"
    obo_wc.config.authenticate.return_value = {"Authorization": "Bearer obo-user-token"}

    with (
        patch("dao_ai.config.AIGatewayChatOpenAI") as mock_chat_openai,
        patch("dao_ai.config.ChatDatabricks") as mock_chat_databricks,
    ):
        mock_chat_openai.return_value = MagicMock()
        model.chat_model_for_workspace_client(obo_wc)

    mock_chat_databricks.assert_not_called()
    mock_chat_openai.assert_called_once()
    kwargs = mock_chat_openai.call_args.kwargs
    assert kwargs["base_url"] == "https://obo-host.databricks.com/ai-gateway/mlflow/v1"
    assert kwargs["model"] == "databricks-claude-opus-4-6"
    # Token provider must close over the OBO workspace_client — not the
    # config-level one — so OBO-scoped requests carry the user's token.
    assert callable(kwargs["api_key"])
    assert kwargs["api_key"]() == "obo-user-token"


def test_chat_model_for_workspace_client_legacy_path_passes_workspace_client() -> None:
    """Without ai_gateway, OBO factory falls through to ChatDatabricks bound
    to the OBO workspace_client (unchanged legacy behavior)."""
    model = InferenceEndpointModel(
        name="databricks-meta-llama-3-3-70b-instruct",
        on_behalf_of_user=True,
        temperature=0.0,
        max_tokens=128,
    )
    obo_wc = MagicMock()

    with (
        patch("dao_ai.config.ChatDatabricks") as mock_chat_databricks,
        patch("dao_ai.config.AIGatewayChatOpenAI") as mock_chat_openai,
    ):
        mock_chat_databricks.return_value = MagicMock()
        model.chat_model_for_workspace_client(obo_wc)

    mock_chat_openai.assert_not_called()
    mock_chat_databricks.assert_called_once()
    kwargs = mock_chat_databricks.call_args.kwargs
    assert kwargs["workspace_client"] is obo_wc
    assert kwargs["model"] == "databricks-meta-llama-3-3-70b-instruct"


# ---------------------------------------------------------------------------
# Tool calling — unit-level wiring
# ---------------------------------------------------------------------------


def test_as_chat_model_returns_runnable_with_bind_tools() -> None:
    """The returned client must expose bind_tools() so dao-ai agent loops
    (which always call .bind_tools(tools)) work transparently against the
    AI Gateway path."""
    model = InferenceEndpointModel(name="databricks-claude-opus-4-6", ai_gateway=True)
    with patch.object(
        InferenceEndpointModel,
        "_resolve_ai_gateway_credentials",
        return_value=("https://x", lambda: "t"),
    ):
        client = model.as_chat_model()
    assert hasattr(client, "bind_tools"), (
        "as_chat_model() must return a client with bind_tools() — agent loops "
        "depend on this. Got: " + type(client).__name__
    )


def test_heterogeneous_fallbacks_compose() -> None:
    """AI Gateway primary + legacy fallback should chain via with_fallbacks."""
    primary = InferenceEndpointModel(
        name="databricks-claude-opus-4-6",
        ai_gateway=True,
        fallbacks=["databricks-meta-llama-3-3-70b-instruct"],
    )

    chat_openai_instance = MagicMock(name="ChatOpenAI-instance")
    chat_databricks_instance = MagicMock(name="ChatDatabricks-instance")
    chained = MagicMock(name="with_fallbacks-result")
    chat_openai_instance.with_fallbacks.return_value = chained

    with (
        patch.object(
            InferenceEndpointModel,
            "_resolve_ai_gateway_credentials",
            return_value=("https://h", lambda: "t"),
        ),
        patch("dao_ai.config.AIGatewayChatOpenAI", return_value=chat_openai_instance),
        patch("dao_ai.config.ChatDatabricks", return_value=chat_databricks_instance),
    ):
        result = primary.as_chat_model()

    chat_openai_instance.with_fallbacks.assert_called_once()
    (fallback_list,) = chat_openai_instance.with_fallbacks.call_args.args
    assert fallback_list == [chat_databricks_instance]
    assert result is chained


# ---------------------------------------------------------------------------
# AIGatewayChatOpenAI — name-field stripping
#
# Background: the AI Gateway's OpenAI-compatible validator rejects ``name``
# on user/assistant/system messages with 400 BAD_REQUEST. LangGraph's
# supervisor pattern attaches ``name`` to AIMessages for routing, so the
# subclass strips it on the request-payload boundary.
# ---------------------------------------------------------------------------


def _build_ai_gateway_chat():
    from dao_ai.config import AIGatewayChatOpenAI

    return AIGatewayChatOpenAI(
        model="databricks-claude-opus-4-6",
        base_url="https://example.databricks.com/ai-gateway/mlflow/v1",
        api_key="dapi-test-token",
    )


def test_ai_gateway_chat_strips_name_from_user_messages() -> None:
    from langchain_core.messages import HumanMessage

    chat = _build_ai_gateway_chat()
    payload = chat._get_request_payload([HumanMessage(content="hi", name="alice")])
    user_msgs = [m for m in payload["messages"] if m.get("role") == "user"]
    assert user_msgs, "expected a user message in the payload"
    assert all("name" not in m for m in user_msgs)


def test_ai_gateway_chat_strips_name_from_assistant_messages() -> None:
    from langchain_core.messages import AIMessage

    chat = _build_ai_gateway_chat()
    payload = chat._get_request_payload(
        [AIMessage(content="hello there", name="agent_a")]
    )
    assistant_msgs = [m for m in payload["messages"] if m.get("role") == "assistant"]
    assert assistant_msgs
    assert all("name" not in m for m in assistant_msgs)


def test_ai_gateway_chat_strips_name_from_system_messages() -> None:
    from langchain_core.messages import SystemMessage

    chat = _build_ai_gateway_chat()
    payload = chat._get_request_payload(
        [SystemMessage(content="you are helpful", name="sys")]
    )
    system_msgs = [m for m in payload["messages"] if m.get("role") == "system"]
    assert system_msgs
    assert all("name" not in m for m in system_msgs)


def test_ai_gateway_chat_leaves_tool_messages_alone() -> None:
    """Tool messages may legitimately carry ``name`` (function name); leave them
    untouched — OpenAI's tool message schema is the upstream contract."""
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        ToolMessage,
    )

    chat = _build_ai_gateway_chat()
    payload = chat._get_request_payload(
        [
            HumanMessage(content="run my_tool", name="user"),
            AIMessage(
                content="",
                name="agent_a",
                tool_calls=[
                    {"id": "t1", "name": "my_tool", "args": {}, "type": "tool_call"}
                ],
            ),
            ToolMessage(content="ok", tool_call_id="t1", name="my_tool"),
        ]
    )
    tool_msgs = [m for m in payload["messages"] if m.get("role") == "tool"]
    assert tool_msgs, "expected a tool message in the payload"
    # We didn't strip from role=tool. Whatever LangChain put there (likely
    # tool_call_id only — no top-level name) is what we send to the gateway.
    # The assertion is just "we didn't crash and we didn't reach into tool".


def test_as_chat_model_ai_gateway_returns_subclass_instance() -> None:
    """When ai_gateway=True, as_chat_model() must return the name-stripping
    subclass — not a vanilla ChatOpenAI."""
    from dao_ai.config import AIGatewayChatOpenAI

    model = InferenceEndpointModel(name="databricks-claude-opus-4-6", ai_gateway=True)
    with patch.object(
        InferenceEndpointModel,
        "_resolve_ai_gateway_credentials",
        return_value=("https://x", lambda: "t"),
    ):
        client = model.as_chat_model()
    assert isinstance(client, AIGatewayChatOpenAI)


def test_as_chat_model_legacy_path_is_not_subclass() -> None:
    """Regression: ai_gateway=False returns ChatDatabricks (no subclass involvement)."""
    from dao_ai.config import AIGatewayChatOpenAI

    model = InferenceEndpointModel(name="databricks-meta-llama-3-3-70b-instruct")
    with patch("dao_ai.config.ChatDatabricks") as mock_chat_databricks:
        mock_chat_databricks.return_value = MagicMock(spec=[])
        client = model.as_chat_model()
    assert not isinstance(client, AIGatewayChatOpenAI)


def test_chat_model_for_workspace_client_ai_gateway_returns_subclass() -> None:
    """OBO factory must also return the subclass when ai_gateway=True."""
    from dao_ai.config import AIGatewayChatOpenAI

    model = InferenceEndpointModel(
        name="databricks-claude-opus-4-6",
        ai_gateway=True,
        on_behalf_of_user=True,
    )
    obo_wc = MagicMock()
    obo_wc.config.host = "https://obo-host.databricks.com"
    obo_wc.config.authenticate.return_value = {"Authorization": "Bearer obo-tok"}

    client = model.chat_model_for_workspace_client(obo_wc)
    assert isinstance(client, AIGatewayChatOpenAI)


def test_as_open_ai_client_ai_gateway_returns_subclass() -> None:
    """as_open_ai_client (raw client factory) must also return the subclass
    when ai_gateway=True."""
    from dao_ai.config import AIGatewayChatOpenAI

    model = InferenceEndpointModel(name="databricks-claude-opus-4-6", ai_gateway=True)
    with patch.object(
        InferenceEndpointModel,
        "_resolve_ai_gateway_credentials",
        return_value=("https://h", lambda: "t"),
    ):
        client = model.as_open_ai_client()
    assert isinstance(client, AIGatewayChatOpenAI)


# ---------------------------------------------------------------------------
# Live AI Gateway smoke (integration)
# ---------------------------------------------------------------------------


import os  # noqa: E402


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("DAO_AI_INTEGRATION"),
    reason="Set DAO_AI_INTEGRATION=1 (with DATABRICKS_HOST/DATABRICKS_TOKEN or a configured profile) to run live AI Gateway smoke.",
)
def test_ai_gateway_chat_handles_supervisor_named_messages_live() -> None:
    """Live AI Gateway round-trip with supervisor-tagged AIMessage(name=...).

    Pre-fix this would 400 with ``messages.N.name: Extra inputs are not permitted``.
    Post-fix the name is stripped on the request boundary and the call succeeds.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    endpoint = InferenceEndpointModel(
        name="databricks-claude-opus-4-6",
        ai_gateway=True,
        max_tokens=64,
    )
    chat = endpoint.as_chat_model()
    out = chat.invoke(
        [
            HumanMessage(content="Say a one-word greeting.", name="user_alice"),
            AIMessage(content="hello there", name="loyalty_lead"),
            HumanMessage(content="Now say a one-word farewell."),
        ]
    )
    assert isinstance(out, AIMessage)
    assert out.content
